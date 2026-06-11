use std::collections::HashSet;

use crate::avl::node::Node;
use crate::error::{Error, Result, UnsupportedFeature};
use crate::hash::{Hash, NULL_HASH};

use super::recording::{get_recording, range_recording};
use super::sparse::Trace;
use super::write_tracing::{
    node_skeleton_from_trace, replay_delete_prefix_on_node_tree, replay_writes_on_node_tree,
};
#[cfg(test)]
use super::ReadResults;
use super::{prefix_successor, BatchOp, SparseMerkNode};

/// Verifier-derived read results, one entry per read transcript step. Test/bench
/// harness only — the production [`TraceVerifier`] returns results per call.
#[cfg(test)]
pub(crate) type VerifiedReadResults = Vec<ReadResults>;

/// Externalized AVL trace verify core — the module-scoped peer of
/// [`crate::mrt::TraceVerifier`].
///
/// Public surface is decode + authenticate + read ([`Self::from_trace`] /
/// [`Self::decode_trace`], [`Self::verify_root`] / [`Self::root_hash`],
/// [`Self::get`] / [`Self::collect_range`] / [`Self::collect_prefix`]). The
/// sequenced **write** path (apply caller-supplied ops, then re-authenticate) is
/// the public handle [`TraceReplayer`], which wraps this; the in-place apply
/// primitives below are `pub(crate)` — there is no public batch-write API. The
/// trace is the proof; steps and roots are supplied by the caller (the
/// changelog), not bundled into a self-contained envelope.
///
/// Soundness: a read yields a value only for a key whose value the trace
/// actually carries (a `Full` node, or one written during replay). Reading a
/// presence-only node (`FullStorageHash` / `FullOmitted`) fails with
/// [`Error::ValueOmitted`]; descending into a `Pruned` node fails with
/// `Error::PrunedNode`.
pub struct TraceVerifier {
    state: Option<Node>,
    readable_keys: HashSet<Vec<u8>>,
    /// Set iff the trace is fully pruned (no materializable `Node` skeleton — a
    /// transcript that accessed nothing). [`Self::verify_root`] authenticates
    /// against this hash, but any read or replay fails `PrunedNode`: you cannot
    /// read from or mutate an unmaterialized tree. (MRT represents this with a
    /// `PrunedHash` verify-node; AVL replays real `Node`s, which cannot be a
    /// bare-hash root, so the case is tracked here instead — keeping `from_trace`
    /// infallible, like `mrt::TraceVerifier::from_trace`.)
    pruned_root: Option<Hash>,
}

impl TraceVerifier {
    /// Build a verifier from an in-memory trace.
    pub fn from_trace(trace: &Trace) -> Self {
        let mut readable_keys = HashSet::new();
        collect_full_value_keys(trace, &mut readable_keys);
        // A fully-pruned trace has no `Node` form; defer rather than fail —
        // `verify_root` authenticates via its root hash, reads/replay error.
        match node_skeleton_from_trace(trace) {
            Ok(state) => Self {
                state,
                readable_keys,
                pruned_root: None,
            },
            Err(_) => Self {
                state: None,
                readable_keys,
                pruned_root: Some(trace.hash()),
            },
        }
    }

    /// Decode an encoded trace and build a verifier.
    pub fn decode_trace(bytes: &[u8]) -> Result<Self> {
        Ok(Self::from_trace(&Trace::decode_exact(bytes)?))
    }

    /// Authenticate the current tree against `expected`.
    pub fn verify_root(&mut self, expected: Hash) -> Result<()> {
        if let Some(pruned) = self.pruned_root {
            return if pruned == expected {
                Ok(())
            } else {
                Err(Error::HashMismatch(expected, pruned))
            };
        }
        verify_node_root(self.state.as_mut(), expected)
    }

    /// Commit and return the current tree's root hash.
    pub fn root_hash(&mut self) -> Result<Hash> {
        if let Some(pruned) = self.pruned_root {
            return Ok(pruned);
        }
        Ok(match self.state.as_mut() {
            Some(tree) => {
                tree.commit();
                tree.hash()
            }
            None => NULL_HASH,
        })
    }

    /// Point read. Returns the value iff the trace carries it; fails
    /// `ValueOmitted` for a presence-only node and `PrunedNode` for a pruned
    /// path (or a fully-pruned trace).
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.reject_if_pruned("read")?;
        match self.state.as_ref() {
            Some(root) => match get_recording(root, key, &mut |_| {})? {
                Some(value) => {
                    reject_hash_only_read_result(key, &self.readable_keys)?;
                    Ok(Some(value))
                }
                None => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// Half-open range read `[start, end)` (open upper bound when `end` is
    /// `None`).
    pub fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_pruned("range read")?;
        collect_node_range(self.state.as_ref(), start, end, &self.readable_keys)
    }

    /// Prefix read (all keys starting with `prefix`).
    pub fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let end = prefix_successor(prefix);
        self.collect_range(prefix, end.as_deref())
    }

    /// Replay a write batch against the current tree. Test-only — production
    /// drives writes through [`TraceReplayer`] (op-by-op `replay_batch_ops_in_place`).
    #[cfg(test)]
    pub(crate) fn replay_batch_ops(&mut self, ops: &[BatchOp]) -> Result<()> {
        self.replay_batch_ops_in_place(ops)
    }

    /// In-place [`replay_batch_ops`](Self::replay_batch_ops).
    ///
    /// The AVL verifier's existing batch replay already mutates in place, so this
    /// method is the symmetric hot-path name shared with the MRT verifier.
    pub(crate) fn replay_batch_ops_in_place(&mut self, ops: &[BatchOp]) -> Result<()> {
        self.reject_if_pruned("replay")?;
        self.state = replay_writes_on_node_tree(self.state.take(), ops)?;
        update_readable_keys_for_write_ops(&mut self.readable_keys, ops);
        Ok(())
    }

    /// Delete every key with byte-prefix `prefix`, in place. Peer of
    /// [`crate::mrt::TraceVerifier`]'s `delete_prefix_in_place`.
    pub(crate) fn delete_prefix_in_place(&mut self, prefix: &[u8]) -> Result<()> {
        self.reject_if_pruned("delete_prefix replay")?;
        self.state = replay_delete_prefix_on_node_tree(self.state.take(), prefix)?;
        self.readable_keys.retain(|key| !key.starts_with(prefix));
        Ok(())
    }

    /// AVL has no subtree-relocate primitive.
    pub(crate) fn move_prefix(&mut self, _from: &[u8], _to: &[u8]) -> Result<()> {
        Err(Error::Unsupported(UnsupportedFeature::MovePrefix))
    }

    /// A fully-pruned trace can be authenticated (`verify_root`) but not read or
    /// mutated — those would answer/derive from nodes the trace never revealed.
    fn reject_if_pruned(&self, op: &str) -> Result<()> {
        if self.pruned_root.is_some() {
            return Err(Error::PrunedNode(format!(
                "{op} against a fully-pruned trace (no revealed nodes)"
            )));
        }
        Ok(())
    }
}

/// Verify-side traced handle for the AVL backend, bound to an expected start
/// root at construction. The externalized peer of [`crate::mrt::TraceReplayer`].
///
/// A thin wrapper over [`TraceVerifier`] that gives consumers the same sequenced
/// read/write calls ([`TraceInterface`](crate::tracer::TraceInterface)) the
/// prove side uses, with the trace decoded once and authenticated against the
/// caller's expected pre-state.
///
/// Build it with [`new_verified`](Self::new_verified) or
/// [`new_unverified`](Self::new_unverified). Read soundness comes straight from
/// the wrapped verifier: a pruned non-membership path or an incomplete
/// range/prefix witness fails with [`Error::PrunedNode`] rather than returning
/// `Ok(None)` or a truncated list.
///
/// **Single-shot on error:** mutations apply in place, so any method returning
/// `Err` may leave the replayer partially mutated. Discard the handle after an
/// error — do not keep using it. This matches the zkVM verify model, where any
/// error rejects the whole proof and drops the tree.
pub struct TraceReplayer {
    verifier: TraceVerifier,
    /// Set once an [`apply`](crate::tracer::TraceInterface::apply) op fails. A
    /// poisoned replayer rejects every subsequent fallible op (reads, further
    /// `apply`, root verification) with [`Error::Poisoned`] — the per-op COW may
    /// have left the tree empty on the failed op, so reads/replay must not be
    /// trusted again.
    poisoned: bool,
}

impl TraceReplayer {
    /// Decode `trace_bytes`, authenticate the trace's root against
    /// `expected_start_root`, and bind the replayer to that pre-state.
    ///
    /// Fails closed with [`Error::HashMismatch`] when the decoded trace's root
    /// does not match `expected_start_root` — *before* any read can be trusted —
    /// or with a decode error when the bytes are malformed. This is the
    /// constructor production callers use: a successful return means subsequent
    /// reads are authenticated against the expected start root.
    pub fn new_verified(trace_bytes: &[u8], expected_start_root: Hash) -> Result<Self> {
        let mut verifier = TraceVerifier::decode_trace(trace_bytes)?;
        verifier.verify_root(expected_start_root)?;
        Ok(Self {
            verifier,
            poisoned: false,
        })
    }

    /// Decode `trace_bytes` without binding them to a start root.
    ///
    /// For tests, diagnostics, and callers that intentionally bind later: reads
    /// through the returned handle are **not** trusted until the caller checks
    /// `root_hash() == expected_start_root` itself. Prefer
    /// [`new_verified`](Self::new_verified) in production.
    pub fn new_unverified(trace_bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            verifier: TraceVerifier::decode_trace(trace_bytes)?,
            poisoned: false,
        })
    }

    /// Reject any fallible op once the handle is poisoned.
    fn reject_if_poisoned(&self, op: &str) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned(format!(
                "{op} on a poisoned AVL replayer (an earlier apply failed)"
            )));
        }
        Ok(())
    }

    /// The current tree's root hash. Fails [`Error::Poisoned`] if a prior `apply`
    /// op failed (the post-state root would be untrustworthy).
    pub fn root_hash(&mut self) -> Result<Hash> {
        self.reject_if_poisoned("root_hash")?;
        self.verifier.root_hash()
    }
}

impl super::TraceReader for TraceReplayer {
    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.reject_if_poisoned("get")?;
        self.verifier.get(key)
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_range")?;
        self.verifier.collect_range(start, Some(end))
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_prefix")?;
        self.verifier.collect_prefix(prefix)
    }
}

/// Replays each `WriteOp` **one at a time, in issue order** — identical to the
/// [`TraceRecorder`](super::finalization::TraceRecorder) and to the MRT backend.
///
/// AVL is **insertion-order sensitive**: applying the same keys as a sorted batch
/// versus one at a time produces different trees and roots. So there is no
/// batching or sorting here — each op lowers to a single-element [`BatchOp`]
/// in-place replay (point/range), the prefix-delete primitive, or AVL's
/// unsupported `MovePrefix`; a `Vec<WriteOp>` is never gathered into a sorted
/// multi-op batch. Because the recorder reveals exactly the nodes each sequential
/// write touches, the replayer touches the same set and never needs an unrevealed
/// node. An empty batch is a no-op. The first failing op poisons the replayer.
impl super::TraceInterface for TraceReplayer {
    fn apply(&mut self, ops: &[super::WriteOp]) -> Result<()> {
        use super::WriteOp;
        self.reject_if_poisoned("apply")?;
        for op in ops {
            let result = match op {
                WriteOp::Put { key, value } => {
                    self.verifier.replay_batch_ops_in_place(&[BatchOp::Put {
                        key: key.clone(),
                        value: value.clone(),
                    }])
                }
                WriteOp::Delete { key } => self
                    .verifier
                    .replay_batch_ops_in_place(&[BatchOp::Delete { key: key.clone() }]),
                WriteOp::DeleteRange { start, end } => {
                    self.verifier
                        .replay_batch_ops_in_place(&[BatchOp::DeleteRange {
                            start: start.clone(),
                            end: end.clone(),
                        }])
                }
                WriteOp::DeletePrefix { prefix } => self.verifier.delete_prefix_in_place(prefix),
                WriteOp::MovePrefix { from, to } => self.verifier.move_prefix(from, to),
            };
            if let Err(err) = result {
                self.poisoned = true;
                return Err(err);
            }
        }
        Ok(())
    }
}

fn verify_node_root(tree: Option<&mut Node>, expected_root: Hash) -> Result<()> {
    let actual = match tree {
        Some(tree) => {
            tree.commit();
            tree.hash()
        }
        None => NULL_HASH,
    };

    if actual == expected_root {
        Ok(())
    } else {
        Err(Error::HashMismatch(expected_root, actual))
    }
}

fn collect_node_range(
    root: Option<&Node>,
    start: &[u8],
    end: Option<&[u8]>,
    readable_keys: &HashSet<Vec<u8>>,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let Some(root) = root else {
        return Ok(Vec::new());
    };

    let results = range_recording(root, start, end, &mut |_| {})?;
    for (key, _) in &results {
        reject_hash_only_read_result(key, readable_keys)?;
    }
    Ok(results)
}

fn reject_hash_only_read_result(key: &[u8], readable_keys: &HashSet<Vec<u8>>) -> Result<()> {
    if readable_keys.contains(key) {
        Ok(())
    } else {
        Err(Error::ValueOmitted(format!(
            "read tried to yield hash-only node {key:?}"
        )))
    }
}

fn update_readable_keys_for_write_ops(readable_keys: &mut HashSet<Vec<u8>>, ops: &[BatchOp]) {
    for op in ops {
        match op {
            BatchOp::Put { key, .. } => {
                readable_keys.insert(key.clone());
            }
            BatchOp::Delete { key } => {
                readable_keys.remove(key);
            }
            BatchOp::DeleteRange { start, end } => {
                readable_keys.retain(|key| key.as_slice() < start || key.as_slice() >= end);
            }
        }
    }
}

fn collect_full_value_keys(trace: &SparseMerkNode, out: &mut HashSet<Vec<u8>>) {
    match trace {
        SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => {}
        SparseMerkNode::Full {
            key, left, right, ..
        } => {
            out.insert(key.clone());
            collect_full_value_keys(left, out);
            collect_full_value_keys(right, out);
        }
        SparseMerkNode::FullStorageHash { left, right, .. }
        | SparseMerkNode::FullOmitted { left, right, .. } => {
            collect_full_value_keys(left, out);
            collect_full_value_keys(right, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avl::in_memory::{Checkpoint, InMemoryMerk};
    use crate::hash::{kv_hash, Hasher, HASH_LENGTH};
    use crate::tracer::test_support::avl::{create_trace, replay_trace, root_after_writes};
    use crate::tracer::test_support::Step;
    use crate::tracer::{
        ReadOp, TraceInterface, TraceReader, WriteOp, SMALL_VALUE_INLINE_THRESHOLD,
    };

    fn build_tree(entries: &[(&[u8], &[u8])]) -> Checkpoint {
        let merk = InMemoryMerk::new();
        for &(key, value) in entries {
            merk.put(key, value).unwrap();
        }
        merk.checkpoint()
    }

    fn large_value(byte: u8) -> Vec<u8> {
        vec![byte; SMALL_VALUE_INLINE_THRESHOLD + 1]
    }

    /// A basic single-read transcript over a small tree: returns the witness
    /// trace, the caller steps, and the authenticated start/end roots.
    fn basic_read_inputs() -> (Trace, Vec<Step>, Hash, Hash) {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps = vec![Step::Read(vec![ReadOp::Key(b"c".to_vec())])];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let end = root_after_writes(&root, &steps);
        (trace, steps, start, end)
    }

    #[test]
    fn verifier_accepts_empty_transcript_with_pruned_root() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps: Vec<Step> = Vec::new();
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();

        assert!(matches!(trace, SparseMerkNode::Pruned { .. }));
        assert!(replay_trace(&trace, start, &steps, start)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn pruned_root_verifier_authenticates_but_rejects_reads_and_replay() {
        // A non-empty tree traced with no accessed nodes yields a fully-pruned
        // root: `from_trace` is still infallible (matching mrt), `verify_root`
        // authenticates against the trace root, but reads/replay error.
        let snapshot = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let trace = create_trace(&snapshot, &[]).unwrap();
        assert!(matches!(trace, SparseMerkNode::Pruned { .. }));
        let root = trace.hash();

        let mut v = TraceVerifier::from_trace(&trace);
        v.verify_root(root).unwrap();
        assert_eq!(v.root_hash().unwrap(), root);
        assert!(matches!(v.get(b"a"), Err(Error::PrunedNode(_))));
        assert!(matches!(
            v.collect_range(b"a", Some(b"z")),
            Err(Error::PrunedNode(_))
        ));
        assert!(matches!(
            v.replay_batch_ops(&[BatchOp::Put {
                key: b"a".to_vec(),
                value: b"x".to_vec()
            }]),
            Err(Error::PrunedNode(_))
        ));
        assert!(matches!(
            v.verify_root([9; HASH_LENGTH]),
            Err(Error::HashMismatch(_, _))
        ));
    }

    fn find_full_mut<'a>(
        trace: &'a mut SparseMerkNode,
        target: &[u8],
    ) -> Option<&'a mut SparseMerkNode> {
        let key = trace.key()?;
        if target == key {
            return Some(trace);
        }
        let go_left = target < key;

        match trace {
            SparseMerkNode::Full { left, right, .. }
            | SparseMerkNode::FullStorageHash { left, right, .. }
            | SparseMerkNode::FullOmitted { left, right, .. } => {
                find_full_mut(if go_left { left } else { right }, target)
            }
            SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => None,
        }
    }

    fn prune_child_on_path(trace: &mut SparseMerkNode, parent_key: &[u8], left_child: bool) {
        let parent = find_full_mut(trace, parent_key).expect("parent exists");
        let child_slot = match parent {
            SparseMerkNode::Full { left, right, .. }
            | SparseMerkNode::FullStorageHash { left, right, .. }
            | SparseMerkNode::FullOmitted { left, right, .. } => {
                if left_child {
                    left
                } else {
                    right
                }
            }
            SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => unreachable!(),
        };
        let key = child_slot.key().expect("child must have key").to_vec();
        let hash = child_slot.hash();
        let child_heights = match child_slot.as_ref() {
            SparseMerkNode::Empty => unreachable!(),
            SparseMerkNode::Pruned { child_heights, .. } => *child_heights,
            SparseMerkNode::Full { left, right, .. }
            | SparseMerkNode::FullStorageHash { left, right, .. }
            | SparseMerkNode::FullOmitted { left, right, .. } => (left.height(), right.height()),
        };
        **child_slot = SparseMerkNode::Pruned {
            key,
            hash,
            child_heights,
        };
    }

    #[test]
    fn verifier_accepts_interleaved_reads_writes_and_delete_range() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"c", b"3"),
            (b"e", b"5"),
            (b"g", b"7"),
            (b"i", b"9"),
        ]);
        let steps = vec![
            Step::Read(vec![ReadOp::Key(b"a".to_vec())]),
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"updated".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Range {
                start: b"a".to_vec(),
                end: b"h".to_vec(),
            }]),
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"e".to_vec(),
                end: b"h".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Prefix(b"g".to_vec())]),
        ];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let end = root_after_writes(&root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();

        assert_eq!(reads.len(), 3);
        assert_eq!(reads[0][0].results, vec![(b"a".to_vec(), b"1".to_vec())]);
        assert_eq!(
            reads[1][0].results,
            vec![
                (b"a".to_vec(), b"1".to_vec()),
                (b"c".to_vec(), b"updated".to_vec()),
                (b"e".to_vec(), b"5".to_vec()),
                (b"g".to_vec(), b"7".to_vec()),
            ]
        );
        assert!(reads[2][0].results.is_empty());
    }

    #[test]
    fn corrupt_start_root_fails() {
        let (trace, steps, _start, end) = basic_read_inputs();
        assert!(matches!(
            replay_trace(&trace, [9; HASH_LENGTH], &steps, end),
            Err(Error::HashMismatch(_, _))
        ));
    }

    #[test]
    fn corrupt_end_root_fails() {
        let (trace, steps, start, _end) = basic_read_inputs();
        assert!(matches!(
            replay_trace(&trace, start, &steps, [9; HASH_LENGTH]),
            Err(Error::HashMismatch(_, _))
        ));
    }

    #[test]
    fn remove_required_path_node_fails() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps = vec![Step::Read(vec![ReadOp::Key(b"a".to_vec())])];
        let mut trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();

        // Pruning a required node preserves the root hash (it leaves a stub), so
        // start authentication still passes — the read then hits the pruned node.
        prune_child_on_path(&mut trace, b"c", true);

        assert!(matches!(
            replay_trace(&trace, start, &steps, start),
            Err(Error::PrunedNode(_))
        ));
    }

    #[test]
    fn demote_read_target_full_node_to_omitted_fails() {
        let big = large_value(0xAA);
        let root = build_tree(&[(b"a", &big), (b"c", &big), (b"e", &big)]);
        let steps = vec![Step::Read(vec![ReadOp::Key(b"c".to_vec())])];
        let mut trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();

        let node = find_full_mut(&mut trace, b"c").expect("read target exists");
        match node {
            SparseMerkNode::Full {
                key,
                value,
                left,
                right,
            } => {
                *node = SparseMerkNode::FullOmitted {
                    key: key.clone(),
                    kv_hash: kv_hash::<Hasher>(key, value).unwrap(),
                    left: left.clone(),
                    right: right.clone(),
                };
            }
            other => panic!("expected Full read target, got {:?}", other),
        }

        assert!(matches!(
            replay_trace(&trace, start, &steps, start),
            Err(Error::ValueOmitted(_))
        ));
    }

    #[test]
    fn wrong_write_key_fails_end_root() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps = vec![Step::Write(vec![BatchOp::Put {
            key: b"c".to_vec(),
            value: b"updated".to_vec(),
        }])];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let end = root_after_writes(&root, &steps);

        // The trace only reveals the path for the `c` write. Replaying a write to
        // a different key against it cannot reproduce `end` (it hits a pruned node
        // or a wrong post-root).
        let corrupted = vec![Step::Write(vec![BatchOp::Put {
            key: b"e".to_vec(),
            value: b"updated".to_vec(),
        }])];
        assert!(replay_trace(&trace, start, &corrupted, end).is_err());
    }

    #[test]
    fn full_storage_hash_point_read_fails() {
        let kv_hash = kv_hash::<Hasher>(b"k", b"hidden").unwrap();
        let trace = SparseMerkNode::FullStorageHash {
            key: b"k".to_vec(),
            kv_hash,
            left: Box::new(SparseMerkNode::Empty),
            right: Box::new(SparseMerkNode::Empty),
        };
        let root = trace.hash();
        let steps = vec![Step::Read(vec![ReadOp::Key(b"k".to_vec())])];

        assert!(matches!(
            replay_trace(&trace, root, &steps, root),
            Err(Error::ValueOmitted(_))
        ));
    }

    #[test]
    fn full_storage_hash_range_result_fails() {
        let kv_hash = kv_hash::<Hasher>(b"k", b"hidden").unwrap();
        let trace = SparseMerkNode::FullStorageHash {
            key: b"k".to_vec(),
            kv_hash,
            left: Box::new(SparseMerkNode::Full {
                key: b"a".to_vec(),
                value: b"left".to_vec(),
                left: Box::new(SparseMerkNode::Empty),
                right: Box::new(SparseMerkNode::Empty),
            }),
            right: Box::new(SparseMerkNode::Full {
                key: b"z".to_vec(),
                value: b"right".to_vec(),
                left: Box::new(SparseMerkNode::Empty),
                right: Box::new(SparseMerkNode::Empty),
            }),
        };
        let root = trace.hash();
        let steps = vec![Step::Read(vec![ReadOp::Range {
            start: b"a".to_vec(),
            end: b"z".to_vec(),
        }])];

        assert!(matches!(
            replay_trace(&trace, root, &steps, root),
            Err(Error::ValueOmitted(_))
        ));
    }

    // ── Stage 4: TraceReplayer root binding and reads ────────────────────────

    #[test]
    fn replayer_new_verified_accepts_honest_trace_and_reads() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps = vec![Step::Read(vec![
            ReadOp::Key(b"c".to_vec()),
            ReadOp::Range {
                start: b"a".to_vec(),
                end: b"z".to_vec(),
            },
        ])];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let bytes = trace.encode().unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        assert_eq!(replayer.get(b"c").unwrap(), Some(b"3".to_vec()));
        assert_eq!(
            replayer.get_range(b"a", b"z").unwrap(),
            vec![
                (b"a".to_vec(), b"1".to_vec()),
                (b"c".to_vec(), b"3".to_vec()),
                (b"e".to_vec(), b"5".to_vec()),
            ]
        );
        // Read-only: the post-state root equals the bound start root.
        assert_eq!(replayer.root_hash().unwrap(), start);
    }

    #[test]
    fn replayer_new_verified_rejects_wrong_root_before_reads() {
        let (trace, _steps, start, _end) = basic_read_inputs();
        let bytes = trace.encode().unwrap();
        // Honest bytes, wrong expected start root: construction fails closed,
        // before any read can be trusted.
        assert!(matches!(
            TraceReplayer::new_verified(&bytes, [9; HASH_LENGTH]),
            Err(Error::HashMismatch(_, _))
        ));
        // The same bytes bound to the correct root succeed.
        assert!(TraceReplayer::new_verified(&bytes, start).is_ok());
    }

    #[test]
    fn replayer_new_unverified_root_matches_decode_trace() {
        let (trace, _steps, start, _end) = basic_read_inputs();
        let bytes = trace.encode().unwrap();
        let mut replayer = TraceReplayer::new_unverified(&bytes).unwrap();
        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        assert_eq!(replayer.root_hash().unwrap(), verifier.root_hash().unwrap());
        assert_eq!(replayer.root_hash().unwrap(), start);
    }

    #[test]
    fn replayer_reads_match_verifier_reads() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let steps = vec![Step::Read(vec![
            ReadOp::Key(b"c".to_vec()),
            ReadOp::Range {
                start: b"a".to_vec(),
                end: b"h".to_vec(),
            },
            ReadOp::Prefix(b"e".to_vec()),
        ])];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let bytes = trace.encode().unwrap();

        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        verifier.verify_root(start).unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();

        assert_eq!(replayer.get(b"c").unwrap(), verifier.get(b"c").unwrap());
        assert_eq!(
            replayer.get_range(b"a", b"h").unwrap(),
            verifier.collect_range(b"a", Some(b"h")).unwrap()
        );
        assert_eq!(
            replayer.get_prefix(b"e").unwrap(),
            verifier.collect_prefix(b"e").unwrap()
        );
    }

    /// A narrow-reveal trace: it proves only the path to `a`, pruning everything
    /// else. Reads outside the revealed region must fail closed (the basis for
    /// the adversarial soundness tests below).
    fn narrow_reveal_replayer() -> TraceReplayer {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps = vec![Step::Read(vec![ReadOp::Key(b"a".to_vec())])];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let bytes = trace.encode().unwrap();
        TraceReplayer::new_verified(&bytes, start).unwrap()
    }

    #[test]
    fn replayer_get_absent_pruned_path_fails_closed() {
        let mut replayer = narrow_reveal_replayer();
        // `d` is absent and its non-membership path was never revealed: must be
        // `PrunedNode`, not `Ok(None)`.
        assert!(matches!(replayer.get(b"d"), Err(Error::PrunedNode(_))));
    }

    #[test]
    fn replayer_range_interior_pruned_fails_closed() {
        let mut replayer = narrow_reveal_replayer();
        // Must be `PrunedNode`, not a silently truncated `[(a, 1)]`.
        assert!(matches!(
            replayer.get_range(b"a", b"z"),
            Err(Error::PrunedNode(_))
        ));
    }

    #[test]
    fn replayer_range_boundary_pruned_fails_closed() {
        let mut replayer = narrow_reveal_replayer();
        assert!(matches!(
            replayer.get_range(b"e", b"z"),
            Err(Error::PrunedNode(_))
        ));
    }

    #[test]
    fn replayer_prefix_pruned_fails_closed() {
        let mut replayer = narrow_reveal_replayer();
        assert!(matches!(
            replayer.get_prefix(b"e"),
            Err(Error::PrunedNode(_))
        ));
    }

    // ── Stage 5: TraceReplayer sequenced writes ─────────────────────────────

    fn replayer_for_steps(root: &Checkpoint, steps: &[Step]) -> (TraceReplayer, Hash) {
        let trace = create_trace(root, steps).unwrap();
        let start = trace.hash();
        let end = root_after_writes(root, steps);
        let bytes = trace.encode().unwrap();
        (TraceReplayer::new_verified(&bytes, start).unwrap(), end)
    }

    #[test]
    fn replayer_sequenced_writes_match_old_verifier_replay() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"updated".to_vec(),
            }]),
            Step::Write(vec![BatchOp::Delete { key: b"a".to_vec() }]),
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"e".to_vec(),
                end: b"h".to_vec(),
            }]),
        ];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let end = root_after_writes(&root, &steps);
        let bytes = trace.encode().unwrap();

        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        verifier.verify_root(start).unwrap();
        for step in &steps {
            let Step::Write(ops) = step else {
                unreachable!()
            };
            verifier.replay_batch_ops_in_place(ops).unwrap();
        }

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.put(b"c", b"updated").unwrap();
        replayer.delete(b"a").unwrap();
        replayer.delete_range(b"e", b"h").unwrap();

        assert_eq!(replayer.root_hash().unwrap(), verifier.root_hash().unwrap());
        assert_eq!(replayer.root_hash().unwrap(), end);
    }

    #[test]
    fn replayer_direct_put_delete_preserves_duplicate_key_order() {
        let root = build_tree(&[]);
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"k".to_vec(),
                value: b"first".to_vec(),
            }]),
            Step::Write(vec![BatchOp::Delete { key: b"k".to_vec() }]),
            Step::Write(vec![BatchOp::Put {
                key: b"k".to_vec(),
                value: b"second".to_vec(),
            }]),
            Step::Write(vec![BatchOp::Put {
                key: b"gone".to_vec(),
                value: b"temp".to_vec(),
            }]),
            Step::Write(vec![BatchOp::Delete {
                key: b"gone".to_vec(),
            }]),
        ];
        let (mut replayer, end) = replayer_for_steps(&root, &steps);

        replayer.put(b"k", b"first").unwrap();
        replayer.delete(b"k").unwrap();
        replayer.put(b"k", b"second").unwrap();
        replayer.put(b"gone", b"temp").unwrap();
        replayer.delete(b"gone").unwrap();

        assert_eq!(replayer.get(b"k").unwrap(), Some(b"second".to_vec()));
        assert_eq!(replayer.get(b"gone").unwrap(), None);
        assert_eq!(replayer.root_hash().unwrap(), end);
    }

    #[test]
    fn replayer_delete_range_and_prefix_preserve_call_order() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"ba", b"old"),
            (b"bb", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
        ]);

        let range_then_put = vec![
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"b".to_vec(),
                end: b"d".to_vec(),
            }]),
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"new".to_vec(),
            }]),
        ];
        let (mut range_first, range_first_end) = replayer_for_steps(&root, &range_then_put);
        range_first.delete_range(b"b", b"d").unwrap();
        range_first.put(b"c", b"new").unwrap();

        let put_then_range = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"new".to_vec(),
            }]),
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"b".to_vec(),
                end: b"d".to_vec(),
            }]),
        ];
        let (mut range_last, range_last_end) = replayer_for_steps(&root, &put_then_range);
        range_last.put(b"c", b"new").unwrap();
        range_last.delete_range(b"b", b"d").unwrap();

        assert_eq!(range_first.root_hash().unwrap(), range_first_end);
        assert_eq!(range_last.root_hash().unwrap(), range_last_end);
        assert_ne!(range_first_end, range_last_end);

        let prefix_then_put = vec![
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"b".to_vec(),
                end: b"c".to_vec(),
            }]),
            Step::Write(vec![BatchOp::Put {
                key: b"ba".to_vec(),
                value: b"new".to_vec(),
            }]),
        ];
        let (mut prefix_first, prefix_first_end) = replayer_for_steps(&root, &prefix_then_put);
        prefix_first.delete_prefix(b"b").unwrap();
        prefix_first.put(b"ba", b"new").unwrap();

        let put_then_prefix = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"ba".to_vec(),
                value: b"new".to_vec(),
            }]),
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"b".to_vec(),
                end: b"c".to_vec(),
            }]),
        ];
        let (mut prefix_last, prefix_last_end) = replayer_for_steps(&root, &put_then_prefix);
        prefix_last.put(b"ba", b"new").unwrap();
        prefix_last.delete_prefix(b"b").unwrap();

        assert_eq!(prefix_first.root_hash().unwrap(), prefix_first_end);
        assert_eq!(prefix_last.root_hash().unwrap(), prefix_last_end);
        assert_ne!(prefix_first_end, prefix_last_end);
    }

    #[test]
    fn replayer_write_and_move_errors_use_expected_buckets() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let trace = create_trace(&root, &[]).unwrap();
        let start = trace.hash();
        let bytes = trace.encode().unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        assert!(matches!(
            replayer.put(b"a", b"x"),
            Err(Error::PrunedNode(_))
        ));

        let empty = build_tree(&[]);
        let trace = create_trace(&empty, &[]).unwrap();
        let start = trace.hash();
        let bytes = trace.encode().unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        assert!(matches!(
            replayer.move_prefix(b"from", b"to"),
            Err(Error::Unsupported(UnsupportedFeature::MovePrefix))
        ));
    }

    #[test]
    fn replayer_write_against_interior_pruned_witness_returns_pruned_node() {
        let mut replayer = narrow_reveal_replayer();
        // Op-by-op: the write is applied immediately, so the insufficient-witness
        // error surfaces at the `put` call rather than at a later barrier.
        assert!(matches!(
            replayer.put(b"d", b"x"),
            Err(Error::PrunedNode(_))
        ));
    }

    // ── Stage 3: WriteOp `apply` poison ─────────────────────────────────────

    #[test]
    fn apply_failure_poisons_avl_replayer() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        // A write step reveals `c`'s path so the first op below can apply.
        let steps = vec![Step::Write(vec![BatchOp::Put {
            key: b"c".to_vec(),
            value: b"updated".to_vec(),
        }])];
        let trace = create_trace(&root, &steps).unwrap();
        let start = trace.hash();
        let bytes = trace.encode().unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();

        // The Put applies in place, then the AVL-unsupported MovePrefix fails: the
        // change is rejected and the replayer poisoned (per-op COW may have left
        // the tree mutated, so it must not be trusted again).
        let err = replayer
            .apply(&[
                WriteOp::Put {
                    key: b"c".to_vec(),
                    value: b"updated".to_vec(),
                },
                WriteOp::MovePrefix {
                    from: b"a".to_vec(),
                    to: b"z".to_vec(),
                },
            ])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Unsupported(UnsupportedFeature::MovePrefix)
        ));

        // Every fallible op now fails closed: reads, root verification, further apply.
        assert!(matches!(replayer.get(b"c"), Err(Error::Poisoned(_))));
        assert!(matches!(replayer.root_hash(), Err(Error::Poisoned(_))));
        assert!(matches!(
            replayer.apply(&[WriteOp::Delete { key: b"a".to_vec() }]),
            Err(Error::Poisoned(_))
        ));
    }
}
