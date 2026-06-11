use std::collections::HashMap;
use std::sync::Arc;

use super::cursor::{get_descent, Cursor};
use super::trace::Trace;
use super::tree::{
    self, delete_prefix_with_trace, delete_range_with_trace, delete_with_trace, MrtNode,
    MrtNodeInner, RoutePrefix,
};
use super::validate_mrt_apply_batch;
use super::Checkpoint;
use crate::error::{Error, Result};
use crate::ops::{Batch, Op};
use crate::tracer::ReadOp;

pub(crate) struct MrtPartialBuilder {
    nodes: HashMap<usize, Arc<MrtNodeInner>>,
    root_ident: Option<usize>,
}

impl MrtPartialBuilder {
    pub(crate) fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root_ident: None,
        }
    }

    pub(crate) fn install(&mut self, ident: usize, arc: Arc<MrtNodeInner>) {
        self.nodes.entry(ident).or_insert(arc);
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, ident: usize) -> bool {
        self.nodes.contains_key(&ident)
    }

    pub(crate) fn set_root(&mut self, ident: usize) {
        self.root_ident = Some(ident);
    }

    pub(crate) fn into_parts(self) -> (HashMap<usize, Arc<MrtNodeInner>>, Option<usize>) {
        (self.nodes, self.root_ident)
    }
}

pub(crate) fn arc_ident(arc: &Arc<MrtNodeInner>) -> usize {
    Arc::as_ptr(arc) as usize
}

pub(crate) fn install_arc(builder: &mut MrtPartialBuilder, arc: &Arc<MrtNodeInner>) {
    builder.install(arc_ident(arc), Arc::clone(arc));
}

pub(crate) fn record_snapshot_root(builder: &mut MrtPartialBuilder, snapshot: &Checkpoint) {
    if let Some(root) = snapshot.root.as_ref() {
        let ident = arc_ident(root);
        builder.set_root(ident);
        install_arc(builder, root);
    }
}

pub(crate) fn assemble_mrt_pruned(builder: MrtPartialBuilder) -> Trace {
    let (nodes, root_ident) = builder.into_parts();
    let Some(root_ident) = root_ident else {
        return Trace(None);
    };
    let Some(root) = nodes.get(&root_ident) else {
        return Trace(None);
    };
    Trace(Some(walk_keep_visited(root, &nodes)))
}

fn walk_keep_visited(
    arc: &Arc<MrtNodeInner>,
    nodes: &HashMap<usize, Arc<MrtNodeInner>>,
) -> Arc<MrtNodeInner> {
    match arc.node() {
        MrtNode::Leaf { .. } | MrtNode::PrunedHash => Arc::clone(arc),
        MrtNode::Branch { skip, left, right } => {
            let new_left = if nodes.contains_key(&arc_ident(left)) {
                walk_keep_visited(left, nodes)
            } else {
                // Pruning a real subtree: carry its (authenticated) depth_below so a
                // parent re-hash and the move overflow check read the true value.
                MrtNodeInner::pruned(left.hash(), left.depth_below())
            };
            let new_right = if nodes.contains_key(&arc_ident(right)) {
                walk_keep_visited(right, nodes)
            } else {
                MrtNodeInner::pruned(right.hash(), right.depth_below())
            };
            if Arc::as_ptr(&new_left) == Arc::as_ptr(left)
                && Arc::as_ptr(&new_right) == Arc::as_ptr(right)
            {
                Arc::clone(arc)
            } else {
                MrtNodeInner::branch(skip.clone(), new_left, new_right)
            }
        }
    }
}

pub(crate) fn trace_get(
    snapshot: &Checkpoint,
    builder: &mut MrtPartialBuilder,
    key: &[u8],
) -> Result<Option<Vec<u8>>> {
    // Query-proof point-get rides the shared `get_descent` over the snapshot, with
    // `install_arc` as its visit hook — the same key-route descent the transcript
    // tracer and the verifier use, so every get surface shares one descent.
    tree::validate_key_len(key, "trace get")?;
    let Some(root) = snapshot.root.as_ref() else {
        return Ok(None);
    };
    let mut visit = |arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
    Ok(get_descent(root, key, &mut visit)?.map(|value| value.to_vec()))
}

pub(crate) fn trace_range(
    snapshot: &Checkpoint,
    builder: &mut MrtPartialBuilder,
    start: &[u8],
    end: &[u8],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    tree::validate_key_len(start, "trace range start")?;
    tree::validate_key_len(end, "trace range end")?;
    assert!(
        start <= end,
        "inverted half-open range: start {:?} must be <= end {:?}",
        start,
        end
    );
    if start == end {
        record_snapshot_root(builder, snapshot);
        return Ok(Vec::new());
    }
    trace_forward_scan(snapshot, builder, start, |key| key >= end)
}

pub(crate) fn trace_range_inclusive(
    snapshot: &Checkpoint,
    builder: &mut MrtPartialBuilder,
    start: &[u8],
    end: &[u8],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    tree::validate_key_len(start, "trace inclusive range start")?;
    tree::validate_key_len(end, "trace inclusive range end")?;
    assert!(
        start <= end,
        "inverted inclusive range: start {:?} must be <= end {:?}",
        start,
        end
    );
    trace_forward_scan(snapshot, builder, start, |key| key > end)
}

fn trace_forward_scan(
    snapshot: &Checkpoint,
    builder: &mut MrtPartialBuilder,
    start: &[u8],
    stop: impl Fn(&[u8]) -> bool,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let Some(root) = snapshot.root.as_ref() else {
        return Ok(Vec::new());
    };

    let mut cursor = Cursor::empty();
    let mut visit = |_: &RoutePrefix, arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
    cursor.reset_to_ge(root, start, &mut visit)?;

    let mut out = Vec::new();
    while let Some((key, value)) = cursor.next_leaf(&mut visit)? {
        if stop(&key) {
            break;
        }
        out.push((key, value.to_vec()));
    }
    Ok(out)
}

/// Scans `[start, …)` over the *current* (post-writes-so-far) tree until `stop`
/// fires on a key, returning the read values and **installing every node the scan
/// visits** straight into `builder` via the cursor's `visit` hook. Because the
/// verifier replays reads on a tree structurally identical to the current one and
/// runs this same cursor, the set installed here is exactly the set its range scan
/// touches. Visited nodes unmodified since the snapshot share the snapshot's
/// `Arc`, so installing them records the pre-state node `assemble_mrt_pruned`
/// keeps; nodes created by earlier writes are unreachable from the snapshot root
/// and are harmlessly dropped at assembly.
fn scan_current_install(
    current_root: Option<&Arc<MrtNodeInner>>,
    builder: &mut MrtPartialBuilder,
    start: &[u8],
    stop: impl Fn(&[u8]) -> bool,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let Some(root) = current_root else {
        return Ok(Vec::new());
    };

    let mut visit = |_: &RoutePrefix, arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
    let mut cursor = Cursor::empty();
    cursor.reset_to_ge(root, start, &mut visit)?;

    let mut out = Vec::new();
    while let Some((key, value)) = cursor.next_leaf(&mut visit)? {
        if stop(&key) {
            break;
        }
        out.push((key, value.to_vec()));
    }
    Ok(out)
}

pub(crate) struct MrtTracer {
    snapshot: Checkpoint,
    current: Checkpoint,
    builder: MrtPartialBuilder,
}

impl MrtTracer {
    pub(crate) fn new(snapshot: Checkpoint) -> Self {
        Self {
            current: snapshot.clone(),
            snapshot,
            builder: MrtPartialBuilder::new(),
        }
    }

    pub(crate) fn record_snapshot_root(&mut self) {
        record_snapshot_root(&mut self.builder, &self.snapshot);
    }

    pub(crate) fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Drive the shared `get_descent` over the *current* tree with `install_arc`
        // as its visit hook: the verifier's get runs the same descent over its
        // structurally-identical replayed tree, so the nodes recorded here are
        // exactly the ones it routes into. Visited nodes unmodified since the
        // snapshot share its `Arc` (recorded as pre-state); modified ones are
        // unreachable from the snapshot root and dropped at assembly. The value is
        // read from the current (post-writes-so-far) tree along the descent.
        // Bind `root`/`builder` to locals first so the `visit` closure borrows
        // `self.builder` without conflicting with the `self.current.root` read.
        // Validate before the empty-tree early return (which would otherwise skip
        // `get_descent`'s own check), so an over-long key is rejected regardless of
        // tree occupancy — matching `MrtVerifyTree::get`.
        tree::validate_key_len(key, "trace get")?;
        let root = self.current.root.clone();
        let builder = &mut self.builder;
        let Some(root) = root.as_ref() else {
            return Ok(None);
        };
        let mut visit = |arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
        Ok(get_descent(root, key, &mut visit)?.map(|value| value.to_vec()))
    }

    pub(crate) fn range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Mirror the verifier's `collect_range`: validate both bounds first, then
        // treat an empty/inverted range as touching nothing (record nothing).
        // Validating up front (before the early return, which can skip the cursor's
        // own check) keeps `create_trace` from accepting a range that the verifier
        // would later reject.
        tree::validate_key_len(start, "trace range start")?;
        tree::validate_key_len(end, "trace range end")?;
        if start >= end {
            return Ok(Vec::new());
        }
        let root = self.current.root.clone();
        scan_current_install(root.as_ref(), &mut self.builder, start, |key| key >= end)
    }

    pub(crate) fn prefix(&mut self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Mirror the verifier's `collect_prefix`: validate before the (possibly
        // empty-tree) early return inside `scan_current_install`, so an invalid
        // prefix is rejected here rather than only at verify time.
        tree::validate_key_len(prefix, "trace prefix")?;
        let root = self.current.root.clone();
        scan_current_install(root.as_ref(), &mut self.builder, prefix, |key| {
            !key.starts_with(prefix)
        })
    }

    pub(crate) fn apply_step(&mut self, batch: &Batch) -> Result<()> {
        self.record_snapshot_root();
        validate_mrt_apply_batch(batch)?;

        // Apply each op to the CURRENT tree, recording **by touched node** in every
        // case: `insert_with_trace`/`delete_with_trace`/`delete_range_with_trace`
        // hand each node the op dereferences to the visit closure, which installs it
        // directly — the same record-what-you-touch the reads use, so the verifier
        // (descending the structurally-identical current tree) routes into exactly
        // these nodes. Unmodified nodes share the snapshot `Arc` (kept at assembly);
        // nodes built by earlier writes/moves are unreachable from the snapshot root
        // and dropped. Recording Put on the *current* tree (rather than the snapshot
        // path) is load-bearing once `move_prefix` is in play: a relocated key no
        // longer sits at its snapshot route, so a snapshot-path reveal would miss the
        // moved subtree's interior and the verifier would hit a pruned node.
        let mut root = self.current.root.clone();
        for (key, op) in batch.iter() {
            let builder = &mut self.builder;
            match op {
                Op::Put(value) => {
                    let mut collect = |arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
                    root = Some(tree::insert_with_trace(
                        root,
                        key.clone(),
                        value.clone(),
                        &mut collect,
                    )?);
                }
                Op::Delete => {
                    let mut collect =
                        |_p: &RoutePrefix, arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
                    let (next, _) = delete_with_trace(root, key, &mut collect)?;
                    root = next;
                }
                Op::DeleteRange(end) => {
                    let mut collect =
                        |_p: &RoutePrefix, arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
                    root = delete_range_with_trace(root, key, end, &mut collect)?;
                }
            }
        }

        self.current = Checkpoint { root };
        Ok(())
    }

    pub(crate) fn delete_prefix(&mut self, prefix: &[u8]) -> Result<()> {
        self.record_snapshot_root();
        let root = self.current.root.clone();
        let builder = &mut self.builder;
        let mut collect = |_p: &RoutePrefix, arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
        let root = delete_prefix_with_trace(root, prefix, &mut collect)?;
        self.current = Checkpoint { root };
        Ok(())
    }

    /// Applies a `MovePrefix` step to the CURRENT tree, recording **by touched
    /// node** exactly like Delete/DeleteRange: `move_prefix_with_trace` navigates
    /// the real detach-then-splice and hands each pre-state node it touches (the
    /// `from`-path, `S`'s original root, the source-collapse survivor, and the
    /// `to`-splice path on the post-detach tree) to the install hook. Synthetic
    /// nodes the op builds are not snapshot `Arc`s, so `walk_keep_visited` drops
    /// them; the verifier rebuilds them at replay. `S`'s interior stays pruned.
    pub(crate) fn move_prefix(&mut self, from: &[u8], to: &[u8]) -> Result<()> {
        self.record_snapshot_root();
        tree::validate_move_prefix_args(from, to)?;
        let root = self.current.root.clone();
        let builder = &mut self.builder;
        let mut record = |arc: &Arc<MrtNodeInner>| install_arc(builder, arc);
        let new_root = tree::move_prefix_with_trace(root, from, to, &mut record)?;
        self.current = Checkpoint {
            root: Some(new_root),
        };
        Ok(())
    }

    pub(crate) fn into_trace(self) -> Trace {
        assemble_mrt_pruned(self.builder)
    }
}

/// Prove-side traced read handle for the MRT backend — the externalized peer of
/// [`crate::avl::TraceRecorder`].
///
/// Built over a live [`Checkpoint`], it answers sequenced reads
/// ([`TraceReader`](crate::tracer::TraceReader)) and sequenced writes
/// ([`TraceInterface`](crate::tracer::TraceInterface)) against a working copy of
/// the snapshot while recording every node they touch, then emits the witness with
/// [`finalize_trace`](Self::finalize_trace) in the same `Trace` wire format
/// `create_trace` produces.
pub struct TraceRecorder {
    tracer: MrtTracer,
    reads: Vec<ReadOp>,
    /// Set once an [`apply`](crate::tracer::TraceInterface::apply) op fails. A
    /// poisoned recorder rejects every subsequent fallible op (reads, further
    /// `apply`, `finalize_trace`) with [`Error::Poisoned`]; `reads()` stays
    /// infallible (see its doc).
    poisoned: bool,
}

impl TraceRecorder {
    /// Start recording traced operations against `snapshot`.
    pub fn new(snapshot: &Checkpoint) -> Self {
        let mut tracer = MrtTracer::new(snapshot.clone());
        // Record the snapshot root so an empty (no-read) recorder still emits a
        // trace that binds to the start root — matching `create_trace`.
        tracer.record_snapshot_root();
        Self {
            tracer,
            reads: Vec::new(),
            poisoned: false,
        }
    }

    /// The read ops issued through this recorder, in order — the read-set a
    /// consumer claims alongside the trace.
    ///
    /// **Poison exemption (precondition).** This stays infallible even after a
    /// failed `apply` poisons the handle, returning the pre-poison read log. That
    /// is sound only if the caller issued *all* reads before the failing write —
    /// merk cannot enforce caller ordering, so honoring this ordering is the
    /// caller's precondition.
    pub fn reads(&self) -> &[ReadOp] {
        &self.reads
    }

    /// Reject any fallible op once the handle is poisoned.
    fn reject_if_poisoned(&self, op: &str) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned(format!(
                "{op} on a poisoned MRT recorder (an earlier apply failed)"
            )));
        }
        Ok(())
    }

    /// Finish recording and emit the witness as encoded `Trace` bytes — the bytes
    /// [`crate::mrt::TraceReplayer::new_verified`] consumes. Semantically
    /// equivalent to the legacy `create_trace` output for the same reads, though
    /// not contractually byte-for-byte identical. Fails [`Error::Poisoned`] if a
    /// prior `apply` op failed.
    pub fn finalize_trace(self) -> Result<Vec<u8>> {
        self.reject_if_poisoned("finalize_trace")?;
        // Encoding a trace into an in-memory buffer is infallible.
        Ok(self
            .tracer
            .into_trace()
            .encode()
            .expect("encoding an MRT trace to an in-memory buffer is infallible"))
    }
}

impl crate::tracer::TraceReader for TraceRecorder {
    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.reject_if_poisoned("get")?;
        let value = self.tracer.get(key)?;
        self.reads.push(ReadOp::Key(key.to_vec()));
        Ok(value)
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_range")?;
        let rows = self.tracer.range(start, end)?;
        self.reads.push(ReadOp::Range {
            start: start.to_vec(),
            end: end.to_vec(),
        });
        Ok(rows)
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_prefix")?;
        let rows = self.tracer.prefix(prefix)?;
        self.reads.push(ReadOp::Prefix(prefix.to_vec()));
        Ok(rows)
    }
}

/// Records each `WriteOp` **in vector order, one at a time**, lowering it to the
/// same per-op primitive the imperative methods used: point/range ops to a
/// single-op `apply_step`, `DeletePrefix`/`MovePrefix` to the tracer's prefix and
/// move primitives. A `Vec<WriteOp>` is never gathered into a sorted multi-op
/// batch. An empty batch is a no-op. The first failing op poisons the recorder.
impl crate::tracer::TraceInterface for TraceRecorder {
    fn apply(&mut self, ops: &[crate::tracer::WriteOp]) -> Result<()> {
        use crate::tracer::WriteOp;
        self.reject_if_poisoned("apply")?;
        for op in ops {
            let result = match op {
                WriteOp::Put { key, value } => self
                    .tracer
                    .apply_step(&[(key.clone(), Op::Put(value.clone()))]),
                WriteOp::Delete { key } => self.tracer.apply_step(&[(key.clone(), Op::Delete)]),
                WriteOp::DeleteRange { start, end } => self
                    .tracer
                    .apply_step(&[(start.clone(), Op::DeleteRange(end.clone()))]),
                WriteOp::DeletePrefix { prefix } => self.tracer.delete_prefix(prefix),
                WriteOp::MovePrefix { from, to } => self.tracer.move_prefix(from, to),
            };
            if let Err(err) = result {
                self.poisoned = true;
                return Err(err);
            }
        }
        Ok(())
    }
}

// ── Stage 6/7: TraceRecorder handle tests ──────────────────────────────────
#[cfg(test)]
mod recorder_tests {
    use super::TraceRecorder;
    use crate::error::Error;
    use crate::mrt::tree::{node_visits, reset_node_visits};
    use crate::mrt::{Checkpoint, TraceReplayer, Tree};
    use crate::tracer::test_support::mrt::create_trace;
    use crate::tracer::{prefix_successor, ReadOp, TraceInterface, TraceReader, WriteOp};

    fn build(entries: &[(&[u8], &[u8])]) -> Checkpoint {
        let tree = Tree::new();
        for (key, value) in entries {
            tree.put(*key, *value).unwrap();
        }
        tree.checkpoint()
    }

    #[test]
    fn recorder_reads_match_snapshot_and_record_ops() {
        let snapshot = build(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let mut recorder = TraceRecorder::new(&snapshot);

        assert_eq!(recorder.get(b"c").unwrap(), Some(b"3".to_vec()));
        // Witnessed non-membership reads back as `Ok(None)`.
        assert_eq!(recorder.get(b"b").unwrap(), None);
        assert_eq!(
            recorder.get_range(b"a", b"f").unwrap(),
            snapshot.collect_range(b"a", Some(b"f")).unwrap()
        );
        assert_eq!(
            recorder.get_prefix(b"e").unwrap(),
            snapshot.collect_prefix(b"e").unwrap()
        );

        // `reads()` records exactly the ops issued, in order.
        assert_eq!(
            recorder.reads(),
            &[
                ReadOp::Key(b"c".to_vec()),
                ReadOp::Key(b"b".to_vec()),
                ReadOp::Range {
                    start: b"a".to_vec(),
                    end: b"f".to_vec(),
                },
                ReadOp::Prefix(b"e".to_vec()),
            ]
        );
    }

    #[test]
    fn recorder_trace_round_trips_through_replayer() {
        let snapshot = build(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let start = snapshot.root_hash();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.get(b"c").unwrap();
        recorder.get_range(b"a", b"z").unwrap();
        let bytes = recorder.finalize_trace().unwrap();

        // The recorder bytes decode, bind to the start root, and replay the same
        // reads — an honest recorder trace never causes a read-side `PrunedNode`.
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
    fn empty_recorder_binds_to_start_root() {
        let snapshot = build(&[(b"aaa", b"1"), (b"mmm", b"2"), (b"zzz", b"3")]);
        let start = snapshot.root_hash();

        let recorder = TraceRecorder::new(&snapshot);
        assert!(recorder.reads().is_empty());
        let bytes = recorder.finalize_trace().unwrap();
        // No reads: the pruned-children root still authenticates the start root.
        assert!(TraceReplayer::new_verified(&bytes, start).is_ok());
    }

    #[test]
    fn recorder_is_object_safe() {
        let snapshot = build(&[(b"a", b"1")]);
        let mut recorder = TraceRecorder::new(&snapshot);
        let reader: &mut dyn TraceReader = &mut recorder;
        assert_eq!(reader.get(b"a").unwrap(), Some(b"1".to_vec()));
    }

    #[test]
    fn recorder_sequenced_writes_round_trip_through_replayer() {
        let snapshot = build(&[
            (b"aa", b"1"),
            (b"bb", b"2"),
            (b"cc", b"3"),
            (b"user:1", b"u1"),
            (b"user:2", b"u2"),
            (b"zz", b"z"),
        ]);
        let start = snapshot.root_hash();

        let expected = Tree::new();
        expected.restore(Some(snapshot.clone()));
        expected.put(b"dd".to_vec(), b"4".to_vec()).unwrap();
        expected.delete(b"aa".to_vec()).unwrap();
        expected.delete_range(b"b".to_vec(), b"d".to_vec()).unwrap();
        expected
            .delete_range(b"user:".to_vec(), prefix_successor(b"user:").unwrap())
            .unwrap();
        expected.put(b"acct:3".to_vec(), b"3".to_vec()).unwrap();
        expected
            .move_prefix(b"acct:".to_vec(), b"arch:".to_vec())
            .unwrap();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.put(b"dd", b"4").unwrap();
        assert_eq!(recorder.get(b"dd").unwrap(), Some(b"4".to_vec()));
        recorder.delete(b"aa").unwrap();
        assert_eq!(recorder.get(b"aa").unwrap(), None);
        recorder.delete_range(b"b", b"d").unwrap();
        assert!(recorder.get_range(b"b", b"d").unwrap().is_empty());
        recorder.delete_prefix(b"user:").unwrap();
        assert!(recorder.get_prefix(b"user:").unwrap().is_empty());
        recorder.put(b"acct:3", b"3").unwrap();
        recorder.move_prefix(b"acct:", b"arch:").unwrap();
        assert_eq!(recorder.get(b"acct:3").unwrap(), None);
        assert_eq!(recorder.get(b"arch:3").unwrap(), Some(b"3".to_vec()));

        let bytes = recorder.finalize_trace().unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.put(b"dd", b"4").unwrap();
        assert_eq!(replayer.get(b"dd").unwrap(), Some(b"4".to_vec()));
        replayer.delete(b"aa").unwrap();
        assert_eq!(replayer.get(b"aa").unwrap(), None);
        replayer.delete_range(b"b", b"d").unwrap();
        assert!(replayer.get_range(b"b", b"d").unwrap().is_empty());
        replayer.delete_prefix(b"user:").unwrap();
        assert!(replayer.get_prefix(b"user:").unwrap().is_empty());
        replayer.put(b"acct:3", b"3").unwrap();
        replayer.move_prefix(b"acct:", b"arch:").unwrap();
        assert_eq!(replayer.get(b"acct:3").unwrap(), None);
        assert_eq!(replayer.get(b"arch:3").unwrap(), Some(b"3".to_vec()));

        assert_eq!(replayer.root_hash().unwrap(), expected.root_hash());
    }

    #[test]
    fn recorder_delete_prefix_handles_no_successor_prefix() {
        let snapshot = build(&[
            (&[0xfe][..], b"before".as_slice()),
            (&[0xff, 0x00][..], b"first".as_slice()),
            (&[0xff, 0x10][..], b"second".as_slice()),
        ]);
        let start = snapshot.root_hash();
        let expected = build(&[(&[0xfe][..], b"before".as_slice())]).root_hash();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.delete_prefix(&[0xff]).unwrap();
        assert!(recorder.get_prefix(&[0xff]).unwrap().is_empty());
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.delete_prefix(&[0xff]).unwrap();
        assert!(replayer.get_prefix(&[0xff]).unwrap().is_empty());
        assert_eq!(replayer.root_hash().unwrap(), expected);
    }

    #[test]
    fn recorder_write_errors_use_expected_buckets() {
        let snapshot = build(&[(b"ab", b"1")]);
        // A failing `apply` op poisons the handle, so each error-bucket check
        // needs a fresh recorder/replayer — otherwise the second op would return
        // `Error::Poisoned` instead of its own bucket.
        let mut recorder = TraceRecorder::new(&snapshot);
        assert!(matches!(recorder.put(b"a", b"x"), Err(Error::Key(_))));
        let mut recorder = TraceRecorder::new(&snapshot);
        assert!(matches!(
            recorder.delete_range(b"z", b"a"),
            Err(Error::BatchKey(_))
        ));

        let trace = create_trace(&snapshot, &[]).unwrap();
        let start = trace.root_hash();
        let bytes = trace.encode().unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        assert!(matches!(replayer.put(b"a", b"x"), Err(Error::Key(_))));
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        assert!(matches!(
            replayer.delete_range(b"z", b"a"),
            Err(Error::BatchKey(_))
        ));
    }

    fn recorder_generic_write<T: TraceInterface>(handle: &mut T) -> Option<Vec<u8>> {
        handle.put(b"k", b"v1").unwrap();
        handle.delete(b"k").unwrap();
        handle.put(b"k", b"v2").unwrap();
        handle.get(b"k").unwrap()
    }

    #[test]
    fn recorder_trace_interface_object_and_generic_paths_run() {
        let snapshot = build(&[]);
        let mut object_recorder = TraceRecorder::new(&snapshot);
        let handle: &mut dyn TraceInterface = &mut object_recorder;
        handle.put(b"a", b"1").unwrap();
        assert_eq!(handle.get(b"a").unwrap(), Some(b"1".to_vec()));
        handle.delete_prefix(b"a").unwrap();
        assert_eq!(handle.get(b"a").unwrap(), None);

        let mut generic_recorder = TraceRecorder::new(&snapshot);
        assert_eq!(
            recorder_generic_write(&mut generic_recorder),
            Some(b"v2".to_vec())
        );
    }

    // ── Stage 3: WriteOp `apply` surface ────────────────────────────────────

    /// Drive a recorder over `snapshot` with the given `apply` calls (each inner
    /// slice is one `apply()` invocation) and then point-read `read_keys`;
    /// finalize and replay the exact same sequence through a verified replayer.
    /// Asserts the recorder and replayer agree on every read, then returns the
    /// replayer's end root and the (agreed) point-read results.
    fn drive_apply_and_reads(
        snapshot: &Checkpoint,
        apply_calls: &[&[WriteOp]],
        read_keys: &[&[u8]],
    ) -> (crate::Hash, Vec<Option<Vec<u8>>>) {
        let start = snapshot.root_hash();

        let mut recorder = TraceRecorder::new(snapshot);
        for call in apply_calls {
            recorder.apply(call).unwrap();
        }
        let recorder_reads: Vec<Option<Vec<u8>>> =
            read_keys.iter().map(|k| recorder.get(k).unwrap()).collect();
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        for call in apply_calls {
            replayer.apply(call).unwrap();
        }
        let replayer_reads: Vec<Option<Vec<u8>>> =
            read_keys.iter().map(|k| replayer.get(k).unwrap()).collect();
        let root = replayer.root_hash().unwrap();

        assert_eq!(
            recorder_reads, replayer_reads,
            "recorder and replayer disagree on reads"
        );
        (root, recorder_reads)
    }

    fn apply_batch_then_read(
        snapshot: &Checkpoint,
        ops: &[WriteOp],
        read_keys: &[&[u8]],
    ) -> Vec<Option<Vec<u8>>> {
        drive_apply_and_reads(snapshot, &[ops], read_keys).1
    }

    fn put(key: &[u8], value: &[u8]) -> WriteOp {
        WriteOp::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        }
    }

    // C1 conflict table (duplicates allowed, last-writer-wins by position). MRT
    // keys must be mutually prefix-free, so the cases use fixed-width keys.

    #[test]
    fn apply_conflict_put_put_keeps_last() {
        let snap = build(&[(b"aa", b"1"), (b"zz", b"26")]);
        let got = apply_batch_then_read(&snap, &[put(b"kk", b"a"), put(b"kk", b"b")], &[b"kk"]);
        assert_eq!(got, vec![Some(b"b".to_vec())]);
    }

    #[test]
    fn apply_conflict_put_delete_is_absent() {
        let snap = build(&[(b"aa", b"1"), (b"zz", b"26")]);
        let got = apply_batch_then_read(
            &snap,
            &[
                put(b"kk", b"a"),
                WriteOp::Delete {
                    key: b"kk".to_vec(),
                },
            ],
            &[b"kk"],
        );
        assert_eq!(got, vec![None]);
    }

    #[test]
    fn apply_conflict_delete_range_then_put_in_range_keeps_put() {
        let snap = build(&[(b"aa", b"1"), (b"mm", b"13"), (b"zz", b"26")]);
        let ops = vec![
            WriteOp::DeleteRange {
                start: b"aa".to_vec(),
                end: b"nn".to_vec(),
            },
            put(b"mm", b"a"),
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"mm"]);
        assert_eq!(got, vec![Some(b"a".to_vec())]);
    }

    #[test]
    fn apply_conflict_put_then_delete_range_over_it_is_absent() {
        let snap = build(&[(b"aa", b"1"), (b"zz", b"26")]);
        let ops = vec![
            put(b"mm", b"a"),
            WriteOp::DeleteRange {
                start: b"aa".to_vec(),
                end: b"nn".to_vec(),
            },
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"mm"]);
        assert_eq!(got, vec![None]);
    }

    #[test]
    fn apply_conflict_put_then_delete_prefix_is_absent() {
        let snap = build(&[(b"aa", b"1"), (b"zz", b"26")]);
        let ops = vec![
            put(b"pre_k", b"a"),
            WriteOp::DeletePrefix {
                prefix: b"pre_".to_vec(),
            },
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"pre_k"]);
        assert_eq!(got, vec![None]);
    }

    #[test]
    fn apply_conflict_delete_prefix_then_put_keeps_put() {
        let snap = build(&[(b"pre_a", b"1"), (b"pre_b", b"2"), (b"zz", b"26")]);
        let ops = vec![
            WriteOp::DeletePrefix {
                prefix: b"pre_".to_vec(),
            },
            put(b"pre_k", b"a"),
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"pre_k", b"pre_a"]);
        assert_eq!(got, vec![Some(b"a".to_vec()), None]);
    }

    #[test]
    fn apply_conflict_put_under_from_is_moved() {
        let snap = build(&[(b"user:a", b"1"), (b"zz", b"26")]);
        let ops = vec![
            put(b"user:new", b"a"),
            WriteOp::MovePrefix {
                from: b"user:".to_vec(),
                to: b"acct:".to_vec(),
            },
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"acct:new", b"user:new"]);
        assert_eq!(got, vec![Some(b"a".to_vec()), None]);
    }

    #[test]
    fn apply_move_prefix_overwrites_destination() {
        let snap = build(&[(b"user:a", b"1"), (b"zz", b"26")]);
        let mut recorder = TraceRecorder::new(&snap);
        // A Put lands a key under the destination prefix, then the move overwrites
        // the whole destination: the change succeeds and `acct:` is replaced by
        // `user:`'s subtree (the Put under `acct:` is discarded, not merged).
        recorder
            .apply(&[
                put(b"acct:x", b"a"),
                WriteOp::MovePrefix {
                    from: b"user:".to_vec(),
                    to: b"acct:".to_vec(),
                },
            ])
            .unwrap();
        assert_eq!(recorder.get(b"acct:a").unwrap(), Some(b"1".to_vec())); // user:a -> acct:a
        assert_eq!(recorder.get(b"acct:x").unwrap(), None); // discarded by overwrite
        assert_eq!(recorder.get(b"user:a").unwrap(), None); // relocated away
        assert_eq!(recorder.get(b"zz").unwrap(), Some(b"26".to_vec())); // untouched
        assert!(recorder.finalize_trace().is_ok());
    }

    #[test]
    fn apply_delete_prefix_0xff_clears_the_whole_span() {
        let snap = build(&[
            (&[0xfe, 0x00][..], b"keep".as_slice()),
            (&[0xff, 0x00][..], b"a".as_slice()),
            (&[0xff, 0x80][..], b"b".as_slice()),
            (&[0xff, 0xff][..], b"c".as_slice()),
        ]);
        let start = snap.root_hash();
        let ops = [WriteOp::DeletePrefix { prefix: vec![0xff] }];

        let mut recorder = TraceRecorder::new(&snap);
        recorder.apply(&ops).unwrap();
        assert!(recorder.get_prefix(&[0xff]).unwrap().is_empty());
        assert_eq!(recorder.get(&[0xfe, 0x00]).unwrap(), Some(b"keep".to_vec()));
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.apply(&ops).unwrap();
        assert!(replayer.get_prefix(&[0xff]).unwrap().is_empty());
        assert_eq!(replayer.get(&[0xfe, 0x00]).unwrap(), Some(b"keep".to_vec()));

        let expected = build(&[(&[0xfe, 0x00][..], b"keep".as_slice())]).root_hash();
        assert_eq!(replayer.root_hash().unwrap(), expected);
    }

    #[test]
    fn apply_empty_batch_is_a_noop() {
        let snap = build(&[(b"aa", b"1"), (b"cc", b"3")]);
        let start = snap.root_hash();

        let mut recorder = TraceRecorder::new(&snap);
        recorder.apply(&[]).unwrap();
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.apply(&[]).unwrap();
        assert_eq!(replayer.root_hash().unwrap(), start);
    }

    #[test]
    fn apply_batch_equivalent_to_individual_calls() {
        let snap = build(&[
            (b"aa", b"1"),
            (b"bb", b"2"),
            (b"cc", b"3"),
            (b"user:1", b"u1"),
            (b"user:2", b"u2"),
            (b"zz", b"z"),
        ]);
        let ops = vec![
            put(b"dd", b"4"),
            WriteOp::Delete {
                key: b"aa".to_vec(),
            },
            WriteOp::DeleteRange {
                start: b"b".to_vec(),
                end: b"d".to_vec(),
            },
            WriteOp::DeletePrefix {
                prefix: b"user:".to_vec(),
            },
            put(b"acct:3", b"3"),
            WriteOp::MovePrefix {
                from: b"acct:".to_vec(),
                to: b"arch:".to_vec(),
            },
        ];
        let read_keys: &[&[u8]] = &[b"dd", b"aa", b"arch:3", b"acct:3"];

        // One apply() call with the whole batch …
        let (batch_root, batch_reads) = drive_apply_and_reads(&snap, &[&ops], read_keys);
        // … vs the same ops, one apply() call each. Equivalent by construction.
        let individual: Vec<&[WriteOp]> = ops.iter().map(std::slice::from_ref).collect();
        let (indiv_root, indiv_reads) = drive_apply_and_reads(&snap, &individual, read_keys);

        assert_eq!(batch_root, indiv_root);
        assert_eq!(batch_reads, indiv_reads);

        // …and both match a live store applying the same ops one at a time
        // (DeletePrefix emulated by its half-open range, which has a successor).
        let oracle = Tree::new();
        oracle.restore(Some(snap.clone()));
        oracle.put(b"dd".to_vec(), b"4".to_vec()).unwrap();
        oracle.delete(b"aa".to_vec()).unwrap();
        oracle.delete_range(b"b".to_vec(), b"d".to_vec()).unwrap();
        oracle
            .delete_range(b"user:".to_vec(), prefix_successor(b"user:").unwrap())
            .unwrap();
        oracle.put(b"acct:3".to_vec(), b"3".to_vec()).unwrap();
        oracle
            .move_prefix(b"acct:".to_vec(), b"arch:".to_vec())
            .unwrap();
        assert_eq!(batch_root, oracle.root_hash());
    }

    #[test]
    fn apply_failure_poisons_mrt_recorder() {
        let snap = build(&[(b"user:a", b"1"), (b"zz", b"26")]);
        let mut recorder = TraceRecorder::new(&snap);
        // A pre-poison read is preserved by the `reads()` exemption.
        recorder.get(b"user:a").unwrap();

        // The Put applies, then an invalid MovePrefix (equal source/destination)
        // fails: the whole change is rejected and the handle poisoned.
        let err = recorder
            .apply(&[
                put(b"acct:x", b"a"),
                WriteOp::MovePrefix {
                    from: b"user:".to_vec(),
                    to: b"user:".to_vec(),
                },
            ])
            .unwrap_err();
        // The first failure is the op's own error (equal prefixes), not `Poisoned`.
        assert!(matches!(err, Error::Key(_)));

        // Every subsequent fallible op fails closed with `Poisoned`.
        assert!(matches!(recorder.get(b"user:a"), Err(Error::Poisoned(_))));
        assert!(matches!(
            recorder.apply(&[put(b"xx", b"x")]),
            Err(Error::Poisoned(_))
        ));
        // `reads()` stays infallible and returns the pre-poison log.
        assert_eq!(recorder.reads(), &[ReadOp::Key(b"user:a".to_vec())]);
        assert!(matches!(recorder.finalize_trace(), Err(Error::Poisoned(_))));
    }

    /// Perf gate: an N-op `apply` over an M-entry tree must stay path-bounded
    /// (~O(N·log M)), never whole-tree (~O(N·M)). Proxy: `MrtNodeInner::node()`
    /// dereferences during recording.
    fn apply_node_visits(m: usize) -> usize {
        let merk = Tree::new();
        for i in 0..m {
            merk.put(format!("{i:08}").into_bytes(), b"v".to_vec())
                .unwrap();
        }
        let snapshot = merk.checkpoint();
        let ops: Vec<WriteOp> = (0..8)
            .map(|i| put(format!("{i:08}").as_bytes(), b"v2"))
            .collect();

        let mut recorder = TraceRecorder::new(&snapshot);
        reset_node_visits();
        recorder.apply(&ops).unwrap();
        let visits = node_visits();
        // Finalize so the recorder is fully exercised (and the trace is valid).
        recorder.finalize_trace().unwrap();
        visits
    }

    #[test]
    fn apply_visits_scale_with_path_not_tree_size() {
        let small = apply_node_visits(1000);
        let large = apply_node_visits(4000);
        assert!(
            large <= small * 2,
            "apply node visits scaled with tree size, not accessed path: \
             {} at n=1000 -> {} at n=4000 (O(n^2) regression?)",
            small,
            large
        );
        assert!(
            large < 1000,
            "expected ~path-bounded node visits, got {} for n=4000",
            large
        );
    }
}
