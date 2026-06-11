use std::collections::{HashMap, HashSet};

use crate::avl::child::Child;
use crate::avl::in_memory::InMemoryMerk;
use crate::avl::node::Node;
use crate::avl::walker::{Fetch, Walker};
use crate::error::{Error, Result};
use crate::hash::{kv_hash, Hash, Hasher};
use crate::ops::{BatchEntry, Op};

use super::assembly::{accessed_keys_from_nodes, assemble_sparse_trace};
use super::{BatchOp, RecordingSource, SparseMerkNode};

/// `Fetch` source for verifier-side replay over a sparse trace skeleton.
///
/// Opened trace nodes are converted to resident [`Node`]s. `Pruned` children
/// become [`Child::Pruned`] placeholders, and any attempted descent into one
/// means the write proof omitted a node required by the AVL mutation path.
#[derive(Clone)]
struct TraceFetch;

impl Fetch for TraceFetch {
    fn fetch_by_key(&self, key: &[u8]) -> Result<Option<Node>> {
        Err(Error::PrunedNode(format!(
            "AVL replay descended into pruned subtree rooted at {key:?}"
        )))
    }
}

#[derive(Clone, Default)]
struct HashOnlyProvenance {
    omitted: HashMap<Vec<u8>, Hash>,
    storage_hash: HashMap<Vec<u8>, Hash>,
}

impl HashOnlyProvenance {
    fn collect(trace: &SparseMerkNode) -> Self {
        let mut provenance = Self::default();
        provenance.collect_inner(trace);
        provenance
    }

    fn collect_inner(&mut self, trace: &SparseMerkNode) {
        match trace {
            SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => {}
            SparseMerkNode::Full { left, right, .. } => {
                self.collect_inner(left);
                self.collect_inner(right);
            }
            SparseMerkNode::FullStorageHash {
                key,
                kv_hash,
                left,
                right,
            } => {
                self.storage_hash.insert(key.clone(), *kv_hash);
                self.collect_inner(left);
                self.collect_inner(right);
            }
            SparseMerkNode::FullOmitted {
                key,
                kv_hash,
                left,
                right,
            } => {
                self.omitted.insert(key.clone(), *kv_hash);
                self.collect_inner(left);
                self.collect_inner(right);
            }
        }
    }

    fn remove_key(&mut self, key: &[u8]) {
        self.omitted.remove(key);
        self.storage_hash.remove(key);
    }

    fn remove_range(&mut self, start: &[u8], end: &[u8]) {
        self.omitted
            .retain(|key, _| key.as_slice() < start || key.as_slice() >= end);
        self.storage_hash
            .retain(|key, _| key.as_slice() < start || key.as_slice() >= end);
    }
}

/// Trace keys that must be emitted as full values after replay because the
/// proof write step supplied materialized bytes for them.
#[derive(Clone, Default)]
struct ReplayOutputPolicy {
    provenance: HashOnlyProvenance,
    full_value_keys: HashSet<Vec<u8>>,
}

impl ReplayOutputPolicy {
    fn from_trace_and_ops(trace: &SparseMerkNode, ops: &[BatchOp]) -> Self {
        let mut policy = Self {
            provenance: HashOnlyProvenance::collect(trace),
            full_value_keys: HashSet::new(),
        };

        for op in ops {
            match op {
                BatchOp::Put { key, .. } => {
                    policy.provenance.remove_key(key);
                    policy.full_value_keys.insert(key.clone());
                }
                BatchOp::Delete { key } => {
                    policy.provenance.remove_key(key);
                    policy.full_value_keys.remove(key);
                }
                BatchOp::DeleteRange { start, end } => {
                    policy.provenance.remove_range(start, end);
                    policy
                        .full_value_keys
                        .retain(|key| key.as_slice() < start || key.as_slice() >= end);
                }
            }
        }

        policy
    }
}

/// Replay a proof-level write batch over a sparse AVL trace.
///
/// It converts the sparse trace into a `Node` / `Child` skeleton, applies the
/// point-write and split/join delete-range algorithms, and converts the result
/// **back** into a sparse trace.
///
/// ⚠️ Each call is two O(trace) conversions (`node_skeleton_from_trace` +
/// `trace_from_node_skeleton`) plus a `commit()`. Calling it once per change in a
/// transcript is therefore **O(n²)** over the sequence. For a change sequence,
/// decode once with [`node_skeleton_from_trace`] and replay each batch in place
/// with [`replay_writes_on_node_tree`], committing once at the end (this is what
/// [`TraceVerifier`] does). Prefer this only for a single, one-shot replay.
///
/// [`TraceVerifier`]: crate::avl::TraceVerifier
pub fn replay_sparse_writes(trace: SparseMerkNode, ops: &[BatchOp]) -> Result<SparseMerkNode> {
    replay_sparse_writes_with_read_targets(trace, ops, &HashSet::new())
}

/// Replay writes while forcing known read-target keys to stay `Full` when the
/// replayed skeleton still contains materialized value bytes for them.
pub fn replay_sparse_writes_with_read_targets(
    trace: SparseMerkNode,
    ops: &[BatchOp],
    read_target_keys: &HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    if ops.is_empty() {
        return Ok(trace);
    }

    let batch = batch_entries_from_ops(ops);
    InMemoryMerk::validate_batch(&batch)?;

    let policy = ReplayOutputPolicy::from_trace_and_ops(&trace, ops);
    let tree = node_skeleton_from_trace(&trace)?;
    let mut maybe_tree = apply_batch_entries(tree, &batch, TraceFetch)?;
    if let Some(ref mut tree) = maybe_tree {
        tree.commit();
    }

    trace_from_node_skeleton(maybe_tree.as_ref(), &policy, read_target_keys)
}

/// Replay a write batch over an already-built verifier tree, in place.
///
/// Unlike [`replay_sparse_writes_with_read_targets`], this does **not** convert
/// the result back to a sparse trace or commit the tree — it is O(batch), not
/// O(trace). The optimized verifier ([`TraceVerifier`]) decodes the trace into a
/// `Node` tree once with [`node_skeleton_from_trace`], keeps that tree alive
/// across transcript steps replaying each write batch with this function, and
/// commits + checks the root once at the end. Driving a change sequence this way
/// is O(total writes); using [`replay_sparse_writes`] per change instead is
/// O(n²) (two O(trace) conversions + a commit per change).
///
/// [`TraceVerifier`]: crate::avl::TraceVerifier
pub fn replay_writes_on_node_tree(tree: Option<Node>, ops: &[BatchOp]) -> Result<Option<Node>> {
    if ops.is_empty() {
        return Ok(tree);
    }

    let batch = batch_entries_from_ops(ops);
    InMemoryMerk::validate_batch(&batch)?;
    apply_batch_entries(tree, &batch, TraceFetch)
}

/// Replay a sequenced `delete_prefix(prefix)` over an already-built verifier
/// tree. This is intentionally narrower than a public open-ended range API: the
/// no-successor prefix case is the only place AVL needs `[prefix, ∞)` internally.
pub(super) fn replay_delete_prefix_on_node_tree(
    tree: Option<Node>,
    prefix: &[u8],
) -> Result<Option<Node>> {
    match super::prefix_successor(prefix) {
        Some(end) => replay_writes_on_node_tree(
            tree,
            &[BatchOp::DeleteRange {
                start: prefix.to_vec(),
                end,
            }],
        ),
        None => {
            let Some(tree) = tree else {
                return Ok(None);
            };
            let walker = Walker::new(tree, TraceFetch);
            let (left, _deleted_suffix) = walker.split_at(prefix)?;
            Ok(left)
        }
    }
}

/// Trace a materialized write batch against a full original tree, returning
/// the sparse pre-state proof tree and the post-state AVL tree.
///
/// The returned sparse tree authenticates against `root`; replaying `ops`
/// against it with [`replay_sparse_writes`] reproduces `post_tree`'s root hash.
pub fn trace_and_apply_writes(
    root: Option<&Node>,
    ops: &[BatchOp],
) -> Result<(SparseMerkNode, Option<Node>)> {
    let (mut accessed_keys, post_tree) = apply_writes_recording_accessed(root.cloned(), ops)?;
    accessed_keys.retain(|key| original_tree_contains(root, key));
    let sparse = assemble_sparse_trace(root, &accessed_keys, &HashSet::new())?;
    Ok((sparse, post_tree))
}

/// Apply a write batch while recording every node visited during the mutation.
///
/// Returns the set of visited keys and the post-mutation tree. The caller is
/// responsible for filtering the accessed keys (e.g. retaining only keys
/// present in the original tree) before sparse-tree assembly.
pub(super) fn apply_writes_recording_accessed(
    root: Option<Node>,
    ops: &[BatchOp],
) -> Result<(HashSet<Vec<u8>>, Option<Node>)> {
    let batch = batch_entries_from_ops(ops);
    InMemoryMerk::validate_batch(&batch)?;

    let source = RecordingSource::new();
    let mut post_tree = apply_batch_entries(root, &batch, source.clone())?;
    if let Some(ref mut tree) = post_tree {
        tree.commit();
    }

    let visits = source.into_visits()?;
    let accessed_keys = accessed_keys_from_nodes(&visits);
    Ok((accessed_keys, post_tree))
}

/// Apply a sequenced `delete_prefix(prefix)` while recording every node visited
/// during the mutation.
///
/// When `prefix` has a byte successor this is the usual half-open range delete.
/// The no-successor edge (`[]` and all-`0xff` prefixes) is an internal open-ended
/// split at `prefix`, matching verifier replay's `delete_prefix` path without
/// exposing an unbounded public range API.
pub(super) fn apply_delete_prefix_recording_accessed(
    root: Option<Node>,
    prefix: &[u8],
) -> Result<(HashSet<Vec<u8>>, Option<Node>)> {
    if let Some(end) = super::prefix_successor(prefix) {
        return apply_writes_recording_accessed(
            root,
            &[BatchOp::DeleteRange {
                start: prefix.to_vec(),
                end,
            }],
        );
    }

    let source = RecordingSource::new();
    let mut post_tree = match root {
        Some(tree) => {
            let walker = Walker::new(tree, source.clone());
            let (left, _deleted_suffix) = walker.split_at(prefix)?;
            left
        }
        None => None,
    };
    if let Some(ref mut tree) = post_tree {
        tree.commit();
    }

    let visits = source.into_visits()?;
    let accessed_keys = accessed_keys_from_nodes(&visits);
    Ok((accessed_keys, post_tree))
}

pub(super) fn original_tree_contains(root: Option<&Node>, key: &[u8]) -> bool {
    let mut cursor = match root {
        Some(root) => root,
        None => return false,
    };

    loop {
        if key == cursor.key() {
            return true;
        }
        let left = key < cursor.key();
        match cursor.child_ref(left) {
            Some(Child::Resident(child)) => cursor = child,
            Some(Child::Pruned(_)) | None => return false,
        }
    }
}

/// Collect the BST path from `root` to `key`, returning every ancestor key
/// (including `key` itself if found). Used to ensure that assembly can reach
/// every accessed key through the original tree structure even when the
/// current tree has been rebalanced by earlier writes.
pub(super) fn original_tree_path_keys(root: Option<&Node>, key: &[u8]) -> Vec<Vec<u8>> {
    let mut path = Vec::new();
    let mut cursor = match root {
        Some(root) => root,
        None => return path,
    };

    loop {
        path.push(cursor.key().to_vec());
        if key == cursor.key() {
            return path;
        }
        let left = key < cursor.key();
        match cursor.child_ref(left) {
            Some(Child::Resident(child)) => cursor = child,
            Some(Child::Pruned(_)) | None => return path,
        }
    }
}

fn batch_entries_from_ops(ops: &[BatchOp]) -> Vec<BatchEntry> {
    ops.iter().map(BatchOp::to_batch_entry).collect()
}

fn apply_batch_entries<S>(
    root: Option<Node>,
    batch: &[BatchEntry],
    source: S,
) -> Result<Option<Node>>
where
    S: Fetch + Sized + Send + Clone,
{
    let mut current_tree = root;
    let mut i = 0;

    while i < batch.len() {
        if let Op::DeleteRange(ref end) = batch[i].1 {
            let start = &batch[i].0;
            let maybe_walker = current_tree
                .take()
                .map(|node| Walker::new(node, source.clone()));
            current_tree = Walker::delete_range_apply_to(maybe_walker, start, end)?;
            i += 1;
        } else {
            let segment_start = i;
            while i < batch.len() && !matches!(batch[i].1, Op::DeleteRange(_)) {
                i += 1;
            }
            let segment = &batch[segment_start..i];
            let maybe_walker = current_tree
                .take()
                .map(|node| Walker::new(node, source.clone()));
            let (new_tree, _deleted_keys) =
                Walker::apply_to_mut(maybe_walker, &mut segment.to_vec(), source.clone())?;
            current_tree = new_tree;
        }
    }

    Ok(current_tree)
}

/// Decodes an authenticated sparse proof into a live `Node` verifier tree (pruned
/// subtrees become hash-only stubs). Done **once** by the optimized verifier,
/// which then reads + replays writes against the live tree in place — see
/// [`replay_writes_on_node_tree`].
pub fn node_skeleton_from_trace(trace: &SparseMerkNode) -> Result<Option<Node>> {
    match trace {
        SparseMerkNode::Empty => Ok(None),
        SparseMerkNode::Pruned { key, .. } => Err(Error::PrunedNode(format!(
            "root of sparse AVL trace is pruned at {key:?}"
        ))),
        SparseMerkNode::Full { .. }
        | SparseMerkNode::FullStorageHash { .. }
        | SparseMerkNode::FullOmitted { .. } => trace_to_resident_node(trace).map(Some),
    }
}

fn trace_to_child(trace: &SparseMerkNode) -> Result<Option<Child>> {
    match trace {
        SparseMerkNode::Empty => Ok(None),
        SparseMerkNode::Pruned {
            key,
            hash,
            child_heights,
        } => {
            trace.try_height()?;
            Ok(Some(Child::pruned(key.clone(), *hash, *child_heights)))
        }
        SparseMerkNode::Full { .. }
        | SparseMerkNode::FullStorageHash { .. }
        | SparseMerkNode::FullOmitted { .. } => {
            trace_to_resident_node(trace).map(Child::Resident).map(Some)
        }
    }
}

fn trace_to_resident_node(trace: &SparseMerkNode) -> Result<Node> {
    let mut node = match trace {
        SparseMerkNode::Full {
            key,
            value,
            left,
            right,
        } => Node::from_fields(
            key.clone(),
            value.clone(),
            kv_hash::<Hasher>(key, value)?,
            trace_to_child(left)?,
            trace_to_child(right)?,
        ),
        SparseMerkNode::FullStorageHash {
            key,
            kv_hash,
            left,
            right,
        }
        | SparseMerkNode::FullOmitted {
            key,
            kv_hash,
            left,
            right,
        } => Node::from_fields(
            key.clone(),
            Vec::new(),
            *kv_hash,
            trace_to_child(left)?,
            trace_to_child(right)?,
        ),
        SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => {
            unreachable!("trace_to_resident_node called for non-resident trace")
        }
    };
    node.commit();
    Ok(node)
}

fn trace_from_node_skeleton(
    node: Option<&Node>,
    policy: &ReplayOutputPolicy,
    read_target_keys: &HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    match node {
        None => Ok(SparseMerkNode::Empty),
        Some(node) => resident_node_to_trace(node, policy, read_target_keys),
    }
}

fn child_to_trace(
    child: Option<&Child>,
    policy: &ReplayOutputPolicy,
    read_target_keys: &HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    match child {
        None => Ok(SparseMerkNode::Empty),
        Some(Child::Pruned(pruned)) => Ok(SparseMerkNode::Pruned {
            key: pruned.key().to_vec(),
            hash: *pruned.node_hash(),
            child_heights: pruned.child_heights(),
        }),
        Some(Child::Resident(node)) => resident_node_to_trace(node, policy, read_target_keys),
    }
}

fn resident_node_to_trace(
    node: &Node,
    policy: &ReplayOutputPolicy,
    read_target_keys: &HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    let left = child_to_trace(node.child_ref(true), policy, read_target_keys)?;
    let right = child_to_trace(node.child_ref(false), policy, read_target_keys)?;
    let key = node.key().to_vec();

    if policy.full_value_keys.contains(node.key()) {
        ensure_materialized_value_matches(node)?;
        return Ok(SparseMerkNode::Full {
            key,
            value: node.value().to_vec(),
            left: Box::new(left),
            right: Box::new(right),
        });
    }

    if let Some(kv_hash) = policy.provenance.storage_hash.get(node.key()) {
        ensure_kv_hash_matches(node, kv_hash)?;
        return Ok(SparseMerkNode::FullStorageHash {
            key,
            kv_hash: *kv_hash,
            left: Box::new(left),
            right: Box::new(right),
        });
    }

    if let Some(kv_hash) = policy.provenance.omitted.get(node.key()) {
        ensure_kv_hash_matches(node, kv_hash)?;
        if read_target_keys.contains(node.key()) {
            return Err(Error::ValueOmitted(format!(
                "read target {:?} remained proof-omitted after write replay",
                node.key()
            )));
        }
        return Ok(SparseMerkNode::FullOmitted {
            key,
            kv_hash: *kv_hash,
            left: Box::new(left),
            right: Box::new(right),
        });
    }

    ensure_materialized_value_matches(node)?;
    SparseMerkNode::full_or_omitted(
        key,
        node.value().to_vec(),
        read_target_keys.contains(node.key()),
        left,
        right,
    )
}

fn ensure_kv_hash_matches(node: &Node, kv_hash: &Hash) -> Result<()> {
    if *kv_hash == *node.kv_hash() {
        Ok(())
    } else {
        Err(Error::Tree(format!(
            "hash-only provenance for key {:?} does not match replayed node",
            node.key()
        )))
    }
}

fn ensure_materialized_value_matches(node: &Node) -> Result<()> {
    let materialized_kv_hash = kv_hash::<Hasher>(node.key(), node.value())?;
    ensure_kv_hash_matches(node, &materialized_kv_hash).map_err(|_| {
        Error::Tree(format!(
            "replayed node {:?} has no materialized value provenance",
            node.key()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::HASH_LENGTH;

    fn build_tree(entries: &[(Vec<u8>, Vec<u8>)]) -> Option<Node> {
        let merk = InMemoryMerk::new();
        for (key, value) in entries {
            merk.put(key.clone(), value.clone()).unwrap();
        }
        merk.checkpoint().into_root()
    }

    fn full_trace_from_node(node: Option<&Node>) -> SparseMerkNode {
        match node {
            None => SparseMerkNode::Empty,
            Some(node) => SparseMerkNode::Full {
                key: node.key().to_vec(),
                value: node.value().to_vec(),
                left: Box::new(full_trace_from_node(node.child(true))),
                right: Box::new(full_trace_from_node(node.child(false))),
            },
        }
    }

    fn find_trace_node<'a>(trace: &'a SparseMerkNode, key: &[u8]) -> Option<&'a SparseMerkNode> {
        match trace {
            SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => None,
            SparseMerkNode::Full {
                key: node_key,
                left,
                right,
                ..
            }
            | SparseMerkNode::FullStorageHash {
                key: node_key,
                left,
                right,
                ..
            }
            | SparseMerkNode::FullOmitted {
                key: node_key,
                left,
                right,
                ..
            } => {
                if key == node_key.as_slice() {
                    Some(trace)
                } else if key < node_key.as_slice() {
                    find_trace_node(left, key)
                } else {
                    find_trace_node(right, key)
                }
            }
        }
    }

    fn large_value(byte: u8) -> Vec<u8> {
        vec![byte; crate::tracer::SMALL_VALUE_INLINE_THRESHOLD + 1]
    }

    #[test]
    fn replay_full_trace_matches_live_point_writes() {
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u8..8).map(|i| (vec![i], vec![i + 10])).collect();
        let root = build_tree(&entries);
        let trace = full_trace_from_node(root.as_ref());
        let ops = vec![
            BatchOp::Put {
                key: vec![2],
                value: b"two".to_vec(),
            },
            BatchOp::Delete { key: vec![5] },
            BatchOp::Put {
                key: vec![10],
                value: b"ten".to_vec(),
            },
        ];

        let live = InMemoryMerk::new();
        for (key, value) in &entries {
            live.put(key.clone(), value.clone()).unwrap();
        }
        let batch = batch_entries_from_ops(&ops);
        live.apply_sorted_batch_ops(&batch).unwrap();

        let replayed = replay_sparse_writes(trace, &ops).unwrap();
        assert_eq!(replayed.hash(), live.root_hash());
        replayed.verify_root(live.root_hash()).unwrap();
    }

    #[test]
    fn replay_errors_when_write_descends_into_pruned_child() {
        let root = SparseMerkNode::Full {
            key: b"m".to_vec(),
            value: b"root".to_vec(),
            left: Box::new(SparseMerkNode::Pruned {
                key: b"a".to_vec(),
                hash: [7; HASH_LENGTH],
                child_heights: (0, 0),
            }),
            right: Box::new(SparseMerkNode::Empty),
        };
        let ops = vec![BatchOp::Put {
            key: b"a".to_vec(),
            value: b"new".to_vec(),
        }];

        let err = replay_sparse_writes(root, &ops).unwrap_err();
        assert!(matches!(err, Error::PrunedNode(_)), "got {:?}", err);
    }

    #[test]
    fn replay_errors_when_rotation_needs_pruned_heavy_child_balance() {
        let root = SparseMerkNode::Full {
            key: vec![10],
            value: b"root".to_vec(),
            left: Box::new(SparseMerkNode::Pruned {
                key: vec![5],
                hash: [5; HASH_LENGTH],
                child_heights: (2, 2),
            }),
            right: Box::new(SparseMerkNode::Full {
                key: vec![20],
                value: b"right".to_vec(),
                left: Box::new(SparseMerkNode::Full {
                    key: vec![15],
                    value: b"right-left".to_vec(),
                    left: Box::new(SparseMerkNode::Empty),
                    right: Box::new(SparseMerkNode::Empty),
                }),
                right: Box::new(SparseMerkNode::Empty),
            }),
        };
        let ops = vec![BatchOp::Delete { key: vec![15] }];

        let err = replay_sparse_writes(root, &ops).unwrap_err();
        assert!(matches!(err, Error::PrunedNode(_)), "got {:?}", err);
    }

    #[test]
    fn replay_preserves_omitted_nodes_and_expands_materialized_puts() {
        let entries = vec![
            (vec![0], large_value(0)),
            (vec![1], large_value(1)),
            (vec![2], large_value(2)),
            (vec![3], large_value(3)),
        ];
        let root = build_tree(&entries);
        let mut trace = full_trace_from_node(root.as_ref());

        fn omit_all(trace: &mut SparseMerkNode) {
            match trace {
                SparseMerkNode::Full {
                    key,
                    value,
                    left,
                    right,
                } => {
                    omit_all(left);
                    omit_all(right);
                    *trace = SparseMerkNode::FullOmitted {
                        key: key.clone(),
                        kv_hash: kv_hash::<Hasher>(key, value).unwrap(),
                        left: left.clone(),
                        right: right.clone(),
                    };
                }
                SparseMerkNode::FullStorageHash { left, right, .. }
                | SparseMerkNode::FullOmitted { left, right, .. } => {
                    omit_all(left);
                    omit_all(right);
                }
                SparseMerkNode::Empty | SparseMerkNode::Pruned { .. } => {}
            }
        }
        omit_all(&mut trace);

        let ops = vec![BatchOp::Put {
            key: vec![2],
            value: b"updated-two".to_vec(),
        }];
        let replayed = replay_sparse_writes(trace, &ops).unwrap();

        assert!(matches!(
            find_trace_node(&replayed, &[2]).unwrap(),
            SparseMerkNode::Full { value, .. } if value == b"updated-two"
        ));
        assert!(matches!(
            find_trace_node(&replayed, &[0]).unwrap(),
            SparseMerkNode::FullOmitted { .. }
        ));
    }

    #[test]
    fn replay_preserves_input_full_large_values() {
        let entries = vec![
            (vec![0], large_value(0)),
            (vec![1], large_value(1)),
            (vec![2], large_value(2)),
        ];
        let root = build_tree(&entries);
        let trace = full_trace_from_node(root.as_ref());

        let replayed = replay_sparse_writes(
            trace,
            &[BatchOp::Put {
                key: vec![9],
                value: b"unrelated".to_vec(),
            }],
        )
        .unwrap();

        assert!(matches!(
            find_trace_node(&replayed, &[1]).unwrap(),
            SparseMerkNode::Full { value, .. } if value == &large_value(1)
        ));
        assert_eq!(
            replayed.get(&[1]).unwrap(),
            Some(large_value(1)),
            "materialized input-Full values must remain readable after replay"
        );
    }

    #[test]
    fn trace_and_replay_materialized_write_batch_matches_post_root() {
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u8..16)
            .map(|i| (vec![i], format!("v{i}").into_bytes()))
            .collect();
        let root = build_tree(&entries).unwrap();
        let ops = vec![
            BatchOp::Delete { key: vec![3] },
            BatchOp::Put {
                key: vec![20],
                value: b"twenty".to_vec(),
            },
        ];

        let (sparse, post_tree) = trace_and_apply_writes(Some(&root), &ops).unwrap();
        let replayed = replay_sparse_writes(sparse, &ops).unwrap();

        assert_eq!(replayed.hash(), post_tree.unwrap().hash());
    }

    #[test]
    fn trace_and_replay_delete_range_uses_split_join_result() {
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u8..24)
            .map(|i| (vec![i], format!("v{i}").into_bytes()))
            .collect();
        let root = build_tree(&entries).unwrap();
        let ops = vec![BatchOp::DeleteRange {
            start: vec![5],
            end: vec![18],
        }];

        let (sparse, post_tree) = trace_and_apply_writes(Some(&root), &ops).unwrap();
        let replayed = replay_sparse_writes(sparse, &ops).unwrap();

        assert_eq!(replayed.hash(), post_tree.unwrap().hash());
    }
}
