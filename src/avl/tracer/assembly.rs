use std::collections::HashSet;

use crate::avl::child::Child;
use crate::avl::node::Node;
use crate::error::{Error, Result};
use crate::hash::{node_hash, Hasher, NULL_HASH};

use super::SparseMerkNode;

/// Collect the unique keys for a set of accessed AVL nodes.
pub fn accessed_keys_from_nodes<'a, I>(nodes: I) -> HashSet<Vec<u8>>
where
    I: IntoIterator<Item = &'a Node>,
{
    nodes.into_iter().map(|node| node.key().to_vec()).collect()
}

/// Assemble a sparse AVL trace from an original full tree plus access sets.
///
/// Nodes whose keys are in `accessed_keys` are opened as
/// [`SparseMerkNode::Full`]. Untouched subtrees are represented as
/// [`SparseMerkNode::Pruned`].
pub fn assemble_sparse_trace(
    root: Option<&Node>,
    accessed_keys: &HashSet<Vec<u8>>,
    read_target_keys: &HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    let mut opened_keys = HashSet::new();
    let mut opened_read_targets = HashSet::new();
    let trace = assemble_optional_node(
        root,
        accessed_keys,
        read_target_keys,
        &mut opened_keys,
        &mut opened_read_targets,
    )?;

    if let Some(key) = accessed_keys.difference(&opened_keys).next() {
        return Err(Error::Tree(format!(
            "accessed key {key:?} was not found in the original AVL tree"
        )));
    }

    if let Some(key) = read_target_keys.difference(&opened_read_targets).next() {
        return Err(Error::Tree(format!(
            "read target key {key:?} was not opened as a full sparse proof node"
        )));
    }

    Ok(trace)
}

fn assemble_optional_node(
    node: Option<&Node>,
    accessed_keys: &HashSet<Vec<u8>>,
    read_target_keys: &HashSet<Vec<u8>>,
    opened_keys: &mut HashSet<Vec<u8>>,
    opened_read_targets: &mut HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    match node {
        Some(node) => assemble_node(
            node,
            accessed_keys,
            read_target_keys,
            opened_keys,
            opened_read_targets,
        ),
        None => Ok(SparseMerkNode::Empty),
    }
}

fn assemble_child(
    child: Option<&Child>,
    accessed_keys: &HashSet<Vec<u8>>,
    read_target_keys: &HashSet<Vec<u8>>,
    opened_keys: &mut HashSet<Vec<u8>>,
    opened_read_targets: &mut HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    match child {
        None => Ok(SparseMerkNode::Empty),
        Some(Child::Resident(node)) => assemble_node(
            node,
            accessed_keys,
            read_target_keys,
            opened_keys,
            opened_read_targets,
        ),
        Some(Child::Pruned(pruned)) => Ok(SparseMerkNode::Pruned {
            key: pruned.key().to_vec(),
            hash: *pruned.node_hash(),
            child_heights: pruned.child_heights(),
        }),
    }
}

fn assemble_node(
    node: &Node,
    accessed_keys: &HashSet<Vec<u8>>,
    read_target_keys: &HashSet<Vec<u8>>,
    opened_keys: &mut HashSet<Vec<u8>>,
    opened_read_targets: &mut HashSet<Vec<u8>>,
) -> Result<SparseMerkNode> {
    if !accessed_keys.contains(node.key()) {
        return Ok(SparseMerkNode::Pruned {
            key: node.key().to_vec(),
            hash: resident_node_hash(node),
            child_heights: node.child_heights(),
        });
    }

    opened_keys.insert(node.key().to_vec());
    let read_target = read_target_keys.contains(node.key());
    if read_target {
        opened_read_targets.insert(node.key().to_vec());
    }

    // Validate that opened nodes still have materialized value bytes matching
    // their stored kv_hash before emitting them as Full proof nodes.
    materialized_kv_hash(node)?;
    let left = assemble_child(
        node.child_ref(true),
        accessed_keys,
        read_target_keys,
        opened_keys,
        opened_read_targets,
    )?;
    let right = assemble_child(
        node.child_ref(false),
        accessed_keys,
        read_target_keys,
        opened_keys,
        opened_read_targets,
    )?;

    Ok(SparseMerkNode::Full {
        key: node.key().to_vec(),
        value: node.value().to_vec(),
        left: Box::new(left),
        right: Box::new(right),
    })
}

fn materialized_kv_hash(node: &Node) -> Result<crate::Hash> {
    let kv_hash = crate::hash::kv_hash::<Hasher>(node.key(), node.value())?;
    if kv_hash != *node.kv_hash() {
        return Err(Error::Tree(format!(
            "cannot open hash-only AVL node {:?} without materialized value bytes",
            node.key()
        )));
    }
    Ok(kv_hash)
}

#[cfg(test)]
thread_local! {
    /// Counts `resident_node_hash` invocations so the regression guard test
    /// (`assembly_visits_scale_with_path_not_tree_size`) can assert sparse-trace
    /// assembly cost scales with the accessed path, not the tree size.
    static RESIDENT_HASH_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn resident_node_hash(node: &Node) -> crate::Hash {
    #[cfg(test)]
    RESIDENT_HASH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let left = child_hash(node.child_ref(true));
    let right = child_hash(node.child_ref(false));
    node_hash::<Hasher>(node.kv_hash(), &left, &right)
}

fn child_hash(child: Option<&Child>) -> crate::Hash {
    match child {
        None => NULL_HASH,
        Some(Child::Pruned(pruned)) => *pruned.node_hash(),
        // Committed resident subtrees (the only kind the tracer is handed in
        // practice — `replace_root` -> `commit()` runs after every batch) have
        // their Merkle hash cached in `node_hash`, so read it in O(1). Recursing
        // here instead made per-change sparse-trace assembly O(tree size), i.e.
        // O(n^2) over a bulk seed, because pruning an accessed path's siblings
        // re-hashed ~the whole resident tree. Fall back to recomputation only
        // for uncommitted nodes (no cached hash), preserving correctness for
        // arbitrarily deep uncommitted subtrees.
        Some(Child::Resident(node)) => match node.node_hash() {
            Some(hash) => *hash,
            None => resident_node_hash(node),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::SMALL_VALUE_INLINE_THRESHOLD;

    fn committed_tree() -> Node {
        let left = Node::new(b"a".to_vec(), b"left value".to_vec()).unwrap();
        let right = Node::new(b"z".to_vec(), vec![9; SMALL_VALUE_INLINE_THRESHOLD + 1]).unwrap();
        let mut root = Node::new(b"m".to_vec(), b"root value".to_vec())
            .unwrap()
            .attach(true, Some(left))
            .attach(false, Some(right));
        root.commit();
        root
    }

    fn set(keys: &[&[u8]]) -> HashSet<Vec<u8>> {
        keys.iter().map(|key| key.to_vec()).collect()
    }

    #[test]
    fn assemble_empty_tree_is_empty_trace() {
        let trace =
            assemble_sparse_trace(None, &HashSet::new(), &HashSet::new()).expect("assemble");

        assert_eq!(trace, SparseMerkNode::Empty);
    }

    #[test]
    fn assemble_prunes_unaccessed_resident_subtrees() {
        let root = committed_tree();
        let accessed = set(&[b"m"]);
        let read_targets = set(&[b"m"]);

        let trace = assemble_sparse_trace(Some(&root), &accessed, &read_targets).unwrap();

        assert_eq!(trace.hash(), root.hash());
        match trace {
            SparseMerkNode::Full { left, right, .. } => {
                assert!(matches!(
                    left.as_ref(),
                    SparseMerkNode::Pruned { key, .. } if key == b"a"
                ));
                assert!(matches!(
                    right.as_ref(),
                    SparseMerkNode::Pruned { key, .. } if key == b"z"
                ));
            }
            other => panic!("expected full root, got {:?}", other),
        }
    }

    #[test]
    fn assemble_prunes_uncommitted_resident_subtrees() {
        let leaf = Node::new(b"a".to_vec(), b"leaf value".to_vec()).unwrap();
        let left = Node::new(b"c".to_vec(), b"left value".to_vec())
            .unwrap()
            .attach(true, Some(leaf));
        let root = Node::new(b"m".to_vec(), b"root value".to_vec())
            .unwrap()
            .attach(true, Some(left));
        let mut committed = root.clone();
        committed.commit();
        let accessed = set(&[b"m"]);
        let read_targets = set(&[b"m"]);

        let trace = assemble_sparse_trace(Some(&root), &accessed, &read_targets).unwrap();

        assert_eq!(trace.hash(), committed.hash());
        match trace {
            SparseMerkNode::Full { left, .. } => {
                assert!(matches!(
                    left.as_ref(),
                    SparseMerkNode::Pruned { key, .. } if key == b"c"
                ));
            }
            other => panic!("expected full root, got {:?}", other),
        }
    }

    #[test]
    fn assemble_read_target_keeps_large_value_full() {
        let root = committed_tree();
        let accessed = set(&[b"m", b"z"]);
        let read_targets = set(&[b"z"]);

        let trace = assemble_sparse_trace(Some(&root), &accessed, &read_targets).unwrap();

        assert_eq!(trace.hash(), root.hash());
        assert_eq!(
            trace.get(b"z").unwrap(),
            Some(vec![9; SMALL_VALUE_INLINE_THRESHOLD + 1])
        );
    }

    #[test]
    fn assemble_non_target_large_values_are_full() {
        let mut exact = Node::new(b"k".to_vec(), vec![1; SMALL_VALUE_INLINE_THRESHOLD]).unwrap();
        exact.commit();
        let accessed = set(&[b"k"]);
        let trace = assemble_sparse_trace(Some(&exact), &accessed, &HashSet::new()).unwrap();
        assert!(matches!(trace, SparseMerkNode::Full { .. }));

        let mut large =
            Node::new(b"k".to_vec(), vec![1; SMALL_VALUE_INLINE_THRESHOLD + 1]).unwrap();
        large.commit();
        let trace = assemble_sparse_trace(Some(&large), &accessed, &HashSet::new()).unwrap();
        assert!(matches!(trace, SparseMerkNode::Full { .. }));
    }

    #[test]
    fn assemble_large_value_hashes_like_full_and_can_be_read() {
        let mut root = Node::new(b"k".to_vec(), vec![1; SMALL_VALUE_INLINE_THRESHOLD + 1]).unwrap();
        root.commit();
        let accessed = set(&[b"k"]);

        let trace = assemble_sparse_trace(Some(&root), &accessed, &HashSet::new()).unwrap();

        assert_eq!(trace.hash(), root.hash());
        assert_eq!(
            trace.get(b"k").unwrap(),
            Some(vec![1; SMALL_VALUE_INLINE_THRESHOLD + 1])
        );
    }

    #[test]
    fn assemble_fails_when_read_target_was_not_opened() {
        let root = committed_tree();
        let accessed = set(&[b"m"]);
        let read_targets = set(&[b"z"]);

        let err = assemble_sparse_trace(Some(&root), &accessed, &read_targets).unwrap_err();

        assert!(matches!(err, Error::Tree(msg) if msg.contains("read target key")));
    }

    #[test]
    fn assemble_fails_when_accessed_key_is_not_reachable_through_open_nodes() {
        let root = committed_tree();
        let accessed = set(&[b"z"]);

        let err = assemble_sparse_trace(Some(&root), &accessed, &HashSet::new()).unwrap_err();

        assert!(matches!(err, Error::Tree(msg) if msg.contains("accessed key")));
    }

    /// Collect the root-to-target path keys (the set the real tracer records as
    /// "accessed" for a single-key operation). Every ancestor must be present or
    /// `assemble_sparse_trace` cannot open its way down to the target.
    fn path_keys(root: &Node, target: &[u8]) -> HashSet<Vec<u8>> {
        let mut keys = HashSet::new();
        let mut cursor = Some(root);
        while let Some(node) = cursor {
            keys.insert(node.key().to_vec());
            if target == node.key() {
                break;
            }
            cursor = node.child(target < node.key());
        }
        keys
    }

    /// Build an N-key committed tree, assemble a single-key sparse trace, and
    /// return how many subtree hashes assembly had to (re)compute.
    fn assembly_resident_hash_calls(n: usize) -> usize {
        let merk = crate::avl::in_memory::InMemoryMerk::new();
        for i in 0..n {
            // Fixed-width keys so the same target exists in trees of any size.
            merk.put(format!("{i:08}").into_bytes(), b"v".to_vec())
                .unwrap();
        }
        let root = merk.checkpoint().into_root().expect("non-empty tree");
        let target = b"00000000";
        let accessed = path_keys(&root, target);

        RESIDENT_HASH_CALLS.with(|calls| calls.set(0));
        assemble_sparse_trace(Some(&root), &accessed, &HashSet::new()).expect("assemble");
        RESIDENT_HASH_CALLS.with(|calls| calls.get())
    }

    /// Guards against re-introducing the O(n^2) regression: assembling a
    /// single-key trace must cost O(accessed path), not O(tree size). The buggy
    /// version recursed into every pruned subtree, so 4x the data did ~4x the
    /// hashing; the fix reads the cached `node_hash` and grows only ~log n.
    #[test]
    fn assembly_visits_scale_with_path_not_tree_size() {
        let small = assembly_resident_hash_calls(1000);
        let large = assembly_resident_hash_calls(4000);

        // 4x the data must not ~4x the work. Linear assembly would; a
        // path-bounded one barely moves. 2x leaves ample room for AVL height
        // growth while still catching a return to linear (let alone quadratic).
        assert!(
            large <= small * 2,
            "sparse-trace assembly scaled with tree size, not accessed path: \
             {} hashes at n=1000 -> {} at n=4000 (O(n^2) regression?)",
            small,
            large
        );
        // ...and the absolute count must stay near the tree height, not n.
        assert!(
            large < 100,
            "expected ~path-length hashing, got {} for n=4000 (O(n) assembly?)",
            large
        );
    }

    #[test]
    fn accessed_keys_from_nodes_deduplicates_visit_log() {
        let root = committed_tree();
        let left = root.child(true).unwrap();
        let keys = accessed_keys_from_nodes([&root, left, &root]);

        assert_eq!(keys, set(&[b"m", b"a"]));
    }
}
