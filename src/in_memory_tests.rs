use super::*;
use crate::hash::NULL_HASH;
use crate::node::Node;
#[cfg(not(use_box))]
use crate::ops::PanicSource;
use crate::proofs::Query;
#[cfg(not(use_box))]
use crate::walker::Walker;
use crate::Op;

fn query_for_keys(keys: &[&[u8]]) -> Query {
    let mut query = Query::new();
    for key in keys {
        query.insert_key((*key).to_vec());
    }
    query
}

#[test]
fn test_new_empty() {
    let merk = InMemoryMerk::new();
    assert_eq!(merk.root_hash(), NULL_HASH);
    assert!(merk.get(b"foo").is_none());
}

#[test]
fn test_apply_and_get() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (vec![1], Op::Put(vec![10])),
        (vec![2], Op::Put(vec![20])),
        (vec![3], Op::Put(vec![30])),
    ])
    .unwrap();

    assert_eq!(merk.get(&[1]), Some(vec![10]));
    assert_eq!(merk.get(&[2]), Some(vec![20]));
    assert_eq!(merk.get(&[3]), Some(vec![30]));
    assert!(merk.get(&[4]).is_none());
    assert_ne!(merk.root_hash(), NULL_HASH);
}

#[test]
fn test_apply_delete() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[(vec![1], Op::Put(vec![10]))]).unwrap();
    assert!(merk.get(&[1]).is_some());

    merk.apply_batch(&[(vec![1], Op::Delete)]).unwrap();
    assert!(merk.get(&[1]).is_none());
    assert_eq!(merk.root_hash(), NULL_HASH);
}

#[test]
fn test_snapshot_isolation() {
    let merk = InMemoryMerk::new();
    merk.put(vec![1], vec![10]).unwrap();

    let snap = merk.snapshot().unwrap();
    let snap_hash = snap.hash();

    merk.put(vec![2], vec![20]).unwrap();

    assert_eq!(snap.hash(), snap_hash);
    assert!(snap.get(&[2]).is_none());
    assert_eq!(snap.get(&[1]), Some(vec![10]));
    assert_ne!(merk.root_hash(), snap_hash);
}

#[test]
fn test_prove_and_verify() {
    let merk = InMemoryMerk::new();
    let batch = vec![
        (b"a".to_vec(), Op::Put(b"val_a".to_vec())),
        (b"b".to_vec(), Op::Put(b"val_b".to_vec())),
        (b"c".to_vec(), Op::Put(b"val_c".to_vec())),
    ];
    merk.apply_batch(&batch).unwrap();

    let keys = [b"b".as_slice(), b"missing".as_slice()];
    let proof_bytes = merk.prove(query_for_keys(&keys)).unwrap();

    let root_hash = merk.root_hash();
    let result =
        crate::proofs::query::verify_query(&proof_bytes, &query_for_keys(&keys), root_hash)
            .unwrap();
    assert_eq!(result, vec![(b"b".to_vec(), b"val_b".to_vec())]);
}

#[test]
fn test_tree_empty() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

#[test]
fn test_tree_traversable() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"a".to_vec(), Op::Put(b"val_a".to_vec())),
        (b"b".to_vec(), Op::Put(b"val_b".to_vec())),
        (b"c".to_vec(), Op::Put(b"val_c".to_vec())),
    ])
    .unwrap();

    let snap = merk.snapshot().unwrap();
    let tree = snap.clone();

    fn collect_keys(node: &Node, keys: &mut Vec<Vec<u8>>) {
        if let Some(left) = node.child(true) {
            collect_keys(left, keys);
        }
        keys.push(node.key().to_vec());
        if let Some(right) = node.child(false) {
            collect_keys(right, keys);
        }
    }

    let mut keys = Vec::new();
    collect_keys(&tree, &mut keys);
    assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
}

#[test]
fn test_snapshot_empty() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

#[test]
fn test_with_tree_traversal() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"a".to_vec(), Op::Put(b"val_a".to_vec())),
        (b"b".to_vec(), Op::Put(b"val_b".to_vec())),
        (b"c".to_vec(), Op::Put(b"val_c".to_vec())),
    ])
    .unwrap();

    let root = merk.snapshot().unwrap();
    fn collect(node: &Node, keys: &mut Vec<Vec<u8>>) {
        if let Some(left) = node.child(true) {
            collect(left, keys);
        }
        keys.push(node.key().to_vec());
        if let Some(right) = node.child(false) {
            collect(right, keys);
        }
    }
    let mut keys = Vec::new();
    collect(&root, &mut keys);

    assert_eq!(keys, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
}

fn make_merk_abcde() -> InMemoryMerk {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"a".to_vec(), Op::Put(b"val_a".to_vec())),
        (b"b".to_vec(), Op::Put(b"val_b".to_vec())),
        (b"c".to_vec(), Op::Put(b"val_c".to_vec())),
        (b"d".to_vec(), Op::Put(b"val_d".to_vec())),
        (b"e".to_vec(), Op::Put(b"val_e".to_vec())),
    ])
    .unwrap();
    merk
}

#[cfg(not(use_box))]
fn make_merk_abcdefg() -> InMemoryMerk {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"a".to_vec(), Op::Put(b"1".to_vec())),
        (b"b".to_vec(), Op::Put(b"2".to_vec())),
        (b"c".to_vec(), Op::Put(b"3".to_vec())),
        (b"d".to_vec(), Op::Put(b"4".to_vec())),
        (b"e".to_vec(), Op::Put(b"5".to_vec())),
        (b"f".to_vec(), Op::Put(b"6".to_vec())),
        (b"g".to_vec(), Op::Put(b"7".to_vec())),
    ])
    .unwrap();
    merk
}

fn keys(entries: &[(Vec<u8>, Vec<u8>)]) -> Vec<&[u8]> {
    entries.iter().map(|(k, _)| k.as_slice()).collect()
}

fn collect_iter(root: &Node) -> Vec<(Vec<u8>, Vec<u8>)> {
    root.iter().collect()
}

fn collect_reverse_iter(root: &Node) -> Vec<(Vec<u8>, Vec<u8>)> {
    root.reverse_iter().collect()
}

fn collect_prefix(root: &Node, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    root.iter_from(prefix)
        .take_while(|(key, _)| key.starts_with(prefix))
        .collect()
}

fn collect_prefix_reverse(root: &Node, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut entries = collect_prefix(root, prefix);
    entries.reverse();
    entries
}

fn collect_range(root: &Node, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    if start >= end {
        return Vec::new();
    }
    root.iter_from(start)
        .take_while(|(key, _)| key.as_slice() < end)
        .collect()
}

fn collect_range_reverse(root: &Node, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut entries = collect_range(root, start, end);
    entries.reverse();
    entries
}

#[cfg(not(use_box))]
fn child_with_key<'a>(node: &'a Node, left: bool, key: &[u8]) -> &'a Node {
    let child = node.child(left).expect("expected resident child");
    assert_eq!(child.key(), key);
    child
}

#[cfg(not(use_box))]
mod cow_tests {
    use super::*;

    fn node_addr(node: &Node) -> usize {
        std::sync::Arc::as_ptr(node.inner_arc()) as usize
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct SevenNodeAddrs {
        root_d: usize,
        left_b: usize,
        left_a: usize,
        left_c: usize,
        right_f: usize,
        right_e: usize,
        right_g: usize,
    }

    fn seven_node_addrs(root: &Node) -> SevenNodeAddrs {
        assert_eq!(root.key(), b"d");

        let left = child_with_key(root, true, b"b");
        let left_left = child_with_key(left, true, b"a");
        let left_right = child_with_key(left, false, b"c");

        let right = child_with_key(root, false, b"f");
        let right_left = child_with_key(right, true, b"e");
        let right_right = child_with_key(right, false, b"g");

        SevenNodeAddrs {
            root_d: node_addr(root),
            left_b: node_addr(left),
            left_a: node_addr(left_left),
            left_c: node_addr(left_right),
            right_f: node_addr(right),
            right_e: node_addr(right_left),
            right_g: node_addr(right_right),
        }
    }

    fn collect_node_addrs(root: &Node) -> std::collections::BTreeMap<Vec<u8>, usize> {
        fn collect(node: &Node, addrs: &mut std::collections::BTreeMap<Vec<u8>, usize>) {
            if let Some(left) = node.child(true) {
                collect(left, addrs);
            }
            addrs.insert(node.key().to_vec(), node_addr(node));
            if let Some(right) = node.child(false) {
                collect(right, addrs);
            }
        }

        let mut addrs = std::collections::BTreeMap::new();
        collect(root, &mut addrs);
        addrs
    }

    fn changed_node_keys(
        before: &std::collections::BTreeMap<Vec<u8>, usize>,
        after: &std::collections::BTreeMap<Vec<u8>, usize>,
    ) -> std::collections::BTreeSet<Vec<u8>> {
        before
            .iter()
            .filter_map(|(key, before_addr)| {
                let after_addr = after
                    .get(key)
                    .unwrap_or_else(|| panic!("key {:?} missing after update", key));
                (after_addr != before_addr).then(|| key.clone())
            })
            .collect()
    }

    // --- CoW structural tests ---

    #[test]
    fn test_cow_clone_root_preserves_pointer_identity_and_increments_strong_count() {
        use std::sync::Arc as StdArc;

        let merk = make_merk_abcdefg();
        let root = merk.snapshot().unwrap();
        let strong_count_before = StdArc::strong_count(root.inner_arc());

        let root_clone = root.clone();

        assert!(StdArc::ptr_eq(root.inner_arc(), root_clone.inner_arc()));
        assert_eq!(
            StdArc::strong_count(root.inner_arc()),
            strong_count_before + 1
        );

        drop(root_clone);
        assert_eq!(StdArc::strong_count(root.inner_arc()), strong_count_before);
    }

    #[test]
    fn test_cow_mutating_cloned_root_copies_only_modified_path() {
        use std::sync::Arc as StdArc;

        let merk = make_merk_abcdefg();
        let snapshot_root = merk.snapshot().unwrap();
        let before = seven_node_addrs(&snapshot_root);

        let cloned_root = snapshot_root.clone();
        assert!(StdArc::ptr_eq(
            snapshot_root.inner_arc(),
            cloned_root.inner_arc()
        ));

        let updated_root = apply_update_to_root(cloned_root, b"e", b"updated");
        let snapshot_after = seven_node_addrs(&snapshot_root);
        let updated = seven_node_addrs(&updated_root);

        assert_eq!(snapshot_after, before);
        assert_eq!(value_at(&snapshot_root, b"e"), b"5".to_vec());
        assert_eq!(value_at(&updated_root, b"e"), b"updated".to_vec());

        assert_ne!(updated.root_d, before.root_d);
        assert_ne!(updated.right_f, before.right_f);
        assert_ne!(updated.right_e, before.right_e);

        assert_eq!(updated.left_b, before.left_b);
        assert_eq!(updated.left_a, before.left_a);
        assert_eq!(updated.left_c, before.left_c);
        assert_eq!(updated.right_g, before.right_g);
    }

    #[test]
    fn test_cow_apply_batch_copies_modified_path_without_live_snapshot() {
        let merk = make_merk_abcdefg();

        let before = seven_node_addrs(&merk.snapshot().unwrap());

        merk.apply_batch(&[(b"e".to_vec(), Op::Put(b"updated".to_vec()))])
            .unwrap();

        let after = seven_node_addrs(&merk.snapshot().unwrap());

        assert_ne!(after.root_d, before.root_d);
        assert_ne!(after.right_f, before.right_f);
        assert_ne!(after.right_e, before.right_e);

        assert_eq!(after.left_b, before.left_b);
        assert_eq!(after.left_a, before.left_a);
        assert_eq!(after.left_c, before.left_c);
        assert_eq!(after.right_g, before.right_g);
        assert_eq!(merk.get(b"e"), Some(b"updated".to_vec()));
    }

    #[test]
    fn test_cow_read_only_ops_preserve_resident_node_identities() {
        let merk = make_merk_abcdefg();
        let before = seven_node_addrs(&merk.snapshot().unwrap());

        let _ = merk.get(b"e");
        let _ = merk.root_hash();
        let _ = collect_iter(&merk.snapshot().unwrap());
        let _ = merk.snapshot().unwrap().iter_from(b"e").collect::<Vec<_>>();
        let _ = collect_prefix(&merk.snapshot().unwrap(), b"a");
        let _ = collect_range(&merk.snapshot().unwrap(), b"b", b"f");

        let mut query = crate::proofs::Query::new();
        query.insert_key(b"e".to_vec());
        let _ = merk.prove(query).unwrap();

        let snap = merk.snapshot().unwrap();
        let _ = snap.get(b"e");
        let _ = snap.hash();
        let _ = collect_iter(&snap);

        let _ = &snap;

        let after = seven_node_addrs(&merk.snapshot().unwrap());
        assert_eq!(after, before);
    }

    #[test]
    fn test_prove_does_not_trigger_cow() {
        use std::sync::Arc as StdArc;

        let merk = InMemoryMerk::new();
        merk.apply_batch(&[
            (b"a".to_vec(), Op::Put(b"1".to_vec())),
            (b"b".to_vec(), Op::Put(b"2".to_vec())),
            (b"c".to_vec(), Op::Put(b"3".to_vec())),
        ])
        .unwrap();

        let snap = merk.snapshot().unwrap();
        let root_before = snap.clone();

        let mut query = crate::proofs::Query::new();
        query.insert_key(b"b".to_vec());
        let _proof = merk.prove(query).unwrap();

        let root_after = snap.clone();
        assert!(StdArc::ptr_eq(
            root_before.inner_arc(),
            root_after.inner_arc()
        ));
    }

    #[test]
    fn test_snapshot_prove_does_not_trigger_cow() {
        use std::sync::Arc as StdArc;

        let merk = InMemoryMerk::new();
        merk.apply_batch(&[
            (b"a".to_vec(), Op::Put(b"1".to_vec())),
            (b"b".to_vec(), Op::Put(b"2".to_vec())),
            (b"c".to_vec(), Op::Put(b"3".to_vec())),
        ])
        .unwrap();

        let snap = merk.snapshot().unwrap();
        let root_before = snap.clone();

        let mut query = crate::proofs::Query::new();
        query.insert_key(b"b".to_vec());
        let _proof = snap.prove(query).unwrap();

        let root_after = snap.clone();
        assert!(StdArc::ptr_eq(
            root_before.inner_arc(),
            root_after.inner_arc()
        ));
    }

    #[test]
    fn test_read_ops_preserve_pointer_identity() {
        use std::sync::Arc as StdArc;

        let merk = InMemoryMerk::new();
        merk.apply_batch(&[
            (b"a".to_vec(), Op::Put(b"1".to_vec())),
            (b"b".to_vec(), Op::Put(b"2".to_vec())),
            (b"c".to_vec(), Op::Put(b"3".to_vec())),
        ])
        .unwrap();

        let root_before = merk.snapshot().unwrap();

        let _ = merk.get(b"b");
        let _ = merk.root_hash();
        let _ = collect_iter(&merk.snapshot().unwrap());
        let _ = merk.snapshot().unwrap().iter_from(b"b").collect::<Vec<_>>();
        let _ = collect_prefix(&merk.snapshot().unwrap(), b"a");
        let _ = collect_range(&merk.snapshot().unwrap(), b"a", b"c");

        let mut query = crate::proofs::Query::new();
        query.insert_key(b"b".to_vec());
        let _ = merk.prove(query).unwrap();

        let root_after = merk.snapshot().unwrap();
        assert!(StdArc::ptr_eq(
            root_before.inner_arc(),
            root_after.inner_arc()
        ));
    }

    // --- Stress and performance gates ---

    #[test]
    fn cow_updates_copy_only_touched_paths_not_whole_tree() {
        use std::collections::BTreeSet;

        let merk = InMemoryMerk::new();
        let initial: Vec<_> = (0u8..31)
            .map(|key| (vec![key], Op::Put(vec![key])))
            .collect();
        merk.apply_batch(&initial).unwrap();

        let before_root = merk.snapshot().expect("expected root");
        let before = collect_node_addrs(&before_root);

        let update_keys = [vec![5], vec![25]];
        let expected_copied: BTreeSet<_> = update_keys
            .iter()
            .flat_map(|key| path_keys(&before_root, key))
            .collect();

        let batch = vec![
            (update_keys[0].clone(), Op::Put(b"updated-left".to_vec())),
            (update_keys[1].clone(), Op::Put(b"updated-right".to_vec())),
        ];
        merk.apply_batch(&batch).unwrap();

        let after_root = merk.snapshot().expect("expected root");
        let after = collect_node_addrs(&after_root);
        let copied = changed_node_keys(&before, &after);

        assert_eq!(before.len(), 31);
        assert_eq!(after.len(), before.len());
        assert_eq!(copied, expected_copied);
        assert!(copied.len() < before.len());

        for key in before.keys() {
            if !copied.contains(key) {
                assert_eq!(
                    before.get(key),
                    after.get(key),
                    "untouched subtree node {:?} should remain shared",
                    key
                );
            }
        }
    }

    #[test]
    fn extra_snapshots_do_not_increase_copied_path_count() {
        fn copied_keys_with_live_snapshots(
            extra_snapshots: usize,
        ) -> (
            std::collections::BTreeSet<Vec<u8>>,
            std::collections::BTreeSet<Vec<u8>>,
        ) {
            let merk = InMemoryMerk::new();
            let initial: Vec<_> = (0u8..31)
                .map(|key| (vec![key], Op::Put(vec![key])))
                .collect();
            merk.apply_batch(&initial).unwrap();

            let snapshots: Vec<_> = (0..extra_snapshots)
                .map(|_| merk.snapshot().unwrap())
                .collect();
            let before_root = merk.snapshot().expect("expected root");
            let expected_copied = path_keys(&before_root, &[21]);
            let before = collect_node_addrs(&before_root);

            merk.put(vec![21], b"updated").unwrap();

            for snap in &snapshots {
                assert_eq!(snap.get(&[21]), Some(vec![21]));
            }

            let after_root = merk.snapshot().expect("expected root");
            let after = collect_node_addrs(&after_root);
            (changed_node_keys(&before, &after), expected_copied)
        }

        let (baseline, expected_baseline) = copied_keys_with_live_snapshots(0);
        let (with_snapshots, expected_with_snapshots) = copied_keys_with_live_snapshots(8);

        assert_eq!(baseline, expected_baseline);
        assert_eq!(with_snapshots, expected_with_snapshots);
        assert_eq!(baseline, with_snapshots);
        assert!(baseline.len() < 31);
    }

    // --- CoW pointer-identity tests on large trees ---
    //
    // These tests verify CoW properties by comparing Arc pointer addresses
    // before and after mutations. No production code changes needed.

    #[test]
    fn cow_large_single_key_update_copies_only_path() {
        let n = 10_000;
        let merk = make_large_merk(n);

        let before_root = merk.snapshot().unwrap();
        let before = collect_node_addrs(&before_root);
        let update_key = 0u32.to_be_bytes().to_vec();
        let expected_path = path_keys(&before_root, &update_key);

        merk.put(update_key, b"updated").unwrap();

        let after_root = merk.snapshot().unwrap();
        let after = collect_node_addrs(&after_root);
        let changed = changed_node_keys(&before, &after);

        assert_eq!(before.len(), n);
        assert_eq!(after.len(), n);
        assert_eq!(
            changed, expected_path,
            "only nodes on the root-to-key path should change address",
        );
        assert!(
            changed.len() < n / 10,
            "changed {} of {} nodes — expected O(log n), not O(n)",
            changed.len(),
            n,
        );

        crate::test_utils::assert_tree_invariants(&merk.snapshot().unwrap());
    }

    #[test]
    fn cow_large_batch_update_copies_path_union() {
        let n = 10_000;
        let merk = make_large_merk(n);

        let before_root = merk.snapshot().unwrap();
        let before = collect_node_addrs(&before_root);

        let update_keys: Vec<Vec<u8>> = (0..10)
            .map(|i| (i as u32 * (n as u32 / 10)).to_be_bytes().to_vec())
            .collect();
        let expected_paths: std::collections::BTreeSet<_> = update_keys
            .iter()
            .flat_map(|key| path_keys(&before_root, key))
            .collect();

        let batch: Vec<_> = update_keys
            .iter()
            .map(|k| (k.clone(), Op::Put(b"batch-updated".to_vec())))
            .collect();
        merk.apply_batch(&batch).unwrap();

        let after_root = merk.snapshot().unwrap();
        let after = collect_node_addrs(&after_root);
        let changed = changed_node_keys(&before, &after);

        assert_eq!(changed, expected_paths);
        assert!(
            changed.len() < n / 5,
            "changed {} of {} nodes — expected O(k log n), not O(n)",
            changed.len(),
            n,
        );

        crate::test_utils::assert_tree_invariants(&merk.snapshot().unwrap());
    }

    #[test]
    fn cow_large_snapshot_preserves_all_pointers() {
        let n = 10_000;
        let merk = make_large_merk(n);

        let before = collect_node_addrs(&merk.snapshot().unwrap());
        let _snap = merk.snapshot().unwrap();
        let after = collect_node_addrs(&merk.snapshot().unwrap());

        assert_eq!(
            before, after,
            "snapshot should not change any node pointers"
        );
    }

    #[test]
    fn cow_large_read_ops_preserve_all_pointers() {
        let n = 10_000;
        let merk = make_large_merk(n);
        let _snap = merk.snapshot().unwrap();

        let before = collect_node_addrs(&merk.snapshot().unwrap());

        let _ = merk.get(&(5000u32).to_be_bytes());
        let _ = merk.root_hash();
        let _ = collect_iter(&merk.snapshot().unwrap());
        let _ = merk
            .snapshot()
            .unwrap()
            .iter_from(&(5000u32).to_be_bytes())
            .collect::<Vec<_>>();
        let _ = collect_range(
            &merk.snapshot().unwrap(),
            &(1000u32).to_be_bytes(),
            &(2000u32).to_be_bytes(),
        );

        let mut query = crate::proofs::Query::new();
        query.insert_key((5000u32).to_be_bytes().to_vec());
        let _ = merk.prove(query).unwrap();

        let after = collect_node_addrs(&merk.snapshot().unwrap());
        assert_eq!(
            before, after,
            "read ops should not change any node pointers"
        );
    }

    #[test]
    fn cow_large_extra_snapshots_same_changed_set() {
        let n = 10_000;
        let update_key = 0u32.to_be_bytes().to_vec();

        let changed_with_snapshots = |num_snaps: usize| -> std::collections::BTreeSet<Vec<u8>> {
            let merk = make_large_merk(n);
            let snaps: Vec<_> = (0..num_snaps).map(|_| merk.snapshot().unwrap()).collect();
            let before = collect_node_addrs(&merk.snapshot().unwrap());
            merk.put(update_key.clone(), b"x").unwrap();
            let after = collect_node_addrs(&merk.snapshot().unwrap());
            for snap in &snaps {
                assert_eq!(snap.get(&update_key), Some(vec![0u8; 64]));
            }
            changed_node_keys(&before, &after)
        };

        let baseline = changed_with_snapshots(0);
        let with_10 = changed_with_snapshots(10);
        assert_eq!(
            baseline, with_10,
            "extra snapshots should not change which nodes get copied",
        );
    }

    #[test]
    fn cow_large_scaling_changed_count_grows_logarithmically() {
        fn changed_for_update(n: usize) -> usize {
            let merk = make_large_merk(n);
            let before = collect_node_addrs(&merk.snapshot().unwrap());
            let key = 0u32.to_be_bytes().to_vec();
            merk.put(key, b"x").unwrap();
            let after = collect_node_addrs(&merk.snapshot().unwrap());
            changed_node_keys(&before, &after).len()
        }

        let c_1k = changed_for_update(1_000);
        let c_10k = changed_for_update(10_000);

        assert!(
            c_10k < c_1k * 3,
            "changed nodes should grow ~logarithmically: 1K->{}, 10K->{} (ratio {}x)",
            c_1k,
            c_10k,
            c_10k / c_1k.max(1),
        );
        assert!(c_1k > 0 && c_10k > 0);
    }

    #[test]
    fn cow_large_tree_stress_with_snapshots_and_invariants() {
        use crate::test_utils::assert_tree_invariants;
        use rand::prelude::*;
        use std::collections::BTreeMap;

        let n = 2_000;
        let merk = make_large_merk(n);

        let mut model: BTreeMap<Vec<u8>, Vec<u8>> = (0..n)
            .map(|i| {
                let key = (i as u32).to_be_bytes().to_vec();
                let value = vec![i as u8; 64];
                (key, value)
            })
            .collect();

        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        let mut snapshots = Vec::new();

        for round in 0u32..50 {
            let snap = merk.snapshot().unwrap();
            let expected_hash = snap.hash();
            let expected_entries: Vec<_> =
                model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            let mut ops = BTreeMap::new();
            for _ in 0..20 {
                let key_num = rng.gen_range(0u32..(n as u32 + 500));
                let key = key_num.to_be_bytes().to_vec();
                if rng.gen_range(0u8..5) == 0 {
                    ops.insert(key, Op::Delete);
                } else {
                    let val_byte = rng.gen::<u8>();
                    let value = vec![round as u8, val_byte];
                    ops.insert(key, Op::Put(value));
                }
            }
            let batch: Vec<_> = ops.into_iter().collect();

            merk.apply_batch(&batch).unwrap();

            // Verify old snapshot is still frozen
            assert_eq!(snap.hash(), expected_hash);
            for (key, value) in expected_entries.iter().step_by(100) {
                assert_eq!(snap.get(key), Some(value.clone()));
            }
            snapshots.push((snap, expected_hash, expected_entries));

            // Update model
            for (key, op) in &batch {
                match op {
                    Op::Put(v) => {
                        model.insert(key.clone(), v.clone());
                    }
                    Op::Delete => {
                        model.remove(key);
                    }
                }
            }

            // Verify live tree
            let live_entries = collect_iter(&merk.snapshot().unwrap());
            let model_entries: Vec<_> = model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            assert_eq!(live_entries, model_entries);

            if let Some(root) = merk.snapshot() {
                assert_tree_invariants(&root);
            }
        }

        // Final check: all accumulated snapshots still valid
        for (snap, expected_hash, _) in &snapshots {
            assert_eq!(snap.hash(), *expected_hash);
        }
    }

    // --- Edge case tests for CoW correctness ---

    #[test]
    fn cow_prove_against_old_snapshot_after_rotation_causing_writes() {
        let merk = InMemoryMerk::new();
        merk.apply_batch(&[
            (b"a".to_vec(), Op::Put(b"1".to_vec())),
            (b"b".to_vec(), Op::Put(b"2".to_vec())),
            (b"c".to_vec(), Op::Put(b"3".to_vec())),
            (b"d".to_vec(), Op::Put(b"4".to_vec())),
            (b"e".to_vec(), Op::Put(b"5".to_vec())),
        ])
        .unwrap();

        let snap = merk.snapshot().unwrap();
        let snap_hash = snap.hash();

        // Heavy mutations that will cause rotations
        merk.apply_batch(&[
            (b"a".to_vec(), Op::Delete),
            (b"f".to_vec(), Op::Put(b"6".to_vec())),
            (b"g".to_vec(), Op::Put(b"7".to_vec())),
            (b"h".to_vec(), Op::Put(b"8".to_vec())),
        ])
        .unwrap();

        assert_ne!(merk.root_hash(), snap_hash);

        // Prove against the old snapshot — must use the snapshot's tree, not
        // the live (rotated) tree.
        let keys = [b"b".as_slice(), b"d".as_slice()];
        let proof = snap.prove(query_for_keys(&keys)).unwrap();
        let result =
            crate::proofs::query::verify_query(&proof, &query_for_keys(&keys), snap_hash).unwrap();
        assert_eq!(
            result,
            vec![
                (b"b".to_vec(), b"2".to_vec()),
                (b"d".to_vec(), b"4".to_vec())
            ]
        );

        // Snapshot should still see deleted key 'a'
        assert_eq!(snap.get(b"a"), Some(b"1".to_vec()));
        // Snapshot should not see new keys
        assert!(snap.get(b"f").is_none());
    }

    #[test]
    fn cow_delete_all_then_reinsert() {
        let merk = InMemoryMerk::new();
        merk.apply_batch(&[
            (vec![1], Op::Put(vec![10])),
            (vec![2], Op::Put(vec![20])),
            (vec![3], Op::Put(vec![30])),
        ])
        .unwrap();

        let snap = merk.snapshot().unwrap();
        let snap_hash = snap.hash();

        // Delete everything
        merk.apply_batch(&[
            (vec![1], Op::Delete),
            (vec![2], Op::Delete),
            (vec![3], Op::Delete),
        ])
        .unwrap();
        assert_eq!(merk.root_hash(), NULL_HASH);
        assert!(merk.snapshot().is_none());

        // Reinsert
        merk.apply_batch(&[(vec![4], Op::Put(vec![40])), (vec![5], Op::Put(vec![50]))])
            .unwrap();
        assert_ne!(merk.root_hash(), NULL_HASH);
        assert_eq!(merk.get(&[4]), Some(vec![40]));
        assert_eq!(merk.get(&[5]), Some(vec![50]));
        assert!(merk.get(&[1]).is_none());

        // Old snapshot still has original keys
        assert_eq!(snap.hash(), snap_hash);
        assert_eq!(snap.get(&[1]), Some(vec![10]));
        assert!(snap.get(&[4]).is_none());

        crate::test_utils::assert_tree_invariants(&merk.snapshot().unwrap());
    }

    #[test]
    fn cow_snapshot_tree_structure_valid_after_live_rotation() {
        use crate::test_utils::assert_tree_invariants;

        let merk = InMemoryMerk::new();
        // Build a left-heavy tree
        merk.apply_batch(&[
            (vec![10], Op::Put(vec![10])),
            (vec![20], Op::Put(vec![20])),
            (vec![30], Op::Put(vec![30])),
        ])
        .unwrap();

        let snap = merk.snapshot().unwrap();
        let snap_root = snap.clone();

        // Force rotations by inserting many keys on one side
        merk.apply_batch(&[
            (vec![40], Op::Put(vec![40])),
            (vec![50], Op::Put(vec![50])),
            (vec![60], Op::Put(vec![60])),
            (vec![70], Op::Put(vec![70])),
            (vec![80], Op::Put(vec![80])),
        ])
        .unwrap();

        // Snapshot's tree must still be structurally valid
        assert_tree_invariants(&snap_root);

        // Snapshot's tree must have the original keys
        let snap_entries = collect_iter(&snap);
        assert_eq!(snap_entries.len(), 3);
        assert_eq!(snap_entries[0].0, vec![10]);
        assert_eq!(snap_entries[2].0, vec![30]);

        // Live tree must also be valid with all 8 keys
        assert_tree_invariants(&merk.snapshot().unwrap());
        let live_entries = collect_iter(&merk.snapshot().unwrap());
        assert_eq!(live_entries.len(), 8);
    }

    #[test]
    fn cow_multiple_independent_snapshots() {
        let merk = InMemoryMerk::new();
        merk.put(vec![1], vec![10]).unwrap();

        let snap1 = merk.snapshot().unwrap();
        let hash1 = snap1.hash();

        merk.put(vec![2], vec![20]).unwrap();

        let snap2 = merk.snapshot().unwrap();
        let hash2 = snap2.hash();

        merk.put(vec![3], vec![30]).unwrap();

        let snap3 = merk.snapshot().unwrap();
        let hash3 = snap3.hash();

        // Each snapshot is frozen at its point in time
        assert_ne!(hash1, hash2);
        assert_ne!(hash2, hash3);

        assert_eq!(collect_iter(&snap1).len(), 1);
        assert_eq!(collect_iter(&snap2).len(), 2);
        assert_eq!(collect_iter(&snap3).len(), 3);

        assert!(snap1.get(&[2]).is_none());
        assert_eq!(snap2.get(&[2]), Some(vec![20]));
        assert!(snap2.get(&[3]).is_none());
        assert_eq!(snap3.get(&[3]), Some(vec![30]));

        // More writes don't affect any snapshot
        merk.put(vec![4], vec![40]).unwrap();

        assert_eq!(snap1.hash(), hash1);
        assert_eq!(snap2.hash(), hash2);
        assert_eq!(snap3.hash(), hash3);

        // Drop order shouldn't matter
        drop(snap2);
        assert_eq!(snap1.hash(), hash1);
        assert_eq!(snap3.hash(), hash3);
    }

    #[test]
    fn cow_prove_old_snapshot_after_heavy_mutation() {
        use crate::proofs::Query;

        let n = 1_000;
        let merk = make_large_merk(n);

        let snap = merk.snapshot().unwrap();
        let snap_hash = snap.hash();

        // Heavy mutation: delete half, insert new half
        let mut batch: Vec<_> = (0..n / 2)
            .map(|i| ((i as u32).to_be_bytes().to_vec(), Op::Delete))
            .collect();
        for i in n..(n + n / 2) {
            batch.push(((i as u32).to_be_bytes().to_vec(), Op::Put(vec![i as u8])));
        }
        batch.sort_by(|a, b| a.0.cmp(&b.0));
        merk.apply_batch(&batch).unwrap();

        // Prove against old snapshot for a key that was deleted in the live tree
        let prove_key = (100u32).to_be_bytes().to_vec();
        let proof = snap.prove(vec![prove_key.clone()]).unwrap();
        let mut query = Query::new();
        query.insert_key(prove_key.clone());
        let result = crate::proofs::query::verify_query(&proof, &query, snap_hash).unwrap();
        assert_eq!(result, vec![(prove_key.clone(), vec![100u8; 64])]);

        // That key is gone from the live tree
        assert!(merk.get(&prove_key).is_none());

        if let Some(root) = merk.snapshot() {
            crate::test_utils::assert_tree_invariants(&root);
        }
    }

    #[test]
    fn cow_large_scale_delete_preserves_cow_properties() {
        use crate::test_utils::assert_tree_invariants;

        let n = 1_000;
        let merk = make_large_merk(n);

        let snap = merk.snapshot().unwrap();
        let snap_hash = snap.hash();

        let before = collect_node_addrs(&merk.snapshot().unwrap());

        // Delete 90% of keys
        let batch: Vec<_> = (0..((n * 9) / 10))
            .map(|i| ((i as u32).to_be_bytes().to_vec(), Op::Delete))
            .collect();
        merk.apply_batch(&batch).unwrap();

        let remaining = collect_iter(&merk.snapshot().unwrap());
        assert_eq!(remaining.len(), n / 10);

        if let Some(root) = merk.snapshot() {
            assert_tree_invariants(&root);
            let after = collect_node_addrs(&root);
            let shared = before
                .iter()
                .filter(|(k, addr)| after.get(*k) == Some(addr))
                .count();
            assert!(
                shared > 0,
                "expected some surviving nodes to be shared via CoW, but none are",
            );
        }

        // Snapshot still has all original keys
        assert_eq!(snap.hash(), snap_hash);
        assert_eq!(collect_iter(&snap).len(), n);
    }

    #[test]
    fn cow_snapshot_drop_order_does_not_corrupt() {
        let merk = InMemoryMerk::new();
        merk.apply_batch(&[
            (vec![1], Op::Put(vec![10])),
            (vec![2], Op::Put(vec![20])),
            (vec![3], Op::Put(vec![30])),
        ])
        .unwrap();

        let snap_a = merk.snapshot().unwrap();
        merk.put(vec![4], vec![40]).unwrap();

        let snap_b = merk.snapshot().unwrap();
        merk.put(vec![5], vec![50]).unwrap();

        let snap_c = merk.snapshot().unwrap();

        // One more write after snap_c
        merk.put(vec![6], vec![60]).unwrap();

        // Drop in reverse order
        let hash_a = snap_a.hash();
        let hash_b = snap_b.hash();
        let hash_c = snap_c.hash();

        drop(snap_c);
        // snap_a and snap_b should still work
        assert_eq!(snap_a.hash(), hash_a);
        assert_eq!(snap_b.hash(), hash_b);
        assert_eq!(snap_a.get(&[1]), Some(vec![10]));
        assert!(snap_a.get(&[4]).is_none());

        drop(snap_a);
        // snap_b should still work
        assert_eq!(snap_b.hash(), hash_b);
        assert_eq!(snap_b.get(&[4]), Some(vec![40]));
        assert!(snap_b.get(&[5]).is_none());

        // Live merk has diverged from all snapshots
        assert_eq!(merk.get(&[6]), Some(vec![60]));
        assert_ne!(merk.root_hash(), hash_c);

        drop(snap_b);
        // Live merk still works
        assert_eq!(merk.get(&[1]), Some(vec![10]));
        crate::test_utils::assert_tree_invariants(&merk.snapshot().unwrap());
    }
}

#[cfg(not(use_box))]
fn value_at(root: &Node, key: &[u8]) -> Vec<u8> {
    root.get(key).expect("expected key to exist")
}

#[cfg(not(use_box))]
fn apply_update_to_root(root: Node, key: &[u8], value: &[u8]) -> Node {
    let batch = [(key.to_vec(), Op::Put(value.to_vec()))];
    let walker = Walker::new(root, PanicSource {});
    let mut root = Walker::apply_to_mut(Some(walker), &mut batch.to_vec(), PanicSource {})
        .unwrap()
        .0
        .expect("expected updated root");
    root.commit();
    root
}

fn apply_batch_to_model(model: &mut std::collections::BTreeMap<Vec<u8>, Vec<u8>>, batch: &Batch) {
    for (key, op) in batch {
        match op {
            Op::Put(value) => {
                model.insert(key.clone(), value.clone());
            }
            Op::Delete => {
                model.remove(key);
            }
        }
    }
}

fn model_entries(model: &std::collections::BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<(Vec<u8>, Vec<u8>)> {
    model
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn assert_snapshot_still_matches(
    snap: &Node,
    expected_hash: Hash,
    expected_entries: &[(Vec<u8>, Vec<u8>)],
) {
    assert_eq!(snap.hash(), expected_hash);
    assert_eq!(collect_iter(snap), expected_entries.to_vec());

    for (key, value) in expected_entries.iter().step_by(7) {
        assert_eq!(snap.get(key), Some(value.clone()));
    }
    assert!(snap.get(&[250]).is_none());
}

#[cfg(not(use_box))]
fn path_keys(root: &Node, key: &[u8]) -> std::collections::BTreeSet<Vec<u8>> {
    let mut path = std::collections::BTreeSet::new();
    let mut cursor = root;

    loop {
        path.insert(cursor.key().to_vec());
        if cursor.key() == key {
            return path;
        }
        cursor = cursor
            .child(key < cursor.key())
            .expect("expected key to exist in resident tree");
    }
}

// --- iter ---

#[test]
fn test_iter_empty() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

#[test]
fn test_iter_single() {
    let merk = InMemoryMerk::new();
    merk.put(b"x", b"v").unwrap();
    let fwd = collect_iter(&merk.snapshot().unwrap());
    assert_eq!(keys(&fwd), vec![b"x".as_slice()]);
    let rev = collect_reverse_iter(&merk.snapshot().unwrap());
    assert_eq!(keys(&rev), vec![b"x".as_slice()]);
}

#[test]
fn test_iter_forward() {
    let merk = make_merk_abcde();
    let entries = collect_iter(&merk.snapshot().unwrap());
    assert_eq!(
        keys(&entries),
        vec![
            b"a".as_slice(),
            b"b".as_slice(),
            b"c".as_slice(),
            b"d".as_slice(),
            b"e".as_slice()
        ]
    );
}

#[test]
fn test_iter_reverse() {
    let merk = make_merk_abcde();
    let entries = collect_reverse_iter(&merk.snapshot().unwrap());
    assert_eq!(
        keys(&entries),
        vec![
            b"e".as_slice(),
            b"d".as_slice(),
            b"c".as_slice(),
            b"b".as_slice(),
            b"a".as_slice()
        ]
    );
}

// --- iter_from ---

#[test]
fn test_iter_from_empty() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

#[test]
fn test_iter_from_forward_existing() {
    let merk = make_merk_abcde();
    let entries = merk.snapshot().unwrap().iter_from(b"c").collect::<Vec<_>>();
    assert_eq!(
        keys(&entries),
        vec![b"c".as_slice(), b"d".as_slice(), b"e".as_slice()]
    );
}

#[test]
fn test_iter_from_forward_missing_lower_bound() {
    let merk = make_merk_abcde();
    let entries = merk
        .snapshot()
        .unwrap()
        .iter_from(b"bb")
        .collect::<Vec<_>>();
    assert_eq!(
        keys(&entries),
        vec![b"c".as_slice(), b"d".as_slice(), b"e".as_slice()]
    );
}

#[test]
fn test_iter_from_forward_past_end() {
    let merk = make_merk_abcde();
    let entries = merk.snapshot().unwrap().iter_from(b"z").collect::<Vec<_>>();
    assert!(entries.is_empty());
}

#[test]
fn test_iter_from_reverse_existing() {
    let merk = make_merk_abcde();
    let entries = merk
        .snapshot()
        .unwrap()
        .reverse_iter_from(b"c")
        .collect::<Vec<_>>();
    assert_eq!(
        keys(&entries),
        vec![b"c".as_slice(), b"b".as_slice(), b"a".as_slice()]
    );
}

#[test]
fn test_iter_from_reverse_missing_lower_bound() {
    let merk = make_merk_abcde();
    let entries = merk
        .snapshot()
        .unwrap()
        .reverse_iter_from(b"bb")
        .collect::<Vec<_>>();
    assert_eq!(keys(&entries), vec![b"b".as_slice(), b"a".as_slice()]);
}

#[test]
fn test_iter_from_reverse_before_start() {
    let merk = make_merk_abcde();
    let entries = merk
        .snapshot()
        .unwrap()
        .reverse_iter_from(b"\x00")
        .collect::<Vec<_>>();
    assert!(entries.is_empty());
}

// --- iter_prefix ---

#[test]
fn test_iter_prefix_empty_tree() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

#[test]
fn test_iter_prefix_matching() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"px_1".to_vec(), Op::Put(b"v1".to_vec())),
        (b"px_2".to_vec(), Op::Put(b"v2".to_vec())),
        (b"px_3".to_vec(), Op::Put(b"v3".to_vec())),
        (b"qx_1".to_vec(), Op::Put(b"v4".to_vec())),
    ])
    .unwrap();

    let fwd = collect_prefix(&merk.snapshot().unwrap(), b"px");
    assert_eq!(
        keys(&fwd),
        vec![b"px_1".as_slice(), b"px_2".as_slice(), b"px_3".as_slice()]
    );

    let rev = collect_prefix_reverse(&merk.snapshot().unwrap(), b"px");
    assert_eq!(
        keys(&rev),
        vec![b"px_3".as_slice(), b"px_2".as_slice(), b"px_1".as_slice()]
    );
}

#[test]
fn test_iter_prefix_missing() {
    let merk = make_merk_abcde();
    assert!(collect_prefix(&merk.snapshot().unwrap(), b"z").is_empty());
}

#[test]
fn test_iter_prefix_empty_prefix_matches_all() {
    let merk = make_merk_abcde();
    let entries = collect_prefix(&merk.snapshot().unwrap(), b"");
    assert_eq!(entries.len(), 5);
}

// --- iter_range ---

#[test]
fn test_iter_range_empty_tree() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

#[test]
fn test_iter_range_forward() {
    let merk = make_merk_abcde();
    let entries = collect_range(&merk.snapshot().unwrap(), b"b", b"d");
    assert_eq!(keys(&entries), vec![b"b".as_slice(), b"c".as_slice()]);
}

#[test]
fn test_iter_range_reverse() {
    let merk = make_merk_abcde();
    let entries = collect_range_reverse(&merk.snapshot().unwrap(), b"b", b"e");
    assert_eq!(
        keys(&entries),
        vec![b"d".as_slice(), b"c".as_slice(), b"b".as_slice()]
    );
}

#[test]
fn test_iter_range_half_open_boundary() {
    let merk = make_merk_abcde();
    let entries = collect_range(&merk.snapshot().unwrap(), b"a", b"f");
    assert_eq!(entries.len(), 5);

    let entries = collect_range(&merk.snapshot().unwrap(), b"c", b"c");
    assert!(entries.is_empty());
}

#[test]
fn test_iter_range_missing_bounds() {
    let merk = make_merk_abcde();
    let entries = collect_range(&merk.snapshot().unwrap(), b"aa", b"cc");
    assert_eq!(keys(&entries), vec![b"b".as_slice(), b"c".as_slice()]);
}

#[test]
fn test_iter_range_inverted_returns_empty() {
    let merk = make_merk_abcde();
    let snap = merk.snapshot().unwrap();
    assert!(collect_range(&snap, b"z", b"a").is_empty());
    assert!(collect_range_reverse(&snap, b"z", b"a").is_empty());
}

// --- snapshot scan helpers ---

#[test]
fn test_snapshot_iter() {
    let merk = make_merk_abcde();
    let snap = merk.snapshot().unwrap();
    let entries = collect_iter(&snap);
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[0].0, b"a");
    assert_eq!(entries[4].0, b"e");
}

#[test]
fn test_snapshot_iter_from() {
    let merk = make_merk_abcde();
    let snap = merk.snapshot().unwrap();
    let entries = snap.iter_from(b"c").collect::<Vec<_>>();
    assert_eq!(
        keys(&entries),
        vec![b"c".as_slice(), b"d".as_slice(), b"e".as_slice()]
    );
}

#[test]
fn test_snapshot_iter_prefix() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"px_1".to_vec(), Op::Put(b"v1".to_vec())),
        (b"px_2".to_vec(), Op::Put(b"v2".to_vec())),
        (b"qx_1".to_vec(), Op::Put(b"v3".to_vec())),
    ])
    .unwrap();
    let snap = merk.snapshot().unwrap();
    let entries = collect_prefix(&snap, b"px");
    assert_eq!(keys(&entries), vec![b"px_1".as_slice(), b"px_2".as_slice()]);
}

#[test]
fn test_snapshot_iter_range() {
    let merk = make_merk_abcde();
    let snap = merk.snapshot().unwrap();
    let entries = collect_range_reverse(&snap, b"b", b"d");
    assert_eq!(keys(&entries), vec![b"c".as_slice(), b"b".as_slice()]);
}

#[test]
fn test_snapshot_with_tree() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"a".to_vec(), Op::Put(b"val_a".to_vec())),
        (b"b".to_vec(), Op::Put(b"val_b".to_vec())),
    ])
    .unwrap();
    let snap = merk.snapshot().unwrap();
    let root_key = Some(snap.key().to_vec());
    assert!(root_key.is_some());
}

#[test]
fn test_snapshot_with_tree_empty() {
    let merk = InMemoryMerk::new();
    assert!(merk.snapshot().is_none());
}

// --- snapshot isolation for scan helpers ---

#[test]
fn test_snapshot_scan_isolation() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[
        (b"a".to_vec(), Op::Put(b"v1".to_vec())),
        (b"b".to_vec(), Op::Put(b"v2".to_vec())),
    ])
    .unwrap();

    let snap = merk.snapshot().unwrap();

    merk.put(b"c", b"v3").unwrap();

    let snap_entries = collect_iter(&snap);
    assert_eq!(snap_entries.len(), 2);

    let merk_entries = collect_iter(&merk.snapshot().unwrap());
    assert_eq!(merk_entries.len(), 3);
}

// --- Clone-mutate-publish tests ---

#[test]
fn test_apply_failure_leaves_root_unchanged() {
    let merk = InMemoryMerk::new();
    merk.put(vec![1], vec![10]).unwrap();
    let hash_before = merk.root_hash();

    // Unsorted batch should fail validation without modifying the tree
    let result = merk.apply_batch(&[(vec![5], Op::Put(vec![50])), (vec![2], Op::Put(vec![20]))]);
    assert!(result.is_err());

    assert_eq!(merk.root_hash(), hash_before);
    assert_eq!(merk.get(&[1]), Some(vec![10]));
    assert!(merk.get(&[2]).is_none());
}

#[test]
fn snapshot_stress_random_batches_keep_old_versions_stable() {
    use crate::test_utils::assert_tree_invariants;
    use rand::prelude::*;
    use std::collections::BTreeMap;

    let merk = InMemoryMerk::new();
    let initial: Vec<_> = (0u8..64)
        .map(|key| (vec![key], Op::Put(vec![key, key.wrapping_mul(3)])))
        .collect();
    merk.apply_batch(&initial).unwrap();

    let mut model = BTreeMap::new();
    apply_batch_to_model(&mut model, &initial);

    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut snapshots = Vec::new();

    for round in 0u8..32 {
        let snap = merk.snapshot().unwrap();
        let expected_hash = snap.hash();
        let expected_entries = model_entries(&model);

        let mut ops = BTreeMap::new();
        for op_index in 0u8..8 {
            let key = rng.gen_range(0u8..96);
            let op = if rng.gen_range(0u8..4) == 0 {
                Op::Delete
            } else {
                Op::Put(vec![round, op_index, key, rng.gen_range(0u8..=u8::MAX)])
            };
            ops.insert(vec![key], op);
        }
        let batch: Vec<_> = ops.into_iter().collect();

        merk.apply_batch(&batch).unwrap();

        assert_snapshot_still_matches(&snap, expected_hash, &expected_entries);
        snapshots.push((snap, expected_hash, expected_entries));

        apply_batch_to_model(&mut model, &batch);
        assert_eq!(
            collect_iter(&merk.snapshot().unwrap()),
            model_entries(&model)
        );
        if let Some(root) = merk.snapshot() {
            assert_tree_invariants(&root);
        }
    }

    for (snap, expected_hash, expected_entries) in snapshots {
        assert_snapshot_still_matches(&snap, expected_hash, &expected_entries);
    }
}

#[cfg(not(use_box))]
fn make_large_merk(n: usize) -> InMemoryMerk {
    let merk = InMemoryMerk::new();
    let batch: Vec<_> = (0..n)
        .map(|i| {
            let key = (i as u32).to_be_bytes().to_vec();
            let value = vec![i as u8; 64];
            (key, Op::Put(value))
        })
        .collect();
    merk.apply_batch(&batch).unwrap();
    merk
}

#[test]
fn apply_batch_rejects_unsorted_batch() {
    let merk = InMemoryMerk::new();
    let batch = vec![(vec![2], Op::Put(vec![20])), (vec![1], Op::Put(vec![10]))];
    let err = merk.apply_batch(&batch).unwrap_err();
    assert!(err.to_string().contains("sorted"));
}

#[test]
fn apply_batch_rejects_duplicate_keys() {
    let merk = InMemoryMerk::new();
    let batch = vec![(vec![1], Op::Put(vec![10])), (vec![1], Op::Put(vec![20]))];
    let err = merk.apply_batch(&batch).unwrap_err();
    assert!(err.to_string().contains("unique"));
}

#[test]
fn apply_batch_accepts_empty_batch() {
    let merk = InMemoryMerk::new();
    merk.apply_batch(&[]).unwrap();
    assert_eq!(merk.root_hash(), NULL_HASH);
}

#[test]
fn apply_batch_accepts_single_element() {
    let merk = InMemoryMerk::new();
    let batch = vec![(vec![1], Op::Put(vec![10]))];
    merk.apply_batch(&batch).unwrap();
    assert_eq!(merk.get(&[1]), Some(vec![10]));
}

#[test]
fn apply_batch_owned_applies_batch() {
    let merk = InMemoryMerk::new();

    merk.apply_batch_owned(vec![
        (vec![1], Op::Put(vec![10])),
        (vec![2], Op::Put(vec![20])),
    ])
    .unwrap();

    assert_eq!(merk.get(&[1]), Some(vec![10]));
    assert_eq!(merk.get(&[2]), Some(vec![20]));
}

#[test]
fn apply_batch_owned_rejects_unsorted_batch() {
    let merk = InMemoryMerk::new();
    let err = merk
        .apply_batch_owned(vec![
            (vec![2], Op::Put(vec![20])),
            (vec![1], Op::Put(vec![10])),
        ])
        .unwrap_err();
    assert!(err.to_string().contains("sorted"));
}

#[test]
fn put_and_get() {
    let merk = InMemoryMerk::new();
    merk.put(b"hello", b"world").unwrap();
    assert_eq!(merk.get(b"hello"), Some(b"world".to_vec()));
}

#[test]
fn put_overwrite() {
    let merk = InMemoryMerk::new();
    merk.put(b"key", b"v1").unwrap();
    merk.put(b"key", b"v2").unwrap();
    assert_eq!(merk.get(b"key"), Some(b"v2".to_vec()));
}

#[test]
fn delete_existing() {
    let merk = InMemoryMerk::new();
    merk.put(b"key", b"val").unwrap();
    assert!(merk.get(b"key").is_some());

    merk.delete(b"key").unwrap();
    assert!(merk.get(b"key").is_none());
    assert_eq!(merk.root_hash(), NULL_HASH);
}

#[test]
fn delete_nonexistent() {
    let merk = InMemoryMerk::new();
    merk.put(b"key", b"val").unwrap();
    let hash_before = merk.root_hash();

    merk.delete(b"other").unwrap();
    assert_eq!(merk.get(b"key"), Some(b"val".to_vec()));
    assert_eq!(merk.root_hash(), hash_before);
}

#[test]
fn put_multiple_then_delete() {
    let merk = InMemoryMerk::new();
    merk.put(vec![1], vec![10]).unwrap();
    merk.put(vec![2], vec![20]).unwrap();
    merk.put(vec![3], vec![30]).unwrap();

    assert_eq!(merk.get(&[2]), Some(vec![20]));

    merk.delete(vec![2]).unwrap();
    assert_eq!(merk.get(&[2]), None);
    assert_eq!(merk.get(&[1]), Some(vec![10]));
    assert_eq!(merk.get(&[3]), Some(vec![30]));
}

#[test]
fn put_accepts_various_types() {
    let merk = InMemoryMerk::new();
    // [u8; N]
    merk.put([1, 2, 3], [4, 5, 6]).unwrap();
    // Vec<u8>
    merk.put(vec![7, 8], vec![9, 10]).unwrap();
    // b"literal"
    merk.put(b"hello", b"world").unwrap();

    assert_eq!(merk.get(&[1, 2, 3]), Some(vec![4, 5, 6]));
    assert_eq!(merk.get(&[7, 8]), Some(vec![9, 10]));
    assert_eq!(merk.get(b"hello"), Some(b"world".to_vec()));
}
