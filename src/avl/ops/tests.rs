use super::*;
use crate::avl::node::*;
use crate::test_utils::{make_batch_seq, put_entry_value, seq_key};
use std::convert::TryFrom;

fn assert_valid_avl(tree: &Node) {
    let bf = tree.balance_factor();
    assert!(
        bf.abs() <= 1,
        "AVL violation at key {:?}: bf={}",
        tree.key(),
        bf
    );
    let expected_h = 1 + std::cmp::max(tree.child_height(true), tree.child_height(false));
    assert_eq!(
        tree.height(),
        expected_h,
        "Height mismatch at key {:?}",
        tree.key()
    );
    if let Some(left) = tree.child(true) {
        assert!(
            left.key() < tree.key(),
            "BST violation: left {:?} >= root {:?}",
            left.key(),
            tree.key()
        );
        assert_valid_avl(left);
    }
    if let Some(right) = tree.child(false) {
        assert!(
            right.key() > tree.key(),
            "BST violation: right {:?} <= root {:?}",
            right.key(),
            tree.key()
        );
        assert_valid_avl(right);
    }
}

fn collect_keys(tree: &Node) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    collect_keys_inner(tree, &mut keys);
    keys
}

fn collect_keys_inner(tree: &Node, keys: &mut Vec<Vec<u8>>) {
    if let Some(left) = tree.child(true) {
        collect_keys_inner(left, keys);
    }
    keys.push(tree.key().to_vec());
    if let Some(right) = tree.child(false) {
        collect_keys_inner(right, keys);
    }
}

// apply_sorted_batch_ops validation tests

#[test]
fn validate_batch_sorted_points() {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch: Vec<BatchEntry> = vec![
        (vec![1], Op::Put(vec![2])),
        (vec![3], Op::Delete),
        (vec![5], Op::Put(vec![6])),
    ];
    assert!(merk.apply_sorted_batch_ops(&batch).is_ok());
}

#[test]
fn validate_batch_unsorted_rejected() {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch: Vec<BatchEntry> = vec![(vec![5], Op::Put(vec![6])), (vec![1], Op::Put(vec![2]))];
    assert!(merk.apply_sorted_batch_ops(&batch).is_err());
}

#[test]
fn validate_batch_range_invalid_bounds() {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch: Vec<BatchEntry> = vec![(vec![5], Op::DeleteRange(vec![3]))];
    assert!(merk.apply_sorted_batch_ops(&batch).is_err());
}

#[test]
fn validate_batch_adjacent_range_and_point_ok() {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    // Insert some keys first
    let setup: Vec<BatchEntry> = vec![
        (vec![1], Op::Put(vec![10])),
        (vec![3], Op::Put(vec![30])),
        (vec![5], Op::Put(vec![50])),
    ];
    merk.apply_sorted_batch_ops(&setup).unwrap();
    // DeleteRange [1, 5) followed by Put at key 5 is valid (different segments)
    let batch: Vec<BatchEntry> = vec![
        (vec![1], Op::DeleteRange(vec![5])),
        (vec![5], Op::Put(vec![6])),
    ];
    assert!(merk.apply_sorted_batch_ops(&batch).is_ok());
}

// apply_sorted_batch_ops with mixed ops tests

#[test]
fn apply_batch_mixed_empty() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch = make_batch_seq(0..10);
    merk.apply_sorted_batch_ops(&batch).unwrap();
    let tree = merk.checkpoint().into_root().unwrap();
    let original_keys = collect_keys(&tree);

    merk.apply_sorted_batch_ops(&[]).unwrap();
    let tree2 = merk.checkpoint().into_root().unwrap();
    assert_eq!(collect_keys(&tree2), original_keys);
    Ok(())
}

#[test]
fn apply_batch_mixed_puts_only() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch = make_batch_seq(0..5);
    merk.apply_sorted_batch_ops(&batch).unwrap();

    let batch2: Vec<BatchEntry> = vec![
        (seq_key(10), Op::Put(put_entry_value())),
        (seq_key(11), Op::Put(put_entry_value())),
    ];
    merk.apply_sorted_batch_ops(&batch2).unwrap();
    let tree = merk.checkpoint().into_root().unwrap();
    let keys = collect_keys(&tree);
    assert_eq!(keys.len(), 7);
    assert!(keys.contains(&seq_key(10)));
    assert!(keys.contains(&seq_key(11)));
    Ok(())
}

#[test]
fn apply_batch_mixed_delete_range_only() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch = make_batch_seq(0..20);
    merk.apply_sorted_batch_ops(&batch).unwrap();

    let batch2: Vec<BatchEntry> = vec![(seq_key(5), Op::DeleteRange(seq_key(15)))];
    merk.apply_sorted_batch_ops(&batch2).unwrap();
    let tree = merk.checkpoint().into_root().unwrap();
    let keys = collect_keys(&tree);
    assert_eq!(keys.len(), 10);
    for k in &keys {
        let n = u64::from_be_bytes(<[u8; 8]>::try_from(k.as_slice()).unwrap());
        assert!(!(5..15).contains(&n));
    }
    Ok(())
}

#[test]
fn apply_batch_mixed_puts_and_range() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch = make_batch_seq(0..20);
    merk.apply_sorted_batch_ops(&batch).unwrap();

    let batch2: Vec<BatchEntry> = vec![
        (seq_key(2), Op::Put(vec![99])),
        (seq_key(3), Op::Delete),
        (seq_key(10), Op::DeleteRange(seq_key(15))),
        (seq_key(50), Op::Put(put_entry_value())),
    ];
    merk.apply_sorted_batch_ops(&batch2).unwrap();
    let tree = merk.checkpoint().into_root().unwrap();
    assert_valid_avl(&tree);
    let keys = collect_keys(&tree);
    assert!(keys.contains(&seq_key(2)));
    assert!(!keys.contains(&seq_key(3)));
    assert!(keys.contains(&seq_key(50)));
    for n in 10..15 {
        assert!(!keys.contains(&seq_key(n)));
    }
    Ok(())
}

#[test]
fn apply_batch_mixed_on_empty_tree() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch: Vec<BatchEntry> = vec![
        (seq_key(1), Op::Put(put_entry_value())),
        (seq_key(2), Op::Put(put_entry_value())),
    ];
    merk.apply_sorted_batch_ops(&batch).unwrap();
    let tree = merk.checkpoint().into_root().unwrap();
    let keys = collect_keys(&tree);
    assert_eq!(keys, vec![seq_key(1), seq_key(2)]);
    Ok(())
}

#[test]
fn apply_batch_mixed_multiple_ranges() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    let merk = InMemoryMerk::new();
    let batch = make_batch_seq(0..50);
    merk.apply_sorted_batch_ops(&batch).unwrap();

    let batch2: Vec<BatchEntry> = vec![
        (seq_key(5), Op::DeleteRange(seq_key(10))),
        (seq_key(25), Op::Put(vec![42])),
        (seq_key(30), Op::DeleteRange(seq_key(40))),
    ];
    merk.apply_sorted_batch_ops(&batch2).unwrap();
    let tree = merk.checkpoint().into_root().unwrap();
    assert_valid_avl(&tree);
    let keys = collect_keys(&tree);
    for n in 5..10 {
        assert!(!keys.contains(&seq_key(n)));
    }
    for n in 30..40 {
        assert!(!keys.contains(&seq_key(n)));
    }
    assert!(keys.contains(&seq_key(25)));
    assert_eq!(keys.len(), 50 - 5 - 10); // 50 original - 5 from first range - 10 from second
    Ok(())
}
