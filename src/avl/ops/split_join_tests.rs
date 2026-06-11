use super::*;
use crate::avl::node::*;
use crate::error::Error;
use crate::hash::Hash;
use crate::test_utils::{apply_to_memonly, make_batch_seq, put_entry_value, seq_key};
use rand::prelude::*;
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

fn assert_valid_avl_opt(tree: &Option<Node>) {
    if let Some(t) = tree {
        assert_valid_avl(t);
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

fn collect_keys_opt(tree: &Option<Node>) -> Vec<Vec<u8>> {
    tree.as_ref().map_or_else(Vec::new, collect_keys)
}

fn build_tree(start: u64, count: u64) -> Node {
    let batch = make_batch_seq(start..start + count);
    apply_to_memonly(None, &batch).unwrap()
}

fn build_tree_from_keys(keys: &[u64]) -> Node {
    let batch: Vec<BatchEntry> = keys
        .iter()
        .map(|key| (seq_key(*key), Op::Put(put_entry_value())))
        .collect();
    apply_to_memonly(None, &batch).unwrap()
}

fn commit_opt(mut tree: Option<Node>) -> Option<Node> {
    if let Some(ref mut t) = tree {
        t.commit();
        assert_valid_avl(t);
    }
    tree
}

fn delete_range_memonly(tree: Node, start: &[u8], end: &[u8]) -> Result<Option<Node>> {
    let result = Walker::new(tree, PanicSource {}).delete_range(start, end)?;
    Ok(commit_opt(result))
}

fn delete_range_oracle_memonly(tree: Node, start: &[u8], end: &[u8]) -> Result<Option<Node>> {
    let mut delete_batch: Vec<BatchEntry> = collect_keys(&tree)
        .into_iter()
        .filter(|key| key.as_slice() >= start && key.as_slice() < end)
        .map(|key| (key, Op::Delete))
        .collect();
    let result = Walker::apply_to_mut(
        Some(Walker::new(tree, PanicSource {})),
        &mut delete_batch,
        PanicSource {},
    )?
    .0;
    Ok(commit_opt(result))
}

fn hash_opt(tree: &Option<Node>) -> Option<Hash> {
    tree.as_ref().map(|tree| tree.hash())
}

fn assert_delete_range_entries_match_oracle(tree: Node, start: &[u8], end: &[u8]) -> Result<()> {
    let result = delete_range_memonly(tree.clone(), start, end)?;
    let oracle = delete_range_oracle_memonly(tree, start, end)?;

    assert_eq!(
        collect_keys_opt(&result),
        collect_keys_opt(&oracle),
        "in-order entries differ for range [{start:?}, {end:?})"
    );
    Ok(())
}

fn assert_delete_range_hash_matches_oracle(tree: Node, start: &[u8], end: &[u8]) -> Result<()> {
    let result = delete_range_memonly(tree.clone(), start, end)?;
    let oracle = delete_range_oracle_memonly(tree, start, end)?;

    assert_eq!(
        collect_keys_opt(&result),
        collect_keys_opt(&oracle),
        "in-order entries differ for range [{start:?}, {end:?})"
    );
    assert_eq!(
        hash_opt(&result),
        hash_opt(&oracle),
        "root hash differs for range [{start:?}, {end:?})"
    );
    Ok(())
}

// Join tests

#[test]
fn join_both_empty() -> Result<()> {
    let pivot = Node::new(seq_key(5), put_entry_value())?;
    let result = Walker::<PanicSource>::join_trees(None, pivot, None, PanicSource {})?;
    assert_valid_avl(&result);
    assert_eq!(collect_keys(&result), vec![seq_key(5)]);
    Ok(())
}

#[test]
fn join_left_empty() -> Result<()> {
    let right = build_tree(10, 5);
    let pivot = Node::new(seq_key(5), put_entry_value())?;
    let result = Walker::<PanicSource>::join_trees(None, pivot, Some(right), PanicSource {})?;
    assert_valid_avl(&result);
    let keys = collect_keys(&result);
    assert!(keys.contains(&seq_key(5)));
    assert_eq!(keys.len(), 6);
    Ok(())
}

#[test]
fn join_right_empty() -> Result<()> {
    let left = build_tree(0, 5);
    let pivot = Node::new(seq_key(100), put_entry_value())?;
    let result = Walker::<PanicSource>::join_trees(Some(left), pivot, None, PanicSource {})?;
    assert_valid_avl(&result);
    let keys = collect_keys(&result);
    assert!(keys.contains(&seq_key(100)));
    assert_eq!(keys.len(), 6);
    Ok(())
}

#[test]
fn join_equal_height() -> Result<()> {
    let left = build_tree(0, 7);
    let right = build_tree(100, 7);
    let pivot = Node::new(seq_key(50), put_entry_value())?;
    let result = Walker::<PanicSource>::join_trees(Some(left), pivot, Some(right), PanicSource {})?;
    assert_valid_avl(&result);
    assert_eq!(collect_keys(&result).len(), 15);
    Ok(())
}

#[test]
fn join_single_nodes() -> Result<()> {
    let left = Node::new(seq_key(1), put_entry_value())?;
    let pivot = Node::new(seq_key(5), put_entry_value())?;
    let right = Node::new(seq_key(10), put_entry_value())?;
    let result = Walker::<PanicSource>::join_trees(Some(left), pivot, Some(right), PanicSource {})?;
    assert_valid_avl(&result);
    assert_eq!(
        collect_keys(&result),
        vec![seq_key(1), seq_key(5), seq_key(10)]
    );
    Ok(())
}

#[test]
fn join_left_much_taller() -> Result<()> {
    let left = build_tree(0, 63);
    let pivot = Node::new(seq_key(100), put_entry_value())?;
    let right = Node::new(seq_key(200), put_entry_value())?;
    let result = Walker::<PanicSource>::join_trees(Some(left), pivot, Some(right), PanicSource {})?;
    assert_valid_avl(&result);
    assert_eq!(collect_keys(&result).len(), 65);
    Ok(())
}

#[test]
fn join_right_much_taller() -> Result<()> {
    let left = Node::new(seq_key(0), put_entry_value())?;
    let pivot = Node::new(seq_key(50), put_entry_value())?;
    let right = build_tree(100, 63);
    let result = Walker::<PanicSource>::join_trees(Some(left), pivot, Some(right), PanicSource {})?;
    assert_valid_avl(&result);
    assert_eq!(collect_keys(&result).len(), 65);
    Ok(())
}

#[test]
fn join_all_size_combos() -> Result<()> {
    for left_n in [0u64, 1, 3, 7, 15, 31] {
        for right_n in [0u64, 1, 3, 7, 15, 31] {
            let left = if left_n > 0 {
                Some(build_tree(0, left_n))
            } else {
                None
            };
            let right = if right_n > 0 {
                Some(build_tree(1000, right_n))
            } else {
                None
            };
            let pivot = Node::new(seq_key(500), put_entry_value())?;
            let result = Walker::<PanicSource>::join_trees(left, pivot, right, PanicSource {})?;
            assert_valid_avl(&result);
            assert_eq!(
                collect_keys(&result).len(),
                (left_n + 1 + right_n) as usize,
                "left_n={left_n} right_n={right_n}: wrong count"
            );
        }
    }
    Ok(())
}

// Join2 tests

#[test]
fn join2_both_empty() -> Result<()> {
    let result = Walker::<PanicSource>::join2_trees(None, None, PanicSource {})?;
    assert!(result.is_none());
    Ok(())
}

#[test]
fn join2_left_empty() -> Result<()> {
    let right = build_tree(0, 7);
    let expected_keys = collect_keys(&right);
    let result = Walker::<PanicSource>::join2_trees(None, Some(right), PanicSource {})?;
    assert_valid_avl_opt(&result);
    assert_eq!(collect_keys_opt(&result), expected_keys);
    Ok(())
}

#[test]
fn join2_right_empty() -> Result<()> {
    let left = build_tree(0, 7);
    let expected_keys = collect_keys(&left);
    let result = Walker::<PanicSource>::join2_trees(Some(left), None, PanicSource {})?;
    assert_valid_avl_opt(&result);
    assert_eq!(collect_keys_opt(&result), expected_keys);
    Ok(())
}

#[test]
fn join2_both_nonempty() -> Result<()> {
    let left = build_tree(0, 7);
    let right = build_tree(100, 7);
    let result = Walker::<PanicSource>::join2_trees(Some(left), Some(right), PanicSource {})?;
    assert_valid_avl_opt(&result);
    assert_eq!(collect_keys_opt(&result).len(), 14);
    Ok(())
}

#[test]
fn join2_large_height_diff() -> Result<()> {
    let left = build_tree(0, 63);
    let right = Node::new(seq_key(500), put_entry_value())?;
    let result = Walker::<PanicSource>::join2_trees(Some(left), Some(right), PanicSource {})?;
    assert_valid_avl_opt(&result);
    assert_eq!(collect_keys_opt(&result).len(), 64);
    Ok(())
}

// Split tests

#[test]
fn split_before_first_key() -> Result<()> {
    let tree = build_tree(5, 10);
    let walker = Walker::new(tree, PanicSource {});
    let split_key = seq_key(0);
    let (left, right) = walker.split_at(&split_key)?;
    assert!(left.is_none());
    assert_valid_avl_opt(&right);
    assert_eq!(collect_keys_opt(&right).len(), 10);
    Ok(())
}

#[test]
fn split_after_last_key() -> Result<()> {
    let tree = build_tree(0, 10);
    let walker = Walker::new(tree, PanicSource {});
    let big_key = seq_key(100);
    let (left, right) = walker.split_at(&big_key)?;
    assert_valid_avl_opt(&left);
    assert_eq!(collect_keys_opt(&left).len(), 10);
    assert!(right.is_none());
    Ok(())
}

#[test]
fn split_at_existing_key() -> Result<()> {
    let tree = build_tree(0, 10);
    let walker = Walker::new(tree, PanicSource {});
    let split_key = seq_key(5);
    let (left, right) = walker.split_at(&split_key)?;
    assert_valid_avl_opt(&left);
    assert_valid_avl_opt(&right);

    let left_keys = collect_keys_opt(&left);
    let right_keys = collect_keys_opt(&right);

    for k in &left_keys {
        assert!(k.as_slice() < split_key.as_slice());
    }
    for k in &right_keys {
        assert!(k.as_slice() >= split_key.as_slice());
    }
    assert!(right_keys.contains(&split_key));
    assert_eq!(left_keys.len() + right_keys.len(), 10);
    Ok(())
}

#[test]
fn split_preserves_all_keys() -> Result<()> {
    for n in [3u64, 7, 15, 31, 50] {
        let tree = build_tree(0, n);
        let all_keys = collect_keys(&tree);
        let split_key = seq_key(n / 2);
        let walker = Walker::new(tree, PanicSource {});
        let (left, right) = walker.split_at(&split_key)?;
        assert_valid_avl_opt(&left);
        assert_valid_avl_opt(&right);
        let mut result_keys = collect_keys_opt(&left);
        result_keys.extend(collect_keys_opt(&right));
        assert_eq!(result_keys, all_keys, "n={n}: keys not preserved");
    }
    Ok(())
}

#[test]
fn split_then_join2_roundtrip() -> Result<()> {
    for n in [3u64, 7, 15, 31, 50] {
        let tree = build_tree(0, n);
        let original_keys = collect_keys(&tree);
        let split_key = seq_key(n / 2);
        let walker = Walker::new(tree, PanicSource {});
        let (left, right) = walker.split_at(&split_key)?;
        let rejoined = Walker::<PanicSource>::join2_trees(left, right, PanicSource {})?;
        assert_valid_avl_opt(&rejoined);
        assert_eq!(
            collect_keys_opt(&rejoined),
            original_keys,
            "n={n}: roundtrip failed"
        );
    }
    Ok(())
}

// Delete range tests

#[test]
fn delete_range_rejects_empty_range() -> Result<()> {
    let tree = build_tree(0, 10);
    let start = seq_key(5);
    let result = Walker::new(tree, PanicSource {}).delete_range(&start, &start);
    assert!(matches!(result, Err(Error::Bound(_))));
    Ok(())
}

#[test]
fn delete_range_rejects_reversed_range() -> Result<()> {
    let tree = build_tree(0, 10);
    let start = seq_key(8);
    let end = seq_key(3);
    let result = Walker::new(tree, PanicSource {}).delete_range(&start, &end);
    assert!(matches!(result, Err(Error::Bound(_))));
    Ok(())
}

#[test]
fn delete_range_before_first_key_matches_single_deletes() -> Result<()> {
    let tree = build_tree(10, 10);
    assert_delete_range_hash_matches_oracle(tree, &seq_key(0), &seq_key(5))
}

#[test]
fn delete_range_after_last_key_matches_single_deletes() -> Result<()> {
    let tree = build_tree(0, 10);
    assert_delete_range_hash_matches_oracle(tree, &seq_key(20), &seq_key(30))
}

#[test]
fn delete_range_entire_tree_matches_single_deletes() -> Result<()> {
    let tree = build_tree(0, 31);
    let result = delete_range_memonly(tree.clone(), &seq_key(0), &seq_key(31))?;
    let oracle = delete_range_oracle_memonly(tree, &seq_key(0), &seq_key(31))?;
    assert!(result.is_none());
    assert!(oracle.is_none());
    assert_eq!(hash_opt(&result), hash_opt(&oracle));
    Ok(())
}

#[test]
fn delete_range_many_contiguous_keys_matches_single_deletes() -> Result<()> {
    let tree = build_tree(0, 100);
    assert_delete_range_entries_match_oracle(tree, &seq_key(20), &seq_key(80))
}

#[test]
fn delete_range_randomized_matches_single_deletes() -> Result<()> {
    for seed in 0..32u64 {
        let keys: Vec<u64> = (0..80).map(|i| i * 4 + ((i * seed + 7) % 3)).collect();
        let tree = build_tree_from_keys(&keys);
        let start_n = (seed * 17 + 5) % 250;
        let end_n = start_n + 1 + ((seed * 29 + 11) % 90);
        assert_delete_range_entries_match_oracle(tree, &seq_key(start_n), &seq_key(end_n))?;
    }
    Ok(())
}

#[test]
fn delete_range_result_height_is_logarithmic() -> Result<()> {
    let tree = build_tree(0, 1000);
    let result = delete_range_memonly(tree, &seq_key(200), &seq_key(800))?;
    let result = result.expect("should have remaining keys");
    let remaining = collect_keys(&result).len();
    assert_eq!(remaining, 400);
    let max_height = (remaining as f64).log2().ceil() as u8 + 2;
    assert!(
        result.height() <= max_height,
        "Height {} exceeds expected O(log {}) = {}",
        result.height(),
        remaining,
        max_height
    );
    Ok(())
}

// Edge cases

#[test]
fn delete_range_on_empty_tree() -> Result<()> {
    let result = Walker::<PanicSource>::delete_range_apply_to(None, &seq_key(0), &seq_key(10))?;
    assert!(result.is_none());
    Ok(())
}

#[test]
fn delete_range_single_node_covered() -> Result<()> {
    let tree = Node::new(seq_key(5), put_entry_value())?;
    let result = Walker::new(tree, PanicSource {}).delete_range(&seq_key(0), &seq_key(10))?;
    assert!(result.is_none());
    Ok(())
}

#[test]
fn delete_range_single_node_not_covered() -> Result<()> {
    let tree = Node::new(seq_key(5), put_entry_value())?;
    let result = Walker::new(tree, PanicSource {}).delete_range(&seq_key(6), &seq_key(10))?;
    let result = result.expect("node should remain");
    assert_eq!(result.key(), seq_key(5).as_slice());
    Ok(())
}

#[test]
fn delete_range_covers_exactly_one_key() -> Result<()> {
    let tree = build_tree(0, 10);
    let result = delete_range_memonly(tree.clone(), &seq_key(5), &seq_key(6))?;
    let result = result.expect("should have remaining keys");
    let keys = collect_keys(&result);
    assert_eq!(keys.len(), 9);
    assert!(!keys.contains(&seq_key(5)));
    Ok(())
}

#[test]
fn delete_range_between_existing_keys() -> Result<()> {
    // Keys are 0,2,4,6,8 (even only). Delete range [3, 7) should remove 4 and 6.
    let keys: Vec<u64> = (0..5).map(|i| i * 2).collect();
    let tree = build_tree_from_keys(&keys);
    let result = delete_range_memonly(tree, &seq_key(3), &seq_key(7))?;
    let result = result.expect("should have remaining keys");
    let remaining = collect_keys(&result);
    assert_eq!(remaining, vec![seq_key(0), seq_key(2), seq_key(8)]);
    Ok(())
}

#[test]
fn delete_range_adjacent_ranges() -> Result<()> {
    let tree = build_tree(0, 30);
    let result = Walker::new(tree, PanicSource {}).delete_range(&seq_key(5), &seq_key(15))?;
    let result = commit_opt(result).expect("should have remaining");
    let result = Walker::new(result, PanicSource {}).delete_range(&seq_key(15), &seq_key(25))?;
    let result = commit_opt(result).expect("should have remaining");
    assert_valid_avl(&result);
    let keys = collect_keys(&result);
    assert_eq!(keys.len(), 10);
    for k in &keys {
        let n = u64::from_be_bytes(<[u8; 8]>::try_from(k.as_slice()).unwrap());
        assert!(!(5..25).contains(&n));
    }
    Ok(())
}

// Contract / semantic property tests

#[test]
fn delete_range_boundary_inclusive_exclusive() -> Result<()> {
    // [start, end) - start IS deleted, end is NOT
    let tree = build_tree(0, 20);
    let result = delete_range_memonly(tree, &seq_key(5), &seq_key(15))?;
    let result = result.expect("should have remaining keys");
    let keys = collect_keys(&result);

    assert!(!keys.contains(&seq_key(5)), "start key should be deleted");
    assert!(!keys.contains(&seq_key(10)), "middle key should be deleted");
    assert!(
        !keys.contains(&seq_key(14)),
        "key before end should be deleted"
    );
    assert!(keys.contains(&seq_key(15)), "end key should NOT be deleted");
    assert!(keys.contains(&seq_key(4)), "key before start should remain");
    Ok(())
}

#[test]
fn delete_range_sequential_differs_from_combined() -> Result<()> {
    // delete_range(a,b) then delete_range(b,c) may produce a different
    // tree structure (and hash) than delete_range(a,c)
    let tree = build_tree(0, 50);

    // Combined: delete [10, 40) in one shot
    let combined = delete_range_memonly(tree.clone(), &seq_key(10), &seq_key(40))?;

    // Sequential: delete [10, 25) then [25, 40)
    let step1 = Walker::new(tree, PanicSource {}).delete_range(&seq_key(10), &seq_key(25))?;
    let step1 = commit_opt(step1).expect("should have remaining");
    let sequential = delete_range_memonly(step1, &seq_key(25), &seq_key(40))?;

    // Same key set
    assert_eq!(
        collect_keys_opt(&combined),
        collect_keys_opt(&sequential),
        "key sets should match"
    );
    // But structure (and hash) may differ; this documents the property.
    // (If they happen to match for this input, the test still passes;
    // the important thing is correctness, not that they differ.)
    Ok(())
}

#[test]
fn delete_range_idempotent() -> Result<()> {
    let tree = build_tree(0, 30);
    let first = delete_range_memonly(tree, &seq_key(10), &seq_key(20))?;
    let first = first.expect("should have remaining");
    let hash_after_first = first.hash();
    let keys_after_first = collect_keys(&first);

    // Second delete_range with same bounds should be a no-op
    let second = delete_range_memonly(first, &seq_key(10), &seq_key(20))?;
    let second = second.expect("should have remaining");
    assert_eq!(collect_keys(&second), keys_after_first);
    assert_eq!(second.hash(), hash_after_first);
    Ok(())
}

#[test]
fn delete_range_variable_length_keys() -> Result<()> {
    // Build a tree with variable-length keys
    let batch: Vec<BatchEntry> = vec![
        (vec![1], Op::Put(put_entry_value())),
        (vec![1, 0], Op::Put(put_entry_value())),
        (vec![1, 1], Op::Put(put_entry_value())),
        (vec![2], Op::Put(put_entry_value())),
        (vec![2, 0], Op::Put(put_entry_value())),
        (vec![3], Op::Put(put_entry_value())),
        (vec![3, 0, 0], Op::Put(put_entry_value())),
    ];
    let tree = apply_to_memonly(None, &batch).unwrap();
    let all_keys = collect_keys(&tree);
    assert_eq!(all_keys.len(), 7);

    // Delete range [vec![1, 1], vec![3]) should remove [1,1], [2], [2,0].
    let result = Walker::new(tree, PanicSource {}).delete_range(&[1, 1], &[3])?;
    let result = commit_opt(result).expect("should have remaining");
    assert_valid_avl(&result);
    let keys = collect_keys(&result);
    assert!(keys.contains(&vec![1u8]));
    assert!(keys.contains(&vec![1u8, 0]));
    assert!(!keys.contains(&vec![1u8, 1]));
    assert!(!keys.contains(&vec![2u8]));
    assert!(!keys.contains(&vec![2u8, 0]));
    assert!(keys.contains(&vec![3u8]));
    assert!(keys.contains(&vec![3u8, 0, 0]));
    Ok(())
}

#[test]
fn delete_range_non_overlapping_commutes() -> Result<()> {
    let tree = build_tree(0, 50);

    // Order A: delete [5,10) then [30,40)
    let a1 = Walker::new(tree.clone(), PanicSource {}).delete_range(&seq_key(5), &seq_key(10))?;
    let a1 = commit_opt(a1).expect("remaining");
    let order_a = delete_range_memonly(a1, &seq_key(30), &seq_key(40))?;

    // Order B: delete [30,40) then [5,10)
    let b1 = Walker::new(tree, PanicSource {}).delete_range(&seq_key(30), &seq_key(40))?;
    let b1 = commit_opt(b1).expect("remaining");
    let order_b = delete_range_memonly(b1, &seq_key(5), &seq_key(10))?;

    // Same key set regardless of order
    assert_eq!(
        collect_keys_opt(&order_a),
        collect_keys_opt(&order_b),
        "non-overlapping delete_ranges should commute (key set)"
    );
    Ok(())
}

// Randomized stress test

#[test]
#[ignore]
fn fuzz_delete_range() -> Result<()> {
    use crate::avl::in_memory::InMemoryMerk;
    use crate::proofs::Query;

    fn parse_env(name: &str) -> Option<u64> {
        std::env::var(name).ok().and_then(|v| v.parse().ok())
    }
    let base_seed: u64 = parse_env("FUZZ_SEED")
        .expect("Required: FUZZ_SEED=<n> FUZZ_ITERS=<n> (e.g. FUZZ_SEED=99 FUZZ_ITERS=10000)");
    let num_iters: u64 = parse_env("FUZZ_ITERS")
        .expect("Required: FUZZ_SEED=<n> FUZZ_ITERS=<n> (e.g. FUZZ_SEED=99 FUZZ_ITERS=10000)");
    let start_epoch: u64 = parse_env("FUZZ_EPOCH").unwrap_or(0);

    let mut total_reads = 0u64;
    let mut total_read_keys = 0u64;
    let mut total_deletes = 0u64;
    let mut total_deleted_keys = 0u64;
    let mut total_inserts = 0u64;
    let mut total_inserted_keys = 0u64;
    let mut total_proofs_verified = 0u64;
    let mut iter = 0u64;
    let mut epoch = start_epoch;

    while iter < num_iters {
        // Each epoch gets its own deterministic seed, so any epoch
        // can be reproduced independently with FUZZ_EPOCH=N FUZZ_ITERS=50
        let mut rng = SmallRng::seed_from_u64(base_seed.wrapping_add(epoch));
        let n = rng.gen_range(1..=2000u64) as usize;
        let cycles_this_epoch = (n as u64 / 5).max(1).min(num_iters - iter);

        // Generate n random unique keys, sorted
        let mut keys: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let len = rng.gen_range(1..=16usize);
                let mut k = vec![0u8; len];
                rng.fill_bytes(&mut k);
                k
            })
            .collect();
        keys.sort();
        keys.dedup();
        let n = keys.len();
        if n < 2 {
            epoch += 1;
            continue;
        }

        // Build tree: value = key ++ 0xAB repeated (distinguishable)
        let merk = InMemoryMerk::new();
        let batch: Vec<BatchEntry> = keys
            .iter()
            .map(|k| {
                let mut v = k.clone();
                v.extend_from_slice(&[0xAB; 8]);
                (k.clone(), Op::Put(v))
            })
            .collect();
        merk.apply_sorted_batch_ops(&batch).unwrap();

        let value_for = |k: &[u8]| -> Vec<u8> {
            let mut v = k.to_vec();
            v.extend_from_slice(&[0xAB; 8]);
            v
        };

        for _cycle in 0..cycles_this_epoch {
            // Pick a random contiguous slice of the key array [idx_lo, idx_hi)
            let idx_a = rng.gen_range(0..=n);
            let idx_b = rng.gen_range(0..=n);
            let (mut idx_lo, mut idx_hi) = if idx_a < idx_b {
                (idx_a, idx_b)
            } else {
                (idx_b, idx_a)
            };
            // 2% chance: force start to include (0) or exclude (1) first key
            if rng.gen_ratio(1, 50) {
                idx_lo = if rng.gen_bool(0.5) { 0 } else { 1 };
            }
            // 2% chance: force end to include (n) or exclude (n-1) last key
            if rng.gen_ratio(1, 50) {
                idx_hi = if rng.gen_bool(0.5) { n } else { n - 1 };
            }
            if idx_lo >= idx_hi {
                continue;
            }
            let range_keys = &keys[idx_lo..idx_hi];
            let range_size = range_keys.len();

            // Range boundaries for delete_range: use actual key values
            // start = keys[idx_lo], end = one-past keys[idx_hi-1]
            let start_key = &keys[idx_lo];
            // end_key must be > all keys in range. Use keys[idx_hi] if it exists,
            // otherwise append 0xFF to last key in range.
            let end_key = if idx_hi < n {
                keys[idx_hi].clone()
            } else {
                let mut ek = keys[idx_hi - 1].clone();
                ek.push(0xFF);
                ek
            };

            // Read phase: 1-4 queries with independent boundaries.
            let num_reads = rng.gen_range(1..=4usize);
            // Pick split indices within [idx_lo, idx_hi)
            let mut read_idxs: Vec<usize> = (0..num_reads.saturating_sub(1))
                .map(|_| rng.gen_range(idx_lo..idx_hi))
                .collect();
            read_idxs.sort();
            read_idxs.dedup();
            let mut read_boundaries = vec![start_key.clone()];
            for &idx in &read_idxs {
                read_boundaries.push(keys[idx].clone());
            }
            read_boundaries.push(end_key.clone());

            for window in read_boundaries.windows(2) {
                let (r_start, r_end) = (&window[0], &window[1]);
                if r_start >= r_end {
                    continue;
                }
                let mut query = Query::new();
                query.insert_range(r_start.clone()..r_end.clone());
                let mut query2 = Query::new();
                query2.insert_range(r_start.clone()..r_end.clone());
                let proof_bytes = merk.prove(query).unwrap();
                let root_hash = merk.root_hash();
                let results =
                    crate::proofs::query::verify_query(&proof_bytes, &query2, root_hash).unwrap();

                for (key, value) in &results {
                    assert_eq!(
                        value,
                        &value_for(key),
                        "read mismatch [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                        base_seed,
                        epoch
                    );
                }

                total_reads += 1;
                total_read_keys += results.len() as u64;
                total_proofs_verified += 1;
            }

            let delete_mode = if range_size >= 3 {
                rng.gen_range(0..3)
            } else if rng.gen_bool(0.5) {
                0
            } else {
                2
            };

            if delete_mode == 0 {
                // Mixed batch: DeleteRange + re-inserts in a single apply_sorted_batch_ops.
                let mut batch: Vec<BatchEntry> = Vec::new();
                batch.push((start_key.clone(), Op::DeleteRange(end_key.clone())));
                // Re-insert a random subset of the deleted keys in the same batch
                let mut reinserted_keys: Vec<&Vec<u8>> = Vec::new();
                for k in range_keys {
                    if rng.gen_bool(0.5) {
                        batch.push((k.clone(), Op::Put(value_for(k))));
                        reinserted_keys.push(k);
                    }
                }
                merk.apply_sorted_batch_ops(&batch).unwrap();
                total_deletes += 1;
                total_inserts += 1;

                // Verify via point gets
                for k in range_keys {
                    let was_reinserted = reinserted_keys.contains(&k);
                    if was_reinserted {
                        assert_eq!(
                            merk.get(k),
                            Some(value_for(k)),
                            "key {:?} should be present [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                            k, base_seed, epoch
                        );
                    } else {
                        assert!(
                            merk.get(k).is_none(),
                            "key {:?} should be deleted [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                            k, base_seed, epoch
                        );
                    }
                }

                // Re-insert the remaining keys to restore the tree
                let remaining: Vec<BatchEntry> = range_keys
                    .iter()
                    .filter(|k| merk.get(k).is_none())
                    .map(|k| (k.clone(), Op::Put(value_for(k))))
                    .collect();
                if !remaining.is_empty() {
                    merk.apply_sorted_batch_ops(&remaining).unwrap();
                }
            } else if delete_mode == 1 {
                // Mixed batch: overlapping DeleteRanges in a single apply_sorted_batch_ops.
                let first_end_idx = rng.gen_range(idx_lo + 2..=idx_hi);
                let overlap_start_idx = rng.gen_range(idx_lo + 1..first_end_idx);
                let first_end_key = if first_end_idx < n {
                    keys[first_end_idx].clone()
                } else {
                    end_key.clone()
                };
                let overlap_start_key = keys[overlap_start_idx].clone();
                let batch = vec![
                    (start_key.clone(), Op::DeleteRange(first_end_key)),
                    (overlap_start_key, Op::DeleteRange(end_key.clone())),
                ];
                merk.apply_sorted_batch_ops(&batch).unwrap();
                total_deletes += 2;

                for k in range_keys {
                    assert!(
                        merk.get(k).is_none(),
                        "key {:?} should be deleted by overlapping ranges [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                        k, base_seed, epoch
                    );
                }

                let reinsert: Vec<BatchEntry> = range_keys
                    .iter()
                    .map(|k| (k.clone(), Op::Put(value_for(k))))
                    .collect();
                merk.apply_sorted_batch_ops(&reinsert).unwrap();
                total_inserts += 1;
            } else {
                // Separate calls: 1-4 delete_ranges then verify via proof.
                let num_deletes = rng.gen_range(1..=4usize);
                let mut del_idxs: Vec<usize> = (0..num_deletes.saturating_sub(1))
                    .map(|_| rng.gen_range(idx_lo..idx_hi))
                    .collect();
                del_idxs.sort();
                del_idxs.dedup();
                let mut del_boundaries = vec![start_key.clone()];
                for &idx in &del_idxs {
                    del_boundaries.push(keys[idx].clone());
                }
                del_boundaries.push(end_key.clone());

                let mut del_ranges: Vec<(Vec<u8>, Vec<u8>)> = del_boundaries
                    .windows(2)
                    .map(|w| (w[0].clone(), w[1].clone()))
                    .filter(|(s, e)| s < e)
                    .collect();
                del_ranges.shuffle(&mut rng);

                for (d_start, d_end) in &del_ranges {
                    merk.apply_sorted_batch_ops(&[(
                        d_start.clone(),
                        Op::DeleteRange(d_end.clone()),
                    )])
                    .unwrap();
                    total_deletes += 1;
                }

                // Verify deletion via a range proof over a random sub-range
                // overlapping the deleted region
                let q_lo = rng.gen_range(idx_lo..idx_hi);
                let q_hi = rng.gen_range(q_lo + 1..=idx_hi);
                let q_start = &keys[q_lo];
                let q_end = if q_hi < n {
                    keys[q_hi].clone()
                } else {
                    let mut ek = keys[q_hi - 1].clone();
                    ek.push(0xFF);
                    ek
                };

                let mut query = Query::new();
                query.insert_range(q_start.clone()..q_end.clone());
                let mut query2 = Query::new();
                query2.insert_range(q_start.clone()..q_end.clone());
                let results = match merk.prove(query) {
                    Ok(proof_bytes) => {
                        let root_hash = merk.root_hash();
                        let results =
                            crate::proofs::query::verify_query(&proof_bytes, &query2, root_hash)
                                .unwrap();
                        total_proofs_verified += 1;
                        results
                    }
                    Err(Error::Proof(msg)) if msg == "Cannot create proof for empty tree" => {
                        assert_eq!(merk.root_hash(), crate::hash::NULL_HASH);
                        Vec::new()
                    }
                    Err(err) => panic!(
                        "proof failed: {:?} [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                        err, base_seed, epoch
                    ),
                };

                // The query range overlaps the deleted region [start_key, end_key).
                // Only keys outside the deleted range should appear.
                for (rk, rv) in &results {
                    assert!(
                        rk.as_slice() < start_key.as_slice()
                            || rk.as_slice() >= end_key.as_slice(),
                        "key {:?} should have been deleted but appeared in proof [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                        rk, base_seed, epoch
                    );
                    assert_eq!(
                        rv,
                        &value_for(rk),
                        "value mismatch for key {:?} [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                        rk, base_seed, epoch
                    );
                }

                // Re-insert all deleted keys in 1-4 batch chunks
                let num_inserts = rng.gen_range(1..=4usize);
                let mut ins_idxs: Vec<usize> = (0..num_inserts.saturating_sub(1))
                    .map(|_| rng.gen_range(idx_lo..idx_hi))
                    .collect();
                ins_idxs.sort();
                ins_idxs.dedup();
                let mut ins_slices: Vec<&[Vec<u8>]> = Vec::new();
                let mut prev = idx_lo;
                for &idx in &ins_idxs {
                    if idx > prev {
                        ins_slices.push(&keys[prev..idx]);
                    }
                    prev = idx;
                }
                if prev < idx_hi {
                    ins_slices.push(&keys[prev..idx_hi]);
                }

                for slice in &ins_slices {
                    let reinsert: Vec<BatchEntry> = slice
                        .iter()
                        .map(|k| (k.clone(), Op::Put(value_for(k))))
                        .collect();
                    if !reinsert.is_empty() {
                        merk.apply_sorted_batch_ops(&reinsert).unwrap();
                        total_inserts += 1;
                    }
                }
            }
            total_deleted_keys += range_size as u64;
            total_inserted_keys += range_size as u64;

            // Verify all keys are back
            for k in range_keys {
                assert_eq!(
                    merk.get(k),
                    Some(value_for(k)),
                    "key {:?} should be re-inserted [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
                    k, base_seed, epoch
                );
            }
        }

        // Full integrity check at end of each epoch
        let snap = merk.checkpoint().into_root().unwrap();
        assert_valid_avl(&snap);
        let current_keys = collect_keys(&snap);
        assert_eq!(
            current_keys, keys,
            "Key set mismatch [reproduce: FUZZ_SEED={} FUZZ_EPOCH={} FUZZ_ITERS=50]",
            base_seed, epoch
        );

        iter += cycles_this_epoch;
        epoch += 1;

        eprintln!(
            "  epoch {} (n={}, {} cycles, iter {}/{}): {} reads, {} deletes, {} inserts, {} proofs",
            epoch,
            n,
            cycles_this_epoch,
            iter,
            num_iters,
            total_reads,
            total_deletes,
            total_inserts,
            total_proofs_verified,
        );
    }

    eprintln!(
        "STRESS TEST COMPLETE: {} iters across {} epochs",
        iter, epoch
    );
    eprintln!(
        "  {} range reads ({} keys), {} delete_ranges ({} keys), {} batch inserts ({} keys), {} proofs",
        total_reads, total_read_keys,
        total_deletes, total_deleted_keys,
        total_inserts, total_inserted_keys,
        total_proofs_verified,
    );

    Ok(())
}
