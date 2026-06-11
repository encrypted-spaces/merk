#![allow(missing_docs)]

use crate::avl::node::Node;
use crate::avl::walker::Walker;
use crate::avl::PanicSource;
use crate::ops::{Batch, BatchEntry, Op};
use rand::prelude::*;
use std::convert::TryInto;
use std::ops::Range;

pub fn assert_tree_invariants(tree: &Node) {
    assert!(tree.balance_factor().abs() < 2);

    let lh = tree.child_ref(true).map_or(0, |c| c.height());
    let rh = tree.child_ref(false).map_or(0, |c| c.height());
    assert_eq!(
        tree.height(),
        1 + std::cmp::max(lh, rh),
        "cached height mismatch at key {:?}: cached={}, expected={}",
        tree.key(),
        tree.height(),
        1 + std::cmp::max(lh, rh),
    );

    let maybe_left = tree.child_ref(true);
    if let Some(left) = maybe_left {
        assert!(left.key() < tree.key());
        assert!(!left.is_modified());
    }

    let maybe_right = tree.child_ref(false);
    if let Some(right) = maybe_right {
        assert!(right.key() > tree.key());
        assert!(!right.is_modified());
    }

    if let Some(left) = tree.child(true) {
        assert_tree_invariants(left);
    }
    if let Some(right) = tree.child(false) {
        assert_tree_invariants(right);
    }
}

#[cfg(not(use_box))]
pub fn apply_memonly_unchecked(tree: Node, batch: &Batch) -> Node {
    let walker = Walker::<PanicSource>::new(tree, PanicSource {});
    let mut tree = Walker::<PanicSource>::apply_cow(Some(walker), batch, PanicSource {})
        .expect("apply failed")
        .0
        .expect("expected tree");
    tree.commit();
    tree
}

#[cfg(use_box)]
pub fn apply_memonly_unchecked(tree: Node, batch: &Batch) -> Node {
    let mut walker = Walker::<PanicSource>::new(tree, PanicSource {});
    let mut batch = batch.to_vec();
    walker.apply_in_place(&mut batch).expect("apply failed");
    let mut tree = walker.into_inner();
    tree.commit();
    tree
}

pub fn apply_memonly(tree: Node, batch: &Batch) -> Node {
    let tree = apply_memonly_unchecked(tree, batch);
    assert_tree_invariants(&tree);
    tree
}

#[cfg(not(use_box))]
pub fn apply_to_memonly(maybe_tree: Option<Node>, batch: &Batch) -> Option<Node> {
    let maybe_walker = maybe_tree.map(|tree| Walker::<PanicSource>::new(tree, PanicSource {}));
    Walker::<PanicSource>::apply_cow(maybe_walker, batch, PanicSource {})
        .expect("apply failed")
        .0
        .map(|mut tree| {
            tree.commit();
            println!("{:?}", &tree);
            assert_tree_invariants(&tree);
            tree
        })
}

#[cfg(use_box)]
pub fn apply_to_memonly(maybe_tree: Option<Node>, batch: &Batch) -> Option<Node> {
    match maybe_tree {
        Some(tree) => {
            let mut walker = Walker::<PanicSource>::new(tree, PanicSource {});
            let mut batch = batch.to_vec();
            match walker.apply_in_place(&mut batch) {
                Ok(_) => {
                    let mut tree = walker.into_inner();
                    tree.commit();
                    println!("{:?}", &tree);
                    assert_tree_invariants(&tree);
                    Some(tree)
                }
                Err(_) => None,
            }
        }
        None => {
            // No tree yet — use the functional path for initial construction
            let maybe_walker: Option<Walker<PanicSource>> = None;
            Walker::<PanicSource>::apply_to_mut(maybe_walker, &mut batch.to_vec(), PanicSource {})
                .expect("apply failed")
                .0
                .map(|mut tree| {
                    tree.commit();
                    println!("{:?}", &tree);
                    assert_tree_invariants(&tree);
                    tree
                })
        }
    }
}

pub fn seq_key(n: u64) -> Vec<u8> {
    n.to_be_bytes().to_vec()
}

pub fn put_entry_value() -> Vec<u8> {
    vec![123; 60]
}

pub fn put_entry(n: u64) -> BatchEntry {
    (seq_key(n), Op::Put(put_entry_value()))
}

pub fn del_entry(n: u64) -> BatchEntry {
    (seq_key(n), Op::Delete)
}

pub fn make_batch_seq(range: Range<u64>) -> Vec<BatchEntry> {
    let mut batch = Vec::with_capacity((range.end - range.start).try_into().unwrap());
    for n in range {
        batch.push(put_entry(n));
    }
    batch
}

pub fn make_del_batch_seq(range: Range<u64>) -> Vec<BatchEntry> {
    let mut batch = Vec::with_capacity((range.end - range.start).try_into().unwrap());
    for n in range {
        batch.push(del_entry(n));
    }
    batch
}

pub fn make_batch_rand(size: u64, seed: u64) -> Vec<BatchEntry> {
    let mut rng: SmallRng = SeedableRng::seed_from_u64(seed);
    let mut batch = Vec::with_capacity(size.try_into().unwrap());
    for _ in 0..size {
        let n = rng.gen::<u64>();
        batch.push(put_entry(n));
    }
    batch.sort_by(|a, b| a.0.cmp(&b.0));
    batch
}

pub fn make_del_batch_rand(size: u64, seed: u64) -> Vec<BatchEntry> {
    let mut rng: SmallRng = SeedableRng::seed_from_u64(seed);
    let mut batch = Vec::with_capacity(size.try_into().unwrap());
    for _ in 0..size {
        let n = rng.gen::<u64>();
        batch.push(del_entry(n));
    }
    batch.sort_by(|a, b| a.0.cmp(&b.0));
    batch
}

pub fn make_tree_rand(node_count: u64, batch_size: u64, initial_seed: u64) -> Node {
    assert!(node_count >= batch_size);
    assert!(node_count.is_multiple_of(batch_size));

    let value = vec![123; 60];
    let mut tree = Node::new(vec![0; 20], value).expect("Node construction failed");

    let batch_count = node_count / batch_size;
    for seed in initial_seed..(initial_seed + batch_count) {
        let batch = make_batch_rand(batch_size, seed);
        tree = apply_memonly(tree, &batch);
    }

    tree
}

pub fn make_tree_seq(node_count: u64) -> Node {
    let batch_size = if node_count >= 10_000 {
        assert!(node_count.is_multiple_of(10_000));
        10_000
    } else {
        node_count
    };

    let value = vec![123; 60];
    let mut tree = Node::new(vec![0; 20], value).expect("Node construction failed");

    let batch_count = node_count / batch_size;
    for i in 0..batch_count {
        let batch = make_batch_seq((i * batch_size)..((i + 1) * batch_size));
        tree = apply_memonly(tree, &batch);
    }

    tree
}
