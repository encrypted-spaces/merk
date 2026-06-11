use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use sha2::{Digest, Sha256};

use super::cursor::Cursor;
use super::tree::*;
use super::*;
use crate::hash::{Hash, NULL_HASH};
use crate::ops::{BatchEntry, Op};
use crate::proofs::query::{Query, QueryItem};
use crate::tracer::test_support::mrt::{create_trace, replay_trace, root_after_writes};
use crate::tracer::test_support::Step;
use crate::tracer::{BatchOp, ReadOp, WriteOp};
use crate::Error;

fn root_hash(root: Option<&Arc<MrtNodeInner>>) -> Hash {
    root.map_or(NULL_HASH, |node| node.hash())
}

fn build_tree(entries: &[(Vec<u8>, Vec<u8>)]) -> Option<Arc<MrtNodeInner>> {
    let mut root = None;
    for (key, value) in entries {
        root = Some(insert(root, key.clone(), value.clone()).unwrap());
    }
    root
}

fn get_owned(root: Option<&Arc<MrtNodeInner>>, key: &[u8]) -> Option<Vec<u8>> {
    get(root, key).unwrap().map(ToOwned::to_owned)
}

type TestCursor<'a> = Cursor<'a, &'a Arc<MrtNodeInner>>;

fn collect_cursor_forward(mut cursor: TestCursor<'_>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut visit = |_: &RoutePrefix, _: &Arc<MrtNodeInner>| {};
    while let Some((key, value)) = cursor.next(&mut visit).unwrap() {
        out.push((key.to_vec(), value.to_vec()));
    }
    out
}

fn collect_cursor_reverse(mut cursor: TestCursor<'_>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut visit = |_: &RoutePrefix, _: &Arc<MrtNodeInner>| {};
    while let Some((key, value)) = cursor.prev(&mut visit).unwrap() {
        out.push((key.to_vec(), value.to_vec()));
    }
    out
}

fn cursor_seek_ge<'a>(root: &'a Arc<MrtNodeInner>, start: &[u8]) -> Result<TestCursor<'a>> {
    let mut visit = |_: &RoutePrefix, _: &Arc<MrtNodeInner>| {};
    Cursor::seek_ge(root, start, &mut visit)
}

fn cursor_seek_le<'a>(root: &'a Arc<MrtNodeInner>, start: &[u8]) -> Result<TestCursor<'a>> {
    let mut visit = |_: &RoutePrefix, _: &Arc<MrtNodeInner>| {};
    Cursor::seek_le(root, start, &mut visit)
}

fn cursor_seek_lt<'a>(root: &'a Arc<MrtNodeInner>, end: &[u8]) -> Result<TestCursor<'a>> {
    let mut visit = |_: &RoutePrefix, _: &Arc<MrtNodeInner>| {};
    Cursor::seek_lt(root, end, &mut visit)
}

fn collect_hashes(arcs: &[Arc<MrtNodeInner>]) -> Vec<Hash> {
    arcs.iter().map(|arc| arc.hash()).collect()
}

/// Suffix-relative leaf hash of a *root* leaf (skip = full route of `key`):
/// `H(skip.bit_len:u32-BE ‖ packed ‖ value ‖ 0x01)` (§3).
fn manual_leaf_hash(key: &[u8], value: &[u8]) -> Hash {
    let skip = route_bits_of(key);
    let mut h = Sha256::new();
    h.update((skip.bit_len() as u32).to_be_bytes());
    h.update(skip.packed_bytes());
    h.update(value);
    h.update([0x01]);
    h.finalize().into()
}

/// `H(left ‖ right ‖ left_depth:u32-BE ‖ right_depth:u32-BE ‖ skip.bit_len:u32-BE
/// ‖ packed ‖ 0x00)` — the branch preimage now commits each child's depth_below.
fn manual_branch_hash(
    skip: &RouteBits,
    left_hash: Hash,
    right_hash: Hash,
    left_depth: u16,
    right_depth: u16,
) -> Hash {
    let mut h = Sha256::new();
    h.update(left_hash);
    h.update(right_hash);
    h.update((left_depth as u32).to_be_bytes());
    h.update((right_depth as u32).to_be_bytes());
    h.update((skip.bit_len() as u32).to_be_bytes());
    h.update(skip.packed_bytes());
    h.update([0x00]);
    h.finalize().into()
}

fn route_cmp(a: &[u8], b: &[u8]) -> Ordering {
    let max_bits = route_len(a).max(route_len(b));
    for position in 0..max_bits {
        match (route_bit_at(a, position), route_bit_at(b, position)) {
            (false, true) => return Ordering::Less,
            (true, false) => return Ordering::Greater,
            _ => {}
        }
    }
    Ordering::Equal
}

#[derive(Clone, Debug)]
enum ModelPayload {
    Value(Vec<u8>),
}

#[derive(Clone, Debug)]
struct ModelValue {
    visible_value: Vec<u8>,
    payload: ModelPayload,
}

type Model = BTreeMap<Vec<u8>, ModelValue>;

fn payload_to_op(payload: &ModelPayload) -> Op {
    match payload {
        ModelPayload::Value(value) => Op::Put(value.clone()),
    }
}

fn reference_root_hash(model: &Model) -> Hash {
    let merk = Tree::new();
    let batch: Vec<BatchEntry> = model
        .iter()
        .map(|(key, value)| (key.clone(), payload_to_op(&value.payload)))
        .collect();
    merk.apply_sorted_batch_ops(&batch).unwrap();
    merk.root_hash()
}

fn apply_to_model(model: &mut Model, batch: &[BatchEntry]) {
    for (key, op) in batch {
        match op {
            Op::Put(value) => {
                model.insert(
                    key.clone(),
                    ModelValue {
                        visible_value: value.clone(),
                        payload: ModelPayload::Value(value.clone()),
                    },
                );
            }
            Op::Delete => {
                model.remove(key);
            }
            Op::DeleteRange(end) => {
                let keys: Vec<Vec<u8>> = model
                    .range(key.clone()..end.clone())
                    .map(|(k, _)| k.clone())
                    .collect();
                for key in keys {
                    model.remove(&key);
                }
            }
        }
    }
}

fn visible_entries(model: &Model) -> Vec<(Vec<u8>, Vec<u8>)> {
    model
        .iter()
        .map(|(key, value)| (key.clone(), value.visible_value.clone()))
        .collect()
}

fn random_bytes<R: RngCore>(rng: &mut R, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    rng.fill_bytes(&mut bytes);
    bytes
}

fn random_key<R: RngCore>(rng: &mut R) -> Vec<u8> {
    // Keys are fixed-length, hence prefix-free (no key can be a byte-prefix of
    // another) — the property the tree now assumes. Shared random prefixes still
    // produce branches at varied bit depths, so skip-length coverage is retained.
    random_bytes(rng, 4)
}

fn random_value<R: RngCore>(rng: &mut R) -> Vec<u8> {
    let len = match rng.gen_range(0..=10) {
        0 => 0,
        1..=4 => 1,
        5..=8 => 8,
        9 => 32,
        _ => 128,
    };
    random_bytes(rng, len)
}

fn choose_existing_key<R: Rng>(rng: &mut R, model: &Model) -> Option<Vec<u8>> {
    if model.is_empty() {
        None
    } else {
        let idx = rng.gen_range(0..model.len());
        Some(model.keys().nth(idx).unwrap().clone())
    }
}

fn random_range_end<R: RngCore>(rng: &mut R, start: &[u8]) -> Vec<u8> {
    if rng.gen_bool(0.35) {
        let mut end = start.to_vec();
        end.push(rng.gen_range(1..=0xff));
        return end;
    }

    for _ in 0..10 {
        let candidate = random_key(rng);
        if candidate.as_slice() > start {
            return candidate;
        }
    }

    let mut end = start.to_vec();
    end.push(1);
    end
}

fn random_batch_with_ranges<R: RngCore>(rng: &mut R, model: &Model) -> Vec<BatchEntry> {
    let n = rng.gen_range(0..=8);
    let mut entries = Vec::new();
    let mut used_keys = BTreeSet::new();

    for _ in 0..n {
        let key = if let Some(key) = choose_existing_key(rng, model).filter(|_| rng.gen_bool(0.5)) {
            key
        } else {
            random_key(rng)
        };

        if used_keys.contains(&key) {
            continue;
        }

        if rng.gen_bool(0.28) {
            let end = random_range_end(rng, &key);
            entries.push((key.clone(), Op::DeleteRange(end)));
            if rng.gen_bool(0.35) {
                let op = match rng.gen_range(0..=1) {
                    0 => Op::Put(random_value(rng)),
                    _ => Op::Delete,
                };
                entries.push((key.clone(), op));
            }
        } else {
            let delete = if model.contains_key(&key) {
                rng.gen_bool(0.30)
            } else {
                rng.gen_bool(0.08)
            };
            let op = if delete {
                Op::Delete
            } else {
                Op::Put(random_value(rng))
            };
            entries.push((key.clone(), op));
        }

        used_keys.insert(key);
    }

    entries.sort_by(|(a_key, a_op), (b_key, b_op)| {
        a_key.cmp(b_key).then_with(|| {
            let a_is_range = matches!(a_op, Op::DeleteRange(_));
            let b_is_range = matches!(b_op, Op::DeleteRange(_));
            b_is_range.cmp(&a_is_range)
        })
    });
    entries
}

fn assert_merk_matches_model(merk: &Tree, model: &Model, seed: u64, step: usize) {
    let expected_root = reference_root_hash(model);
    let snapshot = merk.checkpoint();
    assert_eq!(
        merk.root_hash(),
        expected_root,
        "seed={seed} step={step}: root_hash"
    );
    assert_eq!(
        snapshot.root_hash(),
        expected_root,
        "seed={seed} step={step}: snapshot root_hash"
    );

    if model.is_empty() {
        assert_eq!(merk.root_hash(), NULL_HASH, "seed={seed} step={step}");
        assert!(snapshot.is_empty(), "seed={} step={}", seed, step);
    } else {
        assert_ne!(merk.root_hash(), NULL_HASH, "seed={seed} step={step}");
        assert!(!snapshot.is_empty(), "seed={} step={}", seed, step);
    }

    for (key, value) in model.iter().take(8) {
        assert_eq!(
            merk.get(key),
            Some(value.visible_value.clone()),
            "seed={seed} step={step}: get({key:?})"
        );
        assert_eq!(
            snapshot.get(key),
            Some(value.visible_value.clone()),
            "seed={seed} step={step}: snapshot get({key:?})"
        );
    }

    let forward = visible_entries(model);
    assert_eq!(
        snapshot.iter().collect::<Vec<_>>(),
        forward,
        "seed={seed} step={step}: snapshot iter"
    );

    let mut reverse = forward.clone();
    reverse.reverse();
    assert_eq!(
        snapshot.reverse_iter().collect::<Vec<_>>(),
        reverse,
        "seed={seed} step={step}: snapshot reverse_iter"
    );

    let mut probe_keys = vec![Vec::new(), vec![0x00], vec![0xff]];
    if let Some(first) = model.keys().next() {
        probe_keys.push(first.clone());
        let mut after_first = first.clone();
        after_first.push(0);
        probe_keys.push(after_first);
    }
    if let Some(mid) = model.keys().nth(model.len() / 2) {
        probe_keys.push(mid.clone());
    }
    if let Some(last) = model.keys().next_back() {
        probe_keys.push(last.clone());
    }

    for probe in probe_keys {
        let expected_from: Vec<_> = model
            .range(probe.clone()..)
            .map(|(key, value)| (key.clone(), value.visible_value.clone()))
            .collect();
        assert_eq!(
            snapshot.iter_from(&probe).collect::<Vec<_>>(),
            expected_from,
            "seed={seed} step={step}: snapshot iter_from({probe:?})"
        );

        let mut expected_reverse_from: Vec<_> = model
            .range(..=probe.clone())
            .map(|(key, value)| (key.clone(), value.visible_value.clone()))
            .collect();
        expected_reverse_from.reverse();
        assert_eq!(
            snapshot.reverse_iter_from(&probe).collect::<Vec<_>>(),
            expected_reverse_from,
            "seed={seed} step={step}: snapshot reverse_iter_from({probe:?})"
        );
    }
}

#[test]
fn mrt_insert_single_key_changes_hash() {
    let root = insert(None, b"a".to_vec(), b"va".to_vec()).unwrap();

    assert_ne!(root_hash(Some(&root)), NULL_HASH);
    assert_eq!(get_owned(Some(&root), b"a"), Some(b"va".to_vec()));
}

#[test]
fn mrt_insert_multiple_keys_gets_values() {
    // Prefix-free (equal-length) key set.
    let entries = [
        (b"aa".to_vec(), b"va".to_vec()),
        (b"bb".to_vec(), b"vb".to_vec()),
        (b"cc".to_vec(), b"vc".to_vec()),
        (b"ab".to_vec(), b"vabc".to_vec()),
        (b"az".to_vec(), b"vabcd".to_vec()),
    ];
    let root = build_tree(&entries);

    for (key, value) in entries {
        assert_eq!(get_owned(root.as_ref(), &key), Some(value));
    }
    assert_eq!(get_owned(root.as_ref(), b"zz"), None);
}

#[test]
fn mrt_get_misses_prefix_and_extension_queries() {
    // A *query* need not be prefix-free; it simply isn't present. A query that
    // extends a stored key, or is a prefix of one, misses (exact-length match).
    let single = build_tree(&[(b"a".to_vec(), b"va".to_vec())]);
    assert_eq!(get_owned(single.as_ref(), b"a"), Some(b"va".to_vec()));
    assert_eq!(get_owned(single.as_ref(), b"ab"), None); // extends "a"

    let longer = build_tree(&[(b"ab".to_vec(), b"vab".to_vec())]);
    assert_eq!(get_owned(longer.as_ref(), b"ab"), Some(b"vab".to_vec()));
    assert_eq!(get_owned(longer.as_ref(), b"a"), None); // prefix of "ab"
}

#[test]
fn mrt_insert_rejects_prefix_keys() {
    // Storing a key that is a byte-prefix of an existing one — in either order —
    // violates the prefix-free assumption and is rejected.
    let base = build_tree(&[(b"ab".to_vec(), b"v".to_vec())]);
    let err = insert(base, b"a".to_vec(), b"v".to_vec()).unwrap_err();
    assert!(matches!(err, Error::Key(_)), "got {:?}", err);

    let base = Some(insert(None, b"a".to_vec(), b"v".to_vec()).unwrap());
    let err = insert(base, b"abc".to_vec(), b"v".to_vec()).unwrap_err();
    assert!(matches!(err, Error::Key(_)), "got {:?}", err);
}

#[test]
fn mrt_duplicate_insert_replaces_value() {
    let tree1 = build_tree(&[
        (b"a".to_vec(), b"v1".to_vec()),
        (b"b".to_vec(), b"x".to_vec()),
        (b"a".to_vec(), b"v2".to_vec()),
    ]);
    assert_eq!(get_owned(tree1.as_ref(), b"a"), Some(b"v2".to_vec()));
    assert_eq!(get_owned(tree1.as_ref(), b"b"), Some(b"x".to_vec()));

    let tree2 = build_tree(&[
        (b"a".to_vec(), b"v2".to_vec()),
        (b"b".to_vec(), b"x".to_vec()),
    ]);
    assert_eq!(root_hash(tree1.as_ref()), root_hash(tree2.as_ref()));
}

#[test]
fn mrt_leaf_branch_and_pruned_hashes_match_fixtures() {
    let key = b"known-key".to_vec();
    let value = b"known-value".to_vec();
    let leaf = MrtNodeInner::leaf_value(key.clone(), value.clone());
    assert_eq!(leaf.hash(), manual_leaf_hash(&key, &value));

    let left_hash = [0x11; 32];
    let right_hash = [0x22; 32];
    let left_depth = 11u16;
    let right_depth = 22u16;
    let skip = RouteBits::from_key_range(b"branch-key", 0, 37);
    let branch = MrtNodeInner::branch(
        skip.clone(),
        MrtNodeInner::pruned(left_hash, left_depth),
        MrtNodeInner::pruned(right_hash, right_depth),
    );
    assert_eq!(
        branch.hash(),
        manual_branch_hash(&skip, left_hash, right_hash, left_depth, right_depth)
    );
    // The branch rolls up its own depth_below from the children: skip + 1 + max.
    assert_eq!(branch.depth_below(), 37 + 1 + 22);

    let pruned_hash = [0x7a; 32];
    let pruned = MrtNodeInner::pruned(pruned_hash, 99);
    assert!(matches!(pruned.node(), MrtNode::PrunedHash));
    assert_eq!(pruned.hash(), pruned_hash);
    // A pruned stub reports its carried depth_below.
    assert_eq!(pruned.depth_below(), 99);
}

#[test]
fn mrt_delete_existing_key_removes_it() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ]);

    let (after, deleted) = delete(root, b"b").unwrap();

    assert!(deleted);
    assert_eq!(get_owned(after.as_ref(), b"a"), Some(b"va".to_vec()));
    assert_eq!(get_owned(after.as_ref(), b"b"), None);
    assert_eq!(get_owned(after.as_ref(), b"c"), Some(b"vc".to_vec()));
}

#[test]
fn mrt_delete_nonexistent_key_preserves_hash() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
    ]);
    let before = root_hash(root.as_ref());

    let (after, deleted) = delete(root, b"z").unwrap();

    assert!(!deleted);
    assert_eq!(root_hash(after.as_ref()), before);
}

#[test]
fn mrt_delete_collapses_to_canonical_tree() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ]);
    let (after, deleted) = delete(root, b"b").unwrap();
    assert!(deleted);

    let expected = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ]);
    assert_eq!(root_hash(after.as_ref()), root_hash(expected.as_ref()));
    assert_eq!(get_owned(after.as_ref(), b"a"), Some(b"va".to_vec()));
    assert_eq!(get_owned(after.as_ref(), b"b"), None);
    assert_eq!(get_owned(after.as_ref(), b"c"), Some(b"vc".to_vec()));
}

#[test]
fn mrt_delete_from_two_leaf_tree_collapses_to_single_leaf() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
    ]);

    let (after, deleted) = delete(root, b"a").unwrap();

    assert!(deleted);
    match after.as_ref().unwrap().node() {
        // The survivor is lifted to the root, so it re-skips to "b"'s full route.
        MrtNode::Leaf { skip, value } => {
            assert_eq!(skip, &route_bits_of(b"b"));
            assert_eq!(value, b"vb");
        }
        other => panic!("expected single leaf after collapse, got {:?}", other),
    }
    let expected = MrtNodeInner::leaf_value(b"b".to_vec(), b"vb".to_vec());
    assert_eq!(root_hash(after.as_ref()), expected.hash());
}

#[test]
fn mrt_delete_range_removes_half_open_range() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
        (b"d".to_vec(), b"vd".to_vec()),
        (b"e".to_vec(), b"ve".to_vec()),
    ]);

    let after = delete_range(root, b"b", b"d").unwrap();

    assert_eq!(get_owned(after.as_ref(), b"a"), Some(b"va".to_vec()));
    assert_eq!(get_owned(after.as_ref(), b"b"), None);
    assert_eq!(get_owned(after.as_ref(), b"c"), None);
    assert_eq!(get_owned(after.as_ref(), b"d"), Some(b"vd".to_vec()));
    assert_eq!(get_owned(after.as_ref(), b"e"), Some(b"ve".to_vec()));
}

#[test]
fn mrt_delete_range_all_no_match_and_invalid_bounds() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ]);

    let all_deleted = delete_range(root.clone(), b"a", b"z").unwrap();
    assert!(all_deleted.is_none());

    let before = root_hash(root.as_ref());
    let no_match = delete_range(root.clone(), b"x", b"z").unwrap();
    assert_eq!(root_hash(no_match.as_ref()), before);
    assert_eq!(get_owned(no_match.as_ref(), b"a"), Some(b"va".to_vec()));
    assert_eq!(get_owned(no_match.as_ref(), b"b"), Some(b"vb".to_vec()));
    assert_eq!(get_owned(no_match.as_ref(), b"c"), Some(b"vc".to_vec()));

    let equal_bound = delete_range(root.clone(), b"b", b"b").unwrap_err();
    assert!(matches!(equal_bound, Error::Key(_)));

    let reversed_bound = delete_range(root, b"c", b"b").unwrap_err();
    assert!(matches!(reversed_bound, Error::Key(_)));
}

#[test]
fn mrt_delete_range_rejects_overlong_start_and_end() {
    let too_long = vec![0u8; MAX_KEY_LEN + 1];

    let start_err = delete_range(None, &too_long, &[0xff]).unwrap_err();
    assert!(matches!(start_err, Error::Key(_)));

    let end_err = delete_range(None, &[], &too_long).unwrap_err();
    assert!(matches!(end_err, Error::Key(_)));
}

#[test]
fn mrt_delete_range_into_pruned_node_errors() {
    let leaf = MrtNodeInner::leaf_value(vec![0x00], b"v".to_vec());
    let pruned = MrtNodeInner::pruned([0xee; 32], 0);
    let root = MrtNodeInner::branch(RouteBits::empty(), leaf, pruned);

    // The range [0x80, 0x81) lies entirely in the pruned right subtree.
    let err = delete_range(Some(root), &[0x80], &[0x81]).unwrap_err();

    assert!(matches!(err, Error::PrunedNode(_)));
}

#[test]
fn mrt_delete_range_is_order_independent() {
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u32..16)
        .map(|i| {
            (
                format!("key-{i:08x}").into_bytes(),
                format!("value-{i}").into_bytes(),
            )
        })
        .collect();
    let start: &[u8] = b"key-00000004";
    let end: &[u8] = b"key-0000000c";

    let expected_entries: Vec<(Vec<u8>, Vec<u8>)> = entries
        .iter()
        .filter(|(key, _)| key.as_slice() < start || key.as_slice() >= end)
        .cloned()
        .collect();
    let expected = root_hash(build_tree(&expected_entries).as_ref());

    let canonical_after = delete_range(build_tree(&entries), start, end).unwrap();
    assert_eq!(root_hash(canonical_after.as_ref()), expected);

    let mut rng = SmallRng::seed_from_u64(0xD371E7E);
    for trial in 0..120 {
        let mut shuffled = entries.clone();
        shuffled.shuffle(&mut rng);
        let after = delete_range(build_tree(&shuffled), start, end).unwrap();
        assert_eq!(
            root_hash(after.as_ref()),
            expected,
            "trial {trial} produced a different root"
        );
    }
}

#[test]
fn mrt_delete_range_handles_clustered_keys() {
    // Range-delete over a clustered, prefix-free key set: partial sub-ranges leave
    // the right survivors. (Range bounds themselves may be arbitrary.)
    let entries = [
        (b"abca".to_vec(), b"v-abca".to_vec()),
        (b"abcb".to_vec(), b"v-abcb".to_vec()),
        (b"abda".to_vec(), b"v-abda".to_vec()),
        (b"abdb".to_vec(), b"v-abdb".to_vec()),
    ];

    // [abca, abcc) removes the two "abc*" leaves, leaving the "abd*" cluster.
    let after = delete_range(build_tree(&entries), b"abca", b"abcc").unwrap();
    let expected = build_tree(&[
        (b"abda".to_vec(), b"v-abda".to_vec()),
        (b"abdb".to_vec(), b"v-abdb".to_vec()),
    ]);
    assert_eq!(root_hash(after.as_ref()), root_hash(expected.as_ref()));
    assert_eq!(get_owned(after.as_ref(), b"abca"), None);
    assert_eq!(get_owned(after.as_ref(), b"abcb"), None);
    assert_eq!(get_owned(after.as_ref(), b"abda"), Some(b"v-abda".to_vec()));

    // A mid-range bound that splits a cluster: [abcb, abda) removes only "abcb".
    let after_mid = delete_range(build_tree(&entries), b"abcb", b"abda").unwrap();
    let expected_mid = build_tree(&[
        (b"abca".to_vec(), b"v-abca".to_vec()),
        (b"abda".to_vec(), b"v-abda".to_vec()),
        (b"abdb".to_vec(), b"v-abdb".to_vec()),
    ]);
    assert_eq!(
        root_hash(after_mid.as_ref()),
        root_hash(expected_mid.as_ref())
    );
    assert_eq!(
        get_owned(after_mid.as_ref(), b"abca"),
        Some(b"v-abca".to_vec())
    );
    assert_eq!(get_owned(after_mid.as_ref(), b"abcb"), None);
    assert_eq!(
        get_owned(after_mid.as_ref(), b"abda"),
        Some(b"v-abda".to_vec())
    );
}

// ─── move_prefix stage 1: locate + detach prefix subtree ─────────────────────

/// Every (key, value) stored in `root`, in iteration order — used to derive the
/// independent prefix-partition oracle below.
fn all_entries(root: Option<&Arc<MrtNodeInner>>) -> Vec<(Vec<u8>, Vec<u8>)> {
    match root {
        None => Vec::new(),
        Some(root) => collect_cursor_forward(Cursor::first(root)),
    }
}

/// Independent oracle: rebuild `root` keeping only the keys that do **not** have
/// byte-prefix `p` (i.e. the result of dropping the prefix-`p` subtree).
fn model_drop_prefix(root: Option<&Arc<MrtNodeInner>>, p: &[u8]) -> Option<Arc<MrtNodeInner>> {
    let kept: Vec<_> = all_entries(root)
        .into_iter()
        .filter(|(k, _)| !k.starts_with(p))
        .collect();
    build_tree(&kept)
}

/// Independent oracle for Stage 2: rebuild the whole tree after moving every
/// `p || suffix` key to `q || suffix`, preserving values. Returns `None` if a
/// moved key would exceed the MRT key-length cap.
fn model_move_prefix(
    entries: &[(Vec<u8>, Vec<u8>)],
    p: &[u8],
    q: &[u8],
) -> Option<Arc<MrtNodeInner>> {
    let mut moved = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        if k.starts_with(p) {
            let suffix_len = k.len() - p.len();
            if q.len().saturating_add(suffix_len) > MAX_KEY_LEN {
                return None;
            }
            let mut key = q.to_vec();
            key.extend_from_slice(&k[p.len()..]);
            moved.push((key, v.clone()));
        } else {
            moved.push((k.clone(), v.clone()));
        }
    }
    build_tree(&moved)
}

/// Number of leaves under `node` — the count of distinct keys in a (sub)tree.
fn count_leaves(node: &Arc<MrtNodeInner>) -> usize {
    match node.node() {
        MrtNode::Leaf { .. } => 1,
        MrtNode::Branch { left, right, .. } => count_leaves(left) + count_leaves(right),
        MrtNode::PrunedHash => 0,
    }
}

#[test]
fn mrt_locate_prefix_subtree_root_leaf() {
    // A single-key tree: the root *is* a leaf, so any byte-prefix of that key
    // resolves to the whole tree at entry_depth 0.
    let root = build_tree(&[(b"k".to_vec(), b"v".to_vec())]).unwrap();

    // p == the full key.
    let locus = locate_prefix_subtree(&root, b"k").unwrap();
    assert!(Arc::ptr_eq(locus.s_root, &root));
    assert_eq!(locus.entry_depth, 0);
    assert_eq!(locus.strip_prefix_bits, 8); // 8·|"k"| − 0

    // p == "" (empty prefix): still the root leaf, stripping nothing.
    let empty = locate_prefix_subtree(&root, b"").unwrap();
    assert!(Arc::ptr_eq(empty.s_root, &root));
    assert_eq!(empty.entry_depth, 0);
    assert_eq!(empty.strip_prefix_bits, 0);
}

#[test]
fn mrt_locate_prefix_subtree_root_branch() {
    // Two keys sharing byte 0x61 then diverging at the very next (byte-aligned)
    // bit: the root is a branch deciding at bit 8, and prefix "a" stops *at* it.
    let root = build_tree(&[
        (vec![0x61, 0x00], b"v0".to_vec()),
        (vec![0x61, 0x80], b"v1".to_vec()),
    ])
    .unwrap();

    let locus = locate_prefix_subtree(&root, &[0x61]).unwrap();
    assert!(Arc::ptr_eq(locus.s_root, &root)); // both keys share "a" → whole tree
    assert_eq!(locus.entry_depth, 0);
    assert_eq!(locus.strip_prefix_bits, 8);
}

#[test]
fn mrt_locate_prefix_subtree_mid_skip_and_at_branch() {
    // "abc"/"abd" share a long run of bits (a multi-bit branch skip) and "z"
    // splits off near the root. The "ab*" cluster hangs under the root's left
    // child, entered at depth 4 (root decides at bit 3: 'a'=0x61 vs 'z'=0x7a).
    let root = build_tree(&[
        (b"abc".to_vec(), b"v-abc".to_vec()),
        (b"abd".to_vec(), b"v-abd".to_vec()),
        (b"z".to_vec(), b"v-z".to_vec()),
    ])
    .unwrap();

    // p = "ab" ends *mid-skip* of the "ab*" branch (it spans bits [4, 21)).
    let mid = locate_prefix_subtree(&root, b"ab").unwrap();
    assert_eq!(mid.entry_depth, 4);
    assert_eq!(mid.strip_prefix_bits, 16 - 4); // 8·|"ab"| − entry_depth
    assert_eq!(count_leaves(mid.s_root), 2); // exactly "abc", "abd"

    // p = "a" ends within the same skip, one byte earlier.
    let shorter = locate_prefix_subtree(&root, b"a").unwrap();
    assert_eq!(shorter.entry_depth, 4);
    assert_eq!(shorter.strip_prefix_bits, 8 - 4);
    assert!(Arc::ptr_eq(shorter.s_root, mid.s_root)); // same subtree S
}

#[test]
fn mrt_locate_prefix_subtree_at_branch_non_root() {
    // The "a*" cluster ({0x61,0x00},{0x61,0x80}) hangs under the root's left
    // child (root decides at bit 3 vs "z"); that child branch decides at the
    // byte-aligned bit 8, so prefix "a" stops *at* a non-root branch decision.
    let root = build_tree(&[
        (vec![0x61, 0x00], b"v0".to_vec()),
        (vec![0x61, 0x80], b"v1".to_vec()),
        (b"z".to_vec(), b"v-z".to_vec()),
    ])
    .unwrap();

    let locus = locate_prefix_subtree(&root, &[0x61]).unwrap();
    assert_eq!(locus.entry_depth, 4); // root skip [0,3) + decision bit → child at 4
    assert_eq!(locus.strip_prefix_bits, 8 - 4);
    assert_eq!(count_leaves(locus.s_root), 2);
    assert!(!Arc::ptr_eq(locus.s_root, &root)); // a proper subtree, not the whole tree
}

#[test]
fn mrt_locate_prefix_subtree_non_root_leaf() {
    // "abc" is the only key under the root's left child, so that child is a leaf;
    // prefix "ab" is exhausted within it.
    let root = build_tree(&[
        (b"abc".to_vec(), b"v-abc".to_vec()),
        (b"z".to_vec(), b"v-z".to_vec()),
    ])
    .unwrap();

    let locus = locate_prefix_subtree(&root, b"ab").unwrap();
    assert!(matches!(locus.s_root.node(), MrtNode::Leaf { .. }));
    assert_eq!(locus.entry_depth, 4);
    assert_eq!(locus.strip_prefix_bits, 16 - 4);
    assert_eq!(count_leaves(locus.s_root), 1);
}

#[test]
fn mrt_locate_prefix_subtree_deep_subtree() {
    // A deeper tree: the "user:" namespace clusters several levels down.
    let root = build_tree(&[
        (b"user:alice".to_vec(), b"1".to_vec()),
        (b"user:alex".to_vec(), b"2".to_vec()),
        (b"user:bob".to_vec(), b"3".to_vec()),
        (b"sys:log".to_vec(), b"4".to_vec()),
        (b"zzz".to_vec(), b"5".to_vec()),
    ])
    .unwrap();

    let locus = locate_prefix_subtree(&root, b"user:").unwrap();
    assert_eq!(count_leaves(locus.s_root), 3); // alice, alex, bob
    assert!(locus.entry_depth > 0);
    assert_eq!(locus.strip_prefix_bits, 8 * 5 - locus.entry_depth);
}

#[test]
fn mrt_locate_prefix_subtree_all_ff_is_a_normal_locus() {
    // An all-0xFF prefix needs no special-casing — it is just a normal locus.
    let root = build_tree(&[
        (vec![0xff, 0x00], b"a".to_vec()),
        (vec![0xff, 0x80], b"b".to_vec()),
        (vec![0x00], b"c".to_vec()),
    ])
    .unwrap();

    let locus = locate_prefix_subtree(&root, &[0xff]).unwrap();
    assert_eq!(count_leaves(locus.s_root), 2); // the two 0xff* keys
    assert_eq!(locus.entry_depth, 1); // root decides at bit 0, 0xff* child at 1
    assert_eq!(locus.strip_prefix_bits, 8 - 1);
}

#[test]
fn mrt_locate_prefix_subtree_absent_is_err() {
    let root = build_tree(&[
        (b"abc".to_vec(), b"1".to_vec()),
        (b"abd".to_vec(), b"2".to_vec()),
        (b"z".to_vec(), b"3".to_vec()),
    ])
    .unwrap();

    // Diverges at the root (no key begins with 'm').
    assert!(matches!(
        locate_prefix_subtree(&root, b"m"),
        Err(Error::Key(_))
    ));
    // Diverges below the root ("ax" routes into the "ab*" branch then mismatches).
    assert!(matches!(
        locate_prefix_subtree(&root, b"ax"),
        Err(Error::Key(_))
    ));

    // A prefix that extends *past* a stored key is absent (key is a strict prefix
    // of `p`, so nothing stored has prefix `p`).
    let single = build_tree(&[(b"ab".to_vec(), b"v".to_vec())]).unwrap();
    assert!(matches!(
        locate_prefix_subtree(&single, b"abc"),
        Err(Error::Key(_))
    ));
}

/// Drives the three Stage-1 detach gates on one `(tree, prefix)`: the captured
/// `S` matches the located subtree; dropping `S` equals the `delete_prefix`
/// oracle *and* the independent model; and dropping `S` then re-inserting every
/// prefix-`p` entry reproduces the original tree byte-for-byte.
fn check_detach(entries: &[(Vec<u8>, Vec<u8>)], p: &[u8]) {
    let tree = build_tree(entries);
    let original = root_hash(tree.as_ref());

    // The prefix-p entries (what S must carry) and the remainder.
    let prefix_entries: Vec<_> = entries
        .iter()
        .filter(|(k, _)| k.starts_with(p))
        .cloned()
        .collect();
    assert!(
        !prefix_entries.is_empty(),
        "test prefix must exist in the tree"
    );

    let located = locate_prefix_subtree(tree.as_ref().unwrap(), p).unwrap();
    let (after, captured) = detach_prefix_subtree(tree.clone(), p).unwrap();

    // Capture correctness: detach is CoW, so it captures the *same* Arc the
    // read-only descent locates, with the same prefix-tail length, holding
    // exactly the prefix-p keys.
    assert!(Arc::ptr_eq(&captured.s_root, located.s_root));
    assert_eq!(captured.strip_prefix_bits, located.strip_prefix_bits);
    assert_eq!(count_leaves(&captured.s_root), prefix_entries.len());

    // Drop equality: removing S equals deleting exactly the prefix-p keys, by both
    // the range-machinery oracle and the independent filter-and-rebuild model.
    let via_delete_prefix = delete_prefix(tree.clone(), p).unwrap();
    let via_model = model_drop_prefix(tree.as_ref(), p);
    assert_eq!(
        root_hash(after.as_ref()),
        root_hash(via_delete_prefix.as_ref())
    );
    assert_eq!(root_hash(after.as_ref()), root_hash(via_model.as_ref()));

    // Round-trip: dropping S then re-inserting the prefix-p entries reproduces the
    // original tree (detach removed exactly S, nothing more or less).
    let mut rebuilt = after;
    for (k, v) in prefix_entries {
        rebuilt = Some(insert(rebuilt, k, v).unwrap());
    }
    assert_eq!(root_hash(rebuilt.as_ref()), original);

    // The original tree is untouched (pure CoW).
    assert_eq!(root_hash(tree.as_ref()), original);
}

#[test]
fn mrt_detach_prefix_subtree_round_trip_shapes() {
    // S a deep cluster, with keys outside it.
    check_detach(
        &[
            (b"user:alice".to_vec(), b"1".to_vec()),
            (b"user:alex".to_vec(), b"2".to_vec()),
            (b"user:bob".to_vec(), b"3".to_vec()),
            (b"sys:log".to_vec(), b"4".to_vec()),
            (b"zzz".to_vec(), b"5".to_vec()),
        ],
        b"user:",
    );

    // S ends mid-skip of an internal branch.
    check_detach(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"abd".to_vec(), b"2".to_vec()),
            (b"z".to_vec(), b"3".to_vec()),
        ],
        b"ab",
    );

    // S a non-root leaf.
    check_detach(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"z".to_vec(), b"2".to_vec()),
        ],
        b"abc",
    );

    // S ends at a non-root branch decision (byte-aligned).
    check_detach(
        &[
            (vec![0x61, 0x00], b"1".to_vec()),
            (vec![0x61, 0x80], b"2".to_vec()),
            (b"z".to_vec(), b"3".to_vec()),
        ],
        &[0x61],
    );

    // All-0xFF prefix (exercises the open-upper-bound delete_prefix oracle path).
    check_detach(
        &[
            (vec![0xff, 0x00], b"1".to_vec()),
            (vec![0xff, 0x80], b"2".to_vec()),
            (vec![0x00], b"3".to_vec()),
        ],
        &[0xff],
    );
}

#[test]
fn mrt_detach_prefix_subtree_whole_tree() {
    // Single-key tree: S is the whole tree (a lone leaf), detach empties it.
    let single = build_tree(&[(b"only".to_vec(), b"v".to_vec())]);
    let (after, captured) = detach_prefix_subtree(single.clone(), b"only").unwrap();
    assert!(after.is_none());
    assert!(Arc::ptr_eq(&captured.s_root, single.as_ref().unwrap()));
    assert_eq!(captured.strip_prefix_bits, 8 * 4); // whole prefix, entry_depth 0

    // All keys share prefix "ab": S is the whole-tree root branch.
    let shared = build_tree(&[
        (b"abc".to_vec(), b"1".to_vec()),
        (b"abd".to_vec(), b"2".to_vec()),
    ]);
    let (after, captured) = detach_prefix_subtree(shared.clone(), b"ab").unwrap();
    assert!(after.is_none());
    assert!(Arc::ptr_eq(&captured.s_root, shared.as_ref().unwrap()));
    assert!(matches!(captured.s_root.node(), MrtNode::Branch { .. }));
    assert_eq!(captured.strip_prefix_bits, 8 * 2);
}

#[test]
fn mrt_detach_prefix_subtree_absent_is_err() {
    let root = build_tree(&[
        (b"abc".to_vec(), b"1".to_vec()),
        (b"z".to_vec(), b"2".to_vec()),
    ]);
    assert!(matches!(
        detach_prefix_subtree(root.clone(), b"m"),
        Err(Error::Key(_))
    ));
    // A prefix extending past a stored key is absent.
    assert!(matches!(
        detach_prefix_subtree(root, b"abcd"),
        Err(Error::Key(_))
    ));
    // An empty tree has no prefixes.
    assert!(matches!(
        detach_prefix_subtree(None, b"x"),
        Err(Error::Key(_))
    ));
}

#[test]
fn mrt_detach_prefix_subtree_is_order_independent() {
    // Detach must produce the same post-detach root regardless of insertion order.
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u32..24)
        .map(|i| {
            (
                format!("ns:{i:04x}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .chain((0u32..8).map(|i| {
            (
                format!("other-{i}").into_bytes(),
                format!("o{i}").into_bytes(),
            )
        }))
        .collect();
    let p: &[u8] = b"ns:";

    let canonical = detach_prefix_subtree(build_tree(&entries), p).unwrap();
    let canonical_after = root_hash(canonical.0.as_ref());
    let canonical_s = canonical.1.s_root.hash();

    let mut rng = SmallRng::seed_from_u64(0xDE7AC4);
    for trial in 0..64 {
        let mut shuffled = entries.clone();
        shuffled.shuffle(&mut rng);
        let (after, captured) = detach_prefix_subtree(build_tree(&shuffled), p).unwrap();
        assert_eq!(
            root_hash(after.as_ref()),
            canonical_after,
            "trial {trial} after-root"
        );
        assert_eq!(
            captured.s_root.hash(),
            canonical_s,
            "trial {trial} captured S"
        );
    }
}

// ─── move_prefix stage 2: reskin + splice prefix subtree ─────────────────────

fn reskin_captured_for_test(
    captured: Captured,
    prepend_q_tail_bits: &RouteBits,
) -> Arc<MrtNodeInner> {
    let strip_prefix_bits = captured.strip_prefix_bits;
    match captured.s_root.node() {
        MrtNode::Leaf { skip, value } => MrtNodeInner::leaf(
            reskin_root(skip, strip_prefix_bits, prepend_q_tail_bits).unwrap(),
            value.clone(),
        ),
        MrtNode::Branch { skip, left, right } => MrtNodeInner::branch(
            reskin_root(skip, strip_prefix_bits, prepend_q_tail_bits).unwrap(),
            left.clone(),
            right.clone(),
        ),
        MrtNode::PrunedHash => panic!("test captured root must be materialized"),
    }
}

fn check_detach_splice(entries: &[(Vec<u8>, Vec<u8>)], p: &[u8], q: &[u8]) {
    let tree = build_tree(entries);
    let original = root_hash(tree.as_ref());
    let expected = model_move_prefix(entries, p, q);

    let (after_detach, captured) = detach_prefix_subtree(tree.clone(), p).unwrap();
    let moved = splice_subtree_at(after_detach, q, captured).unwrap();

    assert_eq!(root_hash(Some(&moved)), root_hash(expected.as_ref()));
    assert_eq!(all_entries(Some(&moved)), all_entries(expected.as_ref()));

    let (back_detach, back_captured) = detach_prefix_subtree(Some(moved), q).unwrap();
    let back = splice_subtree_at(back_detach, p, back_captured).unwrap();
    assert_eq!(root_hash(Some(&back)), original);

    // The original tree is untouched (pure CoW across detach + splice).
    assert_eq!(root_hash(tree.as_ref()), original);
}

#[test]
fn mrt_reskin_root_rebuild_property() {
    // Leaf root: re-skinning carries the value and rewrites only the root skip.
    let leaf_entries = [(b"pa".to_vec(), b"leaf".to_vec())];
    let (_empty, captured) = detach_prefix_subtree(build_tree(&leaf_entries), b"pa").unwrap();
    let q_tail = route_bits_of(b"qb");
    let actual = reskin_captured_for_test(captured, &q_tail);
    let expected = build_tree(&[(b"qb".to_vec(), b"leaf".to_vec())]);
    assert_eq!(root_hash(Some(&actual)), root_hash(expected.as_ref()));

    // Random branch/leaf roots: a whole-tree capture re-skinned to q must match a
    // fresh build with every key's prefix changed from p to q.
    let mut rng = SmallRng::seed_from_u64(0x4E57_2A11);
    for trial in 0..128 {
        let prefix_len = rng.gen_range(1..=3);
        let p = random_bytes(&mut rng, prefix_len);
        let mut q = random_bytes(&mut rng, prefix_len);
        while q == p {
            q = random_bytes(&mut rng, prefix_len);
        }

        let count = rng.gen_range(1..=10);
        let mut suffixes = BTreeSet::new();
        while suffixes.len() < count {
            suffixes.insert(random_bytes(&mut rng, 2));
        }

        let entries: Vec<_> = suffixes
            .iter()
            .enumerate()
            .map(|(i, suffix)| {
                let mut key = p.clone();
                key.extend_from_slice(suffix);
                (key, format!("v{trial}-{i}").into_bytes())
            })
            .collect();
        let expected = model_move_prefix(&entries, &p, &q);

        let (_empty, captured) = detach_prefix_subtree(build_tree(&entries), &p).unwrap();
        let q_tail = route_bits_of(&q);
        let actual = reskin_captured_for_test(captured, &q_tail);
        assert_eq!(
            root_hash(Some(&actual)),
            root_hash(expected.as_ref()),
            "trial {trial} p={p:?} q={q:?}"
        );
    }

    assert!(matches!(
        reskin_root(&RouteBits::empty(), 1, &RouteBits::empty()),
        Err(Error::Tree(_))
    ));
}

#[test]
fn mrt_splice_subtree_at_detach_then_splice_shapes() {
    // S a non-root leaf.
    check_detach_splice(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"z".to_vec(), b"2".to_vec()),
        ],
        b"abc",
        b"def",
    );

    // S a deep subtree.
    check_detach_splice(
        &[
            (b"user:alice".to_vec(), b"1".to_vec()),
            (b"user:alex".to_vec(), b"2".to_vec()),
            (b"user:bob".to_vec(), b"3".to_vec()),
            (b"sys:log".to_vec(), b"4".to_vec()),
            (b"zzz".to_vec(), b"5".to_vec()),
        ],
        b"user:",
        b"acct:",
    );

    // S ends mid-skip of an internal branch.
    check_detach_splice(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"abd".to_vec(), b"2".to_vec()),
            (b"z".to_vec(), b"3".to_vec()),
        ],
        b"ab",
        b"xy",
    );

    // S ends at a non-root branch decision (byte-aligned).
    check_detach_splice(
        &[
            (vec![0x61, 0x00], b"1".to_vec()),
            (vec![0x61, 0x80], b"2".to_vec()),
            (b"z".to_vec(), b"3".to_vec()),
        ],
        &[0x61],
        &[0x62],
    );

    // q shares leading bits with p, so the splice descends freshly through the
    // post-detach survivor rather than through the pre-detach source branch.
    check_detach_splice(
        &[
            (b"aa0".to_vec(), b"1".to_vec()),
            (b"aa1".to_vec(), b"2".to_vec()),
            (b"ac0".to_vec(), b"3".to_vec()),
            (b"ad0".to_vec(), b"4".to_vec()),
            (b"zz".to_vec(), b"5".to_vec()),
        ],
        b"aa",
        b"ab",
    );

    // All-0xFF source prefix is just another prefix locus.
    check_detach_splice(
        &[
            (vec![0xff, 0x00], b"1".to_vec()),
            (vec![0xff, 0x80], b"2".to_vec()),
            (vec![0x00], b"3".to_vec()),
        ],
        &[0xff],
        &[0x7f],
    );
}

#[test]
fn mrt_splice_subtree_at_whole_tree_no_connector() {
    // Single-key tree: detach empties the tree and splice returns the re-skinned
    // leaf as the root directly.
    check_detach_splice(&[(b"only".to_vec(), b"v".to_vec())], b"only", b"next");

    // Whole-tree branch: every key shares p, so the empty-destination path must
    // re-skin the branch root directly, without adding a connector.
    check_detach_splice(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"abd".to_vec(), b"2".to_vec()),
        ],
        b"ab",
        b"xy",
    );
}

#[test]
fn mrt_splice_subtree_at_overwrites_destination_but_rejects_prefix_violation() {
    // q is a prefix of existing keys (za, zb): OVERWRITE discards them and splices.
    let tree = build_tree(&[
        (b"aa".to_vec(), b"1".to_vec()),
        (b"ab".to_vec(), b"2".to_vec()),
        (b"za".to_vec(), b"3".to_vec()),
        (b"zb".to_vec(), b"4".to_vec()),
    ]);
    let (after_detach, captured) = detach_prefix_subtree(tree, b"a").unwrap();
    assert!(splice_subtree_at(after_detach, b"z", captured).is_ok());

    // q is an exact existing key: OVERWRITE replaces it.
    let tree = build_tree(&[
        (b"aa".to_vec(), b"1".to_vec()),
        (b"m".to_vec(), b"2".to_vec()),
        (b"zz".to_vec(), b"3".to_vec()),
    ]);
    let (after_detach, captured) = detach_prefix_subtree(tree, b"aa").unwrap();
    assert!(splice_subtree_at(after_detach, b"zz", captured).is_ok());

    // An existing leaf key is a prefix of q: still rejected (the result would
    // violate the prefix-free key invariant — this is not a destination overwrite).
    let tree = build_tree(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"zz".to_vec(), b"2".to_vec()),
    ]);
    let (after_detach, captured) = detach_prefix_subtree(tree, b"zz").unwrap();
    assert!(matches!(
        splice_subtree_at(after_detach, b"ab", captured),
        Err(Error::Key(_))
    ));
}

#[test]
fn mrt_snapshot_arc_isolation_after_mutation() {
    let snapshot = build_tree(&[(b"a".to_vec(), b"va".to_vec())]);
    let snapshot_hash = root_hash(snapshot.as_ref());

    let live = insert(snapshot.clone(), b"b".to_vec(), b"vb".to_vec()).unwrap();

    assert_eq!(root_hash(snapshot.as_ref()), snapshot_hash);
    assert_eq!(get_owned(snapshot.as_ref(), b"a"), Some(b"va".to_vec()));
    assert_eq!(get_owned(snapshot.as_ref(), b"b"), None);
    assert_eq!(get_owned(Some(&live), b"b"), Some(b"vb".to_vec()));
    assert_ne!(live.hash(), snapshot_hash);
}

#[test]
fn mrt_hash_is_deterministic_across_insertion_orders() {
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u32..16)
        .map(|i| {
            (
                format!("key-{i:08x}").into_bytes(),
                format!("value-{i}").into_bytes(),
            )
        })
        .collect();
    let canonical = root_hash(build_tree(&entries).as_ref());

    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    for trial in 0..120 {
        let mut shuffled = entries.clone();
        shuffled.shuffle(&mut rng);
        let h = root_hash(build_tree(&shuffled).as_ref());
        assert_eq!(h, canonical, "trial {trial} produced a different root");
    }
}

#[test]
fn mrt_empty_single_and_long_prefix_edges() {
    let empty: Option<Arc<MrtNodeInner>> = None;
    assert_eq!(root_hash(empty.as_ref()), NULL_HASH);
    assert_eq!(get_owned(empty.as_ref(), b"any"), None);

    let single = insert(None, Vec::new(), b"empty".to_vec()).unwrap();
    assert_eq!(get_owned(Some(&single), b""), Some(b"empty".to_vec()));
    let (after_single_delete, deleted) = delete(Some(single), b"").unwrap();
    assert!(deleted);
    assert!(after_single_delete.is_none());

    // A prefix-free set with varied lengths (no key is a byte-prefix of another).
    let varied_entries = [
        (vec![0x00, 0x01], b"v-0001".to_vec()),
        (vec![0x00, 0x02], b"v-0002".to_vec()),
        (vec![0x01], b"v-01".to_vec()),
        (vec![0x02, 0x03, 0x04, 0x05], b"v-02030405".to_vec()),
    ];
    let root = build_tree(&varied_entries);
    for (key, value) in varied_entries {
        assert_eq!(get_owned(root.as_ref(), &key), Some(value));
    }
}

#[test]
fn mrt_insert_with_trace_records_descent_path() {
    let root = build_tree(&[
        (vec![0x10], b"v10".to_vec()),
        (vec![0x18], b"v18".to_vec()),
        (vec![0x12], b"v12".to_vec()),
    ]);
    let before = root.clone().unwrap();
    let mut visited = Vec::new();

    let after = insert_with_trace(root, vec![0x13], b"v13".to_vec(), &mut |node| {
        visited.push(node.clone())
    })
    .unwrap();

    assert!(visited.len() >= 2);
    assert_eq!(visited[0].hash(), before.hash());
    assert_eq!(get_owned(Some(&after), &[0x13]), Some(b"v13".to_vec()));
}

#[test]
fn mrt_delete_with_trace_records_survivor_sibling_on_collapse() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
    ]);
    let survivor_hash = get_child_hash(root.as_ref().unwrap(), b"b");
    let mut visited = Vec::new();

    let (after, deleted) =
        delete_with_trace(root, b"a", &mut |_prefix, node| visited.push(node.clone())).unwrap();

    assert!(deleted);
    assert_eq!(get_owned(after.as_ref(), b"b"), Some(b"vb".to_vec()));
    assert!(collect_hashes(&visited).contains(&survivor_hash));
}

#[test]
fn mrt_delete_range_with_trace_records_survivor_on_collapse() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
    ]);
    let survivor_hash = get_child_hash(root.as_ref().unwrap(), b"b");
    let mut visited = Vec::new();

    let after = delete_range_with_trace(root, b"a", b"b", &mut |_prefix, node| {
        visited.push(node.clone())
    })
    .unwrap();

    assert_eq!(get_owned(after.as_ref(), b"b"), Some(b"vb".to_vec()));
    assert!(collect_hashes(&visited).contains(&survivor_hash));
}

#[test]
fn mrt_get_with_trace_records_without_mutating() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ]);
    let before = root_hash(root.as_ref());
    let mut visited = Vec::new();

    let got = get_with_trace(root.as_ref(), b"b", &mut |node| visited.push(node.clone())).unwrap();

    assert_eq!(got, Some(b"vb".as_slice()));
    assert!(!visited.is_empty());
    assert_eq!(root_hash(root.as_ref()), before);
}

#[test]
fn mrt_route_bits_pack_and_validate() {
    let empty = RouteBits::from_packed(0, &[]).unwrap();
    assert_eq!(empty.bit_len(), 0);
    // `bit_len` is framed as u32-BE on the wire (§3), so empty packs to 4 bytes.
    assert_eq!(empty.encoded_len(), 4);

    let one = RouteBits::from_packed(1, &[0x80]).unwrap();
    assert!(one.bit_at(0));
    assert!(RouteBits::from_packed(1, &[0x81]).is_err());
    assert!(RouteBits::from_packed(9, &[0x80, 0x80]).is_ok());
    assert!(RouteBits::from_packed(9, &[0x80, 0x81]).is_err());
    assert!(RouteBits::from_packed(MAX_ROUTE_BITS + 1, &[]).is_err());

    let key = vec![0u8; MAX_KEY_LEN];
    let max = RouteBits::from_key_range(&key, 0, route_len(&key));
    assert_eq!(max.bit_len(), MAX_ROUTE_BITS);
    // With the raw 8-bit encoding there is no terminator, so the full-key route
    // and the all-bytes prefix route coincide at 8·len = MAX_ROUTE_BITS.
    assert_eq!(prefix_route_bits(&key).unwrap(), MAX_ROUTE_BITS);
    assert!(matches!(
        max.matches_key_at(&key, 0),
        MatchResult::FullMatch
    ));
}

#[test]
fn mrt_route_bit_at_encodes_raw_bits() {
    // Raw encoding: the route bits are the key's bytes, MSB-first, 8 per byte —
    // no presence bit, no terminator. 'a' = 0x61 = 0b0110_0001.
    let key = b"a";
    let expected_bits = [false, true, true, false, false, false, false, true];

    for (offset, expected) in expected_bits.iter().copied().enumerate() {
        assert_eq!(route_bit_at(key, offset as u16), expected);
    }
    // Position 8 is past the (1-byte) key's end → false.
    assert!(!route_bit_at(key, 8));
}

#[test]
fn mrt_route_bit_at_beyond_key_end_is_false() {
    let key = b"a";

    assert!(!route_bit_at(key, 9));
    assert!(!route_bit_at(key, 10));
    assert!(!route_bit_at(key, 18));
    assert!(!route_bit_at(key, 1024));
}

#[test]
fn mrt_route_encoding_preserves_raw_byte_order() {
    // Among prefix-free keys (here: equal-length, or differing within the shorter)
    // route-bit order equals byte order. Prefix pairs like ([], [0x00]) are
    // intentionally *not* distinguishable by route bits in the raw encoding — that
    // is exactly why the tree requires keys to be prefix-free.
    let pairs: &[(&[u8], &[u8])] = &[
        (&[0x61], &[0x62]),
        (&[0x7f], &[0x80]),
        (&[0x00], &[0x01]),
        (&[0x00, 0x00], &[0x00, 0x01]),
        (&[0x61, 0x61], &[0x61, 0x62]),
        (&[0x61, 0xff], &[0x62, 0x00]),
    ];

    for (a, b) in pairs {
        assert_eq!(a.cmp(b), Ordering::Less);
        assert_eq!(route_cmp(a, b), Ordering::Less);
        assert_eq!(route_cmp(b, a), Ordering::Greater);
    }
}

#[test]
fn mrt_pruned_nodes_error_on_descent() {
    let leaf = MrtNodeInner::leaf_value(vec![0x00], b"v".to_vec());
    let pruned = MrtNodeInner::pruned([0xee; 32], 0);
    let root = MrtNodeInner::branch(RouteBits::empty(), leaf, pruned);

    // [0x80]'s top route bit is set, so every op routes into the pruned right
    // child. (Both stored sides are 1-byte, so the key set stays prefix-free.)
    let get_err = get(Some(&root), &[0x80]).unwrap_err();
    assert!(matches!(get_err, Error::PrunedNode(_)));

    let insert_err = insert(Some(root.clone()), vec![0x80], b"v".to_vec()).unwrap_err();
    assert!(matches!(insert_err, Error::PrunedNode(_)));

    let delete_err = delete(Some(root), &[0x80]).unwrap_err();
    assert!(matches!(delete_err, Error::PrunedNode(_)));
}

#[test]
fn mrt_splices_below_and_above_compressed_range() {
    // Splicing keys below and above a compressed range must keep every key
    // retrievable. The exact branch/skip structure is pinned by the randomized
    // model-equivalence and query-proof tests (root-hash equality); here we check
    // behavior. All keys are 1 byte, hence prefix-free.
    let mut tree = build_tree(&[
        (vec![0x10u8], b"v10".to_vec()),
        (vec![0x18], b"v18".to_vec()),
    ]);
    assert!(matches!(
        tree.as_ref().unwrap().node(),
        MrtNode::Branch { .. }
    ));

    tree = Some(insert(tree, vec![0x12], b"v12".to_vec()).unwrap());
    tree = Some(insert(tree, vec![0x40], b"v40".to_vec()).unwrap());

    for (k, v) in [
        (&[0x10u8][..], b"v10".as_slice()),
        (&[0x12], b"v12"),
        (&[0x18], b"v18"),
        (&[0x40], b"v40"),
    ] {
        assert_eq!(get_owned(tree.as_ref(), k), Some(v.to_vec()));
    }
}

#[test]
fn mrt_shorter_non_prefix_key_splices_above_branch() {
    // A shorter, non-prefix key (0x13 vs 0x12xx) must splice in above the existing
    // branch with all keys still retrievable. Exact skip structure is covered by
    // the differential tests.
    let tree = build_tree(&[
        (vec![0x12u8, 0x34], b"v1234".to_vec()),
        (vec![0x12, 0x35], b"v1235".to_vec()),
    ]);

    let after = Some(insert(tree, vec![0x13], b"v13".to_vec()).unwrap());
    assert!(matches!(
        after.as_ref().unwrap().node(),
        MrtNode::Branch { .. }
    ));
    for (k, v) in [
        (&[0x12u8, 0x34][..], b"v1234".as_slice()),
        (&[0x12, 0x35], b"v1235"),
        (&[0x13], b"v13"),
    ] {
        assert_eq!(get_owned(after.as_ref(), k), Some(v.to_vec()));
    }
}

#[test]
fn mrt_merk_new_has_null_root() {
    let merk = Tree::new();
    let snapshot = merk.checkpoint();

    assert_eq!(merk.root_hash(), NULL_HASH);
    assert_eq!(snapshot.root_hash(), NULL_HASH);
    assert!(snapshot.is_empty());
    assert_eq!(merk.get(b"foo"), None);
}

#[test]
fn mrt_merk_delete_and_delete_range() {
    let merk = Tree::new();
    for i in 0u8..8 {
        merk.put(vec![i], vec![i * 10]).unwrap();
    }

    merk.delete(vec![2]).unwrap();
    merk.delete_range(vec![4], vec![7]).unwrap();

    assert_eq!(merk.get(&[0]), Some(vec![0]));
    assert_eq!(merk.get(&[2]), None);
    assert_eq!(merk.get(&[4]), None);
    assert_eq!(merk.get(&[5]), None);
    assert_eq!(merk.get(&[6]), None);
    assert_eq!(merk.get(&[7]), Some(vec![70]));
}

#[test]
fn mrt_merk_apply_batch_mixed_put_delete_and_range() {
    let merk = Tree::new();
    let setup: Vec<_> = (0u8..10)
        .map(|i| (vec![i], Op::Put(vec![i * 10])))
        .collect();
    merk.apply_sorted_batch_ops(&setup).unwrap();

    let mixed = vec![
        (vec![2], Op::Delete),
        (vec![4], Op::DeleteRange(vec![7])),
        (vec![4], Op::Put(vec![44])),
        (vec![8], Op::Put(vec![88])),
    ];
    merk.apply_sorted_batch_ops(&mixed).unwrap();

    assert_eq!(merk.get(&[2]), None);
    assert_eq!(merk.get(&[3]), Some(vec![30]));
    assert_eq!(merk.get(&[4]), Some(vec![44]));
    assert_eq!(merk.get(&[5]), None);
    assert_eq!(merk.get(&[6]), None);
    assert_eq!(merk.get(&[7]), Some(vec![70]));
    assert_eq!(merk.get(&[8]), Some(vec![88]));
}

#[test]
fn mrt_merk_snapshot_isolated_from_subsequent_mutations() {
    let merk = Tree::new();
    merk.put(b"a".to_vec(), b"va".to_vec()).unwrap();
    let snapshot = merk.checkpoint();
    let snapshot_hash = snapshot.root_hash();

    merk.put(b"b".to_vec(), b"vb".to_vec()).unwrap();
    merk.delete(b"a".to_vec()).unwrap();

    assert_eq!(snapshot.root_hash(), snapshot_hash);
    assert_eq!(snapshot.get(b"a"), Some(b"va".to_vec()));
    assert_eq!(snapshot.get(b"b"), None);
    assert_eq!(merk.get(b"a"), None);
    assert_eq!(merk.get(b"b"), Some(b"vb".to_vec()));
}

#[test]
fn mrt_merk_batch_validation_rejects_invalid_batches_without_mutation() {
    let merk = Tree::new();
    merk.put(vec![1], vec![10]).unwrap();
    let before = merk.root_hash();

    let unsorted = vec![(vec![2], Op::Put(vec![20])), (vec![1], Op::Put(vec![10]))];
    assert!(merk.apply_sorted_batch_ops(&unsorted).is_err());

    let duplicate_points = vec![(vec![1], Op::Put(vec![11])), (vec![1], Op::Delete)];
    assert!(merk.apply_sorted_batch_ops(&duplicate_points).is_err());

    let duplicate_ranges = vec![
        (vec![1], Op::DeleteRange(vec![3])),
        (vec![1], Op::DeleteRange(vec![4])),
    ];
    assert!(merk.apply_sorted_batch_ops(&duplicate_ranges).is_err());

    let invalid_range = vec![(vec![3], Op::DeleteRange(vec![3]))];
    assert!(merk.apply_sorted_batch_ops(&invalid_range).is_err());

    let too_long = vec![0u8; MAX_KEY_LEN + 1];
    assert!(merk
        .apply_sorted_batch_ops(&[(too_long.clone(), Op::Put(vec![1]))])
        .is_err());
    assert!(merk
        .apply_sorted_batch_ops(&[(Vec::new(), Op::DeleteRange(too_long))])
        .is_err());

    assert_eq!(merk.root_hash(), before);
    assert_eq!(merk.get(&[1]), Some(vec![10]));
}

fn build_apply_writes_mrt_fixture() -> Tree {
    let merk = Tree::new();
    for (key, value) in [
        (b"keep".to_vec(), b"keep".to_vec()),
        (b"gone".to_vec(), b"gone".to_vec()),
        (b"pre:old".to_vec(), b"pre-old".to_vec()),
        (b"del:a".to_vec(), b"del-a".to_vec()),
        (b"user:old".to_vec(), b"user-old".to_vec()),
        (vec![0xff, 0x00], b"ff0".to_vec()),
        (vec![0xff, 0x10], b"ff1".to_vec()),
    ] {
        merk.put(key, value).unwrap();
    }
    merk
}

#[test]
fn mrt_apply_writes_ordered_matches_individual_writes() {
    let ops = vec![
        WriteOp::Put {
            key: b"dup:key".to_vec(),
            value: b"a".to_vec(),
        },
        WriteOp::Put {
            key: b"dup:key".to_vec(),
            value: b"b".to_vec(),
        },
        WriteOp::Delete {
            key: b"gone".to_vec(),
        },
        WriteOp::DeleteRange {
            start: b"del:".to_vec(),
            end: b"del;".to_vec(),
        },
        WriteOp::Put {
            key: b"del:z".to_vec(),
            value: b"after-range".to_vec(),
        },
        WriteOp::DeletePrefix {
            prefix: b"pre:".to_vec(),
        },
        WriteOp::Put {
            key: b"pre:new".to_vec(),
            value: b"after-prefix".to_vec(),
        },
        WriteOp::DeletePrefix { prefix: vec![0xff] },
        WriteOp::Put {
            key: b"user:new".to_vec(),
            value: b"user-new".to_vec(),
        },
        WriteOp::MovePrefix {
            from: b"user:".to_vec(),
            to: b"acct:".to_vec(),
        },
    ];

    let batched = build_apply_writes_mrt_fixture();
    let individual = build_apply_writes_mrt_fixture();

    let before_empty = batched.root_hash();
    batched.apply_write_ops(&[]).unwrap();
    assert_eq!(batched.root_hash(), before_empty);

    batched.apply_write_ops(&ops).unwrap();
    for op in &ops {
        individual
            .apply_write_ops(std::slice::from_ref(op))
            .unwrap();
    }

    assert_eq!(batched.root_hash(), individual.root_hash());
    assert_eq!(
        batched.checkpoint().iter().collect::<Vec<_>>(),
        individual.checkpoint().iter().collect::<Vec<_>>()
    );
    assert_eq!(batched.get(b"dup:key"), Some(b"b".to_vec()));
    assert_eq!(batched.get(b"gone"), None);
    assert_eq!(batched.get(b"del:a"), None);
    assert_eq!(batched.get(b"del:z"), Some(b"after-range".to_vec()));
    assert_eq!(batched.get(b"pre:old"), None);
    assert_eq!(batched.get(b"pre:new"), Some(b"after-prefix".to_vec()));
    assert_eq!(batched.get(&[0xff, 0x00]), None);
    assert_eq!(batched.get(&[0xff, 0x10]), None);
    assert_eq!(batched.get(b"user:old"), None);
    assert_eq!(batched.get(b"user:new"), None);
    assert_eq!(batched.get(b"acct:old"), Some(b"user-old".to_vec()));
    assert_eq!(batched.get(b"acct:new"), Some(b"user-new".to_vec()));
    assert_eq!(batched.get(b"keep"), Some(b"keep".to_vec()));
}

#[test]
fn mrt_apply_writes_rolls_back_on_move_prefix_precondition_failure() {
    let merk = Tree::new();
    merk.put(b"user:old".to_vec(), b"user-old".to_vec())
        .unwrap();
    let before_hash = merk.root_hash();
    let before_entries = merk.checkpoint().iter().collect::<Vec<_>>();

    let err = merk
        .apply_write_ops(&[
            WriteOp::Put {
                key: b"acct:new".to_vec(),
                value: b"rolled-back".to_vec(),
            },
            // Equal source/destination prefixes → precondition failure mid-batch
            // (move-onto-occupied no longer fails — it overwrites — so use a still-
            // invalid move to exercise candidate-commit rollback).
            WriteOp::MovePrefix {
                from: b"user:".to_vec(),
                to: b"user:".to_vec(),
            },
            WriteOp::Put {
                key: b"after".to_vec(),
                value: b"after".to_vec(),
            },
        ])
        .unwrap_err();

    assert!(matches!(err, Error::Key(_) | Error::Tree(_)));
    assert_eq!(merk.root_hash(), before_hash);
    assert_eq!(merk.checkpoint().iter().collect::<Vec<_>>(), before_entries);
    assert_eq!(merk.get(b"acct:new"), None);
    assert_eq!(merk.get(b"user:old"), Some(b"user-old".to_vec()));

    merk.apply_write_ops(&[WriteOp::Put {
        key: b"after".to_vec(),
        value: b"after".to_vec(),
    }])
    .unwrap();
    assert_eq!(merk.get(b"after"), Some(b"after".to_vec()));
}

fn apply_writes_node_visits(n: usize, op_count: usize) -> usize {
    let merk = Tree::new();
    for i in 0..n {
        merk.put(format!("{i:08}").into_bytes(), b"v".to_vec())
            .unwrap();
    }

    let ops: Vec<_> = (0..op_count)
        .map(|i| {
            let index = i * (n / op_count);
            WriteOp::Put {
                key: format!("{index:08}").into_bytes(),
                value: b"updated".to_vec(),
            }
        })
        .collect();

    reset_node_visits();
    merk.apply_write_ops(&ops).unwrap();
    node_visits()
}

#[test]
fn mrt_apply_writes_visits_scale_with_path_not_tree_size() {
    let small = apply_writes_node_visits(1000, 8);
    let large = apply_writes_node_visits(4000, 8);

    assert!(
        large <= small * 2,
        "MRT apply_write_ops scaled with tree size, not accessed paths: \
         {} node visits at n=1000 -> {} at n=4000",
        small,
        large
    );
    assert!(
        large < 1000,
        "expected path-bounded apply_write_ops node visits, got {} for n=4000",
        large
    );
}

#[test]
fn mrt_merk_get_overlong_key_returns_none() {
    let merk = Tree::new();
    merk.put(vec![1], vec![10]).unwrap();
    let too_long = vec![0u8; MAX_KEY_LEN + 1];

    assert_eq!(merk.get(&too_long), None);
    assert_eq!(merk.checkpoint().get(&too_long), None);
}

#[test]
fn mrt_snapshot_iterates_forward_and_reverse() {
    let merk = Tree::new();
    let entries = vec![
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
        (b"d".to_vec(), b"vd".to_vec()),
    ];
    for (key, value) in &entries {
        merk.put(key.clone(), value.clone()).unwrap();
    }
    let snapshot = merk.checkpoint();

    assert_eq!(snapshot.iter().collect::<Vec<_>>(), entries);

    let mut reverse = entries.clone();
    reverse.reverse();
    assert_eq!(snapshot.reverse_iter().collect::<Vec<_>>(), reverse);
}

#[test]
fn mrt_snapshot_iter_from_starts_at_present_and_absent_keys() {
    let merk = Tree::new();
    for key in [b"a".as_slice(), b"c".as_slice(), b"e".as_slice()] {
        merk.put(key.to_vec(), [b'v', key[0]].to_vec()).unwrap();
    }
    let snapshot = merk.checkpoint();

    assert_eq!(
        snapshot.iter_from(b"c").collect::<Vec<_>>(),
        vec![
            (b"c".to_vec(), b"vc".to_vec()),
            (b"e".to_vec(), b"ve".to_vec())
        ]
    );
    assert_eq!(
        snapshot.iter_from(b"b").collect::<Vec<_>>(),
        vec![
            (b"c".to_vec(), b"vc".to_vec()),
            (b"e".to_vec(), b"ve".to_vec())
        ]
    );
    assert!(snapshot.iter_from(b"z").next().is_none());
}

#[test]
fn mrt_snapshot_reverse_iter_from_starts_at_present_and_absent_keys() {
    let merk = Tree::new();
    for key in [b"a".as_slice(), b"c".as_slice(), b"e".as_slice()] {
        merk.put(key.to_vec(), [b'v', key[0]].to_vec()).unwrap();
    }
    let snapshot = merk.checkpoint();

    assert_eq!(
        snapshot.reverse_iter_from(b"c").collect::<Vec<_>>(),
        vec![
            (b"c".to_vec(), b"vc".to_vec()),
            (b"a".to_vec(), b"va".to_vec())
        ]
    );
    assert_eq!(
        snapshot.reverse_iter_from(b"d").collect::<Vec<_>>(),
        vec![
            (b"c".to_vec(), b"vc".to_vec()),
            (b"a".to_vec(), b"va".to_vec())
        ]
    );
    assert!(snapshot.reverse_iter_from(b"").next().is_none());
}

#[test]
fn mrt_snapshot_iteration_empty_and_single_element() {
    let empty = Tree::new().checkpoint();
    assert!(empty.iter().next().is_none());
    assert!(empty.iter_from(b"a").next().is_none());
    assert!(empty.reverse_iter().next().is_none());
    assert!(empty.reverse_iter_from(b"a").next().is_none());

    let merk = Tree::new();
    merk.put(b"k".to_vec(), b"v".to_vec()).unwrap();
    let snapshot = merk.checkpoint();
    assert_eq!(
        snapshot.iter().collect::<Vec<_>>(),
        vec![(b"k".to_vec(), b"v".to_vec())]
    );
    assert_eq!(
        snapshot.reverse_iter().collect::<Vec<_>>(),
        vec![(b"k".to_vec(), b"v".to_vec())]
    );
    assert_eq!(
        snapshot.iter_from(b"k").collect::<Vec<_>>(),
        vec![(b"k".to_vec(), b"v".to_vec())]
    );
    assert_eq!(
        snapshot.reverse_iter_from(b"k").collect::<Vec<_>>(),
        vec![(b"k".to_vec(), b"v".to_vec())]
    );
}

#[test]
fn mrt_snapshot_iteration_after_delete_range() {
    let merk = Tree::new();
    for i in 0u8..8 {
        merk.put(vec![i], vec![i + 10]).unwrap();
    }
    merk.delete_range(vec![2], vec![6]).unwrap();
    let snapshot = merk.checkpoint();

    let expected = vec![
        (vec![0], vec![10]),
        (vec![1], vec![11]),
        (vec![6], vec![16]),
        (vec![7], vec![17]),
    ];
    assert_eq!(snapshot.iter().collect::<Vec<_>>(), expected);

    let mut reverse = expected.clone();
    reverse.reverse();
    assert_eq!(snapshot.reverse_iter().collect::<Vec<_>>(), reverse);
    assert_eq!(
        snapshot.iter_from(&[1]).collect::<Vec<_>>(),
        expected[1..].to_vec()
    );
    assert_eq!(
        snapshot.reverse_iter_from(&[6]).collect::<Vec<_>>(),
        vec![
            (vec![6], vec![16]),
            (vec![1], vec![11]),
            (vec![0], vec![10])
        ]
    );
}

#[test]
fn mrt_cursor_seek_lt_excludes_exact_match() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ])
    .unwrap();

    assert_eq!(
        collect_cursor_reverse(cursor_seek_lt(&root, b"b").unwrap()),
        vec![(b"a".to_vec(), b"va".to_vec())]
    );
    assert!(collect_cursor_reverse(cursor_seek_lt(&root, b"a").unwrap()).is_empty());
}

#[test]
fn mrt_cursor_seek_past_end_returns_empty() {
    let root = build_tree(&[
        (b"a".to_vec(), b"va".to_vec()),
        (b"b".to_vec(), b"vb".to_vec()),
        (b"c".to_vec(), b"vc".to_vec()),
    ])
    .unwrap();

    assert!(collect_cursor_forward(cursor_seek_ge(&root, b"z").unwrap()).is_empty());
    assert!(collect_cursor_reverse(cursor_seek_le(&root, &[]).unwrap()).is_empty());
}

#[test]
fn mrt_cursor_seek_ge_below_entire_tree() {
    let keys: &[u8] = &[0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0];
    let entries: Vec<(Vec<u8>, Vec<u8>)> =
        keys.iter().map(|key| (vec![*key], vec![*key])).collect();
    let root = build_tree(&entries).unwrap();

    let got = collect_cursor_forward(cursor_seek_ge(&root, &[0x70]).unwrap());

    assert_eq!(got, entries);
}

#[test]
fn mrt_cursor_seek_le_above_entire_tree() {
    let keys: &[u8] = &[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70];
    let entries: Vec<(Vec<u8>, Vec<u8>)> =
        keys.iter().map(|key| (vec![*key], vec![*key])).collect();
    let root = build_tree(&entries).unwrap();

    let got = collect_cursor_reverse(cursor_seek_le(&root, &[0xf0]).unwrap());
    let mut expected = entries;
    expected.reverse();

    assert_eq!(got, expected);
}

#[test]
fn mrt_cursor_handles_clustered_keys() {
    let entries = [
        (b"abca".to_vec(), b"v-abca".to_vec()),
        (b"abcb".to_vec(), b"v-abcb".to_vec()),
        (b"abcc".to_vec(), b"v-abcc".to_vec()),
    ];
    let root = build_tree(&entries).unwrap();

    assert_eq!(collect_cursor_forward(Cursor::first(&root)), entries);
    assert_eq!(
        collect_cursor_forward(cursor_seek_ge(&root, b"abcb").unwrap()),
        vec![
            (b"abcb".to_vec(), b"v-abcb".to_vec()),
            (b"abcc".to_vec(), b"v-abcc".to_vec())
        ]
    );
}

#[test]
fn mrt_cursor_handles_empty_key() {
    // An empty key can only be the sole key (it is a prefix of every other key).
    let root = build_tree(&[(Vec::new(), b"empty".to_vec())]).unwrap();
    let entries = [(Vec::new(), b"empty".to_vec())];

    assert_eq!(collect_cursor_forward(Cursor::first(&root)), entries);
    assert_eq!(
        collect_cursor_forward(cursor_seek_ge(&root, &[]).unwrap()),
        entries
    );
}

/// A branch whose left child is the real 1-byte key `[0x00]` and whose right child
/// is pruned. Any query whose top route bit is set routes into the pruned side.
fn pruned_right_tree() -> Arc<MrtNodeInner> {
    let leaf = MrtNodeInner::leaf(RouteBits::from_key_range(&[0x00], 1, 8), b"v".to_vec());
    let pruned = MrtNodeInner::pruned([0xee; 32], 0);
    MrtNodeInner::branch(RouteBits::empty(), leaf, pruned)
}

#[test]
fn mrt_cursor_pruned_seek_errors() {
    let root = pruned_right_tree();

    let ge_err = match cursor_seek_ge(&root, &[0x80]) {
        Ok(_) => panic!("expected seek_ge to hit pruned node"),
        Err(err) => err,
    };
    assert!(matches!(ge_err, Error::PrunedNode(_)));

    let le_err = match cursor_seek_le(&root, &[0xff]) {
        Ok(_) => panic!("expected seek_le to hit pruned node"),
        Err(err) => err,
    };
    assert!(matches!(le_err, Error::PrunedNode(_)));
}

#[test]
fn mrt_cursor_next_and_prev_report_pruned_descent() {
    let root = pruned_right_tree();

    let mut forward = Cursor::first(&root);
    let mut visit = |_: &RoutePrefix, _: &Arc<MrtNodeInner>| {};
    let first = forward
        .next(&mut visit)
        .unwrap()
        .expect("left leaf should be yielded");
    assert_eq!(first.0, vec![0x00]);
    let next_err = forward.next(&mut visit).unwrap_err();
    assert!(matches!(next_err, Error::PrunedNode(_)));

    let mut reverse = Cursor::last(&root);
    let prev_err = reverse.prev(&mut visit).unwrap_err();
    assert!(matches!(prev_err, Error::PrunedNode(_)));
}

#[test]
fn mrt_merk_model_equivalence_under_random_batches() {
    for seed in 1..=64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let merk = Tree::new();
        let mut model = Model::new();

        for step in 0..80 {
            let batch = random_batch_with_ranges(&mut rng, &model);
            merk.apply_sorted_batch_ops(&batch).unwrap();
            apply_to_model(&mut model, &batch);
            assert_merk_matches_model(&merk, &model, seed, step);

            for _ in 0..4 {
                let key =
                    choose_existing_key(&mut rng, &model).unwrap_or_else(|| random_key(&mut rng));
                let expected = model.get(&key).map(|value| value.visible_value.clone());
                assert_eq!(
                    merk.get(&key),
                    expected,
                    "seed={seed} step={step}: probe get({key:?})"
                );
                assert_eq!(
                    merk.checkpoint().get(&key),
                    expected,
                    "seed={seed} step={step}: snapshot probe get({key:?})"
                );
            }
        }
    }
}

fn get_child_hash(root: &MrtNodeInner, key: &[u8]) -> Hash {
    match root.node() {
        MrtNode::Branch { left, right, .. } => {
            if get(Some(left), key).unwrap().is_some() {
                left.hash()
            } else {
                right.hash()
            }
        }
        other => panic!("expected branch root, got {:?}", other),
    }
}

fn build_merk(entries: &[(Vec<u8>, Vec<u8>)]) -> Tree {
    let merk = Tree::new();
    let batch: Vec<_> = entries
        .iter()
        .map(|(key, value)| (key.clone(), Op::Put(value.clone())))
        .collect();
    merk.apply_sorted_batch_ops(&batch).unwrap();
    merk
}

fn encode_trace(trace: &Trace) -> Vec<u8> {
    let mut bytes = Vec::new();
    ed::Encode::encode_into(trace, &mut bytes).unwrap();
    assert_eq!(bytes.len(), ed::Encode::encoding_length(trace).unwrap());
    bytes
}

fn normalize_items(items: &[QueryItem]) -> Vec<QueryItem> {
    let mut query = Query::new();
    for item in items.iter().cloned() {
        query.insert_item(item);
    }
    query.into_iter().collect()
}

fn expected_query_entries(
    entries: &[(Vec<u8>, Vec<u8>)],
    items: &[QueryItem],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let normalized = normalize_items(items);
    entries
        .iter()
        .filter(|(key, _)| normalized.iter().any(|item| item.contains(key)))
        .cloned()
        .collect()
}

#[test]
fn mrt_trace_encode_decode_roundtrips_full_tree() {
    let entries: Vec<_> = (0u8..16).map(|i| (vec![i], vec![i, i ^ 0xa5])).collect();
    let merk = build_merk(&entries);
    let snapshot = merk.checkpoint();
    let trace = Trace(snapshot.root.clone());
    let decoded = Trace::decode_exact(&encode_trace(&trace)).unwrap();

    assert_eq!(decoded.root_hash(), snapshot.root_hash());
    assert_eq!(decoded.collect_all().unwrap(), entries);
}

/// Asserts two pruned trees return the same `get` result (same value, or both
/// erroring — e.g. on a pruned-descent key).
fn assert_same_get(a: &TraceVerifier, b: &TraceVerifier, key: &[u8]) {
    match (a.get(key), b.get(key)) {
        (Ok(x), Ok(y)) => assert_eq!(x, y, "get({:?}) value mismatch", key),
        (Err(_), Err(_)) => {}
        (x, y) => panic!("get({:?}) result kind mismatch: {:?} vs {:?}", key, x, y),
    }
}

/// `decode_trace` (flat bytes -> verify tree directly) must be behaviorally
/// identical to the existing `decode_exact` -> `from_trace` path, on a full tree.
#[test]
fn decode_trace_matches_from_trace_full_tree() {
    let entries: Vec<_> = (0u8..40).map(|i| (vec![i], vec![i, i ^ 0x3c])).collect();
    let merk = build_merk(&entries);
    let snapshot = merk.checkpoint();
    let bytes = encode_trace(&Trace(snapshot.root.clone()));

    let via_trace = TraceVerifier::from_trace(&Trace::decode_exact(&bytes).unwrap());
    let direct = TraceVerifier::decode_trace(&bytes).unwrap();

    // Same start-root verification.
    let mut a = via_trace.clone();
    let mut b = direct.clone();
    a.verify_root(snapshot.root_hash()).unwrap();
    b.verify_root(snapshot.root_hash()).unwrap();

    // Identical reads (present + absent) and range.
    for i in 0u8..48 {
        assert_same_get(&via_trace, &direct, &[i]);
    }
    assert_eq!(
        direct.collect_range(&[0], Some(&[48])).unwrap(),
        via_trace.collect_range(&[0], Some(&[48])).unwrap()
    );

    // Identical replay: both reach the same end root.
    let ops = vec![
        BatchOp::Put {
            key: vec![5],
            value: b"changed".to_vec(),
        },
        BatchOp::Delete { key: vec![6] },
        BatchOp::Put {
            key: vec![200],
            value: b"new".to_vec(),
        },
    ];
    let live = build_merk(&entries);
    live.put(vec![5], b"changed".to_vec()).unwrap();
    live.delete(vec![6]).unwrap();
    live.put(vec![200], b"new".to_vec()).unwrap();
    let end_root = live.root_hash();

    a.replay_batch_ops(&ops).unwrap();
    b.replay_batch_ops(&ops).unwrap();
    a.verify_root(end_root).unwrap();
    b.verify_root(end_root).unwrap();
}

/// Same equivalence on a real *pruned* proof trace (hash stubs on untouched
/// subtrees), exercising reads on opened, absent, and pruned-descent keys.
#[test]
fn decode_trace_matches_from_trace_on_pruned_proof() {
    let merk = build_merk(&[
        (b"alpha".to_vec(), b"1".to_vec()),
        (b"alpine".to_vec(), b"2".to_vec()),
        (b"beta".to_vec(), b"3".to_vec()),
        (b"delta".to_vec(), b"5".to_vec()),
        (b"gamma".to_vec(), b"4".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![
        Step::Read(vec![
            ReadOp::Key(b"beta".to_vec()),
            ReadOp::Prefix(b"alp".to_vec()),
        ]),
        Step::Write(vec![BatchOp::Put {
            key: b"beta".to_vec(),
            value: b"33".to_vec(),
        }]),
    ];
    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let bytes = encode_trace(&trace);

    let via_trace = TraceVerifier::from_trace(&Trace::decode_exact(&bytes).unwrap());
    let direct = TraceVerifier::decode_trace(&bytes).unwrap();

    let mut a = via_trace.clone();
    let mut b = direct.clone();
    a.verify_root(start_root).unwrap();
    b.verify_root(start_root).unwrap();

    for key in [&b"beta"[..], b"alpha", b"alpine", b"gamma", b"zzz"] {
        assert_same_get(&via_trace, &direct, key);
    }
    assert_eq!(
        direct.collect_prefix(b"alp").unwrap(),
        via_trace.collect_prefix(b"alp").unwrap()
    );

    // Replay the write step; both reach the trace's end root.
    let ops = vec![BatchOp::Put {
        key: b"beta".to_vec(),
        value: b"33".to_vec(),
    }];
    a.replay_batch_ops(&ops).unwrap();
    b.replay_batch_ops(&ops).unwrap();
    a.verify_root(end_root).unwrap();
    b.verify_root(end_root).unwrap();
}

/// `decode_trace` handles the empty-tree marker and rejects malformed tracees
/// (truncated / trailing / bad tag / empty) without panicking — it parses
/// untrusted guest input.
#[test]
fn decode_trace_handles_empty_and_rejects_malformed() {
    let empty_bytes = encode_trace(&Trace(None));
    let mut empty = TraceVerifier::decode_trace(&empty_bytes).unwrap();
    empty.verify_root(NULL_HASH).unwrap();
    assert!(empty.get(b"anything").unwrap().is_none());

    let merk = build_merk(&[
        (b"k1".to_vec(), b"v".to_vec()),
        (b"k2".to_vec(), b"v2".to_vec()),
    ]);
    let bytes = encode_trace(&Trace(merk.checkpoint().root.clone()));
    TraceVerifier::decode_trace(&bytes).unwrap(); // sanity: well-formed decodes

    // Truncated → error (a length read runs off the end), not a panic.
    assert!(TraceVerifier::decode_trace(&bytes[..bytes.len() - 1]).is_err());
    // Trailing garbage → error.
    let mut trailing = bytes.clone();
    trailing.push(0xff);
    assert!(TraceVerifier::decode_trace(&trailing).is_err());
    // Unknown node tag and empty input → error.
    assert!(TraceVerifier::decode_trace(&[0x7f]).is_err());
    assert!(TraceVerifier::decode_trace(&[]).is_err());
}

/// `depth_below` correctness: for every node, the cached value must equal the
/// deepest leaf's absolute bit-depth in that subtree minus the node's own entry
/// depth (the relative max-subtree-depth). Checked on a skewed (deep chain) tree
/// and a bushy (first-byte-diverging) tree.
#[test]
fn mrt_depth_below_matches_deepest_leaf() {
    // Returns the deepest leaf's absolute bit-depth under `node`, asserting each
    // node's depth_below == deepest − entry_depth on the way up.
    fn check(node: &Arc<MrtNodeInner>, entry_depth: u16) -> u16 {
        let deepest = match node.node() {
            MrtNode::Leaf { skip, .. } => entry_depth + skip.bit_len(),
            MrtNode::Branch { skip, left, right } => {
                let child_depth = entry_depth + skip.bit_len() + 1;
                check(left, child_depth).max(check(right, child_depth))
            }
            MrtNode::PrunedHash => unreachable!("hand-built trees are fully materialized"),
        };
        assert_eq!(
            node.depth_below(),
            deepest - entry_depth,
            "depth_below mismatch at entry_depth {entry_depth}"
        );
        deepest
    }

    // Skewed: long shared prefixes make a deep chain of branches.
    let skewed = build_tree(&[
        (b"aaaaaaaa".to_vec(), b"1".to_vec()),
        (b"aaaaaaab".to_vec(), b"2".to_vec()),
        (b"aaaaaabc".to_vec(), b"3".to_vec()),
        (b"aaaaabcd".to_vec(), b"4".to_vec()),
    ])
    .unwrap();
    check(&skewed, 0);

    // Bushy: first-byte divergence gives a shallower, wider tree.
    let bushy = build_tree(&[
        (b"0".to_vec(), b"a".to_vec()),
        (b"4".to_vec(), b"b".to_vec()),
        (b"8".to_vec(), b"c".to_vec()),
        (b"c".to_vec(), b"d".to_vec()),
        (vec![0xff], b"e".to_vec()),
    ])
    .unwrap();
    check(&bushy, 0);
}

/// Malformed-trace rejection across **every** decode path: an over-deep pruned
/// stub, an over-deep branch roll-up, and a bare pruned root must each be rejected
/// (clean `Err`, no panic) by `Trace::decode_exact`, the verifier's
/// `decode_trace`, and the query-proof `verify`. The honest encoder never
/// emits these; the in-memory (unchecked) host constructors let us build them.
#[test]
fn mrt_malformed_trace_depth_rejected_across_decoders() {
    let over = MAX_ROUTE_BITS + 1;

    // (a) over-deep pruned stub under a materialized branch (so it isn't a bare
    // pruned root — that case is (c)). Built via the `*_unchecked` constructors,
    // which skip the trusted-path `debug_assert`, so we can encode the malformed wire
    // and prove the *decoders* reject it.
    let leaf = MrtNodeInner::leaf(RouteBits::from_key_range(&[0x00], 1, 8), b"v".to_vec());
    let stub = MrtNodeInner::pruned_unchecked([0xab; 32], over);
    let bad_pruned = encode_trace(&Trace(Some(MrtNodeInner::branch_unchecked(
        RouteBits::empty(),
        leaf,
        stub,
    ))));

    // (b) over-deep branch roll-up: a near-max branch skip plus a real child pushes
    // skip + 1 + max(child) past MAX_ROUTE_BITS, while every node alone is in range.
    let big_skip = RouteBits::from_key_range(&vec![0x5au8; 4096], 0, MAX_ROUTE_BITS - 4);
    let l = MrtNodeInner::leaf(RouteBits::from_key_range(&[0xff], 1, 8), b"l".to_vec());
    let r = MrtNodeInner::leaf(RouteBits::from_key_range(&[0x00], 1, 8), b"r".to_vec());
    let bad_branch = encode_trace(&Trace(Some(MrtNodeInner::branch_unchecked(big_skip, l, r))));

    // (c) bare pruned root (depth 0 is in range — the rejection is the bare-root rule,
    // not the depth bound).
    let bad_root = encode_trace(&Trace(Some(MrtNodeInner::pruned([0xcd; 32], 0))));

    for bytes in [&bad_pruned, &bad_branch, &bad_root] {
        assert!(Trace::decode_exact(bytes).is_err(), "decode_exact accepted");
        assert!(
            TraceVerifier::decode_trace(bytes).is_err(),
            "decode_trace accepted"
        );
        let query = vec![QueryItem::Key(vec![0x00])];
        assert!(verify(bytes, query, NULL_HASH).is_err(), "verify accepted");
    }
}

/// Randomized differential: across many random trees (full and pruned), the
/// direct `decode_trace` must produce a tree behaviorally identical to the
/// `decode_exact` -> `from_trace` path — same start root, same reads. Exercises
/// varied skip-bit lengths and structures a few hand-built cases would miss.
#[test]
fn decode_trace_matches_from_trace_randomized() {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF);
    for _ in 0..300 {
        let n = rng.gen_range(0..40);
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .map(|_| {
                // Fixed-length keys are prefix-free, as the tree now requires.
                let key = (0..4).map(|_| rng.gen::<u8>()).collect();
                let vlen = rng.gen_range(0..40);
                let value = (0..vlen).map(|_| rng.gen::<u8>()).collect();
                (key, value)
            })
            .collect();
        entries.sort();
        entries.dedup_by(|a, b| a.0 == b.0); // apply_sorted_batch_ops needs sorted, unique keys

        let merk = build_merk(&entries);
        let snapshot = merk.checkpoint();

        // Half full-tree tracees, half pruned proofs (reads over random keys).
        let trace = if entries.is_empty() || rng.gen_bool(0.5) {
            Trace(snapshot.root.clone())
        } else {
            let reads = (0..rng.gen_range(1..=4))
                .map(|_| ReadOp::Key(entries[rng.gen_range(0..entries.len())].0.clone()))
                .collect();
            create_trace(&snapshot, &[Step::Read(reads)]).unwrap()
        };

        let bytes = encode_trace(&trace);
        let via = TraceVerifier::from_trace(&Trace::decode_exact(&bytes).unwrap());
        let direct = TraceVerifier::decode_trace(&bytes).unwrap();

        // Identical start-root verification (pruning preserves the root hash).
        let mut a = via.clone();
        let mut b = direct.clone();
        a.verify_root(snapshot.root_hash()).unwrap();
        b.verify_root(snapshot.root_hash()).unwrap();

        // Identical reads for every key (opened keys agree on value; pruned-path
        // keys agree on error).
        for (key, _) in &entries {
            assert_same_get(&via, &direct, key);
        }
    }
}

#[test]
fn mrt_trace_replay_matches_live_apply() {
    let entries: Vec<_> = (0u8..8).map(|i| (vec![i], vec![i])).collect();
    let merk = build_merk(&entries);
    let trace = Trace(merk.checkpoint().root.clone());
    let batch = vec![
        (vec![1], Op::Put(b"updated".to_vec())),
        (vec![3], Op::Delete),
        (vec![20], Op::Put(b"new".to_vec())),
    ];

    let live = build_merk(&entries);
    live.apply_sorted_batch_ops(&batch).unwrap();
    let replayed = super::trace::replay(trace, &batch).unwrap();

    assert_eq!(replayed.root_hash(), live.root_hash());
    assert_eq!(
        replayed.collect_all().unwrap(),
        live.checkpoint().iter().collect::<Vec<_>>()
    );
}

#[test]
fn mrt_partial_builder_contains_installed_node() {
    let root = MrtNodeInner::leaf_value(b"k".to_vec(), b"v".to_vec());
    let mut builder = super::tracer::MrtPartialBuilder::new();
    let ident = super::tracer::arc_ident(&root);
    assert!(!builder.contains(ident));
    super::tracer::install_arc(&mut builder, &root);
    assert!(builder.contains(ident));
}

#[test]
fn mrt_trace_pruned_descent_is_fallible() {
    let leaf = MrtNodeInner::leaf(RouteBits::from_key_range(&[0x00], 1, 8), b"v".to_vec());
    let pruned = MrtNodeInner::pruned([0xee; 32], 0);
    let trace = Trace(Some(MrtNodeInner::branch(RouteBits::empty(), leaf, pruned)));
    // [0x80] routes into the pruned right subtree.
    let err = trace.get(&[0x80]).unwrap_err();
    assert!(matches!(err, Error::PrunedNode(_)), "got {:?}", err);
}

#[test]
fn mrt_query_proof_verifies_present_and_absent_keys() {
    let entries: Vec<_> = (0u8..16).map(|i| (vec![i], vec![i + 10])).collect();
    let merk = build_merk(&entries);

    let present = vec![QueryItem::Key(vec![7])];
    let bytes = merk.prove(present.clone()).unwrap();
    let result = verify(&bytes, present, merk.root_hash()).unwrap();
    assert_eq!(result, vec![(vec![7], vec![17])]);

    let absent = vec![QueryItem::Key(vec![100])];
    let bytes = merk.prove(absent.clone()).unwrap();
    let result = verify(&bytes, absent, merk.root_hash()).unwrap();
    assert!(result.is_empty());
}

#[test]
fn mrt_query_proof_verifies_ranges_and_empty_ranges() {
    let entries: Vec<_> = (0u8..16).map(|i| (vec![i], vec![i, i])).collect();
    let merk = build_merk(&entries);

    let range = vec![QueryItem::Range(vec![5]..vec![11])];
    let bytes = merk.checkpoint().prove(range.clone()).unwrap();
    let result = verify(&bytes, range.clone(), merk.root_hash()).unwrap();
    assert_eq!(result, expected_query_entries(&entries, &range));

    let empty = vec![QueryItem::Range(vec![90]..vec![100])];
    let bytes = merk.prove(empty.clone()).unwrap();
    let result = verify(&bytes, empty, merk.root_hash()).unwrap();
    assert!(result.is_empty());

    let empty_prefix_style = vec![QueryItem::Range(b"zz".to_vec()..b"z{".to_vec())];
    let bytes = merk.prove(empty_prefix_style.clone()).unwrap();
    let result = verify(&bytes, empty_prefix_style, merk.root_hash()).unwrap();
    assert!(result.is_empty());
}

#[test]
fn mrt_query_proof_verifies_empty_query_and_empty_tree() {
    let entries: Vec<_> = (0u8..4).map(|i| (vec![i], vec![i])).collect();
    let merk = build_merk(&entries);
    let empty_query: Vec<QueryItem> = Vec::new();
    let bytes = merk.prove(empty_query.clone()).unwrap();
    let decoded = Trace::decode_exact(&bytes).unwrap();
    decoded.verify_root(merk.root_hash()).unwrap();
    let result = verify(&bytes, empty_query, merk.root_hash()).unwrap();
    assert!(result.is_empty());

    let empty = Tree::new();
    let query = vec![QueryItem::Key(b"missing".to_vec())];
    let bytes = empty.prove(query.clone()).unwrap();
    let result = verify(&bytes, query, empty.root_hash()).unwrap();
    assert!(result.is_empty());
}

#[test]
fn mrt_query_proof_verifies_inclusive_single_key_range() {
    // Prefix-free keys: an inclusive range that pins one key returns exactly it.
    let entries = vec![
        (vec![0x10], b"a".to_vec()),
        (vec![0x11], b"d".to_vec()),
        (vec![0x20], b"e".to_vec()),
    ];
    let merk = build_merk(&entries);
    let query = vec![QueryItem::RangeInclusive(std::ops::RangeInclusive::new(
        vec![0x10],
        vec![0x10],
    ))];

    let bytes = merk.prove(query.clone()).unwrap();
    let result = verify(&bytes, query, merk.root_hash()).unwrap();
    assert_eq!(result, vec![(vec![0x10], b"a".to_vec())]);
}

#[test]
fn mrt_query_proof_rejects_wrong_root_and_incomplete_trace() {
    let entries: Vec<_> = (0u8..16).map(|i| (vec![i], vec![i])).collect();
    let merk = build_merk(&entries);
    let query = vec![QueryItem::Key(vec![7])];
    let bytes = merk.prove(query.clone()).unwrap();
    let mut wrong = merk.root_hash();
    wrong[0] ^= 0xff;
    let err = verify(&bytes, query.clone(), wrong).unwrap_err();
    assert!(matches!(err, Error::HashMismatch(_, _)), "got {:?}", err);

    // A fully-pruned root is now rejected at *decode* (a bare pruned root carries
    // no parent edge to authenticate its depth and supports no query) — before the
    // old query-descent `PrunedNode` path is even reached.
    let root_only = Trace(Some(MrtNodeInner::pruned(merk.root_hash(), 0)));
    let err = verify(&encode_trace(&root_only), query, merk.root_hash()).unwrap_err();
    assert!(matches!(err, Error::Ed(_)), "got {:?}", err);
}

#[test]
fn mrt_query_proof_normalizes_overlapping_query_items() {
    let entries: Vec<_> = (0u8..16).map(|i| (vec![i], vec![i + 1])).collect();
    let merk = build_merk(&entries);
    let query = vec![
        QueryItem::Range(vec![6]..vec![10]),
        QueryItem::Key(vec![3]),
        QueryItem::Range(vec![8]..vec![12]),
        QueryItem::Key(vec![3]),
    ];
    let bytes = merk.prove(query.clone()).unwrap();
    let result = verify(&bytes, query.clone(), merk.root_hash()).unwrap();
    assert_eq!(result, expected_query_entries(&entries, &query));
}

#[test]
fn mrt_query_proof_randomized_roundtrip_matches_model() {
    let mut rng = SmallRng::seed_from_u64(0xb970_5a4e);
    for round in 0..32 {
        let mut model = BTreeMap::new();
        while model.len() < 24 {
            let key = random_key(&mut rng);
            let value = vec![round as u8, model.len() as u8];
            model.insert(key, value);
        }
        let entries: Vec<_> = model
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let merk = build_merk(&entries);

        for _ in 0..4 {
            let query = match rng.gen_range(0..=2) {
                0 => vec![QueryItem::Key(random_key(&mut rng))],
                1 => {
                    let mut lo = random_key(&mut rng);
                    let mut hi = random_key(&mut rng);
                    if lo > hi {
                        std::mem::swap(&mut lo, &mut hi);
                    }
                    vec![QueryItem::Range(lo..hi)]
                }
                _ => {
                    let mut lo = random_key(&mut rng);
                    let mut hi = random_key(&mut rng);
                    if lo > hi {
                        std::mem::swap(&mut lo, &mut hi);
                    }
                    vec![QueryItem::RangeInclusive(std::ops::RangeInclusive::new(
                        lo, hi,
                    ))]
                }
            };
            let bytes = merk.prove(query.clone()).unwrap();
            let result = verify(&bytes, query.clone(), merk.root_hash()).unwrap();
            assert_eq!(
                result,
                expected_query_entries(&entries, &query),
                "round {round} query {query:?}"
            );
        }
    }
}

#[test]
fn mrt_tracer_single_write_step_verifies_and_matches_live_root() {
    let merk = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Write(vec![BatchOp::Put {
        key: b"c".to_vec(),
        value: b"updated".to_vec(),
    }])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let live = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
    ]);
    live.put(b"c".to_vec(), b"updated".to_vec()).unwrap();

    assert_eq!(end_root, live.root_hash());
    replay_trace(&trace, start_root, &steps, end_root).unwrap();
}

/// Build an n-key tree and return how many `MrtNodeInner::node()` dereferences
/// a single-key write trace performs — a path-bounded vs whole-tree proxy.
fn tracer_node_visits(n: usize) -> usize {
    let merk = Tree::new();
    for i in 0..n {
        // Fixed-width keys so the target exists in trees of any size.
        merk.put(format!("{i:08}").into_bytes(), b"v".to_vec())
            .unwrap();
    }
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Write(vec![BatchOp::Put {
        key: b"00000000".to_vec(),
        value: b"v2".to_vec(),
    }])];

    reset_node_visits();
    create_trace(&snapshot, &steps).unwrap();
    node_visits()
}

/// Guards against re-introducing the O(n^2) MRT pre-state index: per-change
/// trace generation must walk accessed paths, not the whole tree. The removed
/// `MrtPreStateIndex` build called `node()` once per tree node (O(n)); the
/// on-demand snapshot recorders call it only along paths (~log n).
#[test]
fn mrt_tracer_visits_scale_with_path_not_tree_size() {
    let small = tracer_node_visits(1000);
    let large = tracer_node_visits(4000);

    // 4x the data must not ~4x the work; a whole-tree walk (the old index)
    // would, path-bounded recording grows only ~log n.
    assert!(
        large <= small * 2,
        "MRT trace generation scaled with tree size, not accessed path: \
         {} node visits at n=1000 -> {} at n=4000 (O(n^2) regression?)",
        small,
        large
    );
    // ...and the absolute count must stay far below the tree size.
    assert!(
        large < 1000,
        "expected ~path-bounded node visits, got {} for n=4000 (O(n)?)",
        large
    );
}

#[test]
fn mrt_tracer_single_read_absent_range_and_prefix_verify() {
    let merk = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
        (b"e".to_vec(), b"5".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Read(vec![
        ReadOp::Key(b"c".to_vec()),
        ReadOp::Key(b"z".to_vec()),
        ReadOp::Range {
            start: b"m".to_vec(),
            end: b"n".to_vec(),
        },
        ReadOp::Prefix(b"pre_".to_vec()),
    ])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0][0].results, vec![(b"c".to_vec(), b"3".to_vec())]);
    assert!(reads[0][1].results.is_empty());
    assert!(reads[0][2].results.is_empty());
    assert!(reads[0][3].results.is_empty());
}

#[test]
fn mrt_tracer_interleaved_read_write_read_sees_current_state() {
    let merk = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
        (b"e".to_vec(), b"5".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![
        Step::Read(vec![ReadOp::Key(b"c".to_vec())]),
        Step::Write(vec![
            BatchOp::Put {
                key: b"c".to_vec(),
                value: b"updated".to_vec(),
            },
            BatchOp::Put {
                key: b"d".to_vec(),
                value: b"4".to_vec(),
            },
        ]),
        Step::Read(vec![ReadOp::Range {
            start: b"a".to_vec(),
            end: b"f".to_vec(),
        }]),
    ];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert_eq!(reads[0][0].results, vec![(b"c".to_vec(), b"3".to_vec())]);
    assert_eq!(
        reads[1][0].results,
        vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"c".to_vec(), b"updated".to_vec()),
            (b"d".to_vec(), b"4".to_vec()),
            (b"e".to_vec(), b"5".to_vec()),
        ]
    );
}

#[test]
fn mrt_tracer_delete_and_delete_range_steps_verify() {
    let merk = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"2".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
        (b"d".to_vec(), b"4".to_vec()),
        (b"e".to_vec(), b"5".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![
        Step::Write(vec![
            BatchOp::Delete { key: b"b".to_vec() },
            BatchOp::DeleteRange {
                start: b"c".to_vec(),
                end: b"e".to_vec(),
            },
        ]),
        Step::Read(vec![ReadOp::Range {
            start: b"a".to_vec(),
            end: b"z".to_vec(),
        }]),
    ];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert_eq!(
        reads[0][0].results,
        vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"e".to_vec(), b"5".to_vec())
        ]
    );
}

#[test]
fn mrt_tracer_empty_tree_transcript_verifies() {
    let merk = Tree::new();
    let snapshot = merk.checkpoint();
    let steps = vec![
        Step::Read(vec![ReadOp::Key(b"missing".to_vec())]),
        Step::Write(vec![BatchOp::Put {
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        }]),
        Step::Read(vec![ReadOp::Prefix(b"k".to_vec())]),
    ];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert!(reads[0][0].results.is_empty());
    assert_eq!(reads[1][0].results, vec![(b"k".to_vec(), b"v".to_vec())]);
}

#[test]
fn mrt_tracer_empty_transcript_non_empty_snapshot_verifies() {
    let merk = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps: Vec<Step> = Vec::new();

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert_eq!(start_root, merk.root_hash());
    assert_eq!(end_root, merk.root_hash());
    assert!(reads.is_empty());
}

#[test]
fn mrt_tracer_tampered_root_read_and_write_fail() {
    let merk = build_merk(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
    ]);
    let snapshot = merk.checkpoint();

    // Read transcript: a corrupted start root must fail authentication.
    let steps = vec![Step::Read(vec![ReadOp::Key(b"c".to_vec())])];
    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let mut wrong_start = start_root;
    wrong_start[0] ^= 0xff;
    assert!(matches!(
        replay_trace(&trace, wrong_start, &steps, end_root),
        Err(Error::HashMismatch(_, _))
    ));

    // Write transcript: a corrupted end root must fail the post-replay check.
    let steps = vec![Step::Write(vec![BatchOp::Put {
        key: b"c".to_vec(),
        value: b"updated".to_vec(),
    }])];
    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let mut wrong_end = end_root;
    wrong_end[0] ^= 0xff;
    assert!(replay_trace(&trace, start_root, &steps, wrong_end).is_err());
}

/// Counts the revealed (non-stub) nodes in a sparse proof tree — every opened
/// `Leaf`/`Branch`, excluding `PrunedHash` stubs. The A2 tightening gates pin the
/// exact reveal-set size: fuzz catches *under*-reveals (pruned-stub verify
/// failures), but an *over*-reveal still verifies, so only a pinned count/stub
/// fixture catches it.
fn count_revealed(trace: &Trace) -> usize {
    fn walk(arc: &Arc<MrtNodeInner>) -> usize {
        match arc.node() {
            MrtNode::Leaf { .. } => 1,
            MrtNode::PrunedHash => 0,
            MrtNode::Branch { left, right, .. } => 1 + walk(left) + walk(right),
        }
    }
    trace.0.as_ref().map_or(0, walk)
}

/// Descends the sparse proof along `key`'s route and reports whether it lands on a
/// `PrunedHash` stub — i.e. `key`'s leaf was *not* revealed. The tightening gates
/// use this to assert an over-revealed boundary/sibling/successor leaf is a stub.
fn proof_leaf_is_stub(trace: &Trace, key: &[u8]) -> bool {
    let Some(root) = trace.0.as_ref() else {
        return false;
    };
    let mut cur = root;
    let mut depth = 0u16;
    loop {
        match cur.node() {
            MrtNode::PrunedHash => return true,
            MrtNode::Leaf { .. } => return false,
            MrtNode::Branch { skip, left, right } => {
                if matches!(
                    skip.matches_key_at(key, depth),
                    MatchResult::Mismatch { .. }
                ) {
                    return false;
                }
                let branch_depth = depth + skip.bit_len();
                let side = route_bit_at(key, branch_depth);
                depth = branch_depth + 1;
                cur = if !side { left } else { right };
            }
        }
    }
}

/// Range tightening (A2 fall-out): a range whose lower bound skips the first leaf
/// must leave that leaf a *stub* — the cursor never routes into it, so the tracer
/// (recording via the cursor's visit hook) never reveals it. Under the old
/// leftmost-key recording the boundary leaf was revealed `Full`.
#[test]
fn range_boundary_leaf_pruned() {
    let merk = build_merk(&[
        (vec![0x10], b"a".to_vec()),
        (vec![0x20], b"b".to_vec()),
        (vec![0x30], b"c".to_vec()),
        (vec![0x40], b"d".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Read(vec![ReadOp::Range {
        start: vec![0x25],
        end: vec![0x45],
    }])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert_eq!(
        reads[0][0].results,
        vec![(vec![0x30], b"c".to_vec()), (vec![0x40], b"d".to_vec())]
    );
    // The boundary leaf below the lower bound is never routed into → stub.
    assert!(proof_leaf_is_stub(&trace, &[0x10]));
    assert_eq!(count_revealed(&trace), 6);
}

/// Point-get tightening (A2 fall-out): a present-key get reveals exactly its
/// root→leaf path; every sibling subtree stays a stub.
#[test]
fn point_get_present_reveals_only_its_path() {
    let merk = build_merk(&[
        (vec![0x10], b"a".to_vec()),
        (vec![0x20], b"b".to_vec()),
        (vec![0x30], b"c".to_vec()),
        (vec![0x40], b"d".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Read(vec![ReadOp::Key(vec![0x30])])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert_eq!(reads[0][0].results, vec![(vec![0x30], b"c".to_vec())]);
    assert!(!proof_leaf_is_stub(&trace, &[0x30]));
    for sibling in [&[0x10][..], &[0x20], &[0x40]] {
        assert!(
            proof_leaf_is_stub(&trace, sibling),
            "sibling {:?} should be a stub",
            sibling
        );
    }
    assert_eq!(count_revealed(&trace), 4);
}

/// Point-get tightening (A2 fall-out): an *absent*-key get reveals exactly the
/// root→mismatch path and never the successor side — guarding against a
/// `seek_ge`-style get that would route onto (and reveal) the successor.
#[test]
fn point_get_absent_reveals_only_mismatch_path() {
    let merk = build_merk(&[
        (vec![0x10], b"a".to_vec()),
        (vec![0x20], b"b".to_vec()),
        (vec![0x30], b"c".to_vec()),
        (vec![0x40], b"d".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    // 0x15 is absent; it routes to the 0x10 leaf and mismatches there. Its
    // successor is 0x20 — a seek_ge get would reveal it; a key-route get must not.
    let steps = vec![Step::Read(vec![ReadOp::Key(vec![0x15])])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    let reads = replay_trace(&trace, start_root, &steps, end_root).unwrap();

    assert!(reads[0][0].results.is_empty());
    // Successor side and the far subtree stay stubs.
    assert!(proof_leaf_is_stub(&trace, &[0x20]));
    assert!(proof_leaf_is_stub(&trace, &[0x40]));
    assert_eq!(count_revealed(&trace), 3);
}

/// Delete tightening (A): deleting `0x10` collapses the root onto the `{0x20,0x30}`
/// branch (the survivor). The verifier reads only the survivor *node's* skip to
/// re-skin it and relinks its children as-is — it never descends into `0x20`/`0x30`,
/// so those leaves must be stubs. The old `leftmost_key` recording revealed the
/// survivor's whole leftmost spine.
#[test]
fn delete_survivor_spine_pruned() {
    let merk = build_merk(&[
        (vec![0x10], b"a".to_vec()),
        (vec![0x20], b"b".to_vec()),
        (vec![0x30], b"c".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Write(vec![BatchOp::Delete { key: vec![0x10] }])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    replay_trace(&trace, start_root, &steps, end_root).unwrap();

    // The collapse survivor's descendants are never routed into → stubs.
    assert!(proof_leaf_is_stub(&trace, &[0x20]));
    assert!(proof_leaf_is_stub(&trace, &[0x30]));
    // Revealed: root + survivor branch + the deleted `0x10` leaf.
    assert_eq!(count_revealed(&trace), 3);
}

/// Delete tightening (A): a DeleteRange that wholly covers the `{0x20,0x30}` branch
/// — bounds *outside* it on both sides, so it is reached via the `(None,None)`
/// full-delete fast path. The verifier drops it there **without inspecting it**, so
/// that subtree root must be a pruned stub. The old recording ran `record(cur)`
/// before the fast path and revealed it (and, via `leftmost_key`, its spine).
///
/// `[0x18, 0x40)`: `0x18` routes left of the `{0x20,0x30}` branch (so its lo is
/// None) and `0x40` is past the whole tree (so its hi is None) — the branch is
/// fully in range with neither bound landing inside it. `0x10` (< `0x18`) is kept
/// and becomes the collapse survivor.
#[test]
fn deleterange_full_subtree_stays_stub() {
    let merk = build_merk(&[
        (vec![0x10], b"a".to_vec()),
        (vec![0x20], b"b".to_vec()),
        (vec![0x30], b"c".to_vec()),
    ]);
    let snapshot = merk.checkpoint();
    let steps = vec![Step::Write(vec![BatchOp::DeleteRange {
        start: vec![0x18],
        end: vec![0x40],
    }])];

    let trace = create_trace(&snapshot, &steps).unwrap();
    let start_root = snapshot.root_hash();
    let end_root = root_after_writes(&snapshot, &steps);
    replay_trace(&trace, start_root, &steps, end_root).unwrap();

    // The wholly-deleted subtree root (on the 0x20/0x30 path) is a stub.
    assert!(proof_leaf_is_stub(&trace, &[0x20]));
    assert!(proof_leaf_is_stub(&trace, &[0x30]));
    // Revealed: root + the collapse survivor `0x10`.
    assert_eq!(count_revealed(&trace), 2);
}

/// Read-bound validation must match the verifier and not depend on tree
/// occupancy: `create_trace` rejects an over-long key/range/prefix bound up
/// front, rather than building a trace the verifier (which validates) would later
/// reject. Covers the early-return and empty-tree paths where the cursor's /
/// `get_descent`'s own validation is otherwise skipped.
#[test]
fn mrt_tracer_reads_reject_overlong_bounds() {
    let merk = build_merk(&[(b"a".to_vec(), b"1".to_vec())]);
    let snapshot = merk.checkpoint();
    let empty = Tree::new().checkpoint();
    let long = vec![0u8; MAX_KEY_LEN + 1];

    // Over-long `end` is validated even though `start >= end` short-circuits the scan.
    assert!(create_trace(
        &snapshot,
        &[Step::Read(vec![ReadOp::Range {
            start: b"a".to_vec(),
            end: long.clone(),
        }])],
    )
    .is_err());

    // Over-long prefix on an *empty* tree (the path that skips the cursor's check).
    assert!(create_trace(&empty, &[Step::Read(vec![ReadOp::Prefix(long.clone())])]).is_err());

    // Over-long point-get key is rejected on both empty and non-empty trees — the
    // empty tree is the path that would otherwise skip `get_descent`'s validation.
    assert!(create_trace(&empty, &[Step::Read(vec![ReadOp::Key(long.clone())])]).is_err());
    assert!(create_trace(&snapshot, &[Step::Read(vec![ReadOp::Key(long)])]).is_err());
}
