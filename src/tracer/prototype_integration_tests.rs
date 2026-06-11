//! Prototype integration tests: end-to-end trace generation + verification across
//! the full tracer pipeline. These drive the `create_trace` (witness) +
//! `replay_trace` (verify) test-support shims — not public API; production runs the
//! handle (`TraceRecorder`/`TraceReplayer` via `apply`) — which exercise the same
//! prove/verify logic.

use crate::avl::in_memory::{Checkpoint, InMemoryMerk};
use crate::avl::tracer::SparseMerkNode;
use crate::error::Error;
use crate::hash::{Hasher, NULL_HASH};
use crate::ops::Op;
use crate::tracer::test_support::avl::{create_trace, replay_trace, root_after_writes};
use crate::tracer::test_support::Step;
use crate::tracer::{BatchOp, ReadOp, SMALL_VALUE_INLINE_THRESHOLD};

fn build_tree(entries: &[(&[u8], &[u8])]) -> Checkpoint {
    let merk = InMemoryMerk::new();
    for &(k, v) in entries {
        merk.put(k, v).unwrap();
    }
    merk.checkpoint()
}

fn large_value(byte: u8) -> Vec<u8> {
    vec![byte; SMALL_VALUE_INLINE_THRESHOLD + 1]
}

// ---------------------------------------------------------------------------
// Read-after-write: a value written then read back is readable through the trace
// ---------------------------------------------------------------------------

#[test]
fn read_after_write_keeps_full_value_for_point_read() {
    let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
    let big_value = large_value(0xCC);
    let steps = vec![
        Step::Write(vec![BatchOp::Put {
            key: b"c".to_vec(),
            value: big_value.clone(),
        }]),
        Step::Read(vec![ReadOp::Key(b"c".to_vec())]),
    ];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    let verified = replay_trace(&trace, start, &steps, end).unwrap();

    assert_eq!(verified[0][0].results, vec![(b"c".to_vec(), big_value)]);
}

#[test]
fn read_after_write_keeps_full_value_for_range_read() {
    let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
    let big_value = large_value(0xDD);
    let steps = vec![
        Step::Write(vec![BatchOp::Put {
            key: b"c".to_vec(),
            value: big_value.clone(),
        }]),
        Step::Read(vec![ReadOp::Range {
            start: b"b".to_vec(),
            end: b"d".to_vec(),
        }]),
    ];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    let verified = replay_trace(&trace, start, &steps, end).unwrap();

    assert_eq!(verified[0][0].results, vec![(b"c".to_vec(), big_value)]);
}

#[test]
fn read_after_write_keeps_full_value_for_prefix_read() {
    let root = build_tree(&[(b"pre_a", b"1"), (b"pre_c", b"3"), (b"xyz", b"5")]);
    let big_value = large_value(0xEE);
    let steps = vec![
        Step::Write(vec![BatchOp::Put {
            key: b"pre_c".to_vec(),
            value: big_value.clone(),
        }]),
        Step::Read(vec![ReadOp::Prefix(b"pre_".to_vec())]),
    ];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    let verified = replay_trace(&trace, start, &steps, end).unwrap();

    let pre_c = verified[0][0]
        .results
        .iter()
        .find(|(k, _)| k == b"pre_c")
        .expect("prefix read should include the written key");
    assert_eq!(pre_c.1, big_value);
}

// ---------------------------------------------------------------------------
// Absent reads open their descent path as Full nodes (no hash-only opened nodes)
// ---------------------------------------------------------------------------

#[test]
fn absent_point_read_keeps_opened_path_values_full() {
    let big = large_value(0xAA);
    let root = build_tree(&[(b"a", &big), (b"c", &big), (b"e", &big)]);
    let steps = vec![Step::Read(vec![ReadOp::Key(b"b".to_vec())])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();

    fn count_full_and_hash_only(trace: &SparseMerkNode) -> (usize, usize) {
        match trace {
            SparseMerkNode::Full { left, right, .. } => {
                let (left_full, left_hash_only) = count_full_and_hash_only(left);
                let (right_full, right_hash_only) = count_full_and_hash_only(right);
                (1 + left_full + right_full, left_hash_only + right_hash_only)
            }
            SparseMerkNode::FullOmitted { left, right, .. }
            | SparseMerkNode::FullStorageHash { left, right, .. } => {
                let (left_full, left_hash_only) = count_full_and_hash_only(left);
                let (right_full, right_hash_only) = count_full_and_hash_only(right);
                (left_full + right_full, 1 + left_hash_only + right_hash_only)
            }
            _ => (0, 0),
        }
    }
    let (full, hash_only) = count_full_and_hash_only(&trace);
    assert!(
        full > 0,
        "absent read should open its descent path as Full nodes"
    );
    assert_eq!(
        hash_only, 0,
        "generated absent-read trace should not contain hash-only opened nodes"
    );

    let verified = replay_trace(&trace, start, &steps, start).unwrap();
    assert!(verified[0][0].results.is_empty());
}

#[test]
fn absent_read_in_empty_range_succeeds() {
    let root = build_tree(&[(b"a", b"1"), (b"z", b"9")]);
    let steps = vec![Step::Read(vec![ReadOp::Range {
        start: b"m".to_vec(),
        end: b"n".to_vec(),
    }])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let verified = replay_trace(&trace, start, &steps, start).unwrap();
    assert!(verified[0][0].results.is_empty());
}

#[test]
fn absent_prefix_read_succeeds() {
    let root = build_tree(&[(b"alpha", b"1"), (b"beta", b"2"), (b"gamma", b"3")]);
    let steps = vec![Step::Read(vec![ReadOp::Prefix(b"delta".to_vec())])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let verified = replay_trace(&trace, start, &steps, start).unwrap();
    assert!(verified[0][0].results.is_empty());
}

// ---------------------------------------------------------------------------
// Range/prefix reads return only full-value result nodes
// ---------------------------------------------------------------------------

#[test]
fn range_read_returns_only_full_value_result_nodes() {
    let big = large_value(0xBB);
    let root = build_tree(&[
        (b"a", &big),
        (b"c", b"small_c"),
        (b"e", &big),
        (b"g", b"small_g"),
        (b"i", &big),
    ]);
    let steps = vec![Step::Read(vec![ReadOp::Range {
        start: b"c".to_vec(),
        end: b"h".to_vec(),
    }])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let verified = replay_trace(&trace, start, &steps, start).unwrap();

    assert_eq!(verified[0][0].results.len(), 3);
    assert_eq!(
        verified[0][0].results[0],
        (b"c".to_vec(), b"small_c".to_vec())
    );
    assert_eq!(verified[0][0].results[1], (b"e".to_vec(), big.clone()));
    assert_eq!(
        verified[0][0].results[2],
        (b"g".to_vec(), b"small_g".to_vec())
    );
    for (key, value) in &verified[0][0].results {
        assert!(
            !value.is_empty(),
            "range result for key {:?} must have full value",
            key
        );
    }
}

#[test]
fn prefix_read_returns_only_full_value_result_nodes() {
    let big = large_value(0xCC);
    let root = build_tree(&[
        (b"ns_a", &big),
        (b"ns_b", b"small_b"),
        (b"ns_c", &big),
        (b"other", b"not_in_prefix"),
    ]);
    let steps = vec![Step::Read(vec![ReadOp::Prefix(b"ns_".to_vec())])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let verified = replay_trace(&trace, start, &steps, start).unwrap();

    assert_eq!(verified[0][0].results.len(), 3);
    let keys: Vec<&[u8]> = verified[0][0]
        .results
        .iter()
        .map(|(k, _)| k.as_slice())
        .collect();
    assert!(keys.contains(&b"ns_a".as_slice()));
    assert!(keys.contains(&b"ns_b".as_slice()));
    assert!(keys.contains(&b"ns_c".as_slice()));
    for (key, value) in &verified[0][0].results {
        assert!(
            !value.is_empty(),
            "prefix result for key {:?} must have full value",
            key
        );
    }
}

#[test]
fn verifier_rejects_omitted_node_in_range_results() {
    use crate::hash::{kv_hash, node_hash};

    let kv_a = kv_hash::<Hasher>(b"a", b"va").unwrap();
    let kv_c = kv_hash::<Hasher>(b"c", b"vc").unwrap();
    let kv_e = kv_hash::<Hasher>(b"e", b"ve").unwrap();

    let hash_a = node_hash::<Hasher>(&kv_a, &NULL_HASH, &NULL_HASH);
    let hash_e = node_hash::<Hasher>(&kv_e, &NULL_HASH, &NULL_HASH);
    let hash_c = node_hash::<Hasher>(&kv_c, &hash_a, &hash_e);

    // A range that includes a `FullOmitted` (value-elided) node must be rejected.
    let trace = SparseMerkNode::FullOmitted {
        key: b"c".to_vec(),
        kv_hash: kv_c,
        left: Box::new(SparseMerkNode::Full {
            key: b"a".to_vec(),
            value: b"va".to_vec(),
            left: Box::new(SparseMerkNode::Empty),
            right: Box::new(SparseMerkNode::Empty),
        }),
        right: Box::new(SparseMerkNode::Full {
            key: b"e".to_vec(),
            value: b"ve".to_vec(),
            left: Box::new(SparseMerkNode::Empty),
            right: Box::new(SparseMerkNode::Empty),
        }),
    };
    let root_hash = trace.hash();
    assert_eq!(root_hash, hash_c);

    let steps = vec![Step::Read(vec![ReadOp::Range {
        start: b"a".to_vec(),
        end: b"f".to_vec(),
    }])];

    let result = replay_trace(&trace, root_hash, &steps, root_hash);
    assert!(
        matches!(result, Err(Error::ValueOmitted(_))),
        "verifier must reject range that includes FullOmitted node, got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Delete-range traces replay split/join and match the real post-delete root
// ---------------------------------------------------------------------------

#[test]
fn delete_range_trace_matches_post_delete_root() {
    let root = build_tree(&[
        (b"a", b"1"),
        (b"c", b"3"),
        (b"e", b"5"),
        (b"g", b"7"),
        (b"i", b"9"),
        (b"k", b"11"),
    ]);
    let start_hash = root.root_hash();

    let live = InMemoryMerk::new();
    for &(k, v) in &[
        (b"a".as_slice(), b"1".as_slice()),
        (b"c", b"3"),
        (b"e", b"5"),
        (b"g", b"7"),
        (b"i", b"9"),
        (b"k", b"11"),
    ] {
        live.put(k, v).unwrap();
    }
    live.delete_range(b"c", b"i").unwrap();
    let expected_end_hash = live.root_hash();

    let steps = vec![Step::Write(vec![BatchOp::DeleteRange {
        start: b"c".to_vec(),
        end: b"i".to_vec(),
    }])];

    let trace = create_trace(&root, &steps).unwrap();
    assert_eq!(trace.hash(), start_hash);
    let end = root_after_writes(&root, &steps);
    assert_eq!(end, expected_end_hash);

    replay_trace(&trace, start_hash, &steps, expected_end_hash).unwrap();
}

#[test]
fn delete_range_then_read_verifies_correctly() {
    let root = build_tree(&[
        (b"a", b"1"),
        (b"c", b"3"),
        (b"e", b"5"),
        (b"g", b"7"),
        (b"i", b"9"),
    ]);
    let steps = vec![
        Step::Write(vec![BatchOp::DeleteRange {
            start: b"c".to_vec(),
            end: b"h".to_vec(),
        }]),
        Step::Read(vec![ReadOp::Range {
            start: b"a".to_vec(),
            end: b"z".to_vec(),
        }]),
    ];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    let verified = replay_trace(&trace, start, &steps, end).unwrap();

    let keys: Vec<&[u8]> = verified[0][0]
        .results
        .iter()
        .map(|(k, _)| k.as_slice())
        .collect();
    assert!(keys.contains(&b"a".as_slice()));
    assert!(keys.contains(&b"i".as_slice()));
    assert!(!keys.contains(&b"c".as_slice()));
    assert!(!keys.contains(&b"e".as_slice()));
    assert!(!keys.contains(&b"g".as_slice()));
}

#[test]
fn delete_range_large_tree_matches_split_join_root() {
    let entries: Vec<(&[u8], &[u8])> = (0u8..32)
        .map(|i| {
            let key: &'static [u8] = Box::leak(vec![i].into_boxed_slice());
            let val: &'static [u8] = Box::leak(format!("v{i}").into_bytes().into_boxed_slice());
            (key, val)
        })
        .collect();
    let root = build_tree(&entries);

    let live = InMemoryMerk::new();
    for &(k, v) in &entries {
        live.put(k, v).unwrap();
    }
    live.delete_range([8u8], [24u8]).unwrap();
    let expected_end = live.root_hash();

    let steps = vec![Step::Write(vec![BatchOp::DeleteRange {
        start: vec![8],
        end: vec![24],
    }])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    assert_eq!(end, expected_end);
    replay_trace(&trace, start, &steps, expected_end).unwrap();
}

#[test]
fn delete_range_combined_with_point_ops_verifies() {
    let root = build_tree(&[
        (b"a", b"1"),
        (b"c", b"3"),
        (b"e", b"5"),
        (b"g", b"7"),
        (b"i", b"9"),
        (b"k", b"11"),
    ]);

    let live = InMemoryMerk::new();
    for &(k, v) in &[
        (b"a".as_slice(), b"1".as_slice()),
        (b"c", b"3"),
        (b"e", b"5"),
        (b"g", b"7"),
        (b"i", b"9"),
        (b"k", b"11"),
    ] {
        live.put(k, v).unwrap();
    }
    live.apply_sorted_batch_ops(&[
        (b"a".to_vec(), Op::Put(b"updated_a".to_vec())),
        (b"c".to_vec(), Op::DeleteRange(b"g".to_vec())),
        (b"k".to_vec(), Op::Put(b"updated_k".to_vec())),
    ])
    .unwrap();
    let expected_end = live.root_hash();

    let steps = vec![Step::Write(vec![
        BatchOp::Put {
            key: b"a".to_vec(),
            value: b"updated_a".to_vec(),
        },
        BatchOp::DeleteRange {
            start: b"c".to_vec(),
            end: b"g".to_vec(),
        },
        BatchOp::Put {
            key: b"k".to_vec(),
            value: b"updated_k".to_vec(),
        },
    ])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    assert_eq!(end, expected_end);
    replay_trace(&trace, start, &steps, expected_end).unwrap();
}

#[test]
fn delete_range_entire_tree_produces_empty_root() {
    let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
    let steps = vec![Step::Write(vec![BatchOp::DeleteRange {
        start: b"a".to_vec(),
        end: b"f".to_vec(),
    }])];

    let trace = create_trace(&root, &steps).unwrap();
    let start = trace.hash();
    let end = root_after_writes(&root, &steps);
    assert_eq!(end, NULL_HASH);
    replay_trace(&trace, start, &steps, NULL_HASH).unwrap();
}
