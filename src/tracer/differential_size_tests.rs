//! Differential tracer checks and trace-size accounting.
//!
//! These tests compare traced transcripts against a live `InMemoryMerk` oracle
//! and keep explicit size buckets for full-value, read-heavy, and large-value
//! trace shapes. Generated AVL traces keep opened nodes full and use hash
//! pruning only for untouched subtrees. Size is measured on the **encoded
//! trace** — the only artifact production ships (the steps and roots are
//! caller-supplied, not bundled).

use ed::Encode;

use crate::avl::in_memory::InMemoryMerk;
use crate::avl::node::Node;
use crate::hash::Hash;
use crate::ops::Op;

use super::test_support::avl::{create_trace, replay_trace, root_after_writes};
use super::test_support::Step;
use super::{
    prefix_successor, BatchOp, ProvenRead, ReadOp, ReadResults, SMALL_VALUE_INLINE_THRESHOLD,
};
use crate::avl::tracer::{SparseMerkNode, VerifiedReadResults};

#[derive(Clone)]
enum TranscriptStep {
    Read(Vec<ReadOp>),
    Write(Vec<BatchOp>),
}

struct DifferentialRun {
    trace: SparseMerkNode,
    verified_reads: VerifiedReadResults,
    end_root: Hash,
}

#[derive(Debug, Default)]
struct TraceCounts {
    full: usize,
    omitted: usize,
    pruned: usize,
}

fn build_store(entries: &[(Vec<u8>, Vec<u8>)]) -> InMemoryMerk {
    let merk = InMemoryMerk::new();
    for (key, value) in entries {
        merk.put(key.clone(), value.clone()).unwrap();
    }
    merk
}

fn large_value(byte: u8) -> Vec<u8> {
    vec![byte; SMALL_VALUE_INLINE_THRESHOLD + 96]
}

fn seq_entries(count: u8) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..count)
        .map(|i| (vec![i], format!("value-{i}").into_bytes()))
        .collect()
}

fn large_entries(count: u8) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..count)
        .map(|i| (vec![i], large_value(i.wrapping_add(1))))
        .collect()
}

fn prefixed_entries() -> Vec<(Vec<u8>, Vec<u8>)> {
    vec![
        (b"aa".to_vec(), b"v-aa".to_vec()),
        (b"ab".to_vec(), b"v-ab".to_vec()),
        (b"ac".to_vec(), b"v-ac".to_vec()),
        (b"ba".to_vec(), b"v-ba".to_vec()),
        (b"bb".to_vec(), b"v-bb".to_vec()),
        (b"zz".to_vec(), b"v-zz".to_vec()),
    ]
}

fn to_input_step(step: &TranscriptStep) -> Step {
    match step {
        TranscriptStep::Read(reads) => Step::Read(reads.clone()),
        TranscriptStep::Write(ops) => Step::Write(ops.clone()),
    }
}

fn run_differential(entries: &[(Vec<u8>, Vec<u8>)], steps: &[TranscriptStep]) -> DifferentialRun {
    let live = build_store(entries);
    let start_root = live.root_hash();
    let start_snapshot = live.checkpoint();
    let input_steps: Vec<Step> = steps.iter().map(to_input_step).collect();

    // Evolve the live oracle to derive expected reads (reads see current state)
    // and the authentic end root.
    let mut expected_reads = Vec::new();
    for step in steps {
        match step {
            TranscriptStep::Read(reads) => expected_reads.push(live_read_step(&live, reads)),
            // Apply each write individually, in issue order — AVL is
            // insertion-order sensitive, so the live oracle must evolve the same
            // way the traced path does (one write at a time).
            TranscriptStep::Write(ops) => {
                for op in ops {
                    live.apply_sorted_batch_ops_owned(vec![op.to_batch_entry()])
                        .unwrap();
                }
            }
        }
    }
    let end_root = live.root_hash();

    let trace = create_trace(&start_snapshot, &input_steps).unwrap();
    assert_eq!(trace.hash(), start_root);
    assert_eq!(
        root_after_writes(&start_snapshot, &input_steps),
        end_root,
        "independent end-root oracle must match the live store"
    );

    let verified_reads = replay_trace(&trace, start_root, &input_steps, end_root).unwrap();
    assert_eq!(verified_reads, expected_reads);

    DifferentialRun {
        trace,
        verified_reads,
        end_root,
    }
}

fn batch_entries(ops: &[BatchOp]) -> Vec<(Vec<u8>, Op)> {
    ops.iter().map(BatchOp::to_batch_entry).collect()
}

fn live_read_step(live: &InMemoryMerk, reads: &[ReadOp]) -> ReadResults {
    let snapshot = live.checkpoint();
    reads
        .iter()
        .map(|op| ProvenRead {
            op: op.clone(),
            results: live_read(snapshot.root(), op),
        })
        .collect()
}

fn live_read(root: Option<&Node>, op: &ReadOp) -> Vec<(Vec<u8>, Vec<u8>)> {
    match op {
        ReadOp::Key(key) => root
            .and_then(|root| root.get(key).map(|value| vec![(key.clone(), value)]))
            .unwrap_or_default(),
        ReadOp::Range { start, end } => live_range(root, start, Some(end)),
        ReadOp::Prefix(prefix) => live_range(root, prefix, prefix_successor(prefix).as_deref())
            .into_iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .collect(),
    }
}

fn live_range(root: Option<&Node>, start: &[u8], end: Option<&[u8]>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let Some(root) = root else {
        return Vec::new();
    };

    root.iter_from(start)
        .take_while(|(key, _)| end.is_none_or(|end| key.as_slice() < end))
        .collect()
}

fn put(key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) -> BatchOp {
    BatchOp::Put {
        key: key.into(),
        value: value.into(),
    }
}

fn delete(key: impl Into<Vec<u8>>) -> BatchOp {
    BatchOp::Delete { key: key.into() }
}

fn delete_range(start: impl Into<Vec<u8>>, end: impl Into<Vec<u8>>) -> BatchOp {
    BatchOp::DeleteRange {
        start: start.into(),
        end: end.into(),
    }
}

/// Encoded size of the witness trace — the only artifact production ships.
fn trace_size_bytes(trace: &SparseMerkNode) -> usize {
    Encode::encoding_length(trace).unwrap()
}

fn count_trace(trace: &SparseMerkNode) -> TraceCounts {
    let mut counts = TraceCounts::default();
    count_trace_inner(trace, &mut counts);
    counts
}

fn count_trace_inner(trace: &SparseMerkNode, counts: &mut TraceCounts) {
    match trace {
        SparseMerkNode::Empty => {}
        SparseMerkNode::Pruned { .. } => counts.pruned += 1,
        SparseMerkNode::Full { left, right, .. } => {
            counts.full += 1;
            count_trace_inner(left, counts);
            count_trace_inner(right, counts);
        }
        SparseMerkNode::FullStorageHash { left, right, .. }
        | SparseMerkNode::FullOmitted { left, right, .. } => {
            counts.omitted += 1;
            count_trace_inner(left, counts);
            count_trace_inner(right, counts);
        }
    }
}

fn root_hash_after(entries: &[(Vec<u8>, Vec<u8>)], ops: &[BatchOp]) -> Hash {
    let live = build_store(entries);
    live.apply_sorted_batch_ops(&batch_entries(ops)).unwrap();
    live.root_hash()
}

#[test]
fn differential_point_reads_match_live_store() {
    let entries = prefixed_entries();
    let steps = vec![TranscriptStep::Read(vec![
        ReadOp::Key(b"ab".to_vec()),
        ReadOp::Key(b"missing".to_vec()),
    ])];

    let run = run_differential(&entries, &steps);

    assert_eq!(
        run.verified_reads[0][0].results,
        vec![(b"ab".to_vec(), b"v-ab".to_vec())]
    );
    assert!(run.verified_reads[0][1].results.is_empty());
}

#[test]
fn differential_range_and_prefix_reads_match_live_store() {
    let entries = prefixed_entries();
    let steps = vec![TranscriptStep::Read(vec![
        ReadOp::Range {
            start: b"ab".to_vec(),
            end: b"bb".to_vec(),
        },
        ReadOp::Prefix(b"a".to_vec()),
    ])];

    let run = run_differential(&entries, &steps);

    assert_eq!(
        run.verified_reads[0][0].results,
        vec![
            (b"ab".to_vec(), b"v-ab".to_vec()),
            (b"ac".to_vec(), b"v-ac".to_vec()),
            (b"ba".to_vec(), b"v-ba".to_vec()),
        ]
    );
    assert_eq!(
        run.verified_reads[0][1].results,
        vec![
            (b"aa".to_vec(), b"v-aa".to_vec()),
            (b"ab".to_vec(), b"v-ab".to_vec()),
            (b"ac".to_vec(), b"v-ac".to_vec()),
        ]
    );
}

#[test]
fn differential_writes_and_mixed_read_write_transcripts_match_live_store() {
    let entries = seq_entries(8);
    let steps = vec![
        TranscriptStep::Write(vec![
            put(vec![2], b"two-updated".to_vec()),
            delete(vec![4]),
            put(vec![20], b"twenty".to_vec()),
        ]),
        TranscriptStep::Read(vec![
            ReadOp::Key(vec![2]),
            ReadOp::Key(vec![4]),
            ReadOp::Range {
                start: vec![0],
                end: vec![21],
            },
        ]),
        TranscriptStep::Write(vec![put(vec![6], b"six-updated".to_vec())]),
        TranscriptStep::Read(vec![ReadOp::Key(vec![6])]),
    ];

    let run = run_differential(&entries, &steps);

    assert_eq!(
        run.verified_reads[0][0].results,
        vec![(vec![2], b"two-updated".to_vec())]
    );
    assert!(run.verified_reads[0][1].results.is_empty());
    assert_eq!(
        run.verified_reads[1][0].results,
        vec![(vec![6], b"six-updated".to_vec())]
    );
}

#[test]
fn differential_delete_range_split_join_transcripts_match_live_store() {
    let entries = seq_entries(32);
    let delete_ops = vec![delete_range(vec![8], vec![24])];
    let expected_end = root_hash_after(&entries, &delete_ops);
    let steps = vec![
        TranscriptStep::Write(delete_ops),
        TranscriptStep::Read(vec![
            ReadOp::Range {
                start: vec![0],
                end: vec![32],
            },
            ReadOp::Key(vec![12]),
            ReadOp::Key(vec![28]),
        ]),
    ];

    let run = run_differential(&entries, &steps);

    assert_eq!(run.end_root, expected_end);
    assert!(run.verified_reads[0][0]
        .results
        .iter()
        .all(|(key, _)| key.as_slice() < &[8] || key.as_slice() >= &[24]));
    assert!(run.verified_reads[0][1].results.is_empty());
    assert_eq!(
        run.verified_reads[0][2].results,
        vec![(vec![28], b"value-28".to_vec())]
    );
}

#[test]
fn trace_size_buckets_track_read_heavy_and_large_value_omission_traces() {
    let entries = large_entries(16);
    let read_heavy_steps = vec![TranscriptStep::Read(vec![
        ReadOp::Range {
            start: vec![0],
            end: vec![16],
        },
        ReadOp::Key(vec![7]),
        ReadOp::Key(vec![99]),
    ])];
    let large_value_steps = vec![TranscriptStep::Read(vec![ReadOp::Key(vec![99])])];

    let read_heavy = run_differential(&entries, &read_heavy_steps);
    let large_value = run_differential(&entries, &large_value_steps);

    // Structural pruning: a read-heavy trace keeps yielded values full and is
    // larger than an absent-key read that only opens its descent path.
    assert!(
        trace_size_bytes(&read_heavy.trace) > trace_size_bytes(&large_value.trace),
        "read-heavy trace must keep yielded large values full while absent large-value reads prune untouched subtrees"
    );

    let read_heavy_counts = count_trace(&read_heavy.trace);
    assert!(
        read_heavy_counts.full >= entries.len(),
        "read-heavy range trace should keep yielded values full, got {:?}",
        read_heavy_counts
    );
    assert_eq!(
        read_heavy_counts.pruned, 0,
        "read-heavy full-range trace should not prune yielded subtrees"
    );

    let large_value_counts = count_trace(&large_value.trace);
    assert_eq!(
        large_value_counts.omitted, 0,
        "generated large-value absent-read trace should not use hash-only opened nodes, got {:?}",
        large_value_counts
    );
    assert!(
        large_value_counts.full > 0,
        "large-value absent-read trace should open its descent path as Full nodes, got {:?}",
        large_value_counts
    );
    assert!(
        large_value_counts.pruned > 0,
        "large-value absent-read trace should prune untouched side subtrees, got {:?}",
        large_value_counts
    );
}
