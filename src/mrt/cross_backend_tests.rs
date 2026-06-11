use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};

use super::tree::{MrtNode, MrtNodeInner, MAX_KEY_LEN};
use super::{verify_query as verify_mrt_query, Checkpoint, Trace, Tree};
use crate::avl::node::Node;
use crate::avl::tracer::{AvlTrace, VerifiedReadResults};
use crate::hash::{Hash, NULL_HASH};
use crate::ops::Op;
use crate::proofs::query::{verify_query, Query, QueryItem};
use crate::tracer::test_support::Step;
use crate::tracer::test_support::{avl as avl_ts, mrt as mrt_ts};
use crate::tracer::{BatchOp, ProvenRead, ReadOp, WriteOp};
use crate::{avl::Tree as InMemoryMerk, Error, GetResult, UnsupportedFeature};

type Model = BTreeMap<Vec<u8>, Vec<u8>>;
type KeySet = Vec<Vec<u8>>;
type Entries = Vec<(Vec<u8>, Vec<u8>)>;

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

/// Lower a transcript's point/range `BatchOp`s into the `WriteOp` vocabulary the
/// p2-facing durable path (`Tree::apply_write_ops`) consumes.
fn write_ops(ops: &[BatchOp]) -> Vec<WriteOp> {
    ops.iter()
        .map(|op| match op {
            BatchOp::Put { key, value } => WriteOp::Put {
                key: key.clone(),
                value: value.clone(),
            },
            BatchOp::Delete { key } => WriteOp::Delete { key: key.clone() },
            BatchOp::DeleteRange { start, end } => WriteOp::DeleteRange {
                start: start.clone(),
                end: end.clone(),
            },
        })
        .collect()
}

fn build_avl(entries: &[(Vec<u8>, Vec<u8>)]) -> InMemoryMerk {
    let merk = InMemoryMerk::new();
    if !entries.is_empty() {
        let batch: Vec<_> = entries
            .iter()
            .map(|(key, value)| (key.clone(), Op::Put(value.clone())))
            .collect();
        merk.apply_sorted_batch_ops(&batch).unwrap();
    }
    merk
}

fn build_mrt(entries: &[(Vec<u8>, Vec<u8>)]) -> Tree {
    let merk = Tree::new();
    if !entries.is_empty() {
        let batch: Vec<_> = entries
            .iter()
            .map(|(key, value)| (key.clone(), Op::Put(value.clone())))
            .collect();
        merk.apply_sorted_batch_ops(&batch).unwrap();
    }
    merk
}

/// The `create_trace` (witness) + `replay_trace` (verify, `TraceVerifier`)
/// test-support shims round-trip a mixed read/write/move transcript. They are not
/// the public API — production drives the handle (`TraceRecorder`/`TraceReplayer`
/// via `apply`) — but exercise the same prove/verify logic the prototype's prover
/// and fast-forward verifier run.
#[test]
fn create_trace_then_replay_roundtrips_mixed_transcript() {
    let mrt = build_mrt(&[
        (b"aa".to_vec(), b"1".to_vec()),
        (b"ab".to_vec(), b"2".to_vec()),
        (b"zz".to_vec(), b"3".to_vec()),
    ]);
    let snapshot = mrt.checkpoint();
    let steps = vec![
        Step::Read(vec![ReadOp::Key(b"aa".to_vec())]),
        Step::Write(vec![BatchOp::Put {
            key: b"ac".to_vec(),
            value: b"9".to_vec(),
        }]),
        Step::MovePrefix {
            from: b"z".to_vec(),
            to: b"y".to_vec(),
        },
    ];
    let trace = mrt_ts::create_trace(&snapshot, &steps).unwrap();
    let start = snapshot.root_hash();
    let end = mrt_ts::root_after_writes(&snapshot, &steps);
    mrt_ts::replay_trace(&trace, start, &steps, end).unwrap();
}

fn model_from_entries(entries: &[(Vec<u8>, Vec<u8>)]) -> Model {
    entries.iter().cloned().collect()
}

fn avl_entries(snapshot: Option<&Node>) -> Vec<(Vec<u8>, Vec<u8>)> {
    snapshot.map_or_else(Vec::new, |root| root.iter().collect())
}

fn mrt_entries(snapshot: &Checkpoint) -> Vec<(Vec<u8>, Vec<u8>)> {
    snapshot.iter().collect()
}

fn assert_live_states_equal(avl: &InMemoryMerk, mrt: &Tree) {
    let avl_snapshot = avl.checkpoint();
    let mrt_snapshot = mrt.checkpoint();
    assert_eq!(avl_entries(avl_snapshot.root()), mrt_entries(&mrt_snapshot));

    for (key, value) in mrt_snapshot.iter() {
        assert_eq!(avl.get(&key), Some(value.clone()), "AVL get({key:?})");
        assert_eq!(mrt.get(&key), Some(value), "MRT get({key:?})");
    }
}

fn apply_writes_to_live(avl: &InMemoryMerk, mrt: &Tree, steps: &[Step]) {
    for step in steps {
        if let Step::Write(ops) = step {
            // Drive the p2-facing durable path (`apply_write_ops`) on BOTH backends,
            // so the differential — and the callers' `*_end == live root_hash()`
            // assertions, where `*_end` comes from the traced path — actually
            // exercise it (the prior version used the sorted host batch, leaving
            // `apply_write_ops` unguarded). `apply_write_ops` applies in issue order,
            // no sort, matching the trace. These cross-backend transcripts are
            // move-free (AVL rejects `MovePrefix`), so AVL never hits `Unsupported`.
            let writes = write_ops(ops);
            avl.apply_write_ops(&writes).unwrap();
            mrt.apply_write_ops(&writes).unwrap();
        }
    }
}

fn apply_writes_to_model(model: &mut Model, ops: &[BatchOp]) {
    for op in ops {
        match op {
            BatchOp::Put { key, value } => {
                model.insert(key.clone(), value.clone());
            }
            BatchOp::Delete { key } => {
                model.remove(key);
            }
            BatchOp::DeleteRange { start, end } => {
                let keys: Vec<_> = model
                    .range(start.clone()..end.clone())
                    .map(|(key, _)| key.clone())
                    .collect();
                for key in keys {
                    model.remove(&key);
                }
            }
        }
    }
}

fn model_read(model: &Model, op: &ReadOp) -> Vec<(Vec<u8>, Vec<u8>)> {
    match op {
        ReadOp::Key(key) => model
            .get(key)
            .map(|value| vec![(key.clone(), value.clone())])
            .unwrap_or_default(),
        ReadOp::Range { start, end } => model
            .range(start.clone()..end.clone())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        ReadOp::Prefix(prefix) => model
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    }
}

/// Relocate every prefix-`from` key to prefix `to` in the model (values kept).
/// The naive byte-identity oracle for `move_prefix`: callers that build a result
/// model and `build_mrt` it get the tree a "delete each, re-insert at `to`" pass
/// would produce. Assumes the move's preconditions hold (the transcript generator
/// only emits valid moves); a precondition-aware variant lives in
/// [`model_move_prefix`].
fn apply_move_prefix_to_model(model: &mut Model, from: &[u8], to: &[u8]) {
    // Collect + remove the moved (`from`-prefixed) entries first, so this is correct
    // even when `from` lies under `to`.
    let moved: Vec<_> = model
        .keys()
        .filter(|key| key.starts_with(from))
        .cloned()
        .collect();
    let moved_kv: Vec<(Vec<u8>, Vec<u8>)> = moved
        .into_iter()
        .map(|key| {
            let value = model.remove(&key).expect("key just collected");
            (key, value)
        })
        .collect();
    // OVERWRITE semantics: discard the entire destination subtree (every surviving
    // key under `to`) before re-inserting the moved entries there.
    let dst: Vec<_> = model
        .keys()
        .filter(|key| key.starts_with(to))
        .cloned()
        .collect();
    for key in dst {
        model.remove(&key);
    }
    for (key, value) in moved_kv {
        let mut new_key = to.to_vec();
        new_key.extend_from_slice(&key[from.len()..]);
        model.insert(new_key, value);
    }
}

/// Precondition-aware naive `move_prefix`: returns the relocated model on success,
/// or `None` when the real op would `Err` (equal prefixes, overlong prefix args
/// or resulting moved keys, an absent `from`, or a non-empty / prefix-violating
/// `to`). The byte-identity gate.
fn model_move_prefix(model: &Model, from: &[u8], to: &[u8]) -> Option<Model> {
    if from.len() > MAX_KEY_LEN || to.len() > MAX_KEY_LEN || from == to {
        return None;
    }
    let deepest_moved_len = model
        .keys()
        .filter(|key| key.starts_with(from))
        .map(|key| to.len() + key.len() - from.len())
        .max();
    let Some(deepest_moved_len) = deepest_moved_len else {
        return None; // absent `from`
    };
    if deepest_moved_len > MAX_KEY_LEN {
        return None;
    }
    // Prefix-free against the *surviving* keys: no surviving key may be a byte-prefix
    // of `to` (that would make it an ancestor of the moved keys). Surviving keys
    // *under* `to` are NOT rejected — OVERWRITE discards them (see
    // `apply_move_prefix_to_model`).
    for key in model.keys() {
        if key.starts_with(from) {
            continue; // relocated away
        }
        // Reject only a *strict* prefix of `to` (a surviving ancestor that would
        // become a byte-prefix of the moved keys). A surviving key equal to `to`,
        // or under `to`, is discarded by OVERWRITE — not a violation.
        if to.starts_with(key.as_slice()) && key.len() < to.len() {
            return None;
        }
    }
    let mut out = model.clone();
    apply_move_prefix_to_model(&mut out, from, to);
    Some(out)
}

fn expected_reads(mut model: Model, steps: &[Step]) -> VerifiedReadResults {
    let mut out = Vec::new();
    for step in steps {
        match step {
            Step::Read(reads) => {
                out.push(
                    reads
                        .iter()
                        .map(|op| ProvenRead {
                            op: op.clone(),
                            results: model_read(&model, op),
                        })
                        .collect(),
                );
            }
            Step::Write(ops) => apply_writes_to_model(&mut model, ops),
            Step::MovePrefix { from, to } => apply_move_prefix_to_model(&mut model, from, to),
        }
    }
    out
}

fn assert_tracer_transcript(entries: &[(Vec<u8>, Vec<u8>)], steps: &[Step]) {
    let start_model = model_from_entries(entries);
    let avl = build_avl(entries);
    let mrt = build_mrt(entries);
    let avl_snapshot = avl.checkpoint();
    let mrt_snapshot = mrt.checkpoint();

    let avl_trace = avl_ts::create_trace(&avl_snapshot, steps).unwrap();
    let mrt_trace = mrt_ts::create_trace(&mrt_snapshot, steps).unwrap();

    let avl_start = avl_trace.hash();
    let mrt_start = mrt_snapshot.root_hash();
    let avl_end = avl_ts::root_after_writes(&avl_snapshot, steps);
    let mrt_end = mrt_ts::root_after_writes(&mrt_snapshot, steps);

    let avl_reads = avl_ts::replay_trace(&avl_trace, avl_start, steps, avl_end).unwrap();
    let mrt_reads = mrt_ts::replay_trace(&mrt_trace, mrt_start, steps, mrt_end).unwrap();
    let expected_reads = expected_reads(start_model, steps);
    assert_eq!(avl_reads, expected_reads);
    assert_eq!(mrt_reads, expected_reads);

    let avl_live = build_avl(entries);
    let mrt_live = build_mrt(entries);
    apply_writes_to_live(&avl_live, &mrt_live, steps);
    assert_eq!(avl_end, avl_live.root_hash());
    assert_eq!(mrt_end, mrt_live.root_hash());
    assert_live_states_equal(&avl_live, &mrt_live);

    assert_sparse_trace_roundtrips(&avl_trace, avl_start);
    assert_mrt_trace_roundtrips(&mrt_trace, mrt_start);

    // End-root corruption is detected by replay (the post-state authentication).
    let mut bad_avl_end = avl_end;
    bad_avl_end[0] ^= 0x80;
    assert!(avl_ts::replay_trace(&avl_trace, avl_start, steps, bad_avl_end).is_err());
    let mut bad_mrt_end = mrt_end;
    bad_mrt_end[0] ^= 0x80;
    assert!(mrt_ts::replay_trace(&mrt_trace, mrt_start, steps, bad_mrt_end).is_err());
}

fn assert_sparse_trace_roundtrips(trace: &AvlTrace, expected_root: Hash) {
    let decoded = AvlTrace::decode_exact(&trace.encode().unwrap()).unwrap();
    decoded.verify_root(expected_root).unwrap();
}

fn assert_mrt_trace_roundtrips(trace: &Trace, expected_root: Hash) {
    let decoded = Trace::decode_exact(&trace.encode().unwrap()).unwrap();
    decoded.verify_root(expected_root).unwrap();
}

fn count_revealed_mrt(trace: &Trace) -> usize {
    fn walk(node: &Arc<MrtNodeInner>) -> usize {
        match node.node() {
            MrtNode::Leaf { .. } => 1,
            MrtNode::PrunedHash => 0,
            MrtNode::Branch { left, right, .. } => 1 + walk(left) + walk(right),
        }
    }

    trace.0.as_ref().map_or(0, walk)
}

fn normalize_query_items(query: Vec<QueryItem>) -> Vec<QueryItem> {
    let mut normalized = Query::new();
    for item in query {
        normalized.insert_item(item);
    }
    normalized.into()
}

fn assert_query_proofs_match(avl: &InMemoryMerk, mrt: &Tree, query: Vec<QueryItem>) {
    let query = normalize_query_items(query);
    let avl_bytes = avl.prove(query.clone()).unwrap();
    let mrt_bytes = mrt.prove(query.clone()).unwrap();
    let avl_query = Query::from(query.clone());

    let avl_result = verify_query(&avl_bytes, &avl_query, avl.root_hash()).unwrap();
    let mrt_query = Query::from(query);
    let mrt_result = verify_mrt_query(&mrt_bytes, &mrt_query, mrt.root_hash()).unwrap();
    assert_eq!(avl_result, mrt_result);
}

fn assert_serde_surface<T>()
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
}

#[test]
fn mrt_trace_serde_surface_roundtrips() {
    assert_serde_surface::<Trace>();

    let mrt = build_mrt(&[(b"a".to_vec(), b"1".to_vec())]);
    let snapshot = mrt.checkpoint();
    let steps = vec![Step::Read(vec![ReadOp::Key(b"a".to_vec())])];
    let trace = mrt_ts::create_trace(&snapshot, &steps).unwrap();
    let start = snapshot.root_hash();
    let end = mrt_ts::root_after_writes(&snapshot, &steps);

    let decoded_trace: Trace = serde_json::from_slice(&serde_json::to_vec(&trace).unwrap())
        .expect("MRT trace should serde round-trip");
    assert_eq!(decoded_trace, trace);
    assert_eq!(
        mrt_ts::replay_trace(&decoded_trace, start, &steps, end).unwrap(),
        expected_reads(
            model_from_entries(&[(b"a".to_vec(), b"1".to_vec())]),
            &steps
        )
    );
}

#[test]
fn mrt_serde_rejects_top_level_pruned_root() {
    // `Trace` is a public serde surface, so the "bare pruned root is rejected at
    // decode" rule must hold on the serde `Deserialize` path too (not only the flat
    // `ed` decoder). Serializing a pruned-root trace succeeds; deserializing it back
    // must fail — `trace_from_wire` rejects a top-level `Pruned`.
    let pruned_root = Trace(Some(MrtNodeInner::pruned([0x5c; 32], 0)));
    let bytes = serde_json::to_vec(&pruned_root).expect("serialization is unrestricted");
    let decoded: Result<Trace, _> = serde_json::from_slice(&bytes);
    assert!(
        decoded.is_err(),
        "serde decode accepted a bare pruned root: {:?}",
        decoded
    );

    // A normal (materialized) trace still round-trips through the same surface.
    let ok = Trace(
        build_mrt(&[(b"a".to_vec(), b"1".to_vec())])
            .checkpoint()
            .root
            .clone(),
    );
    let round: Trace = serde_json::from_slice(&serde_json::to_vec(&ok).unwrap())
        .expect("materialized trace round-trips");
    assert_eq!(round, ok);
}

#[test]
fn mrt_snapshot_public_read_helpers_report_results() {
    let mrt = build_mrt(&[
        (b"aa".to_vec(), b"1".to_vec()),
        (b"ab".to_vec(), b"2".to_vec()),
        (b"b".to_vec(), b"3".to_vec()),
        (b"c".to_vec(), b"4".to_vec()),
    ]);
    let snapshot = mrt.checkpoint();

    match snapshot.get_result(b"aa").unwrap() {
        GetResult::Found(value) => assert_eq!(value, b"1".to_vec()),
        other => panic!("unexpected get_result for aa: {:?}", other),
    }
    assert!(matches!(
        snapshot.get_result(b"z").unwrap(),
        GetResult::NotFound
    ));
    assert_eq!(
        snapshot.collect_range(b"a", Some(b"c")).unwrap(),
        vec![
            (b"aa".to_vec(), b"1".to_vec()),
            (b"ab".to_vec(), b"2".to_vec()),
            (b"b".to_vec(), b"3".to_vec())
        ]
    );
    assert_eq!(
        snapshot.collect_prefix(b"a").unwrap(),
        vec![
            (b"aa".to_vec(), b"1".to_vec()),
            (b"ab".to_vec(), b"2".to_vec())
        ]
    );
}

#[test]
fn non_empty_avl_and_mrt_snapshots_share_basic_read_surface() {
    let entries = [
        (b"aa".to_vec(), b"1".to_vec()),
        (b"ab".to_vec(), b"2".to_vec()),
        (b"b".to_vec(), b"3".to_vec()),
        (b"c".to_vec(), b"4".to_vec()),
    ];
    let avl = build_avl(&entries);
    let mrt = build_mrt(&entries);
    // Both backends expose the same `Checkpoint` read surface.
    let avl_snapshot = avl.checkpoint();
    let mrt_snapshot = mrt.checkpoint();

    assert_eq!(avl_snapshot.root_hash(), avl.root_hash());
    assert_eq!(mrt_snapshot.root_hash(), mrt.root_hash());
    assert_eq!(
        avl_snapshot.get_result(b"ab").unwrap(),
        mrt_snapshot.get_result(b"ab").unwrap()
    );
    assert_eq!(
        avl_snapshot.collect_range(b"a", Some(b"c")).unwrap(),
        mrt_snapshot.collect_range(b"a", Some(b"c")).unwrap()
    );
    assert_eq!(
        avl_snapshot.collect_prefix(b"a").unwrap(),
        mrt_snapshot.collect_prefix(b"a").unwrap()
    );

    let query_items = vec![
        QueryItem::Key(b"ab".to_vec()),
        QueryItem::Range(b"a".to_vec()..b"c".to_vec()),
    ];
    let query = Query::from(query_items.clone());
    let avl_proof = avl_snapshot.prove(query_items.clone()).unwrap();
    let mrt_proof = mrt_snapshot.prove(query_items).unwrap();
    assert_eq!(
        verify_query(&avl_proof, &query, avl_snapshot.root_hash()).unwrap(),
        verify_mrt_query(&mrt_proof, &query, mrt_snapshot.root_hash()).unwrap()
    );
}

#[test]
fn cross_backend_public_api_and_proofs_align() {
    let avl = InMemoryMerk::new();
    let mrt = Tree::new();
    assert_eq!(avl.root_hash(), NULL_HASH);
    assert_eq!(mrt.root_hash(), NULL_HASH);
    assert!(mrt.checkpoint().is_empty());

    avl.put(b"a".to_vec(), b"1".to_vec()).unwrap();
    mrt.put(b"a".to_vec(), b"1".to_vec()).unwrap();
    avl.apply_sorted_batch_ops_owned(vec![(b"c".to_vec(), Op::Put(b"3".to_vec()))])
        .unwrap();
    mrt.apply_sorted_batch_ops_owned(vec![(b"c".to_vec(), Op::Put(b"3".to_vec()))])
        .unwrap();
    assert_live_states_equal(&avl, &mrt);
    assert_ne!(avl.root_hash(), NULL_HASH);
    assert_ne!(mrt.root_hash(), NULL_HASH);
    assert_ne!(avl.root_hash(), mrt.root_hash());

    let avl_snapshot = avl.checkpoint();
    let mrt_snapshot = mrt.checkpoint();
    assert_eq!(avl_entries(avl_snapshot.root()), mrt_entries(&mrt_snapshot));

    avl.put(b"b".to_vec(), b"2".to_vec()).unwrap();
    mrt.put(b"b".to_vec(), b"2".to_vec()).unwrap();
    avl.delete(b"a".to_vec()).unwrap();
    mrt.delete(b"a".to_vec()).unwrap();
    assert_live_states_equal(&avl, &mrt);
    assert_eq!(
        avl_entries(avl_snapshot.root()),
        vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"c".to_vec(), b"3".to_vec())
        ]
    );
    assert_eq!(avl_entries(avl_snapshot.root()), mrt_entries(&mrt_snapshot));

    avl.delete_range(b"b".to_vec(), b"d".to_vec()).unwrap();
    mrt.delete_range(b"b".to_vec(), b"d".to_vec()).unwrap();
    assert_live_states_equal(&avl, &mrt);
    assert!(avl.checkpoint().is_empty());
    assert!(mrt.checkpoint().is_empty());

    let avl = build_avl(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"2".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
    ]);
    let mrt = build_mrt(&[
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"2".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
    ]);
    assert_query_proofs_match(
        &avl,
        &mrt,
        vec![
            QueryItem::Key(b"b".to_vec()),
            QueryItem::Key(b"z".to_vec()),
            QueryItem::Range(b"a".to_vec()..b"d".to_vec()),
        ],
    );
}

#[test]
fn tracer_scenario_regressions() {
    let large = vec![7u8; 64];
    let cases = vec![
        (
            vec![
                (b"alpha".to_vec(), b"a".to_vec()),
                (b"bravo".to_vec(), b"b".to_vec()),
            ],
            vec![
                Step::Write(vec![put(b"charlie", b"c")]),
                Step::Read(vec![ReadOp::Key(b"charlie".to_vec())]),
            ],
        ),
        (
            vec![
                (b"key1".to_vec(), b"old1".to_vec()),
                (b"key2".to_vec(), b"old2".to_vec()),
                (b"key3".to_vec(), b"old3".to_vec()),
            ],
            vec![
                Step::Read(vec![ReadOp::Key(b"key1".to_vec())]),
                Step::Write(vec![put(b"key1", b"new1")]),
                Step::Read(vec![
                    ReadOp::Key(b"key1".to_vec()),
                    ReadOp::Key(b"key2".to_vec()),
                ]),
            ],
        ),
        (
            vec![
                (b"del".to_vec(), b"val".to_vec()),
                (b"keep".to_vec(), b"val2".to_vec()),
                (b"stay".to_vec(), b"val3".to_vec()),
            ],
            vec![
                Step::Write(vec![delete(b"del")]),
                Step::Read(vec![
                    ReadOp::Key(b"del".to_vec()),
                    ReadOp::Key(b"keep".to_vec()),
                ]),
            ],
        ),
        (
            (0u8..8)
                .map(|i| (vec![i], format!("v{i}").into_bytes()))
                .collect(),
            vec![
                Step::Read(vec![ReadOp::Key(vec![2])]),
                Step::Write(vec![put(vec![2], b"two-new"), put(vec![20], b"twenty")]),
                Step::Read(vec![ReadOp::Key(vec![2]), ReadOp::Key(vec![20])]),
                Step::Write(vec![delete(vec![4])]),
                Step::Read(vec![ReadOp::Key(vec![4]), ReadOp::Key(vec![0])]),
            ],
        ),
        (
            vec![
                (b"aa".to_vec(), b"1".to_vec()),
                (b"ab".to_vec(), b"2".to_vec()),
                (b"ba".to_vec(), b"3".to_vec()),
                (b"ca".to_vec(), b"4".to_vec()),
            ],
            vec![
                Step::Write(vec![put(b"ac", b"new"), delete(b"ba")]),
                Step::Read(vec![
                    ReadOp::Range {
                        start: b"aa".to_vec(),
                        end: b"b".to_vec(),
                    },
                    ReadOp::Prefix(b"a".to_vec()),
                ]),
            ],
        ),
        (
            vec![
                (b"aa".to_vec(), b"1".to_vec()),
                (b"bb".to_vec(), b"2".to_vec()),
            ],
            vec![
                Step::Write(vec![put(b"cc", b"3")]),
                Step::Read(vec![
                    ReadOp::Range {
                        start: b"dd".to_vec(),
                        end: b"ee".to_vec(),
                    },
                    ReadOp::Prefix(b"z".to_vec()),
                ]),
            ],
        ),
        (
            (0u8..8).map(|i| (vec![i], vec![i; 2])).collect(),
            vec![
                Step::Write(vec![put(vec![10], b"ten")]),
                Step::Write(vec![delete(vec![3])]),
            ],
        ),
        (
            (0u8..6).map(|i| (vec![i], vec![i; 3])).collect(),
            vec![Step::Write(Vec::new())],
        ),
        (
            vec![(b"a".to_vec(), b"1".to_vec())],
            vec![
                Step::Write(vec![put(b"large", large)]),
                Step::Read(vec![ReadOp::Key(b"a".to_vec())]),
            ],
        ),
    ];

    for (entries, steps) in cases {
        assert_tracer_transcript(&entries, &steps);
    }
}

#[test]
fn tracer_differential_regression_cases() {
    let cases = vec![
        (
            vec![
                (b"a".to_vec(), b"va".to_vec()),
                (b"c".to_vec(), b"vc".to_vec()),
            ],
            vec![put(b"b", b"vb")],
            vec![ReadOp::Range {
                start: b"a".to_vec(),
                end: b"c".to_vec(),
            }],
        ),
        (
            vec![
                (vec![0x00], b"v00".to_vec()),
                (vec![0x40], b"v40".to_vec()),
                (vec![0x80], b"v80".to_vec()),
                (vec![0xc0], b"vc0".to_vec()),
            ],
            vec![delete(vec![0x40])],
            vec![ReadOp::Range {
                start: vec![0x00],
                end: vec![0x41],
            }],
        ),
        (
            vec![
                (b"a".to_vec(), b"va".to_vec()),
                (b"c".to_vec(), b"vc".to_vec()),
            ],
            vec![delete(b"b")],
            vec![
                ReadOp::Key(b"b".to_vec()),
                ReadOp::Range {
                    start: b"a".to_vec(),
                    end: b"c".to_vec(),
                },
            ],
        ),
        (
            vec![
                (b"aa".to_vec(), b"vaa".to_vec()),
                (b"ab".to_vec(), b"vab".to_vec()),
                (b"ba".to_vec(), b"vba".to_vec()),
                (vec![0xff], b"vff".to_vec()),
            ],
            vec![put(b"ac", b"vac")],
            vec![ReadOp::Prefix(b"a".to_vec())],
        ),
        (
            vec![
                (b"a".to_vec(), b"va".to_vec()),
                (b"b".to_vec(), b"vb".to_vec()),
                (b"c".to_vec(), b"vc".to_vec()),
                (b"d".to_vec(), b"vd".to_vec()),
            ],
            vec![delete(b"b")],
            vec![
                ReadOp::Range {
                    start: b"b".to_vec(),
                    end: b"d".to_vec(),
                },
                ReadOp::Prefix(b"c".to_vec()),
            ],
        ),
        (
            vec![
                (b"aa".to_vec(), b"vaa".to_vec()),
                (b"ab".to_vec(), b"vab".to_vec()),
                (b"b".to_vec(), b"vb".to_vec()),
            ],
            vec![put(b"ab", b"updated-ab")],
            vec![
                ReadOp::Key(b"ab".to_vec()),
                ReadOp::Range {
                    start: b"aa".to_vec(),
                    end: b"b".to_vec(),
                },
                ReadOp::Prefix(b"a".to_vec()),
            ],
        ),
    ];

    for (entries, ops, reads) in cases {
        let steps = vec![Step::Write(ops), Step::Read(reads)];
        assert_tracer_transcript(&entries, &steps);
    }
}

#[test]
fn mrt_tracer_targeted_scan_regressions() {
    let cases = vec![
        (
            vec![
                (vec![0x00u8], b"val-a".to_vec()),
                (vec![0x80u8], b"val-b".to_vec()),
            ],
            vec![
                Step::Write(vec![delete(vec![0x00u8])]),
                Step::Write(vec![put(vec![0xc0u8], b"val-c")]),
                Step::Read(vec![ReadOp::Range {
                    start: vec![0x00],
                    end: vec![0xff],
                }]),
            ],
        ),
        (
            vec![(vec![0x00u8], b"val-a".to_vec())],
            vec![
                Step::Write(vec![put(vec![0x80u8], b"val-b")]),
                Step::Read(vec![ReadOp::Key(vec![0x80])]),
                Step::Write(vec![put(vec![0xc0u8], b"val-c")]),
                Step::Read(vec![ReadOp::Key(vec![0xc0])]),
            ],
        ),
        (
            vec![
                (b"aa".to_vec(), b"va".to_vec()),
                (b"ab".to_vec(), b"vab".to_vec()),
                (b"ba".to_vec(), b"vb".to_vec()),
            ],
            vec![
                Step::Write(vec![delete_range(b"a", b"b")]),
                Step::Read(vec![ReadOp::Range {
                    start: b"a".to_vec(),
                    end: b"c".to_vec(),
                }]),
            ],
        ),
    ];

    for (entries, steps) in cases {
        let mrt = build_mrt(&entries);
        let snapshot = mrt.checkpoint();
        let trace = mrt_ts::create_trace(&snapshot, &steps).unwrap();
        let start = snapshot.root_hash();
        let end = mrt_ts::root_after_writes(&snapshot, &steps);
        let reads = mrt_ts::replay_trace(&trace, start, &steps, end).unwrap();
        assert_eq!(reads, expected_reads(model_from_entries(&entries), &steps));
    }

    let mrt = build_mrt(&[
        (b"aa".to_vec(), b"1".to_vec()),
        (b"bb".to_vec(), b"2".to_vec()),
        (b"cc".to_vec(), b"3".to_vec()),
    ]);
    let query = vec![QueryItem::RangeInclusive(std::ops::RangeInclusive::new(
        b"dd".to_vec(),
        b"ee".to_vec(),
    ))];
    let bytes = mrt.prove(query.clone()).unwrap();
    let query = Query::from(query);
    let result = verify_mrt_query(&bytes, &query, mrt.root_hash()).unwrap();
    assert!(result.is_empty());
}

fn random_bytes(rng: &mut SmallRng, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    rng.fill_bytes(&mut out);
    out
}

fn random_key(rng: &mut SmallRng) -> Vec<u8> {
    // Fixed length ⇒ prefix-free, which the MRT now requires. Shared random
    // prefixes still produce branches at varied bit depths.
    random_bytes(rng, 3)
}

fn random_value(rng: &mut SmallRng, tag: u8) -> Vec<u8> {
    let len = match rng.gen_range(0..5) {
        0 => 0,
        1 => 1,
        2 => 8,
        3 => 32,
        _ => 80,
    };
    let mut value = vec![tag; len];
    rng.fill_bytes(&mut value);
    value
}

fn random_end_after(rng: &mut SmallRng, start: &[u8]) -> Vec<u8> {
    if rng.gen_bool(0.35) {
        let mut end = start.to_vec();
        end.push(rng.gen_range(1..=0xff));
        return end;
    }
    for _ in 0..16 {
        let candidate = random_key(rng);
        if candidate.as_slice() > start {
            return candidate;
        }
    }
    let mut end = start.to_vec();
    end.push(1);
    end
}

fn choose_probe_key(rng: &mut SmallRng, keys: &[Vec<u8>], model: &Model) -> Vec<u8> {
    if !model.is_empty() && rng.gen_bool(0.55) {
        let index = rng.gen_range(0..model.len());
        model.keys().nth(index).unwrap().clone()
    } else {
        keys[rng.gen_range(0..keys.len())].clone()
    }
}

fn random_read_op(rng: &mut SmallRng, keys: &[Vec<u8>], model: &Model) -> ReadOp {
    match rng.gen_range(0..3) {
        0 => ReadOp::Key(choose_probe_key(rng, keys, model)),
        1 => {
            let a = choose_probe_key(rng, keys, model);
            let b = choose_probe_key(rng, keys, model);
            let (start, end) = if a < b {
                (a, b)
            } else if b < a {
                (b, a)
            } else {
                let end = random_end_after(rng, &a);
                (a, end)
            };
            ReadOp::Range { start, end }
        }
        _ => {
            let key = choose_probe_key(rng, keys, model);
            let len = if key.is_empty() {
                0
            } else {
                rng.gen_range(0..=key.len())
            };
            ReadOp::Prefix(key[..len].to_vec())
        }
    }
}

fn random_write_ops(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &Model,
    max_ops: usize,
    tag: u8,
) -> Vec<BatchOp> {
    let max_unique_ops = max_ops.min(keys.len());
    if max_unique_ops == 0 {
        return Vec::new();
    }
    let op_count = rng.gen_range(1..=max_unique_ops);
    let mut used = BTreeSet::new();
    let mut ops = Vec::new();

    while ops.len() < op_count {
        let key = choose_probe_key(rng, keys, model);
        if !used.insert(key.clone()) {
            continue;
        }

        let op = match rng.gen_range(0..5) {
            0 if model.contains_key(&key) => delete(key),
            1 => delete_range(key.clone(), random_end_after(rng, &key)),
            _ => put(key, random_value(rng, tag)),
        };
        ops.push(op);
    }

    ops.sort_by(|a, b| a.key().cmp(b.key()));
    ops
}

fn random_steps(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    start_model: &Model,
    max_batches: usize,
    max_ops_per_batch: usize,
) -> Vec<Step> {
    let step_count = rng.gen_range(1..=max_batches);
    let mut model = start_model.clone();
    let mut steps = Vec::with_capacity(step_count);

    for step in 0..step_count {
        if rng.gen_bool(0.45) {
            let read_count = rng.gen_range(1..=3);
            let reads = (0..read_count)
                .map(|_| random_read_op(rng, keys, &model))
                .collect();
            steps.push(Step::Read(reads));
        } else {
            let ops = random_write_ops(rng, keys, &model, max_ops_per_batch, step as u8);
            apply_writes_to_model(&mut model, &ops);
            steps.push(Step::Write(ops));
        }
    }

    steps
}

fn random_initial_entries(rng: &mut SmallRng) -> (KeySet, Entries) {
    let key_count = rng.gen_range(8..=64);
    let mut keys = BTreeSet::new();
    // The whole key pool must be prefix-free; `random_key` is fixed-length, so the
    // min/max edge seeds are the all-zero and all-one keys of that same length.
    keys.insert(vec![0x00, 0x00, 0x00]);
    keys.insert(vec![0xff, 0xff, 0xff]);
    while keys.len() < key_count {
        keys.insert(random_key(rng));
    }
    let keys: Vec<_> = keys.into_iter().collect();
    let mut entries = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        if rng.gen_bool(0.65) {
            entries.push((key.clone(), random_value(rng, index as u8)));
        }
    }
    (keys, entries)
}

fn require_env(name: &str) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| {
            panic!(
                "Set FUZZ_SEED=<n> and FUZZ_ITERS=<n>.\n\
                 Run: FUZZ_SEED=0 FUZZ_ITERS=1000 cargo test --release --lib \
                 mrt::cross_backend_tests::fuzz_tracer_differential -- --ignored\n\
                 Reproduce: FUZZ_SEED=<seed> FUZZ_ITERS=1 cargo test --release --lib \
                 mrt::cross_backend_tests::fuzz_tracer_differential -- --ignored"
            )
        })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .max(1)
}

fn env_usize_allow_zero(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn fuzz_tracer_differential() {
    let start_seed = require_env("FUZZ_SEED");
    let iterations = require_env("FUZZ_ITERS");
    let max_transcripts = env_usize("FUZZ_MAX_TRANSCRIPTS", 10);
    let max_batches = env_usize("FUZZ_MAX_BATCHES", 10);
    let max_ops_per_batch = env_usize("FUZZ_MAX_OPS_PER_BATCH", 10);
    let progress_interval = env_usize_allow_zero("FUZZ_PROGRESS_INTERVAL", 10);

    for iter in 0..iterations {
        let seed = start_seed.wrapping_add(iter);
        let mut rng = SmallRng::seed_from_u64(seed);
        let transcript_count = rng.gen_range(1..=max_transcripts);

        for transcript in 0..transcript_count {
            let (keys, entries) = random_initial_entries(&mut rng);
            let start_model = model_from_entries(&entries);
            let steps = random_steps(
                &mut rng,
                &keys,
                &start_model,
                max_batches,
                max_ops_per_batch,
            );

            let result = std::panic::catch_unwind(|| assert_tracer_transcript(&entries, &steps));
            if let Err(payload) = result {
                eprintln!(
                    "fuzz_tracer_differential failed: seed {seed:#x} transcript {transcript}"
                );
                eprintln!("entries: {entries:?}");
                eprintln!("steps: {steps:?}");
                std::panic::resume_unwind(payload);
            }
        }

        if progress_interval > 0 && (iter + 1) % progress_interval as u64 == 0 {
            eprintln!(
                "fuzz_tracer_differential: completed {}/{} iterations; last seed {seed:#x}",
                iter + 1,
                iterations
            );
        }
    }
}

// ─── move_prefix step (stage 3): MRT vs naive byte-identity model ────────────
//
// `MovePrefix` is MRT-only, so these transcripts run through the MRT tracer +
// verifier and are compared to the precondition-aware naive model
// (`model_move_prefix`) rather than the AVL backend. The differential checks both
// directions: a valid move yields the same end root as a "delete each, re-insert
// at `to`" rebuild, and an invalid `(from, to)` is rejected identically (both the
// live op and the model `Err`/`None`).

fn model_entries(model: &Model) -> Entries {
    model.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Replay a MRT-only transcript on a live tree through the p2-facing
/// `apply_write_ops` path — `Write` and `MovePrefix` both as `WriteOp`s (MRT
/// supports move), `Read` ignored. Exercises `apply_write_ops`'s full vocabulary,
/// including `MovePrefix`, against the live tree (the AVL-free counterpart to
/// [`apply_writes_to_live`], which is move-free).
fn apply_steps_to_mrt_live(mrt: &Tree, steps: &[Step]) {
    for step in steps {
        match step {
            Step::Write(ops) => mrt.apply_write_ops(&write_ops(ops)).unwrap(),
            Step::MovePrefix { from, to } => mrt
                .apply_write_ops(&[WriteOp::MovePrefix {
                    from: from.clone(),
                    to: to.clone(),
                }])
                .unwrap(),
            Step::Read(_) => {}
        }
    }
}

/// The model after folding every mutating step (`Write` + `MovePrefix`).
fn final_model(entries: &[(Vec<u8>, Vec<u8>)], steps: &[Step]) -> Model {
    let mut model = model_from_entries(entries);
    for step in steps {
        match step {
            Step::Write(ops) => apply_writes_to_model(&mut model, ops),
            Step::MovePrefix { from, to } => apply_move_prefix_to_model(&mut model, from, to),
            Step::Read(_) => {}
        }
    }
    model
}

/// Full MRT-only transcript gate: trace, verify, live-replay, and naive-model
/// agreement, plus trace round-trip and end-root corruption detection. Every
/// `MovePrefix` step in `steps` must be a *valid* move (the generators guarantee
/// it); rejection is checked separately by [`assert_move_op_matches_model`].
fn assert_mrt_move_transcript(entries: &[(Vec<u8>, Vec<u8>)], steps: &[Step]) {
    // `build_mrt`/`apply_sorted_batch_ops` require sorted, unique keys; the model comparisons
    // are order-independent, so sort the fixture entries up front.
    let mut entries = entries.to_vec();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let entries = entries.as_slice();

    let mrt = build_mrt(entries);
    let snapshot = mrt.checkpoint();

    let trace = mrt_ts::create_trace(&snapshot, steps).unwrap();
    let start = snapshot.root_hash();
    let end = mrt_ts::root_after_writes(&snapshot, steps);
    let reads = mrt_ts::replay_trace(&trace, start, steps, end).unwrap();
    assert_eq!(reads, expected_reads(model_from_entries(entries), steps));

    let live = build_mrt(entries);
    apply_steps_to_mrt_live(&live, steps);
    assert_eq!(end, live.root_hash());

    // Naive byte-identity oracle: the live tree must equal a from-scratch rebuild
    // of the folded model.
    let model = final_model(entries, steps);
    let expected = build_mrt(&model_entries(&model));
    assert_eq!(live.root_hash(), expected.root_hash());
    assert_eq!(mrt_entries(&live.checkpoint()), model_entries(&model));

    assert_mrt_trace_roundtrips(&trace, start);

    let mut bad_end = end;
    bad_end[0] ^= 0x80;
    assert!(mrt_ts::replay_trace(&trace, start, steps, bad_end).is_err());
}

/// Assert the live `move_prefix` op agrees with the precondition-aware model:
/// either both succeed with byte-identical results, or both reject (and a rejected
/// move leaves the tree untouched). Successful moves additionally run the full
/// trace+verify transcript gate.
fn assert_move_op_matches_model(entries: &[(Vec<u8>, Vec<u8>)], from: &[u8], to: &[u8]) {
    let mut entries = entries.to_vec();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let entries = entries.as_slice();

    let model = model_from_entries(entries);
    let mrt = build_mrt(entries);
    let before = mrt.root_hash();
    let result = mrt.move_prefix(from.to_vec(), to.to_vec());

    match model_move_prefix(&model, from, to) {
        Some(expected_model) => {
            result.unwrap_or_else(|err| {
                panic!("move {:?}->{:?} should succeed: {:?}", from, to, err)
            });
            let expected = build_mrt(&model_entries(&expected_model));
            assert_eq!(
                mrt.root_hash(),
                expected.root_hash(),
                "move {:?}->{:?} root mismatch vs model",
                from,
                to
            );
            assert_eq!(
                mrt_entries(&mrt.checkpoint()),
                model_entries(&expected_model)
            );

            let steps = vec![Step::MovePrefix {
                from: from.to_vec(),
                to: to.to_vec(),
            }];
            assert_mrt_move_transcript(entries, &steps);
        }
        None => {
            assert!(
                result.is_err(),
                "move {:?}->{:?} should be rejected but succeeded",
                from,
                to
            );
            assert_eq!(
                mrt.root_hash(),
                before,
                "rejected move {:?}->{:?} must not mutate the tree",
                from,
                to
            );
        }
    }
}

#[test]
fn mrt_move_prefix_model_oracle_hand_fixtures() {
    // Gate the naive model itself on hand-computed expectations before trusting it
    // as the differential oracle.
    let model = model_from_entries(&[
        (b"user:alice".to_vec(), b"1".to_vec()),
        (b"user:bob".to_vec(), b"2".to_vec()),
        (b"sys:log".to_vec(), b"3".to_vec()),
    ]);

    // Valid relocate: user:* -> acct:* (same length), values preserved.
    let moved = model_move_prefix(&model, b"user:", b"acct:").expect("valid move");
    assert_eq!(
        model_entries(&moved),
        vec![
            (b"acct:alice".to_vec(), b"1".to_vec()),
            (b"acct:bob".to_vec(), b"2".to_vec()),
            (b"sys:log".to_vec(), b"3".to_vec()),
        ]
    );

    // Lengthening and shortening are valid when the destination is empty and the
    // deepest resulting key stays within MAX_KEY_LEN.
    let lengthened = model_move_prefix(&model, b"user:", b"accounts:").expect("valid lengthening");
    assert_eq!(
        model_entries(&lengthened),
        vec![
            (b"accounts:alice".to_vec(), b"1".to_vec()),
            (b"accounts:bob".to_vec(), b"2".to_vec()),
            (b"sys:log".to_vec(), b"3".to_vec()),
        ]
    );
    let shortened = model_move_prefix(&model, b"user:", b"u").expect("valid shortening");
    assert_eq!(
        model_entries(&shortened),
        vec![
            (b"sys:log".to_vec(), b"3".to_vec()),
            (b"ualice".to_vec(), b"1".to_vec()),
            (b"ubob".to_vec(), b"2".to_vec()),
        ]
    );

    // Equal prefixes and an absent source reject.
    assert!(model_move_prefix(&model, b"user:", b"user:").is_none());
    assert!(model_move_prefix(&model, b"zzz", b"yyy").is_none());

    // The destination prefix may be within MAX_KEY_LEN while the deepest moved
    // suffix still overflows the resulting key length.
    let mut max_key = vec![0x10; MAX_KEY_LEN];
    max_key[1] = 0x20;
    let overflow = model_from_entries(&[(max_key, b"deep".to_vec())]);
    let overflow_to = vec![0x30; MAX_KEY_LEN];
    assert!(model_move_prefix(&overflow, &[0x10], &overflow_to).is_none());

    // Occupied destination: OVERWRITE discards the destination subtree (`sys:log`,
    // under `sys:l`) and relocates `user:*` there.
    let overwritten = model_move_prefix(&model, b"user:", b"sys:l").expect("overwrite is valid");
    assert_eq!(
        model_entries(&overwritten),
        vec![
            (b"sys:lalice".to_vec(), b"1".to_vec()),
            (b"sys:lbob".to_vec(), b"2".to_vec()),
        ]
    );

    // Exact-key destination is an overwrite (not a violation): `zz` -> `ab` replaces
    // the existing `ab`.
    let two = model_from_entries(&[
        (b"ab".to_vec(), b"x".to_vec()),
        (b"zz".to_vec(), b"y".to_vec()),
    ]);
    assert_eq!(
        model_entries(&model_move_prefix(&two, b"zz", b"ab").expect("overwrite")),
        vec![(b"ab".to_vec(), b"y".to_vec())],
    );
    // A clean divergence at `to` is fine.
    assert!(model_move_prefix(&two, b"zz", b"ac").is_some());

    // Prefix-free violation: a *surviving* key is a strict byte-prefix of `to`.
    let ancestor = model_from_entries(&[
        (b"a".to_vec(), b"x".to_vec()),
        (b"zz".to_vec(), b"y".to_vec()),
    ]);
    assert!(model_move_prefix(&ancestor, b"zz", b"ab").is_none());
}

#[test]
fn mrt_move_prefix_deterministic_fixtures() {
    // Move a (non-root) leaf subtree.
    assert_move_op_matches_model(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"z".to_vec(), b"2".to_vec()),
        ],
        b"abc",
        b"def",
    );

    // Move a deep multi-key subtree (a whole namespace rename).
    assert_move_op_matches_model(
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

    // Lengthening that fits: the moved suffixes remain below MAX_KEY_LEN and the
    // proof verifies against the rebuilt model.
    assert_move_op_matches_model(
        &[
            (b"user:alice".to_vec(), b"1".to_vec()),
            (b"user:bob".to_vec(), b"2".to_vec()),
            (b"sys:log".to_vec(), b"3".to_vec()),
        ],
        b"user:",
        b"accounts:",
    );

    // Shortening is also valid when the destination prefix is empty in the
    // post-detach tree.
    assert_move_op_matches_model(
        &[
            (b"user:alice".to_vec(), b"1".to_vec()),
            (b"user:bob".to_vec(), b"2".to_vec()),
            (b"sys:log".to_vec(), b"3".to_vec()),
        ],
        b"user:",
        b"u",
    );

    // Whole-tree move: single-key tree (empty destination, no connector).
    assert_move_op_matches_model(&[(b"only".to_vec(), b"v".to_vec())], b"only", b"next");

    // Whole-tree move: every key shares `from` (re-skin the branch root, no
    // connector, e_src == 0).
    assert_move_op_matches_model(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"abd".to_vec(), b"2".to_vec()),
        ],
        b"ab",
        b"xy",
    );

    // `from` ends mid-skip of an internal branch.
    assert_move_op_matches_model(
        &[
            (b"abc".to_vec(), b"1".to_vec()),
            (b"abd".to_vec(), b"2".to_vec()),
            (b"z".to_vec(), b"3".to_vec()),
        ],
        b"ab",
        b"xy",
    );

    // `from` ends at a (byte-aligned) non-root branch decision.
    assert_move_op_matches_model(
        &[
            (vec![0x61, 0x00], b"1".to_vec()),
            (vec![0x61, 0x80], b"2".to_vec()),
            (b"z".to_vec(), b"3".to_vec()),
        ],
        &[0x61],
        &[0x62],
    );

    // `from` shares leading bits with `to`: the splice routes through the
    // post-detach survivor, not the pre-detach source branch.
    assert_move_op_matches_model(
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

    // All-0xFF source prefix is just another locus.
    assert_move_op_matches_model(
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
fn mrt_move_prefix_rejections() {
    let four = [
        (b"aa".to_vec(), b"1".to_vec()),
        (b"ab".to_vec(), b"2".to_vec()),
        (b"za".to_vec(), b"3".to_vec()),
        (b"zb".to_vec(), b"4".to_vec()),
    ];

    // Absent source prefix.
    assert_move_op_matches_model(&four, b"m", b"n");
    // Destination not empty (`za`,`zb` survive under `z`).
    assert_move_op_matches_model(&four, b"a", b"z");
    // Shortening can still reject when `to` is a non-empty surviving prefix.
    assert_move_op_matches_model(&four, b"aa", b"a");
    // Stateless: equal prefixes.
    assert_move_op_matches_model(&four, b"aa", b"aa");

    // Lengthening overflows: `to` itself is within the key cap, but appending the
    // moved suffix would exceed MAX_KEY_LEN.
    let mut max_key = vec![0x10; MAX_KEY_LEN];
    max_key[1] = 0x20;
    let overflow_to = vec![0x30; MAX_KEY_LEN];
    assert_move_op_matches_model(&[(max_key, b"deep".to_vec())], &[0x10], &overflow_to);

    // Hidden-suffix overflow: a shallow source prefix captures a deep subtree.
    // Checking only `to.len()` would accept this; the committed depth rejects it.
    let mut deep_a = vec![0x40; MAX_KEY_LEN];
    deep_a[1] = 0x00;
    let mut deep_b = vec![0x40; MAX_KEY_LEN];
    deep_b[1] = 0x80;
    let hidden_overflow_to = vec![0x60; MAX_KEY_LEN - 1];
    assert_move_op_matches_model(
        &[
            (deep_a, b"a".to_vec()),
            (deep_b, b"b".to_vec()),
            (vec![0x20], b"survivor".to_vec()),
        ],
        &[0x40],
        &hidden_overflow_to,
    );

    let mut verify_key = vec![0x70; MAX_KEY_LEN];
    verify_key[1] = 0x10;
    let verify_from = vec![0x70];
    let verify_to = vec![0x90; MAX_KEY_LEN];
    let verify_mrt = build_mrt(&[(verify_key, b"deep".to_vec())]);
    let forged_trace = Trace(verify_mrt.checkpoint().root.clone());
    let forged_steps = vec![Step::MovePrefix {
        from: verify_from,
        to: verify_to,
    }];
    let root = verify_mrt.root_hash();
    // The move would overflow the max key length; replay must reject it.
    assert!(mrt_ts::replay_trace(&forged_trace, root, &forged_steps, root).is_err());

    // Prefix-free violation: an existing key (`m`) becomes a byte-prefix of `to`.
    let prefix_free = [
        (b"m".to_vec(), b"1".to_vec()),
        (b"zz".to_vec(), b"2".to_vec()),
    ];
    assert_move_op_matches_model(&prefix_free, b"zz", b"ma");

    // Empty tree: any source prefix is absent.
    assert_move_op_matches_model(&[], b"a", b"b");
}

#[test]
fn mrt_move_prefix_interleaved_with_reads_and_writes() {
    // A move is atomic across the transcript with surrounding read/write steps.
    let entries = vec![
        (b"user:alice".to_vec(), b"1".to_vec()),
        (b"user:bob".to_vec(), b"2".to_vec()),
        (b"sys:cfg".to_vec(), b"3".to_vec()),
        (b"zzz".to_vec(), b"4".to_vec()),
    ];
    let steps = vec![
        Step::Read(vec![ReadOp::Prefix(b"user:".to_vec())]),
        Step::Write(vec![put(b"user:carol", b"9")]),
        Step::MovePrefix {
            from: b"user:".to_vec(),
            to: b"acct:".to_vec(),
        },
        Step::Read(vec![
            ReadOp::Prefix(b"acct:".to_vec()),
            ReadOp::Key(b"user:alice".to_vec()),
        ]),
        Step::Write(vec![put(b"acct:dave", b"7")]),
        Step::MovePrefix {
            from: b"acct:".to_vec(),
            to: b"org01".to_vec(),
        },
        Step::Read(vec![ReadOp::Range {
            start: b"a".to_vec(),
            end: b"zzzz".to_vec(),
        }]),
    ];
    assert_mrt_move_transcript(&entries, &steps);
}

/// A valid `(from, to)` for the current model, or `None` if no fresh same-length
/// destination could be found in a few tries. The broad arbitrary-length
/// candidate fuzz below covers length-changing successes/rejections; this
/// interleaved transcript generator stays length-preserving so later random
/// writes continue to draw from a fixed-length prefix-free key pool.
fn random_valid_move(rng: &mut SmallRng, model: &Model) -> Option<(Vec<u8>, Vec<u8>)> {
    if model.is_empty() {
        return None;
    }
    let keys: Vec<Vec<u8>> = model.keys().cloned().collect();
    for _ in 0..8 {
        let key = &keys[rng.gen_range(0..keys.len())];
        let from_len = rng.gen_range(1..=key.len());
        let from = key[..from_len].to_vec();
        for _ in 0..8 {
            let to = random_bytes(rng, from_len);
            if model_move_prefix(model, &from, &to).is_some() {
                return Some((from, to));
            }
        }
    }
    None
}

fn random_candidate_to_len(rng: &mut SmallRng, from_len: usize, force_overflow: bool) -> usize {
    if force_overflow {
        return rng.gen_range(MAX_KEY_LEN.saturating_sub(2)..=MAX_KEY_LEN);
    }
    match rng.gen_range(0..20) {
        0 => from_len,
        1 => rng.gen_range(0..=from_len.saturating_add(2).min(6)),
        2 => rng.gen_range(0..=3),
        3 => rng.gen_range(4..=8),
        4 => rng.gen_range(MAX_KEY_LEN.saturating_sub(2)..=MAX_KEY_LEN),
        _ => rng.gen_range(0..=from_len.saturating_add(2).min(6)),
    }
}

/// A candidate `(from, to)` that may be valid or invalid — biased toward existing
/// prefixes, but with arbitrary destination lengths (including near-MAX_KEY_LEN)
/// so success/reject agreement covers lengthening, shortening, and overflow.
fn random_move_candidate(rng: &mut SmallRng, model: &Model) -> (Vec<u8>, Vec<u8>) {
    let force_overflow = rng.gen_bool(0.05);
    let from = if !model.is_empty() && rng.gen_bool(0.7) {
        let keys: Vec<Vec<u8>> = model.keys().cloned().collect();
        let key = keys[rng.gen_range(0..keys.len())].clone();
        let max_from_len = if force_overflow && key.len() > 1 {
            key.len() - 1
        } else {
            key.len()
        };
        let from_len = rng.gen_range(1..=max_from_len);
        key[..from_len].to_vec()
    } else {
        let from_len = rng.gen_range(1..=3);
        random_bytes(rng, from_len)
    };
    let to_len = random_candidate_to_len(rng, from.len(), force_overflow);
    (from, random_bytes(rng, to_len))
}

fn random_steps_with_moves(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    start_model: &Model,
    max_batches: usize,
    max_ops_per_batch: usize,
) -> Vec<Step> {
    let step_count = rng.gen_range(1..=max_batches);
    let mut model = start_model.clone();
    let mut steps = Vec::with_capacity(step_count);

    for step in 0..step_count {
        match rng.gen_range(0..3) {
            0 => {
                let read_count = rng.gen_range(1..=3);
                let reads = (0..read_count)
                    .map(|_| random_read_op(rng, keys, &model))
                    .collect();
                steps.push(Step::Read(reads));
            }
            1 => {
                if let Some((from, to)) = random_valid_move(rng, &model) {
                    apply_move_prefix_to_model(&mut model, &from, &to);
                    steps.push(Step::MovePrefix { from, to });
                } else {
                    let ops = random_write_ops(rng, keys, &model, max_ops_per_batch, step as u8);
                    apply_writes_to_model(&mut model, &ops);
                    steps.push(Step::Write(ops));
                }
            }
            _ => {
                let ops = random_write_ops(rng, keys, &model, max_ops_per_batch, step as u8);
                apply_writes_to_model(&mut model, &ops);
                steps.push(Step::Write(ops));
            }
        }
    }

    steps
}

#[test]
fn mrt_move_prefix_differential() {
    for seed in 0..200u64 {
        // Mix in a tag so this walks a different RNG stream than the no-move fuzz.
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x4D4F_5645_5F33);
        let (keys, entries) = random_initial_entries(&mut rng);
        let model = model_from_entries(&entries);

        // Valid interleaved transcript (reads / writes / valid moves): trace,
        // verify, live-replay, and model all agree.
        let steps = random_steps_with_moves(&mut rng, &keys, &model, 8, 6);
        assert_mrt_move_transcript(&entries, &steps);

        // Hammer the move op (valid + invalid candidates) against the oracle to
        // exercise both success and every precondition rejection.
        for _ in 0..12 {
            let (from, to) = random_move_candidate(&mut rng, &model);
            assert_move_op_matches_model(&entries, &from, &to);
        }
    }
}

#[test]
fn mrt_move_prefix_proof_rejects_pruned_collapse_survivor() {
    use super::tree::{MrtNode, MrtNodeInner};

    // Move "ma": the {ma,mb} branch collapses onto its "mb" survivor, while the
    // destination "qz" splices on the y-side — so the splice never descends through
    // the survivor and cannot catch it being pruned. A crafted proof prunes the
    // survivor root and sets `expected_end_root` to the bogus root the unsound
    // (survivor-not-re-skipped) replay would produce; a sound verifier must reject.
    let entries = [
        (b"ma".to_vec(), b"1".to_vec()),
        (b"mb".to_vec(), b"2".to_vec()),
        (b"ya".to_vec(), b"3".to_vec()),
        (b"yb".to_vec(), b"4".to_vec()),
    ];
    let mrt = build_mrt(&entries);
    let steps = vec![Step::MovePrefix {
        from: b"ma".to_vec(),
        to: b"qz".to_vec(),
    }];
    let snapshot = mrt.checkpoint();
    let trace = mrt_ts::create_trace(&snapshot, &steps).unwrap();
    let start = snapshot.root_hash();
    let end = mrt_ts::root_after_writes(&snapshot, &steps);
    mrt_ts::replay_trace(&trace, start, &steps, end).unwrap(); // honest trace verifies

    // Prune the "mb" survivor (root.left's right child) in the sparse tree.
    let root = trace.0.as_ref().unwrap();
    let MrtNode::Branch {
        skip: root_skip,
        left,
        right,
    } = root.node()
    else {
        panic!("expected a branch root");
    };
    let MrtNode::Branch {
        skip: left_skip,
        left: ma_child,
        right: mb_child,
    } = left.node()
    else {
        panic!("expected an ma/mb branch on the left");
    };
    let tampered_left = MrtNodeInner::branch(
        left_skip.clone(),
        ma_child.clone(),
        // Carry the survivor's real depth so the tampered tree still authenticates
        // against the start root — the tamper is the missing materialization.
        MrtNodeInner::pruned(mb_child.hash(), mb_child.depth_below()),
    );
    let tampered_sparse = Trace(Some(MrtNodeInner::branch(
        root_skip.clone(),
        tampered_left,
        right.clone(),
    )));

    // The bogus end root an unsound replay (survivor kept, not re-skipped) yields —
    // the host `move_prefix` reproduces it because the splice routes away from the
    // pruned survivor and so never errors on it.
    let bogus = super::tree::move_prefix(tampered_sparse.0.clone(), b"ma", b"qz").unwrap();

    // Self-consistent for an unsound verifier (the tampered tree authenticates
    // against the honest start root, and the bogus end root matches what an
    // unsound replay yields), yet the sound verifier rejects the pruned survivor.
    assert!(mrt_ts::replay_trace(&tampered_sparse, start, &steps, bogus.hash()).is_err());
}

#[test]
fn move_prefix_proof_rejects_forged_stub_depth() {
    use super::tree::{MrtNode, MrtNodeInner};

    // `depth_below` is *authenticated*, not advisory: it is committed in the parent
    // branch's hash. The attack it would otherwise enable is understating a pruned
    // stub's depth so a lengthening move's overflow check can't see a deep hidden
    // subtree. But understating it changes the parent hash, so the sparse tree no
    // longer authenticates against the honest `expected_start_root` and is rejected
    // at the start-root check — before the move (or any step) even runs.
    let entries = [
        (b"ma".to_vec(), b"1".to_vec()),
        (b"mb".to_vec(), b"2".to_vec()),
        (b"ya".to_vec(), b"3".to_vec()),
    ];
    let mrt = build_mrt(&entries);
    let honest_root = mrt.root_hash();
    let root = mrt.checkpoint().root.expect("non-empty tree");

    let MrtNode::Branch { skip, left, right } = root.node() else {
        panic!("expected a branch root");
    };
    // Prune the right child to a stub carrying a *forged* (understated) depth_below.
    let real_depth = right.depth_below();
    let forged_depth = if real_depth == 0 { 1 } else { 0 };
    let tampered = Trace(Some(MrtNodeInner::branch(
        skip.clone(),
        left.clone(),
        MrtNodeInner::pruned(right.hash(), forged_depth),
    )));
    // The forged depth alone changes the branch hash → the root no longer matches.
    assert_ne!(
        tampered.0.as_ref().unwrap().hash(),
        honest_root,
        "depth_below must be bound by the branch hash"
    );

    // With the forged depth, the tree no longer authenticates against the honest
    // start root, so replay is rejected at the start-root check (no steps needed).
    let no_steps: Vec<Step> = Vec::new();
    assert!(mrt_ts::replay_trace(&tampered, honest_root, &no_steps, honest_root).is_err());
}

/// The headline trace-size gate, parameterized on the destination prefix so the
/// same large-hidden-subtree scenario can be driven through an *equal-length* move
/// and a *lengthening* one. `from` is 2 bytes; the moved keys form a 256-leaf
/// subtree that the proof must NOT reveal interior-of. Asserts the proof verifies,
/// its end root matches the naive model, and `count_revealed` stays O(depth) and
/// sublinear in the moved-subtree size.
fn assert_move_trace_depth_bounded(from: &[u8], to: &[u8]) {
    let moved_keys = 256usize;
    let mut model = Model::new();

    for i in 0..moved_keys {
        let mut key = from.to_vec();
        key.push(i as u8);
        key.push((255 - i) as u8);
        model.insert(key, vec![i as u8]);
    }
    // A few surviving keys force real source collapse and destination splice
    // paths, while staying clear of the destination prefix.
    model.insert(vec![0x10, 0x00, 0x00, 0x00], b"lo".to_vec());
    model.insert(vec![0x40, 0x20, 0x00, 0x00], b"src-survivor".to_vec());
    model.insert(vec![0xc0, 0x00, 0x00, 0x00], b"hi".to_vec());

    let entries = model_entries(&model);
    let mrt = build_mrt(&entries);
    let steps = vec![Step::MovePrefix {
        from: from.to_vec(),
        to: to.to_vec(),
    }];

    let snapshot = mrt.checkpoint();
    let trace = mrt_ts::create_trace(&snapshot, &steps).unwrap();
    let start = snapshot.root_hash();
    let end = mrt_ts::root_after_writes(&snapshot, &steps);
    mrt_ts::replay_trace(&trace, start, &steps, end).unwrap();

    let expected_model = model_move_prefix(&model, from, to).expect("valid move");
    let expected = build_mrt(&model_entries(&expected_model));
    assert_eq!(end, expected.root_hash());

    let revealed = count_revealed_mrt(&trace);
    let max_key_depth_bits = entries
        .iter()
        .map(|(key, _)| key.len() * 8)
        .max()
        .unwrap_or(0);
    let depth_bound = 3 * max_key_depth_bits;

    assert!(
        revealed < depth_bound,
        "move_prefix {:?}->{:?} revealed {} nodes for {} moved keys; \
         expected < 3 * depth ({})",
        from,
        to,
        revealed,
        moved_keys,
        depth_bound
    );
    assert!(
        revealed < moved_keys / 4,
        "move_prefix {:?}->{:?} trace should stay sublinear in subtree size: \
         revealed {} nodes for {} moved keys",
        from,
        to,
        revealed,
        moved_keys
    );
}

#[test]
fn move_prefix_trace_is_depth_bounded() {
    // Equal-length relocation over a 256-leaf hidden subtree.
    assert_move_trace_depth_bounded(&[0x40, 0x10], &[0x80, 0x10]);
}

#[test]
fn move_prefix_lengthening_trace_is_depth_bounded() {
    // Same large hidden subtree, but a *lengthening* destination (2 -> 3 bytes).
    // The committed `depth_below` lets the O(1) overflow check read the subtree's
    // deepest key straight off the captured root — so lengthening adds NO interior
    // reveals and the trace stays O(depth), exactly like the equal-length move.
    assert_move_trace_depth_bounded(&[0x40, 0x10], &[0x80, 0x10, 0x00]);
}

#[test]
#[ignore]
fn fuzz_move_prefix_differential() {
    // MRT-only long fuzz that *does* emit `MovePrefix` steps (the cross-backend
    // `fuzz_tracer_differential` can't — it replays the same steps on the AVL
    // backend, which rejects moves). Same env interface:
    //   FUZZ_SEED=0 FUZZ_ITERS=4000 cargo test --release --lib \
    //     mrt::cross_backend_tests::fuzz_move_prefix_differential -- --ignored
    let start_seed = require_env("FUZZ_SEED");
    let iterations = require_env("FUZZ_ITERS");
    let max_transcripts = env_usize("FUZZ_MAX_TRANSCRIPTS", 10);
    let max_batches = env_usize("FUZZ_MAX_BATCHES", 10);
    let max_ops_per_batch = env_usize("FUZZ_MAX_OPS_PER_BATCH", 10);
    let move_candidates = env_usize("FUZZ_MOVE_CANDIDATES", 12);
    let progress_interval = env_usize_allow_zero("FUZZ_PROGRESS_INTERVAL", 10);

    for iter in 0..iterations {
        let seed = start_seed.wrapping_add(iter);
        let mut rng = SmallRng::seed_from_u64(seed);
        let transcript_count = rng.gen_range(1..=max_transcripts);

        for transcript in 0..transcript_count {
            let (keys, entries) = random_initial_entries(&mut rng);
            let model = model_from_entries(&entries);
            let steps =
                random_steps_with_moves(&mut rng, &keys, &model, max_batches, max_ops_per_batch);
            // Precompute candidates from the same stream for determinism, so the
            // catch_unwind closure borrows only `Unwind`-safe data.
            let candidates: Vec<(Vec<u8>, Vec<u8>)> = (0..move_candidates)
                .map(|_| random_move_candidate(&mut rng, &model))
                .collect();

            let result = std::panic::catch_unwind(|| {
                assert_mrt_move_transcript(&entries, &steps);
                for (from, to) in &candidates {
                    assert_move_op_matches_model(&entries, from, to);
                }
            });
            if let Err(payload) = result {
                eprintln!(
                    "fuzz_move_prefix_differential failed: seed {seed:#x} transcript {transcript}"
                );
                eprintln!("entries: {entries:?}");
                eprintln!("steps: {steps:?}");
                eprintln!("candidates: {candidates:?}");
                std::panic::resume_unwind(payload);
            }
        }

        if progress_interval > 0 && (iter + 1) % progress_interval as u64 == 0 {
            eprintln!(
                "fuzz_move_prefix_differential: completed {}/{} iterations; last seed {seed:#x}",
                iter + 1,
                iterations
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 4: verify-path symmetry
//
// The two backends expose an identical verify surface, so a cfg-switched
// consumer can `use merk::mrt as backend` / `use merk::avl as backend` and share
// the whole verify path. The trait below is a *compile-time* proof of that
// parity — it only compiles because `crate::{avl,mrt}::TraceVerifier` have
// matching signatures on `decode_trace`/`verify_root`/`root_hash`/`get`/
// `collect_range`/`collect_prefix`/`replay_batch_ops_in_place`/`move_prefix`.
// Construction differs (`avl::from_trace` is fallible), so the shared entry
// point is `decode_trace`.
// -----------------------------------------------------------------------------

trait VerifyBackend: Sized {
    fn decode_trace(bytes: &[u8]) -> crate::error::Result<Self>;
    fn verify_root(&mut self, expected: Hash) -> crate::error::Result<()>;
    fn root_hash(&mut self) -> crate::error::Result<Hash>;
    fn get(&self, key: &[u8]) -> crate::error::Result<Option<Vec<u8>>>;
    fn collect_range(&self, start: &[u8], end: Option<&[u8]>) -> crate::error::Result<Entries>;
    fn collect_prefix(&self, prefix: &[u8]) -> crate::error::Result<Entries>;
    fn replay_batch_ops(&mut self, ops: &[BatchOp]) -> crate::error::Result<()>;
    fn replay_batch_ops_in_place(&mut self, ops: &[BatchOp]) -> crate::error::Result<()>;
    fn move_prefix(&mut self, from: &[u8], to: &[u8]) -> crate::error::Result<()>;
}

macro_rules! impl_verify_backend {
    ($t:ty) => {
        impl VerifyBackend for $t {
            fn decode_trace(bytes: &[u8]) -> crate::error::Result<Self> {
                Self::decode_trace(bytes)
            }
            fn verify_root(&mut self, expected: Hash) -> crate::error::Result<()> {
                Self::verify_root(self, expected)
            }
            fn root_hash(&mut self) -> crate::error::Result<Hash> {
                Self::root_hash(self)
            }
            fn get(&self, key: &[u8]) -> crate::error::Result<Option<Vec<u8>>> {
                Self::get(self, key)
            }
            fn collect_range(
                &self,
                start: &[u8],
                end: Option<&[u8]>,
            ) -> crate::error::Result<Entries> {
                Self::collect_range(self, start, end)
            }
            fn collect_prefix(&self, prefix: &[u8]) -> crate::error::Result<Entries> {
                Self::collect_prefix(self, prefix)
            }
            fn replay_batch_ops(&mut self, ops: &[BatchOp]) -> crate::error::Result<()> {
                Self::replay_batch_ops(self, ops)
            }
            fn replay_batch_ops_in_place(&mut self, ops: &[BatchOp]) -> crate::error::Result<()> {
                Self::replay_batch_ops_in_place(self, ops)
            }
            fn move_prefix(&mut self, from: &[u8], to: &[u8]) -> crate::error::Result<()> {
                Self::move_prefix(self, from, to)
            }
        }
    };
}

impl_verify_backend!(crate::avl::TraceVerifier);
impl_verify_backend!(crate::mrt::TraceVerifier);

/// Backend-agnostic verify driver — the exact sequence a consumer runs over a
/// `use merk::X as backend` import.
fn agnostic_replay<V: VerifyBackend>(
    trace: &[u8],
    start_root: Hash,
    steps: &[Step],
    end_root: Hash,
) -> crate::error::Result<()> {
    let mut v = V::decode_trace(trace)?;
    v.verify_root(start_root)?;
    for step in steps {
        match step {
            Step::Read(reads) => {
                for op in reads {
                    match op {
                        ReadOp::Key(key) => {
                            v.get(key)?;
                        }
                        ReadOp::Range { start, end } => {
                            v.collect_range(start, Some(end))?;
                        }
                        ReadOp::Prefix(prefix) => {
                            v.collect_prefix(prefix)?;
                        }
                    }
                }
            }
            Step::Write(ops) => v.replay_batch_ops_in_place(ops)?,
            // MRT-only step; not exercised by this shared driver.
            Step::MovePrefix { .. } => {}
        }
    }
    v.verify_root(end_root)?;
    // `root_hash` is part of the shared surface too; confirm it agrees.
    assert_eq!(v.root_hash()?, end_root);
    Ok(())
}

fn replay_noop_batch<V: VerifyBackend>(v: &mut V) -> crate::error::Result<()> {
    v.replay_batch_ops(&[])
}

fn replay_move_prefix<V: VerifyBackend>(
    v: &mut V,
    from: &[u8],
    to: &[u8],
) -> crate::error::Result<()> {
    v.move_prefix(from, to)
}

#[test]
fn verify_surface_is_backend_agnostic() {
    let entries = vec![
        (b"a".to_vec(), b"1".to_vec()),
        (b"c".to_vec(), b"3".to_vec()),
        (b"e".to_vec(), b"5".to_vec()),
    ];
    let steps = vec![
        Step::Read(vec![ReadOp::Key(b"c".to_vec())]),
        Step::Write(vec![put(b"c", b"updated")]),
    ];

    // Build each backend's trace + start/end roots from live trees.
    let avl = build_avl(&entries);
    let mrt = build_mrt(&entries);
    let avl_trace = avl_ts::create_trace(&avl.checkpoint(), &steps).unwrap();
    let mrt_trace = mrt_ts::create_trace(&mrt.checkpoint(), &steps).unwrap();
    let avl_start = avl_trace.hash();
    let mrt_start = mrt.root_hash();

    let avl_live = build_avl(&entries);
    let mrt_live = build_mrt(&entries);
    apply_writes_to_live(&avl_live, &mrt_live, &steps);
    let avl_end = avl_live.root_hash();
    let mrt_end = mrt_live.root_hash();

    // Encode each trace to the wire (the public `Trace::encode`, inverse of
    // `decode_trace`), then drive both through the *same* generic verify path.
    let avl_bytes = avl_trace.encode().unwrap();
    let mrt_bytes = mrt_trace.encode().unwrap();

    agnostic_replay::<crate::avl::TraceVerifier>(&avl_bytes, avl_start, &steps, avl_end).unwrap();
    agnostic_replay::<crate::mrt::TraceVerifier>(&mrt_bytes, mrt_start, &steps, mrt_end).unwrap();
}

#[test]
fn verify_surface_exposes_in_place_replay_and_move_prefix_on_both_backends() {
    let empty_avl = build_avl(&[]);
    let avl_trace = avl_ts::create_trace(&empty_avl.checkpoint(), &[]).unwrap();
    let avl_bytes = avl_trace.encode().unwrap();
    let mut avl = <crate::avl::TraceVerifier as VerifyBackend>::decode_trace(&avl_bytes).unwrap();
    replay_noop_batch(&mut avl).unwrap();
    let err = replay_move_prefix(&mut avl, b"user:", b"acct:").unwrap_err();
    assert!(matches!(
        err,
        Error::Unsupported(UnsupportedFeature::MovePrefix)
    ));

    let entries = vec![
        (b"sys:root".to_vec(), b"0".to_vec()),
        (b"user:alice".to_vec(), b"1".to_vec()),
        (b"user:bob".to_vec(), b"2".to_vec()),
    ];
    let steps = vec![Step::MovePrefix {
        from: b"user:".to_vec(),
        to: b"acct:".to_vec(),
    }];
    let traced = build_mrt(&entries);
    let start_root = traced.root_hash();
    let trace = mrt_ts::create_trace(&traced.checkpoint(), &steps).unwrap();
    let bytes = trace.encode().unwrap();

    let mut mrt = <crate::mrt::TraceVerifier as VerifyBackend>::decode_trace(&bytes).unwrap();
    mrt.verify_root(start_root).unwrap();
    replay_noop_batch(&mut mrt).unwrap();
    replay_move_prefix(&mut mrt, b"user:", b"acct:").unwrap();

    let expected = build_mrt(&entries);
    expected
        .move_prefix(b"user:".to_vec(), b"acct:".to_vec())
        .unwrap();
    assert_eq!(mrt.root_hash().unwrap(), expected.root_hash());
}
