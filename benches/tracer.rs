use std::sync::OnceLock;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use merk::avl;
use merk::mrt;
use merk::tracer::{ReadOp, TraceInterface};
use merk::{Hash, Op};

/// A bench-local write op. The bench workloads only ever do point Put/Delete —
/// no DeleteRange/MovePrefix.
#[derive(Clone)]
enum WriteOp {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
}

/// A bench transcript step (the trace API is the op-by-op handle; benches keep a
/// step list and drive it through the public `TraceRecorder`/`TraceReplayer`).
#[derive(Clone)]
enum Step {
    Read(Vec<ReadOp>),
    Write(Vec<WriteOp>),
}

/// Drive a recorder/replayer over the steps, one op at a time in issue order.
fn drive<H: TraceInterface>(handle: &mut H, steps: &[Step]) -> merk::Result<()> {
    for step in steps {
        match step {
            Step::Read(ops) => {
                for op in ops {
                    match op {
                        ReadOp::Key(key) => {
                            handle.get(key)?;
                        }
                        ReadOp::Range { start, end } => {
                            handle.get_range(start, end)?;
                        }
                        ReadOp::Prefix(prefix) => {
                            handle.get_prefix(prefix)?;
                        }
                    }
                }
            }
            Step::Write(ops) => {
                for op in ops {
                    match op {
                        WriteOp::Put(key, value) => handle.put(key, value)?,
                        WriteOp::Delete(key) => handle.delete(key)?,
                    }
                }
            }
        }
    }
    Ok(())
}

/// Build the AVL witness for `steps` over `snapshot` via the public recorder.
fn record_avl(snapshot: &avl::Checkpoint, steps: &[Step]) -> Vec<u8> {
    let mut recorder = avl::TraceRecorder::new(snapshot);
    drive(&mut recorder, steps).expect("AVL bench recording should succeed");
    recorder
        .finalize_trace()
        .expect("AVL bench finalize_trace should succeed")
}

/// Build the MRT witness for `steps` over `snapshot` via the public recorder.
fn record_mrt(snapshot: &mrt::Checkpoint, steps: &[Step]) -> Vec<u8> {
    let mut recorder = mrt::TraceRecorder::new(snapshot);
    drive(&mut recorder, steps).expect("MRT bench recording should succeed");
    recorder
        .finalize_trace()
        .expect("MRT bench finalize_trace should succeed")
}

const PREPOPULATE_ROWS: usize = 100_000;
const PREPOPULATE_LIST_ITEMS: usize = 132;
const OPS_PER_BATCH: usize = 100;

#[derive(Clone, Copy)]
enum LogicalOp {
    Insert,
    Update,
    Delete,
    ListAppend,
    ListInsert,
    ListUpdate,
    ListDelete,
}

struct Workload {
    name: &'static str,
    logical_ops: usize,
    avl_root: avl::Checkpoint,
    mrt_snapshot: mrt::Checkpoint,
    steps: Vec<Step>,
    avl_trace: Vec<u8>,
    avl_start_root: Hash,
    mrt_trace: Vec<u8>,
    mrt_start_root: Hash,
}

struct InitialState {
    avl_root: avl::Checkpoint,
    mrt_snapshot: mrt::Checkpoint,
}

fn workloads() -> &'static [Workload] {
    static WORKLOADS: OnceLock<Vec<Workload>> = OnceLock::new();
    WORKLOADS.get_or_init(build_workloads)
}

fn build_workloads() -> Vec<Workload> {
    let initial = build_initial_state();
    let specs = [
        ("insert_single", LogicalOp::Insert, 1),
        ("insert_10", LogicalOp::Insert, 10),
        ("insert_batch", LogicalOp::Insert, OPS_PER_BATCH),
        ("update_single", LogicalOp::Update, 1),
        ("update_10", LogicalOp::Update, 10),
        ("update_batch", LogicalOp::Update, OPS_PER_BATCH),
        ("delete_single", LogicalOp::Delete, 1),
        ("delete_10", LogicalOp::Delete, 10),
        ("delete_batch", LogicalOp::Delete, OPS_PER_BATCH),
        ("list_append", LogicalOp::ListAppend, 1),
        ("list_append_10", LogicalOp::ListAppend, 10),
        ("list_append_batch", LogicalOp::ListAppend, OPS_PER_BATCH),
        ("list_insert", LogicalOp::ListInsert, 1),
        ("list_insert_10", LogicalOp::ListInsert, 10),
        ("list_insert_batch", LogicalOp::ListInsert, OPS_PER_BATCH),
        ("list_update", LogicalOp::ListUpdate, 1),
        ("list_update_10", LogicalOp::ListUpdate, 10),
        ("list_update_batch", LogicalOp::ListUpdate, OPS_PER_BATCH),
        ("list_delete", LogicalOp::ListDelete, 1),
        ("list_delete_10", LogicalOp::ListDelete, 10),
        ("list_delete_batch", LogicalOp::ListDelete, OPS_PER_BATCH),
    ];

    specs
        .iter()
        .copied()
        .map(|(name, op, logical_ops)| {
            let steps = steps_for(op, logical_ops);
            let avl_trace = record_avl(&initial.avl_root, &steps);
            let avl_start_root = initial.avl_root.root_hash();
            let mrt_trace = record_mrt(&initial.mrt_snapshot, &steps);
            let mrt_start_root = initial.mrt_snapshot.root_hash();

            // Sanity: each trace authenticates against its start root and replays
            // the (externally supplied) steps — the same externalized verify path
            // that `bench_tracer_verify` measures.
            avl_replay_to_root(&avl_trace, avl_start_root, &steps)
                .expect("AVL bench trace should replay");
            mrt_replay_to_root(&mrt_trace, mrt_start_root, &steps)
                .expect("MRT bench trace should replay");

            Workload {
                name,
                logical_ops,
                avl_root: initial.avl_root.clone(),
                mrt_snapshot: initial.mrt_snapshot.clone(),
                steps,
                avl_trace,
                avl_start_root,
                mrt_trace,
                mrt_start_root,
            }
        })
        .collect()
}

fn build_initial_state() -> InitialState {
    let avl = avl::Tree::new();
    let mrt = mrt::Tree::new();
    let mut batch = Vec::with_capacity(1 + PREPOPULATE_ROWS * 2 + PREPOPULATE_LIST_ITEMS + 1);

    batch.push((
        insert_counter_key(),
        Op::Put(u64_value(PREPOPULATE_ROWS as u64)),
    ));
    batch.push((
        list_tail_key(),
        Op::Put(u64_value((PREPOPULATE_LIST_ITEMS - 1) as u64)),
    ));

    for row in 1..=PREPOPULATE_ROWS as u64 {
        batch.push((table_name_key(row), Op::Put(value_bytes(row, 0))));
        batch.push((table_price_key(row), Op::Put(value_bytes(row, 1))));
    }

    for i in 0..PREPOPULATE_LIST_ITEMS as u64 {
        batch.push((list_item_key(i * 1_000), Op::Put(list_value(i, false))));
    }

    // `apply_sorted_batch_ops` is now `pub(crate)`; prepopulate the fixture
    // through the public per-op API instead (order is immaterial for the fixture).
    batch.sort_by(|a, b| a.0.cmp(&b.0));
    for (key, op) in &batch {
        match op {
            Op::Put(value) => {
                avl.put(key.clone(), value.clone())
                    .expect("prepopulate AVL merk");
                mrt.put(key.clone(), value.clone())
                    .expect("prepopulate MRT merk");
            }
            Op::Delete => {
                avl.delete(key.clone()).expect("prepopulate AVL merk");
                mrt.delete(key.clone()).expect("prepopulate MRT merk");
            }
            Op::DeleteRange(end) => {
                avl.delete_range(key.clone(), end.clone())
                    .expect("prepopulate AVL merk");
                mrt.delete_range(key.clone(), end.clone())
                    .expect("prepopulate MRT merk");
            }
        }
    }

    InitialState {
        avl_root: avl.checkpoint(),
        mrt_snapshot: mrt.checkpoint(),
    }
}

fn steps_for(op: LogicalOp, count: usize) -> Vec<Step> {
    match op {
        LogicalOp::Insert => table_insert_steps(count),
        LogicalOp::Update => table_update_steps(count),
        LogicalOp::Delete => table_delete_steps(count),
        LogicalOp::ListAppend => list_append_steps(count),
        LogicalOp::ListInsert => list_insert_steps(count),
        LogicalOp::ListUpdate => list_update_steps(count),
        LogicalOp::ListDelete => list_delete_steps(count),
    }
}

fn table_insert_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    for i in 0..count as u64 {
        let row = PREPOPULATE_ROWS as u64 + 1 + i;
        steps.push(Step::Read(vec![ReadOp::Key(insert_counter_key())]));
        steps.push(Step::Write(vec![
            WriteOp::Put(insert_counter_key(), u64_value(row)),
            WriteOp::Put(table_name_key(row), value_bytes(row, 2)),
            WriteOp::Put(table_price_key(row), value_bytes(row, 3)),
        ]));
    }
    steps
}

fn table_update_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    for row in 1..=count as u64 {
        steps.push(Step::Read(vec![ReadOp::Range {
            start: table_row_start(row),
            end: table_row_end(row),
        }]));
        steps.push(Step::Write(vec![
            WriteOp::Put(table_name_key(row), value_bytes(row, 4)),
            WriteOp::Put(table_price_key(row), value_bytes(row, 5)),
        ]));
    }
    steps
}

fn table_delete_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    for row in 1..=count as u64 {
        steps.push(Step::Read(vec![ReadOp::Range {
            start: table_row_start(row),
            end: table_row_end(row),
        }]));
        steps.push(Step::Write(vec![
            WriteOp::Delete(table_name_key(row)),
            WriteOp::Delete(table_price_key(row)),
        ]));
    }
    steps
}

fn list_append_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    let mut tail = (PREPOPULATE_LIST_ITEMS - 1) as u64 * 1_000;
    for i in 0..count as u64 {
        let next = tail + 1_000;
        steps.push(Step::Read(vec![ReadOp::Key(list_tail_key())]));
        steps.push(Step::Write(vec![
            WriteOp::Put(list_tail_key(), u64_value(next)),
            WriteOp::Put(list_item_key(next), list_value(i, false)),
        ]));
        tail = next;
    }
    steps
}

fn list_insert_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    let mut prev = 16_000u64;
    let next_existing = 17_000u64;
    for i in 0..count as u64 {
        let inserted = 16_001 + i;
        steps.push(Step::Read(vec![
            ReadOp::Key(list_item_key(prev)),
            ReadOp::Key(list_item_key(next_existing)),
        ]));
        steps.push(Step::Write(vec![WriteOp::Put(
            list_item_key(inserted),
            list_value(i, false),
        )]));
        prev = inserted;
    }
    steps
}

fn list_update_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    for i in 0..count as u64 {
        let key = list_item_key(i * 1_000);
        steps.push(Step::Read(vec![ReadOp::Key(key.clone())]));
        steps.push(Step::Write(vec![WriteOp::Put(key, list_value(i, true))]));
    }
    steps
}

fn list_delete_steps(count: usize) -> Vec<Step> {
    let mut steps = Vec::with_capacity(count * 2);
    for i in 0..count as u64 {
        let key = list_item_key(i * 1_000);
        steps.push(Step::Read(vec![ReadOp::Key(key.clone())]));
        steps.push(Step::Write(vec![WriteOp::Delete(key)]));
    }
    steps
}

/// Externalized AVL verify: wrap the trace bytes, authenticate the start root,
/// replay the caller-supplied `steps`, and return the resulting root — mirroring
/// `mrt_replay_to_root` and how the changelog drives an `avl::TraceReplayer`.
fn avl_replay_to_root(trace_bytes: &[u8], start_root: Hash, steps: &[Step]) -> merk::Result<Hash> {
    let mut replayer = avl::TraceReplayer::new_verified(trace_bytes, start_root)?;
    drive(&mut replayer, steps)?;
    replayer.root_hash()
}

/// Externalized MRT verify: wrap the trace bytes, authenticate the start root, replay the
/// caller-supplied `steps` (reads recomputed, writes/moves applied), and return the
/// resulting root — mirroring how the prototype's fast-forward verifier drives a
/// `mrt::TraceReplayer`. (MRT has no self-contained proof; the verifier supplies the steps.)
fn mrt_replay_to_root(trace_bytes: &[u8], start_root: Hash, steps: &[Step]) -> merk::Result<Hash> {
    let mut replayer = mrt::TraceReplayer::new_verified(trace_bytes, start_root)?;
    drive(&mut replayer, steps)?;
    replayer.root_hash()
}

fn insert_counter_key() -> Vec<u8> {
    vec![0x00]
}

fn list_tail_key() -> Vec<u8> {
    vec![0x01, 0x00]
}

fn table_name_key(row: u64) -> Vec<u8> {
    table_col_key(row, 0)
}

fn table_price_key(row: u64) -> Vec<u8> {
    table_col_key(row, 1)
}

fn table_col_key(row: u64, col: u8) -> Vec<u8> {
    let mut key = table_row_start(row);
    key.push(col);
    key
}

fn table_row_start(row: u64) -> Vec<u8> {
    let mut key = vec![0x10];
    key.extend_from_slice(&row.to_be_bytes());
    key
}

fn table_row_end(row: u64) -> Vec<u8> {
    let mut key = table_row_start(row);
    key.push(0xff);
    key
}

fn list_item_key(position: u64) -> Vec<u8> {
    let mut key = vec![0x20];
    key.extend_from_slice(&position.to_be_bytes());
    key
}

fn value_bytes(row: u64, tag: u8) -> Vec<u8> {
    let mut value = Vec::with_capacity(32);
    value.extend_from_slice(&row.to_be_bytes());
    value.push(tag);
    while value.len() < 32 {
        value.push(tag.wrapping_add(value.len() as u8));
    }
    value
}

fn list_value(index: u64, done: bool) -> Vec<u8> {
    let mut value = Vec::with_capacity(32);
    value.extend_from_slice(&index.to_be_bytes());
    value.push(u8::from(done));
    while value.len() < 32 {
        value.push((index as u8).wrapping_add(value.len() as u8));
    }
    value
}

fn u64_value(value: u64) -> Vec<u8> {
    value.to_be_bytes().to_vec()
}

fn bench_tracer_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracer/prove");
    for workload in workloads() {
        group.throughput(Throughput::Elements(workload.logical_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("avl", workload.name),
            workload,
            |b, workload| {
                b.iter(|| {
                    let trace =
                        record_avl(black_box(&workload.avl_root), black_box(&workload.steps));
                    black_box(trace);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mrt", workload.name),
            workload,
            |b, workload| {
                b.iter(|| {
                    let proof = record_mrt(
                        black_box(&workload.mrt_snapshot),
                        black_box(&workload.steps),
                    );
                    black_box(proof);
                });
            },
        );
    }
    group.finish();
}

fn bench_tracer_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracer/verify");
    for workload in workloads() {
        group.throughput(Throughput::Elements(workload.logical_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("avl", workload.name),
            workload,
            |b, workload| {
                b.iter(|| {
                    let root = avl_replay_to_root(
                        black_box(&workload.avl_trace),
                        black_box(workload.avl_start_root),
                        black_box(&workload.steps),
                    )
                    .expect("AVL bench trace should verify");
                    black_box(root);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mrt", workload.name),
            workload,
            |b, workload| {
                b.iter(|| {
                    let root = mrt_replay_to_root(
                        black_box(&workload.mrt_trace),
                        black_box(workload.mrt_start_root),
                        black_box(&workload.steps),
                    )
                    .expect("MRT bench trace should verify");
                    black_box(root);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_tracer_prove, bench_tracer_verify);
criterion_main!(benches);
