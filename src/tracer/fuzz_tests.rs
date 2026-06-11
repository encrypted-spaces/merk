//! Randomized differential fuzz tests for the tracer trace pipeline.
//!
//! Each iteration builds a random AVL tree, generates a random multi-step
//! transcript (interleaved reads, writes, and delete-ranges), produces a
//! witness trace via `create_trace`, replays it through the verifier
//! with [`replay_trace`] (authenticating the start/end roots), and checks that
//! verified read results match a [`BTreeMap`] oracle. The end root is
//! independently cross-checked against [`root_after_writes`] and the live store.
//!
//! The long-running variant (`fuzz_tracer_differential`) is `#[ignore]`d and
//! controlled by environment variables:
//!
//! ```text
//! FUZZ_SEED=0 FUZZ_ITERS=1000 cargo test --release -- --ignored fuzz_tracer_differential
//! ```
//!
//! To reproduce a failure:
//!
//! ```text
//! FUZZ_SEED=<seed> FUZZ_ITERS=1 cargo test --release -- --ignored fuzz_tracer_differential
//! ```

use std::collections::{BTreeMap, HashSet};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::avl::in_memory::InMemoryMerk;
use crate::ops::Op;

use super::{
    prefix_successor, BatchOp, ProvenRead, ReadOp, ReadResults, SMALL_VALUE_INLINE_THRESHOLD,
};
use crate::tracer::test_support::avl::{create_trace, replay_trace, root_after_writes};
use crate::tracer::test_support::Step;

// ---------------------------------------------------------------------------
// Model oracle
// ---------------------------------------------------------------------------

fn model_get(model: &BTreeMap<Vec<u8>, Vec<u8>>, key: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    model
        .get(key)
        .map(|v| vec![(key.to_vec(), v.clone())])
        .unwrap_or_default()
}

fn model_range(
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    start: &[u8],
    end: &[u8],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    model
        .range(start.to_vec()..end.to_vec())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn model_prefix(model: &BTreeMap<Vec<u8>, Vec<u8>>, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    match prefix_successor(prefix) {
        Some(end) => model
            .range(prefix.to_vec()..end)
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        None => model
            .range(prefix.to_vec()..)
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

fn model_read(model: &BTreeMap<Vec<u8>, Vec<u8>>, op: &ReadOp) -> Vec<(Vec<u8>, Vec<u8>)> {
    match op {
        ReadOp::Key(key) => model_get(model, key),
        ReadOp::Range { start, end } => model_range(model, start, end),
        ReadOp::Prefix(prefix) => model_prefix(model, prefix),
    }
}

fn apply_ops_to_model(model: &mut BTreeMap<Vec<u8>, Vec<u8>>, ops: &[BatchOp]) {
    for op in ops {
        match op {
            BatchOp::Put { key, value } => {
                model.insert(key.clone(), value.clone());
            }
            BatchOp::Delete { key } => {
                model.remove(key);
            }
            BatchOp::DeleteRange { start, end } => {
                let to_remove: Vec<Vec<u8>> = model
                    .range(start.clone()..end.clone())
                    .map(|(k, _)| k.clone())
                    .collect();
                for k in to_remove {
                    model.remove(&k);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Live store oracle
// ---------------------------------------------------------------------------

/// Apply a write step to the live store one op at a time, in issue order. AVL is
/// insertion-order sensitive, so the oracle must evolve exactly as the traced
/// path does (a write step is a sequence of single writes, not a sorted batch).
fn apply_write_step_to_live(live: &InMemoryMerk, batch: &[(Vec<u8>, Op)]) -> crate::Result<()> {
    for entry in batch {
        live.apply_sorted_batch_ops_owned(vec![entry.clone()])?;
    }
    Ok(())
}

fn build_live_store(model: &BTreeMap<Vec<u8>, Vec<u8>>) -> InMemoryMerk {
    let merk = InMemoryMerk::new();
    let mut batch: Vec<(Vec<u8>, Op)> = model
        .iter()
        .map(|(k, v)| (k.clone(), Op::Put(v.clone())))
        .collect();
    batch.sort_by(|a, b| a.0.cmp(&b.0));
    if !batch.is_empty() {
        merk.apply_sorted_batch_ops(&batch).unwrap();
    }
    merk
}

// ---------------------------------------------------------------------------
// Random generation
// ---------------------------------------------------------------------------

fn random_value(rng: &mut SmallRng, tag: u8) -> Vec<u8> {
    match rng.gen_range(0..=7) {
        0 => Vec::new(),
        1 => vec![tag],
        2 => vec![rng.gen()],
        3 => vec![rng.gen(), rng.gen()],
        4 => b"value".to_vec(),
        5 => vec![tag, rng.gen(), rng.gen()],
        6 => vec![0xff, tag],
        // Occasionally produce values above the small-value threshold
        _ => (0..SMALL_VALUE_INLINE_THRESHOLD + 4)
            .map(|i| tag.wrapping_add(i as u8))
            .collect(),
    }
}

const DEFAULT_KEY_UNIVERSE_LEN: usize = 40;
const LARGE_BATCH_OP_THRESHOLD: usize = 16;
const MAX_MIXED_DELETE_RANGES_PER_BATCH: usize = 16;
const MAX_RANGE_ONLY_DELETE_RANGES_PER_BATCH: usize = 32;

fn key_universe(rng: &mut SmallRng) -> Vec<Vec<u8>> {
    key_universe_with_len(rng, DEFAULT_KEY_UNIVERSE_LEN)
}

fn key_universe_with_len(rng: &mut SmallRng, target_len: usize) -> Vec<Vec<u8>> {
    let mut keys: Vec<Vec<u8>> = vec![
        vec![0x00],
        vec![0x01],
        vec![0x40],
        vec![0x7f],
        vec![0x80],
        vec![0xc0],
        vec![0xfe],
        vec![0xff],
        b"a".to_vec(),
        b"aa".to_vec(),
        b"ab".to_vec(),
        b"ac".to_vec(),
        b"b".to_vec(),
        b"ba".to_vec(),
        b"bb".to_vec(),
        b"m".to_vec(),
        b"z".to_vec(),
    ];
    while keys.len() < target_len {
        let len = rng.gen_range(1..=3);
        let key: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

fn make_initial_model_with_len(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    target_len: usize,
) -> BTreeMap<Vec<u8>, Vec<u8>> {
    assert!(
        target_len <= keys.len(),
        "target model size must fit in the key universe"
    );

    let mut indexes: Vec<usize> = (0..keys.len()).collect();
    for i in 0..target_len {
        let j = rng.gen_range(i..keys.len());
        indexes.swap(i, j);
    }

    let mut model = BTreeMap::new();
    for index in indexes.into_iter().take(target_len) {
        let key = &keys[index];
        let value = random_value(rng, key.first().copied().unwrap_or(0));
        model.insert(key.clone(), value);
    }
    model
}

fn make_initial_model(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    inclusion_probability: f64,
) -> BTreeMap<Vec<u8>, Vec<u8>> {
    let mut model = BTreeMap::new();
    for key in keys {
        if rng.gen_bool(inclusion_probability) {
            let value = random_value(rng, key.first().copied().unwrap_or(0));
            model.insert(key.clone(), value);
        }
    }
    // Ensure at least a few keys
    if model.len() < 3 {
        for key in keys.iter().take(6) {
            let value = random_value(rng, key.first().copied().unwrap_or(0));
            model.insert(key.clone(), value);
        }
    }
    model
}

fn choose_existing_key(rng: &mut SmallRng, model: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<u8> {
    let keys: Vec<&Vec<u8>> = model.keys().collect();
    if keys.is_empty() {
        return vec![0xf0, rng.gen()];
    }
    keys[rng.gen_range(0..keys.len())].clone()
}

fn choose_missing_key(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> Vec<u8> {
    let missing: Vec<&Vec<u8>> = keys.iter().filter(|k| !model.contains_key(*k)).collect();
    if missing.is_empty() {
        vec![0xf0, rng.gen(), rng.gen()]
    } else {
        missing[rng.gen_range(0..missing.len())].clone()
    }
}

fn random_probe_key(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> Vec<u8> {
    if rng.gen_bool(0.5) && !model.is_empty() {
        choose_existing_key(rng, model)
    } else {
        choose_missing_key(rng, keys, model)
    }
}

fn random_read_op(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> ReadOp {
    match rng.gen_range(0..=2) {
        0 => ReadOp::Key(random_probe_key(rng, keys, model)),
        1 => {
            let mut lo = random_probe_key(rng, keys, model);
            let mut hi = random_probe_key(rng, keys, model);
            if rng.gen_bool(0.2) {
                hi = lo.clone();
            } else if lo > hi {
                std::mem::swap(&mut lo, &mut hi);
            }
            ReadOp::Range { start: lo, end: hi }
        }
        _ => {
            let key = random_probe_key(rng, keys, model);
            if key.is_empty() {
                ReadOp::Prefix(Vec::new())
            } else {
                let prefix_len = rng.gen_range(1..=key.len());
                ReadOp::Prefix(key[..prefix_len].to_vec())
            }
        }
    }
}

fn random_write_ops(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    op_count: usize,
) -> Vec<BatchOp> {
    if keys.len() >= 2 {
        match rng.gen_range(0..=9) {
            0..=3 if op_count >= 2 => {
                return random_mixed_range_write_ops(rng, keys, model, op_count);
            }
            4..=5 => {
                return random_range_only_write_ops(rng, keys, model, op_count);
            }
            _ => {}
        }
    }

    random_point_write_ops(rng, keys, model, op_count)
}

fn random_point_write_ops(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    op_count: usize,
) -> Vec<BatchOp> {
    let mut ops = Vec::new();
    let mut keys_used = HashSet::new();

    for _ in 0..op_count {
        match rng.gen_range(0..=2) {
            0 | 1 => {
                // Put (existing or new)
                let key = if rng.gen_bool(0.5) && !model.is_empty() {
                    choose_existing_key(rng, model)
                } else {
                    choose_missing_key(rng, keys, model)
                };
                if keys_used.contains(&key) {
                    continue;
                }
                let value = random_value(rng, key.first().copied().unwrap_or(0));
                keys_used.insert(key.clone());
                ops.push(BatchOp::Put { key, value });
            }
            2 => {
                // Delete
                if model.is_empty() {
                    continue;
                }
                let key = choose_existing_key(rng, model);
                if keys_used.contains(&key) {
                    continue;
                }
                keys_used.insert(key.clone());
                ops.push(BatchOp::Delete { key });
            }
            _ => unreachable!(),
        }
    }

    // Sort point ops by key for batch validity
    ops.sort_by(|a, b| a.key().cmp(b.key()));
    ops.dedup_by(|a, b| a.key() == b.key());
    ops
}

fn random_mixed_range_write_ops(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    op_count: usize,
) -> Vec<BatchOp> {
    let mut ops = Vec::new();
    let mut range_starts = HashSet::new();
    let mut point_keys = HashSet::new();

    // Keep at least one point op in this path. Range-only batches are useful,
    // but the "mixed" path should exercise segment transitions in one batch.
    let max_ranges = op_count
        .saturating_sub(1)
        .clamp(1, MAX_MIXED_DELETE_RANGES_PER_BATCH);
    let range_count = rng.gen_range(1..=max_ranges);
    for (start, end) in random_delete_ranges(rng, keys, model, range_count) {
        if range_starts.insert(start.clone()) {
            ops.push(BatchOp::DeleteRange { start, end });
        }
    }

    let point_budget = op_count.saturating_sub(ops.len());
    for _ in 0..point_budget {
        let key = random_mixed_point_key(rng, keys, model, &ops);
        if !point_keys.insert(key.clone()) {
            continue;
        }

        let op = if rng.gen_bool(0.65) {
            BatchOp::Put {
                value: random_value(rng, key.first().copied().unwrap_or(0)),
                key,
            }
        } else {
            BatchOp::Delete { key }
        };
        ops.push(op);
    }

    sort_valid_batch_ops(&mut ops);
    ops
}

fn random_range_only_write_ops(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    op_count: usize,
) -> Vec<BatchOp> {
    let mut ops = Vec::new();
    let mut range_starts = HashSet::new();
    let max_ranges = op_count.clamp(1, MAX_RANGE_ONLY_DELETE_RANGES_PER_BATCH);
    let range_count = rng.gen_range(1..=max_ranges);

    for (start, end) in random_delete_ranges(rng, keys, model, range_count) {
        if range_starts.insert(start.clone()) {
            ops.push(BatchOp::DeleteRange { start, end });
        }
    }

    sort_valid_batch_ops(&mut ops);
    ops
}

fn random_delete_ranges(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    count: usize,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let boundary_keys = delete_range_boundary_candidates(rng, keys, model);
    if boundary_keys.len() < 2 {
        return Vec::new();
    }

    let mut ranges = if count >= 2 && boundary_keys.len() >= 4 {
        match rng.gen_range(0..=3) {
            0 => partial_overlap_range_pair(rng, &boundary_keys),
            1 => contained_range_pair(rng, &boundary_keys),
            2 if boundary_keys.len() >= 3 => adjacent_range_pair(rng, &boundary_keys),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    while ranges.len() < count {
        if let Some(range) = random_range_bounds_from_keys(rng, &boundary_keys) {
            ranges.push(range);
        } else {
            break;
        }
    }

    ranges
}

fn delete_range_boundary_candidates(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> Vec<Vec<u8>> {
    let mut candidates = Vec::with_capacity(keys.len() + 10);
    candidates.extend(keys.iter().cloned());
    candidates.extend(model.keys().cloned());
    candidates.push(Vec::new());
    candidates.push(vec![0xff, 0xff, 0xff, 0xff]);
    for _ in 0..8 {
        candidates.push(random_synthetic_key(rng));
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

fn random_synthetic_key(rng: &mut SmallRng) -> Vec<u8> {
    let len = rng.gen_range(1..=4);
    (0..len).map(|_| rng.gen()).collect()
}

fn pick_ordered_indexes(rng: &mut SmallRng, len: usize, count: usize) -> Vec<usize> {
    debug_assert!(len >= count);
    let mut indexes = HashSet::new();
    while indexes.len() < count {
        indexes.insert(rng.gen_range(0..len));
    }
    let mut indexes: Vec<usize> = indexes.into_iter().collect();
    indexes.sort_unstable();
    indexes
}

fn partial_overlap_range_pair(
    rng: &mut SmallRng,
    boundary_keys: &[Vec<u8>],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let indexes = pick_ordered_indexes(rng, boundary_keys.len(), 4);
    vec![
        (
            boundary_keys[indexes[0]].clone(),
            boundary_keys[indexes[2]].clone(),
        ),
        (
            boundary_keys[indexes[1]].clone(),
            boundary_keys[indexes[3]].clone(),
        ),
    ]
}

fn contained_range_pair(rng: &mut SmallRng, boundary_keys: &[Vec<u8>]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let indexes = pick_ordered_indexes(rng, boundary_keys.len(), 4);
    vec![
        (
            boundary_keys[indexes[0]].clone(),
            boundary_keys[indexes[3]].clone(),
        ),
        (
            boundary_keys[indexes[1]].clone(),
            boundary_keys[indexes[2]].clone(),
        ),
    ]
}

fn adjacent_range_pair(rng: &mut SmallRng, boundary_keys: &[Vec<u8>]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let indexes = pick_ordered_indexes(rng, boundary_keys.len(), 3);
    vec![
        (
            boundary_keys[indexes[0]].clone(),
            boundary_keys[indexes[1]].clone(),
        ),
        (
            boundary_keys[indexes[1]].clone(),
            boundary_keys[indexes[2]].clone(),
        ),
    ]
}

fn random_range_bounds_from_keys(
    rng: &mut SmallRng,
    boundary_keys: &[Vec<u8>],
) -> Option<(Vec<u8>, Vec<u8>)> {
    if boundary_keys.len() < 2 {
        return None;
    }

    let a = rng.gen_range(0..boundary_keys.len());
    let b = rng.gen_range(0..boundary_keys.len());
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if lo == hi {
        return None;
    }

    Some((boundary_keys[lo].clone(), boundary_keys[hi].clone()))
}

fn random_mixed_point_key(
    rng: &mut SmallRng,
    keys: &[Vec<u8>],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    ops: &[BatchOp],
) -> Vec<u8> {
    let ranges: Vec<(&[u8], &[u8])> = ops
        .iter()
        .filter_map(|op| match op {
            BatchOp::DeleteRange { start, end } => Some((start.as_slice(), end.as_slice())),
            BatchOp::Put { .. } | BatchOp::Delete { .. } => None,
        })
        .collect();

    if !ranges.is_empty() {
        match rng.gen_range(0..=7) {
            0 => {
                let (start, _) = ranges[rng.gen_range(0..ranges.len())];
                return start.to_vec();
            }
            1 => {
                let (_, end) = ranges[rng.gen_range(0..ranges.len())];
                return end.to_vec();
            }
            2 => {
                let (start, end) = ranges[rng.gen_range(0..ranges.len())];
                if let Some(key) = known_key_before(keys, end) {
                    if key.as_slice() > start {
                        return key;
                    }
                }
            }
            3 => {
                let (_, end) = ranges[rng.gen_range(0..ranges.len())];
                if let Some(key) = known_key_after(keys, end) {
                    return key;
                }
            }
            4 | 5 => {
                let (start, end) = ranges[rng.gen_range(0..ranges.len())];
                let candidates: Vec<&Vec<u8>> = model
                    .keys()
                    .chain(keys.iter())
                    .filter(|key| key.as_slice() >= start && key.as_slice() < end)
                    .collect();
                if !candidates.is_empty() {
                    return candidates[rng.gen_range(0..candidates.len())].clone();
                }
            }
            _ => {}
        }
    }

    if rng.gen_bool(0.5) && !model.is_empty() {
        choose_existing_key(rng, model)
    } else {
        choose_missing_key(rng, keys, model)
    }
}

fn known_key_before(keys: &[Vec<u8>], bound: &[u8]) -> Option<Vec<u8>> {
    keys.iter()
        .rev()
        .find(|key| key.as_slice() < bound)
        .cloned()
}

fn known_key_after(keys: &[Vec<u8>], bound: &[u8]) -> Option<Vec<u8>> {
    keys.iter().find(|key| key.as_slice() > bound).cloned()
}

fn sort_valid_batch_ops(ops: &mut [BatchOp]) {
    ops.sort_by(|a, b| {
        a.key().cmp(b.key()).then_with(|| {
            matches!(b, BatchOp::DeleteRange { .. }).cmp(&matches!(a, BatchOp::DeleteRange { .. }))
        })
    });
}

#[derive(Clone, Copy, Debug, Default)]
struct BatchShapeStats {
    delete_ranges: usize,
    range_only_delete_batches: usize,
    mixed_delete_range_batches: usize,
    multi_delete_range_batches: usize,
    overlapping_delete_range_batches: usize,
    contained_delete_range_batches: usize,
    adjacent_delete_range_batches: usize,
    outside_bound_delete_range_batches: usize,
    both_bounds_missing_delete_range_batches: usize,
    before_first_delete_range_batches: usize,
    after_last_delete_range_batches: usize,
    same_start_range_point_batches: usize,
    inside_range_point_batches: usize,
    same_end_range_point_batches: usize,
    before_end_range_point_batches: usize,
    after_end_range_point_batches: usize,
    large_op_batches: usize,
    max_batch_ops: usize,
}

impl BatchShapeStats {
    fn add(&mut self, other: Self) {
        self.delete_ranges += other.delete_ranges;
        self.range_only_delete_batches += other.range_only_delete_batches;
        self.mixed_delete_range_batches += other.mixed_delete_range_batches;
        self.multi_delete_range_batches += other.multi_delete_range_batches;
        self.overlapping_delete_range_batches += other.overlapping_delete_range_batches;
        self.contained_delete_range_batches += other.contained_delete_range_batches;
        self.adjacent_delete_range_batches += other.adjacent_delete_range_batches;
        self.outside_bound_delete_range_batches += other.outside_bound_delete_range_batches;
        self.both_bounds_missing_delete_range_batches +=
            other.both_bounds_missing_delete_range_batches;
        self.before_first_delete_range_batches += other.before_first_delete_range_batches;
        self.after_last_delete_range_batches += other.after_last_delete_range_batches;
        self.same_start_range_point_batches += other.same_start_range_point_batches;
        self.inside_range_point_batches += other.inside_range_point_batches;
        self.same_end_range_point_batches += other.same_end_range_point_batches;
        self.before_end_range_point_batches += other.before_end_range_point_batches;
        self.after_end_range_point_batches += other.after_end_range_point_batches;
        self.large_op_batches += other.large_op_batches;
        self.max_batch_ops = self.max_batch_ops.max(other.max_batch_ops);
    }
}

fn batch_shape_stats(
    ops: &[BatchOp],
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    keys: &[Vec<u8>],
) -> BatchShapeStats {
    let ranges: Vec<(&Vec<u8>, &Vec<u8>)> = ops
        .iter()
        .filter_map(|op| match op {
            BatchOp::DeleteRange { start, end } => Some((start, end)),
            BatchOp::Put { .. } | BatchOp::Delete { .. } => None,
        })
        .collect();
    let point_keys: Vec<&Vec<u8>> = ops
        .iter()
        .filter_map(|op| match op {
            BatchOp::Put { key, .. } | BatchOp::Delete { key } => Some(key),
            BatchOp::DeleteRange { .. } => None,
        })
        .collect();

    let mut stats = BatchShapeStats {
        delete_ranges: ranges.len(),
        range_only_delete_batches: usize::from(!ranges.is_empty() && point_keys.is_empty()),
        mixed_delete_range_batches: usize::from(!ranges.is_empty() && !point_keys.is_empty()),
        multi_delete_range_batches: usize::from(ranges.len() > 1),
        large_op_batches: usize::from(ops.len() >= LARGE_BATCH_OP_THRESHOLD),
        max_batch_ops: ops.len(),
        ..BatchShapeStats::default()
    };

    let first_model_key = model.keys().next();
    let last_model_key = model.keys().next_back();

    for (start, end) in &ranges {
        let start_missing = !model.contains_key(start.as_slice());
        let end_missing = !model.contains_key(end.as_slice());

        if start_missing || end_missing {
            stats.outside_bound_delete_range_batches = 1;
        }
        if start_missing && end_missing {
            stats.both_bounds_missing_delete_range_batches = 1;
        }
        if first_model_key.is_some_and(|first| *start < first) {
            stats.before_first_delete_range_batches = 1;
        }
        if last_model_key.is_some_and(|last| *end > last) {
            stats.after_last_delete_range_batches = 1;
        }
    }

    for (i, (left_start, left_end)) in ranges.iter().enumerate() {
        for (right_start, right_end) in ranges.iter().skip(i + 1) {
            if *left_start < *right_end && *right_start < *left_end {
                stats.overlapping_delete_range_batches = 1;
            }
            if (*left_start <= *right_start && *right_end <= *left_end)
                || (*right_start <= *left_start && *left_end <= *right_end)
            {
                stats.contained_delete_range_batches = 1;
            }
            if left_end == right_start || right_end == left_start {
                stats.adjacent_delete_range_batches = 1;
            }
        }
    }

    for key in point_keys {
        for (start, end) in &ranges {
            if key == *start {
                stats.same_start_range_point_batches = 1;
            }
            if key > *start && key < *end {
                stats.inside_range_point_batches = 1;
            }
            if key == *end {
                stats.same_end_range_point_batches = 1;
            }
            if known_key_before(keys, end) == Some(key.clone()) && key > *start && key < *end {
                stats.before_end_range_point_batches = 1;
            }
            if known_key_after(keys, end) == Some(key.clone()) {
                stats.after_end_range_point_batches = 1;
            }
        }
    }

    stats
}

// ---------------------------------------------------------------------------
// Transcript generation and verification
// ---------------------------------------------------------------------------

struct TranscriptResult {
    verified_reads: Vec<ReadResults>,
    shape_stats: BatchShapeStats,
    start_model_len: usize,
    read_result_rows: usize,
    final_model_len: usize,
}

fn run_random_transcript(
    seed: u64,
    model: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    keys: &[Vec<u8>],
    max_steps: usize,
    max_ops_per_step: usize,
) -> TranscriptResult {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_mul(0x9e37_79b9));

    let start_model_len = model.len();
    let live = build_live_store(model);
    // Snapshot the start tree before any writes — this is the trace's base.
    let start_tree = live.checkpoint();
    let start_root = live.root_hash();

    let num_steps = rng.gen_range(1..=max_steps);
    let mut steps: Vec<Step> = Vec::new();
    let mut model_reads_all: Vec<ReadResults> = Vec::new();
    let mut wrote = false;
    let mut shape_stats = BatchShapeStats::default();

    for _ in 0..num_steps {
        // Require at least one write before reads, then interleave both.
        let is_write = !wrote || rng.gen_bool(0.5);

        if is_write {
            let op_count = rng.gen_range(1..=max_ops_per_step);
            let ops = random_write_ops(&mut rng, keys, model, op_count);
            if ops.is_empty() {
                continue;
            }
            shape_stats.add(batch_shape_stats(&ops, model, keys));

            // Apply to live store
            let batch: Vec<(Vec<u8>, Op)> = ops.iter().map(BatchOp::to_batch_entry).collect();
            if let Err(err) = apply_write_step_to_live(&live, &batch) {
                panic!(
                    "seed {:#x}: live apply_sorted_batch_ops failed: {:?}\nbatch: {:?}",
                    seed, err, batch
                );
            }

            // Apply to model
            apply_ops_to_model(model, &ops);
            steps.push(Step::Write(ops));
            wrote = true;
        } else {
            let num_reads = rng.gen_range(1..=3);
            let read_ops: Vec<ReadOp> = (0..num_reads)
                .map(|_| random_read_op(&mut rng, keys, model))
                .collect();

            // Capture the model's view at this point so we can cross-check the
            // verifier-recomputed reads after replay.
            let model_results: ReadResults = read_ops
                .iter()
                .map(|op| ProvenRead {
                    op: op.clone(),
                    results: model_read(model, op),
                })
                .collect();

            steps.push(Step::Read(read_ops));
            model_reads_all.push(model_results);
        }
    }

    let trace = match create_trace(&start_tree, &steps) {
        Ok(t) => t,
        Err(err) => panic!("seed {:#x}: create_trace failed: {:?}", seed, err),
    };

    // The trace authenticates to the start root.
    assert_eq!(
        trace.hash(),
        start_root,
        "seed {:#x}: trace start root mismatch",
        seed
    );

    // Independent end-root oracle (live storage apply path), cross-checked
    // against the actual live store.
    let end_root = root_after_writes(&start_tree, &steps);
    assert_eq!(
        end_root,
        live.root_hash(),
        "seed {:#x}: end root mismatch",
        seed
    );

    let verified = match replay_trace(&trace, start_root, &steps, end_root) {
        Ok(v) => v,
        Err(err) => panic!("seed {:#x}: replay_trace failed: {:?}", seed, err),
    };

    // Check verified reads match model reads
    assert_eq!(
        verified.len(),
        model_reads_all.len(),
        "seed {:#x}: verified read count mismatch",
        seed
    );
    for (step_idx, (verified_step, model_step)) in
        verified.iter().zip(model_reads_all.iter()).enumerate()
    {
        for (read_idx, (v, m)) in verified_step.iter().zip(model_step.iter()).enumerate() {
            assert_eq!(
                v.results, m.results,
                "seed {seed:#x}: verified read step {step_idx} read {read_idx} diverged.\n\
                 op: {:?}\nverified: {:?}\nmodel: {:?}",
                v.op, v.results, m.results,
            );
        }
    }

    let read_result_rows = verified
        .iter()
        .flat_map(|step| step.iter())
        .map(|read| read.results.len())
        .sum();

    TranscriptResult {
        verified_reads: verified,
        shape_stats,
        start_model_len,
        read_result_rows,
        final_model_len: model.len(),
    }
}

// ---------------------------------------------------------------------------
// Deterministic boundary tests
// ---------------------------------------------------------------------------

const BOUNDARY_SEEDS: [u64; 20] = [
    0x1000_0001,
    0x1000_0002,
    0x1000_0003,
    0x1000_0004,
    0x1000_0005,
    0x1000_0006,
    0x1000_0007,
    0x1000_0008,
    0x1000_0009,
    0x1000_000a,
    0x1000_000b,
    0x1000_000c,
    0x1000_000d,
    0x1000_000e,
    0x1000_000f,
    0x1000_0010,
    0x1000_0011,
    0x1000_0012,
    0x1000_0013,
    0x1000_0014,
];

#[test]
fn random_multi_step_transcripts_replay_reads_writes_and_delete_ranges() {
    for &seed in &BOUNDARY_SEEDS {
        let mut rng = SmallRng::seed_from_u64(seed);
        let keys = key_universe(&mut rng);
        let mut model = make_initial_model(&mut rng, &keys, 0.3);

        run_random_transcript(seed, &mut model, &keys, 6, 5);
    }
}

#[test]
fn random_read_op_allows_empty_prefix_from_empty_key() {
    let keys = vec![Vec::new()];
    let model = BTreeMap::from([(Vec::new(), b"empty-key".to_vec())]);
    let mut rng = SmallRng::seed_from_u64(0x5000_0001);

    for _ in 0..1024 {
        if matches!(
            random_read_op(&mut rng, &keys, &model),
            ReadOp::Prefix(prefix) if prefix.is_empty()
        ) {
            return;
        }
    }

    panic!("deterministic generator should exercise empty-prefix reads");
}

#[test]
fn random_multi_step_transcripts_cover_delete_range_batch_shapes() {
    let mut stats = BatchShapeStats::default();
    let mut read_result_rows = 0;
    let mut nonempty_final_models = 0;

    for seed in 0x4000_0001..=0x4000_0040u64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let keys = key_universe(&mut rng);
        let mut model = make_initial_model(&mut rng, &keys, 0.5);
        let result = run_random_transcript(seed, &mut model, &keys, 12, 8);
        stats.add(result.shape_stats);
        read_result_rows += result.read_result_rows;
        nonempty_final_models += usize::from(result.final_model_len > 0);
    }

    assert!(
        stats.delete_ranges > 0,
        "deterministic fuzz seeds should exercise delete-ranges"
    );
    assert!(
        stats.mixed_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise delete-range plus point-op batches"
    );
    assert!(
        stats.range_only_delete_batches > 0,
        "deterministic fuzz seeds should exercise range-only delete batches"
    );
    assert!(
        stats.multi_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise batches with multiple delete-ranges"
    );
    assert!(
        stats.overlapping_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise overlapping delete-ranges"
    );
    assert!(
        stats.contained_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise contained delete-ranges"
    );
    assert!(
        stats.adjacent_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise adjacent delete-ranges"
    );
    assert!(
        stats.outside_bound_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise delete-ranges with missing bounds"
    );
    assert!(
        stats.both_bounds_missing_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise delete-ranges with both bounds missing"
    );
    assert!(
        stats.before_first_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise delete-ranges before the first model key"
    );
    assert!(
        stats.after_last_delete_range_batches > 0,
        "deterministic fuzz seeds should exercise delete-ranges after the last model key"
    );
    assert!(
        stats.same_start_range_point_batches > 0,
        "deterministic fuzz seeds should exercise point ops at delete-range starts"
    );
    assert!(
        stats.inside_range_point_batches > 0,
        "deterministic fuzz seeds should exercise point ops inside deleted ranges"
    );
    assert!(
        stats.same_end_range_point_batches > 0,
        "deterministic fuzz seeds should exercise point ops at delete-range ends"
    );
    assert!(
        stats.before_end_range_point_batches > 0,
        "deterministic fuzz seeds should exercise point ops just before delete-range ends"
    );
    assert!(
        stats.after_end_range_point_batches > 0,
        "deterministic fuzz seeds should exercise point ops just after delete-range ends"
    );
    assert!(
        read_result_rows > 0,
        "deterministic fuzz seeds should still produce non-empty verified reads"
    );
    assert!(
        nonempty_final_models > 0,
        "deterministic fuzz seeds should not all collapse to empty final models"
    );
}

#[test]
fn mixed_delete_range_batch_reuses_start_and_overlaps() {
    let keys: Vec<Vec<u8>> = (0u8..30).map(|i| vec![i]).collect();
    let mut model: BTreeMap<Vec<u8>, Vec<u8>> =
        keys.iter().map(|key| (key.clone(), key.clone())).collect();

    let live = build_live_store(&model);
    let start_tree = live.checkpoint();
    let start_root = live.root_hash();

    let pre_read = vec![ReadOp::Range {
        start: vec![4],
        end: vec![21],
    }];
    let pre_expected: ReadResults = pre_read
        .iter()
        .map(|op| ProvenRead {
            op: op.clone(),
            results: model_read(&model, op),
        })
        .collect();

    let ops = vec![
        BatchOp::DeleteRange {
            start: vec![5],
            end: vec![15],
        },
        BatchOp::Put {
            key: vec![5],
            value: vec![55],
        },
        BatchOp::Put {
            key: vec![7],
            value: vec![77],
        },
        BatchOp::DeleteRange {
            start: vec![10],
            end: vec![20],
        },
        BatchOp::Put {
            key: vec![12],
            value: vec![120],
        },
        BatchOp::Put {
            key: vec![18],
            value: vec![180],
        },
    ];
    let stats = batch_shape_stats(&ops, &model, &keys);
    assert_eq!(stats.delete_ranges, 2);
    assert_eq!(stats.mixed_delete_range_batches, 1);
    assert_eq!(stats.multi_delete_range_batches, 1);
    assert_eq!(stats.overlapping_delete_range_batches, 1);
    assert_eq!(stats.same_start_range_point_batches, 1);
    assert_eq!(stats.inside_range_point_batches, 1);

    let batch: Vec<(Vec<u8>, Op)> = ops.iter().map(BatchOp::to_batch_entry).collect();
    apply_write_step_to_live(&live, &batch).unwrap();
    apply_ops_to_model(&mut model, &ops);

    let post_read = vec![ReadOp::Range {
        start: vec![4],
        end: vec![21],
    }];
    let post_expected: ReadResults = post_read
        .iter()
        .map(|op| ProvenRead {
            op: op.clone(),
            results: model_read(&model, op),
        })
        .collect();

    let steps = vec![
        Step::Read(pre_read),
        Step::Write(ops),
        Step::Read(post_read),
    ];

    let trace = create_trace(&start_tree, &steps).unwrap();
    assert_eq!(trace.hash(), start_root);
    let end_root = root_after_writes(&start_tree, &steps);
    assert_eq!(end_root, live.root_hash());
    let verified = replay_trace(&trace, start_root, &steps, end_root).unwrap();
    assert_eq!(verified, vec![pre_expected, post_expected]);
}

#[test]
fn delete_range_boundary_batches_are_proven() {
    let keys: Vec<Vec<u8>> = (0u8..15).map(|i| vec![i]).collect();
    let mut model: BTreeMap<Vec<u8>, Vec<u8>> = (1u8..=10).map(|i| (vec![i], vec![i])).collect();

    let live = build_live_store(&model);
    let start_tree = live.checkpoint();
    let start_root = live.root_hash();
    let mut stats = BatchShapeStats::default();

    let range_only = vec![
        BatchOp::DeleteRange {
            start: Vec::new(),
            end: vec![2],
        },
        BatchOp::DeleteRange {
            start: vec![12],
            end: vec![13],
        },
    ];
    stats.add(batch_shape_stats(&range_only, &model, &keys));
    let batch: Vec<(Vec<u8>, Op)> = range_only.iter().map(BatchOp::to_batch_entry).collect();
    apply_write_step_to_live(&live, &batch).unwrap();
    apply_ops_to_model(&mut model, &range_only);

    let boundary_points = vec![
        BatchOp::DeleteRange {
            start: vec![3],
            end: vec![7],
        },
        BatchOp::Put {
            key: vec![6],
            value: vec![60],
        },
        BatchOp::Put {
            key: vec![7],
            value: vec![70],
        },
        BatchOp::Put {
            key: vec![8],
            value: vec![80],
        },
    ];
    stats.add(batch_shape_stats(&boundary_points, &model, &keys));
    let batch: Vec<(Vec<u8>, Op)> = boundary_points
        .iter()
        .map(BatchOp::to_batch_entry)
        .collect();
    apply_write_step_to_live(&live, &batch).unwrap();
    apply_ops_to_model(&mut model, &boundary_points);

    assert_eq!(stats.range_only_delete_batches, 1);
    assert_eq!(stats.outside_bound_delete_range_batches, 1);
    assert_eq!(stats.both_bounds_missing_delete_range_batches, 1);
    assert_eq!(stats.before_first_delete_range_batches, 1);
    assert_eq!(stats.after_last_delete_range_batches, 1);
    assert_eq!(stats.same_end_range_point_batches, 1);
    assert_eq!(stats.before_end_range_point_batches, 1);
    assert_eq!(stats.after_end_range_point_batches, 1);

    let read = vec![ReadOp::Range {
        start: Vec::new(),
        end: vec![15],
    }];
    let expected: ReadResults = read
        .iter()
        .map(|op| ProvenRead {
            op: op.clone(),
            results: model_read(&model, op),
        })
        .collect();

    let steps = vec![
        Step::Write(range_only),
        Step::Write(boundary_points),
        Step::Read(read),
    ];

    let trace = create_trace(&start_tree, &steps).unwrap();
    assert_eq!(trace.hash(), start_root);
    let end_root = root_after_writes(&start_tree, &steps);
    assert_eq!(end_root, live.root_hash());
    let verified = replay_trace(&trace, start_root, &steps, end_root).unwrap();
    assert_eq!(verified, vec![expected]);
}

#[test]
fn random_multi_step_delete_range_heavy() {
    // Use seeds that produce more delete-range operations
    for seed in 0x2000_0001..=0x2000_0010u64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let keys = key_universe(&mut rng);
        let mut model = make_initial_model(&mut rng, &keys, 0.5);

        // Build a transcript with at least one delete-range
        let live = build_live_store(&model);
        let start_tree = live.checkpoint();
        let start_root = live.root_hash();
        let mut steps: Vec<Step> = Vec::new();

        // Initial read
        let read_ops = vec![ReadOp::Range {
            start: keys.first().unwrap().clone(),
            end: keys.last().unwrap().clone(),
        }];
        steps.push(Step::Read(read_ops));

        // Delete-range in the middle
        let model_keys: Vec<Vec<u8>> = model.keys().cloned().collect();
        if model_keys.len() >= 4 {
            let lo_idx = model_keys.len() / 4;
            let hi_idx = 3 * model_keys.len() / 4;
            let lo = model_keys[lo_idx].clone();
            let hi = model_keys[hi_idx].clone();
            if lo < hi {
                let ops = vec![BatchOp::DeleteRange {
                    start: lo.clone(),
                    end: hi.clone(),
                }];
                let batch: Vec<(Vec<u8>, Op)> = ops.iter().map(BatchOp::to_batch_entry).collect();
                apply_write_step_to_live(&live, &batch).unwrap();
                apply_ops_to_model(&mut model, &ops);
                steps.push(Step::Write(ops));
            }
        }

        // Post-delete read
        let post_read = vec![
            ReadOp::Range {
                start: keys.first().unwrap().clone(),
                end: keys.last().unwrap().clone(),
            },
            ReadOp::Key(keys[keys.len() / 2].clone()),
        ];
        let model_results: Vec<_> = post_read.iter().map(|op| model_read(&model, op)).collect();
        steps.push(Step::Read(post_read));

        let trace = create_trace(&start_tree, &steps)
            .unwrap_or_else(|err| panic!("seed {:#x}: create_trace failed: {:?}", seed, err));
        assert_eq!(trace.hash(), start_root, "seed {:#x}: start root", seed);
        let end_root = root_after_writes(&start_tree, &steps);
        assert_eq!(end_root, live.root_hash(), "seed {:#x}: end root", seed);

        let verified = match replay_trace(&trace, start_root, &steps, end_root) {
            Ok(v) => v,
            Err(err) => panic!("seed {:#x}: replay_trace failed: {:?}", seed, err),
        };

        // Two read steps; verified reads must match the model.
        assert_eq!(verified.len(), 2);
        let post_verified = &verified[1];
        for (i, (v, model_r)) in post_verified.iter().zip(model_results.iter()).enumerate() {
            assert_eq!(
                v.results, *model_r,
                "seed {:#x}: post-delete read {} diverged from model",
                seed, i
            );
        }
    }
}

#[test]
fn random_multi_step_write_then_read_all() {
    // Writes followed by a full range read to verify every key
    for seed in 0x3000_0001..=0x3000_0010u64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let keys = key_universe(&mut rng);
        let mut model = make_initial_model(&mut rng, &keys, 0.4);

        let live = build_live_store(&model);
        let start_tree = live.checkpoint();
        let start_root = live.root_hash();
        let mut steps: Vec<Step> = Vec::new();

        // Random writes
        let ops = random_write_ops(&mut rng, &keys, &model, 5);
        if !ops.is_empty() {
            let batch: Vec<(Vec<u8>, Op)> = ops.iter().map(BatchOp::to_batch_entry).collect();
            apply_write_step_to_live(&live, &batch).unwrap();
            apply_ops_to_model(&mut model, &ops);
            steps.push(Step::Write(ops));
        }

        // Full range read
        let read = vec![ReadOp::Range {
            start: vec![0x00],
            end: vec![0xff, 0xff, 0xff, 0xff],
        }];
        let expected: Vec<(Vec<u8>, Vec<u8>)> =
            model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        steps.push(Step::Read(read));

        let trace = create_trace(&start_tree, &steps).unwrap();
        assert_eq!(trace.hash(), start_root);
        let end_root = root_after_writes(&start_tree, &steps);
        assert_eq!(end_root, live.root_hash());
        let verified = replay_trace(&trace, start_root, &steps, end_root).unwrap();
        assert_eq!(
            verified.last().unwrap()[0].results,
            expected,
            "seed {seed:#x}: full range read after writes diverged from model"
        );
    }
}

// ---------------------------------------------------------------------------
// Long-running fuzz test
// ---------------------------------------------------------------------------

/// Long-running differential fuzzer for the tracer pipeline.
///
/// Run with:
/// ```text
/// FUZZ_SEED=0 FUZZ_ITERS=1000 cargo test --release -- --ignored fuzz_tracer_differential
/// ```
///
/// Reproduce a failure:
/// ```text
/// FUZZ_SEED=<seed> FUZZ_ITERS=1 cargo test --release -- --ignored fuzz_tracer_differential
/// ```
#[test]
#[ignore]
fn fuzz_tracer_differential() {
    fn mix_transcript_seed(seed: u64, transcript_index: usize) -> u64 {
        seed ^ ((transcript_index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    fn require_env(name: &str) -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                panic!(
                    "Set FUZZ_SEED=<n> and FUZZ_ITERS=<n>.\n\
                     Run: FUZZ_SEED=0 FUZZ_ITERS=1000 cargo test --release \
                     -- --ignored fuzz_tracer_differential\n\
                     Reproduce: FUZZ_SEED=<seed> FUZZ_ITERS=1 cargo test --release \
                     -- --ignored fuzz_tracer_differential"
                )
            })
    }
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .map(|s| {
                s.parse::<usize>()
                    .unwrap_or_else(|_| panic!("{} must be an integer", name))
            })
            .unwrap_or(default)
            .max(1)
    }
    fn env_usize_allow_zero(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .map(|s| {
                s.parse::<usize>()
                    .unwrap_or_else(|_| panic!("{} must be an integer", name))
            })
            .unwrap_or(default)
    }

    let start_seed: u64 = require_env("FUZZ_SEED");
    let num_iters: u64 = require_env("FUZZ_ITERS");
    let max_transcripts = env_usize("FUZZ_MAX_TRANSCRIPTS", 5);
    let max_steps = env_usize("FUZZ_MAX_STEPS", 8);
    let max_ops_per_step = env_usize("FUZZ_MAX_OPS_PER_STEP", 64);
    let max_initial_entries = env_usize("FUZZ_MAX_INITIAL_ENTRIES", 1000);
    let progress_interval = env_usize_allow_zero("FUZZ_PROGRESS_INTERVAL", 10);

    let mut total_proofs: u64 = 0;
    let mut total_reads_verified: u64 = 0;
    let mut total_stats = BatchShapeStats::default();
    let mut total_start_model_len: u64 = 0;
    let mut total_read_result_rows: u64 = 0;
    let mut max_start_model_len: usize = 0;

    for iter in 0..num_iters {
        let seed = start_seed.wrapping_add(iter);
        let mut rng = SmallRng::seed_from_u64(seed);

        let initial_entries = rng.gen_range(1..=max_initial_entries);
        let key_count = initial_entries
            .saturating_mul(2)
            .max(DEFAULT_KEY_UNIVERSE_LEN);
        let keys = key_universe_with_len(&mut rng, key_count);
        let mut model = make_initial_model_with_len(&mut rng, &keys, initial_entries);

        let transcript_count = rng.gen_range(1..=max_transcripts);

        for transcript_index in 0..transcript_count {
            let transcript_seed = mix_transcript_seed(seed, transcript_index);
            let result = run_random_transcript(
                transcript_seed,
                &mut model,
                &keys,
                max_steps,
                max_ops_per_step,
            );
            total_proofs += 1;
            total_reads_verified += result.verified_reads.len() as u64;
            total_stats.add(result.shape_stats);
            total_start_model_len += result.start_model_len as u64;
            total_read_result_rows += result.read_result_rows as u64;
            max_start_model_len = max_start_model_len.max(result.start_model_len);
        }

        if progress_interval > 0 && (iter + 1) % (progress_interval as u64) == 0 {
            let avg_start_model_len = total_start_model_len as f64 / total_proofs as f64;
            eprintln!(
                "fuzz_tracer_differential: completed {}/{} iterations; last seed {:#x}; \
                 proofs={} avg_start_tree_entries={:.1} max_start_tree_entries={} \
                 read_steps={} read_rows={} delete_ranges={} max_batch_ops={} \
                 large_op_batches={} range_only_batches={} mixed_range_point_batches={} \
                 multi_range_batches={} overlap_batches={} contained_batches={} \
                 adjacent_batches={} outside_bound_batches={} both_bounds_missing_batches={} \
                 before_first_batches={} after_last_batches={} same_start_batches={} \
                 inside_point_batches={} same_end_batches={} before_end_batches={} \
                 after_end_batches={}",
                iter + 1,
                num_iters,
                seed,
                total_proofs,
                avg_start_model_len,
                max_start_model_len,
                total_reads_verified,
                total_read_result_rows,
                total_stats.delete_ranges,
                total_stats.max_batch_ops,
                total_stats.large_op_batches,
                total_stats.range_only_delete_batches,
                total_stats.mixed_delete_range_batches,
                total_stats.multi_delete_range_batches,
                total_stats.overlapping_delete_range_batches,
                total_stats.contained_delete_range_batches,
                total_stats.adjacent_delete_range_batches,
                total_stats.outside_bound_delete_range_batches,
                total_stats.both_bounds_missing_delete_range_batches,
                total_stats.before_first_delete_range_batches,
                total_stats.after_last_delete_range_batches,
                total_stats.same_start_range_point_batches,
                total_stats.inside_range_point_batches,
                total_stats.same_end_range_point_batches,
                total_stats.before_end_range_point_batches,
                total_stats.after_end_range_point_batches,
            );
        }
    }

    let avg_start_model_len = total_start_model_len as f64 / total_proofs as f64;
    eprintln!(
        "fuzz_tracer_differential: finished {} iterations, {} proofs, {} verified read steps, \
         avg start tree entries {:.1}, max start tree entries {}, \
         {} verified read rows, {} delete-ranges, max batch ops {}, \
         {} large-op batches, {} range-only batches, {} mixed range+point batches, \
         {} multi-range batches, {} overlap batches, {} contained batches, \
         {} adjacent batches, {} outside-bound batches, {} both-bounds-missing batches, \
         {} before-first batches, {} after-last batches, {} same-start batches, \
         {} inside-point batches, {} same-end batches, {} before-end batches, \
         {} after-end batches",
        num_iters,
        total_proofs,
        total_reads_verified,
        avg_start_model_len,
        max_start_model_len,
        total_read_result_rows,
        total_stats.delete_ranges,
        total_stats.max_batch_ops,
        total_stats.large_op_batches,
        total_stats.range_only_delete_batches,
        total_stats.mixed_delete_range_batches,
        total_stats.multi_delete_range_batches,
        total_stats.overlapping_delete_range_batches,
        total_stats.contained_delete_range_batches,
        total_stats.adjacent_delete_range_batches,
        total_stats.outside_bound_delete_range_batches,
        total_stats.both_bounds_missing_delete_range_batches,
        total_stats.before_first_delete_range_batches,
        total_stats.after_last_delete_range_batches,
        total_stats.same_start_range_point_batches,
        total_stats.inside_range_point_batches,
        total_stats.same_end_range_point_batches,
        total_stats.before_end_range_point_batches,
        total_stats.after_end_range_point_batches,
    );
}
