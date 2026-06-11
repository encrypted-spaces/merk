use std::sync::{Arc, RwLock};

use crate::avl::node::GetResult;
use crate::error::{Error, Result};
use crate::hash::{Hash, NULL_HASH};
use crate::ops::{Batch, BatchEntry, Op};
use crate::proofs::query::QueryItem;
pub use crate::tracer::WriteOp;
pub use cursor::{Iter, ReverseIter};
pub use query_proof::{verify, verify_query};
pub use trace::Trace;
pub use tracer::TraceRecorder;
use tree::MrtNodeInner;
pub use verify_replay::{TraceReplayer, TraceVerifier};

mod cursor;
pub mod query_proof;
pub(crate) mod trace;
pub(crate) mod tracer;
pub(crate) mod tree;
pub(crate) mod verify_replay;

pub struct Tree {
    root: RwLock<Option<Arc<MrtNodeInner>>>,
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    root: Option<Arc<MrtNodeInner>>,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            root: RwLock::new(None),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let root = self.root.read().unwrap();
        get_complete(root.as_ref(), key)
    }

    pub fn checkpoint(&self) -> Checkpoint {
        let root = self.root.read().unwrap();
        Checkpoint { root: root.clone() }
    }

    /// Restore the tree to a previously captured snapshot in O(1).
    /// See [`crate::avl::Tree::restore`]; MRT nodes are persistent (`Arc`)
    /// too, so this is a cheap pointer swap rather than a rebuild. With
    /// [`checkpoint`](Self::checkpoint) this is also the checkpoint/rollback
    /// transaction primitive (see the AVL doc for the pattern).
    pub fn restore(&self, checkpoint: Option<Checkpoint>) {
        *self.root.write().unwrap() = checkpoint.and_then(|s| s.root);
    }

    pub fn root_hash(&self) -> Hash {
        let root = self.root.read().unwrap();
        root_hash_impl(root.as_ref())
    }

    pub fn prove<Q, I>(&self, query: I) -> Result<Vec<u8>>
    where
        Q: Into<QueryItem>,
        I: IntoIterator<Item = Q>,
    {
        self.checkpoint().prove(query)
    }

    pub fn put<K: Into<Vec<u8>>, V: Into<Vec<u8>>>(&self, key: K, value: V) -> Result<()> {
        self.apply_sorted_batch_ops_owned(vec![(key.into(), Op::Put(value.into()))])
    }

    pub fn delete<K: Into<Vec<u8>>>(&self, key: K) -> Result<()> {
        self.apply_sorted_batch_ops_owned(vec![(key.into(), Op::Delete)])
    }

    pub fn delete_range<K: Into<Vec<u8>>, E: Into<Vec<u8>>>(&self, start: K, end: E) -> Result<()> {
        self.apply_sorted_batch_ops_owned(vec![(start.into(), Op::DeleteRange(end.into()))])
    }

    /// Relocate the whole subtree of keys under byte-prefix `from` to byte-prefix
    /// `to` (every `from ‖ s` becomes `to ‖ s`, values preserved) in O(depth)
    /// work — independent of how many keys live under `from`. Direct mutation, no
    /// proof. **Overwrite semantics:** anything previously under `to` is discarded
    /// (the whole destination subtree is replaced, not merged). `Err` (leaving the
    /// tree untouched) on a precondition violation: equal prefixes, overlong
    /// resulting moved keys, an absent `from`, or a surviving key that is a
    /// byte-prefix of `to` (prefix-free violation). See [`tree::move_prefix`].
    pub fn move_prefix<K: Into<Vec<u8>>, Q: Into<Vec<u8>>>(&self, from: K, to: Q) -> Result<()> {
        let from = from.into();
        let to = to.into();
        let mut root = self.root.write().unwrap();
        let new_root = tree::move_prefix(root.clone(), &from, &to)?;
        *root = Some(new_root);
        Ok(())
    }

    // Sorted host-batch API, kept crate-internal (no longer p2-facing). Production
    // writes use `apply_sorted_batch_ops_owned` (via `put`/`delete`) or
    // `apply_write_ops`; the multi-op `&Batch` entry currently only builds sorted
    // test fixtures, so it is dead in non-test builds.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn apply_sorted_batch_ops(&self, batch: &Batch) -> Result<()> {
        validate_mrt_apply_batch(batch)?;
        let mut root = self.root.write().unwrap();
        let candidate = apply_batch_to_root(root.clone(), batch)?;
        *root = candidate;
        Ok(())
    }

    pub(crate) fn apply_sorted_batch_ops_owned(&self, batch: Vec<BatchEntry>) -> Result<()> {
        validate_mrt_apply_batch(&batch)?;
        let mut root = self.root.write().unwrap();
        let candidate = apply_batch_to_root(root.clone(), &batch)?;
        *root = candidate;
        Ok(())
    }

    /// Apply ordered proven writes to durable storage.
    ///
    /// This is distinct from the sorted host [`Self::apply_sorted_batch_ops`] path: `ops`
    /// are executed in vector order, one at a time, over a candidate root. The
    /// live root is swapped only after the full slice succeeds; on error, the
    /// candidate is dropped and the tree remains unchanged and usable.
    pub fn apply_write_ops(&self, ops: &[WriteOp]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        let mut root = self.root.write().unwrap();
        let candidate = apply_writes_to_root(root.clone(), ops)?;
        *root = candidate;
        Ok(())
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Checkpoint {
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        get_complete(self.root.as_ref(), key)
    }

    pub fn get_result(&self, key: &[u8]) -> Result<GetResult> {
        get_result_impl(self.root.as_ref(), key)
    }

    pub fn root_hash(&self) -> Hash {
        root_hash_impl(self.root.as_ref())
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter::new(self.root.as_ref())
    }

    pub fn iter_from(&self, start_key: &[u8]) -> Iter<'_> {
        Iter::from_key(self.root.as_ref(), start_key)
    }

    pub fn reverse_iter(&self) -> ReverseIter<'_> {
        ReverseIter::new(self.root.as_ref())
    }

    pub fn reverse_iter_from(&self, end_key: &[u8]) -> ReverseIter<'_> {
        ReverseIter::from_key_inclusive(self.root.as_ref(), end_key)
    }

    pub fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        cursor::collect_range(self.root.as_ref(), start, end)
    }

    pub fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        cursor::collect_prefix(self.root.as_ref(), prefix)
    }

    pub fn prove<Q, I>(&self, query: I) -> Result<Vec<u8>>
    where
        Q: Into<QueryItem>,
        I: IntoIterator<Item = Q>,
    {
        query_proof::prove_from_snapshot(self, query)
    }
}

/// Stateless batch validation shared by the host (`Batch`) and verifier-replay
/// (`[BatchOp]`) entry points. `entries` yields `(key, delete_range_end)` per op,
/// where `delete_range_end` is `Some` iff the op is a `DeleteRange`. Checks key/end
/// length bounds, `DeleteRange` start < end, sorted keys, and uniqueness (a key may
/// repeat only immediately after a `DeleteRange`). Takes only borrows, so it
/// allocates nothing and can run *before* any value-copying conversion.
fn validate_batch_entries<'a>(
    entries: impl Iterator<Item = (&'a [u8], Option<&'a [u8]>)>,
) -> Result<()> {
    let mut prev: Option<(&'a [u8], bool)> = None;
    for (key, delete_range_end) in entries {
        if key.len() > tree::MAX_KEY_LEN {
            return Err(Error::BatchKey(format!(
                "key length {} exceeds maximum {}",
                key.len(),
                tree::MAX_KEY_LEN
            )));
        }

        if let Some(end) = delete_range_end {
            if end.len() > tree::MAX_KEY_LEN {
                return Err(Error::BatchKey(format!(
                    "DeleteRange end key length {} exceeds maximum {}",
                    end.len(),
                    tree::MAX_KEY_LEN
                )));
            }
            if key >= end {
                return Err(Error::BatchKey(
                    "DeleteRange start must be less than end".into(),
                ));
            }
        }

        if let Some((prev_key, prev_was_delete_range)) = prev {
            match prev_key.cmp(key) {
                std::cmp::Ordering::Greater => {
                    return Err(Error::BatchKey("Batch keys must be sorted".into()));
                }
                std::cmp::Ordering::Equal => {
                    if !prev_was_delete_range || delete_range_end.is_some() {
                        return Err(Error::BatchKey(
                            "Batch keys must be unique except after DeleteRange".into(),
                        ));
                    }
                }
                std::cmp::Ordering::Less => {}
            }
        }
        prev = Some((key, delete_range_end.is_some()));
    }
    Ok(())
}

pub(crate) fn validate_mrt_apply_batch(batch: &Batch) -> Result<()> {
    validate_batch_entries(batch.iter().map(|(key, op)| {
        let end = match op {
            Op::DeleteRange(end) => Some(end.as_slice()),
            _ => None,
        };
        (key.as_slice(), end)
    }))
}

/// Same stateless checks as [`validate_mrt_apply_batch`] but over `[BatchOp]`
/// directly — so the verifier-replay public API can validate before the
/// value-copying `BatchOp::to_batch_entry` conversion (and before the defensive
/// tree clone), rejecting bad input without any allocation.
pub(crate) fn validate_batch_ops(ops: &[crate::tracer::BatchOp]) -> Result<()> {
    use crate::tracer::BatchOp;
    validate_batch_entries(ops.iter().map(|op| {
        let end = match op {
            BatchOp::DeleteRange { end, .. } => Some(end.as_slice()),
            _ => None,
        };
        (op.key(), end)
    }))
}

fn get_complete(root: Option<&Arc<MrtNodeInner>>, key: &[u8]) -> Option<Vec<u8>> {
    tree::get(root, key).ok().flatten().map(ToOwned::to_owned)
}

fn get_result_impl(root: Option<&Arc<MrtNodeInner>>, key: &[u8]) -> Result<GetResult> {
    tree::validate_key_len(key, "snapshot get_result")?;
    let Some(root) = root else {
        return Ok(GetResult::NotFound);
    };

    let mut cur = root;
    let mut depth = 0u16;
    loop {
        match cur.node() {
            tree::MrtNode::Leaf { skip, value } => {
                if tree::leaf_matches(skip, key, depth) {
                    return Ok(GetResult::Found(value.clone()));
                }
                return Ok(GetResult::NotFound);
            }
            tree::MrtNode::Branch { skip, left, right } => {
                if matches!(
                    skip.matches_key_at(key, depth),
                    tree::MatchResult::Mismatch { .. }
                ) {
                    return Ok(GetResult::NotFound);
                }
                let branch_depth = depth
                    .checked_add(skip.bit_len())
                    .ok_or_else(|| Error::Tree("MRT snapshot get_result depth overflow".into()))?;
                let side = tree::route_bit_at(key, branch_depth);
                depth = branch_depth.checked_add(1).ok_or_else(|| {
                    Error::Tree("MRT snapshot get_result child depth overflow".into())
                })?;
                cur = if !side { left } else { right };
            }
            tree::MrtNode::PrunedHash => {
                return Err(Error::PrunedNode(format!(
                    "MRT snapshot get_result descended into pruned node for key {key:?}"
                )));
            }
        }
    }
}

fn root_hash_impl(root: Option<&Arc<MrtNodeInner>>) -> Hash {
    root.map_or(NULL_HASH, |node| node.hash())
}

fn apply_batch_to_root(
    mut candidate: Option<Arc<MrtNodeInner>>,
    batch: &Batch,
) -> Result<Option<Arc<MrtNodeInner>>> {
    for (key, op) in batch.iter() {
        match op {
            Op::Put(value) => {
                candidate = Some(tree::insert(candidate, key.clone(), value.clone())?);
            }
            Op::Delete => {
                let (next, _) = tree::delete(candidate, key)?;
                candidate = next;
            }
            Op::DeleteRange(end) => {
                candidate = tree::delete_range(candidate, key, end)?;
            }
        }
    }
    Ok(candidate)
}

fn apply_writes_to_root(
    mut candidate: Option<Arc<MrtNodeInner>>,
    ops: &[WriteOp],
) -> Result<Option<Arc<MrtNodeInner>>> {
    for op in ops {
        match op {
            WriteOp::Put { key, value } => {
                candidate = Some(tree::insert(candidate, key.clone(), value.clone())?);
            }
            WriteOp::Delete { key } => {
                let (next, _) = tree::delete(candidate, key)?;
                candidate = next;
            }
            WriteOp::DeleteRange { start, end } => {
                candidate = tree::delete_range(candidate, start, end)?;
            }
            WriteOp::DeletePrefix { prefix } => {
                candidate = tree::delete_prefix(candidate, prefix)?;
            }
            WriteOp::MovePrefix { from, to } => {
                candidate = Some(tree::move_prefix(candidate, from, to)?);
            }
        }
    }
    Ok(candidate)
}

#[cfg(test)]
mod cross_backend_tests;
#[cfg(test)]
mod tests;
