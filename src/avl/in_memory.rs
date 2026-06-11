use std::sync::RwLock;

use crate::avl::node::{GetResult, Node};
use crate::avl::walker::Walker;
use crate::avl::PanicSource;
use crate::error::{Error, Result, UnsupportedFeature};
use crate::hash::{Hash, NULL_HASH};
use crate::ops::{Batch, BatchEntry, Op};
use crate::proofs::query::QueryItem;
use crate::tracer::{prefix_successor, WriteOp};

pub struct InMemoryMerk {
    root: RwLock<Option<Node>>,
}

impl InMemoryMerk {
    pub fn new() -> Self {
        InMemoryMerk {
            root: RwLock::new(None),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let root = self.root.read().unwrap();
        root.as_ref().and_then(|t| t.get(key))
    }

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint::new(self.root.read().unwrap().clone())
    }

    /// Restore the tree to a previously captured snapshot in O(1).
    ///
    /// The snapshot shares the persistent (`Arc`) nodes of the tree it was
    /// taken from, and those nodes are already committed, so this is a cheap
    /// pointer swap rather than a rebuild. `None` clears the tree (matching
    /// [`crate::mrt::Tree::restore`]).
    ///
    /// # Transactions (checkpoint / rollback)
    ///
    /// [`checkpoint`](Self::checkpoint) + `restore` is the transaction primitive:
    /// snapshot to checkpoint, apply changes one at a time, and `restore` the
    /// checkpoint to roll back all-or-nothing on failure. Because the tree is
    /// persistent both halves are O(1), so this is cheaper and more general than
    /// a dedicated atomic batch-apply (you can checkpoint anywhere and nest), and
    /// it stays canonical for **both** backends — important for AVL, whose tree is
    /// insertion-order sensitive, so a transaction must replay ops in issue order
    /// rather than as a sorted batch.
    ///
    /// ```ignore
    /// let savepoint = tree.checkpoint();
    /// for op in &ops {
    ///     if apply(tree, op).is_err() {
    ///         tree.restore(Some(savepoint)); // roll back
    ///         return Err(/* … */);
    ///     }
    /// }
    /// ```
    ///
    /// (On the verify side there is nothing to roll back: any error aborts the
    /// whole guest execution / rejects the proof, so a failed
    /// [`crate::avl::TraceReplayer`] is simply discarded.)
    pub fn restore(&self, checkpoint: Option<Checkpoint>) {
        *self.root.write().unwrap() = checkpoint.and_then(Checkpoint::into_root);
    }

    pub fn root_hash(&self) -> Hash {
        let root = self.root.read().unwrap();
        root.as_ref().map_or(NULL_HASH, |t| t.hash())
    }

    pub fn prove<Q, I>(&self, query: I) -> Result<Vec<u8>>
    where
        Q: Into<QueryItem>,
        I: IntoIterator<Item = Q>,
    {
        let root = self.root.read().unwrap();
        crate::proofs::query::prove_resident(root.as_ref(), query)
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

    // Sorted host-batch API, kept crate-internal (no longer p2-facing). Production
    // writes use `apply_sorted_batch_ops_owned` (via `put`/`delete`) or
    // `apply_write_ops`; the multi-op `&Batch` entry currently only builds sorted
    // test fixtures, so it is dead in non-test builds.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn apply_sorted_batch_ops(&self, batch: &Batch) -> Result<()> {
        Self::validate_batch(batch)?;
        let current = self.root.read().unwrap().clone();
        let maybe_tree = Self::apply_unvalidated_batch_to_root(current, batch)?;
        self.replace_root(maybe_tree);
        Ok(())
    }

    pub(crate) fn apply_sorted_batch_ops_owned(&self, batch: Vec<BatchEntry>) -> Result<()> {
        Self::validate_batch(&batch)?;
        let current = self.root.read().unwrap().clone();
        let maybe_tree = Self::apply_unvalidated_batch_to_root(current, &batch)?;
        self.replace_root(maybe_tree);
        Ok(())
    }

    /// Apply ordered proven writes to durable storage.
    ///
    /// Unlike [`Self::apply_sorted_batch_ops`], this is **not** a sorted host batch API:
    /// `ops` are executed in vector order, one at a time, and the live root is
    /// swapped only after every op succeeds. On error, the candidate is dropped
    /// and the live tree remains usable.
    pub fn apply_write_ops(&self, ops: &[WriteOp]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        let current_root = {
            let root = self.root.read().unwrap();
            root.clone()
        };
        let candidate = Self::apply_writes_to_root(current_root, ops)?;
        self.replace_root(candidate);
        Ok(())
    }

    pub(crate) fn validate_batch(batch: &Batch) -> Result<()> {
        for i in 1..batch.len() {
            if batch[i].0 < batch[i - 1].0 {
                return Err(Error::Bound("Batch keys must be sorted".into()));
            }
            // Range starts are sorted by the batch key. Range ends are not
            // ordered; adjacent, overlapping, and nested ranges are applied
            // sequentially. The only duplicate-key case is a point op after a
            // DeleteRange at the same start key.
            let duplicate_key_allowed = matches!(batch[i - 1].1, Op::DeleteRange(_))
                && !matches!(batch[i].1, Op::DeleteRange(_));
            if batch[i].0 == batch[i - 1].0 && !duplicate_key_allowed {
                return Err(Error::Bound(
                    "Batch keys must be unique except after DeleteRange".into(),
                ));
            }
        }
        // Check DeleteRange bounds
        for (key, op) in batch.iter() {
            if let Op::DeleteRange(ref end) = op {
                if key >= end {
                    return Err(Error::Bound(
                        "DeleteRange start must be less than end".into(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn apply_writes_to_root(
        mut current_tree: Option<Node>,
        ops: &[WriteOp],
    ) -> Result<Option<Node>> {
        for op in ops {
            current_tree = Self::apply_write_to_root(current_tree, op)?;
        }
        Ok(current_tree)
    }

    fn apply_write_to_root(current_tree: Option<Node>, op: &WriteOp) -> Result<Option<Node>> {
        match op {
            WriteOp::Put { key, value } => {
                let batch = [(key.clone(), Op::Put(value.clone()))];
                Self::apply_unvalidated_batch_to_root(current_tree, &batch)
            }
            WriteOp::Delete { key } => {
                let batch = [(key.clone(), Op::Delete)];
                Self::apply_unvalidated_batch_to_root(current_tree, &batch)
            }
            WriteOp::DeleteRange { start, end } => {
                let batch = [(start.clone(), Op::DeleteRange(end.clone()))];
                Self::validate_batch(&batch)?;
                Self::apply_unvalidated_batch_to_root(current_tree, &batch)
            }
            WriteOp::DeletePrefix { prefix } => Self::delete_prefix_from_root(current_tree, prefix),
            WriteOp::MovePrefix { .. } => Err(Error::Unsupported(UnsupportedFeature::MovePrefix)),
        }
    }

    fn delete_prefix_from_root(current_tree: Option<Node>, prefix: &[u8]) -> Result<Option<Node>> {
        if let Some(end) = prefix_successor(prefix) {
            let batch = [(prefix.to_vec(), Op::DeleteRange(end))];
            return Self::apply_unvalidated_batch_to_root(current_tree, &batch);
        }

        let Some(tree) = current_tree else {
            return Ok(None);
        };
        let walker = Walker::new(tree, PanicSource {});
        let (left, _deleted_suffix) = walker.split_at(prefix)?;
        Ok(left)
    }

    fn apply_unvalidated_batch_to_root(
        mut current_tree: Option<Node>,
        batch: &Batch,
    ) -> Result<Option<Node>> {
        let mut i = 0;
        while i < batch.len() {
            if let Op::DeleteRange(ref end) = batch[i].1 {
                let start = &batch[i].0;
                let maybe_walker = current_tree
                    .take()
                    .map(|node| Walker::new(node, PanicSource {}));
                current_tree = Walker::delete_range_apply_to(maybe_walker, start, end)?;
                i += 1;
            } else {
                let seg_start = i;
                while i < batch.len() && !matches!(batch[i].1, Op::DeleteRange(_)) {
                    i += 1;
                }
                let segment = &batch[seg_start..i];
                let maybe_walker = current_tree
                    .take()
                    .map(|node| Walker::new(node, PanicSource {}));
                let (new_tree, _deleted) =
                    Walker::apply_to_mut(maybe_walker, &mut segment.to_vec(), PanicSource {})?;
                current_tree = new_tree;
            }
        }

        Ok(current_tree)
    }

    fn replace_root(&self, mut maybe_tree: Option<Node>) {
        if let Some(ref mut t) = maybe_tree {
            t.commit();
        }

        let mut root = self.root.write().unwrap();
        *root = maybe_tree;
    }
}

impl Default for InMemoryMerk {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time, hash-authenticated read snapshot of an AVL tree — the
/// module-scoped peer of [`crate::mrt::Checkpoint`].
///
/// An empty tree is a valid snapshot (`root` is `None`, [`Self::root_hash`] is
/// [`NULL_HASH`], [`Self::is_empty`] is `true`); there is no separate "no
/// snapshot" state, so [`InMemoryMerk::checkpoint`] always returns a
/// `Checkpoint` rather than an `Option`. The read surface
/// (`root_hash`/`get`/`get_result`/`collect_range`/`collect_prefix`/`prove`)
/// matches `mrt::Checkpoint` so consumers can address either backend through
/// one `backend::Checkpoint`.
#[derive(Clone, Debug, Default)]
pub struct Checkpoint {
    root: Option<Node>,
}

impl Checkpoint {
    pub(crate) fn new(root: Option<Node>) -> Self {
        Self { root }
    }

    /// The underlying tree root, or `None` for an empty tree — the node-level
    /// entry point used by the trace builder and node-level consumers.
    pub fn root(&self) -> Option<&Node> {
        self.root.as_ref()
    }

    /// Consume the snapshot, yielding the underlying tree root.
    pub fn into_root(self) -> Option<Node> {
        self.root
    }

    /// `true` for a snapshot of an empty tree.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Authenticated root hash ([`NULL_HASH`] for an empty tree).
    pub fn root_hash(&self) -> Hash {
        self.root.as_ref().map_or(NULL_HASH, Node::root_hash)
    }

    /// Point read: the value for `key`, or `None` if absent.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.root.as_ref().and_then(|root| root.get(key))
    }

    /// Point read distinguishing absent (`NotFound`) from pruned.
    pub fn get_result(&self, key: &[u8]) -> Result<GetResult> {
        match self.root.as_ref() {
            Some(root) => root.get_result(key),
            None => Ok(GetResult::NotFound),
        }
    }

    /// Half-open range read `[start, end)` (open upper bound when `end` is
    /// `None`).
    pub fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        match self.root.as_ref() {
            Some(root) => root.collect_range(start, end),
            None => Ok(Vec::new()),
        }
    }

    /// Prefix read (all keys starting with `prefix`).
    pub fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        match self.root.as_ref() {
            Some(root) => root.collect_prefix(prefix),
            None => Ok(Vec::new()),
        }
    }

    /// Build a query proof over this snapshot.
    pub fn prove<Q, I>(&self, query: I) -> Result<Vec<u8>>
    where
        Q: Into<QueryItem>,
        I: IntoIterator<Item = Q>,
    {
        crate::proofs::query::prove_resident(self.root.as_ref(), query)
    }
}

#[cfg(test)]
mod tests;
