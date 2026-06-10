use crate::child::Child;
use crate::error::{Error, Result};
use crate::hash::{kv_hash, Hash, Hasher};
use crate::node::Node;
use crate::walker::{Fetch, Walker};
use std::collections::LinkedList;
use std::fmt;
use Op::*;

/// An operation to be applied to a key in the store.
#[derive(Clone, PartialEq)]
pub enum Op {
    /// Inserts or updates the key/value entry to the given value.
    Put(Vec<u8>),
    /// Deletes the key/value entry.
    Delete,
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "{}",
            match self {
                Put(value) => format!("Put({value:?})"),
                Delete => "Delete".to_string(),
            }
        )
    }
}

/// A single `(key, operation)` pair.
pub type BatchEntry = (Vec<u8>, Op);

/// A mapping of keys and operations. Keys should be sorted and unique.
pub type Batch = [BatchEntry];

/// A source of data which panics when called. Useful when creating a store
/// which always keeps the state in memory.
#[derive(Clone)]
pub struct PanicSource {}

impl Fetch for PanicSource {
    fn fetch_by_key(&self, _: &[u8]) -> Result<Option<Node>> {
        unreachable!()
    }
}

impl<S> Walker<S>
where
    S: Fetch + Sized + Send + Clone,
{
    /// Applies a batch using the functional detach/recurse/reattach path.
    /// Safe with live snapshots (COW via Arc). Requires `arc` feature.
    #[cfg(not(use_box))]
    pub fn apply_cow(
        maybe_tree: Option<Self>,
        batch: &Batch,
        source: S,
    ) -> Result<(Option<Node>, LinkedList<Vec<u8>>)> {
        let mut batch = batch.to_vec();
        Self::apply_to_mut(maybe_tree, &mut batch, source)
    }

    /// Like `apply_cow` but takes an owned batch to avoid cloning.
    #[cfg(not(use_box))]
    pub fn apply_cow_owned(
        maybe_tree: Option<Self>,
        mut batch: Vec<BatchEntry>,
        source: S,
    ) -> Result<(Option<Node>, LinkedList<Vec<u8>>)> {
        Self::apply_to_mut(maybe_tree, &mut batch, source)
    }

    /// Applies a batch in-place via `&mut` references, avoiding detach/reattach.
    /// Requires exclusive ownership — will corrupt any live snapshots.
    pub fn apply_in_place(&mut self, batch: &mut Batch) -> Result<LinkedList<Vec<u8>>> {
        let mut deleted_keys = LinkedList::new();
        if !batch.is_empty() {
            let source = self.clone_source();
            let result =
                apply_in_place_recursive(self.tree_mut(), batch, &mut deleted_keys, &source)?;
            if result.node_deleted {
                return Err(Error::Key("apply_in_place: entire tree was deleted".into()));
            }
        }
        Ok(deleted_keys)
    }

    /// Applies a batch using the internal functional path. Handles both tree
    /// creation (when `maybe_tree` is None) and updates. Used by `apply_cow`
    /// and available for initial tree construction with Box nodes.
    pub fn apply_to_mut(
        maybe_tree: Option<Self>,
        batch: &mut Batch,
        source: S,
    ) -> Result<(Option<Node>, LinkedList<Vec<u8>>)> {
        let (maybe_walker, deleted_keys) = if batch.is_empty() {
            (maybe_tree, LinkedList::default())
        } else {
            match maybe_tree {
                None => return Ok((Self::build(batch, source)?, LinkedList::default())),
                Some(tree) => tree.apply_mut(batch)?,
            }
        };

        let maybe_tree = maybe_walker.map(|walker| walker.into_inner());
        Ok((maybe_tree, deleted_keys))
    }

    /// Builds a `Node` from a batch of operations.
    ///
    /// Keys in batch must be sorted and unique.
    fn build(batch: &mut Batch, source: S) -> Result<Option<Node>> {
        if batch.is_empty() {
            return Ok(None);
        }

        let mid_index = batch.len() / 2;
        let mid_op = std::mem::replace(&mut batch[mid_index].1, Op::Delete);
        let mid_tree = match mid_op {
            Delete => {
                let (left_batch, rest) = batch.split_at_mut(mid_index);
                let right_batch = &mut rest[1..];

                let maybe_tree = Self::build(left_batch, source.clone())?
                    .map(|tree| Self::new(tree, source.clone()));
                let maybe_tree = match maybe_tree {
                    Some(tree) => tree.apply_mut(right_batch)?.0,
                    None => Self::build(right_batch, source.clone())?
                        .map(|tree| Self::new(tree, source.clone())),
                };
                return Ok(maybe_tree.map(|tree| tree.into()));
            }
            Put(value) => {
                let key = std::mem::take(&mut batch[mid_index].0);
                Node::new(key, value)?
            }
        };

        let mid_walker = Walker::new(mid_tree, PanicSource {});
        Ok(mid_walker
            .recurse(batch, mid_index, true)?
            .0
            .map(|w| w.into_inner()))
    }

    /// Applies a batch of operations to an existing tree. This is similar to
    /// `Walker<S>::apply`_to, but requires a populated tree.
    ///
    /// Keys in batch must be sorted and unique.
    #[cfg(test)]
    fn apply(self, batch: &Batch) -> Result<(Option<Self>, LinkedList<Vec<u8>>)> {
        let mut batch = batch.to_vec();
        self.apply_mut(&mut batch)
    }

    fn apply_mut(self, batch: &mut Batch) -> Result<(Option<Self>, LinkedList<Vec<u8>>)> {
        // binary search to see if this node's key is in the batch, and to split
        // into left and right batches
        let search = batch.binary_search_by(|(key, _op)| key.as_slice().cmp(self.tree().key()));
        let tree = if let Ok(index) = search {
            // a key matches this node's key, apply op to this node
            match std::mem::replace(&mut batch[index].1, Op::Delete) {
                Put(value) => self.with_value(value),
                Delete => {
                    let source = self.clone_source();
                    let key = self.tree().key().to_vec();

                    let (walker, maybe_left) = self.detach(true)?;
                    let (walker, maybe_right) = walker.detach(false)?;
                    let (left_batch, rest) = batch.split_at_mut(index);
                    let right_batch = &mut rest[1..];

                    let (maybe_left, mut deleted_keys) =
                        Self::apply_to_mut(maybe_left, left_batch, source.clone())?;

                    deleted_keys.push_back(key);

                    let (maybe_right, mut deleted_keys_right) =
                        Self::apply_to_mut(maybe_right, right_batch, source)?;
                    deleted_keys.append(&mut deleted_keys_right);

                    let maybe_walker = walker
                        .attach(true, maybe_left)
                        .attach(false, maybe_right)
                        .remove()?
                        .map(|w| w.maybe_balance())
                        .transpose()?;

                    return Ok((maybe_walker, deleted_keys));
                }
            }
        } else {
            Ok(self)
        };

        let (mid, exclusive) = match search {
            Ok(index) => (index, true),
            Err(index) => (index, false),
        };

        tree?.recurse(batch, mid, exclusive)
    }

    /// Recursively applies operations to the tree's children (if there are any
    /// operations for them).
    ///
    /// This recursion executes serially in the same thread, but in the future
    /// will be dispatched to workers in other threads.
    fn recurse(
        self,
        batch: &mut Batch,
        mid: usize,
        exclusive: bool,
    ) -> Result<(Option<Self>, LinkedList<Vec<u8>>)> {
        let (left_batch, rest) = batch.split_at_mut(mid);
        let right_batch = if exclusive { &mut rest[1..] } else { rest };

        let mut deleted_keys = LinkedList::default();

        let tree = if !left_batch.is_empty() {
            let source = self.clone_source();
            self.walk(true, |maybe_left| {
                let (maybe_left, mut deleted_keys_left) =
                    Self::apply_to_mut(maybe_left, left_batch, source)?;
                deleted_keys.append(&mut deleted_keys_left);
                Ok(maybe_left)
            })?
        } else {
            self
        };

        let tree = if !right_batch.is_empty() {
            let source = tree.clone_source();
            tree.walk(false, |maybe_right| {
                let (maybe_right, mut deleted_keys_right) =
                    Self::apply_to_mut(maybe_right, right_batch, source)?;
                deleted_keys.append(&mut deleted_keys_right);
                Ok(maybe_right)
            })?
        } else {
            tree
        };

        let tree = tree.maybe_balance()?;

        Ok((Some(tree), deleted_keys))
    }

    /// Gets the wrapped tree's balance factor.
    #[inline]
    fn balance_factor(&self) -> i8 {
        self.tree().balance_factor()
    }

    /// Checks if the tree is unbalanced and if so, applies AVL tree rotation(s)
    /// to rebalance the tree and its subtrees. Returns the root node of the
    /// balanced tree after applying the rotations.
    fn maybe_balance(self) -> Result<Self> {
        let balance_factor = self.balance_factor();
        if balance_factor.abs() <= 1 {
            return Ok(self);
        }

        let left = balance_factor < 0;

        // maybe do a double rotation
        let tree = if left == (self.tree().child_ref(left).unwrap().balance_factor() > 0) {
            self.walk_expect(left, |child| Ok(Some(child.rotate(!left)?)))?
        } else {
            self
        };

        tree.rotate(left)
    }

    /// Applies an AVL tree rotation, a constant-time operation which only needs
    /// to swap pointers in order to rebalance a tree.
    fn rotate(self, left: bool) -> Result<Self> {
        let (tree, child) = self.detach_expect(left)?;
        let (child, maybe_grandchild) = child.detach(!left)?;

        // attach grandchild to self
        let tree = tree.attach(left, maybe_grandchild).maybe_balance()?;

        // attach self to child, return child
        child.attach(!left, Some(tree)).maybe_balance()
    }

    /// Removes the root node from the tree. Rearranges and rebalances
    /// descendants (if any) in order to maintain a valid tree.
    pub fn remove(self) -> Result<Option<Self>> {
        let tree = self.tree();
        let has_left = tree.child_ref(true).is_some();
        let has_right = tree.child_ref(false).is_some();
        let left = tree.child_height(true) > tree.child_height(false);

        let maybe_tree = if has_left && has_right {
            // two children, promote edge of taller child
            let (tree, tall_child) = self.detach_expect(left)?;
            let (_, short_child) = tree.detach_expect(!left)?;
            Some(tall_child.promote_edge(!left, short_child)?)
        } else if has_left || has_right {
            // single child, promote it
            Some(self.detach_expect(left)?.1)
        } else {
            // no child
            None
        };

        Ok(maybe_tree)
    }

    /// Traverses to find the tree's edge on the given side, removes it, and
    /// reattaches it at the top in order to fill in a gap when removing a root
    /// node from a tree with both left and right children. Attaches `attach` on
    /// the opposite side. Returns the promoted node.
    fn promote_edge(self, left: bool, attach: Self) -> Result<Self> {
        let (edge, maybe_child) = self.remove_edge(left)?;
        edge.attach(!left, maybe_child)
            .attach(left, Some(attach))
            .maybe_balance()
    }

    /// Traverses to the tree's edge on the given side and detaches it
    /// (reattaching its child, if any, to its former parent). Return value is
    /// `(edge, maybe_updated_tree)`.
    fn remove_edge(self, left: bool) -> Result<(Self, Option<Self>)> {
        if self.tree().child_ref(left).is_some() {
            // this node is not the edge, recurse
            let (tree, child) = self.detach_expect(left)?;
            let (edge, maybe_child) = child.remove_edge(left)?;
            let tree = tree.attach(left, maybe_child).maybe_balance()?;
            Ok((edge, Some(tree)))
        } else {
            // this node is the edge, detach its child if present
            self.detach(!left)
        }
    }
}

// ─── In-place batch apply ────────────────────────────────────────────────────
//
// Alternative to the functional detach/recurse/reattach path. Walks the tree
// via `&mut Node` references, modifying nodes in place. Saves ~2x on
// overlay_apply in the zkVM by eliminating per-level detach/attach overhead
// (Option::take/set, duplicate recompute_height, duplicate node_hash = None).
//
// The functional path (`apply_to_owned`) is preserved for callers that need
// COW semantics (server-side with snapshots).

#[derive(Default)]
struct InPlaceApplyResult {
    /// A value or hash changed on the path (requires node_hash invalidation).
    modified: bool,
    /// A node was inserted, deleted, or rotated (requires recompute_height + balance).
    structure_changed: bool,
    /// The current node was a leaf that was deleted (caller must remove the child slot).
    node_deleted: bool,
}

impl InPlaceApplyResult {
    fn merge(&mut self, other: InPlaceApplyResult) {
        self.modified |= other.modified;
        self.structure_changed |= other.structure_changed;
    }
}

/// Recursively applies operations in-place via `&mut Node` references.
fn apply_in_place_recursive<S: Fetch + Clone + Send>(
    node: &mut Node,
    batch: &mut Batch,
    deleted_keys: &mut LinkedList<Vec<u8>>,
    source: &S,
) -> Result<InPlaceApplyResult> {
    if batch.is_empty() {
        return Ok(InPlaceApplyResult::default());
    }

    let search = batch.binary_search_by(|(key, _)| key.as_slice().cmp(node.key()));

    let (mid, exclusive) = match search {
        Ok(index) => (index, true),
        Err(index) => (index, false),
    };

    let mut result = InPlaceApplyResult::default();

    let is_delete = matches!(search, Ok(idx) if matches!(&batch[idx].1, Delete));

    if search.is_ok() && !is_delete {
        let op = std::mem::replace(&mut batch[mid].1, Op::Delete);
        if apply_node_op_in_place(node, op)? {
            result.modified = true;
        }
    }

    let (left_batch, rest) = batch.split_at_mut(mid);
    let right_batch = if exclusive { &mut rest[1..] } else { rest };

    if !left_batch.is_empty() {
        result.merge(apply_in_place_to_child(
            node,
            true,
            left_batch,
            deleted_keys,
            source,
        )?);
    }

    if !right_batch.is_empty() {
        result.merge(apply_in_place_to_child(
            node,
            false,
            right_batch,
            deleted_keys,
            source,
        )?);
    }

    if is_delete {
        deleted_keys.push_back(node.key().to_vec());
        if remove_in_place(node, source)? {
            result.node_deleted = true;
            return Ok(result);
        }
        result.modified = true;
        result.structure_changed = true;
    }

    if result.modified {
        node.inner_mut().node_hash = None;
    }

    if result.structure_changed {
        node.inner_mut().recompute_height();
        maybe_balance_in_place(node, source)?;
    }

    Ok(result)
}

fn apply_node_op_in_place(node: &mut Node, op: Op) -> Result<bool> {
    let inner = node.inner_mut();

    match op {
        Put(value) => {
            let new_kv_hash = kv_hash::<Hasher>(&inner.key, &value)?;
            if inner.value.as_slice() != value.as_slice() || inner.kv_hash != new_kv_hash {
                let old_kv_hash = inner.kv_hash;
                inner.kv_hash = new_kv_hash;
                inner.value = value;
                if old_kv_hash != inner.kv_hash {
                    inner.node_hash = None;
                    return Ok(true);
                }
            }
        }
        Delete => {
            return Err(Error::Key("apply_in_place does not support Delete".into()));
        }
    }

    Ok(false)
}

/// Fetches a pruned child from the source and replaces it with a resident
/// node. No-op if the child is already resident or absent.
fn materialize_pruned_child<S: Fetch + Clone + Send>(
    node: &mut Node,
    left: bool,
    source: &S,
) -> Result<()> {
    if !matches!(node.child_ref(left), Some(Child::Pruned(_))) {
        return Ok(());
    }

    let inner = node.inner_mut();
    let pruned = inner.child_slot_mut(left).take().unwrap();
    let (hash, child_heights) = match &pruned {
        Child::Pruned(p) => (*p.node_hash(), p.child_heights()),
        _ => unreachable!(),
    };
    let mut fetched = source.fetch(&pruned)?;
    let fi = fetched.inner_mut();
    fi.node_hash = Some(hash);
    fi.height = 1 + std::cmp::max(child_heights.0, child_heights.1);
    *inner.child_slot_mut(left) = Some(Child::Resident(fetched));
    Ok(())
}

fn apply_in_place_to_child<S: Fetch + Clone + Send>(
    node: &mut Node,
    left: bool,
    batch: &mut Batch,
    deleted_keys: &mut LinkedList<Vec<u8>>,
    source: &S,
) -> Result<InPlaceApplyResult> {
    materialize_pruned_child(node, left, source)?;

    let child_result = {
        let inner = node.inner_mut();
        match inner.child_slot_mut(left) {
            Some(Child::Resident(ref mut child)) => Some(apply_in_place_recursive(
                child,
                batch,
                deleted_keys,
                source,
            )?),
            Some(Child::Pruned(_)) => unreachable!("pruned child should have been materialized"),
            None => None,
        }
    };

    if let Some(result) = child_result {
        if result.node_deleted {
            let inner = node.inner_mut();
            *inner.child_slot_mut(left) = None;
            return Ok(InPlaceApplyResult {
                modified: true,
                structure_changed: true,
                node_deleted: false,
            });
        }
        return Ok(result);
    }

    if let Some(child) = build_in_place_subtree(batch)? {
        let inner = node.inner_mut();
        let slot = inner.child_slot_mut(left);
        debug_assert!(slot.is_none());
        *slot = Some(Child::Resident(child));

        Ok(InPlaceApplyResult {
            modified: true,
            structure_changed: true,
            node_deleted: false,
        })
    } else {
        Ok(InPlaceApplyResult::default())
    }
}

fn build_in_place_subtree(batch: &mut Batch) -> Result<Option<Node>> {
    if batch.is_empty() {
        return Ok(None);
    }

    let mid_index = batch.len() / 2;
    let mid_op = std::mem::replace(&mut batch[mid_index].1, Op::Delete);
    let key = std::mem::take(&mut batch[mid_index].0);
    let mut node = match mid_op {
        Put(value) => Node::new(key, value)?,
        Delete => return Err(Error::Key("apply_in_place does not support Delete".into())),
    };

    let (left_batch, rest) = batch.split_at_mut(mid_index);
    let right_batch = &mut rest[1..];

    let left = build_in_place_subtree(left_batch)?;
    let right = build_in_place_subtree(right_batch)?;

    if left.is_some() || right.is_some() {
        let inner = node.inner_mut();
        inner.left = left.map(Child::Resident);
        inner.right = right.map(Child::Resident);
        inner.node_hash = None;
        inner.recompute_height();
        maybe_balance_in_place(&mut node, &PanicSource {})?;
    }

    Ok(Some(node))
}

fn maybe_balance_in_place<S: Fetch + Clone + Send>(node: &mut Node, source: &S) -> Result<()> {
    let balance_factor = node.balance_factor();
    if balance_factor.abs() <= 1 {
        return Ok(());
    }

    let left = balance_factor < 0;
    materialize_pruned_child(node, left, source)?;

    let child_balance_factor = node
        .child_ref(left)
        .ok_or_else(|| Error::Key("apply_in_place: cannot rotate missing child".into()))?
        .balance_factor();

    if left == (child_balance_factor > 0) {
        match node.inner_mut().child_slot_mut(left) {
            Some(Child::Resident(ref mut child)) => {
                materialize_pruned_child(child, !left, source)?;
                rotate_in_place(child, !left, source)?;
            }
            _ => unreachable!("pruned child should have been materialized"),
        }
    }

    rotate_in_place(node, left, source)
}

fn rotate_in_place<S: Fetch + Clone + Send>(node: &mut Node, left: bool, source: &S) -> Result<()> {
    let mut child = take_resident_child(node, left, source)?;
    materialize_pruned_child(&mut child, !left, source)?;
    let grandchild = child.inner_mut().child_slot_mut(!left).take();

    {
        let inner = node.inner_mut();
        *inner.child_slot_mut(left) = grandchild;
        inner.node_hash = None;
        inner.recompute_height();
    }

    maybe_balance_in_place(node, source)?;

    let old_root = std::mem::replace(node, child);

    {
        let inner = node.inner_mut();
        let slot = inner.child_slot_mut(!left);
        debug_assert!(slot.is_none());
        *slot = Some(Child::Resident(old_root));
        inner.node_hash = None;
        inner.recompute_height();
    }

    maybe_balance_in_place(node, source)
}

fn take_resident_child<S: Fetch + Clone + Send>(
    node: &mut Node,
    left: bool,
    source: &S,
) -> Result<Node> {
    materialize_pruned_child(node, left, source)?;

    match node.inner_mut().child_slot_mut(left).take() {
        Some(Child::Resident(child)) => Ok(child),
        Some(Child::Pruned(_)) => unreachable!("pruned child should have been materialized"),
        None => Err(Error::Key("take_resident_child: no child".into())),
    }
}

/// Removes the current node in-place.
/// Returns true if the node is a leaf (caller must remove the child slot).
/// For one child: replaces node with the child via mem::replace.
/// For two children: finds the in-order predecessor/successor, copies its
/// data here, and deletes it from the subtree.
fn remove_in_place<S: Fetch + Clone + Send>(node: &mut Node, source: &S) -> Result<bool> {
    let has_left = node.child_ref(true).is_some();
    let has_right = node.child_ref(false).is_some();

    if !has_left && !has_right {
        return Ok(true);
    }

    let left = node.child_height(true) > node.child_height(false);

    if has_left && has_right {
        let edge_side = !left;
        let (key, value, kv_hash) = find_and_remove_edge(node, left, edge_side, source)?;
        let inner = node.inner_mut();
        inner.key = key;
        inner.value = value;
        inner.kv_hash = kv_hash;
        inner.node_hash = None;
        inner.recompute_height();
        maybe_balance_in_place(node, source)?;
    } else {
        let child = take_resident_child(node, left, source)?;
        *node = child;
    }

    Ok(false)
}

/// Finds the edge node in direction `edge_side` within `node`'s child on
/// `child_side`, removes it, and returns its (key, value, kv_hash).
///
/// At intermediate levels, traverses via `&mut` references (no detach).
/// At the edge itself, takes the node out of the parent's slot (O(1)).
/// Rebalances at each level on the way back up.
fn find_and_remove_edge<S: Fetch + Clone + Send>(
    node: &mut Node,
    child_side: bool,
    edge_side: bool,
    source: &S,
) -> Result<(Vec<u8>, Vec<u8>, Hash)> {
    materialize_pruned_child(node, child_side, source)?;

    let has_further = match node.child(child_side) {
        Some(child) => child.child_ref(edge_side).is_some(),
        None => return Err(Error::Key("find_and_remove_edge: expected child".into())),
    };

    if has_further {
        let inner = node.inner_mut();
        let data = match inner.child_slot_mut(child_side) {
            Some(Child::Resident(ref mut child)) => {
                let data = find_and_remove_edge(child, edge_side, edge_side, source)?;
                let ci = child.inner_mut();
                ci.node_hash = None;
                ci.recompute_height();
                maybe_balance_in_place(child, source)?;
                data
            }
            _ => unreachable!("checked above"),
        };
        Ok(data)
    } else {
        let inner = node.inner_mut();
        let mut edge = match inner.child_slot_mut(child_side).take() {
            Some(Child::Resident(edge)) => edge,
            _ => unreachable!("checked above"),
        };

        let key = edge.key().to_vec();
        let value = edge.value().to_vec();
        let kv_hash = *edge.kv_hash();

        let replacement = edge.inner_mut().child_slot_mut(!edge_side).take();
        *inner.child_slot_mut(child_side) = replacement;

        Ok((key, value, kv_hash))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::child::Child;
    use crate::node::*;
    use crate::test_utils::{apply_memonly, del_entry, make_tree_seq, put_entry, seq_key};
    #[cfg(not(use_box))]
    use crate::test_utils::{assert_tree_invariants, put_entry_value};

    #[test]
    fn simple_insert() -> Result<()> {
        let batch = [(b"foo2".to_vec(), Op::Put(b"bar2".to_vec()))];
        let tree = Node::new(b"foo".to_vec(), b"bar".to_vec())?;
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        let walker = maybe_walker.expect("should be Some");
        assert_eq!(walker.tree().key(), b"foo");
        assert_eq!(walker.into_inner().child(false).unwrap().key(), b"foo2");
        assert!(deleted_keys.is_empty());
        Ok(())
    }

    #[test]
    fn simple_update() -> Result<()> {
        let batch = [(b"foo".to_vec(), Op::Put(b"bar2".to_vec()))];
        let tree = Node::new(b"foo".to_vec(), b"bar".to_vec())?;
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        let walker = maybe_walker.expect("should be Some");
        assert_eq!(walker.tree().key(), b"foo");
        assert_eq!(walker.tree().value(), b"bar2");
        assert!(walker.tree().child_ref(true).is_none());
        assert!(walker.tree().child_ref(false).is_none());
        assert!(deleted_keys.is_empty());
        Ok(())
    }

    #[test]
    fn simple_delete() -> Result<()> {
        let batch = [(b"foo2".to_vec(), Op::Delete)];
        let tree = Node::from_fields(
            b"foo".to_vec(),
            b"bar".to_vec(),
            [123; 32],
            None,
            Some(Child::Resident(Node::new(
                b"foo2".to_vec(),
                b"bar2".to_vec(),
            )?)),
        );
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        let walker = maybe_walker.expect("should be Some");
        assert_eq!(walker.tree().key(), b"foo");
        assert_eq!(walker.tree().value(), b"bar");
        assert!(walker.tree().child_ref(true).is_none());
        assert!(walker.tree().child_ref(false).is_none());
        assert_eq!(deleted_keys.len(), 1);
        assert_eq!(*deleted_keys.front().unwrap(), b"foo2");
        Ok(())
    }

    #[test]
    fn delete_non_existent() -> Result<()> {
        let batch = [(b"foo2".to_vec(), Op::Delete)];
        let tree = Node::new(b"foo".to_vec(), b"bar".to_vec())?;
        Walker::new(tree, PanicSource {}).apply(&batch).unwrap();
        Ok(())
    }

    #[test]
    fn noop_delete_does_not_dirty_committed_tree() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?;
        let batch = [(vec![3], Op::Put(vec![30])), (vec![7], Op::Put(vec![70]))];
        tree = crate::test_utils::apply_memonly(tree, &batch);
        let hash_before = tree.hash();
        assert!(!tree.is_modified());

        let batch = [(vec![99], Op::Delete)];
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        let walker = maybe_walker.expect("should be Some");
        let tree = walker.into_inner();
        assert!(deleted_keys.is_empty());
        assert!(
            !tree.is_modified(),
            "no-op delete should not dirty a committed tree"
        );
        assert_eq!(tree.hash(), hash_before);
        Ok(())
    }

    #[test]
    fn delete_promotes_committed_child_dirties_parent() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?;
        let batch = [
            (vec![3], Op::Put(vec![30])),
            (vec![6], Op::Put(vec![60])),
            (vec![7], Op::Put(vec![70])),
        ];
        tree = crate::test_utils::apply_memonly(tree, &batch);
        let hash_before = tree.hash();

        let batch = [(vec![7], Op::Delete)];
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        let walker = maybe_walker.expect("should be Some");
        let mut tree = walker.into_inner();
        assert_eq!(deleted_keys.len(), 1);
        assert!(
            tree.is_modified(),
            "parent must be dirtied when delete promotes a different child"
        );
        tree.commit();
        assert_ne!(tree.hash(), hash_before);
        crate::test_utils::assert_tree_invariants(&tree);
        Ok(())
    }

    #[test]
    fn same_value_update_does_not_dirty_tree() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?;
        let batch = [(vec![3], Op::Put(vec![30])), (vec![7], Op::Put(vec![70]))];
        tree = crate::test_utils::apply_memonly(tree, &batch);
        let hash_before = tree.hash();
        assert!(!tree.is_modified());

        let batch = [(vec![7], Op::Put(vec![70]))];
        let (maybe_walker, _) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        let walker = maybe_walker.expect("should be Some");
        let tree = walker.into_inner();
        assert!(
            !tree.is_modified(),
            "same-value update should not dirty a committed tree"
        );
        assert_eq!(tree.hash(), hash_before);
        Ok(())
    }

    #[test]
    fn delete_only_node() -> Result<()> {
        let batch = [(b"foo".to_vec(), Op::Delete)];
        let tree = Node::new(b"foo".to_vec(), b"bar".to_vec())?;
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        assert!(maybe_walker.is_none());
        assert_eq!(deleted_keys.len(), 1);
        assert_eq!(deleted_keys.front().unwrap(), b"foo");
        Ok(())
    }

    #[test]
    fn delete_deep() {
        let tree = make_tree_seq(50);
        let batch = [del_entry(5)];
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        maybe_walker.expect("should be Some");
        assert_eq!(deleted_keys.len(), 1);
        assert_eq!(*deleted_keys.front().unwrap(), seq_key(5));
    }

    #[test]
    fn delete_recursive() {
        let tree = make_tree_seq(50);
        let batch = [del_entry(29), del_entry(34)];
        let (maybe_walker, mut deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        maybe_walker.expect("should be Some");
        assert_eq!(deleted_keys.len(), 2);
        assert_eq!(deleted_keys.pop_front().unwrap(), seq_key(29));
        assert_eq!(deleted_keys.pop_front().unwrap(), seq_key(34));
    }

    #[test]
    fn delete_recursive_2() {
        let tree = make_tree_seq(10);
        let batch = [del_entry(7), del_entry(9)];
        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        maybe_walker.expect("should be Some");
        let mut deleted_keys: Vec<&Vec<u8>> = deleted_keys.iter().collect();
        deleted_keys.sort();
        assert_eq!(deleted_keys, vec![&seq_key(7), &seq_key(9)]);
    }

    #[test]
    fn rebalanced_delete() {
        let tree = make_tree_seq(7);

        let walker = Walker::new(tree, PanicSource {})
            .apply(&[(vec![0; 20], Delete)])
            .expect("apply errored")
            .0
            .unwrap();

        let batch = [
            put_entry(0),
            put_entry(1),
            put_entry(2),
            put_entry(3),
            del_entry(4),
            del_entry(5),
            del_entry(6),
        ];
        let (maybe_walker, deleted_keys) = walker.apply(&batch).expect("apply errored");
        let walker = maybe_walker.expect("should be Some");

        let mut deleted_keys: Vec<&Vec<u8>> = deleted_keys.iter().collect();
        deleted_keys.sort();
        assert_eq!(deleted_keys, vec![&seq_key(4), &seq_key(5), &seq_key(6)]);

        let mut iter = walker.tree().iter();
        assert_eq!(iter.next().unwrap().0, seq_key(0));
        assert_eq!(iter.next().unwrap().0, seq_key(1));
        assert_eq!(iter.next().unwrap().0, seq_key(2));
        assert_eq!(iter.next().unwrap().0, seq_key(3));
        assert!(iter.next().is_none());
    }

    #[cfg(not(use_box))]
    #[test]
    fn apply_empty_none() {
        let (maybe_tree, deleted_keys) =
            Walker::<PanicSource>::apply_cow(None, &[], PanicSource {}).expect("apply_to failed");
        assert!(maybe_tree.is_none());
        assert!(deleted_keys.is_empty());
    }

    #[cfg(not(use_box))]
    #[test]
    fn insert_empty_single() {
        let batch = vec![(vec![0], Op::Put(vec![1]))];
        let (maybe_tree, deleted_keys) =
            Walker::<PanicSource>::apply_cow(None, &batch, PanicSource {})
                .expect("apply_to failed");
        let tree = maybe_tree.expect("expected tree");
        assert_eq!(tree.key(), &[0]);
        assert_eq!(tree.value(), &[1]);
        assert_tree_invariants(&tree);
        assert!(deleted_keys.is_empty());
    }

    #[test]
    fn insert_root_single() -> Result<()> {
        let tree = Node::new(vec![5], vec![123])?;
        let batch = vec![(vec![6], Op::Put(vec![123]))];
        let tree = apply_memonly(tree, &batch);
        assert_eq!(tree.key(), &[5]);
        assert!(tree.child(true).is_none());
        assert_eq!(tree.child(false).expect("expected child").key(), &[6]);
        Ok(())
    }

    #[test]
    fn insert_root_double() -> Result<()> {
        let tree = Node::new(vec![5], vec![123])?;
        let batch = vec![(vec![4], Op::Put(vec![123])), (vec![6], Op::Put(vec![123]))];
        let tree = apply_memonly(tree, &batch);
        assert_eq!(tree.key(), &[5]);
        assert_eq!(tree.child(true).expect("expected child").key(), &[4]);
        assert_eq!(tree.child(false).expect("expected child").key(), &[6]);
        Ok(())
    }

    #[test]
    fn insert_rebalance() -> Result<()> {
        let tree = Node::new(vec![5], vec![123])?;

        let batch = vec![(vec![6], Op::Put(vec![123]))];
        let tree = apply_memonly(tree, &batch);

        let batch = vec![(vec![7], Op::Put(vec![123]))];
        let tree = apply_memonly(tree, &batch);

        assert_eq!(tree.key(), &[6]);
        assert_eq!(tree.child(true).expect("expected child").key(), &[5]);
        assert_eq!(tree.child(false).expect("expected child").key(), &[7]);
        Ok(())
    }

    #[test]
    fn insert_100_sequential() -> Result<()> {
        let mut tree = Node::new(vec![0], vec![123])?;

        for i in 0..100 {
            let batch = vec![(vec![i + 1], Op::Put(vec![123]))];
            tree = apply_memonly(tree, &batch);
        }

        assert_eq!(tree.key(), &[63]);
        assert_eq!(tree.child(true).expect("expected child").key(), &[31]);
        assert_eq!(tree.child(false).expect("expected child").key(), &[79]);
        Ok(())
    }

    #[test]
    fn delete_recursive_large() {
        let tree = make_tree_seq(2_500);

        let mut batch = vec![];
        for i in 500..2_000 {
            batch.push(del_entry(i));
        }

        let (maybe_walker, deleted_keys) = Walker::new(tree, PanicSource {})
            .apply(&batch)
            .expect("apply errored");
        maybe_walker.expect("should be Some");
        assert_eq!(deleted_keys.len(), 1_500);
    }

    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_update() -> Result<()> {
        let tree = make_tree_seq(100);

        let batch: Vec<BatchEntry> = vec![
            (seq_key(10), Op::Put(vec![42; 60])),
            (seq_key(20), Op::Put(vec![43; 60])),
            (seq_key(30), Op::Put(vec![44; 60])),
        ];

        let (result, _) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_single_update() -> Result<()> {
        let tree = make_tree_seq(10);

        let batch: Vec<BatchEntry> = vec![(seq_key(5), Op::Put(vec![99; 60]))];

        let (result, _) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);
        assert_eq!(tree_in_place.get(&seq_key(5)), Some(vec![99; 60]));

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_same_value_not_dirty() -> Result<()> {
        let tree = make_tree_seq(10);
        let hash_before = tree.hash();

        let mut batch: Vec<BatchEntry> = vec![
            (seq_key(3), Op::Put(put_entry_value())),
            (seq_key(7), Op::Put(put_entry_value())),
        ];

        let mut walker = Walker::new(tree, PanicSource {});
        walker.apply_in_place(&mut batch)?;
        let tree = walker.into_inner();

        assert!(
            !tree.is_modified(),
            "same-value update should not dirty tree"
        );
        assert_eq!(tree.hash(), hash_before);

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_large_tree() -> Result<()> {
        let tree = make_tree_seq(1000);

        let batch: Vec<BatchEntry> = (0..100u64)
            .map(|i| (seq_key(i * 10), Op::Put(vec![i as u8; 60])))
            .collect();

        let (result, _) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_sequential_inserts() -> Result<()> {
        let mut tree_functional = Node::new(seq_key(0), put_entry_value())?;
        tree_functional.commit();
        let mut tree_in_place = tree_functional.clone();

        for i in 1..128u64 {
            let batch = vec![put_entry(i)];

            let (result, _) = Walker::apply_cow_owned(
                Some(Walker::new(tree_functional, PanicSource {})),
                batch.clone(),
                PanicSource {},
            )?;
            tree_functional = result.unwrap();
            tree_functional.commit();

            let mut walker = Walker::new(tree_in_place, PanicSource {});
            let mut batch_mut = batch;
            walker.apply_in_place(&mut batch_mut)?;
            tree_in_place = walker.into_inner();
            tree_in_place.commit();

            assert_eq!(tree_functional.hash(), tree_in_place.hash());
            assert_tree_invariants(&tree_in_place);
        }

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_random_order_inserts() -> Result<()> {
        let insert_order = [
            42u64, 7, 75, 13, 60, 91, 3, 28, 54, 66, 82, 97, 1, 9, 21, 34, 57, 63, 70, 88, 95, 99,
            11, 24, 31, 45, 52,
        ];

        let mut tree_functional = Node::new(seq_key(insert_order[0]), put_entry_value())?;
        tree_functional.commit();
        let mut tree_in_place = tree_functional.clone();

        for key in insert_order.iter().copied().skip(1) {
            let batch = vec![put_entry(key)];

            let (result, _) = Walker::apply_cow_owned(
                Some(Walker::new(tree_functional, PanicSource {})),
                batch.clone(),
                PanicSource {},
            )?;
            tree_functional = result.unwrap();
            tree_functional.commit();

            let mut walker = Walker::new(tree_in_place, PanicSource {});
            let mut batch_mut = batch;
            walker.apply_in_place(&mut batch_mut)?;
            tree_in_place = walker.into_inner();
            tree_in_place.commit();

            assert_eq!(tree_functional.hash(), tree_in_place.hash());
            assert_tree_invariants(&tree_in_place);
        }

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_single_delete() -> Result<()> {
        let tree = make_tree_seq(10);

        let batch: Vec<BatchEntry> = vec![(seq_key(5), Op::Delete)];

        let (result, deleted_func) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        let deleted_ip = walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);
        assert_eq!(deleted_func.len(), 1);
        assert_eq!(deleted_ip.len(), 1);
        assert!(tree_in_place.get(&seq_key(5)).is_none());

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_delete_leaf() -> Result<()> {
        let tree = make_tree_seq(7);

        for key in 0..7u64 {
            let batch: Vec<BatchEntry> = vec![(seq_key(key), Op::Delete)];

            let (result, _) = Walker::apply_cow_owned(
                Some(Walker::new(tree.clone(), PanicSource {})),
                batch.clone(),
                PanicSource {},
            )?;
            let mut tree_functional = result.unwrap();
            tree_functional.commit();

            let mut walker = Walker::new(tree.clone(), PanicSource {});
            let mut batch_mut = batch;
            walker.apply_in_place(&mut batch_mut)?;
            let mut tree_in_place = walker.into_inner();
            tree_in_place.commit();

            assert_eq!(
                tree_functional.hash(),
                tree_in_place.hash(),
                "hash mismatch deleting key {}",
                key
            );
            assert_tree_invariants(&tree_in_place);
        }

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_multiple_deletes() -> Result<()> {
        let tree = make_tree_seq(50);

        let batch: Vec<BatchEntry> = vec![
            (seq_key(5), Op::Delete),
            (seq_key(15), Op::Delete),
            (seq_key(25), Op::Delete),
            (seq_key(35), Op::Delete),
            (seq_key(45), Op::Delete),
        ];

        let (result, _) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_mixed_insert_update_delete() -> Result<()> {
        let tree = make_tree_seq(100);

        let batch: Vec<BatchEntry> = vec![
            (seq_key(5), Op::Delete),
            (seq_key(10), Op::Put(vec![42; 60])),
            (seq_key(15), Op::Delete),
            (seq_key(20), Op::Put(vec![43; 60])),
            (seq_key(25), Op::Delete),
            (seq_key(100), Op::Put(vec![44; 60])),
            (seq_key(101), Op::Put(vec![45; 60])),
        ];

        let (result, _) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);
        assert!(tree_in_place.get(&seq_key(5)).is_none());
        assert_eq!(tree_in_place.get(&seq_key(10)), Some(vec![42; 60]));
        assert_eq!(tree_in_place.get(&seq_key(100)), Some(vec![44; 60]));

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_delete_large_range() -> Result<()> {
        let tree = make_tree_seq(2500);

        let batch: Vec<BatchEntry> = (500..2000u64).map(|i| (seq_key(i), Op::Delete)).collect();

        let (result, _) = Walker::apply_cow_owned(
            Some(Walker::new(tree.clone(), PanicSource {})),
            batch.clone(),
            PanicSource {},
        )?;
        let mut tree_functional = result.unwrap();
        tree_functional.commit();

        let mut walker = Walker::new(tree, PanicSource {});
        let mut batch_mut = batch;
        walker.apply_in_place(&mut batch_mut)?;
        let mut tree_in_place = walker.into_inner();
        tree_in_place.commit();

        assert_eq!(tree_functional.hash(), tree_in_place.hash());
        assert_tree_invariants(&tree_in_place);

        Ok(())
    }
    #[cfg(not(use_box))]
    #[test]
    fn in_place_parity_sequential_deletes() -> Result<()> {
        let mut tree_functional = make_tree_seq(50);
        let mut tree_in_place = tree_functional.clone();

        for i in (0..50u64).rev() {
            let batch = vec![del_entry(i)];

            let (result, _) = Walker::apply_cow_owned(
                Some(Walker::new(tree_functional, PanicSource {})),
                batch.clone(),
                PanicSource {},
            )?;

            let mut walker = Walker::new(tree_in_place, PanicSource {});
            let mut batch_mut = batch;
            let ip_result = walker.apply_in_place(&mut batch_mut);

            match result {
                Some(mut t) => {
                    t.commit();
                    tree_functional = t;

                    ip_result?;
                    let mut t = walker.into_inner();
                    t.commit();
                    assert_eq!(tree_functional.hash(), t.hash());
                    assert_tree_invariants(&t);
                    tree_in_place = t;
                }
                None => {
                    assert!(
                        ip_result.is_err(),
                        "in-place should error when tree is fully deleted"
                    );
                    return Ok(());
                }
            }
        }

        Ok(())
    }
}
