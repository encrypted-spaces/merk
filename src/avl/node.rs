use std::cmp::max;
#[cfg(not(use_box))]
use std::sync::Arc;

use crate::avl::child::Child;
use crate::error::{Error, Result};
use crate::hash::{kv_hash, node_hash, Hash, Hasher, NULL_HASH};

#[cfg(not(use_box))]
pub(crate) type NodePtr = Arc<NodeInner>;
#[cfg(use_box)]
pub(crate) type NodePtr = Box<NodeInner>;

/// The fields of the `Node` type, stored on the heap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInner {
    pub(crate) left: Option<Child>,
    pub(crate) right: Option<Child>,
    pub(crate) key: Vec<u8>,
    pub(crate) value: Vec<u8>,
    pub(crate) kv_hash: Hash,
    pub(crate) node_hash: Option<Hash>,
    pub(crate) height: u8,
}

impl NodeInner {
    #[inline]
    pub(crate) fn child_slot(&self, left: bool) -> &Option<Child> {
        if left {
            &self.left
        } else {
            &self.right
        }
    }

    #[inline]
    pub(crate) fn child_slot_mut(&mut self, left: bool) -> &mut Option<Child> {
        if left {
            &mut self.left
        } else {
            &mut self.right
        }
    }

    #[inline]
    pub(crate) fn recompute_height(&mut self) {
        let lh = self.left.as_ref().map_or(0, |l| l.height());
        let rh = self.right.as_ref().map_or(0, |r| r.height());
        self.height = 1 + max(lh, rh);
    }
}

/// A binary AVL tree data structure, with Merkle hashes.
///
/// Nodes' inner fields are stored on the heap so that nodes can recursively
/// point to each other, and so we can detach nodes from their parents, then
/// reattach without allocating or freeing heap memory.
#[derive(PartialEq, Eq)]
pub struct Node {
    pub(crate) inner: NodePtr,
}

#[cfg(not(use_box))]
impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(use_box)]
impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node").field("inner", &self.inner).finish()
    }
}

impl Node {
    #[inline]
    pub(crate) fn inner_mut(&mut self) -> &mut NodeInner {
        #[cfg(not(use_box))]
        {
            Arc::make_mut(&mut self.inner)
        }
        #[cfg(use_box)]
        {
            &mut self.inner
        }
    }

    /// Creates a new `Node` with the given key and value, and no children.
    pub fn new(key: Vec<u8>, value: Vec<u8>) -> Result<Self> {
        let hash = kv_hash::<Hasher>(&key, &value)?;
        Ok(Node {
            inner: NodePtr::new(NodeInner {
                key,
                value,
                kv_hash: hash,
                left: None,
                right: None,
                node_hash: None,
                height: 1,
            }),
        })
    }

    /// Creates a `Node` by supplying all the raw struct fields (mainly useful
    /// for testing). The `kv_hash` and `Child`s are not ensured to be correct.
    pub fn from_fields(
        key: Vec<u8>,
        value: Vec<u8>,
        kv_hash_val: Hash,
        left: Option<Child>,
        right: Option<Child>,
    ) -> Node {
        let height = {
            let lh = left.as_ref().map_or(0, |l| l.height());
            let rh = right.as_ref().map_or(0, |l| l.height());
            1 + max(lh, rh)
        };
        Node {
            inner: NodePtr::new(NodeInner {
                key,
                value,
                kv_hash: kv_hash_val,
                left,
                right,
                node_hash: None,
                height,
            }),
        }
    }

    /// Returns a reference to the inner `Arc<NodeInner>` for pointer identity
    /// checks (e.g., `Arc::ptr_eq`).
    #[cfg(all(test, not(use_box)))]
    #[inline]
    pub(crate) fn inner_arc(&self) -> &Arc<NodeInner> {
        &self.inner
    }

    /// Returns the root node's key as a slice.
    #[inline]
    pub fn key(&self) -> &[u8] {
        &self.inner.key
    }

    /// Returns the root node's value as a slice.
    #[inline]
    pub fn value(&self) -> &[u8] {
        &self.inner.value
    }

    /// Returns the hash of the root node's key/value pair.
    #[inline]
    pub fn kv_hash(&self) -> &Hash {
        &self.inner.kv_hash
    }

    /// Returns a reference to the root node's `Child` on the given side, if any.
    /// If there is no child, returns `None`.
    #[inline]
    pub fn child_ref(&self, left: bool) -> Option<&Child> {
        self.inner.child_slot(left).as_ref()
    }

    /// Returns a reference to the root node's child on the given side, if any.
    /// If there is no child, returns `None`.
    #[inline]
    pub fn child(&self, left: bool) -> Option<&Self> {
        match self.child_ref(left) {
            None => None,
            Some(child) => child.as_resident_node(),
        }
    }

    /// Returns the hash of the root node's child on the given side, if any. If
    /// there is no child, returns the null hash (zero-filled).
    #[inline]
    pub fn child_hash(&self, left: bool) -> &Hash {
        self.child_ref(left)
            .map_or(&NULL_HASH, |child| child.hash())
    }

    /// Computes and returns the hash of the root node. If `node_hash` is cached,
    /// returns it; otherwise computes from the kv_hash and child hashes.
    #[inline]
    pub fn hash(&self) -> Hash {
        if let Some(h) = self.inner.node_hash {
            return h;
        }
        node_hash::<Hasher>(
            &self.inner.kv_hash,
            self.child_hash(true),
            self.child_hash(false),
        )
    }

    #[inline]
    pub fn root_hash(&self) -> Hash {
        self.hash()
    }

    /// Returns the cached `node_hash`, if any.
    #[inline]
    pub fn node_hash(&self) -> Option<&Hash> {
        self.inner.node_hash.as_ref()
    }

    /// Returns `true` if this node has been modified (its `node_hash` is stale).
    #[inline]
    pub fn is_modified(&self) -> bool {
        self.inner.node_hash.is_none()
    }

    /// Returns the height of the child on the given side, if any. If there is
    /// no child, returns 0.
    #[inline]
    pub fn child_height(&self, left: bool) -> u8 {
        self.child_ref(left).map_or(0, |child| child.height())
    }

    #[inline]
    pub fn child_heights(&self) -> (u8, u8) {
        (self.child_height(true), self.child_height(false))
    }

    /// Returns the height of the tree (the number of levels). For example, a
    /// single node has height 1, a node with a single descendant has height 2,
    /// etc.
    #[inline]
    pub fn height(&self) -> u8 {
        self.inner.height
    }

    /// Returns the balance factor of the root node. This is the difference
    /// between the height of the right child (if any) and the height of the
    /// left child (if any). For example, a balance factor of 2 means the right
    /// subtree is 2 levels taller than the left subtree.
    #[inline]
    pub fn balance_factor(&self) -> i8 {
        let left_height = self.child_height(true) as i8;
        let right_height = self.child_height(false) as i8;
        right_height - left_height
    }

    /// Attaches the child (if any) to the root node on the given side. Creates
    /// a `Child::Resident` which contains the child.
    ///
    /// Panics if there is already a child on the given side.
    #[inline]
    pub fn attach(mut self, left: bool, maybe_child: Option<Self>) -> Self {
        debug_assert_ne!(
            Some(self.key()),
            maybe_child.as_ref().map(|c| c.key()),
            "Tried to attach tree with same key"
        );

        let inner = self.inner_mut();
        let slot = inner.child_slot_mut(left);
        assert!(
            !slot.is_some(),
            "Tried to attach to {} tree slot, but it is already Some",
            side_to_str(left)
        );

        *slot = maybe_child.map(Child::Resident);
        inner.node_hash = None;
        inner.recompute_height();

        self
    }

    /// Detaches the child on the given side (if any) from the root node, and
    /// returns `(root_node, maybe_child)`.
    ///
    /// One will usually want to reattach (see `attach`) a child on the same
    /// side after applying some operation to the detached child.
    ///
    /// Note: This method preserves `Child::Pruned` in the slot (returning `None`
    /// for the child). This is safe for sparse trees where some nodes are pruned.
    ///
    /// When a resident child is actually removed, sets `node_hash = None` and
    /// recomputes `height`.
    #[inline]
    pub fn detach(mut self, left: bool) -> (Self, Option<Self>) {
        if !matches!(self.child_ref(left), Some(Child::Resident(_))) {
            return (self, None);
        }

        let inner = self.inner_mut();
        let child = match inner.child_slot_mut(left).take() {
            Some(Child::Resident(tree)) => Some(tree),
            _ => unreachable!("resident child checked before detach"),
        };
        inner.node_hash = None;
        inner.recompute_height();

        (self, child)
    }

    /// Detaches the child on the given side from the root node, and
    /// returns `(root_node, child)`.
    ///
    /// Panics if there is no child on the given side.
    ///
    /// One will usually want to reattach (see `attach`) a child on the same
    /// side after applying some operation to the detached child.
    #[inline]
    pub fn detach_expect(self, left: bool) -> (Self, Self) {
        let (parent, maybe_child) = self.detach(left);

        if let Some(child) = maybe_child {
            (parent, child)
        } else {
            panic!(
                "Expected tree to have {} child, but got None",
                side_to_str(left)
            );
        }
    }

    /// Detaches the child on the given side and passes it into `f`, which must
    /// return a new child (either the same child, a new child to take its
    /// place, or `None` to explicitly keep the slot empty).
    ///
    /// This is the same as `detach`, but with the function interface to enforce
    /// at compile-time that an explicit final child value is returned. This is
    /// less error prone that detaching with `detach` and reattaching with
    /// `attach`.
    ///
    /// When no child was detached and the callback returns `None`, the tree is
    /// left unchanged (pruned slots are preserved, no dirtying). When the
    /// returned child has the same `node_hash` as the original, the parent's
    /// `node_hash` is reinstated (the walk was a no-op from the Merkle
    /// perspective).
    #[inline]
    pub fn walk<F>(self, left: bool, f: F) -> Self
    where
        F: FnOnce(Option<Self>) -> Option<Self>,
    {
        let old_child_hash = self.child_ref(left).and_then(|l| match l {
            Child::Resident(t) => t.inner.node_hash,
            Child::Pruned(_) => None,
        });
        let parent_hash = self.inner.node_hash;

        let (mut tree, maybe_child) = self.detach(left);
        let had_child = maybe_child.is_some();
        let new_child = f(maybe_child);

        if !had_child && new_child.is_none() {
            return tree;
        }

        if !had_child {
            tree.child_slot_mut(left).take();
        }

        let new_child_hash = new_child.as_ref().and_then(|c| c.inner.node_hash);
        let child_same = had_child && old_child_hash.is_some() && new_child_hash == old_child_hash;

        if child_same {
            let child = new_child.unwrap();
            let inner = tree.inner_mut();
            let slot = inner.child_slot_mut(left);
            debug_assert!(!slot.is_some());
            *slot = Some(Child::Resident(child));
            inner.recompute_height();
            inner.node_hash = parent_hash;
            tree
        } else {
            tree.attach(left, new_child)
        }
    }

    /// Like `walk`, but panics if there is no child on the given side.
    #[inline]
    pub fn walk_expect<F>(self, left: bool, f: F) -> Self
    where
        F: FnOnce(Self) -> Option<Self>,
    {
        let old_child_hash = match self.child_ref(left) {
            Some(Child::Resident(t)) => t.inner.node_hash,
            _ => None,
        };
        let parent_hash = self.inner.node_hash;

        let (mut tree, child) = self.detach_expect(left);
        let new_child = f(child);

        let new_child_hash = new_child.as_ref().and_then(|c| c.inner.node_hash);
        let child_same = old_child_hash.is_some() && new_child_hash == old_child_hash;

        if child_same {
            let child = new_child.unwrap();
            let inner = tree.inner_mut();
            let slot = inner.child_slot_mut(left);
            debug_assert!(!slot.is_some());
            *slot = Some(Child::Resident(child));
            inner.recompute_height();
            inner.node_hash = parent_hash;
            tree
        } else {
            tree.attach(left, new_child)
        }
    }

    /// Returns a mutable reference to the child slot for the given side.
    #[inline]
    pub(crate) fn child_slot_mut(&mut self, left: bool) -> &mut Option<Child> {
        self.inner_mut().child_slot_mut(left)
    }

    /// Replaces the root node's value with the given value and returns the
    /// modified `Node`. If the new value produces the same `kv_hash`, the
    /// node is not dirtied.
    #[inline]
    pub fn with_value(mut self, value: Vec<u8>) -> Result<Self> {
        let new_kv_hash = kv_hash::<Hasher>(self.key(), value.as_slice())?;
        if self.value() == value.as_slice() && *self.kv_hash() == new_kv_hash {
            return Ok(self);
        }

        let inner = self.inner_mut();
        let old_kv_hash = inner.kv_hash;
        inner.kv_hash = new_kv_hash;
        inner.value = value;
        if inner.kv_hash != old_kv_hash {
            inner.node_hash = None;
        }
        Ok(self)
    }

    /// Recomputes hashes for all modified nodes (those with `node_hash: None`).
    pub fn commit(&mut self) {
        if self.inner.node_hash.is_some() {
            return;
        }

        {
            let inner = self.inner_mut();
            if let Some(Child::Resident(ref mut child)) = inner.left {
                child.commit();
            }
            if let Some(Child::Resident(ref mut child)) = inner.right {
                child.commit();
            }
        }

        let left_hash = *self.child_hash(true);
        let right_hash = *self.child_hash(false);

        let inner = self.inner_mut();
        inner.node_hash = Some(node_hash::<Hasher>(&inner.kv_hash, &left_hash, &right_hash));
    }

    pub fn get_value(&self, key: &[u8]) -> Result<GetResult> {
        let mut cursor = self;

        loop {
            if key == cursor.key() {
                return Ok(GetResult::Found(cursor.value().to_vec()));
            }

            let left = key < cursor.key();
            let child = match cursor.child_ref(left) {
                None => return Ok(GetResult::NotFound),
                Some(child) => child,
            };

            let maybe_child = child.as_resident_node();
            match maybe_child {
                None => return Ok(GetResult::Pruned),
                Some(child) => cursor = child,
            }
        }
    }

    pub fn get_result(&self, key: &[u8]) -> Result<GetResult> {
        self.get_value(key)
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        match self.get_value(key) {
            Ok(GetResult::Found(v)) => Some(v),
            Ok(GetResult::NotFound) => None,
            Ok(GetResult::Pruned) => panic!("Unexpected pruned node"),
            Err(e) => panic!("get failed: {}", e),
        }
    }

    pub fn prove<Q, I>(&self, query: I) -> Result<Vec<u8>>
    where
        Q: Into<crate::proofs::query::QueryItem>,
        I: IntoIterator<Item = Q>,
    {
        crate::proofs::query::prove_resident(Some(self), query)
    }

    pub fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        if end.is_some_and(|end| start >= end) {
            return Ok(Vec::new());
        }

        let mut out = Vec::new();
        self.collect_range_inner(start, end, &mut out)?;
        Ok(out)
    }

    pub fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let end = prefix_successor(prefix);
        self.collect_range(prefix, end.as_deref())
    }

    fn collect_range_inner(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
        out: &mut Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<()> {
        if self.key() > start {
            self.collect_range_child(true, start, end, out)?;
        }

        if self.key() >= start && end.is_none_or(|end| self.key() < end) {
            out.push((self.key().to_vec(), self.value().to_vec()));
        }

        if end.is_none_or(|end| self.key() < end) {
            self.collect_range_child(false, start, end, out)?;
        }

        Ok(())
    }

    fn collect_range_child(
        &self,
        left: bool,
        start: &[u8],
        end: Option<&[u8]>,
        out: &mut Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<()> {
        match self.child_ref(left) {
            Some(Child::Resident(child)) => child.collect_range_inner(start, end, out),
            Some(Child::Pruned(pruned)) => Err(Error::PrunedNode(format!(
                "AVL snapshot range scan descended into pruned node {:?}",
                pruned.key()
            ))),
            None => Ok(()),
        }
    }
}

fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(&last) = end.last() {
        if last < 0xff {
            *end.last_mut().expect("last checked above") += 1;
            return Some(end);
        }
        end.pop();
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GetResult {
    Found(Vec<u8>),
    Pruned,
    NotFound,
}

pub fn side_to_str(left: bool) -> &'static str {
    if left {
        "left"
    } else {
        "right"
    }
}

#[cfg(test)]
mod test {
    use super::Node;
    use crate::avl::child::Child;
    use crate::error::Result;
    use crate::hash::NULL_HASH;

    #[test]
    fn build_tree() -> Result<()> {
        let tree = Node::new(vec![1], vec![101])?;
        assert_eq!(tree.key(), &[1]);
        assert_eq!(tree.value(), &[101]);
        assert!(tree.child(true).is_none());
        assert!(tree.child(false).is_none());

        let tree = tree.attach(true, None);
        assert!(tree.child(true).is_none());
        assert!(tree.child(false).is_none());

        let tree = tree.attach(true, Some(Node::new(vec![2], vec![102])?));
        assert_eq!(tree.key(), &[1]);
        assert_eq!(tree.child(true).unwrap().key(), &[2]);
        assert!(tree.child(false).is_none());

        let tree = Node::new(vec![3], vec![103])?.attach(false, Some(tree));
        assert_eq!(tree.key(), &[3]);
        assert_eq!(tree.child(false).unwrap().key(), &[1]);
        assert!(tree.child(true).is_none());
        Ok(())
    }

    #[should_panic]
    #[test]
    fn attach_existing() {
        Node::new(vec![0], vec![1])
            .expect("tree construction failed")
            .attach(
                true,
                Some(Node::new(vec![2], vec![3]).expect("tree construction failed")),
            )
            .attach(
                true,
                Some(Node::new(vec![4], vec![5]).expect("tree construction failed")),
            );
    }

    #[test]
    fn modify() -> Result<()> {
        let tree = Node::new(vec![0], vec![1])?
            .attach(true, Some(Node::new(vec![2], vec![3])?))
            .attach(false, Some(Node::new(vec![4], vec![5])?));

        let tree = tree.walk(true, |left_opt| {
            assert_eq!(left_opt.as_ref().unwrap().key(), &[2]);
            None
        });
        assert!(tree.child(true).is_none());
        assert!(tree.child(false).is_some());
        let fixed_tree = Some(Node::new(vec![2], vec![3])?);
        let tree = tree.walk(true, |left_opt| {
            assert!(left_opt.is_none());
            fixed_tree
        });
        assert_eq!(tree.child_ref(true).unwrap().key(), &[2]);

        let tree = tree.walk_expect(false, |right| {
            assert_eq!(right.key(), &[4]);
            None
        });
        assert!(tree.child(true).is_some());
        assert!(tree.child(false).is_none());
        Ok(())
    }

    #[test]
    fn child_accessors() -> Result<()> {
        let mut tree =
            Node::new(vec![0], vec![1])?.attach(true, Some(Node::new(vec![2], vec![3])?));
        assert!(tree.child_ref(true).expect("expected child").is_modified());
        assert!(tree.child(true).is_some());
        assert!(tree.child_ref(false).is_none());
        assert!(tree.child(false).is_none());

        tree.commit();
        assert!(!tree.child_ref(true).expect("expected child").is_modified());
        assert!(tree.child(true).is_some());

        let tree = tree.walk(true, |_| None);
        assert!(tree.child_ref(true).is_none());
        assert!(tree.child(true).is_none());
        Ok(())
    }

    #[test]
    fn child_hash() -> Result<()> {
        let mut tree =
            Node::new(vec![0], vec![1])?.attach(true, Some(Node::new(vec![2], vec![3])?));
        tree.commit();
        let mut expected_child = Node::new(vec![2], vec![3])?;
        expected_child.commit();
        assert_eq!(tree.child_hash(true), &expected_child.hash());
        assert_eq!(tree.child_hash(false), &NULL_HASH);
        Ok(())
    }

    #[test]
    fn hash() -> Result<()> {
        use crate::hash::{kv_hash, node_hash, Hasher};
        let tree = Node::new(vec![0], vec![1])?;
        let expected_kv = kv_hash::<Hasher>(&[0], &[1]).unwrap();
        let expected = node_hash::<Hasher>(&expected_kv, &NULL_HASH, &NULL_HASH);
        assert_eq!(tree.hash(), expected);
        Ok(())
    }

    #[test]
    fn height_and_balance() -> Result<()> {
        let tree = Node::new(vec![0], vec![1])?;
        assert_eq!(tree.height(), 1);
        assert_eq!(tree.child_height(true), 0);
        assert_eq!(tree.child_height(false), 0);
        assert_eq!(tree.balance_factor(), 0);

        let tree = tree.attach(true, Some(Node::new(vec![2], vec![3])?));
        assert_eq!(tree.height(), 2);
        assert_eq!(tree.child_height(true), 1);
        assert_eq!(tree.child_height(false), 0);
        assert_eq!(tree.balance_factor(), -1);

        let (tree, maybe_child) = tree.detach(true);
        let tree = tree.attach(false, maybe_child);
        assert_eq!(tree.height(), 2);
        assert_eq!(tree.child_height(true), 0);
        assert_eq!(tree.child_height(false), 1);
        assert_eq!(tree.balance_factor(), 1);
        Ok(())
    }

    #[test]
    fn commit() -> Result<()> {
        let mut tree =
            Node::new(vec![0], vec![1])?.attach(false, Some(Node::new(vec![2], vec![3])?));
        tree.commit();

        assert!(!tree.child_ref(false).expect("expected child").is_modified());
        Ok(())
    }

    #[test]
    fn node_hash_none_for_new() -> Result<()> {
        let tree = Node::new(vec![0], vec![1])?;
        assert!(tree.is_modified());
        assert!(tree.node_hash().is_none());
        Ok(())
    }

    #[test]
    fn node_hash_set_after_commit() -> Result<()> {
        let mut tree = Node::new(vec![0], vec![1])?;
        tree.commit();
        assert!(!tree.is_modified());
        assert!(tree.node_hash().is_some());
        assert_eq!(*tree.node_hash().unwrap(), tree.hash());
        Ok(())
    }

    #[test]
    fn node_hash_cleared_on_value_change() -> Result<()> {
        let mut tree = Node::new(vec![0], vec![1])?;
        tree.commit();
        assert!(!tree.is_modified());
        let tree = tree.with_value(vec![2])?;
        assert!(tree.is_modified());
        Ok(())
    }

    #[test]
    fn height_correct_after_attach_detach() -> Result<()> {
        let tree = Node::new(vec![5], vec![50])?;
        assert_eq!(tree.height(), 1);

        let child = Node::new(vec![3], vec![30])?.attach(true, Some(Node::new(vec![1], vec![10])?));
        assert_eq!(child.height(), 2);

        let tree = tree.attach(true, Some(child));
        assert_eq!(tree.height(), 3);

        let (tree, _child) = tree.detach(true);
        assert_eq!(tree.height(), 1);
        Ok(())
    }

    #[test]
    fn modified_descendants_propagate_none() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?
            .attach(true, Some(Node::new(vec![3], vec![30])?))
            .attach(false, Some(Node::new(vec![7], vec![70])?));
        tree.commit();
        assert!(!tree.is_modified());

        let tree = tree.walk(true, |child| child.map(|c| c.with_value(vec![31]).unwrap()));
        assert!(tree.is_modified());
        assert!(tree.child_ref(true).unwrap().is_modified());
        assert!(!tree.child_ref(false).unwrap().is_modified());
        Ok(())
    }

    #[test]
    fn noop_walk_does_not_dirty() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?
            .attach(true, Some(Node::new(vec![3], vec![30])?))
            .attach(false, Some(Node::new(vec![7], vec![70])?));
        tree.commit();
        let hash_before = tree.hash();
        assert!(!tree.is_modified());

        let tree = tree.walk(true, |child| child);
        assert!(!tree.is_modified());
        assert_eq!(tree.hash(), hash_before);

        let tree = tree.walk(false, |child| child);
        assert!(!tree.is_modified());
        assert_eq!(tree.hash(), hash_before);
        Ok(())
    }

    #[test]
    fn noop_walk_on_empty_slot_does_not_dirty() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?;
        tree.commit();
        assert!(!tree.is_modified());
        let hash_before = tree.hash();

        let tree = tree.walk(true, |child| {
            assert!(child.is_none());
            None
        });
        assert!(!tree.is_modified());
        assert_eq!(tree.hash(), hash_before);
        Ok(())
    }

    #[test]
    fn walk_on_pruned_child_preserves_slot() -> Result<()> {
        let tree = Node::from_fields(
            b"root".to_vec(),
            b"val".to_vec(),
            Default::default(),
            Some(Child::pruned(b"left".to_vec(), [77; 32], (0, 0))),
            None,
        );

        let tree = tree.walk(true, |child| {
            assert!(child.is_none());
            None
        });

        assert!(tree.child_ref(true).unwrap().is_pruned());
        assert_eq!(tree.child_ref(true).unwrap().key(), b"left");
        Ok(())
    }

    #[test]
    fn detach_dirties_parent() -> Result<()> {
        let mut tree =
            Node::new(vec![5], vec![50])?.attach(true, Some(Node::new(vec![3], vec![30])?));
        tree.commit();
        assert!(!tree.is_modified());

        let (tree, _child) = tree.detach(true);
        assert!(tree.is_modified());
        Ok(())
    }

    #[test]
    fn walk_with_different_committed_child_dirties() -> Result<()> {
        let mut tree =
            Node::new(vec![5], vec![50])?.attach(true, Some(Node::new(vec![3], vec![30])?));
        tree.commit();
        assert!(!tree.is_modified());

        let mut replacement = Node::new(vec![2], vec![20])?;
        replacement.commit();

        let tree = tree.walk(true, |_old| Some(replacement));
        assert!(tree.is_modified());
        Ok(())
    }

    #[test]
    fn same_value_update_does_not_dirty() -> Result<()> {
        let mut tree = Node::new(vec![5], vec![50])?;
        tree.commit();
        let hash_before = tree.hash();
        assert!(!tree.is_modified());

        let tree = tree.with_value(vec![50])?;
        assert!(!tree.is_modified());
        assert_eq!(tree.hash(), hash_before);
        Ok(())
    }
}
