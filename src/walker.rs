use crate::child::Child;
use crate::error::{Error, Result};
use crate::hash::Hash;
use crate::node::Node;

/// A source of data to be used by the tree when encountering a pruned node.
pub trait Fetch {
    fn fetch_by_key(&self, key: &[u8]) -> Result<Option<Node>>;

    fn fetch(&self, child: &Child) -> Result<Node> {
        self.fetch_by_key_expect(child.key())
    }

    fn fetch_by_key_expect(&self, key: &[u8]) -> Result<Node> {
        self.fetch_by_key(key)?
            .ok_or_else(|| Error::Key(format!("Key does not exist: {key:?}")))
    }
}

/// Allows immutable traversal of a fully-resident `Node` tree.
/// Returns `None` for pruned or absent children.
pub struct RefWalker<'a> {
    tree: &'a Node,
}

impl<'a> RefWalker<'a> {
    pub fn new(tree: &'a Node) -> Self {
        RefWalker { tree }
    }

    pub fn tree(&self) -> &Node {
        self.tree
    }

    pub fn walk(&self, left: bool) -> Option<RefWalker<'a>> {
        self.tree.child(left).map(RefWalker::new)
    }
}

/// Allows traversal of a `Node`, fetching from the given source when traversing
/// to a pruned node, detaching children as they are traversed.
pub struct Walker<S>
where
    S: Fetch + Sized + Clone + Send,
{
    tree: Node,
    source: S,
}

impl<S> Walker<S>
where
    S: Fetch + Sized + Clone + Send,
{
    /// Creates a `Walker` with the given tree and source.
    pub fn new(tree: Node, source: S) -> Self {
        Walker { tree, source }
    }

    fn reattach_preserving_hash(&mut self, left: bool, child: Node, parent_hash: Option<Hash>) {
        let inner = self.tree.inner_mut();
        let slot = inner.child_slot_mut(left);
        debug_assert!(!slot.is_some());
        *slot = Some(Child::Resident(child));
        inner.recompute_height();
        inner.node_hash = parent_hash;
    }

    /// Similar to `Node#detach`, but yields a `Walker` which fetches from the
    /// same source as `self`. Returned tuple is `(updated_self,
    /// maybe_child_walker)`.
    pub fn detach(mut self, left: bool) -> Result<(Self, Option<Self>)> {
        let child_meta = match self.tree.child_ref(left) {
            None => return Ok((self, None)),
            Some(child) => child,
        };

        let child = if child_meta.as_resident_node().is_some() {
            let (tree, child) = self.tree.detach(left);
            self.tree = tree;
            match child {
                Some(child) => child,
                _ => unreachable!("Expected Some"),
            }
        } else {
            let pruned_child = self.tree.child_slot_mut(left).take();
            let (hash, child_heights) = match &pruned_child {
                Some(Child::Pruned(pruned)) => (*pruned.node_hash(), pruned.child_heights()),
                _ => unreachable!("Expected Some(Child::Pruned)"),
            };
            {
                let inner = self.tree.inner_mut();
                inner.node_hash = None;
                inner.recompute_height();
            }
            let mut child = self.source.fetch(&pruned_child.unwrap())?;
            {
                let child_inner = child.inner_mut();
                child_inner.node_hash = Some(hash);
                child_inner.height = 1 + std::cmp::max(child_heights.0, child_heights.1);
            }
            child
        };

        let child = self.wrap(child);
        Ok((self, Some(child)))
    }

    /// Similar to `Node#detach_expect`, but yields a `Walker` which fetches
    /// from the same source as `self`. Returned tuple is `(updated_self,
    /// child_walker)`.
    pub fn detach_expect(self, left: bool) -> Result<(Self, Self)> {
        let (walker, maybe_child) = self.detach(left)?;
        if let Some(child) = maybe_child {
            Ok((walker, child))
        } else {
            panic!(
                "Expected {} child, got None",
                if left { "left" } else { "right" }
            );
        }
    }

    /// Similar to `Node#walk`, but yields a `Walker` which fetches from the
    /// same source as `self`.
    pub fn walk<F, T>(self, left: bool, f: F) -> Result<Self>
    where
        F: FnOnce(Option<Self>) -> Result<Option<T>>,
        T: Into<Node>,
    {
        let old_child_hash = self.tree.child_ref(left).and_then(|l| match l {
            Child::Resident(t) => t.inner.node_hash,
            Child::Pruned(pruned) => Some(*pruned.node_hash()),
        });
        let parent_hash = self.tree.inner.node_hash;

        let (mut walker, maybe_child) = self.detach(left)?;
        let had_child = maybe_child.is_some();
        let new_child = f(maybe_child)?.map(|t| t.into());

        if !had_child && new_child.is_none() {
            return Ok(walker);
        }

        let new_child_hash = new_child.as_ref().and_then(|c| c.inner.node_hash);
        let child_same = had_child && old_child_hash.is_some() && new_child_hash == old_child_hash;

        if child_same {
            let child = new_child.unwrap();
            walker.reattach_preserving_hash(left, child, parent_hash);
        } else {
            walker.tree = walker.tree.attach(left, new_child);
        }
        Ok(walker)
    }

    /// Similar to `Node#walk_expect` but yields a `Walker` which fetches from
    /// the same source as `self`.
    pub fn walk_expect<F, T>(self, left: bool, f: F) -> Result<Self>
    where
        F: FnOnce(Self) -> Result<Option<T>>,
        T: Into<Node>,
    {
        let old_child_hash = self.tree.child_ref(left).and_then(|l| match l {
            Child::Resident(t) => t.inner.node_hash,
            Child::Pruned(pruned) => Some(*pruned.node_hash()),
        });
        let parent_hash = self.tree.inner.node_hash;

        let (mut walker, child) = self.detach_expect(left)?;
        let new_child = f(child)?.map(|t| t.into());

        let new_child_hash = new_child.as_ref().and_then(|c| c.inner.node_hash);
        let child_same = old_child_hash.is_some() && new_child_hash == old_child_hash;

        if child_same {
            let child = new_child.unwrap();
            walker.reattach_preserving_hash(left, child, parent_hash);
        } else {
            walker.tree = walker.tree.attach(left, new_child);
        }
        Ok(walker)
    }

    /// Returns an immutable reference to the `Node` wrapped by this walker.
    pub fn tree(&self) -> &Node {
        &self.tree
    }

    /// Returns a mutable reference to the `Node` wrapped by this walker.
    pub(crate) fn tree_mut(&mut self) -> &mut Node {
        &mut self.tree
    }

    /// Consumes the `Walker` and returns the `Node` it wraps.
    pub fn into_inner(self) -> Node {
        self.tree
    }

    /// Takes a `Node` and returns a `Walker` which fetches from the same source
    /// as `self`.
    fn wrap(&self, tree: Node) -> Self {
        Walker::new(tree, self.source.clone())
    }

    /// Returns a clone of this `Walker`'s source.
    pub fn clone_source(&self) -> S {
        self.source.clone()
    }

    /// Similar to `Node#attach`, but can also take a `Walker` since it
    /// implements `Into<Node>`.
    pub fn attach<T>(mut self, left: bool, maybe_child: Option<T>) -> Self
    where
        T: Into<Node>,
    {
        self.tree = self.tree.attach(left, maybe_child.map(|t| t.into()));
        self
    }

    /// Similar to `Node#with_value`.
    pub fn with_value(mut self, value: Vec<u8>) -> Result<Self> {
        self.tree = self.tree.with_value(value)?;
        Ok(self)
    }
}

impl<S> From<Walker<S>> for Node
where
    S: Fetch + Sized + Clone + Send,
{
    fn from(walker: Walker<S>) -> Node {
        walker.into_inner()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::child::Child;
    use crate::node::Node;

    #[derive(Clone)]
    struct MockSource {}

    impl Fetch for MockSource {
        fn fetch_by_key(&self, key: &[u8]) -> Result<Option<Node>> {
            Node::new(key.to_vec(), b"foo".to_vec()).map(Some)
        }
    }

    #[test]
    fn walk_modified() -> Result<()> {
        let tree = Node::new(b"test".to_vec(), b"abc".to_vec())?
            .attach(true, Some(Node::new(b"foo".to_vec(), b"bar".to_vec())?));

        let source = MockSource {};
        let walker = Walker::new(tree, source);

        let walker = walker
            .walk(true, |child| -> Result<Option<Node>> {
                assert_eq!(child.expect("should have child").tree().key(), b"foo");
                Ok(None)
            })
            .expect("walk failed");
        assert!(walker.into_inner().child(true).is_none());
        Ok(())
    }

    #[test]
    fn walk_stored() -> Result<()> {
        let mut tree = Node::new(b"test".to_vec(), b"abc".to_vec())?
            .attach(true, Some(Node::new(b"foo".to_vec(), b"bar".to_vec())?));
        tree.commit();

        let source = MockSource {};
        let walker = Walker::new(tree, source);

        let walker = walker
            .walk(true, |child| -> Result<Option<Node>> {
                assert_eq!(child.expect("should have child").tree().key(), b"foo");
                Ok(None)
            })
            .expect("walk failed");
        assert!(walker.into_inner().child(true).is_none());
        Ok(())
    }

    #[test]
    fn walk_pruned() {
        let tree = Node::from_fields(
            b"test".to_vec(),
            b"abc".to_vec(),
            Default::default(),
            Some(Child::pruned(b"foo".to_vec(), Default::default(), (0, 0))),
            None,
        );

        let source = MockSource {};
        let walker = Walker::new(tree, source);

        let walker = walker
            .walk_expect(true, |child| -> Result<Option<Node>> {
                assert_eq!(child.tree().key(), b"foo");
                Ok(None)
            })
            .expect("walk failed");
        assert!(walker.into_inner().child(true).is_none());
    }

    #[test]
    fn walk_none() -> Result<()> {
        let tree = Node::new(b"test".to_vec(), b"abc".to_vec())?;

        let source = MockSource {};
        let walker = Walker::new(tree, source);

        walker
            .walk(true, |child| -> Result<Option<Node>> {
                assert!(child.is_none());
                Ok(None)
            })
            .expect("walk failed");
        Ok(())
    }

    #[test]
    fn pruned_fetch_preserves_hash_and_height() {
        let expected_hash = [42u8; 32];
        let tree = Node::from_fields(
            b"root".to_vec(),
            b"val".to_vec(),
            Default::default(),
            Some(Child::pruned(b"child".to_vec(), expected_hash, (2, 1))),
            None,
        );

        let source = MockSource {};
        let walker = Walker::new(tree, source);

        let (_walker, child) = walker.detach(true).expect("detach failed");
        let child = child.expect("expected child");
        let child_tree = child.tree();
        assert!(child_tree.node_hash().is_some());
        assert_eq!(*child_tree.node_hash().unwrap(), expected_hash);
        assert_eq!(child_tree.height(), 3);
    }

    #[test]
    fn noop_walk_on_committed_tree_does_not_dirty() -> Result<()> {
        let mut tree = Node::new(b"root".to_vec(), b"root_val".to_vec())?
            .attach(
                true,
                Some(Node::new(b"left".to_vec(), b"left_val".to_vec())?),
            )
            .attach(
                false,
                Some(Node::new(b"right".to_vec(), b"right_val".to_vec())?),
            );
        tree.commit();
        let root_hash = tree.hash();
        assert!(!tree.is_modified());

        let source = MockSource {};
        let walker = Walker::new(tree, source);

        let walker = walker
            .walk(true, |child| -> Result<Option<Node>> {
                Ok(child.map(|w| w.into_inner()))
            })
            .expect("walk failed");

        let tree = walker.into_inner();
        assert!(
            !tree.is_modified(),
            "root should not be dirtied by no-op walk"
        );
        assert_eq!(tree.hash(), root_hash);
        assert!(!tree.child_ref(true).unwrap().is_modified());
        assert!(!tree.child_ref(false).unwrap().is_modified());
        Ok(())
    }
}
