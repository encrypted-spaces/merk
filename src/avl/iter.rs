use crate::avl::node::Node;

/// An entry stored on an `Iter`'s stack, containing a reference to a `Node`,
/// and its traversal state.
///
/// The `traversed` field represents whether or not the left child, self, and
/// right child have been visited, respectively (`(left, self, right)`).
struct StackItem<'a> {
    tree: &'a Node,
    traversed: (bool, bool, bool),
}

impl<'a> StackItem<'a> {
    /// Creates a new `StackItem` for the given tree. The `traversed` state will
    /// be `false` since the children and self have not been visited yet, but
    /// will default to `true` for sides that do not have a child.
    fn new(tree: &'a Node) -> Self {
        StackItem {
            tree,
            traversed: (
                tree.child(true).is_none(),
                false,
                tree.child(false).is_none(),
            ),
        }
    }

    /// Gets a tuple to yield from an `Iter`, `(key, value)`.
    fn to_entry(&self) -> (Vec<u8>, Vec<u8>) {
        (self.tree.key().to_vec(), self.tree.value().to_vec())
    }
}

/// An iterator which yields the key/value pairs of the tree, in order, skipping
/// any parts of the tree which are pruned (not currently retained in memory).
pub struct Iter<'a> {
    stack: Vec<StackItem<'a>>,
}

impl<'a> Iter<'a> {
    /// Creates a new iterator for the given tree.
    pub fn new(tree: &'a Node) -> Self {
        let stack = vec![StackItem::new(tree)];
        Iter { stack }
    }

    /// Creates an iterator starting at the first key >= `start_key`.
    /// Walks from root to the seek position in O(log n).
    pub fn from_key(tree: &'a Node, start_key: &[u8]) -> Self {
        let mut stack = Vec::new();
        let mut current = Some(tree);
        while let Some(node) = current {
            if node.key() < start_key {
                current = node.child(false);
            } else {
                stack.push(StackItem {
                    tree: node,
                    traversed: (true, false, node.child(false).is_none()),
                });
                current = node.child(true);
            }
        }
        Iter { stack }
    }
}

impl<'a> Node {
    /// Creates an iterator which yields `(key, value)` tuples for all of the
    /// tree's nodes which are retained in memory (skipping pruned subtrees).
    pub fn iter(&'a self) -> Iter<'a> {
        Iter::new(self)
    }

    /// Creates an iterator starting at the first key >= `start_key`.
    pub fn iter_from(&'a self, start_key: &[u8]) -> Iter<'a> {
        Iter::from_key(self, start_key)
    }

    /// Creates a reverse iterator yielding entries in descending key order.
    pub fn reverse_iter(&'a self) -> ReverseIter<'a> {
        ReverseIter::new(self)
    }

    /// Creates a reverse iterator starting at the last key <= `end_key`.
    pub fn reverse_iter_from(&'a self, end_key: &[u8]) -> ReverseIter<'a> {
        ReverseIter::from_key_inclusive(self, end_key)
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (Vec<u8>, Vec<u8>);

    /// Traverses to and yields the next key/value pair, in key order.
    fn next(&mut self) -> Option<Self::Item> {
        if self.stack.is_empty() {
            return None;
        }

        let last = self.stack.last_mut().unwrap();
        if !last.traversed.0 {
            last.traversed.0 = true;
            let tree = last.tree.child(true).unwrap();
            self.stack.push(StackItem::new(tree));
            self.next()
        } else if !last.traversed.1 {
            last.traversed.1 = true;
            Some(last.to_entry())
        } else if !last.traversed.2 {
            last.traversed.2 = true;
            let tree = last.tree.child(false).unwrap();
            self.stack.push(StackItem::new(tree));
            self.next()
        } else {
            self.stack.pop();
            self.next()
        }
    }
}

/// Stack entry for reverse in-order traversal (right, self, left).
struct ReverseStackItem<'a> {
    tree: &'a Node,
    traversed: (bool, bool, bool),
}

impl<'a> ReverseStackItem<'a> {
    fn new(tree: &'a Node) -> Self {
        ReverseStackItem {
            tree,
            traversed: (
                tree.child(false).is_none(),
                false,
                tree.child(true).is_none(),
            ),
        }
    }

    fn to_entry(&self) -> (Vec<u8>, Vec<u8>) {
        (self.tree.key().to_vec(), self.tree.value().to_vec())
    }
}

/// An iterator which yields key/value pairs in descending key order.
pub struct ReverseIter<'a> {
    stack: Vec<ReverseStackItem<'a>>,
}

impl<'a> ReverseIter<'a> {
    pub fn new(tree: &'a Node) -> Self {
        let stack = vec![ReverseStackItem::new(tree)];
        ReverseIter { stack }
    }

    /// Creates a reverse iterator starting at the last key <= `end_key`.
    /// Walks from root to the seek position in O(log n).
    pub fn from_key_inclusive(tree: &'a Node, end_key: &[u8]) -> Self {
        let mut stack = Vec::new();
        let mut current = Some(tree);
        while let Some(node) = current {
            if node.key() > end_key {
                current = node.child(true);
            } else {
                stack.push(ReverseStackItem {
                    tree: node,
                    traversed: (true, false, node.child(true).is_none()),
                });
                current = node.child(false);
            }
        }
        ReverseIter { stack }
    }
}

impl<'a> Iterator for ReverseIter<'a> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.stack.is_empty() {
            return None;
        }

        let last = self.stack.last_mut().unwrap();
        if !last.traversed.0 {
            last.traversed.0 = true;
            let tree = last.tree.child(false).unwrap();
            self.stack.push(ReverseStackItem::new(tree));
            self.next()
        } else if !last.traversed.1 {
            last.traversed.1 = true;
            Some(last.to_entry())
        } else if !last.traversed.2 {
            last.traversed.2 = true;
            let tree = last.tree.child(true).unwrap();
            self.stack.push(ReverseStackItem::new(tree));
            self.next()
        } else {
            self.stack.pop();
            self.next()
        }
    }
}
