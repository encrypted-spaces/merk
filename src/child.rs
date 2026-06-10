use std::cmp::max;

use std::io::{Read, Write};

use ed::{Decode, Encode, Result, Terminated};

use crate::hash::Hash;
use crate::node::Node;

/// Metadata for a child node that has been pruned from memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrunedNode {
    pub(crate) key: Vec<u8>,
    pub(crate) node_hash: Hash,
    pub(crate) child_heights: (u8, u8),
}

impl PrunedNode {
    #[inline]
    pub fn key(&self) -> &[u8] {
        self.key.as_slice()
    }

    #[inline]
    pub fn node_hash(&self) -> &Hash {
        &self.node_hash
    }

    #[inline]
    pub fn child_heights(&self) -> (u8, u8) {
        self.child_heights
    }

    #[inline]
    pub fn height(&self) -> u8 {
        1 + max(self.child_heights.0, self.child_heights.1)
    }

    #[inline]
    pub fn balance_factor(&self) -> i8 {
        self.child_heights.1 as i8 - self.child_heights.0 as i8
    }
}

/// Represents a child node. Children are either resident (the child `Node` is
/// in memory) or pruned (only metadata is retained, keyed for later fetching).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Child {
    /// Represents a child tree node that is currently loaded in memory.
    /// Whether the subtree's hash is up-to-date is tracked by the child
    /// tree's `node_hash` field (`None` = modified, `Some` = hash-current).
    Resident(Node),

    /// Represents a child tree node which has been pruned from memory.
    Pruned(PrunedNode),
}

impl Child {
    /// Creates a pruned child from its key, node hash, and child heights.
    #[inline]
    pub fn pruned(key: Vec<u8>, node_hash: Hash, child_heights: (u8, u8)) -> Self {
        Child::Pruned(PrunedNode {
            key,
            node_hash,
            child_heights,
        })
    }

    /// Returns `true` if this child is pruned.
    #[inline]
    pub fn is_pruned(&self) -> bool {
        matches!(self, Child::Pruned(_))
    }

    /// Returns `true` if the resident child has been modified (its
    /// `node_hash` is stale/missing). Always `false` for pruned references.
    #[inline]
    pub fn is_modified(&self) -> bool {
        match self {
            Child::Pruned(_) => false,
            Child::Resident(tree) => tree.inner.node_hash.is_none(),
        }
    }

    /// Returns the key of the node represented by this child, as a slice.
    #[inline]
    pub fn key(&self) -> &[u8] {
        match self {
            Child::Pruned(pruned) => pruned.key(),
            Child::Resident(tree) => tree.key(),
        }
    }

    /// Returns the resident `Node`, if this child is resident.
    #[inline]
    pub fn as_resident_node(&self) -> Option<&Node> {
        match self {
            Child::Resident(node) => Some(node),
            Child::Pruned(_) => None,
        }
    }

    /// Returns the hash of the node represented by this child. Panics if the
    /// resident child has not been committed (`node_hash` is `None`).
    #[inline]
    pub fn hash(&self) -> &Hash {
        match self {
            Child::Pruned(pruned) => pruned.node_hash(),
            Child::Resident(tree) => tree
                .inner
                .node_hash
                .as_ref()
                .expect("Cannot get hash from modified child"),
        }
    }

    /// Returns the height of the subtree represented by this child.
    #[inline]
    pub fn height(&self) -> u8 {
        match self {
            Child::Pruned(pruned) => pruned.height(),
            Child::Resident(tree) => tree.inner.height,
        }
    }

    /// Returns the balance factor of the subtree represented by this child.
    #[inline]
    pub fn balance_factor(&self) -> i8 {
        match self {
            Child::Pruned(pruned) => pruned.balance_factor(),
            Child::Resident(tree) => tree.balance_factor(),
        }
    }
}

impl Encode for Child {
    #[inline]
    fn encode_into<W: Write>(&self, out: &mut W) -> Result<()> {
        match self {
            Child::Pruned(pruned) => {
                let (left_height, right_height) = pruned.child_heights();
                let key = pruned.key();
                debug_assert!(key.len() < 65536, "Key length must be less than 65536");
                out.write_all(&(key.len() as u16).to_be_bytes())?;
                out.write_all(key)?;
                out.write_all(pruned.node_hash())?;
                out.write_all(&[left_height, right_height])?;
            }
            Child::Resident(tree) => {
                let hash = tree
                    .inner
                    .node_hash
                    .as_ref()
                    .expect("No encoding for modified Child");
                let key = tree.key();
                let (left_height, right_height) = tree.child_heights();
                debug_assert!(key.len() < 65536, "Key length must be less than 65536");
                out.write_all(&(key.len() as u16).to_be_bytes())?;
                out.write_all(key)?;
                out.write_all(hash)?;
                out.write_all(&[left_height, right_height])?;
            }
        }
        Ok(())
    }

    #[inline]
    fn encoding_length(&self) -> Result<usize> {
        let key_len = match self {
            Child::Pruned(pruned) => pruned.key().len(),
            Child::Resident(tree) => {
                assert!(
                    tree.inner.node_hash.is_some(),
                    "No encoding for modified Child"
                );
                tree.key().len()
            }
        };
        debug_assert!(key_len < 65536, "Key length must be less than 65536");
        // 2 (key_len u16) + key + 32 (hash) + 2 (child heights)
        Ok(2 + key_len + 32 + 2)
    }
}

impl Child {
    #[inline]
    fn default_pruned() -> Self {
        Child::Pruned(PrunedNode {
            key: Vec::with_capacity(64),
            node_hash: Default::default(),
            child_heights: (0, 0),
        })
    }
}

impl Decode for Child {
    #[inline]
    fn decode<R: Read>(input: R) -> Result<Child> {
        let mut child = Child::default_pruned();
        Child::decode_into(&mut child, input)?;
        Ok(child)
    }

    #[inline]
    fn decode_into<R: Read>(&mut self, mut input: R) -> Result<()> {
        if !self.is_pruned() {
            *self = Child::default_pruned();
        }

        if let Child::Pruned(pruned) = self {
            let length = read_u16(&mut input)? as usize;

            pruned.key.resize(length, 0);
            input.read_exact(pruned.key.as_mut())?;

            input.read_exact(&mut pruned.node_hash[..])?;

            pruned.child_heights.0 = read_u8(&mut input)?;
            pruned.child_heights.1 = read_u8(&mut input)?;
        } else {
            unreachable!()
        }

        Ok(())
    }
}

impl Terminated for Child {}

#[inline]
fn read_u16<R: Read>(mut input: R) -> Result<u16> {
    let mut length = [0, 0];
    input.read_exact(length.as_mut())?;
    Ok(u16::from_be_bytes(length))
}

#[inline]
fn read_u8<R: Read>(mut input: R) -> Result<u8> {
    let mut length = [0];
    input.read_exact(length.as_mut())?;
    Ok(length[0])
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hash::NULL_HASH;
    use crate::node::Node;

    #[test]
    fn types() -> std::result::Result<(), crate::error::Error> {
        let hash = NULL_HASH;
        let child_heights = (0, 0);
        let key = vec![0];
        let tree = || Node::new(vec![0], vec![1]);

        let pruned = Child::pruned(key, hash, child_heights);

        // A Resident child whose tree has node_hash: None is "modified"
        let modified = Child::Resident(tree()?);

        // A Resident child whose tree has node_hash: Some(_) is "stored"
        let mut stored_tree = tree()?;
        stored_tree.inner_mut().node_hash = Some(hash);
        let stored = Child::Resident(stored_tree);

        assert!(pruned.is_pruned());
        assert!(!pruned.is_modified());
        assert!(pruned.as_resident_node().is_none());
        assert_eq!(pruned.hash(), &[0; 32]);
        assert_eq!(pruned.height(), 1);

        assert!(!modified.is_pruned());
        assert!(modified.is_modified());
        assert!(modified.as_resident_node().is_some());
        assert_eq!(modified.height(), 1);

        assert!(!stored.is_pruned());
        assert!(!stored.is_modified());
        assert!(stored.as_resident_node().is_some());
        assert_eq!(stored.hash(), &[0; 32]);
        assert_eq!(stored.height(), 1);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Cannot get hash from modified child")]
    fn modified_hash() {
        Node::new(vec![0], vec![1])
            .map(Child::Resident)
            .map(|child| child.hash().to_vec())
            .map(|_| ())
            .unwrap_or_default()
    }

    #[test]
    fn encode_child() {
        let child = Child::pruned(vec![1, 2, 3], [55; 32], (123, 124));
        assert_eq!(child.encoding_length().unwrap(), 39);

        let mut bytes = vec![];
        child.encode_into(&mut bytes).unwrap();
        assert_eq!(bytes.len(), child.encoding_length().unwrap());
        assert_eq!(
            bytes,
            vec![
                0, 3, 1, 2, 3, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 123, 124
            ]
        );
    }

    #[test]
    fn encode_child_long_key_valid() {
        let child = Child::pruned(vec![123; 60_000], [55; 32], (123, 124));
        let mut bytes = vec![];
        child.encode_into(&mut bytes).unwrap();

        let decoded = Child::decode(&bytes[..]).unwrap();
        assert_eq!(decoded, child);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic = "Key length must be less than 65536"]
    fn encode_child_long_key_invalid() {
        let child = Child::pruned(vec![123; 70_000], [55; 32], (123, 124));
        let mut bytes = vec![];
        child.encode_into(&mut bytes).unwrap();
    }
}
