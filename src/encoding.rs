use std::cmp::max;
use std::io::{Read, Write};

use crate::node::NodePtr;

use crate::child::Child;
use crate::hash::{Hash, HASH_LENGTH, NULL_HASH};
use crate::node::{Node, NodeInner};
use ed::{Decode, Encode};

impl Encode for Node {
    #[inline]
    fn encode_into<W: Write>(&self, out: &mut W) -> ed::Result<()> {
        // Encode left child
        match &self.inner.left {
            Some(child) => {
                out.write_all(&[1])?;
                child.encode_into(out)?;
            }
            None => {
                out.write_all(&[0])?;
            }
        }

        // Encode right child
        match &self.inner.right {
            Some(child) => {
                out.write_all(&[1])?;
                child.encode_into(out)?;
            }
            None => {
                out.write_all(&[0])?;
            }
        }

        // Encode kv_hash + value
        out.write_all(&self.inner.kv_hash)?;
        out.write_all(&self.inner.value)?;

        Ok(())
    }

    #[inline]
    fn encoding_length(&self) -> ed::Result<usize> {
        let left_len = match &self.inner.left {
            Some(child) => 1 + child.encoding_length()?,
            None => 1,
        };
        let right_len = match &self.inner.right {
            Some(child) => 1 + child.encoding_length()?,
            None => 1,
        };

        Ok(left_len + right_len + HASH_LENGTH + self.inner.value.len())
    }
}

impl Decode for Node {
    #[inline]
    fn decode<R: Read>(mut input: R) -> ed::Result<Self> {
        // Decode left child
        let mut left_flag = [0u8];
        input.read_exact(&mut left_flag)?;
        let left = if left_flag[0] != 0 {
            Some(Child::decode(&mut input)?)
        } else {
            None
        };

        // Decode right child
        let mut right_flag = [0u8];
        input.read_exact(&mut right_flag)?;
        let right = if right_flag[0] != 0 {
            Some(Child::decode(&mut input)?)
        } else {
            None
        };

        // Decode kv_hash + value
        let mut kv_hash: Hash = NULL_HASH;
        input.read_exact(&mut kv_hash)?;
        let mut value = Vec::with_capacity(128);
        input.read_to_end(&mut value)?;

        let height = {
            let lh = left.as_ref().map_or(0, |l| l.height());
            let rh = right.as_ref().map_or(0, |l| l.height());
            1 + max(lh, rh)
        };

        Ok(Node {
            inner: NodePtr::new(NodeInner {
                left,
                right,
                key: Vec::new(),
                value,
                kv_hash,
                node_hash: None,
                height,
            }),
        })
    }
}

impl Node {
    #[inline]
    pub fn encode(&self) -> Vec<u8> {
        // Vec-backed writes should not fail.
        Encode::encode(self).unwrap()
    }

    #[inline]
    pub fn encode_into(&self, dest: &mut Vec<u8>) {
        // Vec-backed writes should not fail.
        Encode::encode_into(self, dest).unwrap()
    }

    #[inline]
    pub fn encoding_length(&self) -> usize {
        // Length calculation only fails if a child cannot be encoded.
        Encode::encoding_length(self).unwrap()
    }

    #[inline]
    pub fn decode_into(&mut self, key: Vec<u8>, input: &[u8]) {
        Decode::decode_into(self, input).unwrap();
        let inner = self.inner_mut();
        inner.key = key;
        inner.node_hash = None;
        inner.recompute_height();
    }

    #[inline]
    pub fn decode(key: Vec<u8>, input: &[u8]) -> Node {
        let mut tree: Node = Decode::decode(input).unwrap();
        tree.inner_mut().key = key;
        tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child::Child;
    use crate::error::Result;

    #[test]
    fn encode_leaf_tree() {
        let tree = Node::from_fields(vec![0], vec![1], [55; 32], None, None);
        assert_eq!(tree.encoding_length(), 35);
        assert_eq!(
            tree.encode(),
            vec![
                0, 0, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 1,
            ]
        );
    }

    #[test]
    #[should_panic(expected = "No encoding for modified Child")]
    fn encode_modified_tree() {
        let tree = Node::from_fields(
            vec![0],
            vec![1],
            [55; 32],
            Some(Child::Resident(Node::new(vec![2], vec![3]).unwrap())),
            None,
        );
        tree.encode();
    }

    #[test]
    fn encode_stored_tree() -> Result<()> {
        let mut child = Node::new(vec![2], vec![3])?;
        let child_inner = child.inner_mut();
        child_inner.node_hash = Some([66; 32]);
        child_inner.height = 1;
        let tree = Node::from_fields(
            vec![0],
            vec![1],
            [55; 32],
            Some(Child::Resident(child)),
            None,
        );
        assert_eq!(
            tree.encode(),
            vec![
                1, 0, 1, 2, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
                66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 0, 0, 0, 55, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 55, 1
            ]
        );
        Ok(())
    }

    #[test]
    fn encode_pruned_tree() {
        let tree = Node::from_fields(
            vec![0],
            vec![1],
            [55; 32],
            Some(Child::pruned(vec![2], [66; 32], (123, 124))),
            None,
        );
        assert_eq!(tree.encoding_length(), 72);
        assert_eq!(
            tree.encode(),
            vec![
                1, 0, 1, 2, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
                66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 123, 124, 0, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 55, 55, 1
            ]
        );
    }

    #[test]
    fn decode_leaf_tree() {
        // Format: left_flag (0), right_flag (0), kv_hash (32 bytes), value
        let bytes = vec![
            0, 0, // left_flag, right_flag
            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, // kv_hash
            1,  // value
        ];
        let tree = Node::decode(vec![0], bytes.as_slice());
        assert_eq!(tree.key(), &[0]);
        assert_eq!(tree.value(), &[1]);
    }

    #[test]
    fn decode_pruned_tree() {
        let bytes = vec![
            1, 0, 1, 2, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
            66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 123, 124, 0, // left child
            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
            55, 55, 55, 55, 55, 55, 55, 55, 55, 55, // kv_hash
            1,  // value
        ];
        let tree = Node::decode(vec![0], bytes.as_slice());
        assert_eq!(tree.key(), &[0]);
        assert_eq!(tree.value(), &[1]);
        if let Some(Child::Pruned(pruned)) = tree.child_ref(true) {
            assert_eq!(pruned.key(), [2]);
            assert_eq!(pruned.child_heights(), (123_u8, 124_u8));
            assert_eq!(*pruned.node_hash(), [66_u8; 32]);
        } else {
            panic!("Expected Child::Pruned");
        }
    }
}
