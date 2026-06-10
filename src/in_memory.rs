use std::sync::RwLock;

use crate::error::{Error, Result};
use crate::hash::{Hash, NULL_HASH};
use crate::node::Node;
use crate::ops::{Batch, BatchEntry, Op, PanicSource};
use crate::proofs::query::QueryItem;
use crate::walker::Walker;

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

    pub fn snapshot(&self) -> Option<Node> {
        self.root.read().unwrap().clone()
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
        self.apply_batch_owned(vec![(key.into(), Op::Put(value.into()))])
    }

    pub fn delete<K: Into<Vec<u8>>>(&self, key: K) -> Result<()> {
        self.apply_batch_owned(vec![(key.into(), Op::Delete)])
    }

    pub fn apply_batch(&self, batch: &Batch) -> Result<()> {
        Self::validate_batch(batch)?;

        let (maybe_tree, _) =
            Walker::apply_to_mut(self.current_walker(), &mut batch.to_vec(), PanicSource {})?;
        self.replace_root(maybe_tree);
        Ok(())
    }

    pub fn apply_batch_owned(&self, batch: Vec<BatchEntry>) -> Result<()> {
        Self::validate_batch(&batch)?;

        let (maybe_tree, _) =
            Walker::apply_to_mut(self.current_walker(), &mut batch.to_vec(), PanicSource {})?;
        self.replace_root(maybe_tree);
        Ok(())
    }

    fn validate_batch(batch: &Batch) -> Result<()> {
        for i in 1..batch.len() {
            match batch[i].0.cmp(&batch[i - 1].0) {
                std::cmp::Ordering::Less => {
                    return Err(Error::Bound("Batch keys must be sorted".into()));
                }
                std::cmp::Ordering::Equal => {
                    return Err(Error::Bound("Batch keys must be unique".into()));
                }
                std::cmp::Ordering::Greater => {}
            }
        }

        Ok(())
    }

    fn current_walker(&self) -> Option<Walker<PanicSource>> {
        let current_root = {
            let root = self.root.read().unwrap();
            root.clone()
        };
        current_root.map(|node| Walker::new(node, PanicSource {}))
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

#[cfg(test)]
#[path = "in_memory_tests.rs"]
mod tests;
