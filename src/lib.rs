//! A high-performance Merkle key/value store.
//!
//! Merk is a crypto key/value store - more specifically, it's an in-memory
//! Merkle AVL tree with copy-on-write nodes for efficient snapshots.

mod child;
mod encoding;
/// Error and Result types.
mod error;
mod hash;
pub mod in_memory;
mod iter;
mod node;
mod ops;
/// Algorithms for generating and verifying Merkle proofs.
pub mod proofs;
mod walker;

#[cfg(test)]
mod fuzz_tests;
#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

pub use in_memory::InMemoryMerk;

pub use child::{Child, PrunedNode};
pub use error::{Error, Result};
pub use hash::{kv_hash, node_hash, zkvm_hash_tests, Hash, Hasher, HASH_LENGTH, NULL_HASH};
pub use node::{GetResult, Node, NodeInner};
pub use ops::{Batch, BatchEntry, Op, PanicSource};
pub use proofs::query::{prove_resident, verify, verify_query};
pub use walker::{Fetch, RefWalker, Walker};
