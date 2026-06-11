//! A high-performance Merkle key/value store.
//!
//! Merk is a crypto key/value store - more specifically, it's an in-memory
//! Merkle AVL tree with copy-on-write nodes for efficient snapshots.

pub mod avl;
/// Error and Result types.
mod error;
mod hash;
pub mod mrt;
mod ops;
/// Algorithms for generating and verifying Merkle proofs.
pub mod proofs;
/// Shared transcript vocabulary and traced-handle traits.
pub mod tracer;

#[cfg(test)]
mod fuzz_tests;
#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

// The two backends are module-scoped peers: `merk::avl` (this AVL store) and
// `merk::mrt`. Shared, backend-agnostic surface stays flat below.
pub use avl::GetResult;
pub use avl::PanicSource;
pub use error::{Error, Result, UnsupportedFeature};
pub use hash::{kv_hash, node_hash, zkvm_hash_tests, Hash, Hasher, HASH_LENGTH, NULL_HASH};
pub use ops::{Batch, BatchEntry, Op};
pub use tracer::{TraceInterface, TraceReader};
