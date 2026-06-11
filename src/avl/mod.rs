//! AVL backend — the original balanced-tree merk, addressed as a module-scoped
//! peer of [`crate::mrt`].
//!
//! This is the public facade for the AVL backend. The implementation lives in
//! this module's internal submodules; `merk::avl` parallels `merk::mrt`, so a
//! cfg-switched consumer can `use merk::avl as backend` / `use merk::mrt as
//! backend` and share verify-path code.
//!
//! Backend-agnostic vocabulary deliberately stays out of here: the op types
//! (`BatchOp`, `ReadOp`, `ProvenRead`, …) and the traced-handle traits
//! ([`TraceReader`](crate::tracer::TraceReader) /
//! [`TraceInterface`](crate::tracer::TraceInterface)) remain in
//! [`crate::tracer`], and the query types (`Query`, `QueryItem`) remain in
//! [`crate::proofs::query`].
//!
//! Proving and verification are externalized — the trace *is* the proof: record
//! a transcript through [`TraceRecorder`], then verify by decoding the trace and
//! replaying the same operations against a [`TraceReplayer`] / [`TraceVerifier`],
//! mirroring [`crate::mrt`]. There is no self-contained proof object.

pub(crate) mod child;
pub(crate) mod encoding;
pub(crate) mod in_memory;
pub(crate) mod iter;
pub(crate) mod node;
pub(crate) mod ops;
pub(crate) mod tracer;
pub(crate) mod walker;

/// Point-in-time read snapshot (peer of [`crate::mrt::Checkpoint`]).
pub use in_memory::Checkpoint;
/// The live AVL store (was `merk::InMemoryMerk`).
pub use in_memory::InMemoryMerk as Tree;
/// The AVL tree node — also the read/snapshot surface (no separate snapshot type).
pub use node::{GetResult, Node, NodeInner};

pub use child::{Child, PrunedNode};
pub use ops::PanicSource;
pub use walker::{Fetch, RefWalker, Walker};

/// One-shot / `&Query` proof verification.
pub use crate::proofs::query::{prove_resident, verify, verify_query};

/// The ordered traced write vocabulary applied through
/// [`TraceInterface::apply`](crate::tracer::TraceInterface::apply). `MovePrefix` is
/// MRT-only — the AVL handle rejects it with [`Error::Unsupported`](crate::Error).
pub use crate::tracer::WriteOp;

/// Prove-side traced handle that records reads/writes against a snapshot and
/// emits the witness. Peer of [`crate::mrt::TraceRecorder`].
pub use tracer::finalization::TraceRecorder;
/// Verify-side traced read handle bound to an expected start root. Peer of
/// [`crate::mrt::TraceReplayer`].
pub use tracer::verification::TraceReplayer;
/// Externalized verifier: decode the trace, authenticate, replay caller-supplied
/// batches, re-authenticate. Peer of [`crate::mrt::TraceVerifier`].
pub use tracer::verification::TraceVerifier;

/// AVL trace primitives (skeleton build, write replay, read recording).
pub use tracer::assembly::{accessed_keys_from_nodes, assemble_sparse_trace};
pub use tracer::read_tracing::{trace_and_prove_reads, ReadTracer};
pub use tracer::recording::{get_recording, prefix_recording, range_recording, RecordingSource};
/// The sparse AVL proof/trace node (was `SparseMerkNode` / `AvlTrace`).
pub use tracer::sparse::Trace;
pub use tracer::write_tracing::{
    node_skeleton_from_trace, replay_sparse_writes, replay_sparse_writes_with_read_targets,
    replay_writes_on_node_tree, trace_and_apply_writes,
};
