//! AVL trace implementation.
//!
//! Shared traced-handle traits and transcript vocabulary live in
//! [`crate::tracer`]. This module owns only the AVL-specific sparse trace,
//! recorder, and verifier machinery re-exported through [`crate::avl`].

pub(crate) mod assembly;
pub(crate) mod finalization;
pub(crate) mod read_tracing;
pub(crate) mod recording;
pub(crate) mod sparse;
pub(crate) mod verification;
pub(crate) mod write_tracing;

pub(crate) use crate::tracer::extract_reads_sparse;
pub(crate) use crate::tracer::{
    prefix_successor, BatchOp, ProvenRead, ReadOp, ReadResults, TraceInterface, TraceReader,
    WriteOp,
};
pub(crate) use recording::RecordingSource;
pub(crate) use sparse::Trace as SparseMerkNode;

#[cfg(test)]
pub(crate) use sparse::Trace as AvlTrace;
#[cfg(test)]
pub(crate) use verification::VerifiedReadResults;
