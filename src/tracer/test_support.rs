//! Test-only scaffolding for driving the op-by-op trace handle from a step list.
//!
//! The public transcript API (`create_trace` / `InputStep` / `TraceStep`) was
//! deleted; the canonical write path is the handle ([`crate::tracer::TraceReader`]
//! / [`crate::tracer::TraceInterface`], implemented by each backend's
//! `TraceRecorder` / `TraceReplayer`), which applies every operation one at a
//! time in issue order. These helpers let the test corpus keep expressing a
//! transcript as a `&[Step]` list while exercising that handle.

use crate::error::Result;
use crate::tracer::{BatchOp, ProvenRead, ReadOp, ReadResults, TraceInterface};

/// A test transcript step — the in-test stand-in for the deleted public
/// `InputStep`. `MovePrefix` is MRT-only (the AVL handle rejects it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Read(Vec<ReadOp>),
    Write(Vec<BatchOp>),
    MovePrefix { from: Vec<u8>, to: Vec<u8> },
}

/// Per-read-step results, the shape the verify loop yields back.
pub(crate) type StepReads = Vec<ReadResults>;

/// Drive a recorder or replayer over `steps`, applying each op one at a time and
/// returning the per-read-step results. Writes are applied in issue order — AVL
/// is insertion-order sensitive, so this is exactly how the canonical tree, the
/// recorder, and the replayer all evolve.
pub(crate) fn drive<H: TraceInterface>(handle: &mut H, steps: &[Step]) -> Result<StepReads> {
    let mut all_reads = Vec::new();
    for step in steps {
        match step {
            Step::Read(ops) => {
                let mut reads = Vec::with_capacity(ops.len());
                for op in ops {
                    let results = match op {
                        ReadOp::Key(key) => match handle.get(key)? {
                            Some(value) => vec![(key.clone(), value)],
                            None => Vec::new(),
                        },
                        ReadOp::Prefix(prefix) => handle.get_prefix(prefix)?,
                        ReadOp::Range { start, end } => handle.get_range(start, end)?,
                    };
                    reads.push(ProvenRead {
                        op: op.clone(),
                        results,
                    });
                }
                all_reads.push(reads);
            }
            Step::Write(ops) => {
                for op in ops {
                    match op {
                        BatchOp::Put { key, value } => handle.put(key, value)?,
                        BatchOp::Delete { key } => handle.delete(key)?,
                        BatchOp::DeleteRange { start, end } => handle.delete_range(start, end)?,
                    }
                }
            }
            Step::MovePrefix { from, to } => handle.move_prefix(from, to)?,
        }
    }
    Ok(all_reads)
}

/// AVL handle-driven transcript helpers (replace the deleted `create_trace` /
/// `replay_trace` / `root_after_writes`).
pub(crate) mod avl {
    use super::{drive, Step, StepReads};
    use crate::avl::{Checkpoint, Trace, TraceRecorder, TraceReplayer};
    use crate::error::Result;
    use crate::hash::Hash;

    /// Build the witness for `steps` over `snapshot` via `TraceRecorder`,
    /// returning the decoded [`Trace`] so soundness tests can inspect/tamper it.
    pub(crate) fn create_trace(snapshot: &Checkpoint, steps: &[Step]) -> Result<Trace> {
        let mut recorder = TraceRecorder::new(snapshot);
        drive(&mut recorder, steps)?;
        let bytes = recorder.finalize_trace()?;
        Trace::decode_exact(&bytes)
    }

    /// Replay `trace` through `TraceReplayer`, authenticating `start_root` before
    /// and `end_root` after; returns per-read-step results. Fails closed so
    /// soundness tests can pass a tampered trace and assert `Err`.
    pub(crate) fn replay_trace(
        trace: &Trace,
        start_root: Hash,
        steps: &[Step],
        end_root: Hash,
    ) -> Result<StepReads> {
        let bytes = trace.encode()?;
        let mut replayer = TraceReplayer::new_verified(&bytes, start_root)?;
        let reads = drive(&mut replayer, steps)?;
        let actual_end = replayer.root_hash()?;
        if actual_end != end_root {
            return Err(crate::Error::HashMismatch(end_root, actual_end));
        }
        Ok(reads)
    }

    /// Independent end-root oracle: apply the steps' writes to a fresh live AVL
    /// store, one op at a time, in issue order.
    pub(crate) fn root_after_writes(snapshot: &Checkpoint, steps: &[Step]) -> Hash {
        let merk = crate::avl::in_memory::InMemoryMerk::new();
        merk.restore(Some(snapshot.clone()));
        for step in steps {
            if let Step::Write(ops) = step {
                for op in ops {
                    merk.apply_sorted_batch_ops_owned(vec![op.to_batch_entry()])
                        .expect("apply write op to live AVL store");
                }
            }
        }
        merk.root_hash()
    }
}

/// MRT handle-driven transcript helpers.
pub(crate) mod mrt {
    use super::{drive, Step, StepReads};
    use crate::error::Result;
    use crate::hash::Hash;
    use crate::mrt::{Checkpoint, Trace, TraceRecorder, TraceReplayer, Tree};

    pub(crate) fn create_trace(snapshot: &Checkpoint, steps: &[Step]) -> Result<Trace> {
        let mut recorder = TraceRecorder::new(snapshot);
        drive(&mut recorder, steps)?;
        let bytes = recorder.finalize_trace()?;
        Trace::decode_exact(&bytes)
    }

    pub(crate) fn replay_trace(
        trace: &Trace,
        start_root: Hash,
        steps: &[Step],
        end_root: Hash,
    ) -> Result<StepReads> {
        let bytes = trace.encode()?;
        let mut replayer = TraceReplayer::new_verified(&bytes, start_root)?;
        let reads = drive(&mut replayer, steps)?;
        let actual_end = replayer.root_hash()?;
        if actual_end != end_root {
            return Err(crate::Error::HashMismatch(end_root, actual_end));
        }
        Ok(reads)
    }

    /// Independent end-root oracle: apply the steps' writes (incl. `MovePrefix`)
    /// to a fresh live MRT store, one op at a time, in issue order.
    pub(crate) fn root_after_writes(snapshot: &Checkpoint, steps: &[Step]) -> Hash {
        let tree = Tree::new();
        tree.restore(Some(snapshot.clone()));
        for step in steps {
            match step {
                Step::Write(ops) => {
                    for op in ops {
                        tree.apply_sorted_batch_ops_owned(vec![op.to_batch_entry()])
                            .expect("apply write op to live MRT store");
                    }
                }
                Step::MovePrefix { from, to } => {
                    tree.move_prefix(from.clone(), to.clone())
                        .expect("apply move_prefix to live MRT store");
                }
                Step::Read(_) => {}
            }
        }
        tree.root_hash()
    }
}
