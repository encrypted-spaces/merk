use std::collections::HashSet;

use crate::avl::in_memory::Checkpoint;
use crate::avl::node::Node;
use crate::error::{Error, Result, UnsupportedFeature};
#[cfg(test)]
use crate::hash::{Hash, NULL_HASH};

use super::assembly::assemble_sparse_trace;
use super::read_tracing::ReadTracer;
use super::sparse::Trace;
use super::write_tracing::{
    apply_delete_prefix_recording_accessed, apply_writes_recording_accessed,
    original_tree_contains, original_tree_path_keys,
};

/// Walk the original tree toward `key`, adding every ancestor to the
/// accessed-key set. This is the equivalent of the reference branch's
/// `record_original_path`: it ensures the assembly walk can reach `key`
/// through the original tree structure, regardless of how the current tree
/// was rebalanced by earlier writes.
///
/// If `key` does not exist in the original tree the walk stops at a dead end
/// and records the ancestors it did visit — harmless extra nodes that make
/// the proof slightly larger.
fn record_original_path(accessed: &mut HashSet<Vec<u8>>, root: Option<&Node>, key: &[u8]) {
    for ancestor in original_tree_path_keys(root, key) {
        accessed.insert(ancestor);
    }
}
use super::{BatchOp, ProvenRead, ReadOp, ReadResults};

/// Accumulates the accessed-node set while tracing a transcript over an AVL
/// tree, so the sparse **trace** (the witness) can be assembled. This is the
/// whole prove-side core, driven by [`TraceRecorder`].
struct TraceBuilder {
    original_tree: Option<Node>,
    current_tree: Option<Node>,
    all_accessed_keys: HashSet<Vec<u8>>,
    all_read_target_keys: HashSet<Vec<u8>>,
}

impl TraceBuilder {
    fn new(tree: Option<Node>) -> Self {
        Self {
            original_tree: tree.clone(),
            current_tree: tree,
            all_accessed_keys: HashSet::new(),
            all_read_target_keys: HashSet::new(),
        }
    }

    /// Trace a read step against the current tree, recording accessed nodes and
    /// returning the proven results.
    fn add_read_step(&mut self, reads: &[ReadOp]) -> Result<ReadResults> {
        let root = match &self.current_tree {
            Some(root) => root,
            None => {
                return Ok(reads
                    .iter()
                    .map(|op| ProvenRead {
                        op: op.clone(),
                        results: Vec::new(),
                    })
                    .collect());
            }
        };

        let mut tracer = ReadTracer::new();
        let mut proven_reads = Vec::with_capacity(reads.len());
        for op in reads {
            let results = tracer.execute_read_op(root, op)?;
            proven_reads.push(ProvenRead {
                op: op.clone(),
                results,
            });
        }

        let (accessed, targets) = tracer.into_sets();
        for key in &accessed {
            record_original_path(
                &mut self.all_accessed_keys,
                self.original_tree.as_ref(),
                key,
            );
        }
        for key in targets {
            if original_tree_contains(self.original_tree.as_ref(), &key) {
                self.all_read_target_keys.insert(key);
            }
        }

        Ok(proven_reads)
    }

    /// Trace a write step: apply it to the current tree and record accessed
    /// nodes for the final sparse-tree assembly.
    fn add_write_step(&mut self, ops: &[BatchOp]) -> Result<()> {
        if ops.is_empty() {
            if let Some(root) = self.original_tree.as_ref() {
                self.all_accessed_keys.insert(root.key().to_vec());
            }
            return Ok(());
        }

        let (accessed_keys, post_tree) =
            apply_writes_recording_accessed(self.current_tree.clone(), ops)?;

        for key in &accessed_keys {
            record_original_path(
                &mut self.all_accessed_keys,
                self.original_tree.as_ref(),
                key,
            );
        }

        self.current_tree = post_tree;
        Ok(())
    }

    /// Trace a sequenced prefix delete. This keeps `delete_prefix` as a method
    /// operation while reusing the same touched-node recording as range deletes.
    fn add_delete_prefix_step(&mut self, prefix: &[u8]) -> Result<()> {
        let (accessed_keys, post_tree) =
            apply_delete_prefix_recording_accessed(self.current_tree.clone(), prefix)?;

        for key in &accessed_keys {
            record_original_path(
                &mut self.all_accessed_keys,
                self.original_tree.as_ref(),
                key,
            );
        }

        self.current_tree = post_tree;
        Ok(())
    }

    /// Assemble the sparse trace from the accumulated accessed-node set.
    fn finalize_trace(self) -> Result<Trace> {
        assemble_sparse_trace(
            self.original_tree.as_ref(),
            &self.all_accessed_keys,
            &self.all_read_target_keys,
        )
    }
}

/// Prove-side traced read handle for the AVL backend — the externalized peer of
/// [`crate::mrt::TraceRecorder`].
///
/// Built over a live [`Checkpoint`], it answers sequenced reads
/// ([`TraceReader`](crate::tracer::TraceReader)) and sequenced writes
/// ([`TraceInterface`](crate::tracer::TraceInterface)) against a working copy of
/// the snapshot while recording every node they touch, then emits the witness with
/// [`finalize_trace`](Self::finalize_trace) in the same `Trace` wire format the
/// legacy `create_trace` produces.
pub struct TraceRecorder {
    builder: TraceBuilder,
    reads: Vec<ReadOp>,
    /// Set once an [`apply`](crate::tracer::TraceInterface::apply) op fails. A
    /// poisoned recorder rejects every subsequent fallible op (reads, further
    /// `apply`, `finalize_trace`) with [`Error::Poisoned`]; `reads()` stays
    /// infallible (see its doc).
    poisoned: bool,
}

impl TraceRecorder {
    /// Start recording traced operations against `snapshot`.
    pub fn new(snapshot: &Checkpoint) -> Self {
        Self {
            builder: TraceBuilder::new(snapshot.root().cloned()),
            reads: Vec::new(),
            poisoned: false,
        }
    }

    /// The read ops issued through this recorder, in order — the read-set a
    /// consumer claims alongside the trace.
    ///
    /// **Poison exemption (precondition).** This stays infallible even after a
    /// failed `apply` poisons the handle, returning the pre-poison read log. That
    /// is sound only if the caller issued *all* reads before the failing write —
    /// merk cannot enforce caller ordering, so honoring this ordering is the
    /// caller's precondition.
    pub fn reads(&self) -> &[ReadOp] {
        &self.reads
    }

    /// Reject any fallible op once the handle is poisoned.
    fn reject_if_poisoned(&self, op: &str) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned(format!(
                "{op} on a poisoned AVL recorder (an earlier apply failed)"
            )));
        }
        Ok(())
    }

    /// Finish recording and emit the witness as encoded `Trace` bytes — the bytes
    /// [`crate::avl::TraceReplayer::new_verified`] consumes. Semantically
    /// equivalent to the legacy `create_trace` output for the same reads, though
    /// not contractually byte-for-byte identical. Fails [`Error::Poisoned`] if a
    /// prior `apply` op failed.
    pub fn finalize_trace(self) -> Result<Vec<u8>> {
        self.reject_if_poisoned("finalize_trace")?;
        // Assembling an honest trace and encoding it into an in-memory
        // buffer are both infallible; surface a clear panic if that ever changes.
        let trace = self
            .builder
            .finalize_trace()
            .expect("assembling an honest AVL trace is infallible after successful operations");
        Ok(trace
            .encode()
            .expect("encoding a trace to an in-memory buffer is infallible"))
    }
}

impl super::TraceReader for TraceRecorder {
    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.reject_if_poisoned("get")?;
        let op = ReadOp::Key(key.to_vec());
        let proven = self.builder.add_read_step(std::slice::from_ref(&op))?;
        self.reads.push(op);
        Ok(proven
            .into_iter()
            .next()
            .and_then(|read| read.results.into_iter().next())
            .map(|(_, value)| value))
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_range")?;
        let op = ReadOp::Range {
            start: start.to_vec(),
            end: end.to_vec(),
        };
        let proven = self.builder.add_read_step(std::slice::from_ref(&op))?;
        self.reads.push(op);
        Ok(proven
            .into_iter()
            .next()
            .map(|read| read.results)
            .unwrap_or_default())
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_prefix")?;
        let op = ReadOp::Prefix(prefix.to_vec());
        let proven = self.builder.add_read_step(std::slice::from_ref(&op))?;
        self.reads.push(op);
        Ok(proven
            .into_iter()
            .next()
            .map(|read| read.results)
            .unwrap_or_default())
    }
}

/// Records each `WriteOp` **in vector order, one at a time**, lowering it to the
/// same per-op primitive the imperative methods used: point/range ops to a
/// single-element [`BatchOp`] write step, `DeletePrefix` to the prefix-delete
/// step, and `MovePrefix` to AVL's unsupported error. A `Vec<WriteOp>` is never
/// gathered into a sorted multi-op batch. An empty batch loops zero times — it is
/// a true no-op and must **not** route through `add_write_step(&[])` (which would
/// record the root). The first failing op poisons the recorder.
impl super::TraceInterface for TraceRecorder {
    fn apply(&mut self, ops: &[super::WriteOp]) -> Result<()> {
        use super::WriteOp;
        self.reject_if_poisoned("apply")?;
        for op in ops {
            let result = match op {
                WriteOp::Put { key, value } => self.builder.add_write_step(&[BatchOp::Put {
                    key: key.clone(),
                    value: value.clone(),
                }]),
                WriteOp::Delete { key } => self
                    .builder
                    .add_write_step(&[BatchOp::Delete { key: key.clone() }]),
                WriteOp::DeleteRange { start, end } => {
                    self.builder.add_write_step(&[BatchOp::DeleteRange {
                        start: start.clone(),
                        end: end.clone(),
                    }])
                }
                WriteOp::DeletePrefix { prefix } => self.builder.add_delete_prefix_step(prefix),
                WriteOp::MovePrefix { .. } => {
                    Err(Error::Unsupported(UnsupportedFeature::MovePrefix))
                }
            };
            if let Err(err) = result {
                self.poisoned = true;
                return Err(err);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avl::in_memory::InMemoryMerk;
    use crate::error::Error;
    use crate::tracer::test_support::avl::{create_trace, replay_trace, root_after_writes};
    use crate::tracer::test_support::Step;
    use crate::tracer::{prefix_successor, TraceInterface, TraceReader, WriteOp};

    fn build_tree(entries: &[(&[u8], &[u8])]) -> Checkpoint {
        let merk = InMemoryMerk::new();
        for &(k, v) in entries {
            merk.put(k, v).unwrap();
        }
        merk.checkpoint()
    }

    /// Build the witness for `steps` over `snapshot` and return `(trace, start,
    /// end)`: the start root authenticated by the witness, and the end root from
    /// an independent live-store oracle (`root_after_writes`).
    fn build(snapshot: Checkpoint, steps: &[Step]) -> (Trace, Hash, Hash) {
        let trace = create_trace(&snapshot, steps).unwrap();
        let start = trace.hash();
        let end = root_after_writes(&snapshot, steps);
        (trace, start, end)
    }

    #[test]
    fn create_trace_empty_tree_is_empty() {
        let (trace, start, end) = build(build_tree(&[]), &[]);
        assert_eq!(trace, Trace::Empty);
        assert_eq!(start, NULL_HASH);
        assert_eq!(end, NULL_HASH);
        assert!(replay_trace(&trace, start, &[], end).unwrap().is_empty());
    }

    #[test]
    fn start_root_authenticates_original_tree() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let expected = root.root_hash();
        let steps = vec![Step::Read(vec![ReadOp::Key(b"a".to_vec())])];
        let (trace, start, end) = build(root, &steps);
        assert_eq!(start, expected);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert_eq!(reads[0][0].results, vec![(b"a".to_vec(), b"1".to_vec())]);
    }

    #[test]
    fn end_root_reflects_writes() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let start_hash = root.root_hash();
        let steps = vec![Step::Write(vec![BatchOp::Put {
            key: b"c".to_vec(),
            value: b"updated".to_vec(),
        }])];
        let (trace, start, end) = build(root, &steps);
        assert_eq!(start, start_hash);
        assert_ne!(end, start_hash);
        replay_trace(&trace, start, &steps, end).unwrap();
    }

    #[test]
    fn empty_write_step_is_a_noop() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let start_hash = root.root_hash();
        let steps = vec![Step::Write(Vec::new())];
        let (trace, start, end) = build(root, &steps);
        assert_eq!(start, start_hash);
        assert_eq!(end, start_hash);
        replay_trace(&trace, start, &steps, end).unwrap();
    }

    #[test]
    fn write_then_read_sees_post_write_value() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"updated".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Key(b"c".to_vec())]),
        ];
        let (trace, start, end) = build(root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert_eq!(
            reads[0][0].results,
            vec![(b"c".to_vec(), b"updated".to_vec())]
        );
    }

    #[test]
    fn read_on_empty_tree_returns_empty() {
        let steps = vec![Step::Read(vec![ReadOp::Key(b"x".to_vec())])];
        let (trace, start, end) = build(build_tree(&[]), &steps);
        assert_eq!(trace, Trace::Empty);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert!(reads[0][0].results.is_empty());
    }

    #[test]
    fn empty_start_then_write_read() {
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Key(b"k".to_vec())]),
        ];
        let (trace, start, end) = build(build_tree(&[]), &steps);
        assert_eq!(start, NULL_HASH);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert_eq!(reads[0][0].results, vec![(b"k".to_vec(), b"v".to_vec())]);
    }

    #[test]
    fn multiple_interleaved_steps_verify() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let steps = vec![
            Step::Read(vec![ReadOp::Key(b"a".to_vec())]),
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"new_c".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Key(b"c".to_vec())]),
            Step::Write(vec![BatchOp::Delete { key: b"e".to_vec() }]),
        ];
        let (trace, start, end) = build(root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert_eq!(reads[0][0].results, vec![(b"a".to_vec(), b"1".to_vec())]);
        assert_eq!(
            reads[1][0].results,
            vec![(b"c".to_vec(), b"new_c".to_vec())]
        );
    }

    #[test]
    fn delete_range_then_reads() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let steps = vec![
            Step::Write(vec![BatchOp::DeleteRange {
                start: b"c".to_vec(),
                end: b"f".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Key(b"c".to_vec())]),
            Step::Read(vec![ReadOp::Key(b"a".to_vec())]),
        ];
        let (trace, start, end) = build(root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert!(reads[0][0].results.is_empty(), "c should be deleted");
        assert_eq!(reads[1][0].results, vec![(b"a".to_vec(), b"1".to_vec())]);
    }

    #[test]
    fn write_insert_then_point_read_new_key_succeeds() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"b".to_vec(),
                value: b"inserted".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Key(b"b".to_vec())]),
        ];
        let (trace, start, end) = build(root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        assert_eq!(
            reads[0][0].results,
            vec![(b"b".to_vec(), b"inserted".to_vec())]
        );
    }

    #[test]
    fn write_insert_then_range_read_visiting_new_key_succeeds() {
        let root = build_tree(&[(b"a", b"1"), (b"e", b"5")]);
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"c".to_vec(),
                value: b"inserted".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Range {
                start: b"a".to_vec(),
                end: b"f".to_vec(),
            }]),
        ];
        let (trace, start, end) = build(root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        let keys: Vec<&[u8]> = reads[0][0]
            .results
            .iter()
            .map(|(k, _)| k.as_slice())
            .collect();
        assert!(keys.contains(&b"c".as_slice()));
    }

    #[test]
    fn write_insert_then_prefix_read_visiting_new_key_succeeds() {
        let root = build_tree(&[(b"pre_a", b"1"), (b"xyz", b"2")]);
        let steps = vec![
            Step::Write(vec![BatchOp::Put {
                key: b"pre_b".to_vec(),
                value: b"inserted".to_vec(),
            }]),
            Step::Read(vec![ReadOp::Prefix(b"pre_".to_vec())]),
        ];
        let (trace, start, end) = build(root, &steps);
        let reads = replay_trace(&trace, start, &steps, end).unwrap();
        let keys: Vec<&[u8]> = reads[0][0]
            .results
            .iter()
            .map(|(k, _)| k.as_slice())
            .collect();
        assert!(keys.contains(&b"pre_b".as_slice()));
    }

    #[test]
    fn create_trace_accepts_unsorted_point_writes_op_by_op() {
        // Point writes are traced one at a time in issue order, so there is no
        // sorted-batch requirement: an unsorted write step is applied as given.
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let steps = vec![Step::Write(vec![
            BatchOp::Put {
                key: b"c".to_vec(),
                value: b"x".to_vec(),
            },
            BatchOp::Put {
                key: b"a".to_vec(),
                value: b"y".to_vec(),
            },
        ])];
        assert!(create_trace(&root, &steps).is_ok());
    }

    #[test]
    fn move_prefix_is_unsupported_for_avl() {
        let root = build_tree(&[(b"a", b"1")]);
        let steps = vec![Step::MovePrefix {
            from: b"a".to_vec(),
            to: b"b".to_vec(),
        }];
        assert!(matches!(
            create_trace(&root, &steps),
            Err(Error::Unsupported(UnsupportedFeature::MovePrefix))
        ));
    }

    // ── Stage 6: TraceRecorder read-only ────────────────────────────────────

    #[test]
    fn recorder_reads_match_snapshot_and_record_ops() {
        let snapshot = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let mut recorder = TraceRecorder::new(&snapshot);

        assert_eq!(recorder.get(b"c").unwrap(), Some(b"3".to_vec()));
        // Witnessed non-membership reads back as `Ok(None)`.
        assert_eq!(recorder.get(b"b").unwrap(), None);
        assert_eq!(
            recorder.get_range(b"a", b"f").unwrap(),
            snapshot.collect_range(b"a", Some(b"f")).unwrap()
        );
        assert_eq!(
            recorder.get_prefix(b"e").unwrap(),
            snapshot.collect_prefix(b"e").unwrap()
        );

        // `reads()` records exactly the ops issued, in order.
        assert_eq!(
            recorder.reads(),
            &[
                ReadOp::Key(b"c".to_vec()),
                ReadOp::Key(b"b".to_vec()),
                ReadOp::Range {
                    start: b"a".to_vec(),
                    end: b"f".to_vec(),
                },
                ReadOp::Prefix(b"e".to_vec()),
            ]
        );
    }

    #[test]
    fn recorder_trace_round_trips_through_replayer() {
        let snapshot = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let start = snapshot.root_hash();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.get(b"c").unwrap();
        recorder.get_range(b"a", b"z").unwrap();
        let bytes = recorder.finalize_trace().unwrap();

        // The recorder bytes decode, bind to the start root, and replay the same
        // reads — an honest recorder trace never causes a read-side `PrunedNode`.
        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, start).unwrap();
        assert_eq!(replayer.get(b"c").unwrap(), Some(b"3".to_vec()));
        assert_eq!(
            replayer.get_range(b"a", b"z").unwrap(),
            vec![
                (b"a".to_vec(), b"1".to_vec()),
                (b"c".to_vec(), b"3".to_vec()),
                (b"e".to_vec(), b"5".to_vec()),
            ]
        );
        // Read-only: the post-state root equals the bound start root.
        assert_eq!(replayer.root_hash().unwrap(), start);
    }

    #[test]
    fn empty_recorder_binds_to_start_root() {
        let snapshot = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let start = snapshot.root_hash();

        let recorder = TraceRecorder::new(&snapshot);
        assert!(recorder.reads().is_empty());
        let bytes = recorder.finalize_trace().unwrap();
        // No reads: the fully-pruned trace still authenticates the start root.
        assert!(crate::avl::TraceReplayer::new_verified(&bytes, start).is_ok());
    }

    #[test]
    fn recorder_is_object_safe() {
        let snapshot = build_tree(&[(b"a", b"1")]);
        let mut recorder = TraceRecorder::new(&snapshot);
        let reader: &mut dyn TraceReader = &mut recorder;
        assert_eq!(reader.get(b"a").unwrap(), Some(b"1".to_vec()));
    }

    // ── Stage 7: TraceRecorder sequenced writes ────────────────────────────

    #[test]
    fn recorder_sequenced_writes_round_trip_through_replayer() {
        let snapshot = build_tree(&[
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
            (b"pre_a", b"pa"),
            (b"pre_b", b"pb"),
        ]);
        let start = snapshot.root_hash();

        let expected = InMemoryMerk::new();
        expected.restore(Some(snapshot.clone()));
        expected.put(b"b".to_vec(), b"22".to_vec()).unwrap();
        expected.delete(b"a".to_vec()).unwrap();
        expected.delete_range(b"c".to_vec(), b"e".to_vec()).unwrap();
        expected
            .delete_range(b"pre_".to_vec(), prefix_successor(b"pre_").unwrap())
            .unwrap();
        expected.put(b"post".to_vec(), b"p".to_vec()).unwrap();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.put(b"b", b"22").unwrap();
        assert_eq!(recorder.get(b"b").unwrap(), Some(b"22".to_vec()));
        recorder.delete(b"a").unwrap();
        assert_eq!(recorder.get(b"a").unwrap(), None);
        recorder.delete_range(b"c", b"e").unwrap();
        assert!(recorder.get_range(b"c", b"e").unwrap().is_empty());
        recorder.delete_prefix(b"pre_").unwrap();
        assert!(recorder.get_prefix(b"pre_").unwrap().is_empty());
        recorder.put(b"post", b"p").unwrap();
        assert_eq!(recorder.get(b"post").unwrap(), Some(b"p".to_vec()));

        let bytes = recorder.finalize_trace().unwrap();
        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.put(b"b", b"22").unwrap();
        assert_eq!(replayer.get(b"b").unwrap(), Some(b"22".to_vec()));
        replayer.delete(b"a").unwrap();
        assert_eq!(replayer.get(b"a").unwrap(), None);
        replayer.delete_range(b"c", b"e").unwrap();
        assert!(replayer.get_range(b"c", b"e").unwrap().is_empty());
        replayer.delete_prefix(b"pre_").unwrap();
        assert!(replayer.get_prefix(b"pre_").unwrap().is_empty());
        replayer.put(b"post", b"p").unwrap();
        assert_eq!(replayer.get(b"post").unwrap(), Some(b"p".to_vec()));

        assert_eq!(replayer.root_hash().unwrap(), expected.root_hash());
    }

    #[test]
    fn recorder_delete_prefix_handles_no_successor_prefix() {
        let snapshot = build_tree(&[
            (&[0xfe][..], b"before".as_slice()),
            (&[0xff, 0x00][..], b"first".as_slice()),
            (&[0xff, 0x10][..], b"second".as_slice()),
        ]);
        let start = snapshot.root_hash();
        let expected = build_tree(&[(&[0xfe][..], b"before".as_slice())]).root_hash();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.delete_prefix(&[0xff]).unwrap();
        assert!(recorder.get_prefix(&[0xff]).unwrap().is_empty());
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.delete_prefix(&[0xff]).unwrap();
        assert!(replayer.get_prefix(&[0xff]).unwrap().is_empty());
        assert_eq!(replayer.root_hash().unwrap(), expected);
    }

    #[test]
    fn recorder_write_errors_use_expected_buckets() {
        let snapshot = build_tree(&[(b"a", b"1")]);
        // A failing `apply` op poisons the handle, so each error-bucket check
        // needs a fresh recorder — otherwise the second op would return
        // `Error::Poisoned` instead of its own bucket.
        let mut recorder = TraceRecorder::new(&snapshot);
        assert!(matches!(
            recorder.delete_range(b"z", b"a"),
            Err(Error::Bound(_))
        ));
        let mut recorder = TraceRecorder::new(&snapshot);
        assert!(matches!(
            recorder.move_prefix(b"from", b"to"),
            Err(Error::Unsupported(UnsupportedFeature::MovePrefix))
        ));

        let bytes = TraceRecorder::new(&build_tree(&[]))
            .finalize_trace()
            .unwrap();
        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, NULL_HASH).unwrap();
        assert!(matches!(
            replayer.delete_range(b"z", b"a"),
            Err(Error::Bound(_))
        ));
    }

    fn recorder_generic_write<T: TraceInterface>(handle: &mut T) -> Option<Vec<u8>> {
        handle.put(b"k", b"v1").unwrap();
        handle.delete(b"k").unwrap();
        handle.put(b"k", b"v2").unwrap();
        handle.get(b"k").unwrap()
    }

    #[test]
    fn recorder_trace_interface_object_and_generic_paths_run() {
        let snapshot = build_tree(&[]);
        let mut object_recorder = TraceRecorder::new(&snapshot);
        let handle: &mut dyn TraceInterface = &mut object_recorder;
        handle.put(b"a", b"1").unwrap();
        assert_eq!(handle.get(b"a").unwrap(), Some(b"1".to_vec()));
        handle.delete_prefix(b"a").unwrap();
        assert_eq!(handle.get(b"a").unwrap(), None);

        let mut generic_recorder = TraceRecorder::new(&snapshot);
        assert_eq!(
            recorder_generic_write(&mut generic_recorder),
            Some(b"v2".to_vec())
        );
    }

    // ── Stage 3: WriteOp `apply` surface ────────────────────────────────────

    /// Count every materialized/pruned node in a sparse AVL witness — a
    /// path-bounded vs whole-tree proxy for the recorder's per-op work.
    fn count_trace_nodes(trace: &Trace) -> usize {
        match trace {
            Trace::Empty => 0,
            Trace::Pruned { .. } => 1,
            Trace::Full { left, right, .. }
            | Trace::FullStorageHash { left, right, .. }
            | Trace::FullOmitted { left, right, .. } => {
                1 + count_trace_nodes(left) + count_trace_nodes(right)
            }
        }
    }

    /// Drive a recorder over `snapshot` with the given `apply` calls (each inner
    /// slice is one `apply()` invocation) and then point-read `read_keys`;
    /// finalize and replay the exact same sequence through a verified replayer.
    /// Asserts the recorder and replayer agree on every read, then returns the
    /// replayer's end root and the (agreed) point-read results.
    fn drive_apply_and_reads(
        snapshot: &Checkpoint,
        apply_calls: &[&[WriteOp]],
        read_keys: &[&[u8]],
    ) -> (Hash, Vec<Option<Vec<u8>>>) {
        let start = snapshot.root_hash();

        let mut recorder = TraceRecorder::new(snapshot);
        for call in apply_calls {
            recorder.apply(call).unwrap();
        }
        let recorder_reads: Vec<Option<Vec<u8>>> =
            read_keys.iter().map(|k| recorder.get(k).unwrap()).collect();
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, start).unwrap();
        for call in apply_calls {
            replayer.apply(call).unwrap();
        }
        let replayer_reads: Vec<Option<Vec<u8>>> =
            read_keys.iter().map(|k| replayer.get(k).unwrap()).collect();
        let root = replayer.root_hash().unwrap();

        assert_eq!(
            recorder_reads, replayer_reads,
            "recorder and replayer disagree on reads"
        );
        (root, recorder_reads)
    }

    /// Apply a single batch and read `read_keys` back through the prove→verify
    /// cycle; returns the agreed observations.
    fn apply_batch_then_read(
        snapshot: &Checkpoint,
        ops: &[WriteOp],
        read_keys: &[&[u8]],
    ) -> Vec<Option<Vec<u8>>> {
        drive_apply_and_reads(snapshot, &[ops], read_keys).1
    }

    fn put(key: &[u8], value: &[u8]) -> WriteOp {
        WriteOp::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        }
    }

    // C1 conflict table (duplicates allowed, last-writer-wins by position).

    #[test]
    fn apply_conflict_put_put_keeps_last() {
        let snap = build_tree(&[(b"a", b"1"), (b"z", b"26")]);
        let got = apply_batch_then_read(&snap, &[put(b"k", b"a"), put(b"k", b"b")], &[b"k"]);
        assert_eq!(got, vec![Some(b"b".to_vec())]);
    }

    #[test]
    fn apply_conflict_put_delete_is_absent() {
        let snap = build_tree(&[(b"a", b"1"), (b"z", b"26")]);
        let got = apply_batch_then_read(
            &snap,
            &[put(b"k", b"a"), WriteOp::Delete { key: b"k".to_vec() }],
            &[b"k"],
        );
        assert_eq!(got, vec![None]);
    }

    #[test]
    fn apply_conflict_delete_range_then_put_in_range_keeps_put() {
        let snap = build_tree(&[(b"a", b"1"), (b"m", b"13"), (b"z", b"26")]);
        let ops = vec![
            WriteOp::DeleteRange {
                start: b"a".to_vec(),
                end: b"n".to_vec(),
            },
            put(b"m", b"a"),
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"m"]);
        assert_eq!(got, vec![Some(b"a".to_vec())]);
    }

    #[test]
    fn apply_conflict_put_then_delete_range_over_it_is_absent() {
        let snap = build_tree(&[(b"a", b"1"), (b"z", b"26")]);
        let ops = vec![
            put(b"m", b"a"),
            WriteOp::DeleteRange {
                start: b"a".to_vec(),
                end: b"n".to_vec(),
            },
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"m"]);
        assert_eq!(got, vec![None]);
    }

    #[test]
    fn apply_conflict_put_then_delete_prefix_is_absent() {
        let snap = build_tree(&[(b"a", b"1"), (b"z", b"26")]);
        let ops = vec![
            put(b"pre_k", b"a"),
            WriteOp::DeletePrefix {
                prefix: b"pre_".to_vec(),
            },
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"pre_k"]);
        assert_eq!(got, vec![None]);
    }

    #[test]
    fn apply_conflict_delete_prefix_then_put_keeps_put() {
        let snap = build_tree(&[(b"pre_a", b"1"), (b"pre_b", b"2"), (b"z", b"26")]);
        let ops = vec![
            WriteOp::DeletePrefix {
                prefix: b"pre_".to_vec(),
            },
            put(b"pre_k", b"a"),
        ];
        let got = apply_batch_then_read(&snap, &ops, &[b"pre_k", b"pre_a"]);
        assert_eq!(got, vec![Some(b"a".to_vec()), None]);
    }

    #[test]
    fn apply_delete_prefix_0xff_clears_the_whole_span() {
        let snap = build_tree(&[
            (&[0xfe][..], b"keep".as_slice()),
            (&[0xff, 0x00][..], b"a".as_slice()),
            (&[0xff, 0x80][..], b"b".as_slice()),
            (&[0xff, 0xff][..], b"c".as_slice()),
        ]);
        let start = snap.root_hash();
        let ops = [WriteOp::DeletePrefix { prefix: vec![0xff] }];

        let mut recorder = TraceRecorder::new(&snap);
        recorder.apply(&ops).unwrap();
        assert!(recorder.get_prefix(&[0xff]).unwrap().is_empty());
        assert_eq!(recorder.get(&[0xfe]).unwrap(), Some(b"keep".to_vec()));
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.apply(&ops).unwrap();
        assert!(replayer.get_prefix(&[0xff]).unwrap().is_empty());
        assert_eq!(replayer.get(&[0xfe]).unwrap(), Some(b"keep".to_vec()));

        // Independent oracle: deleting the open-ended `[0xff, ∞)` span one op at a
        // time leaves only `0xfe`.
        let expected = build_tree(&[(&[0xfe][..], b"keep".as_slice())]).root_hash();
        assert_eq!(replayer.root_hash().unwrap(), expected);
    }

    #[test]
    fn apply_empty_batch_is_a_noop() {
        let snap = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let start = snap.root_hash();

        let mut recorder = TraceRecorder::new(&snap);
        recorder.apply(&[]).unwrap();
        let bytes = recorder.finalize_trace().unwrap();

        let mut replayer = crate::avl::TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.apply(&[]).unwrap();
        // No op ran, so the post-state root equals the bound start root.
        assert_eq!(replayer.root_hash().unwrap(), start);
    }

    #[test]
    fn apply_batch_equivalent_to_individual_calls() {
        let snap = build_tree(&[
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"pre_x", b"x"),
            (b"pre_y", b"y"),
        ]);
        let ops = vec![
            put(b"b", b"22"),
            WriteOp::Delete { key: b"a".to_vec() },
            WriteOp::DeleteRange {
                start: b"c".to_vec(),
                end: b"d".to_vec(),
            },
            WriteOp::DeletePrefix {
                prefix: b"pre_".to_vec(),
            },
            put(b"new", b"n"),
        ];
        let read_keys: &[&[u8]] = &[b"a", b"b", b"new", b"pre_x"];

        // One apply() call with the whole batch …
        let (batch_root, batch_reads) = drive_apply_and_reads(&snap, &[&ops], read_keys);
        // … vs the same ops, one apply() call each. Equivalent by construction.
        let individual: Vec<&[WriteOp]> = ops.iter().map(std::slice::from_ref).collect();
        let (indiv_root, indiv_reads) = drive_apply_and_reads(&snap, &individual, read_keys);

        // Equal root + reads + replay outcome (not raw bytes).
        assert_eq!(batch_root, indiv_root);
        assert_eq!(batch_reads, indiv_reads);

        // …and both match a live store applying the same ops one at a time
        // (DeletePrefix emulated by its half-open range, which has a successor).
        let oracle = InMemoryMerk::new();
        oracle.restore(Some(snap.clone()));
        oracle.put(b"b".to_vec(), b"22".to_vec()).unwrap();
        oracle.delete(b"a".to_vec()).unwrap();
        oracle.delete_range(b"c".to_vec(), b"d".to_vec()).unwrap();
        oracle
            .delete_range(b"pre_".to_vec(), prefix_successor(b"pre_").unwrap())
            .unwrap();
        oracle.put(b"new".to_vec(), b"n".to_vec()).unwrap();
        assert_eq!(batch_root, oracle.root_hash());
    }

    #[test]
    fn apply_move_prefix_poisons_avl_recorder() {
        let snap = build_tree(&[(b"a", b"1"), (b"b", b"2")]);
        let mut recorder = TraceRecorder::new(&snap);
        // A pre-poison read is preserved by the `reads()` exemption.
        recorder.get(b"a").unwrap();

        // The Put applies, then the AVL-unsupported MovePrefix fails: the whole
        // change is rejected and the handle poisoned.
        let err = recorder
            .apply(&[
                put(b"c", b"3"),
                WriteOp::MovePrefix {
                    from: b"a".to_vec(),
                    to: b"z".to_vec(),
                },
            ])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Unsupported(UnsupportedFeature::MovePrefix)
        ));

        // Every subsequent fallible op fails closed with `Poisoned`.
        assert!(matches!(recorder.get(b"a"), Err(Error::Poisoned(_))));
        assert!(matches!(
            recorder.apply(&[put(b"x", b"x")]),
            Err(Error::Poisoned(_))
        ));
        // `reads()` stays infallible and returns the pre-poison log.
        assert_eq!(recorder.reads(), &[ReadOp::Key(b"a".to_vec())]);
        assert!(matches!(recorder.finalize_trace(), Err(Error::Poisoned(_))));
    }

    #[test]
    fn apply_mid_batch_failure_aborts_rest_and_poisons() {
        let snap = build_tree(&[(b"a", b"1")]);
        let mut recorder = TraceRecorder::new(&snap);
        // The inverted DeleteRange (second op) fails; the third op must not run.
        let err = recorder
            .apply(&[
                put(b"b", b"2"),
                WriteOp::DeleteRange {
                    start: b"z".to_vec(),
                    end: b"a".to_vec(),
                },
                put(b"c", b"3"),
            ])
            .unwrap_err();
        assert!(matches!(err, Error::Bound(_)));
        assert!(matches!(recorder.get(b"b"), Err(Error::Poisoned(_))));
    }

    /// Perf gate: an N-op `apply` over an M-entry tree must stay path-bounded
    /// (~O(N·log M)), never whole-tree (~O(N·M)). Proxy: the witness node count.
    fn apply_witness_nodes(m: usize) -> usize {
        let merk = InMemoryMerk::new();
        for i in 0..m {
            merk.put(format!("{i:08}").into_bytes(), b"v".to_vec())
                .unwrap();
        }
        let snapshot = merk.checkpoint();
        // Eight existing keys present in trees of any size.
        let ops: Vec<WriteOp> = (0..8)
            .map(|i| put(format!("{i:08}").as_bytes(), b"v2"))
            .collect();

        let mut recorder = TraceRecorder::new(&snapshot);
        recorder.apply(&ops).unwrap();
        let bytes = recorder.finalize_trace().unwrap();
        count_trace_nodes(&Trace::decode_exact(&bytes).unwrap())
    }

    #[test]
    fn apply_witness_scales_with_path_not_tree_size() {
        let small = apply_witness_nodes(1000);
        let large = apply_witness_nodes(4000);
        // 4x the data must not ~4x the witness; a whole-tree walk would.
        assert!(
            large <= small * 2,
            "apply witness scaled with tree size, not accessed path: \
             {} nodes at n=1000 -> {} at n=4000 (O(n^2) regression?)",
            small,
            large
        );
        // …and the absolute count stays far below the tree size.
        assert!(
            large < 1000,
            "expected ~path-bounded witness, got {} for n=4000",
            large
        );
    }
}
