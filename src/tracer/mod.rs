//! Shared traced-handle traits and op vocabulary.
//!
//! The backend-agnostic traced interfaces ([`TraceReader`],
//! [`TraceInterface`]), op types (`BatchOp`, `ReadOp`, `ProvenRead`,
//! `ReadResults`), and [`prefix_successor`] are defined here and used by both
//! backends. AVL-specific trace machinery lives under [`crate::avl::tracer`] and
//! is re-exported publicly through [`crate::avl`].

use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::hash::HASH_LENGTH;
use crate::ops::{BatchEntry, Op};

#[cfg(test)]
mod differential_size_tests;
#[cfg(test)]
mod fuzz_tests;
#[cfg(test)]
mod prototype_integration_tests;
#[cfg(test)]
pub(crate) mod test_support;

/// Legacy value-payload threshold from the old compacted proof format. Retained
/// only for tests that construct hash-only opened nodes.
#[cfg(test)]
pub(crate) const SMALL_VALUE_INLINE_THRESHOLD: usize = HASH_LENGTH - 4;

/// A point/range write op — the apply primitive the handle's sequenced
/// [`TraceInterface`] methods (`put` / `delete` / `delete_range`) lower into, and
/// the input to the low-level AVL trace-replay primitives. The verifiers' batch
/// entry points that take it are `pub(crate)`; the public write API is the
/// sequenced handle, never a batch.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum BatchOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    DeleteRange { start: Vec<u8>, end: Vec<u8> },
}

impl BatchOp {
    pub fn key(&self) -> &[u8] {
        match self {
            BatchOp::Put { key, .. } | BatchOp::Delete { key } => key,
            BatchOp::DeleteRange { start, .. } => start,
        }
    }

    pub fn to_op(&self) -> Op {
        match self {
            BatchOp::Put { value, .. } => Op::Put(value.clone()),
            BatchOp::Delete { .. } => Op::Delete,
            BatchOp::DeleteRange { end, .. } => Op::DeleteRange(end.clone()),
        }
    }

    pub fn to_batch_entry(&self) -> BatchEntry {
        (self.key().to_vec(), self.to_op())
    }
}

/// The higher-level **ordered** traced write vocabulary applied through
/// [`TraceInterface::apply`] — distinct from the legacy sorted-batch [`BatchOp`].
///
/// `apply` executes a `&[WriteOp]` **in vector order, one op at a time**, never
/// sorting or merging (sorting would diverge the AVL root). Each variant lowers to
/// an existing per-op primitive:
///
/// - `Put` / `Delete` / `DeleteRange` → the single-op point/range path (a
///   one-element [`BatchOp`]/[`Op`], exactly what `put`/`delete` lower to).
/// - `DeletePrefix` → the prefix-delete primitive directly (handles the
///   no-successor `[0xff]` span; never lowered to a `DeleteRange`).
/// - `MovePrefix` → the MRT subtree-relocate primitive; AVL has none, so it fails
///   [`Error::Unsupported`](crate::error::Error::Unsupported) and poisons the handle.
///
/// A `Vec<WriteOp>` is **never** converted into a sorted multi-op `Vec<BatchOp>`;
/// the only `BatchOp` use is the single-op carrier, one op at a time.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum WriteOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    DeleteRange { start: Vec<u8>, end: Vec<u8> },
    DeletePrefix { prefix: Vec<u8> },
    MovePrefix { from: Vec<u8>, to: Vec<u8> },
}

/// A read query recorded in the trace transcript.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum ReadOp {
    Key(Vec<u8>),
    Prefix(Vec<u8>),
    Range { start: Vec<u8>, end: Vec<u8> },
}

/// A read query plus verifier-populated key/value results.
///
/// Results are not serialized in the prototype postcard proof format; they are
/// derived by replay/verification from the authenticated sparse tree.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ProvenRead {
    pub op: ReadOp,
    #[serde(skip, default)]
    pub results: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Sequenced traced read operations shared by prove- and verify-side handles.
///
/// Reads take `&mut self` because traced reads are transcript operations:
/// implementations may need to record, authenticate, or flush pending writes
/// before answering them.
pub trait TraceReader {
    fn get(&mut self, key: &[u8]) -> crate::Result<Option<Vec<u8>>>;

    /// Half-open range read `[start, end)`.
    fn get_range(&mut self, start: &[u8], end: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Prefix read over every key starting with `prefix`.
    fn get_prefix(&mut self, prefix: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>>;
}

/// Sequenced traced read/write operations.
///
/// Range, prefix, and subtree mutations are explicit ordered operations rather
/// than sortable batch entries.
///
/// [`apply`](Self::apply) is the single required write method: it executes a
/// `&[WriteOp]` **in vector order, one op at a time** (no sort, no merge — sorting
/// would diverge the AVL root). The five imperative methods are sugar that lower
/// to a one-element [`apply`](Self::apply). An empty batch (`apply(&[])`) is a
/// no-op. Implementors (`TraceRecorder` / `TraceReplayer`) **poison** the handle
/// on the first failing op — every subsequent fallible operation then returns
/// [`Error::Poisoned`](crate::error::Error::Poisoned).
pub trait TraceInterface: TraceReader {
    /// Apply `ops` in vector order, one at a time. The only required write method.
    fn apply(&mut self, ops: &[WriteOp]) -> crate::Result<()>;

    fn put(&mut self, key: &[u8], value: &[u8]) -> crate::Result<()> {
        self.apply(&[WriteOp::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        }])
    }

    fn delete(&mut self, key: &[u8]) -> crate::Result<()> {
        self.apply(&[WriteOp::Delete { key: key.to_vec() }])
    }

    fn delete_range(&mut self, start: &[u8], end: &[u8]) -> crate::Result<()> {
        self.apply(&[WriteOp::DeleteRange {
            start: start.to_vec(),
            end: end.to_vec(),
        }])
    }

    fn delete_prefix(&mut self, prefix: &[u8]) -> crate::Result<()> {
        self.apply(&[WriteOp::DeletePrefix {
            prefix: prefix.to_vec(),
        }])
    }

    fn move_prefix(&mut self, from: &[u8], to: &[u8]) -> crate::Result<()> {
        self.apply(&[WriteOp::MovePrefix {
            from: from.to_vec(),
            to: to.to_vec(),
        }])
    }
}

/// Per-step read results.
pub type ReadResults = Vec<ProvenRead>;

/// Compute the exclusive end bound for a prefix scan.
pub fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(&last) = end.last() {
        if last < 0xff {
            *end.last_mut().expect("last checked above") += 1;
            return Some(end);
        }
        end.pop();
    }
    None
}

pub(crate) fn extract_reads_sparse(
    root: &crate::avl::tracer::SparseMerkNode,
    reads: &[ProvenRead],
) -> crate::Result<ReadResults> {
    let mut all_results = Vec::with_capacity(reads.len());
    for proven_read in reads {
        let results = match &proven_read.op {
            ReadOp::Key(key) => match root.get(key)? {
                Some(value) => vec![(key.clone(), value)],
                None => Vec::new(),
            },
            ReadOp::Prefix(prefix) => {
                let end = prefix_successor(prefix);
                root.collect_range(prefix, end.as_deref())?
            }
            ReadOp::Range { start, end } => root.collect_range(start, Some(end))?,
        };
        all_results.push(ProvenRead {
            op: proven_read.op.clone(),
            results,
        });
    }
    Ok(all_results)
}

impl TraceReader for crate::avl::Checkpoint {
    fn get(&mut self, key: &[u8]) -> crate::Result<Option<Vec<u8>>> {
        Ok(crate::avl::Checkpoint::get(self, key))
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::avl::Checkpoint::collect_range(self, start, Some(end))
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::avl::Checkpoint::collect_prefix(self, prefix)
    }
}

impl TraceReader for crate::avl::TraceVerifier {
    fn get(&mut self, key: &[u8]) -> crate::Result<Option<Vec<u8>>> {
        crate::avl::TraceVerifier::get(self, key)
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::avl::TraceVerifier::collect_range(self, start, Some(end))
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::avl::TraceVerifier::collect_prefix(self, prefix)
    }
}

impl TraceReader for crate::mrt::Checkpoint {
    fn get(&mut self, key: &[u8]) -> crate::Result<Option<Vec<u8>>> {
        Ok(crate::mrt::Checkpoint::get(self, key))
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::mrt::Checkpoint::collect_range(self, start, Some(end))
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::mrt::Checkpoint::collect_prefix(self, prefix)
    }
}

impl TraceReader for crate::mrt::TraceVerifier {
    fn get(&mut self, key: &[u8]) -> crate::Result<Option<Vec<u8>>> {
        crate::mrt::TraceVerifier::get(self, key)
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::mrt::TraceVerifier::collect_range(self, start, Some(end))
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        crate::mrt::TraceVerifier::collect_prefix(self, prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Hash;

    #[test]
    fn batch_op_delete_range_maps_to_storage_op() {
        let op = BatchOp::DeleteRange {
            start: b"a".to_vec(),
            end: b"z".to_vec(),
        };
        assert_eq!(op.key(), b"a");
        assert!(matches!(op.to_op(), Op::DeleteRange(end) if end == b"z"));
        assert_eq!(op.to_batch_entry().0, b"a".to_vec());
    }

    #[test]
    fn prefix_successor_handles_edges() {
        assert_eq!(prefix_successor(b"ab"), Some(b"ac".to_vec()));
        assert_eq!(prefix_successor(&[0x12, 0xff]), Some(vec![0x13]));
        assert_eq!(prefix_successor(&[0xff]), None);
    }

    fn build_avl_snapshot(entries: &[(&[u8], &[u8])]) -> crate::avl::Checkpoint {
        let tree = crate::avl::Tree::new();
        for (key, value) in entries {
            tree.put(*key, *value).unwrap();
        }
        tree.checkpoint()
    }

    fn build_mrt_snapshot(entries: &[(&[u8], &[u8])]) -> crate::mrt::Checkpoint {
        let tree = crate::mrt::Tree::new();
        for (key, value) in entries {
            tree.put(*key, *value).unwrap();
        }
        tree.checkpoint()
    }

    fn avl_trace_for_reads(
        snapshot: &crate::avl::Checkpoint,
        reads: Vec<ReadOp>,
    ) -> (crate::avl::Trace, Hash) {
        let trace =
            test_support::avl::create_trace(snapshot, &[test_support::Step::Read(reads)]).unwrap();
        let start_root = trace.hash();
        (trace, start_root)
    }

    fn mrt_trace_for_reads(
        snapshot: &crate::mrt::Checkpoint,
        reads: Vec<ReadOp>,
    ) -> (crate::mrt::Trace, Hash) {
        let trace =
            test_support::mrt::create_trace(snapshot, &[test_support::Step::Read(reads)]).unwrap();
        let start_root = trace.root_hash();
        (trace, start_root)
    }

    #[test]
    fn trace_reader_snapshot_reads_match_direct_reads_for_avl_and_mrt() {
        let mut avl =
            build_avl_snapshot(&[(b"a", b"1"), (b"aa", b"11"), (b"ab", b"12"), (b"b", b"2")]);
        assert_eq!(
            <crate::avl::Checkpoint as TraceReader>::get(&mut avl, b"aa").unwrap(),
            crate::avl::Checkpoint::get(&avl, b"aa")
        );
        assert_eq!(
            <crate::avl::Checkpoint as TraceReader>::get_range(&mut avl, b"a", b"b").unwrap(),
            crate::avl::Checkpoint::collect_range(&avl, b"a", Some(b"b")).unwrap()
        );
        assert_eq!(
            <crate::avl::Checkpoint as TraceReader>::get_prefix(&mut avl, b"a").unwrap(),
            crate::avl::Checkpoint::collect_prefix(&avl, b"a").unwrap()
        );

        let mut mrt =
            build_mrt_snapshot(&[(b"a", b"1"), (b"ba", b"21"), (b"bb", b"22"), (b"c", b"3")]);
        assert_eq!(
            <crate::mrt::Checkpoint as TraceReader>::get(&mut mrt, b"ba").unwrap(),
            crate::mrt::Checkpoint::get(&mrt, b"ba")
        );
        assert_eq!(
            <crate::mrt::Checkpoint as TraceReader>::get_range(&mut mrt, b"b", b"c").unwrap(),
            crate::mrt::Checkpoint::collect_range(&mrt, b"b", Some(b"c")).unwrap()
        );
        assert_eq!(
            <crate::mrt::Checkpoint as TraceReader>::get_prefix(&mut mrt, b"b").unwrap(),
            crate::mrt::Checkpoint::collect_prefix(&mrt, b"b").unwrap()
        );
    }

    #[test]
    fn trace_reader_verifier_reads_match_direct_reads_for_avl_and_mrt() {
        let avl_snapshot =
            build_avl_snapshot(&[(b"a", b"1"), (b"aa", b"11"), (b"ab", b"12"), (b"b", b"2")]);
        let avl_reads = vec![
            ReadOp::Key(b"aa".to_vec()),
            ReadOp::Range {
                start: b"a".to_vec(),
                end: b"b".to_vec(),
            },
            ReadOp::Prefix(b"a".to_vec()),
        ];
        let (avl_trace, avl_start) = avl_trace_for_reads(&avl_snapshot, avl_reads);
        let mut avl = crate::avl::TraceVerifier::from_trace(&avl_trace);
        avl.verify_root(avl_start).unwrap();
        assert_eq!(
            <crate::avl::TraceVerifier as TraceReader>::get(&mut avl, b"aa").unwrap(),
            crate::avl::TraceVerifier::get(&avl, b"aa").unwrap()
        );
        assert_eq!(
            <crate::avl::TraceVerifier as TraceReader>::get_range(&mut avl, b"a", b"b").unwrap(),
            crate::avl::TraceVerifier::collect_range(&avl, b"a", Some(b"b")).unwrap()
        );
        assert_eq!(
            <crate::avl::TraceVerifier as TraceReader>::get_prefix(&mut avl, b"a").unwrap(),
            crate::avl::TraceVerifier::collect_prefix(&avl, b"a").unwrap()
        );

        let mrt_snapshot =
            build_mrt_snapshot(&[(b"a", b"1"), (b"ba", b"21"), (b"bb", b"22"), (b"c", b"3")]);
        let mrt_reads = vec![
            ReadOp::Key(b"ba".to_vec()),
            ReadOp::Range {
                start: b"b".to_vec(),
                end: b"c".to_vec(),
            },
            ReadOp::Prefix(b"b".to_vec()),
        ];
        let (mrt_trace, mrt_start) = mrt_trace_for_reads(&mrt_snapshot, mrt_reads);
        let mut mrt = crate::mrt::TraceVerifier::from_trace(&mrt_trace);
        mrt.verify_root(mrt_start).unwrap();
        assert_eq!(
            <crate::mrt::TraceVerifier as TraceReader>::get(&mut mrt, b"ba").unwrap(),
            crate::mrt::TraceVerifier::get(&mrt, b"ba").unwrap()
        );
        assert_eq!(
            <crate::mrt::TraceVerifier as TraceReader>::get_range(&mut mrt, b"b", b"c").unwrap(),
            crate::mrt::TraceVerifier::collect_range(&mrt, b"b", Some(b"c")).unwrap()
        );
        assert_eq!(
            <crate::mrt::TraceVerifier as TraceReader>::get_prefix(&mut mrt, b"b").unwrap(),
            crate::mrt::TraceVerifier::collect_prefix(&mrt, b"b").unwrap()
        );
    }

    #[test]
    fn trace_reader_prefix_matches_range_when_successor_exists() {
        let mut avl = build_avl_snapshot(&[(b"aa", b"1"), (b"ab", b"2"), (b"b", b"3")]);
        let prefix = b"a";
        let end = prefix_successor(prefix).unwrap();
        assert_eq!(
            <crate::avl::Checkpoint as TraceReader>::get_prefix(&mut avl, prefix).unwrap(),
            <crate::avl::Checkpoint as TraceReader>::get_range(&mut avl, prefix, &end).unwrap()
        );

        let mut mrt = build_mrt_snapshot(&[(b"aa", b"1"), (b"ab", b"2"), (b"b", b"3")]);
        assert_eq!(
            <crate::mrt::Checkpoint as TraceReader>::get_prefix(&mut mrt, prefix).unwrap(),
            <crate::mrt::Checkpoint as TraceReader>::get_range(&mut mrt, prefix, &end).unwrap()
        );
    }

    #[test]
    fn trace_reader_prefix_handles_no_successor_prefix() {
        let expected = vec![
            (vec![0xff, 0x00], b"first".to_vec()),
            (vec![0xff, 0x10], b"second".to_vec()),
        ];

        let mut avl = build_avl_snapshot(&[
            (&[0xfe], b"before"),
            (&[0xff, 0x00], b"first"),
            (&[0xff, 0x10], b"second"),
        ]);
        assert_eq!(
            <crate::avl::Checkpoint as TraceReader>::get_prefix(&mut avl, &[0xff]).unwrap(),
            expected
        );

        let mut mrt = build_mrt_snapshot(&[
            (&[0xfe], b"before"),
            (&[0xff, 0x00], b"first"),
            (&[0xff, 0x10], b"second"),
        ]);
        assert_eq!(
            <crate::mrt::Checkpoint as TraceReader>::get_prefix(&mut mrt, &[0xff]).unwrap(),
            expected
        );
    }

    #[test]
    fn trace_reader_is_object_safe() {
        let mut snapshot = build_avl_snapshot(&[(b"a", b"1")]);
        let reader: &mut dyn TraceReader = &mut snapshot;
        assert_eq!(reader.get(b"a").unwrap(), Some(b"1".to_vec()));
    }

    fn read_with_generic_bound<T: TraceReader>(reader: &mut T, key: &[u8]) -> Option<Vec<u8>> {
        reader.get(key).unwrap()
    }

    #[test]
    fn trace_reader_accepts_generic_bounds() {
        let mut snapshot = build_mrt_snapshot(&[(b"a", b"1")]);
        assert_eq!(
            read_with_generic_bound(&mut snapshot, b"a"),
            Some(b"1".to_vec())
        );
    }

    #[derive(Default)]
    struct InterfaceTestDouble {
        entries: Vec<(Vec<u8>, Vec<u8>)>,
        moved: Vec<(Vec<u8>, Vec<u8>)>,
    }

    impl TraceReader for InterfaceTestDouble {
        fn get(&mut self, key: &[u8]) -> crate::Result<Option<Vec<u8>>> {
            Ok(self
                .entries
                .iter()
                .find(|(entry_key, _)| entry_key == key)
                .map(|(_, value)| value.clone()))
        }

        fn get_range(
            &mut self,
            start: &[u8],
            end: &[u8],
        ) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
            Ok(self
                .entries
                .iter()
                .filter(|(key, _)| key.as_slice() >= start && key.as_slice() < end)
                .cloned()
                .collect())
        }

        fn get_prefix(&mut self, prefix: &[u8]) -> crate::Result<Vec<(Vec<u8>, Vec<u8>)>> {
            Ok(self
                .entries
                .iter()
                .filter(|(key, _)| key.starts_with(prefix))
                .cloned()
                .collect())
        }
    }

    impl TraceInterface for InterfaceTestDouble {
        fn apply(&mut self, ops: &[WriteOp]) -> crate::Result<()> {
            for op in ops {
                match op {
                    WriteOp::Put { key, value } => {
                        self.entries.retain(|(entry_key, _)| entry_key != key);
                        self.entries.push((key.clone(), value.clone()));
                        self.entries.sort_by(|a, b| a.0.cmp(&b.0));
                    }
                    WriteOp::Delete { key } => {
                        self.entries.retain(|(entry_key, _)| entry_key != key);
                    }
                    WriteOp::DeleteRange { start, end } => {
                        self.entries.retain(|(key, _)| {
                            key.as_slice() < start.as_slice() || key.as_slice() >= end.as_slice()
                        });
                    }
                    WriteOp::DeletePrefix { prefix } => {
                        self.entries
                            .retain(|(key, _)| !key.starts_with(prefix.as_slice()));
                    }
                    WriteOp::MovePrefix { from, to } => {
                        self.moved.push((from.clone(), to.clone()));
                    }
                }
            }
            Ok(())
        }
    }

    #[test]
    fn trace_interface_is_object_safe_with_test_double() {
        let mut handle = InterfaceTestDouble::default();
        let interface: &mut dyn TraceInterface = &mut handle;
        interface.put(b"a", b"1").unwrap();
        interface.put(b"b", b"2").unwrap();
        interface.delete_range(b"b", b"c").unwrap();
        assert_eq!(
            interface.get_prefix(b"a").unwrap(),
            vec![(b"a".to_vec(), b"1".to_vec())]
        );
        interface.delete_prefix(b"a").unwrap();
        interface.move_prefix(b"old", b"new").unwrap();
        assert_eq!(interface.get(b"a").unwrap(), None);
    }

    #[test]
    fn trace_interface_apply_is_object_safe() {
        // `apply` (the single required write method) must be callable on a trait
        // object; an empty batch is a no-op and a multi-op batch runs in order.
        let mut handle = InterfaceTestDouble::default();
        let interface: &mut dyn TraceInterface = &mut handle;
        interface.apply(&[]).unwrap();
        interface
            .apply(&[
                WriteOp::Put {
                    key: b"a".to_vec(),
                    value: b"1".to_vec(),
                },
                WriteOp::Put {
                    key: b"a".to_vec(),
                    value: b"2".to_vec(),
                },
                WriteOp::MovePrefix {
                    from: b"x".to_vec(),
                    to: b"y".to_vec(),
                },
            ])
            .unwrap();
        assert_eq!(interface.get(b"a").unwrap(), Some(b"2".to_vec()));
        assert_eq!(handle.moved, vec![(b"x".to_vec(), b"y".to_vec())]);
    }

    fn write_with_generic_bound<T: TraceInterface>(handle: &mut T) -> Option<Vec<u8>> {
        handle.put(b"a", b"1").unwrap();
        handle.put(b"a", b"2").unwrap();
        handle.delete(b"missing").unwrap();
        handle.get(b"a").unwrap()
    }

    #[test]
    fn trace_interface_accepts_generic_bounds_with_test_double() {
        let mut handle = InterfaceTestDouble::default();
        assert_eq!(write_with_generic_bound(&mut handle), Some(b"2".to_vec()));
        assert_eq!(handle.moved, Vec::<(Vec<u8>, Vec<u8>)>::new());
    }

    fn exercise_real_replayer_interface(handle: &mut dyn TraceInterface) {
        handle.put(b"a", b"1").unwrap();
        handle.put(b"a", b"2").unwrap();
        handle.delete(b"missing").unwrap();
        assert_eq!(handle.get(b"a").unwrap(), Some(b"2".to_vec()));
        handle.delete_prefix(b"a").unwrap();
        assert_eq!(handle.get(b"a").unwrap(), None);
    }

    fn exercise_real_replayer_generic<T: TraceInterface>(handle: &mut T) {
        handle.put(b"k", b"v1").unwrap();
        handle.delete(b"k").unwrap();
        handle.put(b"k", b"v2").unwrap();
        assert_eq!(handle.get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn trace_interface_object_and_generic_paths_run_against_replayers() {
        let avl_tree = crate::avl::Tree::new();
        let avl_trace = test_support::avl::create_trace(&avl_tree.checkpoint(), &[]).unwrap();
        let avl_start = avl_trace.hash();
        let avl_bytes = avl_trace.encode().unwrap();
        let mut avl = crate::avl::TraceReplayer::new_verified(&avl_bytes, avl_start).unwrap();
        exercise_real_replayer_interface(&mut avl);

        let mrt_tree = crate::mrt::Tree::new();
        let mrt_trace = test_support::mrt::create_trace(&mrt_tree.checkpoint(), &[]).unwrap();
        let mrt_start = mrt_trace.root_hash();
        let mrt_bytes = mrt_trace.encode().unwrap();
        let mut mrt = crate::mrt::TraceReplayer::new_verified(&mrt_bytes, mrt_start).unwrap();
        exercise_real_replayer_interface(&mut mrt);

        let avl_trace = test_support::avl::create_trace(&avl_tree.checkpoint(), &[]).unwrap();
        let avl_start = avl_trace.hash();
        let avl_bytes = avl_trace.encode().unwrap();
        let mut avl = crate::avl::TraceReplayer::new_verified(&avl_bytes, avl_start).unwrap();
        exercise_real_replayer_generic(&mut avl);

        let mrt_trace = test_support::mrt::create_trace(&mrt_tree.checkpoint(), &[]).unwrap();
        let mrt_start = mrt_trace.root_hash();
        let mrt_bytes = mrt_trace.encode().unwrap();
        let mut mrt = crate::mrt::TraceReplayer::new_verified(&mrt_bytes, mrt_start).unwrap();
        exercise_real_replayer_generic(&mut mrt);
    }
}
