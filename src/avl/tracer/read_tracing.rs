use std::collections::HashSet;

use crate::avl::node::Node;
use crate::error::Result;

use super::assembly::{accessed_keys_from_nodes, assemble_sparse_trace};
use super::recording::{get_recording, prefix_recording, range_recording};
use super::{extract_reads_sparse, ProvenRead, ReadOp, ReadResults, SparseMerkNode};

/// Accumulator for traced read operations against a live AVL tree.
///
/// Collects every visited node and every read-target key across one or more
/// read operations. The accumulated sets feed into [`assemble_sparse_trace`]
/// to produce a sparse proof tree where opened nodes are
/// [`SparseMerkNode::Full`].
pub struct ReadTracer {
    visited: Vec<Node>,
    read_target_keys: HashSet<Vec<u8>>,
}

impl ReadTracer {
    pub fn new() -> Self {
        Self {
            visited: Vec::new(),
            read_target_keys: HashSet::new(),
        }
    }

    /// Point read. Records the descent path; marks the key as a read target
    /// only if present.
    pub fn get(&mut self, root: &Node, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let result = get_recording(root, key, &mut |n: &Node| {
            self.visited.push(n.clone());
        })?;
        if result.is_some() {
            self.read_target_keys.insert(key.to_vec());
        }
        Ok(result)
    }

    /// Range read `[start, end)`. Records every visited node; marks every
    /// yielded key as a read target.
    pub fn range(
        &mut self,
        root: &Node,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let results = range_recording(root, start, end, &mut |n: &Node| {
            self.visited.push(n.clone());
        })?;
        for (key, _) in &results {
            self.read_target_keys.insert(key.clone());
        }
        Ok(results)
    }

    /// Prefix read. Records every visited node; marks every yielded key as a
    /// read target.
    pub fn prefix(&mut self, root: &Node, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let results = prefix_recording(root, prefix, &mut |n: &Node| {
            self.visited.push(n.clone());
        })?;
        for (key, _) in &results {
            self.read_target_keys.insert(key.clone());
        }
        Ok(results)
    }

    /// Execute a [`ReadOp`] and return the raw key/value results.
    pub fn execute_read_op(&mut self, root: &Node, op: &ReadOp) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        match op {
            ReadOp::Key(key) => Ok(self
                .get(root, key)?
                .into_iter()
                .map(|v| (key.clone(), v))
                .collect()),
            ReadOp::Range { start, end } => self.range(root, start, Some(end)),
            ReadOp::Prefix(prefix) => self.prefix(root, prefix),
        }
    }

    /// Decompose into the deduplicated accessed-key set and the read-target
    /// key set. These feed directly into [`assemble_sparse_trace`].
    pub fn into_sets(self) -> (HashSet<Vec<u8>>, HashSet<Vec<u8>>) {
        let accessed = accessed_keys_from_nodes(&self.visited);
        (accessed, self.read_target_keys)
    }

    /// Borrow the current sets without consuming the tracer.
    pub fn sets(&self) -> (HashSet<Vec<u8>>, HashSet<Vec<u8>>) {
        let accessed = accessed_keys_from_nodes(&self.visited);
        (accessed, self.read_target_keys.clone())
    }

    pub fn visit_count(&self) -> usize {
        self.visited.len()
    }

    pub fn read_target_count(&self) -> usize {
        self.read_target_keys.len()
    }
}

impl Default for ReadTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a batch of [`ReadOp`]s with full tracing, then assemble a sparse
/// proof tree and populate verifier-side results from it.
///
/// Returns `(sparse_tree, proven_reads)`. The sparse tree's root hash equals
/// the original tree's root hash. Opened nodes are emitted as
/// [`SparseMerkNode::Full`], while untouched subtrees are pruned by hash.
pub fn trace_and_prove_reads(
    root: &Node,
    reads: &[ReadOp],
) -> Result<(SparseMerkNode, ReadResults)> {
    let mut tracer = ReadTracer::new();

    for op in reads {
        tracer.execute_read_op(root, op)?;
    }

    let (accessed_keys, read_target_keys) = tracer.into_sets();
    let sparse = assemble_sparse_trace(Some(root), &accessed_keys, &read_target_keys)?;

    let stub_reads: Vec<ProvenRead> = reads
        .iter()
        .map(|op| ProvenRead {
            op: op.clone(),
            results: Vec::new(),
        })
        .collect();
    let populated = extract_reads_sparse(&sparse, &stub_reads)?;

    Ok((sparse, populated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avl::in_memory::InMemoryMerk;
    use crate::error::Error;
    use crate::hash::{kv_hash, Hasher};
    use crate::tracer::SMALL_VALUE_INLINE_THRESHOLD;

    fn build_tree(entries: &[(&[u8], &[u8])]) -> Node {
        let merk = InMemoryMerk::new();
        for &(k, v) in entries {
            merk.put(k, v).unwrap();
        }
        merk.checkpoint().into_root().unwrap()
    }

    // -----------------------------------------------------------------------
    // ReadTracer unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn point_read_present_marks_target() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let mut tracer = ReadTracer::new();
        let result = tracer.get(&root, b"c").unwrap();
        assert_eq!(result, Some(b"3".to_vec()));

        let (accessed, targets) = tracer.into_sets();
        assert!(targets.contains(b"c".as_slice()));
        assert!(!accessed.is_empty());
    }

    #[test]
    fn point_read_absent_does_not_mark_target() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let mut tracer = ReadTracer::new();
        let result = tracer.get(&root, b"b").unwrap();
        assert_eq!(result, None);

        let (accessed, targets) = tracer.into_sets();
        assert!(targets.is_empty());
        assert!(!accessed.is_empty());
    }

    #[test]
    fn range_read_marks_yielded_keys_as_targets() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"c", b"3"),
            (b"e", b"5"),
            (b"g", b"7"),
            (b"i", b"9"),
        ]);
        let mut tracer = ReadTracer::new();
        let results = tracer.range(&root, b"c", Some(b"h")).unwrap();

        let result_keys: Vec<&[u8]> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(
            result_keys,
            vec![b"c".as_slice(), b"e".as_slice(), b"g".as_slice()]
        );

        let (accessed, targets) = tracer.into_sets();
        assert!(targets.contains(b"c".as_slice()));
        assert!(targets.contains(b"e".as_slice()));
        assert!(targets.contains(b"g".as_slice()));
        assert!(!targets.contains(b"a".as_slice()));
        assert!(!targets.contains(b"i".as_slice()));
        assert!(accessed.len() >= targets.len());
    }

    #[test]
    fn prefix_read_marks_yielded_keys_as_targets() {
        let root = build_tree(&[
            (b"pre_a", b"1"),
            (b"pre_b", b"2"),
            (b"pre_c", b"3"),
            (b"xyz", b"4"),
        ]);
        let mut tracer = ReadTracer::new();
        let results = tracer.prefix(&root, b"pre_").unwrap();

        let result_keys: Vec<&[u8]> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(result_keys.contains(&b"pre_a".as_slice()));
        assert!(result_keys.contains(&b"pre_b".as_slice()));
        assert!(result_keys.contains(&b"pre_c".as_slice()));

        let (_, targets) = tracer.into_sets();
        assert!(targets.contains(b"pre_a".as_slice()));
        assert!(targets.contains(b"pre_b".as_slice()));
        assert!(targets.contains(b"pre_c".as_slice()));
        assert!(!targets.contains(b"xyz".as_slice()));
    }

    #[test]
    fn multiple_reads_accumulate_targets() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let mut tracer = ReadTracer::new();
        tracer.get(&root, b"a").unwrap();
        tracer.get(&root, b"e").unwrap();
        tracer.range(&root, b"c", Some(b"d")).unwrap();

        let (_, targets) = tracer.into_sets();
        assert!(targets.contains(b"a".as_slice()));
        assert!(targets.contains(b"c".as_slice()));
        assert!(targets.contains(b"e".as_slice()));
        assert!(!targets.contains(b"g".as_slice()));
    }

    #[test]
    fn empty_range_yields_no_targets() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let mut tracer = ReadTracer::new();
        let results = tracer.range(&root, b"f", Some(b"z")).unwrap();
        assert!(results.is_empty());

        let (accessed, targets) = tracer.into_sets();
        assert!(targets.is_empty());
        assert!(!accessed.is_empty());
    }

    #[test]
    fn execute_read_op_dispatches_correctly() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5"), (b"g", b"7")]);
        let mut tracer = ReadTracer::new();

        let kv = tracer
            .execute_read_op(&root, &ReadOp::Key(b"c".to_vec()))
            .unwrap();
        assert_eq!(kv, vec![(b"c".to_vec(), b"3".to_vec())]);

        let range = tracer
            .execute_read_op(
                &root,
                &ReadOp::Range {
                    start: b"a".to_vec(),
                    end: b"d".to_vec(),
                },
            )
            .unwrap();
        assert_eq!(
            range,
            vec![
                (b"a".to_vec(), b"1".to_vec()),
                (b"c".to_vec(), b"3".to_vec()),
            ]
        );

        let prefix = tracer
            .execute_read_op(&root, &ReadOp::Prefix(b"e".to_vec()))
            .unwrap();
        assert_eq!(prefix, vec![(b"e".to_vec(), b"5".to_vec())]);
    }

    #[test]
    fn sets_borrows_without_consuming() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3")]);
        let mut tracer = ReadTracer::new();
        tracer.get(&root, b"a").unwrap();

        let (accessed1, targets1) = tracer.sets();
        assert!(targets1.contains(b"a".as_slice()));

        tracer.get(&root, b"c").unwrap();
        let (accessed2, targets2) = tracer.sets();
        assert!(targets2.contains(b"a".as_slice()));
        assert!(targets2.contains(b"c".as_slice()));
        assert!(accessed2.len() >= accessed1.len());
    }

    #[test]
    fn visit_and_target_counts() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let mut tracer = ReadTracer::new();
        assert_eq!(tracer.visit_count(), 0);
        assert_eq!(tracer.read_target_count(), 0);

        tracer.get(&root, b"a").unwrap();
        assert!(tracer.visit_count() > 0);
        assert_eq!(tracer.read_target_count(), 1);

        tracer.get(&root, b"b").unwrap();
        assert_eq!(tracer.read_target_count(), 1);
    }

    // -----------------------------------------------------------------------
    // End-to-end: trace_and_prove_reads
    // -----------------------------------------------------------------------

    #[test]
    fn prove_point_present() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let (sparse, reads) = trace_and_prove_reads(&root, &[ReadOp::Key(b"c".to_vec())]).unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].results, vec![(b"c".to_vec(), b"3".to_vec())]);
    }

    #[test]
    fn prove_point_absent() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let (sparse, reads) = trace_and_prove_reads(&root, &[ReadOp::Key(b"b".to_vec())]).unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert!(reads[0].results.is_empty());
    }

    #[test]
    fn prove_range() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"c", b"3"),
            (b"e", b"5"),
            (b"g", b"7"),
            (b"i", b"9"),
        ]);
        let (sparse, reads) = trace_and_prove_reads(
            &root,
            &[ReadOp::Range {
                start: b"c".to_vec(),
                end: b"h".to_vec(),
            }],
        )
        .unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert_eq!(
            reads[0].results,
            vec![
                (b"c".to_vec(), b"3".to_vec()),
                (b"e".to_vec(), b"5".to_vec()),
                (b"g".to_vec(), b"7".to_vec()),
            ]
        );
    }

    #[test]
    fn prove_prefix() {
        let root = build_tree(&[
            (b"pre_a", b"1"),
            (b"pre_b", b"2"),
            (b"pre_c", b"3"),
            (b"xyz", b"4"),
        ]);
        let (sparse, reads) =
            trace_and_prove_reads(&root, &[ReadOp::Prefix(b"pre_".to_vec())]).unwrap();

        assert_eq!(sparse.hash(), root.hash());
        let result_keys: Vec<&[u8]> = reads[0].results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(result_keys.contains(&b"pre_a".as_slice()));
        assert!(result_keys.contains(&b"pre_b".as_slice()));
        assert!(result_keys.contains(&b"pre_c".as_slice()));
        assert!(!result_keys.contains(&b"xyz".as_slice()));
    }

    #[test]
    fn prove_mixed_reads() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"c", b"3"),
            (b"e", b"5"),
            (b"g", b"7"),
            (b"i", b"9"),
        ]);
        let (sparse, reads) = trace_and_prove_reads(
            &root,
            &[
                ReadOp::Key(b"c".to_vec()),
                ReadOp::Range {
                    start: b"e".to_vec(),
                    end: b"i".to_vec(),
                },
                ReadOp::Prefix(b"a".to_vec()),
            ],
        )
        .unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert_eq!(reads[0].results, vec![(b"c".to_vec(), b"3".to_vec())]);
        assert_eq!(
            reads[1].results,
            vec![
                (b"e".to_vec(), b"5".to_vec()),
                (b"g".to_vec(), b"7".to_vec()),
            ]
        );
        assert_eq!(reads[2].results, vec![(b"a".to_vec(), b"1".to_vec())]);
    }

    #[test]
    fn prove_empty_range() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let (sparse, reads) = trace_and_prove_reads(
            &root,
            &[ReadOp::Range {
                start: b"f".to_vec(),
                end: b"z".to_vec(),
            }],
        )
        .unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert!(reads[0].results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Large values: opened nodes stay full
    // -----------------------------------------------------------------------

    #[test]
    fn large_value_target_stays_full() {
        let big = vec![0xAA; SMALL_VALUE_INLINE_THRESHOLD + 1];
        let root = build_tree(&[(b"a", &big), (b"c", &big), (b"e", &big)]);
        let (sparse, reads) = trace_and_prove_reads(&root, &[ReadOp::Key(b"c".to_vec())]).unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert_eq!(reads[0].results, vec![(b"c".to_vec(), big)]);
    }

    #[test]
    fn large_value_non_target_stays_full() {
        let big = vec![0xBB; SMALL_VALUE_INLINE_THRESHOLD + 1];
        let root = build_tree(&[
            (b"a", &big),
            (b"b", &big),
            (b"c", &big),
            (b"d", &big),
            (b"e", b"small"),
        ]);

        let mut tracer = ReadTracer::new();
        let result = tracer.get(&root, b"e").unwrap();
        assert_eq!(result, Some(b"small".to_vec()));
        assert!(
            tracer.visit_count() > 1,
            "reading a non-root leaf must visit intermediate nodes"
        );

        let (accessed, targets) = tracer.into_sets();
        assert_eq!(targets.len(), 1);
        assert!(accessed.len() > 1);

        let sparse = assemble_sparse_trace(Some(&root), &accessed, &targets).unwrap();
        assert_eq!(sparse.hash(), root.hash());

        fn count_hash_only(trace: &SparseMerkNode) -> usize {
            match trace {
                SparseMerkNode::FullOmitted { .. } | SparseMerkNode::FullStorageHash { .. } => 1,
                SparseMerkNode::Full { left, right, .. } => {
                    count_hash_only(left) + count_hash_only(right)
                }
                _ => 0,
            }
        }
        assert_eq!(
            count_hash_only(&sparse),
            0,
            "generated read traces should not emit hash-only opened nodes"
        );
    }

    #[test]
    fn absent_read_path_nodes_stay_full() {
        let big = vec![0xCC; SMALL_VALUE_INLINE_THRESHOLD + 1];
        let root = build_tree(&[(b"a", &big), (b"c", &big), (b"e", &big)]);

        let (sparse, reads) = trace_and_prove_reads(&root, &[ReadOp::Key(b"b".to_vec())]).unwrap();

        assert_eq!(sparse.hash(), root.hash());
        assert!(reads[0].results.is_empty());
        assert_eq!(sparse.get(b"b").unwrap(), None);

        fn count_full_and_hash_only(trace: &SparseMerkNode) -> (usize, usize) {
            match trace {
                SparseMerkNode::Full { left, right, .. } => {
                    let (left_full, left_hash_only) = count_full_and_hash_only(left);
                    let (right_full, right_hash_only) = count_full_and_hash_only(right);
                    (1 + left_full + right_full, left_hash_only + right_hash_only)
                }
                SparseMerkNode::FullOmitted { left, right, .. }
                | SparseMerkNode::FullStorageHash { left, right, .. } => {
                    let (left_full, left_hash_only) = count_full_and_hash_only(left);
                    let (right_full, right_hash_only) = count_full_and_hash_only(right);
                    (left_full + right_full, 1 + left_hash_only + right_hash_only)
                }
                _ => (0, 0),
            }
        }
        let (full, hash_only) = count_full_and_hash_only(&sparse);
        assert!(
            full > 0,
            "absent read should open its descent path as Full nodes"
        );
        assert_eq!(
            hash_only, 0,
            "generated absent-read traces should not emit hash-only opened nodes"
        );
    }

    // -----------------------------------------------------------------------
    // Verifier rejection: FullOmitted cannot be yielded
    // -----------------------------------------------------------------------

    #[test]
    fn extract_rejects_omitted_node_in_point_read() {
        let omitted = SparseMerkNode::FullOmitted {
            key: b"k".to_vec(),
            kv_hash: kv_hash::<Hasher>(b"k", b"val").unwrap(),
            left: Box::new(SparseMerkNode::Empty),
            right: Box::new(SparseMerkNode::Empty),
        };

        let reads = vec![ProvenRead {
            op: ReadOp::Key(b"k".to_vec()),
            results: Vec::new(),
        }];
        let err = extract_reads_sparse(&omitted, &reads).unwrap_err();
        assert!(matches!(err, Error::ValueOmitted(_)));
    }

    #[test]
    fn extract_rejects_omitted_node_in_range() {
        let omitted_root = SparseMerkNode::FullOmitted {
            key: b"m".to_vec(),
            kv_hash: kv_hash::<Hasher>(b"m", b"middle").unwrap(),
            left: Box::new(SparseMerkNode::Full {
                key: b"a".to_vec(),
                value: b"left".to_vec(),
                left: Box::new(SparseMerkNode::Empty),
                right: Box::new(SparseMerkNode::Empty),
            }),
            right: Box::new(SparseMerkNode::Full {
                key: b"z".to_vec(),
                value: b"right".to_vec(),
                left: Box::new(SparseMerkNode::Empty),
                right: Box::new(SparseMerkNode::Empty),
            }),
        };

        let reads = vec![ProvenRead {
            op: ReadOp::Range {
                start: b"a".to_vec(),
                end: b"z".to_vec(),
            },
            results: Vec::new(),
        }];
        let err = extract_reads_sparse(&omitted_root, &reads).unwrap_err();
        assert!(matches!(err, Error::ValueOmitted(_)));
    }

    #[test]
    fn extract_routes_through_omitted_without_yielding() {
        let omitted_root = SparseMerkNode::FullOmitted {
            key: b"m".to_vec(),
            kv_hash: kv_hash::<Hasher>(b"m", b"middle").unwrap(),
            left: Box::new(SparseMerkNode::Full {
                key: b"a".to_vec(),
                value: b"left".to_vec(),
                left: Box::new(SparseMerkNode::Empty),
                right: Box::new(SparseMerkNode::Empty),
            }),
            right: Box::new(SparseMerkNode::Full {
                key: b"z".to_vec(),
                value: b"right".to_vec(),
                left: Box::new(SparseMerkNode::Empty),
                right: Box::new(SparseMerkNode::Empty),
            }),
        };

        let reads = vec![ProvenRead {
            op: ReadOp::Range {
                start: b"a".to_vec(),
                end: b"b".to_vec(),
            },
            results: Vec::new(),
        }];
        let populated = extract_reads_sparse(&omitted_root, &reads).unwrap();
        assert_eq!(
            populated[0].results,
            vec![(b"a".to_vec(), b"left".to_vec())]
        );
    }
}
