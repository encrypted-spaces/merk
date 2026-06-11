use std::sync::{Arc, Mutex};

use crate::avl::child::Child;
use crate::avl::node::Node;
use crate::avl::walker::Fetch;
use crate::error::{Error, Result};

/// `Fetch` source that records every visited node into a shared buffer.
///
/// `Walker` calls `record_visit` at each `detach` (for both the parent
/// and the detached child, pre-modification). This gives full coverage of
/// descent paths and rotation ancestors without modifying the `Walker`
/// pipeline.
///
/// `Node::clone` is an `Arc` bump, so recorded nodes cheaply share their
/// `NodeInner` with the live tree. If the live tree later mutates a node
/// via `Arc::make_mut`, the recorded copy retains the pre-state data.
#[derive(Clone)]
pub struct RecordingSource {
    visits: Arc<Mutex<Vec<Node>>>,
}

impl RecordingSource {
    pub fn new() -> Self {
        Self {
            visits: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn into_visits(self) -> Result<Vec<Node>> {
        Arc::try_unwrap(self.visits)
            .map_err(|_| {
                Error::Tree("RecordingSource still referenced after apply returned".into())
            })?
            .into_inner()
            .map_err(|_| Error::Tree("RecordingSource visits mutex poisoned".into()))
    }

    pub fn visit_count(&self) -> usize {
        self.visits.lock().expect("poisoned").len()
    }
}

impl Default for RecordingSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Fetch for RecordingSource {
    fn fetch_by_key(&self, key: &[u8]) -> Result<Option<Node>> {
        Err(Error::PrunedNode(format!(
            "RecordingSource fetch_by_key for {key:?}: \
             in-memory tree must be fully resident"
        )))
    }

    fn record_visit(&self, node: &Node) {
        self.visits
            .lock()
            .expect("RecordingSource visits mutex poisoned")
            .push(node.clone());
    }
}

/// Point lookup that records every node on the descent path.
///
/// Returns `Some(value)` if the key is present, `None` if absent.
/// Errors if the descent hits a pruned child.
pub fn get_recording<F>(root: &Node, key: &[u8], recorder: &mut F) -> Result<Option<Vec<u8>>>
where
    F: FnMut(&Node),
{
    let mut cursor = root;
    loop {
        recorder(cursor);

        if key == cursor.key() {
            return Ok(Some(cursor.value().to_vec()));
        }

        let left = key < cursor.key();
        match cursor.child_ref(left) {
            None => return Ok(None),
            Some(Child::Resident(child)) => cursor = child,
            Some(Child::Pruned(pruned)) => {
                return Err(Error::PrunedNode(format!(
                    "recording get for {key:?} descended into pruned node {:?}",
                    pruned.key()
                )));
            }
        }
    }
}

/// Forward range scan `[start, end)` that records every visited node.
///
/// If `end` is `None`, the scan is unbounded on the right.
/// Errors if the traversal hits a pruned child that lies within the range.
pub fn range_recording<F>(
    root: &Node,
    start: &[u8],
    end: Option<&[u8]>,
    recorder: &mut F,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>>
where
    F: FnMut(&Node),
{
    let mut results = Vec::new();
    range_inner(root, start, end, recorder, &mut results)?;
    Ok(results)
}

fn range_inner<F>(
    node: &Node,
    start: &[u8],
    end: Option<&[u8]>,
    recorder: &mut F,
    results: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<()>
where
    F: FnMut(&Node),
{
    recorder(node);
    let key = node.key();

    if key > start {
        match node.child_ref(true) {
            Some(Child::Resident(left)) => {
                range_inner(left, start, end, recorder, results)?;
            }
            Some(Child::Pruned(pruned)) => {
                return Err(Error::PrunedNode(format!(
                    "range recording descended into pruned left child {:?}",
                    pruned.key()
                )));
            }
            None => {}
        }
    }

    if key >= start && end.is_none_or(|e| key < e) {
        results.push((key.to_vec(), node.value().to_vec()));
    }

    if end.is_none_or(|e| key < e) {
        match node.child_ref(false) {
            Some(Child::Resident(right)) => {
                range_inner(right, start, end, recorder, results)?;
            }
            Some(Child::Pruned(pruned)) => {
                return Err(Error::PrunedNode(format!(
                    "range recording descended into pruned right child {:?}",
                    pruned.key()
                )));
            }
            None => {}
        }
    }

    Ok(())
}

/// Prefix scan that records every visited node.
///
/// Equivalent to `range_recording(root, prefix, prefix_successor, recorder)`.
pub fn prefix_recording<F>(
    root: &Node,
    prefix: &[u8],
    recorder: &mut F,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>>
where
    F: FnMut(&Node),
{
    let end = crate::tracer::prefix_successor(prefix);
    range_recording(root, prefix, end.as_deref(), recorder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avl::in_memory::InMemoryMerk;
    use crate::avl::walker::Walker;
    use crate::avl::PanicSource;

    fn build_tree(entries: &[(&[u8], &[u8])]) -> Node {
        let merk = InMemoryMerk::new();
        for &(k, v) in entries {
            merk.put(k, v).unwrap();
        }
        merk.checkpoint().into_root().unwrap()
    }

    #[test]
    fn recording_source_collects_visits_during_apply() {
        let merk = InMemoryMerk::new();
        for i in 0u8..10 {
            merk.put(vec![i], vec![i + 100]).unwrap();
        }
        let root = merk.checkpoint().into_root().unwrap();

        let source = RecordingSource::new();
        let walker = Walker::new(root, source.clone());

        let mut batch = [(vec![5u8], crate::ops::Op::Put(vec![55]))];
        let (maybe_tree, _) =
            Walker::apply_to_mut(Some(walker), &mut batch, source.clone()).unwrap();
        assert!(maybe_tree.is_some());

        let visits = source.into_visits().unwrap();
        assert!(
            !visits.is_empty(),
            "RecordingSource should have recorded visits"
        );

        let visited_keys: Vec<Vec<u8>> = visits.iter().map(|n| n.key().to_vec()).collect();
        assert!(
            visited_keys.contains(&vec![5u8]),
            "visited keys should include the target key"
        );
    }

    #[test]
    fn recording_source_records_root_key_update() {
        let merk = InMemoryMerk::new();
        merk.put(b"root_key", b"old_value").unwrap();
        let root = merk.checkpoint().into_root().unwrap();
        let root_key = root.key().to_vec();

        let source = RecordingSource::new();
        let walker = Walker::new(root, source.clone());

        let mut batch = [(root_key.clone(), crate::ops::Op::Put(b"new_value".to_vec()))];
        let (maybe_tree, _) =
            Walker::apply_to_mut(Some(walker), &mut batch, source.clone()).unwrap();
        assert!(maybe_tree.is_some());

        let visits = source.into_visits().unwrap();
        let visited_keys: Vec<Vec<u8>> = visits.iter().map(|n| n.key().to_vec()).collect();
        assert!(
            visited_keys.contains(&root_key),
            "single-key root update must record the root node"
        );
    }

    #[test]
    fn recording_source_records_matched_non_root_update() {
        let merk = InMemoryMerk::new();
        for i in 0u8..7 {
            merk.put(vec![i], vec![i + 100]).unwrap();
        }
        let root = merk.checkpoint().into_root().unwrap();

        let source = RecordingSource::new();
        let walker = Walker::new(root, source.clone());

        let mut batch = [(vec![2u8], crate::ops::Op::Put(vec![99]))];
        let (maybe_tree, _) =
            Walker::apply_to_mut(Some(walker), &mut batch, source.clone()).unwrap();
        assert!(maybe_tree.is_some());

        let visits = source.into_visits().unwrap();
        let visited_keys: Vec<Vec<u8>> = visits.iter().map(|n| n.key().to_vec()).collect();
        assert!(
            visited_keys.contains(&vec![2u8]),
            "matched-key update must record the updated node"
        );
    }

    #[test]
    fn recording_source_records_delete_range_visits() {
        let merk = InMemoryMerk::new();
        for i in 0u8..10 {
            merk.put(vec![i], vec![i + 100]).unwrap();
        }
        let root = merk.checkpoint().into_root().unwrap();

        let source = RecordingSource::new();
        let walker = Walker::new(root, source.clone());

        let result = Walker::delete_range_apply_to(Some(walker), &[3u8], &[7u8]);
        assert!(result.is_ok());

        let visits = source.into_visits().unwrap();
        assert!(
            visits.len() >= 4,
            "delete-range should visit at least the split/join path nodes"
        );
    }

    #[test]
    fn panic_source_record_visit_is_noop() {
        let source = PanicSource {};
        let node = Node::new(b"test".to_vec(), b"val".to_vec()).unwrap();
        source.record_visit(&node);
    }

    #[test]
    fn get_recording_records_descent_path() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"c", b"3"),
            (b"e", b"5"),
            (b"g", b"7"),
            (b"i", b"9"),
        ]);

        let mut visited = Vec::new();
        let result = get_recording(&root, b"e", &mut |n: &Node| {
            visited.push(n.key().to_vec());
        })
        .unwrap();

        assert_eq!(result, Some(b"5".to_vec()));
        assert!(!visited.is_empty(), "should have visited at least one node");
        assert!(
            visited.contains(&b"e".to_vec()),
            "visited nodes should include the target"
        );
        assert_eq!(
            *visited.first().unwrap(),
            root.key().to_vec(),
            "first visited should be the root"
        );
    }

    #[test]
    fn get_recording_absent_key_records_path() {
        let root = build_tree(&[(b"b", b"2"), (b"d", b"4"), (b"f", b"6")]);

        let mut visited = Vec::new();
        let result = get_recording(&root, b"c", &mut |n: &Node| {
            visited.push(n.key().to_vec());
        })
        .unwrap();

        assert_eq!(result, None);
        assert!(
            !visited.is_empty(),
            "absent-key lookup should still record the path"
        );
    }

    #[test]
    fn range_recording_records_routing_and_result_nodes() {
        let root = build_tree(&[
            (b"a", b"1"),
            (b"c", b"3"),
            (b"e", b"5"),
            (b"g", b"7"),
            (b"i", b"9"),
        ]);

        let mut visited = Vec::new();
        let results = range_recording(&root, b"c", Some(b"h"), &mut |n: &Node| {
            visited.push(n.key().to_vec());
        })
        .unwrap();

        let result_keys: Vec<&[u8]> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(result_keys.contains(&b"c".as_slice()));
        assert!(result_keys.contains(&b"e".as_slice()));
        assert!(result_keys.contains(&b"g".as_slice()));
        assert!(!result_keys.contains(&b"a".as_slice()));
        assert!(!result_keys.contains(&b"i".as_slice()));

        assert!(
            visited.len() >= results.len(),
            "visited count should be >= result count (routing nodes)"
        );
        assert!(
            visited.contains(&root.key().to_vec()),
            "root should be visited"
        );
    }

    #[test]
    fn range_recording_unbounded_end() {
        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);

        let mut visited = Vec::new();
        let results = range_recording(&root, b"c", None, &mut |n: &Node| {
            visited.push(n.key().to_vec());
        })
        .unwrap();

        let result_keys: Vec<&[u8]> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(result_keys.contains(&b"c".as_slice()));
        assert!(result_keys.contains(&b"e".as_slice()));
        assert!(!result_keys.contains(&b"a".as_slice()));
    }

    #[test]
    fn prefix_recording_delegates_to_range() {
        let root = build_tree(&[
            (b"pre_a", b"1"),
            (b"pre_b", b"2"),
            (b"pre_c", b"3"),
            (b"xyz", b"4"),
        ]);

        let mut visited = Vec::new();
        let results = prefix_recording(&root, b"pre_", &mut |n: &Node| {
            visited.push(n.key().to_vec());
        })
        .unwrap();

        let result_keys: Vec<&[u8]> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(result_keys.contains(&b"pre_a".as_slice()));
        assert!(result_keys.contains(&b"pre_b".as_slice()));
        assert!(result_keys.contains(&b"pre_c".as_slice()));
        assert!(!result_keys.contains(&b"xyz".as_slice()));

        assert!(!visited.is_empty());
    }

    #[test]
    fn get_recording_errors_on_pruned() {
        let tree = Node::from_fields(
            b"root".to_vec(),
            b"val".to_vec(),
            Default::default(),
            Some(Child::pruned(b"left".to_vec(), [77; 32], (0, 0))),
            None,
        );

        let result = get_recording(&tree, b"left", &mut |_: &Node| {});
        assert!(matches!(result, Err(Error::PrunedNode(_))));
    }

    #[test]
    fn range_recording_errors_on_pruned_in_range() {
        let tree = Node::from_fields(
            b"m".to_vec(),
            b"val".to_vec(),
            Default::default(),
            Some(Child::pruned(b"a".to_vec(), [77; 32], (0, 0))),
            None,
        );

        let result = range_recording(&tree, b"a", Some(b"z"), &mut |_: &Node| {});
        assert!(matches!(result, Err(Error::PrunedNode(_))));
    }
}
