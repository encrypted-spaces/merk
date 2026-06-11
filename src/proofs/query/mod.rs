mod map;

use {super::Op, std::collections::LinkedList};

use super::proof_tree::execute;
use super::{Decoder, Node};
use crate::avl::child::Child;
use crate::avl::node::Node as TreeNode;
use crate::avl::walker::RefWalker;
use crate::error::{Error, Result};
use crate::hash::Hash;
use std::cmp::{max, min, Ordering};
use std::collections::BTreeSet;
use std::ops::RangeInclusive;

pub use map::*;

/// `Query` represents one or more keys or ranges of keys, which can be used to
/// resolve a proof which will include all of the requested values.
#[derive(Default)]
pub struct Query {
    items: BTreeSet<QueryItem>,
}

impl Query {
    /// Creates a new query which contains no items.
    pub fn new() -> Self {
        Default::default()
    }

    /// Adds an individual key to the query, so that its value (or its absence)
    /// in the tree will be included in the resulting proof.
    ///
    /// If the key or a range including the key already exists in the query,
    /// this will have no effect. If the query already includes a range that has
    /// a non-inclusive bound equal to the key, the bound will be changed to be
    /// inclusive.
    pub fn insert_key(&mut self, key: Vec<u8>) {
        let key = QueryItem::Key(key);
        self.items.insert(key);
    }

    /// Adds a range to the query, so that all the entries in the tree with keys
    /// in the range will be included in the resulting proof.
    ///
    /// If a range including the range already exists in the query, this will
    /// have no effect. If the query already includes a range that overlaps with
    /// the range, the ranges will be joined together.
    pub fn insert_range(&mut self, range: std::ops::Range<Vec<u8>>) {
        let range = QueryItem::Range(range);
        self.insert_item(range);
    }

    /// Adds an inclusive range to the query, so that all the entries in the
    /// tree with keys in the range will be included in the resulting proof.
    ///
    /// If a range including the range already exists in the query, this will
    /// have no effect. If the query already includes a range that overlaps with
    /// the range, the ranges will be merged together.
    pub fn insert_range_inclusive(&mut self, range: RangeInclusive<Vec<u8>>) {
        let range = QueryItem::RangeInclusive(range);
        self.insert_item(range);
    }

    /// Adds the `QueryItem` to the query, first checking to see if it collides
    /// with any existing ranges or keys. All colliding items will be removed
    /// then merged together so that the query includes the minimum number of
    /// items (with no items covering any duplicate parts of keyspace) while
    /// still including every key or range that has been added to the query.
    pub fn insert_item(&mut self, mut item: QueryItem) {
        // since `QueryItem::eq` considers items equal if they collide at all
        // (including keys within ranges or ranges which partially overlap),
        // `items.take` will remove the first item which collides
        while let Some(existing) = self.items.take(&item) {
            item = item.merge(existing);
        }

        self.items.insert(item);
    }

    pub fn iter(&self) -> impl Iterator<Item = &QueryItem> {
        self.items.iter()
    }
}

impl<Q: Into<QueryItem>> From<Vec<Q>> for Query {
    fn from(other: Vec<Q>) -> Self {
        let items = other.into_iter().map(Into::into).collect();
        Query { items }
    }
}

impl From<Query> for Vec<QueryItem> {
    fn from(q: Query) -> Vec<QueryItem> {
        q.into_iter().collect()
    }
}

impl IntoIterator for Query {
    type Item = QueryItem;
    type IntoIter = <BTreeSet<QueryItem> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

/// A `QueryItem` represents a key or range of keys to be included in a proof.
#[derive(Clone, Debug)]
pub enum QueryItem {
    Key(Vec<u8>),
    Range(std::ops::Range<Vec<u8>>),
    RangeInclusive(RangeInclusive<Vec<u8>>),
}

impl QueryItem {
    pub fn lower_bound(&self) -> &[u8] {
        match self {
            QueryItem::Key(key) => key.as_slice(),
            QueryItem::Range(range) => range.start.as_ref(),
            QueryItem::RangeInclusive(range) => range.start().as_ref(),
        }
    }

    pub fn upper_bound(&self) -> (&[u8], bool) {
        match self {
            QueryItem::Key(key) => (key.as_slice(), true),
            QueryItem::Range(range) => (range.end.as_ref(), false),
            QueryItem::RangeInclusive(range) => (range.end().as_ref(), true),
        }
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        let (bound, inclusive) = self.upper_bound();
        key >= self.lower_bound() && (key < bound || (key == bound && inclusive))
    }

    fn merge(self, other: QueryItem) -> QueryItem {
        // TODO: don't copy into new vecs
        let start = min(self.lower_bound(), other.lower_bound()).to_vec();
        let end = max(self.upper_bound(), other.upper_bound());
        if end.1 {
            QueryItem::RangeInclusive(RangeInclusive::new(start, end.0.to_vec()))
        } else {
            QueryItem::Range(std::ops::Range {
                start,
                end: end.0.to_vec(),
            })
        }
    }
}

impl PartialEq for QueryItem {
    fn eq(&self, other: &QueryItem) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl PartialEq<&[u8]> for QueryItem {
    fn eq(&self, other: &&[u8]) -> bool {
        matches!(self.partial_cmp(other), Some(Ordering::Equal))
    }
}

impl Eq for QueryItem {}

impl Ord for QueryItem {
    fn cmp(&self, other: &QueryItem) -> Ordering {
        let cmp_lu = self.lower_bound().cmp(other.upper_bound().0);
        let cmp_ul = self.upper_bound().0.cmp(other.lower_bound());
        let self_inclusive = self.upper_bound().1;
        let other_inclusive = other.upper_bound().1;

        match (cmp_lu, cmp_ul) {
            (Ordering::Less, Ordering::Less) => Ordering::Less,
            (Ordering::Less, Ordering::Equal) => match self_inclusive {
                true => Ordering::Equal,
                false => Ordering::Less,
            },
            (Ordering::Less, Ordering::Greater) => Ordering::Equal,
            (Ordering::Equal, _) => match other_inclusive {
                true => Ordering::Equal,
                false => Ordering::Greater,
            },
            (Ordering::Greater, _) => Ordering::Greater,
        }
    }
}

impl PartialOrd for QueryItem {
    fn partial_cmp(&self, other: &QueryItem) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd<&[u8]> for QueryItem {
    fn partial_cmp(&self, other: &&[u8]) -> Option<Ordering> {
        let other = QueryItem::Key(other.to_vec());
        Some(self.cmp(&other))
    }
}

impl From<Vec<u8>> for QueryItem {
    fn from(key: Vec<u8>) -> Self {
        QueryItem::Key(key)
    }
}

impl Child {
    /// Creates a `Node::NodeHash` from this child. Panics if the child's hash has
    /// not yet been computed (modified resident child).
    fn to_hash_node(&self) -> Node {
        Node::NodeHash(*self.hash())
    }
}

impl<'a> RefWalker<'a> {
    /// Creates a `Node::KV` from the key/value pair of the root node.
    pub(crate) fn to_kv_node(&self) -> Node {
        Node::KV(self.tree().key().to_vec(), self.tree().value().to_vec())
    }

    /// Creates a `Node::KVHash` from the hash of the key/value pair of the root
    /// node.
    pub(crate) fn to_kvhash_node(&self) -> Node {
        Node::KVHash(*self.tree().kv_hash())
    }

    pub(crate) fn create_proof(
        &self,
        query: &[QueryItem],
    ) -> Result<(LinkedList<Op>, (bool, bool))> {
        let node_key = QueryItem::Key(self.tree().key().to_vec());
        let search = query.binary_search_by(|key| key.cmp(&node_key));

        let (left_items, right_items) = match search {
            Ok(index) => {
                let item = &query[index];
                let left_bound = item.lower_bound();
                let right_bound = item.upper_bound().0;

                let left_query = if left_bound < self.tree().key() {
                    &query[..=index]
                } else {
                    &query[..index]
                };

                let right_query = if right_bound > self.tree().key() {
                    &query[index..]
                } else {
                    &query[index + 1..]
                };

                (left_query, right_query)
            }
            Err(index) => (&query[..index], &query[index..]),
        };

        let (mut proof, left_absence) = self.create_child_proof(true, left_items)?;
        let (mut right_proof, right_absence) = self.create_child_proof(false, right_items)?;

        let (has_left, has_right) = (!proof.is_empty(), !right_proof.is_empty());

        proof.push_back(match search {
            Ok(_) => Op::Push(self.to_kv_node()),
            Err(_) => {
                if left_absence.1 || right_absence.0 {
                    Op::Push(self.to_kv_node())
                } else {
                    Op::Push(self.to_kvhash_node())
                }
            }
        });

        if has_left {
            proof.push_back(Op::Parent);
        }

        if has_right {
            proof.append(&mut right_proof);
            proof.push_back(Op::Child);
        }

        Ok((proof, (left_absence.0, right_absence.1)))
    }

    fn create_child_proof(
        &self,
        left: bool,
        query: &[QueryItem],
    ) -> Result<(LinkedList<Op>, (bool, bool))> {
        Ok(if !query.is_empty() {
            if let Some(child) = self.walk(left) {
                child.create_proof(query)?
            } else {
                (LinkedList::new(), (true, true))
            }
        } else if let Some(child) = self.tree().child_ref(left) {
            let mut proof = LinkedList::new();
            proof.push_back(Op::Push(child.to_hash_node()));
            (proof, (false, false))
        } else {
            (LinkedList::new(), (false, false))
        })
    }
}

/// Verifies the encoded proof against the expected root hash.
///
/// Returns a verified partial view of the proof contents. Callers can query the
/// returned `Map` with `get` or `range`; those accessors return `Err` if the
/// proof does not contain enough data to prove the requested key or range.
pub fn verify(bytes: &[u8], expected_hash: Hash) -> Result<Map> {
    let ops = Decoder::new(bytes);
    let mut map_builder = MapBuilder::new();

    let root = execute(ops, true, |node| map_builder.insert(node))?;

    if root.hash()? != expected_hash {
        return Err(Error::HashMismatch(expected_hash, root.hash()?));
    }

    Ok(map_builder.build())
}

/// Verifies the encoded proof with the given query and expected root hash.
///
/// Every requested key or range is checked against the proof. Missing keys that
/// are proven absent are omitted from the returned values. If the proof does
/// not contain enough data to prove a requested key or range, `Err` is
/// returned.
pub fn verify_query(
    bytes: &[u8],
    query: &Query,
    expected_hash: Hash,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut output = Vec::with_capacity(query.items.len());
    let mut last_push = None;
    let mut query = query.items.iter().peekable();
    let mut in_range = false;

    let ops = Decoder::new(bytes);

    let root = execute(ops, true, |node| {
        if let Node::KV(key, value) = node {
            while let Some(item) = query.peek() {
                let query_item = *item;

                if *query_item > key.as_slice() {
                    break;
                }

                if !in_range {
                    match last_push {
                        _ if key == query_item.lower_bound() => {}
                        None => {}
                        Some(Node::KV(_, _)) => {}
                        Some(_) => {
                            return Err(Error::Bound(
                                "Cannot verify lower bound of queried range".into(),
                            ));
                        }
                    }
                }

                if key.as_slice() >= query_item.upper_bound().0 {
                    query.next();
                    in_range = false;
                } else {
                    in_range = true;
                }

                if query_item.contains(key) {
                    output.push((key.clone(), value.clone()));
                    break;
                }
            }
        } else if in_range {
            return Err(Error::MissingData);
        }

        last_push = Some(node.clone());

        Ok(())
    })?;

    if query.peek().is_some() {
        match last_push {
            Some(Node::KV(_, _)) => {}
            _ => {
                return Err(Error::MissingData);
            }
        }
    }

    if root.hash()? != expected_hash {
        return Err(Error::HashMismatch(expected_hash, root.hash()?));
    }

    Ok(output)
}

pub fn prove_resident<Q, I>(maybe_tree: Option<&TreeNode>, query: I) -> Result<Vec<u8>>
where
    Q: Into<QueryItem>,
    I: IntoIterator<Item = Q>,
{
    let query_vec: Vec<QueryItem> = query.into_iter().map(Into::into).collect();
    let tree =
        maybe_tree.ok_or_else(|| Error::Proof("Cannot create proof for empty tree".into()))?;
    let walker = RefWalker::new(tree);
    let (proof, _) = walker.create_proof(query_vec.as_slice())?;
    let mut bytes = Vec::with_capacity(128);
    super::encode_into(proof.iter(), &mut bytes);
    Ok(bytes)
}

#[cfg(test)]
mod test {
    use super::super::encoding::encode_into;
    use super::super::*;
    use super::*;
    use crate::avl::node::Node as TreeNode;
    use crate::avl::walker::RefWalker;
    use crate::test_utils::make_tree_seq;
    use ed::Encode;

    fn make_3_node_tree() -> Result<TreeNode> {
        let mut tree = TreeNode::new(vec![5], vec![5])?
            .attach(true, Some(TreeNode::new(vec![3], vec![3])?))
            .attach(false, Some(TreeNode::new(vec![7], vec![7])?));
        tree.commit();
        Ok(tree)
    }

    fn assert_verified_values(
        bytes: &[u8],
        expected_hash: Hash,
        expected: &[(Vec<u8>, Option<Vec<u8>>)],
    ) -> Result<()> {
        let mut query = Query::new();
        for (key, _) in expected {
            query.insert_key(key.clone());
        }

        let result = verify_query(bytes, &query, expected_hash)?;
        let mut values = std::collections::HashMap::new();
        for (key, value) in result {
            assert!(values.insert(key, value).is_none());
        }

        for (key, expected_value) in expected {
            assert_eq!(values.get(key), expected_value.as_ref());
        }
        Ok(())
    }

    fn encode_query_proof(tree: &TreeNode, queryitems: &[QueryItem]) -> Vec<u8> {
        let walker = RefWalker::new(tree);
        let (proof, _) = walker
            .create_proof(queryitems)
            .expect("create_proof errored");
        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        bytes
    }

    fn query_from_items(queryitems: &[QueryItem]) -> Query {
        let mut query = Query::new();
        for item in queryitems {
            query.insert_item(item.clone());
        }
        query
    }

    fn assert_verified_query_values(
        bytes: &[u8],
        expected_hash: Hash,
        queryitems: &[QueryItem],
        expected: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<()> {
        let query = query_from_items(queryitems);
        let result = verify_query(bytes, &query, expected_hash)?;
        assert_eq!(result, expected);
        Ok(())
    }

    fn seq_key(key: u64) -> Vec<u8> {
        key.to_be_bytes().to_vec()
    }

    fn verify_keys_test(keys: Vec<Vec<u8>>, expected_result: Vec<Option<Vec<u8>>>) -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let (proof, _) = walker
            .create_proof(
                keys.clone()
                    .into_iter()
                    .map(QueryItem::Key)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .expect("failed to create proof");
        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);

        let expected_hash = [
            26, 197, 74, 4, 89, 161, 76, 41, 37, 166, 197, 214, 68, 53, 6, 228, 91, 221, 131, 222,
            175, 210, 106, 172, 78, 117, 48, 59, 209, 195, 53, 80,
        ];

        let expected: Vec<_> = keys.into_iter().zip(expected_result).collect();
        assert_verified_values(bytes.as_slice(), expected_hash, &expected)
    }

    #[test]
    fn verify_query_keys() -> Result<()> {
        let tree = make_3_node_tree()?;
        let queryitems = vec![QueryItem::Key(vec![3]), QueryItem::Key(vec![7])];
        let bytes = encode_query_proof(&tree, queryitems.as_slice());
        let query = query_from_items(queryitems.as_slice());

        let result = verify_query(bytes.as_slice(), &query, tree.hash())?;

        assert_eq!(result, vec![(vec![3], vec![3]), (vec![7], vec![7])]);
        Ok(())
    }

    #[test]
    fn verify_query_proven_absence_returns_no_value() -> Result<()> {
        let tree = make_3_node_tree()?;
        let queryitems = vec![QueryItem::Key(vec![6])];
        let bytes = encode_query_proof(&tree, queryitems.as_slice());
        let query = query_from_items(queryitems.as_slice());

        let result = verify_query(bytes.as_slice(), &query, tree.hash())?;

        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn verify_query_range() -> Result<()> {
        let tree = make_tree_seq(10);
        let queryitems = vec![QueryItem::Range(seq_key(5)..seq_key(7))];
        let bytes = encode_query_proof(&tree, queryitems.as_slice());
        let query = query_from_items(queryitems.as_slice());

        let result = verify_query(bytes.as_slice(), &query, tree.hash())?;

        assert_eq!(
            result,
            vec![(seq_key(5), vec![123; 60]), (seq_key(6), vec![123; 60])]
        );
        Ok(())
    }

    #[test]
    fn verify_query_errors_when_proof_does_not_cover_query() -> Result<()> {
        let tree = make_3_node_tree()?;
        let bytes = encode_query_proof(&tree, &[QueryItem::Key(vec![5])]);
        let query = query_from_items(&[QueryItem::Key(vec![7])]);

        let result = verify_query(bytes.as_slice(), &query, tree.hash());

        assert!(matches!(result, Err(Error::MissingData)));
        Ok(())
    }

    #[test]
    fn root_verify() -> Result<()> {
        verify_keys_test(vec![vec![5]], vec![Some(vec![5])])
    }

    #[test]
    fn single_verify() -> Result<()> {
        verify_keys_test(vec![vec![3]], vec![Some(vec![3])])
    }

    #[test]
    fn double_verify() -> Result<()> {
        verify_keys_test(vec![vec![3], vec![5]], vec![Some(vec![3]), Some(vec![5])])
    }

    #[test]
    fn double_verify_2() -> Result<()> {
        verify_keys_test(vec![vec![3], vec![7]], vec![Some(vec![3]), Some(vec![7])])
    }

    #[test]
    fn triple_verify() -> Result<()> {
        verify_keys_test(
            vec![vec![3], vec![5], vec![7]],
            vec![Some(vec![3]), Some(vec![5]), Some(vec![7])],
        )
    }

    #[test]
    fn left_edge_absence_verify() -> Result<()> {
        verify_keys_test(vec![vec![2]], vec![None])
    }

    #[test]
    fn right_edge_absence_verify() -> Result<()> {
        verify_keys_test(vec![vec![8]], vec![None])
    }

    #[test]
    fn inner_absence_verify() -> Result<()> {
        verify_keys_test(vec![vec![6]], vec![None])
    }

    #[test]
    fn absent_and_present_verify() -> Result<()> {
        verify_keys_test(vec![vec![5], vec![6]], vec![Some(vec![5]), None])
    }

    #[test]
    fn empty_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let (proof, absence) = walker
            .create_proof(vec![].as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                58, 114, 115, 81, 9, 53, 210, 78, 220, 23, 100, 136, 26, 152, 18, 112, 219, 217,
                48, 154, 227, 92, 100, 5, 35, 234, 194, 199, 112, 53, 188, 16
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                233, 192, 89, 232, 161, 148, 62, 69, 165, 24, 156, 25, 44, 34, 166, 17, 115, 252,
                75, 105, 14, 252, 228, 57, 78, 44, 230, 91, 45, 221, 40, 10
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                164, 34, 118, 190, 242, 9, 177, 28, 85, 32, 6, 231, 131, 61, 119, 22, 48, 13, 117,
                231, 1, 18, 220, 48, 102, 122, 204, 22, 126, 24, 36, 220
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        verify_query(bytes.as_slice(), &Query::new(), tree.hash()).unwrap();
        Ok(())
    }

    #[test]
    fn root_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Key(vec![5])];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                58, 114, 115, 81, 9, 53, 210, 78, 220, 23, 100, 136, 26, 152, 18, 112, 219, 217,
                48, 154, 227, 92, 100, 5, 35, 234, 194, 199, 112, 53, 188, 16
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![5], vec![5]))));
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                164, 34, 118, 190, 242, 9, 177, 28, 85, 32, 6, 231, 131, 61, 119, 22, 48, 13, 117,
                231, 1, 18, 220, 48, 102, 122, 204, 22, 126, 24, 36, 220
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![(vec![5], vec![5])],
        )?;
        assert_verified_values(bytes.as_slice(), tree.hash(), &[(vec![5], Some(vec![5]))])?;
        Ok(())
    }

    #[test]
    fn leaf_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Key(vec![3])];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![3], vec![3]))));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                233, 192, 89, 232, 161, 148, 62, 69, 165, 24, 156, 25, 44, 34, 166, 17, 115, 252,
                75, 105, 14, 252, 228, 57, 78, 44, 230, 91, 45, 221, 40, 10
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                164, 34, 118, 190, 242, 9, 177, 28, 85, 32, 6, 231, 131, 61, 119, 22, 48, 13, 117,
                231, 1, 18, 220, 48, 102, 122, 204, 22, 126, 24, 36, 220
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![(vec![3], vec![3])],
        )?;
        assert_verified_values(bytes.as_slice(), tree.hash(), &[(vec![3], Some(vec![3]))])?;
        Ok(())
    }

    #[test]
    fn double_leaf_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Key(vec![3]), QueryItem::Key(vec![7])];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![3], vec![3]))));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                233, 192, 89, 232, 161, 148, 62, 69, 165, 24, 156, 25, 44, 34, 166, 17, 115, 252,
                75, 105, 14, 252, 228, 57, 78, 44, 230, 91, 45, 221, 40, 10
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![7], vec![7]))));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![(vec![3], vec![3]), (vec![7], vec![7])],
        )?;
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[(vec![3], Some(vec![3])), (vec![7], Some(vec![7]))],
        )?;
        Ok(())
    }

    #[test]
    fn all_nodes_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let queryitems = vec![
            QueryItem::Key(vec![3]),
            QueryItem::Key(vec![5]),
            QueryItem::Key(vec![7]),
        ];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![3], vec![3]))));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![5], vec![5]))));
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![7], vec![7]))));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![(vec![3], vec![3]), (vec![5], vec![5]), (vec![7], vec![7])],
        )?;
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[
                (vec![3], Some(vec![3])),
                (vec![5], Some(vec![5])),
                (vec![7], Some(vec![7])),
            ],
        )?;
        Ok(())
    }

    #[test]
    fn global_edge_absence_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Key(vec![8])];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                58, 114, 115, 81, 9, 53, 210, 78, 220, 23, 100, 136, 26, 152, 18, 112, 219, 217,
                48, 154, 227, 92, 100, 5, 35, 234, 194, 199, 112, 53, 188, 16
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                233, 192, 89, 232, 161, 148, 62, 69, 165, 24, 156, 25, 44, 34, 166, 17, 115, 252,
                75, 105, 14, 252, 228, 57, 78, 44, 230, 91, 45, 221, 40, 10
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![7], vec![7]))));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, true));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(bytes.as_slice(), tree.hash(), queryitems.as_slice(), vec![])?;
        assert_verified_values(bytes.as_slice(), tree.hash(), &[(vec![8], None)])?;
        Ok(())
    }

    #[test]
    fn absence_proof() -> Result<()> {
        let tree = make_3_node_tree()?;
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Key(vec![6])];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                58, 114, 115, 81, 9, 53, 210, 78, 220, 23, 100, 136, 26, 152, 18, 112, 219, 217,
                48, 154, 227, 92, 100, 5, 35, 234, 194, 199, 112, 53, 188, 16
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![5], vec![5]))));
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![7], vec![7]))));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(bytes.as_slice(), tree.hash(), queryitems.as_slice(), vec![])?;
        assert_verified_values(bytes.as_slice(), tree.hash(), &[(vec![6], None)])?;
        Ok(())
    }

    #[test]
    fn doc_proof() -> Result<()> {
        let mut tree = TreeNode::new(vec![5], vec![5])?
            .attach(
                true,
                Some(
                    TreeNode::new(vec![2], vec![2])?
                        .attach(true, Some(TreeNode::new(vec![1], vec![1])?))
                        .attach(
                            false,
                            Some(
                                TreeNode::new(vec![4], vec![4])?
                                    .attach(true, Some(TreeNode::new(vec![3], vec![3])?)),
                            ),
                        ),
                ),
            )
            .attach(
                false,
                Some(
                    TreeNode::new(vec![9], vec![9])?
                        .attach(
                            true,
                            Some(
                                TreeNode::new(vec![7], vec![7])?
                                    .attach(true, Some(TreeNode::new(vec![6], vec![6])?))
                                    .attach(false, Some(TreeNode::new(vec![8], vec![8])?)),
                            ),
                        )
                        .attach(
                            false,
                            Some(
                                TreeNode::new(vec![11], vec![11])?
                                    .attach(true, Some(TreeNode::new(vec![10], vec![10])?)),
                            ),
                        ),
                ),
            );
        tree.commit();

        let walker = RefWalker::new(&tree);

        let queryitems = vec![
            QueryItem::Key(vec![1]),
            QueryItem::Key(vec![2]),
            QueryItem::Key(vec![3]),
            QueryItem::Key(vec![4]),
        ];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![1], vec![1]))));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![2], vec![2]))));
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![3], vec![3]))));
        assert_eq!(iter.next(), Some(&Op::Push(Node::KV(vec![4], vec![4]))));
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                233, 192, 89, 232, 161, 148, 62, 69, 165, 24, 156, 25, 44, 34, 166, 17, 115, 252,
                75, 105, 14, 252, 228, 57, 78, 44, 230, 91, 45, 221, 40, 10
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                185, 150, 80, 49, 7, 210, 18, 118, 201, 115, 155, 49, 189, 239, 64, 11, 31, 167,
                74, 249, 84, 113, 192, 169, 137, 13, 151, 119, 106, 146, 122, 145
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_eq!(
            bytes,
            vec![
                3, 0, 1, 1, 0, 1, 1, 3, 0, 1, 2, 0, 1, 2, 16, 3, 0, 1, 3, 0, 1, 3, 3, 0, 1, 4, 0,
                1, 4, 16, 17, 2, 233, 192, 89, 232, 161, 148, 62, 69, 165, 24, 156, 25, 44, 34,
                166, 17, 115, 252, 75, 105, 14, 252, 228, 57, 78, 44, 230, 91, 45, 221, 40, 10, 16,
                1, 185, 150, 80, 49, 7, 210, 18, 118, 201, 115, 155, 49, 189, 239, 64, 11, 31, 167,
                74, 249, 84, 113, 192, 169, 137, 13, 151, 119, 106, 146, 122, 145, 17
            ]
        );

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![
                (vec![1], vec![1]),
                (vec![2], vec![2]),
                (vec![3], vec![3]),
                (vec![4], vec![4]),
            ],
        )?;
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[
                (vec![1], Some(vec![1])),
                (vec![2], Some(vec![2])),
                (vec![3], Some(vec![3])),
                (vec![4], Some(vec![4])),
            ],
        )?;
        Ok(())
    }

    #[test]
    fn query_item_cmp() {
        assert!(QueryItem::Key(vec![10]) < QueryItem::Key(vec![20]));
        assert!(QueryItem::Key(vec![10]) == QueryItem::Key(vec![10]));
        assert!(QueryItem::Key(vec![20]) > QueryItem::Key(vec![10]));

        assert!(QueryItem::Key(vec![10]) < QueryItem::Range(vec![20]..vec![30]));
        assert!(QueryItem::Key(vec![10]) == QueryItem::Range(vec![10]..vec![20]));
        assert!(QueryItem::Key(vec![15]) == QueryItem::Range(vec![10]..vec![20]));
        assert!(QueryItem::Key(vec![20]) > QueryItem::Range(vec![10]..vec![20]));
        assert!(QueryItem::Key(vec![20]) == QueryItem::RangeInclusive(vec![10]..=vec![20]));
        assert!(QueryItem::Key(vec![30]) > QueryItem::Range(vec![10]..vec![20]));

        assert!(QueryItem::Range(vec![10]..vec![20]) < QueryItem::Range(vec![30]..vec![40]));
        assert!(QueryItem::Range(vec![10]..vec![20]) < QueryItem::Range(vec![20]..vec![30]));
        assert!(
            QueryItem::RangeInclusive(vec![10]..=vec![20]) == QueryItem::Range(vec![20]..vec![30])
        );
        assert!(QueryItem::Range(vec![15]..vec![25]) == QueryItem::Range(vec![20]..vec![30]));
        assert!(QueryItem::Range(vec![20]..vec![30]) > QueryItem::Range(vec![10]..vec![20]));
    }

    #[test]
    fn query_item_merge() {
        let mine = QueryItem::Range(vec![10]..vec![30]);
        let other = QueryItem::Range(vec![15]..vec![20]);
        assert_eq!(mine.merge(other), QueryItem::Range(vec![10]..vec![30]));

        let mine = QueryItem::RangeInclusive(vec![10]..=vec![30]);
        let other = QueryItem::Range(vec![20]..vec![30]);
        assert_eq!(
            mine.merge(other),
            QueryItem::RangeInclusive(vec![10]..=vec![30])
        );

        let mine = QueryItem::Key(vec![5]);
        let other = QueryItem::Range(vec![1]..vec![10]);
        assert_eq!(mine.merge(other), QueryItem::Range(vec![1]..vec![10]));

        let mine = QueryItem::Key(vec![10]);
        let other = QueryItem::RangeInclusive(vec![1]..=vec![10]);
        assert_eq!(
            mine.merge(other),
            QueryItem::RangeInclusive(vec![1]..=vec![10])
        );
    }

    #[test]
    fn query_insert() {
        let mut query = Query::new();
        query.insert_key(vec![2]);
        query.insert_range(vec![3]..vec![5]);
        query.insert_range_inclusive(vec![5]..=vec![7]);
        query.insert_range(vec![4]..vec![6]);
        query.insert_key(vec![5]);

        let mut iter = query.items.iter();
        assert_eq!(format!("{:?}", iter.next()), "Some(Key([2]))");
        assert_eq!(
            format!("{:?}", iter.next()),
            "Some(RangeInclusive([3]..=[7]))"
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn range_proof() {
        let tree = make_tree_seq(10);
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Range(
            vec![0, 0, 0, 0, 0, 0, 0, 5]..vec![0, 0, 0, 0, 0, 0, 0, 7],
        )];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                225, 152, 183, 62, 249, 37, 189, 25, 150, 135, 171, 38, 135, 131, 58, 233, 61, 89,
                243, 198, 60, 200, 234, 170, 190, 197, 90, 25, 52, 189, 2, 65
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                49, 10, 177, 169, 208, 39, 204, 220, 74, 172, 172, 196, 88, 59, 201, 97, 108, 135,
                191, 148, 214, 89, 83, 140, 208, 113, 80, 137, 151, 104, 220, 138
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                218, 57, 4, 102, 52, 65, 175, 249, 60, 207, 219, 176, 186, 249, 46, 163, 9, 9, 237,
                172, 69, 141, 86, 21, 26, 3, 209, 62, 14, 211, 217, 23
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 5],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 6],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 7],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                114, 190, 130, 81, 143, 185, 12, 157, 71, 145, 188, 22, 103, 48, 177, 157, 19, 101,
                136, 144, 109, 124, 95, 88, 228, 213, 5, 149, 226, 7, 106, 2
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![
                (vec![0, 0, 0, 0, 0, 0, 0, 5], vec![123; 60]),
                (vec![0, 0, 0, 0, 0, 0, 0, 6], vec![123; 60]),
            ],
        )
        .unwrap();
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[
                (vec![0, 0, 0, 0, 0, 0, 0, 5], Some(vec![123; 60])),
                (vec![0, 0, 0, 0, 0, 0, 0, 6], Some(vec![123; 60])),
            ],
        )
        .unwrap();
    }

    #[test]
    fn range_proof_inclusive() {
        let tree = make_tree_seq(10);
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::RangeInclusive(
            vec![0, 0, 0, 0, 0, 0, 0, 5]..=vec![0, 0, 0, 0, 0, 0, 0, 7],
        )];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                225, 152, 183, 62, 249, 37, 189, 25, 150, 135, 171, 38, 135, 131, 58, 233, 61, 89,
                243, 198, 60, 200, 234, 170, 190, 197, 90, 25, 52, 189, 2, 65
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                49, 10, 177, 169, 208, 39, 204, 220, 74, 172, 172, 196, 88, 59, 201, 97, 108, 135,
                191, 148, 214, 89, 83, 140, 208, 113, 80, 137, 151, 104, 220, 138
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                218, 57, 4, 102, 52, 65, 175, 249, 60, 207, 219, 176, 186, 249, 46, 163, 9, 9, 237,
                172, 69, 141, 86, 21, 26, 3, 209, 62, 14, 211, 217, 23
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 5],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 6],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 7],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                114, 190, 130, 81, 143, 185, 12, 157, 71, 145, 188, 22, 103, 48, 177, 157, 19, 101,
                136, 144, 109, 124, 95, 88, 228, 213, 5, 149, 226, 7, 106, 2
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![
                (vec![0, 0, 0, 0, 0, 0, 0, 5], vec![123; 60]),
                (vec![0, 0, 0, 0, 0, 0, 0, 6], vec![123; 60]),
                (vec![0, 0, 0, 0, 0, 0, 0, 7], vec![123; 60]),
            ],
        )
        .unwrap();
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[
                (vec![0, 0, 0, 0, 0, 0, 0, 5], Some(vec![123; 60])),
                (vec![0, 0, 0, 0, 0, 0, 0, 6], Some(vec![123; 60])),
                (vec![0, 0, 0, 0, 0, 0, 0, 7], Some(vec![123; 60])),
            ],
        )
        .unwrap();
    }

    #[test]
    fn range_proof_missing_upper_bound() {
        let tree = make_tree_seq(10);
        let walker = RefWalker::new(&tree);

        let queryitems = vec![QueryItem::Range(
            vec![0, 0, 0, 0, 0, 0, 0, 5]..vec![0, 0, 0, 0, 0, 0, 0, 6, 5],
        )];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                225, 152, 183, 62, 249, 37, 189, 25, 150, 135, 171, 38, 135, 131, 58, 233, 61, 89,
                243, 198, 60, 200, 234, 170, 190, 197, 90, 25, 52, 189, 2, 65
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                49, 10, 177, 169, 208, 39, 204, 220, 74, 172, 172, 196, 88, 59, 201, 97, 108, 135,
                191, 148, 214, 89, 83, 140, 208, 113, 80, 137, 151, 104, 220, 138
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                218, 57, 4, 102, 52, 65, 175, 249, 60, 207, 219, 176, 186, 249, 46, 163, 9, 9, 237,
                172, 69, 141, 86, 21, 26, 3, 209, 62, 14, 211, 217, 23
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 5],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 6],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 7],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                114, 190, 130, 81, 143, 185, 12, 157, 71, 145, 188, 22, 103, 48, 177, 157, 19, 101,
                136, 144, 109, 124, 95, 88, 228, 213, 5, 149, 226, 7, 106, 2
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![
                (vec![0, 0, 0, 0, 0, 0, 0, 5], vec![123; 60]),
                (vec![0, 0, 0, 0, 0, 0, 0, 6], vec![123; 60]),
            ],
        )
        .unwrap();
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[
                (vec![0, 0, 0, 0, 0, 0, 0, 5], Some(vec![123; 60])),
                (vec![0, 0, 0, 0, 0, 0, 0, 6], Some(vec![123; 60])),
            ],
        )
        .unwrap();
    }

    #[test]
    fn range_proof_missing_lower_bound() {
        let tree = make_tree_seq(10);
        let walker = RefWalker::new(&tree);

        let queryitems = vec![
            // 7 is not inclusive
            QueryItem::Range(vec![0, 0, 0, 0, 0, 0, 0, 5, 5]..vec![0, 0, 0, 0, 0, 0, 0, 7]),
        ];
        let (proof, absence) = walker
            .create_proof(queryitems.as_slice())
            .expect("create_proof errored");

        let mut iter = proof.iter();
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                225, 152, 183, 62, 249, 37, 189, 25, 150, 135, 171, 38, 135, 131, 58, 233, 61, 89,
                243, 198, 60, 200, 234, 170, 190, 197, 90, 25, 52, 189, 2, 65
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KVHash([
                49, 10, 177, 169, 208, 39, 204, 220, 74, 172, 172, 196, 88, 59, 201, 97, 108, 135,
                191, 148, 214, 89, 83, 140, 208, 113, 80, 137, 151, 104, 220, 138
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                218, 57, 4, 102, 52, 65, 175, 249, 60, 207, 219, 176, 186, 249, 46, 163, 9, 9, 237,
                172, 69, 141, 86, 21, 26, 3, 209, 62, 14, 211, 217, 23
            ])))
        );
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 5],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 6],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::KV(
                vec![0, 0, 0, 0, 0, 0, 0, 7],
                vec![123; 60]
            )))
        );
        assert_eq!(iter.next(), Some(&Op::Parent));
        assert_eq!(
            iter.next(),
            Some(&Op::Push(Node::NodeHash([
                114, 190, 130, 81, 143, 185, 12, 157, 71, 145, 188, 22, 103, 48, 177, 157, 19, 101,
                136, 144, 109, 124, 95, 88, 228, 213, 5, 149, 226, 7, 106, 2
            ])))
        );
        assert_eq!(iter.next(), Some(&Op::Child));
        assert_eq!(iter.next(), Some(&Op::Child));
        assert!(iter.next().is_none());
        assert_eq!(absence, (false, false));

        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);
        assert_verified_query_values(
            bytes.as_slice(),
            tree.hash(),
            queryitems.as_slice(),
            vec![(vec![0, 0, 0, 0, 0, 0, 0, 6], vec![123; 60])],
        )
        .unwrap();
        assert_verified_values(
            bytes.as_slice(),
            tree.hash(),
            &[(vec![0, 0, 0, 0, 0, 0, 0, 6], Some(vec![123; 60]))],
        )
        .unwrap();
    }

    #[test]
    fn query_from_vec() {
        let queryitems = vec![QueryItem::Range(
            vec![0, 0, 0, 0, 0, 0, 0, 5, 5]..vec![0, 0, 0, 0, 0, 0, 0, 7],
        )];
        let query = Query::from(queryitems);

        let mut expected = BTreeSet::new();
        expected.insert(QueryItem::Range(
            vec![0, 0, 0, 0, 0, 0, 0, 5, 5]..vec![0, 0, 0, 0, 0, 0, 0, 7],
        ));
        assert_eq!(query.items, expected);
    }

    #[test]
    fn query_into_vec() {
        let mut query = Query::new();
        query.insert_item(QueryItem::Range(
            vec![0, 0, 0, 0, 0, 0, 5, 5]..vec![0, 0, 0, 0, 0, 0, 0, 7],
        ));
        let query_vec: Vec<QueryItem> = query.into();
        let expected = [QueryItem::Range(
            vec![0, 0, 0, 0, 0, 0, 5, 5]..vec![0, 0, 0, 0, 0, 0, 0, 7],
        )];
        assert_eq!(
            query_vec.first().unwrap().lower_bound(),
            expected.first().unwrap().lower_bound()
        );
        assert_eq!(
            query_vec.first().unwrap().upper_bound(),
            expected.first().unwrap().upper_bound()
        );
    }

    #[test]
    fn query_item_from_vec_u8() {
        let queryitems: Vec<u8> = vec![42];
        let query = QueryItem::from(queryitems);

        let expected = QueryItem::Key(vec![42]);
        assert_eq!(query, expected);
    }

    #[test]
    fn verify_ops() -> Result<()> {
        let mut tree = TreeNode::new(vec![5], vec![5])?;
        tree.commit();

        let root_hash = tree.hash();
        let walker = RefWalker::new(&tree);

        let (proof, _) = walker
            .create_proof(vec![QueryItem::Key(vec![5])].as_slice())
            .expect("failed to create proof");
        let mut bytes = vec![];

        encode_into(proof.iter(), &mut bytes);

        let map = verify(&bytes, root_hash).unwrap();
        assert_eq!(
            map.get(vec![5].as_slice()).unwrap().unwrap(),
            vec![5].as_slice()
        );
        Ok(())
    }

    #[test]
    #[should_panic(expected = "verify failed")]
    fn verify_ops_mismatched_hash() {
        let mut tree = TreeNode::new(vec![5], vec![5]).expect("tree construction failed");
        tree.commit();

        let walker = RefWalker::new(&tree);

        let (proof, _) = walker
            .create_proof(vec![QueryItem::Key(vec![5])].as_slice())
            .expect("failed to create proof");
        let mut bytes = vec![];

        encode_into(proof.iter(), &mut bytes);

        let _map = verify(&bytes, [42; 32]).expect("verify failed");
    }

    #[test]
    #[should_panic(expected = "verify failed")]
    fn verify_query_mismatched_hash() {
        let tree = make_3_node_tree().expect("tree construction failed");
        let walker = RefWalker::new(&tree);
        let (proof, _) = walker
            .create_proof(
                vec![vec![5], vec![7]]
                    .into_iter()
                    .map(QueryItem::Key)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .expect("failed to create proof");
        let mut bytes = vec![];
        encode_into(proof.iter(), &mut bytes);

        let query = query_from_items(&[QueryItem::Key(vec![5]), QueryItem::Key(vec![7])]);
        let _result = verify_query(bytes.as_slice(), &query, [42; 32]).expect("verify failed");
    }

    #[test]
    #[should_panic(expected = "Tried to attach to NodeHash node")]
    fn hash_attach() {
        let target = make_3_node_tree().expect("tree construction failed");

        let proof = vec![
            Op::Push(Node::KV(vec![42], vec![42])),
            Op::Push(Node::NodeHash(target.hash())),
            Op::Parent,
        ];

        let query = query_from_items(&[QueryItem::Key(vec![42])]);
        let result = verify_query(&proof.encode().unwrap(), &query, target.hash()).unwrap();
        assert_eq!(result, vec![(vec![42], vec![42])])
    }
}
