use std::marker::PhantomData;
use std::sync::Arc;

use super::tree::{
    leaf_matches, route_bit_at, validate_key_len, MatchResult, MrtNode, MrtNodeInner, RouteBits,
    RoutePrefix,
};
use crate::error::{Error, Result};

pub(crate) enum NavStep<'a, H> {
    Leaf {
        skip: &'a RouteBits,
        value: &'a [u8],
    },
    Branch {
        skip: &'a RouteBits,
        left: H,
        right: H,
    },
    Pruned,
}

pub(crate) trait TreeNav<'a>: Copy {
    fn step(self) -> NavStep<'a, Self>;
}

impl<'a> TreeNav<'a> for &'a Arc<MrtNodeInner> {
    fn step(self) -> NavStep<'a, Self> {
        match self.node() {
            MrtNode::Leaf { skip, value } => NavStep::Leaf {
                skip,
                value: value.as_slice(),
            },
            MrtNode::Branch { skip, left, right } => NavStep::Branch { skip, left, right },
            MrtNode::PrunedHash => NavStep::Pruned,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IterDirection {
    Forward,
    Reverse,
}

/// The route from the root to a stack entry's node is rebuilt from the cursor's
/// single reusable [`RoutePrefix`] buffer rather than copied per entry: each entry
/// records a rewind `mark` (the bit-length to truncate the buffer to) plus the
/// `ext`ension to re-append — the parent branch's `skip` and the decision bit into
/// this node (`None` only for the root). On pop, the buffer is truncated to `mark`
/// and `ext` re-appended, reconstructing the node's prefix; leaves store only their
/// suffix, so the rebuilt prefix reconstructs absolute keys. This keeps a scan over
/// a shared-prefix subtree at ~one key copy-out per leaf instead of re-copying the
/// whole accumulated prefix per node and per leaf.
type Frame<'a, H> = (u16, Option<(&'a RouteBits, bool)>, H);

pub(crate) struct Cursor<'a, H: TreeNav<'a>> {
    /// One route buffer reused across the whole iteration; restored to each node's
    /// prefix as it is popped (and threaded down during a seek).
    path: RoutePrefix,
    stack: Vec<Frame<'a, H>>,
    direction: IterDirection,
    _marker: PhantomData<&'a ()>,
}

impl<'a, H: TreeNav<'a>> Cursor<'a, H> {
    pub(crate) fn empty() -> Self {
        Self {
            path: RoutePrefix::root(),
            stack: Vec::new(),
            direction: IterDirection::Forward,
            _marker: PhantomData,
        }
    }

    /// Rebuild `self.path` to the prefix of an entry popped with `(mark, ext)`:
    /// truncate to the parent's length, then re-append the parent edge.
    fn restore_path(&mut self, mark: u16, ext: Option<(&'a RouteBits, bool)>) -> Result<()> {
        self.path.truncate_to(mark);
        if let Some((skip, bit)) = ext {
            self.path.descend_in_place(skip, bit)?;
        }
        Ok(())
    }

    pub(crate) fn first(root: H) -> Self {
        let mut cursor = Self::empty();
        cursor.reset_to_first(root);
        cursor
    }

    pub(crate) fn last(root: H) -> Self {
        let mut cursor = Self::empty();
        cursor.reset_to_last(root);
        cursor
    }

    pub(crate) fn seek_ge(
        root: H,
        start: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Self> {
        let mut cursor = Self::empty();
        cursor.reset_to_ge(root, start, visit)?;
        Ok(cursor)
    }

    pub(crate) fn seek_le(
        root: H,
        start: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Self> {
        let mut cursor = Self::empty();
        cursor.reset_to_le(root, start, visit)?;
        Ok(cursor)
    }

    #[cfg(test)]
    pub(crate) fn seek_lt(
        root: H,
        end: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Self> {
        let mut cursor = Self::empty();
        cursor.reset_to_lt(root, end, visit)?;
        Ok(cursor)
    }

    pub(crate) fn reset_to_first(&mut self, root: H) {
        self.direction = IterDirection::Forward;
        self.stack.clear();
        self.path.clear();
        self.stack.push((0, None, root));
    }

    pub(crate) fn reset_to_last(&mut self, root: H) {
        self.direction = IterDirection::Reverse;
        self.stack.clear();
        self.path.clear();
        self.stack.push((0, None, root));
    }

    pub(crate) fn reset_to_ge(
        &mut self,
        root: H,
        start: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<()> {
        validate_key_len(start, "cursor seek_ge")?;
        self.direction = IterDirection::Forward;
        self.stack.clear();
        self.path.clear();
        self.push_lower_bound_forward(root, 0, (0, None, ()), start, visit)?;
        Ok(())
    }

    pub(crate) fn reset_to_le(
        &mut self,
        root: H,
        start: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<()> {
        validate_key_len(start, "cursor seek_le")?;
        self.direction = IterDirection::Reverse;
        self.stack.clear();
        self.path.clear();
        self.push_upper_bound_reverse(root, 0, (0, None, ()), start, false, visit)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn reset_to_lt(
        &mut self,
        root: H,
        end: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<()> {
        validate_key_len(end, "cursor seek_lt")?;
        self.direction = IterDirection::Reverse;
        self.stack.clear();
        self.path.clear();
        self.push_upper_bound_reverse(root, 0, (0, None, ()), end, true, visit)?;
        Ok(())
    }

    pub(crate) fn next_leaf(
        &mut self,
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Option<(Vec<u8>, &'a [u8])>> {
        debug_assert_eq!(self.direction, IterDirection::Forward);
        loop {
            let Some((mark, ext, node)) = self.stack.pop() else {
                return Ok(None);
            };
            self.restore_path(mark, ext)?;
            visit(&self.path, node);
            match node.step() {
                NavStep::Leaf { skip, value } => {
                    let key = self.path.key_with_suffix_scratch(skip)?;
                    return Ok(Some((key, value)));
                }
                NavStep::Branch { skip, left, right } => {
                    // `path` is this branch's prefix; children rewind to it (`mark`)
                    // and re-append `skip ‖ bit`. Push right first so left pops next.
                    let mark = self.path.bit_len();
                    self.stack.push((mark, Some((skip, true)), right));
                    self.stack.push((mark, Some((skip, false)), left));
                }
                NavStep::Pruned => {
                    return Err(Error::PrunedNode("cursor next descent".into()));
                }
            }
        }
    }

    pub(crate) fn next(
        &mut self,
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Option<(Vec<u8>, &'a [u8])>> {
        self.next_leaf(visit)
    }

    pub(crate) fn prev_leaf(
        &mut self,
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Option<(Vec<u8>, &'a [u8])>> {
        debug_assert_eq!(self.direction, IterDirection::Reverse);
        loop {
            let Some((mark, ext, node)) = self.stack.pop() else {
                return Ok(None);
            };
            self.restore_path(mark, ext)?;
            visit(&self.path, node);
            match node.step() {
                NavStep::Leaf { skip, value } => {
                    let key = self.path.key_with_suffix_scratch(skip)?;
                    return Ok(Some((key, value)));
                }
                NavStep::Branch { skip, left, right } => {
                    // Push left first so right pops next (descending order).
                    let mark = self.path.bit_len();
                    self.stack.push((mark, Some((skip, false)), left));
                    self.stack.push((mark, Some((skip, true)), right));
                }
                NavStep::Pruned => {
                    return Err(Error::PrunedNode("cursor prev descent".into()));
                }
            }
        }
    }

    pub(crate) fn prev(
        &mut self,
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<Option<(Vec<u8>, &'a [u8])>> {
        self.prev_leaf(visit)
    }

    // `restore` is how a pushed copy of `node` rebuilds its prefix (see `Frame`);
    // `self.path` holds `node`'s prefix on entry and is threaded down on descent.
    fn push_lower_bound_forward(
        &mut self,
        node: H,
        depth: u16,
        restore: Frame<'a, ()>,
        key: &[u8],
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<bool> {
        let (mark, ext, ()) = restore;
        visit(&self.path, node);
        match node.step() {
            NavStep::Leaf { skip, .. } => {
                let leaf_key = self.path.key_with_suffix_scratch(skip)?;
                if leaf_key.as_slice() >= key {
                    self.stack.push((mark, ext, node));
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            NavStep::Branch { skip, left, right } => match skip.matches_key_at(key, depth) {
                MatchResult::Mismatch { offset } => {
                    let position = depth.checked_add(offset).ok_or_else(|| {
                        Error::Tree("MRT cursor lower-bound depth overflow".into())
                    })?;
                    let key_bit = route_bit_at(key, position);
                    let subtree_bit = skip.bit_at(offset);
                    if !key_bit && subtree_bit {
                        self.stack.push((mark, ext, node));
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                }
                MatchResult::FullMatch => {
                    let branch_depth = depth.checked_add(skip.bit_len()).ok_or_else(|| {
                        Error::Tree("MRT cursor lower-bound branch depth overflow".into())
                    })?;
                    let child_depth = branch_depth.checked_add(1).ok_or_else(|| {
                        Error::Tree("MRT cursor lower-bound child depth overflow".into())
                    })?;
                    let here = self.path.bit_len();
                    if !route_bit_at(key, branch_depth) {
                        // Key goes left: the right subtree is entirely > key — save it,
                        // then thread `path` into the left child for the bound search.
                        self.stack.push((here, Some((skip, true)), right));
                        self.path.descend_in_place(skip, false)?;
                        self.push_lower_bound_forward(
                            left,
                            child_depth,
                            (here, Some((skip, false)), ()),
                            key,
                            visit,
                        )?;
                        Ok(true)
                    } else {
                        // Key goes right: the left subtree is entirely < key — skip it.
                        self.path.descend_in_place(skip, true)?;
                        self.push_lower_bound_forward(
                            right,
                            child_depth,
                            (here, Some((skip, true)), ()),
                            key,
                            visit,
                        )
                    }
                }
            },
            NavStep::Pruned => Err(Error::PrunedNode(format!(
                "cursor lower-bound descent at {key:?}"
            ))),
        }
    }

    fn push_upper_bound_reverse(
        &mut self,
        node: H,
        depth: u16,
        restore: Frame<'a, ()>,
        key: &[u8],
        strict: bool,
        visit: &mut impl FnMut(&RoutePrefix, H),
    ) -> Result<bool> {
        let (mark, ext, ()) = restore;
        visit(&self.path, node);
        match node.step() {
            NavStep::Leaf { skip, .. } => {
                let leaf_key = self.path.key_with_suffix_scratch(skip)?;
                let include = if strict {
                    leaf_key.as_slice() < key
                } else {
                    leaf_key.as_slice() <= key
                };
                if include {
                    self.stack.push((mark, ext, node));
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            NavStep::Branch { skip, left, right } => match skip.matches_key_at(key, depth) {
                MatchResult::Mismatch { offset } => {
                    let position = depth.checked_add(offset).ok_or_else(|| {
                        Error::Tree("MRT cursor upper-bound depth overflow".into())
                    })?;
                    let key_bit = route_bit_at(key, position);
                    let subtree_bit = skip.bit_at(offset);
                    if key_bit && !subtree_bit {
                        self.stack.push((mark, ext, node));
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                }
                MatchResult::FullMatch => {
                    let branch_depth = depth.checked_add(skip.bit_len()).ok_or_else(|| {
                        Error::Tree("MRT cursor upper-bound branch depth overflow".into())
                    })?;
                    let child_depth = branch_depth.checked_add(1).ok_or_else(|| {
                        Error::Tree("MRT cursor upper-bound child depth overflow".into())
                    })?;
                    let here = self.path.bit_len();
                    if route_bit_at(key, branch_depth) {
                        // Key goes right: the left subtree is entirely < key — save it,
                        // then thread `path` into the right child.
                        self.stack.push((here, Some((skip, false)), left));
                        self.path.descend_in_place(skip, true)?;
                        self.push_upper_bound_reverse(
                            right,
                            child_depth,
                            (here, Some((skip, true)), ()),
                            key,
                            strict,
                            visit,
                        )?;
                        Ok(true)
                    } else {
                        self.path.descend_in_place(skip, false)?;
                        self.push_upper_bound_reverse(
                            left,
                            child_depth,
                            (here, Some((skip, false)), ()),
                            key,
                            strict,
                            visit,
                        )
                    }
                }
            },
            NavStep::Pruned => Err(Error::PrunedNode(format!(
                "cursor upper-bound descent at {key:?}"
            ))),
        }
    }
}

/// Point-get navigation: descends from `root` following `key`'s own route
/// (`route_bit_at` / `matches_key_at`), invoking `visit` on every node entered,
/// and returns the value of the leaf the key routes to (`Some` iff that leaf's
/// key equals `key`). This is the read counterpart to [`Cursor`] for point gets:
/// it follows the *key's* path rather than a lower-bound seek, so an absent key
/// never lands on (nor reveals) its successor. The tracer and the verifier both
/// drive it over their respective handles, so a get's reveal set is the verifier's
/// get touch set by construction (same descent, one handle).
pub(crate) fn get_descent<'a, H: TreeNav<'a>>(
    root: H,
    key: &[u8],
    visit: &mut impl FnMut(H),
) -> Result<Option<&'a [u8]>> {
    validate_key_len(key, "get descent")?;
    let mut cur = root;
    let mut depth = 0u16;
    loop {
        visit(cur);
        match cur.step() {
            NavStep::Leaf { skip, value } => {
                return Ok(leaf_matches(skip, key, depth).then_some(value));
            }
            NavStep::Branch { skip, left, right } => {
                if matches!(
                    skip.matches_key_at(key, depth),
                    MatchResult::Mismatch { .. }
                ) {
                    return Ok(None);
                }
                let branch_depth = depth
                    .checked_add(skip.bit_len())
                    .ok_or_else(|| Error::Tree("MRT get descent branch depth overflow".into()))?;
                let side = route_bit_at(key, branch_depth);
                depth = branch_depth
                    .checked_add(1)
                    .ok_or_else(|| Error::Tree("MRT get descent child depth overflow".into()))?;
                cur = if !side { left } else { right };
            }
            NavStep::Pruned => {
                return Err(Error::PrunedNode(format!("MRT get descent at key {key:?}")));
            }
        }
    }
}

// Shared ordered-read helpers over any `TreeNav` handle (the host `&Arc<MrtNodeInner>`
// or the verify `&MrtVerifyNode`): the host snapshot, the query-proof verify reader,
// and the verify tree all route through these, so reads have one implementation
// instead of each re-wrapping `Cursor`/`get_descent`. A reader records nothing, so the
// cursor's visit hook is a no-op.

pub(crate) fn point_get<'a, H: TreeNav<'a>>(
    root: Option<H>,
    key: &[u8],
) -> Result<Option<Vec<u8>>> {
    validate_key_len(key, "mrt get")?;
    let Some(root) = root else {
        return Ok(None);
    };
    Ok(get_descent(root, key, &mut |_| {})?.map(|value| value.to_vec()))
}

#[cfg(test)]
pub(crate) fn collect_all<'a, H: TreeNav<'a>>(root: Option<H>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let mut cursor = Cursor::first(root);
    collect_forward_until(&mut cursor, |_| false)
}

pub(crate) fn collect_range<'a, H: TreeNav<'a>>(
    root: Option<H>,
    start: &[u8],
    end: Option<&[u8]>,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    validate_key_len(start, "mrt range start")?;
    if let Some(end) = end {
        validate_key_len(end, "mrt range end")?;
        if start >= end {
            return Ok(Vec::new());
        }
    }
    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let mut cursor = Cursor::seek_ge(root, start, &mut |_, _| {})?;
    collect_forward_until(&mut cursor, |key| end.is_some_and(|end| key >= end))
}

pub(crate) fn collect_prefix<'a, H: TreeNav<'a>>(
    root: Option<H>,
    prefix: &[u8],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    validate_key_len(prefix, "mrt prefix")?;
    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let mut cursor = Cursor::seek_ge(root, prefix, &mut |_, _| {})?;
    collect_forward_until(&mut cursor, |key| !key.starts_with(prefix))
}

pub(crate) fn collect_range_inclusive<'a, H: TreeNav<'a>>(
    root: Option<H>,
    start: &[u8],
    end: &[u8],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    validate_key_len(start, "mrt inclusive range start")?;
    validate_key_len(end, "mrt inclusive range end")?;
    assert!(
        start <= end,
        "inverted inclusive range: start {:?} must be <= end {:?}",
        start,
        end
    );
    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let mut cursor = Cursor::seek_ge(root, start, &mut |_, _| {})?;
    collect_forward_until(&mut cursor, |key| key > end)
}

fn collect_forward_until<'a, H: TreeNav<'a>>(
    cursor: &mut Cursor<'a, H>,
    stop: impl Fn(&[u8]) -> bool,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut out = Vec::new();
    let mut visit = |_: &RoutePrefix, _: H| {};
    while let Some((key, value)) = cursor.next_leaf(&mut visit)? {
        if stop(&key) {
            break;
        }
        out.push((key, value.to_vec()));
    }
    Ok(out)
}

pub struct Iter<'a> {
    cursor: Option<Cursor<'a, &'a Arc<MrtNodeInner>>>,
}

impl<'a> Iter<'a> {
    pub(crate) fn new(root: Option<&'a Arc<MrtNodeInner>>) -> Self {
        Self {
            cursor: root.map(Cursor::first),
        }
    }

    pub(crate) fn from_key(root: Option<&'a Arc<MrtNodeInner>>, start_key: &[u8]) -> Self {
        Self {
            cursor: root.and_then(|root| {
                let mut visit = |_: &RoutePrefix, _: &'a Arc<MrtNodeInner>| {};
                Cursor::seek_ge(root, start_key, &mut visit).ok()
            }),
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor.as_mut()?;
        let mut visit = |_: &RoutePrefix, _: &'a Arc<MrtNodeInner>| {};
        match cursor
            .next(&mut visit)
            .expect("complete MRT snapshot iteration must not hit pruned nodes")
        {
            Some((key, value)) => Some((key, value.to_vec())),
            None => {
                self.cursor = None;
                None
            }
        }
    }
}

pub struct ReverseIter<'a> {
    cursor: Option<Cursor<'a, &'a Arc<MrtNodeInner>>>,
}

impl<'a> ReverseIter<'a> {
    pub(crate) fn new(root: Option<&'a Arc<MrtNodeInner>>) -> Self {
        Self {
            cursor: root.map(Cursor::last),
        }
    }

    pub(crate) fn from_key_inclusive(root: Option<&'a Arc<MrtNodeInner>>, end_key: &[u8]) -> Self {
        Self {
            cursor: root.and_then(|root| {
                let mut visit = |_: &RoutePrefix, _: &'a Arc<MrtNodeInner>| {};
                Cursor::seek_le(root, end_key, &mut visit).ok()
            }),
        }
    }
}

impl<'a> Iterator for ReverseIter<'a> {
    type Item = (Vec<u8>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor.as_mut()?;
        let mut visit = |_: &RoutePrefix, _: &'a Arc<MrtNodeInner>| {};
        match cursor
            .prev(&mut visit)
            .expect("complete MRT snapshot iteration must not hit pruned nodes")
        {
            Some((key, value)) => Some((key, value.to_vec())),
            None => {
                self.cursor = None;
                None
            }
        }
    }
}
