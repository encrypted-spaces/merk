use crate::avl::child::Child;
use crate::avl::node::Node;
use crate::avl::walker::{Fetch, Walker};
use crate::error::{Error, Result};

impl<S> Walker<S>
where
    S: Fetch + Sized + Send + Clone,
{
    // Split / Join (Blelloch)

    // Split/join uses raw rotations as local rebalancing steps. These helpers
    // intentionally avoid rotate(), whose recursive maybe_balance() calls are
    // part of the existing insert/delete balancing path.
    fn split_join_rotate_left(self) -> Result<Self> {
        let (node_a, child_b) = self.detach_expect(false)?;
        let (child_b, maybe_m) = child_b.detach(true)?;
        let node_a = node_a.attach(false, maybe_m);
        Ok(child_b.attach(true, Some(node_a)))
    }

    fn split_join_rotate_right(self) -> Result<Self> {
        let (node_a, child_b) = self.detach_expect(true)?;
        let (child_b, maybe_m) = child_b.detach(false)?;
        let node_a = node_a.attach(true, maybe_m);
        Ok(child_b.attach(false, Some(node_a)))
    }

    pub(crate) fn join_trees(
        left: Option<Node>,
        pivot: Node,
        right: Option<Node>,
        source: S,
    ) -> Result<Node> {
        let lh = left.as_ref().map_or(0, |t| t.height());
        let rh = right.as_ref().map_or(0, |t| t.height());

        if lh > rh + 1 {
            let left_walker = Walker::new(left.unwrap(), source);
            Self::join_right_spine(left_walker, pivot, right, rh)
        } else if rh > lh + 1 {
            let right_walker = Walker::new(right.unwrap(), source);
            Self::join_left_spine(left, pivot, right_walker, lh)
        } else {
            let mut result = pivot;
            let inner = result.inner_mut();
            inner.left = left.map(Child::Resident);
            inner.right = right.map(Child::Resident);
            inner.node_hash = None;
            inner.recompute_height();
            Ok(result)
        }
    }

    fn join_right_spine(
        left_walker: Self,
        pivot: Node,
        right: Option<Node>,
        shorter_h: u8,
    ) -> Result<Node> {
        if left_walker.tree().height() <= shorter_h + 1 {
            let left_tree = left_walker.into_inner();
            let mut result = pivot;
            let inner = result.inner_mut();
            inner.left = Some(Child::Resident(left_tree));
            inner.right = right.map(Child::Resident);
            inner.node_hash = None;
            inner.recompute_height();
            return Ok(result);
        }

        let (left_walker, maybe_right_child) = left_walker.detach(false)?;

        let joined = match maybe_right_child {
            Some(rc_walker) => Self::join_right_spine(rc_walker, pivot, right, shorter_h)?,
            None => {
                let mut p = pivot;
                let inner = p.inner_mut();
                inner.right = right.map(Child::Resident);
                inner.node_hash = None;
                inner.recompute_height();
                p
            }
        };

        let left_walker = left_walker.attach(false, Some(joined));

        let bf = left_walker.tree().balance_factor();
        if bf > 1 {
            let right_bf = left_walker.child_balance_factor_for_rotation(false)?;
            if right_bf < 0 {
                let (left_walker, right_child) = left_walker.detach_expect(false)?;
                let rotated = right_child.split_join_rotate_right()?;
                let left_walker = left_walker.attach(false, Some(rotated));
                Ok(left_walker.split_join_rotate_left()?.into_inner())
            } else {
                Ok(left_walker.split_join_rotate_left()?.into_inner())
            }
        } else {
            Ok(left_walker.into_inner())
        }
    }

    fn join_left_spine(
        left: Option<Node>,
        pivot: Node,
        right_walker: Self,
        shorter_h: u8,
    ) -> Result<Node> {
        if right_walker.tree().height() <= shorter_h + 1 {
            let right_tree = right_walker.into_inner();
            let mut result = pivot;
            let inner = result.inner_mut();
            inner.left = left.map(Child::Resident);
            inner.right = Some(Child::Resident(right_tree));
            inner.node_hash = None;
            inner.recompute_height();
            return Ok(result);
        }

        let (right_walker, maybe_left_child) = right_walker.detach(true)?;

        let joined = match maybe_left_child {
            Some(lc_walker) => Self::join_left_spine(left, pivot, lc_walker, shorter_h)?,
            None => {
                let mut p = pivot;
                let inner = p.inner_mut();
                inner.left = left.map(Child::Resident);
                inner.node_hash = None;
                inner.recompute_height();
                p
            }
        };

        let right_walker = right_walker.attach(true, Some(joined));

        let bf = right_walker.tree().balance_factor();
        if bf < -1 {
            let left_bf = right_walker.child_balance_factor_for_rotation(true)?;
            if left_bf > 0 {
                let (right_walker, left_child) = right_walker.detach_expect(true)?;
                let rotated = left_child.split_join_rotate_left()?;
                let right_walker = right_walker.attach(true, Some(rotated));
                Ok(right_walker.split_join_rotate_right()?.into_inner())
            } else {
                Ok(right_walker.split_join_rotate_right()?.into_inner())
            }
        } else {
            Ok(right_walker.into_inner())
        }
    }

    pub(crate) fn extract_min(walker: Self) -> Result<(Node, Option<Node>)> {
        if walker.tree().child_ref(true).is_none() {
            let (walker, maybe_right) = walker.detach(false)?;
            let min_node = walker.into_inner();
            return Ok((min_node, maybe_right.map(|w| w.into_inner())));
        }

        let (walker, left_child) = walker.detach_expect(true)?;
        let (min_node, rest) = Self::extract_min(left_child)?;

        let walker = walker.attach(true, rest);

        let bf = walker.tree().balance_factor();
        if bf > 1 {
            let right_bf = walker.child_balance_factor_for_rotation(false)?;
            if right_bf < 0 {
                let (walker, right_child) = walker.detach_expect(false)?;
                let rotated = right_child.split_join_rotate_right()?;
                let walker = walker.attach(false, Some(rotated));
                let result = walker.split_join_rotate_left()?;
                Ok((min_node, Some(result.into_inner())))
            } else {
                let result = walker.split_join_rotate_left()?;
                Ok((min_node, Some(result.into_inner())))
            }
        } else {
            Ok((min_node, Some(walker.into_inner())))
        }
    }

    pub(crate) fn join2_trees(
        left: Option<Node>,
        right: Option<Node>,
        source: S,
    ) -> Result<Option<Node>> {
        match (left, right) {
            (None, r) => Ok(r),
            (l, None) => Ok(l),
            (l, Some(r)) => {
                let right_walker = Self::new(r, source.clone());
                let (min_node, right_rest) = Self::extract_min(right_walker)?;
                Ok(Some(Self::join_trees(l, min_node, right_rest, source)?))
            }
        }
    }

    pub(crate) fn split_at(self, key: &[u8]) -> Result<(Option<Node>, Option<Node>)> {
        let source = self.clone_source();

        if self.tree().key() < key {
            let (walker, maybe_left) = self.detach(true)?;
            let off_path = maybe_left.map(|w| w.into_inner());

            let (walker, maybe_right) = walker.detach(false)?;

            let (rl, rr) = match maybe_right {
                Some(right_walker) => right_walker.split_at(key)?,
                None => (None, None),
            };

            let node = walker.into_inner();
            let left_result = Some(Self::join_trees(off_path, node, rl, source)?);
            Ok((left_result, rr))
        } else {
            let (walker, maybe_right) = self.detach(false)?;
            let off_path = maybe_right.map(|w| w.into_inner());

            let (walker, maybe_left) = walker.detach(true)?;

            let (ll, lr) = match maybe_left {
                Some(left_walker) => left_walker.split_at(key)?,
                None => (None, None),
            };

            let node = walker.into_inner();
            let right_result = Some(Self::join_trees(lr, node, off_path, source)?);
            Ok((ll, right_result))
        }
    }

    pub(crate) fn delete_range(self, start: &[u8], end: &[u8]) -> Result<Option<Node>> {
        if start >= end {
            return Err(Error::Bound(
                "delete_range start must be less than end".into(),
            ));
        }

        let source = self.clone_source();
        let (left, right1) = self.split_at(start)?;
        let right = match right1 {
            Some(right1) => {
                let walker = Self::new(right1, source.clone());
                let (_middle, right) = walker.split_at(end)?;
                right
            }
            None => None,
        };

        Self::join2_trees(left, right, source)
    }

    pub(crate) fn delete_range_apply_to(
        maybe_tree: Option<Self>,
        start: &[u8],
        end: &[u8],
    ) -> Result<Option<Node>> {
        if start >= end {
            return Err(Error::Bound(
                "delete_range start must be less than end".into(),
            ));
        }

        match maybe_tree {
            Some(walker) => walker.delete_range(start, end),
            None => Ok(None),
        }
    }
}
