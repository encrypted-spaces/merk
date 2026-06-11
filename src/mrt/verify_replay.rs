use std::convert::TryFrom;
use std::sync::Arc;

use super::cursor::{self, NavStep, TreeNav};
use super::trace::{Trace, MAX_MRT_TRACE_DECODE_DEPTH};
use super::tree::{self, MatchResult, MrtNode, MrtNodeInner, RouteBits, RoutePrefix};
use super::validate_batch_ops;
#[cfg(test)]
use super::validate_mrt_apply_batch;
use crate::error::{Error, Result};
use crate::hash::{Hash, HASH_LENGTH, NULL_HASH};
#[cfg(test)]
use crate::ops::{Batch, BatchEntry, Op};
use crate::tracer::{prefix_successor, BatchOp};

#[derive(Clone, Debug)]
enum MrtVerifyNode {
    Leaf {
        skip: RouteBits,
        value: Vec<u8>,
        hash_cache: Option<Hash>,
    },
    Branch {
        skip: RouteBits,
        left: Box<MrtVerifyNode>,
        right: Box<MrtVerifyNode>,
        hash_cache: Option<Hash>,
        /// Lazily-built, persistent SHA-256 preimage (`child hash slots(64) ‖ child
        /// depth slots(8) ‖ bit_len ‖ skip ‖ tag ‖ pad`), held as `Box<[u32]>` for
        /// word alignment. Built on first rehash after this branch goes dirty; the
        /// *stable* tail (`[72..]`) is written once, while child hashes are written
        /// into `[0..64]` directly by the SHA syscall and the two child
        /// `depth_below` words at `[64..72]` are rewritten from the children on
        /// every dirty rehash (they are dynamic — child max-depths change on
        /// writes) in [`MrtVerifyNode::rehash_into`]. `None` until then (decoded
        /// branches keep their eager `hash_cache` and never build it unless later
        /// mutated). PROTOTYPE — see `tree::build_branch_preimage`.
        preimage: Option<Box<[u32]>>,
        /// Eager relative max-subtree-depth (the verifier twin of host
        /// `MrtNodeInner::depth_below`): `skip.bit_len() + 1 + max(child depths)`.
        /// Committed in this node's *parent* hash (per child); recomputed by the
        /// constructors that build a branch (`new_branch` / decode), never read from
        /// the trace for a materialized node.
        depth_below: u16,
    },
    /// A pruned stub: `(hash, depth_below)`. The depth is stamped from the parent's
    /// edge at decode (it can't be derived from a bare hash) and is authenticated by
    /// the parent's branch hash → root.
    PrunedHash(Hash, u16),
}

impl<'a> TreeNav<'a> for &'a MrtVerifyNode {
    fn step(self) -> NavStep<'a, Self> {
        match self {
            MrtVerifyNode::Leaf { skip, value, .. } => NavStep::Leaf {
                skip,
                value: value.as_slice(),
            },
            MrtVerifyNode::Branch {
                skip, left, right, ..
            } => NavStep::Branch {
                skip,
                left: left.as_ref(),
                right: right.as_ref(),
            },
            MrtVerifyNode::PrunedHash(..) => NavStep::Pruned,
        }
    }
}

impl MrtVerifyNode {
    /// The eager relative max-subtree-depth (leaf → `skip.bit_len()`; branch /
    /// pruned stub → the stored field). O(1), the verifier twin of
    /// `tree::MrtNodeInner::depth_below`.
    fn depth_below(&self) -> u16 {
        match self {
            MrtVerifyNode::Leaf { skip, .. } => skip.bit_len(),
            MrtVerifyNode::Branch { depth_below, .. } => *depth_below,
            MrtVerifyNode::PrunedHash(_, depth_below) => *depth_below,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MrtVerifyTree {
    root: Option<Box<MrtVerifyNode>>,
}

#[derive(Clone, Debug, Default)]
pub struct TraceVerifier {
    inner: MrtVerifyTree,
}

impl TraceVerifier {
    pub fn from_trace(trace: &Trace) -> Self {
        Self {
            inner: MrtVerifyTree::from_trace(trace),
        }
    }

    /// Decodes a flat MRT trace — the wire format produced by [`Trace`]'s
    /// `ed::Encode` impl — **directly** into the verify tree, skipping the
    /// `postcard`/serde decode and the intermediate `Trace` +
    /// [`from_trace`](Self::from_trace) rebuild. The result is behaviorally
    /// identical to `from_trace(&Trace::decode_exact(bytes)?)`; this is purely
    /// a guest-cycle optimization (the trace decode + rebuild is ~40% of MRT
    /// guest cost). Hashes are computed eagerly during decode, so a subsequent
    /// `verify_root` stays O(1).
    pub fn decode_trace(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            inner: MrtVerifyTree::decode_trace(bytes)?,
        })
    }

    pub fn verify_root(&mut self, expected: Hash) -> Result<()> {
        self.inner.verify_root(expected)
    }

    /// The current (post-replay) root hash. Use after `verify_root(start)` +
    /// replaying the caller-supplied batches to obtain the resulting end root.
    pub fn root_hash(&mut self) -> Result<Hash> {
        self.inner.root_hash()
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.get(key)
    }

    pub fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.inner.collect_range(start, end)
    }

    pub fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.inner.collect_prefix(prefix)
    }

    /// Apply `ops` to the tree. **Atomic**: a mid-batch failure (e.g. a prefix-free
    /// violation) leaves the tree unchanged. (The internal verifier replays in place
    /// and aborts on error, so it skips this clone.)
    ///
    /// Test-only — production drives writes through [`TraceReplayer`] (op-by-op
    /// `replay_batch_ops_in_place`).
    #[cfg(test)]
    pub(crate) fn replay_batch_ops(&mut self, ops: &[BatchOp]) -> Result<()> {
        // Validate directly on the `BatchOp`s — before the value-copying
        // `to_batch_entry` conversion *and* before the O(trace) tree clone — so an
        // invalid batch (over-long key, inverted DeleteRange bounds, unsorted or
        // duplicate keys) is rejected without copying any op value or the tree.
        // The candidate replays the borrowed `BatchOp`s directly, avoiding the old
        // temporary `Vec<BatchEntry>` and the key/end clones it carried.
        validate_batch_ops(ops)?;
        let mut candidate = self.inner.clone();
        candidate.replay_batch_ops_unchecked(ops)?;
        self.inner = candidate;
        Ok(())
    }

    /// In-place [`replay_batch_ops`](Self::replay_batch_ops) that skips the defensive
    /// whole-tree clone.
    ///
    /// **Not atomic:** on `Err`, `self` is left partially mutated and must be
    /// discarded. Intended for the zkVM verify loop — any failure rejects the whole
    /// proof and drops the tree — where the per-call clone in `replay_batch_ops` makes
    /// a sequence of `n` changes O(n × tree_size) instead of O(n × depth).
    pub(crate) fn replay_batch_ops_in_place(&mut self, ops: &[BatchOp]) -> Result<()> {
        validate_batch_ops(ops)?;
        self.inner.replay_batch_ops_unchecked(ops)
    }

    /// Relocate the prefix-`from` subtree to `to`. **Atomic**: the in-place
    /// detach/splice can otherwise leave the tree mid-move on error (source removed,
    /// or a `placeholder_node` after a rejected re-skin) — so this wrapper applies to
    /// a clone and commits only on success. (The internal verifier replay uses the
    /// in-place `MrtVerifyTree::move_prefix` directly.)
    ///
    /// Test-only — production drives moves through [`TraceReplayer`] (op-by-op
    /// `move_prefix_in_place`).
    #[cfg(test)]
    pub(crate) fn move_prefix(&mut self, from: &[u8], to: &[u8]) -> Result<()> {
        // Cheap stateless check *before* the O(trace) clone, so invalid args
        // (equal prefixes or overlong prefix args) don't duplicate the tree.
        tree::validate_move_prefix_args(from, to)?;
        let mut candidate = self.inner.clone();
        candidate.move_prefix(from, to)?;
        self.inner = candidate;
        Ok(())
    }

    /// In-place [`move_prefix`](Self::move_prefix) that skips the defensive whole-tree
    /// clone (see [`replay_batch_ops_in_place`](Self::replay_batch_ops_in_place)).
    /// **Not atomic** — discard `self` on `Err`. For the zkVM verify loop.
    pub(crate) fn move_prefix_in_place(&mut self, from: &[u8], to: &[u8]) -> Result<()> {
        self.inner.move_prefix(from, to)
    }

    fn delete_prefix_in_place(&mut self, prefix: &[u8]) -> Result<()> {
        self.inner.delete_prefix(prefix)
    }
}

/// Verify-side traced handle for the MRT backend, bound to an expected start
/// root at construction. The externalized peer of [`crate::avl::TraceReplayer`].
///
/// A thin wrapper over [`TraceVerifier`] that gives consumers the same sequenced
/// read/write calls ([`TraceInterface`](crate::tracer::TraceInterface)) the
/// prove side uses, with the trace decoded once and authenticated against the
/// caller's expected pre-state.
///
/// Build it with [`new_verified`](Self::new_verified) or
/// [`new_unverified`](Self::new_unverified). Read soundness comes straight from
/// the wrapped verifier: a pruned non-membership path or an incomplete
/// range/prefix witness fails with [`Error::PrunedNode`] rather than returning
/// `Ok(None)` or a truncated list.
///
/// **Single-shot on error:** mutations apply in place, so any method returning
/// `Err` may leave the replayer partially mutated. Discard the handle after an
/// error — do not keep using it. This matches the zkVM verify model, where any
/// error rejects the whole proof and drops the tree.
#[derive(Clone, Debug, Default)]
pub struct TraceReplayer {
    verifier: TraceVerifier,
    /// Set once an [`apply`](crate::tracer::TraceInterface::apply) op fails. A
    /// poisoned replayer rejects every subsequent fallible op (reads, further
    /// `apply`, root verification) with [`Error::Poisoned`] — the in-place replay
    /// may have left the tree partially mutated on the failed op, so reads/replay
    /// must not be trusted again.
    poisoned: bool,
}

impl TraceReplayer {
    /// Decode `trace_bytes`, authenticate the trace's root against
    /// `expected_start_root`, and bind the replayer to that pre-state.
    ///
    /// Fails closed with [`Error::HashMismatch`] when the decoded trace's root
    /// does not match `expected_start_root` — *before* any read can be trusted —
    /// or with a decode error when the bytes are malformed. This is the
    /// constructor production callers use: a successful return means subsequent
    /// reads are authenticated against the expected start root.
    pub fn new_verified(trace_bytes: &[u8], expected_start_root: Hash) -> Result<Self> {
        let mut verifier = TraceVerifier::decode_trace(trace_bytes)?;
        verifier.verify_root(expected_start_root)?;
        Ok(Self {
            verifier,
            poisoned: false,
        })
    }

    /// Decode `trace_bytes` without binding them to a start root.
    ///
    /// For tests, diagnostics, and callers that intentionally bind later: reads
    /// through the returned handle are **not** trusted until the caller checks
    /// `root_hash() == expected_start_root` itself. Prefer
    /// [`new_verified`](Self::new_verified) in production.
    pub fn new_unverified(trace_bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            verifier: TraceVerifier::decode_trace(trace_bytes)?,
            poisoned: false,
        })
    }

    /// Reject any fallible op once the handle is poisoned.
    fn reject_if_poisoned(&self, op: &str) -> Result<()> {
        if self.poisoned {
            return Err(Error::Poisoned(format!(
                "{op} on a poisoned MRT replayer (an earlier apply failed)"
            )));
        }
        Ok(())
    }

    /// The current tree's root hash after any sequenced writes replayed so far.
    /// Fails [`Error::Poisoned`] if a prior `apply` op failed (the post-state root
    /// would be untrustworthy).
    pub fn root_hash(&mut self) -> Result<Hash> {
        self.reject_if_poisoned("root_hash")?;
        self.verifier.root_hash()
    }
}

impl crate::tracer::TraceReader for TraceReplayer {
    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.reject_if_poisoned("get")?;
        self.verifier.get(key)
    }

    fn get_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_range")?;
        self.verifier.collect_range(start, Some(end))
    }

    fn get_prefix(&mut self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.reject_if_poisoned("get_prefix")?;
        self.verifier.collect_prefix(prefix)
    }
}

/// Replays each `WriteOp` **in vector order, one at a time**, lowering it to the
/// same per-op primitive: point/range ops to a single-element [`BatchOp`] in-place
/// replay, `DeletePrefix`/`MovePrefix` to the verifier's in-place prefix and move
/// primitives. A `Vec<WriteOp>` is never gathered into a sorted multi-op batch. An
/// empty batch is a no-op. The first failing op poisons the replayer.
impl crate::tracer::TraceInterface for TraceReplayer {
    fn apply(&mut self, ops: &[crate::tracer::WriteOp]) -> Result<()> {
        use crate::tracer::WriteOp;
        self.reject_if_poisoned("apply")?;
        for op in ops {
            let result = match op {
                WriteOp::Put { key, value } => {
                    self.verifier.replay_batch_ops_in_place(&[BatchOp::Put {
                        key: key.clone(),
                        value: value.clone(),
                    }])
                }
                WriteOp::Delete { key } => self
                    .verifier
                    .replay_batch_ops_in_place(&[BatchOp::Delete { key: key.clone() }]),
                WriteOp::DeleteRange { start, end } => {
                    self.verifier
                        .replay_batch_ops_in_place(&[BatchOp::DeleteRange {
                            start: start.clone(),
                            end: end.clone(),
                        }])
                }
                WriteOp::DeletePrefix { prefix } => self.verifier.delete_prefix_in_place(prefix),
                WriteOp::MovePrefix { from, to } => self.verifier.move_prefix_in_place(from, to),
            };
            if let Err(err) = result {
                self.poisoned = true;
                return Err(err);
            }
        }
        Ok(())
    }
}

impl MrtVerifyTree {
    pub(crate) fn from_trace(trace: &Trace) -> Self {
        Self {
            root: trace.0.as_ref().map(node_from_arc),
        }
    }

    pub(crate) fn decode_trace(bytes: &[u8]) -> Result<Self> {
        let mut reader = TraceReader::new(bytes);
        // Top-level `0x00` is the empty-tree marker (see `Trace`'s `Encode`);
        // anything else is a node tag.
        let tag = reader.read_u8()?;
        let root = match tag {
            0x00 => None,
            // A bare pruned root carries no parent edge to authenticate its depth
            // (and supports no operation), so it is trace-malleable — reject it.
            0x03 => {
                return Err(Error::Proof(
                    "MRT trace root must be a materialized node or empty, not a pruned stub".into(),
                ))
            }
            _ => Some(decode_trace_node(tag, &mut reader, 0)?),
        };
        if reader.remaining() != 0 {
            return Err(Error::Proof(format!(
                "MRT trace has {} trailing byte(s)",
                reader.remaining()
            )));
        }
        Ok(Self { root })
    }

    pub(crate) fn root_hash(&mut self) -> Result<Hash> {
        let Some(node) = self.root.as_mut() else {
            return Ok(NULL_HASH);
        };
        // The root has no parent slot, so hash into a local 4-aligned holder (the
        // zkVM syscall writes the digest as `[u32; 8]`, so `dest` must be aligned).
        #[repr(align(4))]
        struct Aligned([u8; HASH_LENGTH]);
        let mut holder = Aligned(NULL_HASH);
        // SAFETY: `holder.0` is 32 bytes, 4-aligned, and disjoint from every node
        // buffer in the tree.
        unsafe { node.rehash_into(holder.0.as_mut_ptr()) };
        Ok(holder.0)
    }

    pub(crate) fn verify_root(&mut self, expected: Hash) -> Result<()> {
        let actual = self.root_hash()?;
        if actual == expected {
            Ok(())
        } else {
            Err(Error::HashMismatch(expected, actual))
        }
    }

    pub(crate) fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Shared point-get / scan helpers: the verifier and the host drive the same
        // `cursor::*` over their respective handles, so the verifier's touch set is the
        // tracer's reveal set by construction.
        cursor::point_get(self.root.as_deref(), key)
    }

    pub(crate) fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        cursor::collect_range(self.root.as_deref(), start, end)
    }

    pub(crate) fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        cursor::collect_prefix(self.root.as_deref(), prefix)
    }

    #[cfg(test)]
    pub(crate) fn replay(&mut self, batch: &Batch) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        validate_mrt_apply_batch(batch)?;
        self.root = replay_root(self.root.take(), batch)?;
        Ok(())
    }

    fn replay_batch_ops_unchecked(&mut self, ops: &[BatchOp]) -> Result<()> {
        replay_batch_ops_in_root(&mut self.root, ops)
    }

    /// Replays a `MovePrefix` step: relocate the whole prefix-`from` subtree to
    /// prefix `to`, mirroring the host `tree::move_prefix` (detach + splice). The
    /// re-skinned `S` root hash is **derived** here from the authenticated original
    /// (carried child hash-stubs for a branch, or the leaf value), never trusted
    /// from the trace. Validates the stateless args first, like every mutating
    /// path. `Err` on an absent `from` or a non-empty / prefix-violating `to`.
    pub(crate) fn move_prefix(&mut self, from: &[u8], to: &[u8]) -> Result<()> {
        tree::validate_move_prefix_args(from, to)?;
        // In-place: detach `S` out of the tree (collapsing the source path) then
        // splice it back in at `to`, mutating through `&mut` like the in-place
        // insert/delete (work B). Only the collapse survivor and the spliced
        // connector + re-skinned `S` root allocate — not the whole from-/to-path.
        let captured = detach_in_place(&mut self.root, from)?;
        tree::validate_move_prefix_result_depth(
            to,
            captured.s_root.depth_below(),
            captured.strip_prefix_bits,
        )?;
        splice_in_place(&mut self.root, to, captured)
    }

    /// Delete every key with byte-prefix `prefix`, including the no-successor
    /// edge (`[prefix, ∞)`) for all-`0xff` prefixes and the empty prefix.
    pub(crate) fn delete_prefix(&mut self, prefix: &[u8]) -> Result<()> {
        delete_prefix(&mut self.root, prefix)
    }
}

impl MrtVerifyNode {
    /// Computes this node's hash (lazily, memoized in `hash_cache`) and writes the
    /// 32 bytes to `dest`. A clean subtree (`hash_cache == Some`) is skipped — its
    /// cached hash is copied to `dest`. A dirty branch builds its preimage if
    /// needed and recurses with `dest` aimed at its own `[0..32]`/`[32..64]` child
    /// slots, so each child's hash is written **directly into this branch's
    /// preimage** by the syscall — no inter-node hash copy (NEW_ADVICE_MERK.md §1).
    ///
    /// # Safety
    /// `dest` must be valid for 32 writes and 4-byte aligned, and must not alias
    /// `self`'s own preimage buffer. The recursion only ever passes slots of a
    /// *parent's* buffer down to a child, which are always disjoint allocations,
    /// so this holds for every internal call; the root caller supplies an aligned
    /// stack holder.
    unsafe fn rehash_into(&mut self, dest: *mut u8) {
        match self {
            MrtVerifyNode::Leaf {
                skip,
                value,
                hash_cache,
            } => {
                let hash = match *hash_cache {
                    Some(hash) => hash,
                    None => {
                        let hash = tree::compute_leaf_hash(skip, value);
                        *hash_cache = Some(hash);
                        hash
                    }
                };
                core::ptr::copy_nonoverlapping(hash.as_ptr(), dest, HASH_LENGTH);
            }
            MrtVerifyNode::Branch {
                skip,
                left,
                right,
                hash_cache,
                preimage,
                ..
            } => {
                if let Some(hash) = *hash_cache {
                    core::ptr::copy_nonoverlapping(hash.as_ptr(), dest, HASH_LENGTH);
                    return;
                }
                if preimage.is_none() {
                    *preimage = Some(tree::build_branch_preimage(skip));
                }
                // Take the buffer's base pointer and word length, then let the
                // `&mut` borrow of `preimage` end — every access below goes through
                // this one raw provenance (no second reborrow). Children write
                // their hashes directly into the [0..32]/[32..64] slots; the read
                // for hashing reconstructs the slice from the same pointer after
                // those writes complete. The slots live in *this* node's buffer,
                // disjoint from the child allocations the recursion mutates.
                let (pre, words) = {
                    let buf = preimage.as_mut().unwrap();
                    (buf.as_mut_ptr(), buf.len())
                };
                let pre_bytes = pre as *mut u8;
                left.rehash_into(pre_bytes);
                right.rehash_into(pre_bytes.add(HASH_LENGTH));
                // The two child depth_below words at [64..72] are DYNAMIC (a write
                // under a child can change its max-depth), unlike the build-once
                // stable tail [72..]. Rewrite them from the children's current
                // (eager) depth_below on every dirty rehash, alongside the child
                // hashes — caching them once would hash a stale depth after a later
                // write changed a child's max depth. The child borrows above have
                // ended, so these shared reads are sound.
                let l_depth = (left.depth_below() as u32).to_be_bytes();
                let r_depth = (right.depth_below() as u32).to_be_bytes();
                core::ptr::copy_nonoverlapping(l_depth.as_ptr(), pre_bytes.add(64), 4);
                core::ptr::copy_nonoverlapping(r_depth.as_ptr(), pre_bytes.add(68), 4);
                let filled = core::slice::from_raw_parts(pre as *const u32, words);
                tree::hash_branch_preimage_into(dest, filled, skip);
                let mut hash = NULL_HASH;
                core::ptr::copy_nonoverlapping(dest, hash.as_mut_ptr(), HASH_LENGTH);
                *hash_cache = Some(hash);
            }
            MrtVerifyNode::PrunedHash(hash, _) => {
                core::ptr::copy_nonoverlapping(hash.as_ptr(), dest, HASH_LENGTH);
            }
        }
    }
}

fn node_from_arc(node: &Arc<MrtNodeInner>) -> Box<MrtVerifyNode> {
    let hash_cache = Some(node.hash());
    // depth_below comes straight off the (eagerly-maintained) host node — for a
    // branch it is the host branch's own roll-up; for a pruned stub it is the
    // carried value.
    let depth_below = node.depth_below();
    Box::new(match node.node() {
        MrtNode::Leaf { skip, value } => MrtVerifyNode::Leaf {
            skip: skip.clone(),
            value: value.clone(),
            hash_cache,
        },
        MrtNode::Branch { skip, left, right } => MrtVerifyNode::Branch {
            skip: skip.clone(),
            left: node_from_arc(left),
            right: node_from_arc(right),
            hash_cache,
            preimage: None,
            depth_below,
        },
        MrtNode::PrunedHash => MrtVerifyNode::PrunedHash(node.hash(), depth_below),
    })
}

/// The already-computed hash of a freshly decoded verify node (leaf/branch
/// caches are populated by [`decode_trace_node`]; pruned nodes carry it).
fn verify_node_hash(node: &MrtVerifyNode) -> Hash {
    match node {
        MrtVerifyNode::Leaf { hash_cache, .. } | MrtVerifyNode::Branch { hash_cache, .. } => {
            hash_cache.expect("decode_trace populates leaf/branch hash_cache")
        }
        MrtVerifyNode::PrunedHash(hash, _) => *hash,
    }
}

/// Bounds-checked cursor over a flat MRT trace. Every read fails (rather than
/// panics) past the end of the buffer, since the trace is untrusted input.
struct TraceReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> TraceReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if len > self.remaining() {
            return Err(Error::Proof(format!(
                "MRT trace truncated: need {len} more byte(s), {} remaining",
                self.remaining()
            )));
        }
        let out = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_hash(&mut self) -> Result<Hash> {
        let mut hash = NULL_HASH;
        hash.copy_from_slice(self.read_bytes(HASH_LENGTH)?);
        Ok(hash)
    }
}

/// Reads one node (after its tag has been consumed). Tag/layout must match
/// `encode_node` in `trace.rs`; this is the verify-tree twin of
/// `decode_node_with_tag_dyn`, building `MrtVerifyNode`s directly with hashes
/// computed eagerly (so `verify_root` reads cached hashes instead of recomputing).
fn decode_trace_node(
    tag: u8,
    reader: &mut TraceReader<'_>,
    depth: usize,
) -> Result<Box<MrtVerifyNode>> {
    if depth > MAX_MRT_TRACE_DECODE_DEPTH {
        return Err(Error::Proof(format!(
            "MRT trace depth {depth} exceeds limit {MAX_MRT_TRACE_DECODE_DEPTH}"
        )));
    }

    match tag {
        // Leaf: skip(bit_len u32 BE + packed) + value_len(u32 BE) + value.
        0x01 => {
            let skip = read_trace_skip(reader)?;
            let value_len = reader.read_u32()? as usize;
            let value_bytes = reader.read_bytes(value_len)?;
            let hash = tree::compute_leaf_hash(&skip, value_bytes);
            let value = value_bytes.to_vec();
            Ok(Box::new(MrtVerifyNode::Leaf {
                skip,
                value,
                hash_cache: Some(hash),
            }))
        }
        // Branch: skip(bit_len u32 BE + packed) + left node + right node. Each
        // child's depth_below is derived from the (already-decoded) child; the
        // branch's own roll-up is validated (checked) before it is stored.
        0x02 => {
            let skip = read_trace_skip(reader)?;
            let left = read_trace_node(reader, depth + 1)?;
            let right = read_trace_node(reader, depth + 1)?;
            let l_depth = left.depth_below();
            let r_depth = right.depth_below();
            let hash = tree::compute_branch_hash(
                &verify_node_hash(&left),
                &verify_node_hash(&right),
                l_depth,
                r_depth,
                &skip,
            );
            let depth_below = tree::checked_branch_depth_below(skip.bit_len(), l_depth, r_depth)?;
            Ok(Box::new(MrtVerifyNode::Branch {
                skip,
                left,
                right,
                hash_cache: Some(hash),
                preimage: None,
                depth_below,
            }))
        }
        // Pruned: hash + carried depth_below (u32-BE), validated to ≤ MAX_ROUTE_BITS.
        0x03 => {
            let hash = reader.read_hash()?;
            let depth_below = read_trace_depth(reader)?;
            Ok(Box::new(MrtVerifyNode::PrunedHash(hash, depth_below)))
        }
        byte => Err(Error::Proof(format!("MRT trace invalid node tag {byte}"))),
    }
}

/// Reads a node tag and its body. Children are never the empty marker (`0x00`),
/// so an unexpected tag here is rejected by `decode_trace_node`.
fn read_trace_node(reader: &mut TraceReader<'_>, depth: usize) -> Result<Box<MrtVerifyNode>> {
    let tag = reader.read_u8()?;
    decode_trace_node(tag, reader, depth)
}

/// Reads a `RouteBits` framed as `bit_len(u32 BE) + packed`, narrowing the u32
/// to the internal u16 (mirrors `read_skip_dyn` in `trace.rs`).
fn read_trace_skip(reader: &mut TraceReader<'_>) -> Result<RouteBits> {
    let bit_len_u32 = reader.read_u32()?;
    let bit_len = u16::try_from(bit_len_u32).map_err(|_| {
        Error::Proof(format!(
            "MRT trace route bit length {bit_len_u32} exceeds u16"
        ))
    })?;
    let packed_len = (bit_len as usize).div_ceil(8);
    let packed = reader.read_bytes(packed_len)?;
    RouteBits::from_packed(bit_len, packed)
        .map_err(|err| Error::Proof(format!("MRT trace route bits invalid: {err}")))
}

/// Reads a pruned stub's carried `depth_below` (u32-BE on the wire), narrowing to
/// the internal u16 and rejecting a value past `MAX_ROUTE_BITS` (the in-memory
/// invariant). An honest encoder never emits an out-of-range depth.
fn read_trace_depth(reader: &mut TraceReader<'_>) -> Result<u16> {
    let depth_u32 = reader.read_u32()?;
    let depth = u16::try_from(depth_u32)
        .map_err(|_| Error::Proof(format!("MRT trace depth_below {depth_u32} exceeds u16")))?;
    tree::validate_depth_below(depth)
        .map_err(|err| Error::Proof(format!("MRT trace depth_below invalid: {err}")))
}

fn new_leaf(skip: RouteBits, value: Vec<u8>) -> Box<MrtVerifyNode> {
    Box::new(MrtVerifyNode::Leaf {
        skip,
        value,
        hash_cache: None,
    })
}

fn new_branch(
    skip: RouteBits,
    left: Box<MrtVerifyNode>,
    right: Box<MrtVerifyNode>,
) -> Box<MrtVerifyNode> {
    // Eager depth_below roll-up from the (materialized) children — the verifier
    // twin of host `MrtNodeInner::branch`. Saturating like the host: a branch the
    // verifier *builds* during replay is over already-validated nodes.
    let depth_below =
        tree::branch_depth_below(skip.bit_len(), left.depth_below(), right.depth_below());
    Box::new(MrtVerifyNode::Branch {
        skip,
        left,
        right,
        hash_cache: None,
        preimage: None,
        depth_below,
    })
}

#[cfg(test)]
fn replay_root(
    mut root: Option<Box<MrtVerifyNode>>,
    batch: &Batch,
) -> Result<Option<Box<MrtVerifyNode>>> {
    for (key, op) in batch.iter() {
        match op {
            Op::Put(value) => {
                insert(&mut root, key, value.clone())?;
            }
            Op::Delete => {
                delete(&mut root, key)?;
            }
            Op::DeleteRange(end) => {
                delete_range(&mut root, key, end)?;
            }
        }
    }
    Ok(root)
}

fn replay_batch_ops_in_root(root: &mut Option<Box<MrtVerifyNode>>, ops: &[BatchOp]) -> Result<()> {
    for op in ops {
        match op {
            BatchOp::Put { key, value } => {
                insert(root, key, value.clone())?;
            }
            BatchOp::Delete { key } => {
                delete(root, key)?;
            }
            BatchOp::DeleteRange { start, end } => {
                delete_range(root, start, end)?;
            }
        }
    }
    Ok(())
}

/// Inserts `key`→`value`, mutating the tree **in place**. Branches on the
/// descent path are walked through `&mut` — only their cached hash is invalidated,
/// not rebuilt as a fresh `Box` — and `value` is moved straight into the new (or
/// overwritten) leaf rather than re-cloned per level. Only the single
/// split/insertion point allocates. (Was a functional root→leaf path rebuild;
/// see `NEW_ADVICE_MERK.md` lever 1.)
fn insert(root: &mut Option<Box<MrtVerifyNode>>, key: &[u8], value: Vec<u8>) -> Result<()> {
    tree::validate_key_len(key, "verify replay insert")?;
    match root {
        Some(node) => insert_at(node, 0, key, value),
        None => {
            *root = Some(new_leaf(tree::route_bits_of(key), value));
            Ok(())
        }
    }
}

/// Deletes `key`, mutating the tree **in place** (was a functional root→leaf path
/// rebuild). Branches on the not-deleting path keep their cached hash; only branches
/// above an actual deletion drop it, and only a collapse allocates. Returns whether
/// a key was removed.
fn delete(root: &mut Option<Box<MrtVerifyNode>>, key: &[u8]) -> Result<bool> {
    tree::validate_key_len(key, "verify replay delete")?;
    let Some(node) = root.as_deref_mut() else {
        return Ok(false);
    };
    match delete_at(node, 0, key)? {
        DelOutcome::NotFound => Ok(false),
        DelOutcome::Deleted => Ok(true),
        DelOutcome::Emptied => {
            // The root itself was the matching leaf → the tree is now empty.
            *root = None;
            Ok(true)
        }
    }
}

/// Deletes the half-open range `[start, end)`, mutating the tree **in place** (was a
/// functional rebuild). Branches outside the range keep their cached hash; only
/// branches the range actually touches drop it, and only a collapse allocates.
fn delete_range(root: &mut Option<Box<MrtVerifyNode>>, start: &[u8], end: &[u8]) -> Result<()> {
    validate_delete_range_bounds(start, end)?;
    let Some(node) = root.as_deref_mut() else {
        return Ok(());
    };
    match delete_range_at(node, 0, &RoutePrefix::root(), Some(start), Some(end))? {
        RangeOutcome::Kept => Ok(()),
        RangeOutcome::Emptied => {
            *root = None;
            Ok(())
        }
    }
}

/// Deletes every key with byte-prefix `prefix`, mutating in place.
fn delete_prefix(root: &mut Option<Box<MrtVerifyNode>>, prefix: &[u8]) -> Result<()> {
    tree::validate_key_len(prefix, "verify replay delete_prefix")?;
    let Some(node) = root.as_deref_mut() else {
        return Ok(());
    };
    let hi = prefix_successor(prefix);
    match delete_range_at(node, 0, &RoutePrefix::root(), Some(prefix), hi.as_deref())? {
        RangeOutcome::Kept => Ok(()),
        RangeOutcome::Emptied => {
            *root = None;
            Ok(())
        }
    }
}

/// Cheap, non-allocating stand-in moved into `node`'s slot by `mem::replace`
/// while a split takes ownership of the old node's parts. Overwritten before the
/// function returns, so it is never observable.
fn placeholder_node() -> MrtVerifyNode {
    MrtVerifyNode::PrunedHash(NULL_HASH, 0)
}

/// Post-order refresh of a branch whose subtree changed under an **in-place** write
/// (insert / delete / delete_range / move detach / splice): drop the now-stale
/// cached hash **and** recompute the eager `depth_below` roll-up from the
/// (already-updated) children. Both must happen together — unlike the lazy
/// `hash_cache`, `depth_below` is held eagerly, so a stale value would feed a wrong
/// number into this branch's parent hash (and into its own preimage `[64..72]` on
/// the next rehash). A no-op on a non-branch. Mirrors how the host CoW paths get a
/// correct `depth_below` for free by rebuilding each branch via
/// `MrtNodeInner::branch`.
fn refresh_branch_after_write(node: &mut MrtVerifyNode) {
    if let MrtVerifyNode::Branch {
        skip,
        left,
        right,
        hash_cache,
        depth_below,
        ..
    } = node
    {
        *hash_cache = None;
        *depth_below =
            tree::branch_depth_below(skip.bit_len(), left.depth_below(), right.depth_below());
    }
}

/// Inserts `key`→`value` under `node` (reached at `depth`), mutating in place.
///
/// On the descent, a matching branch is left structurally untouched — its skip
/// and both children stay put, only the cached hash is dropped — and we recurse
/// into the one child through its `&mut`. The single allocation point is the
/// split/insert at the bottom (`split_leaf`/`split_branch`). An exact-key match
/// overwrites the leaf's value without disturbing its (bit-identical) suffix.
/// Because keys are assumed prefix-free, an insert that would make one key a
/// prefix of another is rejected with an error (see [`tree::classify_key_vs_skip`]).
fn insert_at(node: &mut MrtVerifyNode, depth: u16, key: &[u8], value: Vec<u8>) -> Result<()> {
    let relation = match &*node {
        MrtVerifyNode::Leaf { skip, .. } | MrtVerifyNode::Branch { skip, .. } => {
            tree::classify_key_vs_skip(skip, key, depth)
        }
        MrtVerifyNode::PrunedHash(..) => {
            return Err(Error::PrunedNode(format!(
                "MRT verify replay insert descent at K={key:?}"
            )));
        }
    };

    let offset = match relation {
        // Same key already present: overwrite the value, keep the (bit-identical)
        // suffix, invalidate the cached hash — no reallocation. At a branch, an
        // exact-length match means the key ends at the branch point, i.e. it is a
        // prefix of the subtree's keys, which is not allowed.
        tree::SkipRelation::Equal => {
            return match node {
                MrtVerifyNode::Leaf {
                    skip,
                    value: slot,
                    hash_cache,
                } => {
                    debug_assert_eq!(*skip, tree::leaf_skip_from_key(key, depth));
                    *slot = value;
                    *hash_cache = None;
                    Ok(())
                }
                MrtVerifyNode::Branch { .. } => Err(tree::prefix_free_violation(key)),
                MrtVerifyNode::PrunedHash(..) => unreachable!("pruned descent returned above"),
            };
        }
        // Key continues past this skip: at a branch, descend into the routed
        // child (skip + children retained), then post-order refresh this branch
        // (drop the cached hash + recompute the eager depth_below); at a leaf it
        // means the leaf's key is a prefix of `key`, which is not allowed.
        tree::SkipRelation::SkipIsPrefix => {
            match node {
                MrtVerifyNode::Branch {
                    skip, left, right, ..
                } => {
                    let branch_depth = checked_replay_branch_depth(depth, skip)?;
                    let side = tree::route_bit_at(key, branch_depth);
                    let child_depth = checked_child_depth(branch_depth)?;
                    let child: &mut MrtVerifyNode = if !side { left } else { right };
                    insert_at(child, child_depth, key, value)?;
                }
                MrtVerifyNode::Leaf { .. } => return Err(tree::prefix_free_violation(key)),
                MrtVerifyNode::PrunedHash(..) => unreachable!("pruned descent returned above"),
            }
            refresh_branch_after_write(node);
            return Ok(());
        }
        // `key` ends within this skip → it is a prefix of an existing key.
        tree::SkipRelation::KeyIsPrefix => return Err(tree::prefix_free_violation(key)),
        // Genuine divergence: split this node into a new branch.
        tree::SkipRelation::Diverge { offset } => offset,
    };

    if matches!(node, MrtVerifyNode::Leaf { .. }) {
        split_leaf(node, depth, offset, key, value)
    } else {
        split_branch(node, depth, offset, key, value)
    }
}

/// Replaces the leaf at `node` (which mismatches `key` at `offset`) with a branch
/// whose children are the re-skipped existing leaf and a new leaf carrying
/// `value`. All fallible/`value`-consuming work happens before the in-place
/// swap, so an early error leaves `node` untouched.
fn split_leaf(
    node: &mut MrtVerifyNode,
    depth: u16,
    offset: u16,
    key: &[u8],
    value: Vec<u8>,
) -> Result<()> {
    let p = checked_add_depth(depth, offset)?;
    let child_depth = checked_child_depth(p)?;
    let branch_skip = RouteBits::from_key_range(key, depth, p);
    let key_bit = tree::route_bit_at(key, p);
    let key_leaf = new_leaf(tree::leaf_skip_from_key(key, child_depth), value);

    let MrtVerifyNode::Leaf {
        skip: leaf_skip,
        value: leaf_value,
        ..
    } = std::mem::replace(node, placeholder_node())
    else {
        unreachable!("split_leaf called on a non-leaf");
    };
    debug_assert_ne!(key_bit, leaf_skip.bit_at(offset));
    // The old leaf's skip is owned here and otherwise dropped, so reuse its buffer
    // for the re-skipped existing leaf instead of allocating a fresh `slice`.
    let existing = new_leaf(leaf_skip.into_suffix(offset + 1), leaf_value);
    let (left, right) = if !key_bit {
        (key_leaf, existing)
    } else {
        (existing, key_leaf)
    };
    // `new_branch` rolls up the eager depth_below from the two children.
    *node = *new_branch(branch_skip, left, right);
    Ok(())
}

/// Replaces the branch at `node` (which mismatches `key` at `offset`) with a new
/// branch: the original branch, re-skipped onto the suffix below the split, and a
/// fresh leaf for `key`. As in [`split_leaf`], all fallible/`value`-consuming
/// work precedes the in-place swap.
fn split_branch(
    node: &mut MrtVerifyNode,
    depth: u16,
    offset: u16,
    key: &[u8],
    value: Vec<u8>,
) -> Result<()> {
    let p = checked_add_depth(depth, offset)?;
    let child_depth = checked_child_depth(p)?;
    let key_bit = tree::route_bit_at(key, p);
    let key_leaf = new_leaf(tree::leaf_skip_from_key(key, child_depth), value);

    let MrtVerifyNode::Branch {
        skip, left, right, ..
    } = std::mem::replace(node, placeholder_node())
    else {
        unreachable!("split_branch called on a non-branch");
    };
    let (prefix, existing_bit, existing_suffix) = skip.split_at_into(offset);
    debug_assert_ne!(key_bit, existing_bit);
    let existing = new_branch(existing_suffix, left, right);
    let (left, right) = if !key_bit {
        (key_leaf, existing)
    } else {
        (existing, key_leaf)
    };
    // `new_branch` rolls up the eager depth_below from the two children.
    *node = *new_branch(prefix, left, right);
    Ok(())
}

/// What a recursive `delete_at` reports to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DelOutcome {
    /// Key absent in this subtree — nothing changed, so the parent must NOT
    /// invalidate its cached hash (matches the old functional non-deleting path).
    NotFound,
    /// Key removed; this node still exists but its cached hash is now stale.
    Deleted,
    /// This node itself is gone — the parent must collapse onto the sibling.
    Emptied,
}

/// Deletes `key` under `node` (reached at `depth`), mutating in place. Mirrors
/// `insert_at`: a non-deleting descent leaves branches structurally untouched; a
/// branch above an actual deletion only drops its cached hash (skip, children, and
/// any preimage stay put); only a collapse allocates (`collapse_in_place`). The
/// matching leaf reports `Emptied`, and its parent collapses onto the survivor.
fn delete_at(node: &mut MrtVerifyNode, depth: u16, key: &[u8]) -> Result<DelOutcome> {
    // Classify against this node (short borrow): route to a child, or return an
    // early outcome for a leaf / mismatch / pruned stub.
    let (side, child_depth) = match node {
        MrtVerifyNode::Leaf { skip, .. } => {
            return Ok(if tree::leaf_matches(skip, key, depth) {
                DelOutcome::Emptied
            } else {
                DelOutcome::NotFound
            });
        }
        MrtVerifyNode::Branch { skip, .. } => {
            if matches!(
                skip.matches_key_at(key, depth),
                MatchResult::Mismatch { .. }
            ) {
                return Ok(DelOutcome::NotFound);
            }
            let branch_depth = checked_replay_branch_depth(depth, skip)?;
            (
                tree::route_bit_at(key, branch_depth),
                checked_child_depth(branch_depth)?,
            )
        }
        MrtVerifyNode::PrunedHash(..) => {
            return Err(Error::PrunedNode(format!(
                "MRT verify replay delete descent at K={key:?}"
            )));
        }
    };

    // Recurse into the routed child; this borrow ends before we touch `node` again.
    let outcome = match node {
        MrtVerifyNode::Branch { left, right, .. } => {
            let child: &mut MrtVerifyNode = if !side { left } else { right };
            delete_at(child, child_depth, key)?
        }
        _ => unreachable!("classified as a branch above"),
    };

    match outcome {
        DelOutcome::NotFound => Ok(DelOutcome::NotFound),
        DelOutcome::Deleted => {
            // A node below changed: drop this branch's stale cached hash AND
            // recompute its eager depth_below (a delete below can shrink the deepest
            // key), keeping skip/children/preimage. Mirrors `insert_at`'s descent.
            refresh_branch_after_write(node);
            Ok(DelOutcome::Deleted)
        }
        DelOutcome::Emptied => {
            // The routed child vanished: collapse this branch onto the survivor.
            collapse_in_place(node, side)?;
            Ok(DelOutcome::Deleted)
        }
    }
}

/// Collapses a branch whose `deleted_side` child was removed onto its surviving
/// child, re-skipped through the parent's skip + the surviving decision bit, and
/// replaces `*node` in place. Reuses `collapse_survivor` verbatim, so the result is
/// byte-identical to the old functional path. (`collapse_survivor` is the only
/// fallible step; an error there aborts the whole verification and the tree — with
/// its parked `placeholder_node()` — is discarded, so the placeholder is unobservable.)
fn collapse_in_place(node: &mut MrtVerifyNode, deleted_side: bool) -> Result<()> {
    let survivor_side = !deleted_side;
    let MrtVerifyNode::Branch {
        skip, left, right, ..
    } = std::mem::replace(node, placeholder_node())
    else {
        unreachable!("collapse_in_place called on a non-branch");
    };
    let survivor = if survivor_side { *right } else { *left };
    *node = *collapse_survivor(&skip, survivor_side, survivor)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundPosition {
    Before,
    After,
    InLeft,
    InRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundEffect<'a> {
    Open,
    Bound(&'a [u8]),
    NoOverlap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildRange<'a> {
    NoOverlap,
    Full,
    Partial {
        lo: Option<&'a [u8]>,
        hi: Option<&'a [u8]>,
    },
}

/// What a recursive `delete_range_at` reports to its parent. Range delete has no
/// "not found" — the lo-After / hi-Before early returns are simply `Kept` unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeOutcome {
    /// This subtree survives (possibly with descendants removed).
    Kept,
    /// This subtree is entirely within the range — gone.
    Emptied,
}

fn delete_range_at<'a>(
    node: &mut MrtVerifyNode,
    depth: u16,
    prefix: &RoutePrefix,
    lo: Option<&'a [u8]>,
    hi: Option<&'a [u8]>,
) -> Result<RangeOutcome> {
    // Whole subtree falls in range (both bounds open): deleted wholesale.
    if lo.is_none() && hi.is_none() {
        return Ok(RangeOutcome::Emptied);
    }

    // Classify against this node (short borrow): a leaf decides Kept/Emptied; a
    // branch handles the lo-After / hi-Before "no overlap" early returns (Kept,
    // untouched — keep the cached hash) and otherwise computes the child ranges.
    let (left_range, right_range, child_depth, left_prefix, right_prefix) = match node {
        MrtVerifyNode::Leaf { skip, .. } => {
            let key = prefix.key_with_suffix(skip)?;
            return Ok(
                if lo.is_some_and(|lo| key.as_slice() < lo)
                    || hi.is_some_and(|hi| key.as_slice() >= hi)
                {
                    RangeOutcome::Kept
                } else {
                    RangeOutcome::Emptied
                },
            );
        }
        MrtVerifyNode::Branch { skip, .. } => {
            let lo_pos = lo.map(|key| classify_bound(key, depth, skip)).transpose()?;
            if lo_pos == Some(BoundPosition::After) {
                return Ok(RangeOutcome::Kept);
            }
            let hi_pos = hi.map(|key| classify_bound(key, depth, skip)).transpose()?;
            if hi_pos == Some(BoundPosition::Before) {
                return Ok(RangeOutcome::Kept);
            }
            let branch_depth = checked_replay_branch_depth(depth, skip)?;
            let child_depth = checked_child_depth(branch_depth)?;
            let left_prefix = prefix.descend(skip, false)?;
            let right_prefix = prefix.descend(skip, true)?;
            let left_range = child_range(
                lower_effect(lo, lo_pos, false),
                upper_effect(hi, hi_pos, false),
            );
            let right_range = child_range(
                lower_effect(lo, lo_pos, true),
                upper_effect(hi, hi_pos, true),
            );
            (
                left_range,
                right_range,
                child_depth,
                left_prefix,
                right_prefix,
            )
        }
        MrtVerifyNode::PrunedHash(..) => {
            return Err(Error::PrunedNode(
                "MRT verify replay delete_range descent".into(),
            ));
        }
    };

    // Recurse into each child in place; these borrows end before we touch `node`.
    let (left_outcome, right_outcome) = match node {
        MrtVerifyNode::Branch { left, right, .. } => {
            let left_outcome = apply_child_range(left, child_depth, &left_prefix, left_range)?;
            let right_outcome = apply_child_range(right, child_depth, &right_prefix, right_range)?;
            (left_outcome, right_outcome)
        }
        _ => unreachable!("classified as a branch above"),
    };

    match (left_outcome, right_outcome) {
        // Both children survive: this branch overlapped the range (it didn't early-
        // return), so its hash is stale — drop the cached hash AND recompute the
        // eager depth_below (descendants may have been deleted under either child),
        // keeping skip/children/preimage. Like `insert_at`'s descent.
        (RangeOutcome::Kept, RangeOutcome::Kept) => {
            refresh_branch_after_write(node);
            Ok(RangeOutcome::Kept)
        }
        // One child emptied: collapse onto the surviving side. `deleted_side` is the
        // emptied side (`collapse_in_place` re-skins the other child).
        (RangeOutcome::Kept, RangeOutcome::Emptied) => {
            collapse_in_place(node, true)?;
            Ok(RangeOutcome::Kept)
        }
        (RangeOutcome::Emptied, RangeOutcome::Kept) => {
            collapse_in_place(node, false)?;
            Ok(RangeOutcome::Kept)
        }
        (RangeOutcome::Emptied, RangeOutcome::Emptied) => Ok(RangeOutcome::Emptied),
    }
}

fn classify_bound(key: &[u8], depth: u16, skip: &RouteBits) -> Result<BoundPosition> {
    for offset in 0..skip.bit_len() {
        let position = checked_add_depth(depth, offset)?;
        let bound_bit = tree::route_bit_at(key, position);
        let skip_bit = skip.bit_at(offset);
        if bound_bit != skip_bit {
            return Ok(if bound_bit {
                BoundPosition::After
            } else {
                BoundPosition::Before
            });
        }
    }

    let branch_depth = checked_replay_branch_depth(depth, skip)?;
    if tree::route_bit_at(key, branch_depth) {
        Ok(BoundPosition::InRight)
    } else {
        Ok(BoundPosition::InLeft)
    }
}

fn lower_effect<'a>(
    lo: Option<&'a [u8]>,
    position: Option<BoundPosition>,
    right_child: bool,
) -> BoundEffect<'a> {
    match (lo, position, right_child) {
        (None, _, _) | (Some(_), Some(BoundPosition::Before), _) => BoundEffect::Open,
        (Some(key), Some(BoundPosition::InLeft), false) => BoundEffect::Bound(key),
        (Some(_), Some(BoundPosition::InLeft), true) => BoundEffect::Open,
        (Some(_), Some(BoundPosition::InRight), false) => BoundEffect::NoOverlap,
        (Some(key), Some(BoundPosition::InRight), true) => BoundEffect::Bound(key),
        (Some(_), Some(BoundPosition::After), _) => BoundEffect::NoOverlap,
        (Some(_), None, _) => unreachable!("lower bound position missing"),
    }
}

fn upper_effect<'a>(
    hi: Option<&'a [u8]>,
    position: Option<BoundPosition>,
    right_child: bool,
) -> BoundEffect<'a> {
    match (hi, position, right_child) {
        (None, _, _) | (Some(_), Some(BoundPosition::After), _) => BoundEffect::Open,
        (Some(key), Some(BoundPosition::InLeft), false) => BoundEffect::Bound(key),
        (Some(_), Some(BoundPosition::InLeft), true) => BoundEffect::NoOverlap,
        (Some(_), Some(BoundPosition::InRight), false) => BoundEffect::Open,
        (Some(key), Some(BoundPosition::InRight), true) => BoundEffect::Bound(key),
        (Some(_), Some(BoundPosition::Before), _) => BoundEffect::NoOverlap,
        (Some(_), None, _) => unreachable!("upper bound position missing"),
    }
}

fn child_range<'a>(lo: BoundEffect<'a>, hi: BoundEffect<'a>) -> ChildRange<'a> {
    match (lo, hi) {
        (BoundEffect::NoOverlap, _) | (_, BoundEffect::NoOverlap) => ChildRange::NoOverlap,
        (BoundEffect::Open, BoundEffect::Open) => ChildRange::Full,
        (BoundEffect::Bound(lo), BoundEffect::Open) => ChildRange::Partial {
            lo: Some(lo),
            hi: None,
        },
        (BoundEffect::Open, BoundEffect::Bound(hi)) => ChildRange::Partial {
            lo: None,
            hi: Some(hi),
        },
        (BoundEffect::Bound(lo), BoundEffect::Bound(hi)) => ChildRange::Partial {
            lo: Some(lo),
            hi: Some(hi),
        },
    }
}

fn apply_child_range<'a>(
    child: &mut MrtVerifyNode,
    child_depth: u16,
    child_prefix: &RoutePrefix,
    range: ChildRange<'a>,
) -> Result<RangeOutcome> {
    match range {
        // Outside the range: untouched, survives (and keeps its cached hash).
        ChildRange::NoOverlap => Ok(RangeOutcome::Kept),
        ChildRange::Full => delete_range_at(child, child_depth, child_prefix, None, None),
        ChildRange::Partial { lo, hi } => delete_range_at(child, child_depth, child_prefix, lo, hi),
    }
}

fn collapse_survivor(
    parent_skip: &RouteBits,
    survivor_side: bool,
    survivor: MrtVerifyNode,
) -> Result<Box<MrtVerifyNode>> {
    match survivor {
        MrtVerifyNode::Branch {
            skip, left, right, ..
        } => {
            let merged = parent_skip.concat_bit_and_skip(survivor_side, &skip)?;
            Ok(new_branch(merged, left, right))
        }
        // A leaf survivor re-skips just like a branch survivor (§5).
        MrtVerifyNode::Leaf { skip, value, .. } => {
            let merged = parent_skip.concat_bit_and_skip(survivor_side, &skip)?;
            Ok(new_leaf(merged, value))
        }
        // A pruned survivor can't be re-skipped; `verify_root` catches the
        // resulting mismatch (never arises in a valid trace).
        other => Ok(Box::new(other)),
    }
}

// ─── move_prefix replay (`detach` + `splice`, mirrors `tree.rs`) ─────────────
//
// The verifier replays the whole move (detach then splice) over its in-place
// `Box<MrtVerifyNode>` tree, structurally identical to the host CoW descent in
// `tree.rs`. The shared *decision* is `tree::classify_key_vs_skip` (host and
// verifier both build their per-node prefix classification on it); the ownership
// plumbing is backend-specific (here: consume-and-rebuild `Box`es, no parent
// stack). The re-skinned `S` root hash is derived from the authenticated original
// (child hash-stubs, or leaf value) — it is never read from the trace.

/// `S` removed from the verify tree, ready to re-skin at the destination prefix.
struct VerifyCaptured {
    s_root: Box<MrtVerifyNode>,
    strip_prefix_bits: u16,
}

/// One step of the prefix-locus descent over the verify tree — the twin of
/// `tree::classify_prefix_step`, decided via the shared `classify_key_vs_skip`.
enum VerifyPrefixStep {
    Stop { strip_prefix_bits: u16 },
    Descend { side: bool, child_depth: u16 },
}

fn move_prefix_absent(p: &[u8]) -> Error {
    Error::Key(format!(
        "MRT verify replay move_prefix source prefix {p:?} is absent"
    ))
}

fn move_prefix_destination_not_empty(q: &[u8]) -> Error {
    Error::Key(format!(
        "MRT verify replay move_prefix destination prefix {q:?} is not empty"
    ))
}

fn classify_prefix_step_verify(
    node: &MrtVerifyNode,
    depth: u16,
    p: &[u8],
) -> Result<VerifyPrefixStep> {
    let p_bits = tree::route_len(p);
    debug_assert!(depth <= p_bits, "prefix-locus descent overran the prefix");
    let stop = VerifyPrefixStep::Stop {
        strip_prefix_bits: p_bits - depth,
    };
    match node {
        MrtVerifyNode::Leaf { skip, .. } => match tree::classify_key_vs_skip(skip, p, depth) {
            tree::SkipRelation::Equal | tree::SkipRelation::KeyIsPrefix => Ok(stop),
            tree::SkipRelation::SkipIsPrefix | tree::SkipRelation::Diverge { .. } => {
                Err(move_prefix_absent(p))
            }
        },
        MrtVerifyNode::Branch { skip, .. } => match tree::classify_key_vs_skip(skip, p, depth) {
            tree::SkipRelation::Equal | tree::SkipRelation::KeyIsPrefix => Ok(stop),
            tree::SkipRelation::SkipIsPrefix => {
                let branch_depth = checked_replay_branch_depth(depth, skip)?;
                let side = tree::route_bit_at(p, branch_depth);
                let child_depth = checked_child_depth(branch_depth)?;
                Ok(VerifyPrefixStep::Descend { side, child_depth })
            }
            tree::SkipRelation::Diverge { .. } => Err(move_prefix_absent(p)),
        },
        MrtVerifyNode::PrunedHash(..) => Err(Error::PrunedNode(format!(
            "MRT verify replay move_prefix detach descent at prefix {p:?}"
        ))),
    }
}

/// What a recursive `detach_at_in_place` reports to its parent.
enum DetachStep {
    /// This node *is* `S`: the parent must remove it and collapse onto the sibling.
    IsS { strip_prefix_bits: u16 },
    /// `S` was removed below; this node still exists but its cached hash is stale.
    Changed,
}

/// In-place source detach for `move_prefix` replay: remove the prefix-`p` subtree
/// `S`, mutating through `&mut` (only the collapse survivor re-skins; ancestors keep
/// their nodes with the cached hash dropped), and return the captured `S` for
/// splicing. `Err` on an absent `p` (incl. an empty tree). Mirrors the in-place
/// `delete_at`, except the emptied subtree is captured out instead of dropped.
fn detach_in_place(root: &mut Option<Box<MrtVerifyNode>>, p: &[u8]) -> Result<VerifyCaptured> {
    let Some(node) = root.as_deref_mut() else {
        return Err(move_prefix_absent(p));
    };
    let mut captured = None;
    match detach_at_in_place(node, 0, p, &mut captured)? {
        // The root itself is `S` (whole-tree move): take it out; the tree empties.
        DetachStep::IsS { strip_prefix_bits } => Ok(VerifyCaptured {
            s_root: root.take().expect("root present above"),
            strip_prefix_bits,
        }),
        // A non-root `S` was captured during the collapse below.
        DetachStep::Changed => {
            Ok(captured.expect("non-root detach must capture S at the collapse"))
        }
    }
}

/// Source-collapse for `move_prefix` replay: like [`collapse_survivor`] but the
/// survivor root **must be materialized**. A `PrunedHash` survivor cannot be
/// re-skipped (its interior skip is hidden), so re-skinning silently keeps a stale
/// hash — a crafted proof could prune the survivor, drive the replay to a bogus
/// post-root, and have its own `expected_end_root` accept it. The trace is
/// required to reveal the source-collapse survivor root (the tracer records it), so
/// a pruned survivor here is a tampered proof and is rejected. (Plain
/// `collapse_survivor`, shared with delete/delete_range replay, is left untouched.)
fn collapse_survivor_materialized(
    parent_skip: &RouteBits,
    survivor_side: bool,
    survivor: MrtVerifyNode,
) -> Result<Box<MrtVerifyNode>> {
    if matches!(survivor, MrtVerifyNode::PrunedHash(..)) {
        return Err(Error::PrunedNode(
            "MRT verify replay move_prefix source-collapse survivor is pruned".into(),
        ));
    }
    collapse_survivor(parent_skip, survivor_side, survivor)
}

/// Recursive in-place detach worker. Descends `p`; when the routed child is `S`,
/// captures it (via `mem::replace`) and collapses this branch onto the surviving
/// sibling (`collapse_survivor_materialized` — a pruned survivor is a tampered
/// proof). The captured `S` is threaded up through `out`; ancestors above the
/// collapse only drop their cached hash.
fn detach_at_in_place(
    node: &mut MrtVerifyNode,
    depth: u16,
    p: &[u8],
    out: &mut Option<VerifyCaptured>,
) -> Result<DetachStep> {
    let (side, child_depth) = match classify_prefix_step_verify(node, depth, p)? {
        VerifyPrefixStep::Stop { strip_prefix_bits } => {
            return Ok(DetachStep::IsS { strip_prefix_bits });
        }
        VerifyPrefixStep::Descend { side, child_depth } => (side, child_depth),
    };

    // Recurse into the routed child; this borrow ends before we touch `node` again.
    let child_outcome = match node {
        MrtVerifyNode::Branch { left, right, .. } => {
            let child: &mut MrtVerifyNode = if !side { left } else { right };
            detach_at_in_place(child, child_depth, p, out)?
        }
        _ => unreachable!("Descend is only returned for a branch node"),
    };

    match child_outcome {
        // The routed child is `S`: capture it and collapse onto the survivor.
        DetachStep::IsS { strip_prefix_bits } => {
            let MrtVerifyNode::Branch {
                skip, left, right, ..
            } = std::mem::replace(node, placeholder_node())
            else {
                unreachable!("Descend classified this as a branch");
            };
            // `S` is on `side`; the survivor is the other child.
            let (s_root, survivor, survivor_side) = if !side {
                (left, *right, true)
            } else {
                (right, *left, false)
            };
            *out = Some(VerifyCaptured {
                s_root,
                strip_prefix_bits,
            });
            *node = *collapse_survivor_materialized(&skip, survivor_side, survivor)?;
            Ok(DetachStep::Changed)
        }
        DetachStep::Changed => {
            // `S` was detached below: drop the stale cached hash AND recompute the
            // eager depth_below (removing `S` can shrink the deepest key here).
            refresh_branch_after_write(node);
            Ok(DetachStep::Changed)
        }
    }
}

/// Re-skin the captured `S` root: strip its source prefix tail, prepend the
/// destination tail, and rebuild the one root node (children/value preserved).
/// `hash_cache: None` (via `new_leaf`/`new_branch`) forces `verify_root` to
/// **recompute** the root hash from the authenticated parts.
fn reskin_captured_root_verify(
    captured: VerifyCaptured,
    prepend_q_tail_bits: &RouteBits,
) -> Result<Box<MrtVerifyNode>> {
    let strip = captured.strip_prefix_bits;
    match *captured.s_root {
        MrtVerifyNode::Leaf { skip, value, .. } => Ok(new_leaf(
            tree::reskin_root(&skip, strip, prepend_q_tail_bits)?,
            value,
        )),
        MrtVerifyNode::Branch {
            skip, left, right, ..
        } => Ok(new_branch(
            tree::reskin_root(&skip, strip, prepend_q_tail_bits)?,
            left,
            right,
        )),
        MrtVerifyNode::PrunedHash(..) => Err(Error::PrunedNode(
            "MRT verify replay move_prefix re-skin of a pruned captured root".into(),
        )),
    }
}

/// In-place destination splice for `move_prefix` replay: insert the captured `S` at
/// prefix `q` in the post-detach tree, mutating through `&mut`. An empty post-detach
/// tree (whole-tree move) makes the re-skinned `S` the new root with no connector.
/// Rejects a non-empty / prefix-violating `q`. Mirrors the in-place `insert_at` split.
fn splice_in_place(
    root: &mut Option<Box<MrtVerifyNode>>,
    q: &[u8],
    captured: VerifyCaptured,
) -> Result<()> {
    tree::validate_key_len(q, "verify replay splice destination prefix")?;
    match root.as_deref_mut() {
        None => {
            let q_tail = RouteBits::from_key_range(q, 0, tree::route_len(q));
            *root = Some(reskin_captured_root_verify(captured, &q_tail)?);
            Ok(())
        }
        Some(node) => splice_at_in_place(node, 0, q, captured),
    }
}

fn splice_at_in_place(
    node: &mut MrtVerifyNode,
    depth: u16,
    q: &[u8],
    captured: VerifyCaptured,
) -> Result<()> {
    let relation = match &*node {
        MrtVerifyNode::Leaf { skip, .. } | MrtVerifyNode::Branch { skip, .. } => {
            tree::classify_key_vs_skip(skip, q, depth)
        }
        MrtVerifyNode::PrunedHash(..) => {
            return Err(Error::PrunedNode(format!(
                "MRT verify replay move_prefix splice descent at prefix {q:?}"
            )));
        }
    };
    match relation {
        // OVERWRITE semantics: `q` lands on/within an existing node, so the
        // destination prefix is occupied. That node is revealed (a `PrunedHash`
        // here is rejected above — a stub carries no skip to classify), so its hash
        // is pinned by the authenticated start root; discard it and put the moved
        // subtree at `q`. Re-skin first so an early error leaves `node` untouched.
        tree::SkipRelation::Equal | tree::SkipRelation::KeyIsPrefix => {
            let q_tail = RouteBits::from_key_range(q, depth, tree::route_len(q));
            *node = *reskin_captured_root_verify(captured, &q_tail)?;
            Ok(())
        }
        // `q` continues past this node: descend a branch, then post-order refresh
        // this branch (drop the cached hash + recompute the eager depth_below — the
        // spliced subtree can deepen it); at a leaf, the existing key is a prefix of
        // `q` (prefix-free violation).
        tree::SkipRelation::SkipIsPrefix => match node {
            MrtVerifyNode::Branch { .. } => {
                let (side, child_depth) = match node {
                    MrtVerifyNode::Branch { skip, .. } => {
                        let branch_depth = checked_replay_branch_depth(depth, skip)?;
                        (
                            tree::route_bit_at(q, branch_depth),
                            checked_child_depth(branch_depth)?,
                        )
                    }
                    _ => unreachable!("matched a branch above"),
                };
                match node {
                    MrtVerifyNode::Branch { left, right, .. } => {
                        let child: &mut MrtVerifyNode = if !side { left } else { right };
                        splice_at_in_place(child, child_depth, q, captured)?;
                    }
                    _ => unreachable!("matched a branch above"),
                }
                refresh_branch_after_write(node);
                Ok(())
            }
            MrtVerifyNode::Leaf { .. } => Err(tree::prefix_free_violation(q)),
            MrtVerifyNode::PrunedHash(..) => unreachable!("pruned handled above"),
        },
        // Real divergence before `q` ends: build the connector in place.
        tree::SkipRelation::Diverge { offset } => {
            build_splice_connector_in_place(node, offset, depth, q, captured)
        }
    }
}

/// Replace `node` (which mismatches `q` at `offset`) **in place** with a connector
/// branch whose children are the re-skipped existing node and the re-skinned moved
/// subtree. All fallible/owning work (the re-skin) happens before the `mem::replace`
/// swap, so an early error leaves `node` untouched.
fn build_splice_connector_in_place(
    node: &mut MrtVerifyNode,
    offset: u16,
    depth: u16,
    q: &[u8],
    captured: VerifyCaptured,
) -> Result<()> {
    let q_div = checked_add_depth(depth, offset)?;
    let q_bits = tree::route_len(q);
    if q_div >= q_bits {
        return Err(move_prefix_destination_not_empty(q));
    }
    let e_dst = checked_child_depth(q_div)?;
    let q_tail = RouteBits::from_key_range(q, e_dst, q_bits);
    let moved = reskin_captured_root_verify(captured, &q_tail)?;
    let q_side = tree::route_bit_at(q, q_div);

    let (connector_skip, existing, existing_side) =
        match std::mem::replace(node, placeholder_node()) {
            MrtVerifyNode::Leaf { skip, value, .. } => {
                let connector_skip = skip.slice(0, offset);
                let existing_side = skip.bit_at(offset);
                let existing = new_leaf(skip.slice(offset + 1, skip.bit_len()), value);
                (connector_skip, existing, existing_side)
            }
            MrtVerifyNode::Branch {
                skip, left, right, ..
            } => {
                let connector_skip = skip.slice(0, offset);
                let existing_side = skip.bit_at(offset);
                let existing = new_branch(skip.slice(offset + 1, skip.bit_len()), left, right);
                (connector_skip, existing, existing_side)
            }
            MrtVerifyNode::PrunedHash(..) => unreachable!("Diverge classified a real node"),
        };
    debug_assert_ne!(q_side, existing_side);
    *node = if !q_side {
        *new_branch(connector_skip, moved, existing)
    } else {
        *new_branch(connector_skip, existing, moved)
    };
    Ok(())
}

fn validate_delete_range_bounds(start: &[u8], end: &[u8]) -> Result<()> {
    tree::validate_key_len(start, "verify replay delete_range start")?;
    tree::validate_key_len(end, "verify replay delete_range end")?;
    if start >= end {
        return Err(Error::Key(format!(
            "MRT verify replay delete_range start key {start:?} must be less than end key {end:?}"
        )));
    }
    Ok(())
}

fn checked_replay_branch_depth(depth: u16, skip: &RouteBits) -> Result<u16> {
    let branch_depth = depth
        .checked_add(skip.bit_len())
        .ok_or_else(|| Error::Tree("MRT verify replay branch depth overflow".into()))?;
    if branch_depth >= tree::MAX_ROUTE_BITS {
        return Err(Error::Tree(format!(
            "MRT verify replay branch depth {branch_depth} exceeds maximum decision bit {}",
            tree::MAX_ROUTE_BITS - 1
        )));
    }
    Ok(branch_depth)
}

fn checked_child_depth(branch_depth: u16) -> Result<u16> {
    branch_depth
        .checked_add(1)
        .ok_or_else(|| Error::Tree("MRT verify replay child depth overflow".into()))
}

fn checked_add_depth(depth: u16, offset: u16) -> Result<u16> {
    depth
        .checked_add(offset)
        .ok_or_else(|| Error::Tree("MRT verify replay route depth overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::{TraceInterface, TraceReader, WriteOp};

    fn build_tree(entries: &[(&[u8], &[u8])]) -> Option<Arc<MrtNodeInner>> {
        let mut root = None;
        for (key, value) in entries {
            root = Some(tree::insert(root, key.to_vec(), value.to_vec()).unwrap());
        }
        root
    }

    #[test]
    fn in_place_replay_and_move_match_atomic() {
        // The clone-free zkVM-loop variants must produce the same result as the atomic
        // (clone-apply-commit) ones. A full tree trace covers every touched node.
        let trace = Trace(build_tree(&[(b"aa", b"1"), (b"ab", b"2"), (b"zz", b"3")]));

        // replay_batch_ops: sorted ops (delete then put, aa < ac).
        let ops = vec![
            BatchOp::Delete {
                key: b"aa".to_vec(),
            },
            BatchOp::Put {
                key: b"ac".to_vec(),
                value: b"9".to_vec(),
            },
        ];
        let mut atomic = TraceVerifier::from_trace(&trace);
        atomic.replay_batch_ops(&ops).unwrap();
        let mut in_place = TraceVerifier::from_trace(&trace);
        in_place.replay_batch_ops_in_place(&ops).unwrap();
        assert_eq!(
            in_place.root_hash().unwrap(),
            atomic.root_hash().unwrap(),
            "in-place batch replay must match atomic"
        );

        // move_prefix: "zz" sits under prefix "z"; move it to the empty "y".
        let mut atomic_m = TraceVerifier::from_trace(&trace);
        atomic_m.move_prefix(b"z", b"y").unwrap();
        let mut in_place_m = TraceVerifier::from_trace(&trace);
        in_place_m.move_prefix_in_place(b"z", b"y").unwrap();
        assert_eq!(
            in_place_m.root_hash().unwrap(),
            atomic_m.root_hash().unwrap(),
            "in-place move_prefix must match atomic"
        );
    }

    #[test]
    fn lazy_tree_hashes_empty_full_and_pruned_traces() {
        let mut empty = MrtVerifyTree::from_trace(&Trace(None));
        assert_eq!(empty.root_hash().unwrap(), NULL_HASH);

        let root = build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]);
        let expected = root.as_ref().unwrap().hash();
        let mut full = MrtVerifyTree::from_trace(&Trace(root));
        assert_eq!(full.root_hash().unwrap(), expected);
        assert_eq!(full.root_hash().unwrap(), expected);

        let left = MrtNodeInner::leaf_value(Vec::new(), b"empty".to_vec());
        let right = MrtNodeInner::pruned([0xee; 32], 0);
        let root = MrtNodeInner::branch(RouteBits::empty(), left, right);
        let expected = root.hash();
        let mut pruned = MrtVerifyTree::from_trace(&Trace(Some(root)));
        assert_eq!(pruned.root_hash().unwrap(), expected);
    }

    #[test]
    fn lazy_tree_reads_match_trace_reads() {
        let trace = Trace(build_tree(&[
            (b"aa", b"11"),
            (b"ab", b"12"),
            (b"bb", b"2"),
            (b"cc", b"3"),
        ]));
        let lazy = MrtVerifyTree::from_trace(&trace);

        assert_eq!(lazy.get(b"bb").unwrap(), trace.get(b"bb").unwrap());
        assert_eq!(lazy.get(b"zz").unwrap(), trace.get(b"zz").unwrap());
        assert_eq!(
            lazy.collect_range(b"ab", Some(b"cc")).unwrap(),
            trace.collect_range(b"ab", Some(b"cc")).unwrap()
        );
        assert_eq!(
            lazy.collect_prefix(b"a").unwrap(),
            trace.collect_prefix(b"a").unwrap()
        );
    }

    #[test]
    fn lazy_tree_reads_fail_on_pruned_descent() {
        let left = MrtNodeInner::leaf(RouteBits::from_key_range(&[0x00], 1, 8), b"v".to_vec());
        let right = MrtNodeInner::pruned([0xee; 32], 0);
        let pruned = MrtVerifyTree::from_trace(&Trace(Some(MrtNodeInner::branch(
            RouteBits::empty(),
            left,
            right,
        ))));
        // [0x80] routes into the pruned right subtree.
        let err = pruned.get(&[0x80]).unwrap_err();
        assert!(matches!(err, Error::PrunedNode(_)), "got {:?}", err);
    }

    #[test]
    fn lazy_replay_matches_trace_replay_for_point_and_range_writes() {
        let trace = Trace(build_tree(&[
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
            (b"e", b"5"),
        ]));
        let batch = vec![
            (b"b".to_vec(), Op::Put(b"updated".to_vec())),
            (b"c".to_vec(), Op::DeleteRange(b"e".to_vec())),
            (b"f".to_vec(), Op::Put(b"6".to_vec())),
        ];

        let eager = super::super::trace::replay(trace.clone(), &batch).unwrap();
        let mut lazy = MrtVerifyTree::from_trace(&trace);
        lazy.replay(&batch).unwrap();

        assert_eq!(lazy.root_hash().unwrap(), eager.root_hash());
        assert_eq!(
            lazy.collect_range(b"a", Some(b"z")).unwrap(),
            eager.collect_range(b"a", Some(b"z")).unwrap()
        );
    }

    #[test]
    fn pruned_tree_wrapper_replays_batch_ops() {
        let trace = Trace(build_tree(&[
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
        ]));
        let start_root = trace.root_hash();
        let ops = vec![
            BatchOp::Put {
                key: b"b".to_vec(),
                value: b"updated".to_vec(),
            },
            BatchOp::DeleteRange {
                start: b"c".to_vec(),
                end: b"e".to_vec(),
            },
            BatchOp::Put {
                key: b"f".to_vec(),
                value: b"6".to_vec(),
            },
        ];
        let batch: Vec<BatchEntry> = ops.iter().map(BatchOp::to_batch_entry).collect();
        let eager = super::super::trace::replay(trace.clone(), &batch).unwrap();

        let mut pruned = TraceVerifier::from_trace(&trace);
        pruned.verify_root(start_root).unwrap();
        assert_eq!(pruned.get(b"b").unwrap(), Some(b"2".to_vec()));
        pruned.replay_batch_ops(&ops).unwrap();
        pruned.verify_root(eager.root_hash()).unwrap();
        assert_eq!(
            pruned.collect_range(b"a", Some(b"z")).unwrap(),
            eager.collect_range(b"a", Some(b"z")).unwrap()
        );
    }

    #[test]
    fn pruned_tree_wrapper_rejects_boundary_incomplete_range() {
        let left = MrtNodeInner::leaf(RouteBits::from_key_range(&[0x00], 1, 8), b"lo".to_vec());
        let right = MrtNodeInner::pruned([0xee; 32], 0);
        let root = MrtNodeInner::branch(RouteBits::empty(), left, right);
        let pruned = TraceVerifier::from_trace(&Trace(Some(root)));

        // The forward scan yields the left leaf, then hits the pruned right child.
        let err = pruned.collect_range(&[0x00], None).unwrap_err();
        assert!(matches!(err, Error::PrunedNode(_)), "got {:?}", err);
    }

    #[test]
    fn lazy_replay_matches_host_move_prefix() {
        // Deep subtree relocate: the verifier in-place detach+splice must produce
        // the same root as the host CoW `tree::move_prefix` (re-skin hash derived,
        // then recomputed by `rehash_into` — the Miri-covered path).
        let entries: &[(&[u8], &[u8])] = &[
            (b"sys:log", b"3"),
            (b"user:alex", b"2"),
            (b"user:alice", b"1"),
            (b"user:bob", b"4"),
            (b"zzz", b"5"),
        ];
        let trace = Trace(build_tree(entries));
        let host = tree::move_prefix(trace.0.clone(), b"user:", b"acct:").unwrap();
        let mut lazy = MrtVerifyTree::from_trace(&trace);
        lazy.move_prefix(b"user:", b"acct:").unwrap();
        assert_eq!(lazy.root_hash().unwrap(), host.hash());

        // Whole-tree single-key move (empty destination, no connector, leaf root).
        let single = Trace(build_tree(&[(b"only", b"v")]));
        let host_single = tree::move_prefix(single.0.clone(), b"only", b"next").unwrap();
        let mut lazy_single = MrtVerifyTree::from_trace(&single);
        lazy_single.move_prefix(b"only", b"next").unwrap();
        assert_eq!(lazy_single.root_hash().unwrap(), host_single.hash());

        // Length-changing move: verifier matches host as long as the deepest
        // resulting moved key stays within the cap.
        let host_shorter = tree::move_prefix(trace.0.clone(), b"user:", b"acct").unwrap();
        let mut lazy_shorter = MrtVerifyTree::from_trace(&trace);
        lazy_shorter.move_prefix(b"user:", b"acct").unwrap();
        assert_eq!(lazy_shorter.root_hash().unwrap(), host_shorter.hash());

        // Overwrite: destination `sys:l` is occupied (by `sys:log`), which is
        // discarded; host CoW and verifier in-place agree on the overwritten root.
        let host_ow = tree::move_prefix(trace.0.clone(), b"user:", b"sys:l").unwrap();
        let mut lazy_ow = MrtVerifyTree::from_trace(&trace);
        lazy_ow.move_prefix(b"user:", b"sys:l").unwrap();
        assert_eq!(lazy_ow.root_hash().unwrap(), host_ow.hash());

        // Rejections: absent source, equal prefixes.
        assert!(MrtVerifyTree::from_trace(&trace)
            .move_prefix(b"nope", b"yarp")
            .is_err());
        assert!(MrtVerifyTree::from_trace(&trace)
            .move_prefix(b"user:", b"user:")
            .is_err());
    }

    #[test]
    fn pruned_tree_move_prefix_rejection_is_atomic() {
        // The public `TraceVerifier::move_prefix` clones-applies-commits, so a move
        // rejected *after* the detach (here a prefix-free violation at the splice)
        // must leave the tree intact — unlike the in-place internal path, which can
        // strand the tree with the source removed. (The internal path is fine only
        // because the verifier aborts on error.)
        let entries: &[(&[u8], &[u8])] = &[
            (b"user:alex", b"2"),
            (b"user:bob", b"4"),
            (b"sys:log", b"3"),
            (b"zzz", b"5"),
        ];
        let trace = Trace(build_tree(entries));
        let root = trace.0.as_ref().unwrap().hash();
        let mut pruned = TraceVerifier::from_trace(&trace);

        // "user:" -> "sys:logx" passes the stateless check but is rejected at the
        // splice with a prefix-free violation: the existing key "sys:log" is a
        // byte-prefix of the destination. (Occupied destinations now overwrite, so
        // the reject-after-detach path is exercised via a prefix conflict instead.)
        assert!(pruned.move_prefix(b"user:", b"sys:logx").is_err());

        // Atomic: still authenticates against the original root and reads unchanged.
        pruned.verify_root(root).unwrap();
        assert_eq!(pruned.get(b"user:alex").unwrap(), Some(b"2".to_vec()));
        assert_eq!(pruned.get(b"zzz").unwrap(), Some(b"5".to_vec()));
    }

    #[test]
    fn move_prefix_rejects_pruned_collapse_survivor() {
        // Move "ma": S is the "ma" leaf, so the {ma,mb} branch collapses onto its
        // "mb" survivor. The destination "qz" routes to the *right* (y-side), so the
        // splice never descends through the re-skinned survivor — i.e. the existing
        // pruned-descent guard in `splice_at_in_place` does NOT cover this. The only
        // line of defense is the collapse rejecting a pruned survivor; without it,
        // the survivor would be kept with its stale (un-re-skipped) hash and the
        // replay would produce a bogus root that a crafted proof's `expected_end_root`
        // could match.
        let full =
            build_tree(&[(b"ma", b"1"), (b"mb", b"2"), (b"ya", b"3"), (b"yb", b"4")]).unwrap();
        let MrtNode::Branch { skip, left, right } = full.node() else {
            panic!("expected a branch root");
        };
        // root.left is the {ma,mb} branch; its right child is the "mb" survivor.
        let MrtNode::Branch {
            skip: lskip,
            left: ma_child,
            right: mb_child,
        } = left.node()
        else {
            panic!("expected an ma/mb branch");
        };
        let tampered_left = MrtNodeInner::branch(
            lskip.clone(),
            ma_child.clone(),
            // survivor pruned — carry its real depth so the start root still
            // authenticates (the tamper is the missing materialization, not depth).
            MrtNodeInner::pruned(mb_child.hash(), mb_child.depth_below()),
        );
        let tampered = Trace(Some(MrtNodeInner::branch(
            skip.clone(),
            tampered_left,
            right.clone(),
        )));
        let mut verify = MrtVerifyTree::from_trace(&tampered);
        verify.verify_root(full.hash()).unwrap(); // start root still authenticates
        assert!(matches!(
            verify.move_prefix(b"ma", b"qz"),
            Err(Error::PrunedNode(_))
        ));

        // With the survivor materialized, the same move succeeds and matches the
        // host op — so the rejection is specific to the missing survivor.
        let host = tree::move_prefix(Some(full.clone()), b"ma", b"qz").unwrap();
        let mut materialized = MrtVerifyTree::from_trace(&Trace(Some(full)));
        materialized.move_prefix(b"ma", b"qz").unwrap();
        assert_eq!(materialized.root_hash().unwrap(), host.hash());
    }

    #[test]
    fn move_prefix_overflow_check_depth_is_authenticated() {
        // The lengthening overflow check reads `depth_below` off the captured root —
        // so soundness rests on that value being *authenticated*, not advisory. This
        // proves it: a trace that **understates** a committed child `depth_below`
        // (claiming a deep hidden subtree is shallow, to sneak an overflowing
        // lengthening past the O(1) check) changes the committing branch's hash and
        // so no longer authenticates against the honest start root.
        //
        // Tree: under prefix "m", a 1-byte-suffix leaf ("ma") and a MAX_KEY_LEN-deep
        // leaf ("mb"…). Lengthening "m" -> "mx" pushes the deep key to
        // MAX_KEY_LEN + 1 bytes, so an honest verifier rejects it on the real,
        // authenticated depth.
        let deep_key = {
            let mut k = vec![b'm', b'b'];
            k.resize(tree::MAX_KEY_LEN, 0x00);
            k
        };
        let mut root = None;
        for (key, value) in [
            (b"ma".to_vec(), b"1".to_vec()),
            (deep_key, b"2".to_vec()),
            (b"zzz".to_vec(), b"3".to_vec()),
        ] {
            root = Some(tree::insert(root, key, value).unwrap());
        }
        let full = root.unwrap();
        let honest_root = full.hash();

        // Honest path: authenticates against the real root, and the move is rejected
        // by the overflow check (the deep key, read via the committed depth, would
        // overflow MAX_KEY_LEN once lengthened).
        let mut honest = MrtVerifyTree::from_trace(&Trace(Some(full.clone())));
        honest.verify_root(honest_root).unwrap();
        assert!(
            matches!(honest.move_prefix(b"m", b"mx"), Err(Error::Key(_))),
            "honest lengthening over a deep subtree must be rejected as overflowing"
        );

        // Forge the trace: prune the deep "mb"… child and carry an *understated*
        // depth (claim a 1-bit-deep leaf). The branch over "m" rolls this up, so the
        // captured root now reports a shallow `depth_below` that the overflow check
        // would wave through.
        let MrtNode::Branch {
            skip: root_skip,
            left: m_side,
            right: z_side,
        } = full.node()
        else {
            panic!("expected an m/z branch root");
        };
        let MrtNode::Branch {
            skip: s_skip,
            left: ma_leaf,
            right: deep_leaf,
        } = m_side.node()
        else {
            panic!("expected an ma/deep branch under \"m\"");
        };
        let understated_depth = 1u16;
        assert!(
            deep_leaf.depth_below() > understated_depth,
            "the deep child must really be deep for the understatement to matter"
        );
        let forged_m_side = MrtNodeInner::branch(
            s_skip.clone(),
            ma_leaf.clone(),
            MrtNodeInner::pruned(deep_leaf.hash(), understated_depth),
        );
        let forged_full = MrtNodeInner::branch(root_skip.clone(), forged_m_side, z_side.clone());
        let forged_root = forged_full.hash();

        // The forged depth changed the committing branch's hash → a different root.
        assert_ne!(honest_root, forged_root);

        // Authentication catches it: the forged trace does NOT hash to the honest
        // committed start root. This is the whole soundness argument — you cannot
        // both understate the depth and match the authenticated root.
        let mut forged_vs_honest = MrtVerifyTree::from_trace(&Trace(Some(forged_full.clone())));
        assert!(
            forged_vs_honest.verify_root(honest_root).is_err(),
            "forged understated depth must break authentication against the honest root"
        );

        // And the understatement *is* what gates the move: taken on its own (forged)
        // root, the lengthening now slips through the overflow check. So nothing but
        // the start-root authentication stands between the forgery and an overflowing
        // move — i.e. `depth_below` is load-bearing and must be authenticated.
        let mut forged = MrtVerifyTree::from_trace(&Trace(Some(forged_full)));
        forged.verify_root(forged_root).unwrap();
        forged
            .move_prefix(b"m", b"mx")
            .expect("understated depth lets the overflowing lengthening pass the O(1) check");
    }

    #[test]
    fn lazy_replay_clears_collapsed_branch_hash_cache() {
        let trace = Trace(build_tree(&[
            (b"a", b"1"),
            (b"b", b"2"),
            (b"c", b"3"),
            (b"d", b"4"),
        ]));
        let batch = vec![(b"a".to_vec(), Op::Delete)];
        let expected = Trace(build_tree(&[(b"b", b"2"), (b"c", b"3"), (b"d", b"4")]));

        let mut lazy = MrtVerifyTree::from_trace(&trace);
        lazy.replay(&batch).unwrap();

        assert_eq!(lazy.root_hash().unwrap(), expected.root_hash());
    }

    /// A decoded branch carries the same eager `depth_below` the host computes —
    /// pinned explicitly (the round-trip tests pin it only implicitly, via root
    /// re-derivation).
    #[test]
    fn decode_trace_carries_depth_below() {
        let root = build_tree(&[(b"aa", b"1"), (b"ab", b"2"), (b"bc", b"3")]).unwrap();
        let mut bytes = Vec::new();
        ed::Encode::encode_into(&Trace(Some(root.clone())), &mut bytes).unwrap();

        let tree = MrtVerifyTree::decode_trace(&bytes).unwrap();
        let decoded_root = tree.root.as_deref().expect("non-empty");
        assert!(matches!(decoded_root, MrtVerifyNode::Branch { .. }));
        // Same as the host's eager roll-up — a real branch depth (skip+1+max),
        // not merely a leaf skip.
        assert_eq!(decoded_root.depth_below(), root.depth_below());
    }

    /// Stale-preimage regression: once a branch's persistent preimage is built, a
    /// later write that changes the MAX depth under it must still hash to the host's
    /// new root — i.e. `rehash_into` rewrites the `[64..72]` child-depth slots on
    /// every rehash, not just at build. A preimage that cached the depth words once
    /// would hash a stale depth here. Exercised in **both** directions: deepen
    /// (insert a deeper key) and shallow (delete the deepest key), with a
    /// `verify_root` between writes so the preimage is reused, not freshly built.
    #[test]
    fn stale_preimage_rewritten_on_rehash_both_directions() {
        let start = Trace(build_tree(&[
            (b"aaaa", b"1"),
            (b"aaab", b"2"),
            (b"bbbb", b"3"),
        ]));
        let mut v = MrtVerifyTree::from_trace(&start);
        v.verify_root(start.root_hash()).unwrap();

        // w1: deepen under the root's "a…" subtree (diverges at byte 3, so no prefix
        // violation). The following verify_root BUILDS the dirtied branches' preimages.
        let w1 = vec![(b"aaac1234".to_vec(), Op::Put(b"x".to_vec()))];
        v.replay(&w1).unwrap();
        let host1 = super::super::trace::replay(start.clone(), &w1).unwrap();
        v.verify_root(host1.root_hash()).unwrap();

        // w2 (deepen further): REUSES those preimages — the [64..72] depth words must
        // be rewritten or the root would be stale-wrong.
        let w2 = vec![(b"aaad12345678".to_vec(), Op::Put(b"y".to_vec()))];
        v.replay(&w2).unwrap();
        let host2 = super::super::trace::replay(host1.clone(), &w2).unwrap();
        v.verify_root(host2.root_hash()).unwrap();

        // w3 (shallow): delete the deepest key, shrinking the max depth under the
        // same (preimage-cached) branches.
        let w3 = vec![(b"aaad12345678".to_vec(), Op::Delete)];
        v.replay(&w3).unwrap();
        let host3 = super::super::trace::replay(host2.clone(), &w3).unwrap();
        v.verify_root(host3.root_hash()).unwrap();
    }

    // ── Stage 4: TraceReplayer root binding and reads ────────────────────────

    /// Encoded bytes + start root for a fully-revealed trace of `{a, c, e}`.
    fn full_trace_bytes_and_root() -> (Vec<u8>, Hash) {
        let trace = Trace(build_tree(&[(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]));
        (trace.encode().unwrap(), trace.root_hash())
    }

    /// A narrow-reveal replayer: the trace proves only the path to `aaa` over
    /// `{aaa, mmm, zzz}`, pruning everything else. Reads outside the revealed
    /// region must fail closed.
    fn narrow_reveal_replayer() -> TraceReplayer {
        let tree = crate::mrt::Tree::new();
        tree.put(b"aaa".to_vec(), b"1".to_vec()).unwrap();
        tree.put(b"mmm".to_vec(), b"2".to_vec()).unwrap();
        tree.put(b"zzz".to_vec(), b"3".to_vec()).unwrap();
        let snapshot = tree.checkpoint();
        let steps = vec![crate::tracer::test_support::Step::Read(vec![
            crate::tracer::ReadOp::Key(b"aaa".to_vec()),
        ])];
        let trace = crate::tracer::test_support::mrt::create_trace(&snapshot, &steps).unwrap();
        let start = trace.root_hash();
        let bytes = trace.encode().unwrap();
        TraceReplayer::new_verified(&bytes, start).unwrap()
    }

    #[test]
    fn replayer_new_verified_accepts_honest_trace_and_reads() {
        use crate::tracer::TraceReader;
        let (bytes, start) = full_trace_bytes_and_root();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
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
    fn replayer_new_verified_rejects_wrong_root_before_reads() {
        let (bytes, start) = full_trace_bytes_and_root();
        // Honest bytes, wrong expected start root: construction fails closed.
        assert!(matches!(
            TraceReplayer::new_verified(&bytes, [9; HASH_LENGTH]),
            Err(Error::HashMismatch(_, _))
        ));
        // The same bytes bound to the correct root succeed.
        assert!(TraceReplayer::new_verified(&bytes, start).is_ok());
    }

    #[test]
    fn replayer_new_unverified_root_matches_decode_trace() {
        let (bytes, start) = full_trace_bytes_and_root();
        let mut replayer = TraceReplayer::new_unverified(&bytes).unwrap();
        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        assert_eq!(replayer.root_hash().unwrap(), verifier.root_hash().unwrap());
        assert_eq!(replayer.root_hash().unwrap(), start);
    }

    #[test]
    fn replayer_reads_match_verifier_reads() {
        use crate::tracer::TraceReader;
        let (bytes, start) = full_trace_bytes_and_root();
        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        verifier.verify_root(start).unwrap();
        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        assert_eq!(replayer.get(b"c").unwrap(), verifier.get(b"c").unwrap());
        assert_eq!(
            replayer.get_range(b"a", b"z").unwrap(),
            verifier.collect_range(b"a", Some(b"z")).unwrap()
        );
        assert_eq!(
            replayer.get_prefix(b"a").unwrap(),
            verifier.collect_prefix(b"a").unwrap()
        );
    }

    #[test]
    fn replayer_get_absent_pruned_path_fails_closed() {
        use crate::tracer::TraceReader;
        let mut replayer = narrow_reveal_replayer();
        // `zzy` is absent and its non-membership path was never revealed: must be
        // `PrunedNode`, not `Ok(None)`.
        assert!(matches!(replayer.get(b"zzy"), Err(Error::PrunedNode(_))));
    }

    #[test]
    fn replayer_range_interior_pruned_fails_closed() {
        use crate::tracer::TraceReader;
        let mut replayer = narrow_reveal_replayer();
        // Must be `PrunedNode`, not a silently truncated list.
        assert!(matches!(
            replayer.get_range(b"aaa", b"zzz"),
            Err(Error::PrunedNode(_))
        ));
    }

    #[test]
    fn replayer_range_boundary_pruned_fails_closed() {
        use crate::tracer::TraceReader;
        let mut replayer = narrow_reveal_replayer();
        assert!(matches!(
            replayer.get_range(b"mmm", b"zzz"),
            Err(Error::PrunedNode(_))
        ));
    }

    #[test]
    fn replayer_prefix_pruned_fails_closed() {
        use crate::tracer::TraceReader;
        let mut replayer = narrow_reveal_replayer();
        assert!(matches!(
            replayer.get_prefix(b"z"),
            Err(Error::PrunedNode(_))
        ));
    }

    // ── Stage 5: TraceReplayer sequenced writes ─────────────────────────────

    fn replayer_for_entries(entries: &[(&[u8], &[u8])]) -> (TraceReplayer, Hash) {
        let trace = Trace(build_tree(entries));
        let start = trace.root_hash();
        let bytes = trace.encode().unwrap();
        (TraceReplayer::new_verified(&bytes, start).unwrap(), start)
    }

    fn live_tree(entries: &[(&[u8], &[u8])]) -> crate::mrt::Tree {
        let tree = crate::mrt::Tree::new();
        for (key, value) in entries {
            tree.put((*key).to_vec(), (*value).to_vec()).unwrap();
        }
        tree
    }

    #[test]
    fn replayer_sequenced_writes_match_old_verifier_replay() {
        let entries = &[
            (b"aa".as_slice(), b"1".as_slice()),
            (b"cc".as_slice(), b"3".as_slice()),
            (b"ee".as_slice(), b"5".as_slice()),
            (b"gg".as_slice(), b"7".as_slice()),
        ];
        let trace = Trace(build_tree(entries));
        let start = trace.root_hash();
        let bytes = trace.encode().unwrap();
        let steps = vec![
            vec![BatchOp::Put {
                key: b"cc".to_vec(),
                value: b"updated".to_vec(),
            }],
            vec![BatchOp::Delete {
                key: b"aa".to_vec(),
            }],
            vec![BatchOp::DeleteRange {
                start: b"ee".to_vec(),
                end: b"hh".to_vec(),
            }],
        ];

        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        verifier.verify_root(start).unwrap();
        for ops in &steps {
            verifier.replay_batch_ops_in_place(ops).unwrap();
        }

        let live = live_tree(entries);
        live.put(b"cc".to_vec(), b"updated".to_vec()).unwrap();
        live.delete(b"aa".to_vec()).unwrap();
        live.delete_range(b"ee".to_vec(), b"hh".to_vec()).unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.put(b"cc", b"updated").unwrap();
        replayer.delete(b"aa").unwrap();
        replayer.delete_range(b"ee", b"hh").unwrap();

        assert_eq!(replayer.root_hash().unwrap(), verifier.root_hash().unwrap());
        assert_eq!(replayer.root_hash().unwrap(), live.root_hash());
    }

    #[test]
    fn replayer_direct_put_delete_preserves_duplicate_key_order() {
        let (mut replayer, _start) = replayer_for_entries(&[]);
        let live = live_tree(&[]);

        replayer.put(b"kk", b"first").unwrap();
        live.put(b"kk".to_vec(), b"first".to_vec()).unwrap();
        replayer.delete(b"kk").unwrap();
        live.delete(b"kk".to_vec()).unwrap();
        replayer.put(b"kk", b"second").unwrap();
        live.put(b"kk".to_vec(), b"second".to_vec()).unwrap();
        replayer.put(b"gone", b"temp").unwrap();
        live.put(b"gone".to_vec(), b"temp".to_vec()).unwrap();
        replayer.delete(b"gone").unwrap();
        live.delete(b"gone".to_vec()).unwrap();

        assert_eq!(replayer.get(b"kk").unwrap(), Some(b"second".to_vec()));
        assert_eq!(replayer.get(b"gone").unwrap(), None);
        assert_eq!(replayer.root_hash().unwrap(), live.root_hash());
    }

    #[test]
    fn replayer_delete_range_and_prefix_preserve_call_order() {
        let entries = &[
            (b"aa".as_slice(), b"1".as_slice()),
            (b"ba".as_slice(), b"old".as_slice()),
            (b"bb".as_slice(), b"2".as_slice()),
            (b"ca".as_slice(), b"3".as_slice()),
            (b"da".as_slice(), b"4".as_slice()),
        ];

        let (mut range_first, _) = replayer_for_entries(entries);
        let range_first_live = live_tree(entries);
        range_first.delete_range(b"b", b"d").unwrap();
        range_first_live
            .delete_range(b"b".to_vec(), b"d".to_vec())
            .unwrap();
        range_first.put(b"ca", b"new").unwrap();
        range_first_live
            .put(b"ca".to_vec(), b"new".to_vec())
            .unwrap();

        let (mut range_last, _) = replayer_for_entries(entries);
        let range_last_live = live_tree(entries);
        range_last.put(b"ca", b"new").unwrap();
        range_last_live
            .put(b"ca".to_vec(), b"new".to_vec())
            .unwrap();
        range_last.delete_range(b"b", b"d").unwrap();
        range_last_live
            .delete_range(b"b".to_vec(), b"d".to_vec())
            .unwrap();

        let range_first_root = range_first.root_hash().unwrap();
        let range_last_root = range_last.root_hash().unwrap();
        assert_eq!(range_first_root, range_first_live.root_hash());
        assert_eq!(range_last_root, range_last_live.root_hash());
        assert_ne!(range_first_root, range_last_root);

        let (mut prefix_first, _) = replayer_for_entries(entries);
        let prefix_first_live = live_tree(entries);
        prefix_first.delete_prefix(b"b").unwrap();
        prefix_first_live
            .delete_range(b"b".to_vec(), b"c".to_vec())
            .unwrap();
        prefix_first.put(b"bc", b"new").unwrap();
        prefix_first_live
            .put(b"bc".to_vec(), b"new".to_vec())
            .unwrap();

        let (mut prefix_last, _) = replayer_for_entries(entries);
        let prefix_last_live = live_tree(entries);
        prefix_last.put(b"bc", b"new").unwrap();
        prefix_last_live
            .put(b"bc".to_vec(), b"new".to_vec())
            .unwrap();
        prefix_last.delete_prefix(b"b").unwrap();
        prefix_last_live
            .delete_range(b"b".to_vec(), b"c".to_vec())
            .unwrap();

        let prefix_first_root = prefix_first.root_hash().unwrap();
        let prefix_last_root = prefix_last.root_hash().unwrap();
        assert_eq!(prefix_first_root, prefix_first_live.root_hash());
        assert_eq!(prefix_last_root, prefix_last_live.root_hash());
        assert_ne!(prefix_first_root, prefix_last_root);
    }

    #[test]
    fn replayer_delete_prefix_handles_no_successor_prefix() {
        let entries = &[
            (&[0xfe][..], b"before".as_slice()),
            (&[0xff, 0x00][..], b"first".as_slice()),
            (&[0xff, 0x10][..], b"second".as_slice()),
        ];
        let (mut replayer, _) = replayer_for_entries(entries);
        replayer.delete_prefix(&[0xff]).unwrap();

        let expected = Trace(build_tree(&[(&[0xfe][..], b"before".as_slice())]));
        assert_eq!(replayer.root_hash().unwrap(), expected.root_hash());
    }

    #[test]
    fn replayer_move_prefix_matches_old_verifier_and_live_tree() {
        let entries = &[
            (b"user:1".as_slice(), b"1".as_slice()),
            (b"user:2".as_slice(), b"2".as_slice()),
            (b"zzzz:1".as_slice(), b"z".as_slice()),
        ];
        let trace = Trace(build_tree(entries));
        let start = trace.root_hash();
        let bytes = trace.encode().unwrap();

        let mut verifier = TraceVerifier::decode_trace(&bytes).unwrap();
        verifier.verify_root(start).unwrap();
        verifier.move_prefix_in_place(b"user:", b"acct:").unwrap();

        let live = live_tree(entries);
        live.move_prefix(b"user:".to_vec(), b"acct:".to_vec())
            .unwrap();

        let mut replayer = TraceReplayer::new_verified(&bytes, start).unwrap();
        replayer.move_prefix(b"user:", b"acct:").unwrap();

        assert_eq!(replayer.root_hash().unwrap(), verifier.root_hash().unwrap());
        assert_eq!(replayer.root_hash().unwrap(), live.root_hash());
    }

    #[test]
    fn replayer_write_against_pruned_witness_returns_pruned_node() {
        let mut replayer = narrow_reveal_replayer();
        assert!(matches!(
            replayer.put(b"zzy", b"x"),
            Err(Error::PrunedNode(_))
        ));
    }

    // ── Stage 3: WriteOp `apply` poison ─────────────────────────────────────

    #[test]
    fn apply_failure_poisons_mrt_replayer() {
        // A fully-revealed witness so the first op can apply.
        let (mut replayer, _start) =
            replayer_for_entries(&[(b"user:a", b"1"), (b"acct:x", b"9"), (b"zz", b"26")]);

        // The Put applies in place, then an invalid MovePrefix (equal
        // source/destination) fails: the change is rejected and the replayer
        // poisoned (the in-place replay may have left the tree mutated).
        let err = replayer
            .apply(&[
                WriteOp::Put {
                    key: b"acct:y".to_vec(),
                    value: b"a".to_vec(),
                },
                WriteOp::MovePrefix {
                    from: b"user:".to_vec(),
                    to: b"user:".to_vec(),
                },
            ])
            .unwrap_err();
        // The first failure is the op's own error (equal prefixes), not `Poisoned`.
        assert!(matches!(err, Error::Key(_)));

        // Every fallible op now fails closed: reads, root verification, further apply.
        assert!(matches!(replayer.get(b"user:a"), Err(Error::Poisoned(_))));
        assert!(matches!(replayer.root_hash(), Err(Error::Poisoned(_))));
        assert!(matches!(
            replayer.apply(&[WriteOp::Delete {
                key: b"zz".to_vec()
            }]),
            Err(Error::Poisoned(_))
        ));
    }
}
