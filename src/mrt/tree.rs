use std::sync::Arc;

#[cfg(not(target_os = "zkvm"))]
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::hash::Hash;

pub(crate) const MAX_KEY_LEN: usize = 4 * 1024;
// A key's route is simply its raw bits, 8 per byte, MSB-first — no per-byte
// presence bit and no terminator. This is NOT prefix-free on its own; the tree
// ASSUMES keys are prefix-free (no key is a byte-prefix of another) and rejects
// violations at insert time (see `classify_key_vs_skip`).
pub(crate) const MAX_ROUTE_BITS: u16 = (MAX_KEY_LEN as u16) * 8;

pub(crate) fn validate_key_len(key: &[u8], context: &str) -> Result<()> {
    if key.len() > MAX_KEY_LEN {
        return Err(Error::Key(format!(
            "MRT {context} key length {} exceeds maximum {}",
            key.len(),
            MAX_KEY_LEN
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MrtNode {
    /// A leaf binds its **suffix** (`skip`, the key's route-bits below its
    /// deepest branch) and `value` — the absolute key is path-derived, never
    /// stored. Invariant: a leaf reached at depth `d` has
    /// `skip.bit_len() == route_len(key) − d` and `skip` equals the key's
    /// route-bits `[d .. route_len(key))`, so `skip` spans to the route
    /// terminus.
    Leaf {
        skip: RouteBits,
        value: Vec<u8>,
    },
    Branch {
        skip: RouteBits,
        left: Arc<MrtNodeInner>,
        right: Arc<MrtNodeInner>,
    },
    PrunedHash,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RouteBits {
    bit_len: u16,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatchResult {
    FullMatch,
    Mismatch { offset: u16 },
}

impl RouteBits {
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            bit_len: 0,
            bytes: Vec::new(),
        }
    }

    pub(crate) fn bit_len(&self) -> u16 {
        self.bit_len
    }

    pub(crate) fn packed_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn encoded_len(&self) -> usize {
        // `bit_len` is framed as u32-BE on the wire (see §3: word-width for
        // RISC0), so the prefix is 4 bytes even though it is held as u16.
        4 + self.bytes.len()
    }

    pub(crate) fn from_packed(bit_len: u16, packed: &[u8]) -> std::result::Result<Self, String> {
        if bit_len > MAX_ROUTE_BITS {
            return Err(format!(
                "route bit length {bit_len} exceeds maximum {MAX_ROUTE_BITS}"
            ));
        }
        let expected = packed_len(bit_len);
        if packed.len() != expected {
            return Err(format!(
                "route bit length {bit_len} requires {expected} packed bytes, got {}",
                packed.len()
            ));
        }
        if !bit_len.is_multiple_of(8) && !packed.is_empty() {
            let unused = 8 - (bit_len % 8);
            let mask = (1u8 << unused) - 1;
            if packed[packed.len() - 1] & mask != 0 {
                return Err("route bits have nonzero unused final-byte bits".into());
            }
        }
        Ok(Self {
            bit_len,
            bytes: packed.to_vec(),
        })
    }

    pub(crate) fn from_key_range(key: &[u8], start: u16, end: u16) -> Self {
        debug_assert!(start <= end);
        debug_assert!(end <= MAX_ROUTE_BITS);
        let bit_len = end - start;
        let mut bytes = vec![0u8; packed_len(bit_len)];
        // Write one route group (key byte) per step rather than one bit per step.
        // Each chunk runs to the next byte (8-bit) boundary or the window end, so a
        // leading partial byte and the full bytes are handled by the same body.
        let mut o = 0u16;
        while o < bit_len {
            let pos = start + o;
            let bit = pos % 8;
            let width = (8 - bit).min(bit_len - o);
            let field = route_group_field(key, (pos / 8) as usize, bit, width);
            set_packed_field(&mut bytes, o, width, field);
            o += width;
        }
        Self { bit_len, bytes }
    }

    pub(crate) fn bit_at(&self, offset: u16) -> bool {
        assert!(offset < self.bit_len);
        packed_bit(&self.bytes, offset)
    }

    pub(crate) fn matches_key_at(&self, key: &[u8], depth: u16) -> MatchResult {
        let base = depth as u32;
        let bit_len = self.bit_len as u32;
        // Route positions are bounded by u16::MAX; a window reaching past it is
        // only possible from an untrusted trace. Mirror the original
        // `checked_add` guard: bits before the overflow point compare normally,
        // and the first offset whose absolute position would overflow is a clean
        // miss. `valid` is the count of in-range offsets (`>= 1`, since
        // `depth <= u16::MAX`).
        let valid = (u16::MAX as u32 + 1).saturating_sub(base);
        let scan = bit_len.min(valid);

        // Compare one route group (key byte) per step: the <=8-bit slice of the
        // byte overlapping this chunk against the same slice of the packed skip,
        // batching what was eight per-bit compares. Each chunk runs to the next
        // byte boundary or to `scan`, whichever comes first.
        let mut o = 0u32;
        while o < scan {
            let pos = base + o;
            let bit = (pos % 8) as u16;
            let width = (8 - bit as u32).min(scan - o) as u16;
            let expected = route_group_field(key, (pos / 8) as usize, bit, width);
            let actual = packed_field(&self.bytes, o as u16, width);
            if expected != actual {
                // First differing route position in this chunk: the high bit of
                // the XOR, counted from the field's MSB (= the lowest offset).
                let diff = expected ^ actual;
                let first = width - 1 - (15 - diff.leading_zeros() as u16);
                return MatchResult::Mismatch {
                    offset: o as u16 + first,
                };
            }
            o += width as u32;
        }
        if scan < bit_len {
            return MatchResult::Mismatch {
                offset: scan as u16,
            };
        }
        MatchResult::FullMatch
    }

    pub(crate) fn slice(&self, start: u16, end: u16) -> Self {
        assert!(start <= end);
        assert!(end <= self.bit_len);
        let bit_len = end - start;
        let mut bytes = vec![0u8; packed_len(bit_len)];
        if start.is_multiple_of(8) {
            // Byte-aligned source: copy whole bytes, then clear the unused tail
            // bits the copy may have pulled in past `end`. (Bit-identical to the
            // loop below; just avoids the per-bit work — e.g. `split_at`'s
            // `slice(0, offset)` is always aligned.)
            let n = packed_len(bit_len);
            let src = (start / 8) as usize;
            bytes.copy_from_slice(&self.bytes[src..src + n]);
            if !bit_len.is_multiple_of(8) {
                let unused = 8 - (bit_len % 8);
                bytes[n - 1] &= !((1u8 << unused) - 1);
            }
        } else {
            // Unaligned source: copy up to one byte per step instead of one bit.
            // Destination offsets stay byte-aligned (steps of 8), so each
            // `set_packed_field` writes a single clean byte; the source slice is
            // read from its (unaligned) 2-byte window by `packed_field`. The
            // trailing partial chunk (`width < 8`) lands MSB-first, bit-identical
            // to the per-bit form, and cuts the loop to ~1/8 the iterations.
            let mut o = 0u16;
            while o < bit_len {
                let width = (bit_len - o).min(8);
                let field = packed_field(&self.bytes, start + o, width);
                set_packed_field(&mut bytes, o, width, field);
                o += width;
            }
        }
        Self { bit_len, bytes }
    }

    pub(crate) fn split_at(&self, bit_offset: u16) -> (Self, bool, Self) {
        assert!(bit_offset < self.bit_len);
        (
            self.slice(0, bit_offset),
            self.bit_at(bit_offset),
            self.slice(bit_offset + 1, self.bit_len),
        )
    }

    /// Consuming form of [`split_at`] that **reuses `self`'s allocation** for the
    /// `[0, bit_offset)` prefix: the prefix is a byte-prefix of the packed buffer,
    /// so it is just a `truncate` + final-byte tail-clear — no new `Vec`. Only the
    /// suffix allocates. Returned values are bit-identical to `split_at`'s. Used by
    /// the in-place verify replay where the branch's old skip is owned and would
    /// otherwise be dropped (see `NEW_ADVICE_MERK.md` lever 2).
    pub(crate) fn split_at_into(mut self, bit_offset: u16) -> (Self, bool, Self) {
        assert!(bit_offset < self.bit_len);
        let bit = self.bit_at(bit_offset);
        // Build the suffix while `self` is still intact (slice borrows it)…
        let suffix = self.slice(bit_offset + 1, self.bit_len);
        // …then repurpose `self`'s buffer as the prefix.
        self.bytes.truncate(packed_len(bit_offset));
        if !bit_offset.is_multiple_of(8) {
            let unused = 8 - (bit_offset % 8);
            let last = self.bytes.len() - 1;
            self.bytes[last] &= !((1u8 << unused) - 1);
        }
        self.bit_len = bit_offset;
        (self, bit, suffix)
    }

    /// Consuming form of `slice(start, self.bit_len())`: returns the suffix
    /// `[start, bit_len)` by **shifting it left into `self`'s own buffer** rather
    /// than allocating a new one. Bit-identical to that `slice`. Used by the leaf
    /// split, where the old leaf's owned skip is free to be reused for the
    /// re-skipped existing leaf (NEW_ADVICE_MERK.md lever 2 — split allocations).
    pub(crate) fn into_suffix(mut self, start: u16) -> Self {
        debug_assert!(start <= self.bit_len);
        let new_len = self.bit_len - start;
        let n = packed_len(new_len);
        let base = (start / 8) as usize;
        let bit = (start % 8) as u32;
        if bit == 0 {
            // Byte-aligned: just move the suffix bytes to the front.
            self.bytes.copy_within(base..base + n, 0);
        } else {
            // Each output byte combines the tail of one source byte with the head
            // of the next (`base + k` is always in bounds; `base + k + 1` may be
            // one past the end → reads 0). Writing `[k]` while reading `[>= k]` has
            // no read-after-write hazard going forward.
            for k in 0..n {
                let hi = self.bytes[base + k] << bit;
                let lo = self.bytes.get(base + k + 1).copied().unwrap_or(0) >> (8 - bit);
                self.bytes[k] = hi | lo;
            }
        }
        self.bytes.truncate(n);
        if !new_len.is_multiple_of(8) && n > 0 {
            let unused = 8 - (new_len % 8);
            self.bytes[n - 1] &= !((1u8 << unused) - 1);
        }
        self.bit_len = new_len;
        self
    }

    pub(crate) fn concat_bit_and_skip(&self, bit: bool, suffix: &RouteBits) -> Result<Self> {
        let total = self
            .bit_len
            .checked_add(1)
            .and_then(|n| n.checked_add(suffix.bit_len))
            .ok_or_else(|| Error::Tree("MRT route skip length overflow".into()))?;
        if total > MAX_ROUTE_BITS {
            return Err(Error::Tree(format!(
                "MRT route skip length {total} exceeds maximum {MAX_ROUTE_BITS}"
            )));
        }
        let mut bytes = vec![0u8; packed_len(total)];
        // Copy the prefix (`self`) whole-byte — its trailing bits are zero — then
        // place the separator bit and blit the (small) suffix.
        bytes[..self.bytes.len()].copy_from_slice(&self.bytes);
        set_packed_bit(&mut bytes, self.bit_len, bit);
        blit_bits(&mut bytes, self.bit_len + 1, suffix);
        Ok(Self {
            bit_len: total,
            bytes,
        })
    }

    /// Returns `self ‖ suffix` (bit concatenation). The building block for
    /// threading an accumulated route prefix down a descent.
    pub(crate) fn concat(&self, suffix: &RouteBits) -> Result<Self> {
        let total = self
            .bit_len
            .checked_add(suffix.bit_len)
            .ok_or_else(|| Error::Tree("MRT route skip length overflow".into()))?;
        if total > MAX_ROUTE_BITS {
            return Err(Error::Tree(format!(
                "MRT route skip length {total} exceeds maximum {MAX_ROUTE_BITS}"
            )));
        }
        let mut bytes = vec![0u8; packed_len(total)];
        // The prefix (`self`) is the large, growing operand when threading a
        // route down a descent — copy it whole-byte (trailing bits are zero)
        // rather than bit-by-bit, then blit the small suffix at the bit offset.
        // This is what makes prefix accumulation linear instead of O(depth^2)
        // bit-ops. Output is bit-identical to the per-bit form.
        bytes[..self.bytes.len()].copy_from_slice(&self.bytes);
        blit_bits(&mut bytes, self.bit_len, suffix);
        Ok(Self {
            bit_len: total,
            bytes,
        })
    }

    /// Returns `self ‖ suffix ‖ bit` in a **single allocation** — the descent
    /// step (accumulate a branch's skip, then its decision bit). Bit-identical to
    /// `self.concat(suffix)?.push_bit(bit)?` but avoids the intermediate buffer.
    pub(crate) fn concat_skip_then_bit(&self, suffix: &RouteBits, bit: bool) -> Result<Self> {
        let total = self
            .bit_len
            .checked_add(suffix.bit_len)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Tree("MRT route skip length overflow".into()))?;
        if total > MAX_ROUTE_BITS {
            return Err(Error::Tree(format!(
                "MRT route skip length {total} exceeds maximum {MAX_ROUTE_BITS}"
            )));
        }
        let mut bytes = vec![0u8; packed_len(total)];
        // Large prefix copied whole-byte; small suffix blitted at the bit offset;
        // then the single decision bit just past the suffix.
        bytes[..self.bytes.len()].copy_from_slice(&self.bytes);
        blit_bits(&mut bytes, self.bit_len, suffix);
        set_packed_bit(&mut bytes, self.bit_len + suffix.bit_len, bit);
        Ok(Self {
            bit_len: total,
            bytes,
        })
    }
}

fn packed_len(bit_len: u16) -> usize {
    (bit_len as usize).div_ceil(8)
}

fn packed_bit(bytes: &[u8], offset: u16) -> bool {
    let byte = bytes[(offset / 8) as usize];
    let mask = 0x80u8 >> (offset % 8);
    byte & mask != 0
}

fn set_packed_bit(bytes: &mut [u8], offset: u16, bit: bool) {
    if bit {
        let byte = &mut bytes[(offset / 8) as usize];
        *byte |= 0x80u8 >> (offset % 8);
    }
}

/// Reads the `width`-bit (1..=9) field at packed `offset`, MSB-first, as an
/// integer whose most-significant bit is the bit at `offset`. The batched form
/// of [`packed_bit`]. A `width`-bit field starting at any bit offset spans at
/// most two bytes, so a 2-byte big-endian window then a shift suffices.
#[inline]
fn packed_field(bytes: &[u8], offset: u16, width: u16) -> u16 {
    let idx = (offset / 8) as usize;
    let hi = bytes[idx] as u16;
    let lo = bytes.get(idx + 1).copied().unwrap_or(0) as u16;
    ((hi << 8 | lo) >> (16 - (offset % 8) - width)) & ((1u16 << width) - 1)
}

/// ORs the `width`-bit (1..=9) `field` (MSB-first, as produced by
/// [`route_group_field`]) into `bytes` at packed `offset`, which must be zero in
/// that range. The batched form of [`set_packed_bit`].
#[inline]
fn set_packed_field(bytes: &mut [u8], offset: u16, width: u16, field: u16) {
    if field == 0 {
        return;
    }
    let idx = (offset / 8) as usize;
    // `field`'s MSB lands at bit `15 - offset%8` of a 2-byte window, so the
    // placed value occupies bits `[shift, shift+width)` with `shift+width <= 16`.
    let placed = (field as u32) << (16 - (offset % 8) - width);
    bytes[idx] |= (placed >> 8) as u8;
    let lo = (placed & 0xFF) as u8;
    if lo != 0 {
        bytes[idx + 1] |= lo;
    }
}

/// Blits `src`'s bits into `dst` starting at bit `dst_off`. `dst` must be zeroed in
/// the target range `[dst_off, dst_off + src.bit_len)` (every caller allocates a
/// zeroed buffer or keeps the tail zeroed), so OR-ing `src` in is exact, while the
/// bits *before* `dst_off` in the partial first byte are preserved.
///
/// O(bytes), not O(bits): a byte-aligned `dst_off` is a bulk `copy_from_slice`;
/// otherwise each source byte is shifted across two destination bytes with the low
/// part carried in a register, so each dst byte is written **once** — no
/// read-modify-write, which matters under the zkVM's per-byte-store cost. Output is
/// bit-identical to the obvious one-bit-at-a-time form (see `route_bits_tests`).
fn blit_bits(dst: &mut [u8], dst_off: u16, src: &RouteBits) {
    if src.bit_len == 0 {
        return;
    }
    let shift = (dst_off & 7) as u32;
    let mut di = (dst_off >> 3) as usize;
    if shift == 0 {
        // Byte-aligned: target range is zeroed, so a bulk copy == OR.
        dst[di..di + src.bytes.len()].copy_from_slice(&src.bytes);
        return;
    }
    let rshift = 8 - shift;
    // `acc` starts as the existing first byte (preserves the pre-`dst_off` bits);
    // thereafter it carries the low `shift` bits of the previous source byte.
    let mut acc = dst[di];
    for &b in &src.bytes {
        dst[di] = acc | (b >> shift);
        acc = b << rshift;
        di += 1;
    }
    // Flush the final carry iff it lands in-buffer; when it doesn't, those bits are
    // past `src.bit_len` and are zero, so dropping them is correct.
    if di < dst.len() {
        dst[di] |= acc;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MrtNodeInner {
    node: MrtNode,
    hash: Hash,
    /// Relative max-subtree-depth: the bit-distance from this node's entry down
    /// to its deepest leaf (leaf → `skip.bit_len()`; branch →
    /// `skip.bit_len() + 1 + max(child depths)`; pruned stub → the value stamped
    /// from its parent's edge at decode). Held **eagerly** (always current, unlike
    /// the lazy verifier `hash_cache`) so `depth_below()` is O(1) and never
    /// recurses — a recursive accessor over materialized children would make
    /// `move_prefix` and branch hashing O(subtree). It is *relative* (measured from
    /// the node's own entry, not absolute position), so a relocated subtree keeps
    /// every interior `depth_below` → keeps every interior hash. Committed in the
    /// **parent's** branch hash (per child), never the node's own hash. See
    /// PLAN_MAX_SUBTREE_DEPTH.md.
    depth_below: u16,
}

#[cfg(test)]
thread_local! {
    /// Counts `MrtNodeInner::node()` dereferences so the regression guard test
    /// (`mrt_tracer_visits_scale_with_path_not_tree_size`) can assert per-change
    /// trace generation walks accessed paths, not the whole tree. The old
    /// `MrtPreStateIndex` build called `node()` once per tree node (O(n)).
    static MRT_NODE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Resets the test-only `node()` visit counter. See [`node_visits`].
#[cfg(test)]
pub(crate) fn reset_node_visits() {
    MRT_NODE_VISITS.with(|visits| visits.set(0));
}

/// Returns the test-only `node()` visit count since the last [`reset_node_visits`].
#[cfg(test)]
pub(crate) fn node_visits() -> usize {
    MRT_NODE_VISITS.with(|visits| visits.get())
}

impl MrtNodeInner {
    /// Test helper: a root-level leaf for `key` — its suffix is the full
    /// canonical route (depth 0), i.e. `route_bits_of(key)`.
    #[cfg(test)]
    pub(crate) fn leaf_value(key: Vec<u8>, value: Vec<u8>) -> Arc<Self> {
        Self::leaf(route_bits_of(&key), value)
    }

    pub(crate) fn leaf(skip: RouteBits, value: Vec<u8>) -> Arc<Self> {
        let hash = compute_leaf_hash(&skip, &value);
        // A leaf's skip spans to the key terminus, so its depth_below *is* its
        // skip length (no children, no decision bit).
        let depth_below = skip.bit_len();
        Arc::new(Self {
            node: MrtNode::Leaf { skip, value },
            hash,
            depth_below,
        })
    }

    fn branch_inner(
        skip: RouteBits,
        left: Arc<MrtNodeInner>,
        right: Arc<MrtNodeInner>,
    ) -> Arc<Self> {
        // The branch hash commits its *children's* depth_below (per-child, in this
        // node's edge); the node's own depth_below rolls them up for its parent.
        let l_depth = left.depth_below;
        let r_depth = right.depth_below;
        let hash = compute_branch_hash(&left.hash, &right.hash, l_depth, r_depth, &skip);
        let depth_below = branch_depth_below(skip.bit_len(), l_depth, r_depth);
        Arc::new(Self {
            node: MrtNode::Branch { skip, left, right },
            hash,
            depth_below,
        })
    }

    /// **Trusted** branch constructor (host ops / re-skin). The `debug_assert` catches
    /// a future *untrusted* caller that should have used [`decode_branch`] — valid
    /// trees always satisfy `depth_below ≤ MAX_ROUTE_BITS` (compiled out in release;
    /// never affects the production verifier).
    pub(crate) fn branch(
        skip: RouteBits,
        left: Arc<MrtNodeInner>,
        right: Arc<MrtNodeInner>,
    ) -> Arc<Self> {
        let node = Self::branch_inner(skip, left, right);
        debug_assert!(
            node.depth_below <= MAX_ROUTE_BITS,
            "branch depth_below {} exceeds {}: untrusted construction must use decode_branch",
            node.depth_below,
            MAX_ROUTE_BITS
        );
        node
    }

    /// Test-only unchecked branch: builds a branch whose rolled-up `depth_below` may
    /// exceed `MAX_ROUTE_BITS`, so a test can encode a malformed trace and assert the
    /// decoders reject it (the trusted [`branch`] would `debug_assert` on it).
    #[cfg(test)]
    pub(crate) fn branch_unchecked(
        skip: RouteBits,
        left: Arc<MrtNodeInner>,
        right: Arc<MrtNodeInner>,
    ) -> Arc<Self> {
        Self::branch_inner(skip, left, right)
    }

    /// A pruned stub carries the `depth_below` it stands for — not derivable from a
    /// bare hash, so it is stamped from the parent's edge at decode (and from the
    /// real child by the tracer when pruning a snapshot subtree). This is the
    /// **trusted** constructor; untrusted decoders must use [`decode_pruned`], which
    /// validates `depth_below ≤ MAX_ROUTE_BITS`. The `debug_assert` catches a future
    /// untrusted caller (compiled out in release).
    pub(crate) fn pruned(hash: Hash, depth_below: u16) -> Arc<Self> {
        debug_assert!(
            depth_below <= MAX_ROUTE_BITS,
            "pruned depth_below {} exceeds {}: untrusted construction must use decode_pruned",
            depth_below,
            MAX_ROUTE_BITS
        );
        Arc::new(Self {
            node: MrtNode::PrunedHash,
            hash,
            depth_below,
        })
    }

    /// Test-only unchecked pruned stub (depth may exceed `MAX_ROUTE_BITS`) — for
    /// building malformed-trace fixtures; see [`branch_unchecked`].
    #[cfg(test)]
    pub(crate) fn pruned_unchecked(hash: Hash, depth_below: u16) -> Arc<Self> {
        Arc::new(Self {
            node: MrtNode::PrunedHash,
            hash,
            depth_below,
        })
    }

    pub(crate) fn node(&self) -> &MrtNode {
        #[cfg(test)]
        MRT_NODE_VISITS.with(|visits| visits.set(visits.get() + 1));
        &self.node
    }

    pub(crate) fn hash(&self) -> Hash {
        self.hash
    }

    /// The eagerly-cached relative max-subtree-depth (see the `depth_below` field).
    /// O(1) — a plain field read, and (unlike [`node`](Self::node)) it does **not**
    /// count as a node visit.
    pub(crate) fn depth_below(&self) -> u16 {
        self.depth_below
    }
}

/// Bit `position` of `key`'s route — i.e. its raw bits, MSB-first, 8 per byte.
/// Positions at or past the key's end read as `false`: that keeps route-bit order
/// equal to byte order (a shorter key sorts before its extensions), which reads
/// and range scans rely on. A *stored* key never legitimately reads past its end
/// during descent (prefix-freeness guarantees divergence first); when it would,
/// insert reports an error rather than reaching here.
pub(crate) fn route_bit_at(key: &[u8], position: u16) -> bool {
    let symbol = (position / 8) as usize;
    if symbol >= key.len() {
        return false;
    }
    let offset = (position % 8) as u8;
    (key[symbol] >> (7 - offset)) & 1 == 1
}

/// The `width`-bit (1..=8) slice of key byte `symbol`, beginning at within-byte
/// offset `bit`, returned MSB-first (the first/lowest route position is the
/// field's most-significant bit). The batched form of [`route_bit_at`]; any
/// off-the-end byte reads as `0`. Walking one byte per step instead of one bit
/// per step cuts the membership and skip-construction loops to ~1/8 the iterations.
#[inline]
fn route_group_field(key: &[u8], symbol: usize, bit: u16, width: u16) -> u16 {
    let group = if symbol < key.len() {
        key[symbol] as u16
    } else {
        0
    };
    (group >> (8 - bit - width)) & ((1u16 << width) - 1)
}

/// The full-key route length in bits: `8·len` (the raw key bits).
pub(crate) fn route_len(key: &[u8]) -> u16 {
    debug_assert!(key.len() <= MAX_KEY_LEN);
    (8 * key.len()) as u16
}

#[cfg(test)]
pub(crate) fn prefix_route_bits(prefix: &[u8]) -> Result<u16> {
    validate_key_len(prefix, "prefix")?;
    Ok((8 * prefix.len()) as u16)
}

/// Full-key route: the key's raw bits `[0 .. 8·len)`. With this encoding the
/// route bits *are* the key bytes, so the packed form equals `key` — a single
/// clone, no per-byte chunk loop. The forward direction; [`key_from_route_bits`]
/// is its (fallible) inverse.
pub(crate) fn route_bits_of(key: &[u8]) -> RouteBits {
    RouteBits {
        bit_len: route_len(key),
        bytes: key.to_vec(),
    }
}

/// The suffix skip a leaf for `key` carries when reached at `depth`: the key's
/// route-bits `[depth .. route_len(key))`. The leaf-skip invariant (§4) in one
/// call site.
pub(crate) fn leaf_skip_from_key(key: &[u8], depth: u16) -> RouteBits {
    RouteBits::from_key_range(key, depth, route_len(key))
}

/// Whether the leaf with suffix `skip`, reached at `depth`, is exactly `key`.
/// Requires both a full bit match **and** an exact length match
/// (`depth + skip.bit_len() == route_len(key)`): a leaf whose skip is a proper
/// prefix of the query's remaining bits is a miss, not a hit (§6). The
/// `checked_add` folds the untrusted-trace overflow case into a clean miss.
pub(crate) fn leaf_matches(skip: &RouteBits, key: &[u8], depth: u16) -> bool {
    matches!(skip.matches_key_at(key, depth), MatchResult::FullMatch)
        && depth.checked_add(skip.bit_len()) == Some(route_len(key))
}

/// How a query `key` (reached at `depth`) relates to a node's `skip`. With the
/// raw, non-prefix-free route encoding, `matches_key_at` alone can no longer tell
/// "same key" from "one key is a prefix of the other" — this folds in the key's
/// remaining route length to recover that distinction, which insert needs both to
/// route correctly and to reject prefix-violating keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SkipRelation {
    /// Key and skip differ at this offset within their shared length → split here.
    Diverge { offset: u16 },
    /// All `skip` bits matched and the key ends exactly there: the same key.
    Equal,
    /// The key's remaining bits are a strict prefix of `skip` (key ends first).
    KeyIsPrefix,
    /// `skip` is a strict prefix of the key's remaining bits (key continues past).
    SkipIsPrefix,
}

/// Classifies `key` (at `depth`) against `skip` (see [`SkipRelation`]). Built on
/// [`RouteBits::matches_key_at`] (which compares `skip.bit_len()` bits, reading
/// the key as `0` past its end) plus the key's remaining length `key_rem`:
/// a mismatch *before* `key_rem` is a real divergence, while a full match — or a
/// mismatch at/after `key_rem` — is resolved into Equal / prefix by comparing
/// lengths. The untrusted-trace overflow guard inside `matches_key_at` is
/// inherited unchanged.
pub(crate) fn classify_key_vs_skip(skip: &RouteBits, key: &[u8], depth: u16) -> SkipRelation {
    let skip_len = skip.bit_len();
    let key_rem = route_len(key).saturating_sub(depth);
    match skip.matches_key_at(key, depth) {
        MatchResult::Mismatch { offset } => {
            if offset < key_rem {
                SkipRelation::Diverge { offset }
            } else {
                // The only difference is that the key ran out first.
                SkipRelation::KeyIsPrefix
            }
        }
        MatchResult::FullMatch => {
            if key_rem > skip_len {
                SkipRelation::SkipIsPrefix
            } else if key_rem == skip_len {
                SkipRelation::Equal
            } else {
                // skip's tail was all-zero, matching the key's off-the-end zeros.
                SkipRelation::KeyIsPrefix
            }
        }
    }
}

/// The error returned (in both the host build and verify replay) when an insert
/// would make one key a byte-prefix of another, violating the prefix-free key
/// assumption this tree is built on.
pub(crate) fn prefix_free_violation(key: &[u8]) -> Error {
    Error::Key(format!(
        "MRT keys must be prefix-free; key {key:?} is a prefix of, or extends, an existing key"
    ))
}

/// Reconstructs the key bytes from a full-key route, the inverse of
/// [`route_bits_of`]. Under the raw 8-bit encoding the route's packed bytes *are*
/// the key, so this **moves** the buffer out as the key — no copy. The only
/// validity requirement (it parses untrusted trace bits) is that the route ends
/// on a byte boundary (`bit_len % 8 == 0`); any deviation returns `Err`.
pub(crate) fn key_from_route_bits(bits: RouteBits) -> Result<Vec<u8>> {
    if !bits.bit_len.is_multiple_of(8) {
        return Err(Error::Tree(format!(
            "MRT route length {} is not a whole number of bytes (expected a multiple of 8)",
            bits.bit_len
        )));
    }
    Ok(bits.bytes)
}

/// Accumulates the route bits consumed from the root during a descent, so a
/// caller can reconstruct the absolute key of any node it reaches once leaves
/// store only suffixes. Each branch step appends `branch.skip ‖ decision_bit`
/// via [`RoutePrefix::descend`]; at a leaf, [`RoutePrefix::key_with_suffix`]
/// appends `leaf.skip` and decodes the completed route with
/// [`key_from_route_bits`].
#[derive(Clone, Debug)]
pub(crate) struct RoutePrefix {
    bits: RouteBits,
}

impl RoutePrefix {
    /// The empty prefix at the tree root.
    pub(crate) fn root() -> Self {
        Self {
            bits: RouteBits {
                bit_len: 0,
                bytes: Vec::new(),
            },
        }
    }

    /// The route bits accumulated so far. Used by tests to assert prefix shape;
    /// production threading needs only `descend`/`key_with_suffix`.
    #[cfg(test)]
    pub(crate) fn bits(&self) -> &RouteBits {
        &self.bits
    }

    /// Extends the prefix past a branch: appends the branch's `skip` then the
    /// decision bit into the chosen child (`false` = left, `true` = right). One
    /// allocation per level, not the two a `concat` + `push_bit` would take.
    pub(crate) fn descend(&self, branch_skip: &RouteBits, side: bool) -> Result<Self> {
        Ok(Self {
            bits: self.bits.concat_skip_then_bit(branch_skip, side)?,
        })
    }

    /// Reconstructs the absolute key of a leaf whose suffix is `leaf_skip`, i.e.
    /// `key_from_route_bits(prefix ‖ leaf_skip)`. The `concat` already owns the
    /// full-key route buffer (its bytes are the key), so it is moved out rather
    /// than copied again.
    pub(crate) fn key_with_suffix(&self, leaf_skip: &RouteBits) -> Result<Vec<u8>> {
        key_from_route_bits(self.bits.concat(leaf_skip)?)
    }

    // --- In-place buffer ops for a reused cursor route (see `Cursor`) ---------
    //
    // [`descend`]/[`key_with_suffix`] each allocate a fresh prefix per call; a range
    // scan over a subtree that shares a long common prefix then re-copies that whole
    // prefix per node and per leaf. These variants instead reuse one growable buffer:
    // descend appends in place, the cursor stores only a rewind mark per stack entry,
    // and on backtrack the buffer is truncated. The buffer's region past `bit_len` is
    // always kept zero (`truncate_to` clears the partial tail, `resize` adds zeros),
    // so the `blit_bits`/`set_packed_bit` OR-writes land in a clean range — output is
    // bit-identical to the allocating path. Pure scratch; no hashed bytes change.

    /// The number of route bits accumulated so far.
    pub(crate) fn bit_len(&self) -> u16 {
        self.bits.bit_len
    }

    /// Reset to the empty root prefix, reusing the existing allocation.
    pub(crate) fn clear(&mut self) {
        self.bits.bit_len = 0;
        self.bits.bytes.clear();
    }

    /// In-place `self ‖ branch_skip ‖ side` — the descent step of [`descend`],
    /// appended into the reused buffer instead of a fresh allocation.
    pub(crate) fn descend_in_place(&mut self, branch_skip: &RouteBits, side: bool) -> Result<()> {
        let total = self
            .bits
            .bit_len
            .checked_add(branch_skip.bit_len)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Tree("MRT route skip length overflow".into()))?;
        if total > MAX_ROUTE_BITS {
            return Err(Error::Tree(format!(
                "MRT route skip length {total} exceeds maximum {MAX_ROUTE_BITS}"
            )));
        }
        let at = self.bits.bit_len;
        self.bits.bytes.resize(packed_len(total), 0);
        blit_bits(&mut self.bits.bytes, at, branch_skip);
        set_packed_bit(&mut self.bits.bytes, at + branch_skip.bit_len, side);
        self.bits.bit_len = total;
        Ok(())
    }

    /// Shrink the buffer back to `bit_len` bits, zeroing the now-unused bits of the
    /// partial trailing byte so a later append still ORs into a zeroed range.
    pub(crate) fn truncate_to(&mut self, bit_len: u16) {
        debug_assert!(bit_len <= self.bits.bit_len);
        let keep_bytes = packed_len(bit_len);
        if !bit_len.is_multiple_of(8) {
            // Keep the top `used` bits of the last retained byte; clear the rest.
            let used = bit_len % 8;
            let mask = 0xFFu8 << (8 - used);
            self.bits.bytes[keep_bytes - 1] &= mask;
        }
        self.bits.bytes.truncate(keep_bytes);
        self.bits.bit_len = bit_len;
    }

    /// `key_from_route_bits(self ‖ leaf_skip)` using the reused buffer: append the
    /// leaf suffix, copy out the (byte-aligned) key bytes, then restore the buffer to
    /// its prior length. Bit-identical to [`key_with_suffix`].
    pub(crate) fn key_with_suffix_scratch(&mut self, leaf_skip: &RouteBits) -> Result<Vec<u8>> {
        let mark = self.bits.bit_len;
        let total = mark
            .checked_add(leaf_skip.bit_len)
            .ok_or_else(|| Error::Tree("MRT route skip length overflow".into()))?;
        if total > MAX_ROUTE_BITS {
            return Err(Error::Tree(format!(
                "MRT route skip length {total} exceeds maximum {MAX_ROUTE_BITS}"
            )));
        }
        self.bits.bytes.resize(packed_len(total), 0);
        blit_bits(&mut self.bits.bytes, mark, leaf_skip);
        self.bits.bit_len = total;
        if !total.is_multiple_of(8) {
            self.truncate_to(mark);
            return Err(Error::Tree(format!(
                "MRT route length {total} is not a whole number of bytes (expected a multiple of 8)"
            )));
        }
        let key = self.bits.bytes.clone();
        self.truncate_to(mark);
        Ok(key)
    }
}

/// `leaf_hash = H( skip.bit_len:u32-BE ‖ skip.packed_bytes ‖ value ‖ 0x01 )`
/// (§3). The value length is **not** framed — SHA commits to the total length,
/// and `skip` is length-prefixed, so the encoding stays injective. The trailing
/// `0x01` tag keeps leaf and branch preimages disjoint.
#[cfg(not(target_os = "zkvm"))]
pub(crate) fn compute_leaf_hash(skip: &RouteBits, value: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update((skip.bit_len() as u32).to_be_bytes());
    h.update(skip.packed_bytes());
    h.update(value);
    h.update([0x01]);
    h.finalize().into()
}

/// `branch_hash = H( left(32) ‖ right(32) ‖ left_depth_below:u32-BE ‖
/// right_depth_below:u32-BE ‖ skip.bit_len:u32-BE ‖ skip.packed_bytes ‖ 0x00 )`.
/// Children-first keeps the 32-byte hashes 4-aligned at the front; the two child
/// `depth_below`s follow as `u32`-BE words (same 4-alignment as `bit_len`), so the
/// branch commits each child's relative depth (per-child commitment — a write under
/// one child can rehash from the pruned sibling's *authenticated* depth). Fixed
/// prefix = 76 bytes, then the packed skip and the `0x00` tag (disjoint from leaf
/// preimages). See PLAN_MAX_SUBTREE_DEPTH.md §The field.
#[cfg(not(target_os = "zkvm"))]
pub(crate) fn compute_branch_hash(
    left_hash: &Hash,
    right_hash: &Hash,
    left_depth_below: u16,
    right_depth_below: u16,
    skip: &RouteBits,
) -> Hash {
    let mut h = Sha256::new();
    h.update(left_hash);
    h.update(right_hash);
    h.update((left_depth_below as u32).to_be_bytes());
    h.update((right_depth_below as u32).to_be_bytes());
    h.update((skip.bit_len() as u32).to_be_bytes());
    h.update(skip.packed_bytes());
    h.update([0x00]);
    h.finalize().into()
}

/// `depth_below` of a branch from its skip-bit length and its two children's
/// `depth_below`s: `skip + 1 + max(left, right)` — the relative bit-distance from
/// the branch's entry (its skip), through the decision bit, down to its deeper
/// child's deepest leaf. Saturating: host/verifier trees are valid by construction
/// (every key ≤ `MAX_ROUTE_BITS`, so every node's true depth_below ≤
/// `MAX_ROUTE_BITS`), so this never actually saturates; [`checked_branch_depth_below`]
/// is the guard for untrusted decoders.
pub(crate) fn branch_depth_below(skip_bits: u16, left: u16, right: u16) -> u16 {
    skip_bits.saturating_add(1).saturating_add(left.max(right))
}

/// Checked [`branch_depth_below`] for untrusted decode paths: `Err` on u16 overflow
/// or a result exceeding `MAX_ROUTE_BITS`. Used by the trace / trace / query-proof
/// branch constructors so a malformed trace can't push `depth_below` past the
/// in-memory `u16 ≤ MAX_ROUTE_BITS` invariant.
pub(crate) fn checked_branch_depth_below(skip_bits: u16, left: u16, right: u16) -> Result<u16> {
    let raw = skip_bits
        .checked_add(1)
        .and_then(|x| x.checked_add(left.max(right)))
        .ok_or_else(|| Error::Tree("MRT branch depth_below overflow".into()))?;
    validate_depth_below(raw)
}

/// Rejects a `depth_below` past `MAX_ROUTE_BITS`, preserving the in-memory
/// `u16 ≤ MAX_ROUTE_BITS` invariant on untrusted input (the wire carries depth as
/// `u32`; an honest encoder never emits an out-of-range value).
pub(crate) fn validate_depth_below(depth_below: u16) -> Result<u16> {
    if depth_below > MAX_ROUTE_BITS {
        return Err(Error::Tree(format!(
            "MRT depth_below {depth_below} exceeds maximum {MAX_ROUTE_BITS}"
        )));
    }
    Ok(depth_below)
}

/// Branch constructor for **untrusted decode paths** (trace / query-proof): builds
/// the branch like [`MrtNodeInner::branch`] but first validates the rolled-up
/// `depth_below` with [`checked_branch_depth_below`] (children's depths come from
/// the already-decoded child nodes), turning an over-deep malformed trace into a
/// clean `Err`. Host ops use the infallible `branch` (their trees are valid by
/// construction).
pub(crate) fn decode_branch(
    skip: RouteBits,
    left: Arc<MrtNodeInner>,
    right: Arc<MrtNodeInner>,
) -> Result<Arc<MrtNodeInner>> {
    checked_branch_depth_below(skip.bit_len(), left.depth_below(), right.depth_below())?;
    Ok(MrtNodeInner::branch(skip, left, right))
}

/// Pruned-stub constructor for untrusted decode paths: stamps the carried
/// `depth_below` after [`validate_depth_below`] rejects an out-of-range value.
pub(crate) fn decode_pruned(hash: Hash, depth_below: u16) -> Result<Arc<MrtNodeInner>> {
    Ok(MrtNodeInner::pruned(
        hash,
        validate_depth_below(depth_below)?,
    ))
}

// ─── RISC0 zkVM hash implementations ────────────────────────────────────────
//
// These produce byte-identical output to the Digest-based versions above, but
// call the SHA-256 accelerator syscall (sys_sha_buffer) directly via
// `crate::hash::zkvm_sha`, bypassing the Sha256::new/update/finalize wrapper.
// This eliminates ~50% of per-hash overhead (hasher lifecycle, internal
// buffering, redundant endianness conversions). The MRT decode/verify path is
// dominated by these two hashes, so the accelerated path is a direct win on
// guest cycles.
//
// Both preimages are variable-length (skip and value sizes vary), so — unlike
// hash.rs's fixed-size `node_hash` — they use the same aligned-stack-buffer
// (with heap fallback) construction as hash.rs's `kv_hash`: assemble the input
// + FIPS 180-4 padding once and make a single syscall. The stack buffer holds
// any preimage whose padded length is ≤ 128 bytes (data_len ≤ 119); larger
// inputs fall back to a heap buffer with the same direct syscall.

// leaf_hash = SHA-256( bit_len:u32-BE(4) || packed_skip || value || 0x01 )
#[cfg(target_os = "zkvm")]
pub(crate) fn compute_leaf_hash(skip: &RouteBits, value: &[u8]) -> Hash {
    let bit_len_be = (skip.bit_len() as u32).to_be_bytes();
    let packed = skip.packed_bytes();
    let value_start = 4 + packed.len();
    let tag_pos = value_start + value.len();
    let data_len = tag_pos + 1;
    if data_len <= 119 {
        let padded = (data_len + 9).div_ceil(64) * 64;
        #[repr(C, align(4))]
        struct Buf([u8; 128]);
        let mut b = core::mem::MaybeUninit::<Buf>::uninit();
        unsafe {
            let p = b.as_mut_ptr() as *mut u8;
            core::ptr::copy_nonoverlapping(bit_len_be.as_ptr(), p, 4);
            core::ptr::copy_nonoverlapping(packed.as_ptr(), p.add(4), packed.len());
            core::ptr::copy_nonoverlapping(value.as_ptr(), p.add(value_start), value.len());
            *p.add(tag_pos) = 0x01;
            *p.add(data_len) = 0x80;
            core::ptr::write_bytes(p.add(data_len + 1), 0, padded - 8 - (data_len + 1));
            let bits = (data_len as u64 * 8).to_be_bytes();
            core::ptr::copy_nonoverlapping(bits.as_ptr(), p.add(padded - 8), 8);
            crate::hash::zkvm_sha::compress_to_hash(p, (padded / 64) as u32)
        }
    } else {
        let mut tmp = vec![0u8; data_len];
        tmp[..4].copy_from_slice(&bit_len_be);
        tmp[4..value_start].copy_from_slice(packed);
        tmp[value_start..tag_pos].copy_from_slice(value);
        tmp[tag_pos] = 0x01;
        crate::hash::zkvm_sha::hash_bytes(&tmp)
    }
}

// branch_hash = SHA-256( left(32) || right(32) || left_depth_below:u32-BE(4) ||
//                        right_depth_below:u32-BE(4) || bit_len:u32-BE(4) ||
//                        packed_skip || 0x00 )
#[cfg(target_os = "zkvm")]
pub(crate) fn compute_branch_hash(
    left_hash: &Hash,
    right_hash: &Hash,
    left_depth_below: u16,
    right_depth_below: u16,
    skip: &RouteBits,
) -> Hash {
    let left_depth_be = (left_depth_below as u32).to_be_bytes();
    let right_depth_be = (right_depth_below as u32).to_be_bytes();
    let bit_len_be = (skip.bit_len() as u32).to_be_bytes();
    let packed = skip.packed_bytes();
    // left(32) + right(32) + left_depth(4) + right_depth(4) + bit_len(4)
    let skip_start = 76;
    let tag_pos = skip_start + packed.len();
    let data_len = tag_pos + 1;
    if data_len <= 119 {
        let padded = (data_len + 9).div_ceil(64) * 64;
        #[repr(C, align(4))]
        struct Buf([u8; 128]);
        let mut b = core::mem::MaybeUninit::<Buf>::uninit();
        unsafe {
            let p = b.as_mut_ptr() as *mut u8;
            core::ptr::copy_nonoverlapping(left_hash.as_ptr(), p, 32);
            core::ptr::copy_nonoverlapping(right_hash.as_ptr(), p.add(32), 32);
            core::ptr::copy_nonoverlapping(left_depth_be.as_ptr(), p.add(64), 4);
            core::ptr::copy_nonoverlapping(right_depth_be.as_ptr(), p.add(68), 4);
            core::ptr::copy_nonoverlapping(bit_len_be.as_ptr(), p.add(72), 4);
            core::ptr::copy_nonoverlapping(packed.as_ptr(), p.add(skip_start), packed.len());
            *p.add(tag_pos) = 0x00;
            *p.add(data_len) = 0x80;
            core::ptr::write_bytes(p.add(data_len + 1), 0, padded - 8 - (data_len + 1));
            let bits = (data_len as u64 * 8).to_be_bytes();
            core::ptr::copy_nonoverlapping(bits.as_ptr(), p.add(padded - 8), 8);
            crate::hash::zkvm_sha::compress_to_hash(p, (padded / 64) as u32)
        }
    } else {
        let mut tmp = vec![0u8; data_len];
        tmp[..32].copy_from_slice(left_hash);
        tmp[32..64].copy_from_slice(right_hash);
        tmp[64..68].copy_from_slice(&left_depth_be);
        tmp[68..72].copy_from_slice(&right_depth_be);
        tmp[72..skip_start].copy_from_slice(&bit_len_be);
        tmp[skip_start..tag_pos].copy_from_slice(packed);
        tmp[tag_pos] = 0x00;
        crate::hash::zkvm_sha::hash_bytes(&tmp)
    }
}

// ─── Materialized-preimage branch hashing (NEW_ADVICE_MERK.md lever 1) ───────
//
// PROTOTYPE for es/p2 measurement. The verify tree's rehash assembles a branch
// SHA preimage (`left ‖ right ‖ depths ‖ bit_len ‖ skip ‖ 0x00 ‖ pad`) on every
// hash and memcpys the 64 bytes of child hashes into it (~63K guest cyc, advice
// §1). Here a branch instead owns its padded preimage buffer: the *stable* part
// (`bit_len ‖ skip ‖ tag ‖ padding`, i.e. `[72..]`) is written once by
// `build_branch_preimage`, and the two child hashes are written **directly into the
// [0..64] slots by the SHA syscall** during rehash (`out_state` aimed at the slot —
// see `MrtVerifyNode::rehash_into`), so they are never copied. The two child
// `depth_below`s at `[64..72]` are *dynamic* (like the child hashes, unlike `skip`),
// so `rehash_into` rewrites them on every dirty rehash. The whole tree then hashes
// with no inter-node hash copies; only this small stable tail is assembled, once.
//
// Whether the per-branch buffer's added working set pays for itself (vs. the
// memcpy + the paging it removes) is the open question — measure in es/p2.

/// Allocates a branch's padded SHA-256 preimage buffer with the **stable** bytes
/// filled: `[0..64]` (the two child-hash slots), `[64..72]` (the two child
/// `depth_below` words), and the `0x00` branch tag are left zero, `[72..76]` holds
/// `bit_len:u32-BE`, `[76..]` the packed skip, then the FIPS 180-4 `0x80 ‖ 0… ‖
/// len64`. The child-hash slots and the child-depth slots are written later, in
/// place, on each rehash (`[64..72]` is dynamic — child depths change on writes —
/// so it is **not** filled here). Returned as `Box<[u32]>` purely for 4-byte
/// alignment (the syscall's `out_state`/`buf` must be word-aligned); it is only ever
/// touched as bytes. Layout matches `compute_branch_hash`, so the hash is
/// byte-identical.
pub(crate) fn build_branch_preimage(skip: &RouteBits) -> Box<[u32]> {
    let packed = skip.packed_bytes();
    // left(32) + right(32) + left_depth(4) + right_depth(4) + bit_len(4) = 76,
    // then skip, then the 1-byte tag.
    let data_len = 76 + packed.len() + 1;
    let padded = (data_len + 9).div_ceil(64) * 64;
    let mut buf = vec![0u32; padded / 4].into_boxed_slice();
    let bit_len_be = (skip.bit_len() as u32).to_be_bytes();
    let len_bits = (data_len as u64 * 8).to_be_bytes();
    // SAFETY: `buf` is `padded` bytes (padded >= 128 > data_len); every offset
    // written is in-bounds, and the buffer is zero-initialized so the tag, the
    // zero-fill, and (for empty skips) bit_len are already correct.
    unsafe {
        let p = buf.as_mut_ptr() as *mut u8;
        core::ptr::copy_nonoverlapping(bit_len_be.as_ptr(), p.add(72), 4);
        core::ptr::copy_nonoverlapping(packed.as_ptr(), p.add(76), packed.len());
        *p.add(data_len) = 0x80;
        core::ptr::copy_nonoverlapping(len_bits.as_ptr(), p.add(padded - 8), 8);
    }
    buf
}

/// Hashes a branch `preimage` (whose `[0..64]` child-hash slots are already
/// populated) and writes the 32-byte result to `dest`.
///
/// # Safety
/// `dest` must be valid for 32 writes and 4-byte aligned (the syscall writes the
/// state as `[u32; 8]`); `preimage` must be the buffer from
/// [`build_branch_preimage`] for `skip`, with both child slots filled and not
/// aliasing `dest`.
#[cfg(target_os = "zkvm")]
pub(crate) unsafe fn hash_branch_preimage_into(dest: *mut u8, preimage: &[u32], _skip: &RouteBits) {
    // preimage.len() u32 words = len*4 bytes = len/16 64-byte blocks.
    let blocks = (preimage.len() / 16) as u32;
    crate::hash::zkvm_sha::sys_sha_buffer(
        dest as *mut [u32; 8],
        &crate::hash::zkvm_sha::SHA256_IV,
        preimage.as_ptr() as *const u8,
        blocks,
    );
}

/// Host twin of the zkVM [`hash_branch_preimage_into`] — byte-identical output via
/// the `Digest` path over the same preimage bytes `[..data_len]`.
///
/// # Safety
/// Same contract as the zkVM variant: `dest` valid for 32 writes, `preimage` the
/// buffer for `skip` with child slots filled, not aliasing `dest`.
#[cfg(not(target_os = "zkvm"))]
pub(crate) unsafe fn hash_branch_preimage_into(dest: *mut u8, preimage: &[u32], skip: &RouteBits) {
    let data_len = 76 + skip.packed_bytes().len() + 1;
    let bytes = core::slice::from_raw_parts(preimage.as_ptr() as *const u8, preimage.len() * 4);
    let mut h = Sha256::new();
    h.update(&bytes[..data_len]);
    let out: Hash = h.finalize().into();
    core::ptr::copy_nonoverlapping(out.as_ptr(), dest, 32);
}

pub(crate) fn insert(
    root: Option<Arc<MrtNodeInner>>,
    key: Vec<u8>,
    value: Vec<u8>,
) -> Result<Arc<MrtNodeInner>> {
    validate_key_len(&key, "insert")?;
    let Some(root) = root else {
        return Ok(MrtNodeInner::leaf(route_bits_of(&key), value));
    };
    let mut record = |_arc: &Arc<MrtNodeInner>| {};
    insert_at(root, 0, &key, &value, &mut record)
}

pub(crate) fn delete(
    root: Option<Arc<MrtNodeInner>>,
    key: &[u8],
) -> Result<(Option<Arc<MrtNodeInner>>, bool)> {
    validate_key_len(key, "delete")?;
    let Some(root) = root else {
        return Ok((None, false));
    };
    let mut record = |_prefix: &RoutePrefix, _arc: &Arc<MrtNodeInner>| {};
    delete_at(root, 0, &RoutePrefix::root(), key, &mut record)
}

pub(crate) fn delete_range(
    root: Option<Arc<MrtNodeInner>>,
    start: &[u8],
    end: &[u8],
) -> Result<Option<Arc<MrtNodeInner>>> {
    validate_delete_range_bounds(start, end)?;
    let Some(root) = root else {
        return Ok(None);
    };
    let mut record = |_prefix: &RoutePrefix, _arc: &Arc<MrtNodeInner>| {};
    delete_range_at(
        root,
        0,
        &RoutePrefix::root(),
        Some(start),
        Some(end),
        &mut record,
    )
}

pub(crate) fn get<'a>(root: Option<&'a Arc<MrtNodeInner>>, key: &[u8]) -> Result<Option<&'a [u8]>> {
    validate_key_len(key, "get")?;
    let Some(root) = root else {
        return Ok(None);
    };
    let mut cur: &Arc<MrtNodeInner> = root;
    let mut depth = 0u16;
    loop {
        match cur.node() {
            MrtNode::Leaf { skip, value } => {
                if leaf_matches(skip, key, depth) {
                    return Ok(Some(value.as_slice()));
                }
                return Ok(None);
            }
            MrtNode::Branch { skip, left, right } => {
                if matches!(
                    skip.matches_key_at(key, depth),
                    MatchResult::Mismatch { .. }
                ) {
                    return Ok(None);
                }
                let branch_depth = checked_branch_depth(depth, skip)?;
                let side = route_bit_at(key, branch_depth);
                depth = checked_child_depth(branch_depth)?;
                cur = if !side { left } else { right };
            }
            MrtNode::PrunedHash => {
                return Err(Error::PrunedNode(format!("get descent at K={key:?}")));
            }
        }
    }
}

pub(crate) fn insert_with_trace(
    root: Option<Arc<MrtNodeInner>>,
    key: Vec<u8>,
    value: Vec<u8>,
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<Arc<MrtNodeInner>> {
    validate_key_len(&key, "insert")?;
    let Some(root) = root else {
        return Ok(MrtNodeInner::leaf(route_bits_of(&key), value));
    };
    insert_at(root, 0, &key, &value, record)
}

pub(crate) fn delete_with_trace(
    root: Option<Arc<MrtNodeInner>>,
    key: &[u8],
    record: &mut impl FnMut(&RoutePrefix, &Arc<MrtNodeInner>),
) -> Result<(Option<Arc<MrtNodeInner>>, bool)> {
    validate_key_len(key, "delete")?;
    let Some(root) = root else {
        return Ok((None, false));
    };
    delete_at(root, 0, &RoutePrefix::root(), key, record)
}

pub(crate) fn delete_range_with_trace(
    root: Option<Arc<MrtNodeInner>>,
    start: &[u8],
    end: &[u8],
    record: &mut impl FnMut(&RoutePrefix, &Arc<MrtNodeInner>),
) -> Result<Option<Arc<MrtNodeInner>>> {
    validate_delete_range_bounds(start, end)?;
    let Some(root) = root else {
        return Ok(None);
    };
    delete_range_at(
        root,
        0,
        &RoutePrefix::root(),
        Some(start),
        Some(end),
        record,
    )
}

pub(crate) fn delete_prefix_with_trace(
    root: Option<Arc<MrtNodeInner>>,
    prefix: &[u8],
    record: &mut impl FnMut(&RoutePrefix, &Arc<MrtNodeInner>),
) -> Result<Option<Arc<MrtNodeInner>>> {
    validate_key_len(prefix, "delete_prefix")?;
    let Some(root) = root else {
        return Ok(None);
    };
    let hi = crate::tracer::prefix_successor(prefix);
    delete_range_at(
        root,
        0,
        &RoutePrefix::root(),
        Some(prefix),
        hi.as_deref(),
        record,
    )
}

#[cfg(test)]
pub(crate) fn get_with_trace<'a>(
    root: Option<&'a Arc<MrtNodeInner>>,
    key: &[u8],
    record: &mut impl FnMut(&'a Arc<MrtNodeInner>),
) -> Result<Option<&'a [u8]>> {
    validate_key_len(key, "get")?;
    let Some(root) = root else {
        return Ok(None);
    };
    let mut cur: &'a Arc<MrtNodeInner> = root;
    let mut depth = 0u16;
    loop {
        record(cur);
        match cur.node() {
            MrtNode::Leaf { skip, value } => {
                if leaf_matches(skip, key, depth) {
                    return Ok(Some(value.as_slice()));
                }
                return Ok(None);
            }
            MrtNode::Branch { skip, left, right } => {
                if matches!(
                    skip.matches_key_at(key, depth),
                    MatchResult::Mismatch { .. }
                ) {
                    return Ok(None);
                }
                let branch_depth = checked_branch_depth(depth, skip)?;
                let side = route_bit_at(key, branch_depth);
                depth = checked_child_depth(branch_depth)?;
                cur = if !side { left } else { right };
            }
            MrtNode::PrunedHash => {
                return Err(Error::PrunedNode(format!("get descent at K={key:?}")));
            }
        }
    }
}

fn checked_branch_depth(depth: u16, skip: &RouteBits) -> Result<u16> {
    let branch_depth = depth
        .checked_add(skip.bit_len())
        .ok_or_else(|| Error::Tree("MRT branch depth overflow".into()))?;
    if branch_depth >= MAX_ROUTE_BITS {
        return Err(Error::Tree(format!(
            "MRT branch depth {branch_depth} exceeds maximum decision bit {}",
            MAX_ROUTE_BITS - 1
        )));
    }
    Ok(branch_depth)
}

fn checked_child_depth(branch_depth: u16) -> Result<u16> {
    branch_depth
        .checked_add(1)
        .ok_or_else(|| Error::Tree("MRT child depth overflow".into()))
}

fn checked_add_depth(depth: u16, offset: u16) -> Result<u16> {
    depth
        .checked_add(offset)
        .ok_or_else(|| Error::Tree("MRT route depth overflow".into()))
}

fn validate_delete_range_bounds(start: &[u8], end: &[u8]) -> Result<()> {
    validate_key_len(start, "delete_range start")?;
    validate_key_len(end, "delete_range end")?;
    if start >= end {
        return Err(Error::Key(format!(
            "MRT delete_range start key {start:?} must be less than end key {end:?}"
        )));
    }
    Ok(())
}

fn insert_at(
    cur: Arc<MrtNodeInner>,
    depth: u16,
    key: &[u8],
    value: &[u8],
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<Arc<MrtNodeInner>> {
    record(&cur);
    match cur.node() {
        MrtNode::Leaf {
            skip: l_skip,
            value: l_value,
        } => match classify_key_vs_skip(l_skip, key, depth) {
            // Same key: replace the value (re-skipping to the identical suffix).
            SkipRelation::Equal => Ok(MrtNodeInner::leaf(
                leaf_skip_from_key(key, depth),
                value.to_vec(),
            )),
            // Diverge at bit `p`: a new branch over the common prefix, with both
            // the existing leaf and the new leaf re-skipped to their suffixes.
            SkipRelation::Diverge { offset } => {
                let p = checked_add_depth(depth, offset)?;
                let branch_skip = RouteBits::from_key_range(key, depth, p);
                let l_bit = l_skip.bit_at(offset);
                let k_bit = route_bit_at(key, p);
                debug_assert_ne!(k_bit, l_bit);
                let child_depth = checked_child_depth(p)?;
                let existing_leaf =
                    MrtNodeInner::leaf(l_skip.slice(offset + 1, l_skip.bit_len()), l_value.clone());
                let k_leaf =
                    MrtNodeInner::leaf(leaf_skip_from_key(key, child_depth), value.to_vec());
                if !k_bit {
                    Ok(MrtNodeInner::branch(branch_skip, k_leaf, existing_leaf))
                } else {
                    Ok(MrtNodeInner::branch(branch_skip, existing_leaf, k_leaf))
                }
            }
            // One key is a prefix of the other — not allowed.
            SkipRelation::KeyIsPrefix | SkipRelation::SkipIsPrefix => {
                Err(prefix_free_violation(key))
            }
        },
        MrtNode::Branch { skip, left, right } => match classify_key_vs_skip(skip, key, depth) {
            SkipRelation::Diverge { offset } => {
                let (prefix, existing_bit, existing_suffix) = skip.split_at(offset);
                let p = checked_add_depth(depth, offset)?;
                let k_bit = route_bit_at(key, p);
                debug_assert_ne!(k_bit, existing_bit);
                let existing = MrtNodeInner::branch(existing_suffix, left.clone(), right.clone());
                let child_depth = checked_child_depth(p)?;
                let k_leaf =
                    MrtNodeInner::leaf(leaf_skip_from_key(key, child_depth), value.to_vec());
                if !k_bit {
                    Ok(MrtNodeInner::branch(prefix, k_leaf, existing))
                } else {
                    Ok(MrtNodeInner::branch(prefix, existing, k_leaf))
                }
            }
            // Key continues past this branch's skip: descend into the routed child.
            SkipRelation::SkipIsPrefix => {
                let branch_depth = checked_branch_depth(depth, skip)?;
                let side = route_bit_at(key, branch_depth);
                let child_depth = checked_child_depth(branch_depth)?;
                if !side {
                    let new_left = insert_at(left.clone(), child_depth, key, value, record)?;
                    Ok(MrtNodeInner::branch(skip.clone(), new_left, right.clone()))
                } else {
                    let new_right = insert_at(right.clone(), child_depth, key, value, record)?;
                    Ok(MrtNodeInner::branch(skip.clone(), left.clone(), new_right))
                }
            }
            // Key ends at or within this branch's skip → it is a prefix of every
            // key in the subtree below — not allowed.
            SkipRelation::Equal | SkipRelation::KeyIsPrefix => Err(prefix_free_violation(key)),
        },
        MrtNode::PrunedHash => Err(Error::PrunedNode(format!("insert descent at K={key:?}"))),
    }
}

fn delete_at(
    cur: Arc<MrtNodeInner>,
    depth: u16,
    prefix: &RoutePrefix,
    key: &[u8],
    record: &mut impl FnMut(&RoutePrefix, &Arc<MrtNodeInner>),
) -> Result<(Option<Arc<MrtNodeInner>>, bool)> {
    record(prefix, &cur);
    match cur.node() {
        MrtNode::Leaf { skip, .. } => {
            if leaf_matches(skip, key, depth) {
                Ok((None, true))
            } else {
                Ok((Some(cur), false))
            }
        }
        MrtNode::Branch { skip, left, right } => {
            if matches!(
                skip.matches_key_at(key, depth),
                MatchResult::Mismatch { .. }
            ) {
                return Ok((Some(cur), false));
            }
            let branch_depth = checked_branch_depth(depth, skip)?;
            let side = route_bit_at(key, branch_depth);
            let child_depth = checked_child_depth(branch_depth)?;
            let child_prefix = prefix.descend(skip, side)?;
            let (new_child, deleted) = if !side {
                delete_at(left.clone(), child_depth, &child_prefix, key, record)?
            } else {
                delete_at(right.clone(), child_depth, &child_prefix, key, record)?
            };
            if !deleted {
                return Ok((Some(cur), false));
            }
            match (side, new_child) {
                (false, None) => {
                    record(&prefix.descend(skip, true)?, right);
                    Ok((Some(collapse_survivor(skip, true, right.clone())?), true))
                }
                (true, None) => {
                    record(&prefix.descend(skip, false)?, left);
                    Ok((Some(collapse_survivor(skip, false, left.clone())?), true))
                }
                (false, Some(new_left)) => Ok((
                    Some(MrtNodeInner::branch(skip.clone(), new_left, right.clone())),
                    true,
                )),
                (true, Some(new_right)) => Ok((
                    Some(MrtNodeInner::branch(skip.clone(), left.clone(), new_right)),
                    true,
                )),
            }
        }
        MrtNode::PrunedHash => Err(Error::PrunedNode(format!("delete descent at K={key:?}"))),
    }
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

fn delete_range_at<'a>(
    cur: Arc<MrtNodeInner>,
    depth: u16,
    prefix: &RoutePrefix,
    lo: Option<&'a [u8]>,
    hi: Option<&'a [u8]>,
    record: &mut impl FnMut(&RoutePrefix, &Arc<MrtNodeInner>),
) -> Result<Option<Arc<MrtNodeInner>>> {
    // Whole subtree falls in range: drop it without recording `cur`. The verifier's
    // delete_range_at returns at this same fast path without inspecting the node, so
    // a wholly-deleted subtree root only needs to be a pruned stub — recording it
    // would over-reveal. `record` must therefore come AFTER this fast path.
    if lo.is_none() && hi.is_none() {
        return Ok(None);
    }
    record(prefix, &cur);

    match cur.node() {
        MrtNode::Leaf { skip, .. } => {
            // Membership now needs the *absolute* key, reconstructed from the
            // accumulated prefix and the leaf's suffix (the stored key is gone).
            let key = prefix.key_with_suffix(skip)?;
            if lo.is_some_and(|lo| key.as_slice() < lo) || hi.is_some_and(|hi| key.as_slice() >= hi)
            {
                Ok(Some(cur))
            } else {
                Ok(None)
            }
        }
        MrtNode::Branch { skip, left, right } => {
            let lo_pos = lo.map(|key| classify_bound(key, depth, skip)).transpose()?;
            if lo_pos == Some(BoundPosition::After) {
                return Ok(Some(cur));
            }

            let hi_pos = hi.map(|key| classify_bound(key, depth, skip)).transpose()?;
            if hi_pos == Some(BoundPosition::Before) {
                return Ok(Some(cur));
            }

            let branch_depth = checked_branch_depth(depth, skip)?;
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

            let new_left =
                apply_child_range(left.clone(), child_depth, &left_prefix, left_range, record)?;
            let new_right = apply_child_range(
                right.clone(),
                child_depth,
                &right_prefix,
                right_range,
                record,
            )?;

            match (new_left, new_right) {
                (Some(new_left), Some(new_right)) => Ok(Some(MrtNodeInner::branch(
                    skip.clone(),
                    new_left,
                    new_right,
                ))),
                (Some(survivor), None) => {
                    record(&left_prefix, &survivor);
                    Ok(Some(collapse_survivor(skip, false, survivor)?))
                }
                (None, Some(survivor)) => {
                    record(&right_prefix, &survivor);
                    Ok(Some(collapse_survivor(skip, true, survivor)?))
                }
                (None, None) => Ok(None),
            }
        }
        MrtNode::PrunedHash => Err(Error::PrunedNode("delete_range descent".into())),
    }
}

fn classify_bound(key: &[u8], depth: u16, skip: &RouteBits) -> Result<BoundPosition> {
    // Reuse the byte-chunked descent compare rather than a per-bit scan (this used to
    // duplicate `matches_key_at`'s logic one bit at a time). First divergence → which
    // side the bound key falls; a full skip match → the branch decision bit picks the
    // child the bound descends into.
    match skip.matches_key_at(key, depth) {
        MatchResult::Mismatch { offset } => {
            let position = checked_add_depth(depth, offset)?;
            Ok(if route_bit_at(key, position) {
                BoundPosition::After
            } else {
                BoundPosition::Before
            })
        }
        MatchResult::FullMatch => {
            let branch_depth = checked_branch_depth(depth, skip)?;
            Ok(if route_bit_at(key, branch_depth) {
                BoundPosition::InRight
            } else {
                BoundPosition::InLeft
            })
        }
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
    child: Arc<MrtNodeInner>,
    child_depth: u16,
    child_prefix: &RoutePrefix,
    range: ChildRange<'a>,
    record: &mut impl FnMut(&RoutePrefix, &Arc<MrtNodeInner>),
) -> Result<Option<Arc<MrtNodeInner>>> {
    match range {
        ChildRange::NoOverlap => Ok(Some(child)),
        ChildRange::Full => delete_range_at(child, child_depth, child_prefix, None, None, record),
        ChildRange::Partial { lo, hi } => {
            delete_range_at(child, child_depth, child_prefix, lo, hi, record)
        }
    }
}

/// Re-roots `survivor` one level up after its sibling was deleted, absorbing the
/// parent's `skip` and the branch decision bit. Under suffix-relative hashing a
/// **leaf** survivor re-skips just like a branch survivor (§5) — the merged skip
/// is `parent_skip ‖ survivor_side ‖ old_skip`. A pruned survivor can't be
/// re-skipped (its interior skip is hidden), so it is left as-is; the resulting
/// hash mismatch is caught by `verify_root` (such a collapse never arises in a
/// valid trace, which records the materialized survivor).
fn collapse_survivor(
    parent_skip: &RouteBits,
    survivor_side: bool,
    survivor: Arc<MrtNodeInner>,
) -> Result<Arc<MrtNodeInner>> {
    match survivor.node() {
        MrtNode::Branch { skip, left, right } => {
            let merged = parent_skip.concat_bit_and_skip(survivor_side, skip)?;
            Ok(MrtNodeInner::branch(merged, left.clone(), right.clone()))
        }
        MrtNode::Leaf { skip, value } => {
            let merged = parent_skip.concat_bit_and_skip(survivor_side, skip)?;
            Ok(MrtNodeInner::leaf(merged, value.clone()))
        }
        MrtNode::PrunedHash => Ok(survivor),
    }
}

// ─── prefix-locus descent (`move_prefix` subtree relocate, stage 1) ──────────
//
// `move_prefix(p, q)` relocates the whole subtree of keys under byte-prefix `p`
// to prefix `q` in O(depth) work. Stage 1 builds the two *source-side* host
// primitives it rests on: `locate_prefix_subtree` (a read-only descent that lands
// on the one subtree `S` carrying every prefix-`p` key) and
// `detach_prefix_subtree` (the CoW collapse that removes `S` from the tree and
// captures it for re-skinning at `q`). Both share `classify_prefix_step`, the
// pure per-node decision; the verifier (stage 3) reuses that same decision over
// its in-place tree.
//
// Stage 3 wires the public `move_prefix` op (`tree::move_prefix` /
// `move_prefix_with_trace`) on top of these, so the detach/splice/re-skin
// primitives are now production code. Only the read-only `locate_prefix_subtree`
// (+ `Locus`) and the `delete_prefix` round-trip oracle stay `#[cfg(test)]` — they
// have no production caller (detach descends directly on `classify_prefix_step`).

/// One step of the prefix-locus descent — the **shared decision core** behind
/// both [`locate_prefix_subtree`] (read-only) and [`detach_prefix_subtree`]
/// (mutating). For a node reached at absolute `depth` while walking prefix `p`,
/// it reports either that `p` is exhausted here (this node is the located subtree
/// `S`) or that `p` continues one level deeper. A byte-prefix in a radix trie
/// always resolves to exactly one subtree, so the only failure is an **absent**
/// `p` (the descent diverges, or runs off a leaf), returned as `Err`.
enum PrefixStep {
    /// `p` ends within/at this node: it is `S`. `strip_prefix_bits` (`= 8·|p| −
    /// depth`) is how many leading bits of `S`'s root skip are the prefix tail —
    /// what re-skinning strips before prepending `q`'s tail.
    Stop { strip_prefix_bits: u16 },
    /// `p` consumed this branch's decision bit: descend into `side` (`false` =
    /// left, `true` = right); the chosen child is entered at `child_depth`.
    Descend { side: bool, child_depth: u16 },
}

/// The error when `p` is not an existing prefix — the locus descent diverges from
/// every stored key, or runs off the end of a leaf whose key is a strict prefix of
/// `p`. (A byte-prefix otherwise always resolves to exactly one subtree, so there
/// is no "ragged" case — only absence.)
fn prefix_absent(p: &[u8]) -> Error {
    Error::Key(format!(
        "MRT prefix {p:?} is absent: no stored key has it as a byte-prefix"
    ))
}

/// Classifies one node of the prefix-locus descent (see [`PrefixStep`]). The
/// branch/leaf landing rules collapse onto [`classify_key_vs_skip`] (treating `p`
/// as the query key): `Equal`/`KeyIsPrefix` mean `p` ends within/at this node →
/// it is `S`; `SkipIsPrefix` at a branch means `p` ran past the decision bit →
/// route to a child (at a leaf it means `p` extends past the key → absent);
/// `Diverge` is always absent. `strip_prefix_bits = 8·|p| − depth` is correct in
/// every `Stop` case (whole-tree `S` at `depth == 0` gives the whole prefix).
fn classify_prefix_step(node: &MrtNode, depth: u16, p: &[u8]) -> Result<PrefixStep> {
    let p_bits = route_len(p);
    debug_assert!(depth <= p_bits, "prefix-locus descent overran the prefix");
    let stop = PrefixStep::Stop {
        strip_prefix_bits: p_bits - depth,
    };
    match node {
        // A leaf is `S` only if `p` is exhausted within its key. If `p` runs past
        // the key (the key is a strict prefix of `p`) or diverges from it, no
        // stored key has prefix `p`.
        MrtNode::Leaf { skip, .. } => match classify_key_vs_skip(skip, p, depth) {
            SkipRelation::Equal | SkipRelation::KeyIsPrefix => Ok(stop),
            SkipRelation::SkipIsPrefix | SkipRelation::Diverge { .. } => Err(prefix_absent(p)),
        },
        // A branch is `S` when `p` ends within/at its skip, *before* consuming the
        // decision bit (then both children carry prefix `p`). If `p` continues
        // past the decision bit, route into the chosen child; if it diverges from
        // the skip, `p` is absent.
        MrtNode::Branch { skip, .. } => match classify_key_vs_skip(skip, p, depth) {
            SkipRelation::Equal | SkipRelation::KeyIsPrefix => Ok(stop),
            SkipRelation::SkipIsPrefix => {
                let branch_depth = checked_branch_depth(depth, skip)?;
                let side = route_bit_at(p, branch_depth);
                let child_depth = checked_child_depth(branch_depth)?;
                Ok(PrefixStep::Descend { side, child_depth })
            }
            SkipRelation::Diverge { .. } => Err(prefix_absent(p)),
        },
        MrtNode::PrunedHash => Err(Error::PrunedNode(format!(
            "prefix-locus descent at prefix {p:?}"
        ))),
    }
}

/// Where prefix `p` resolves in a tree: the one subtree `S` carrying every
/// prefix-`p` key (a node when `p` ends within/at a branch's skip, or a leaf when
/// `p` is exhausted within that leaf's key). Read-only — `s_root` borrows the
/// pre-state tree; re-skinning math reads `entry_depth`/`strip_prefix_bits`.
#[cfg(test)]
pub(crate) struct Locus<'a> {
    /// The located subtree root `S` (a reference into the pre-state tree).
    pub(crate) s_root: &'a Arc<MrtNodeInner>,
    /// Absolute depth at which `S` is entered (`e_src`) — `0` for a whole-tree
    /// `S` (every key has prefix `p`).
    pub(crate) entry_depth: u16,
    /// `8·|p| − entry_depth`: leading bits of `S`'s root skip that are the prefix
    /// tail (what re-skinning strips before prepending `q`'s tail).
    pub(crate) strip_prefix_bits: u16,
}

/// `S` removed from the tree, ready to re-skin at the destination prefix.
pub(crate) struct Captured {
    /// `S`'s original pre-state root node — a **branch** carrying its two child
    /// handles, or a **leaf** carrying its value. Re-skinning at `q` recomputes
    /// only this one node's hash; everything below it is depth- and
    /// prefix-invariant under suffix-relative hashing, so it stays untouched.
    pub(crate) s_root: Arc<MrtNodeInner>,
    /// How many leading bits of `s_root`'s skip are the prefix tail (`8·|p| −
    /// entry_depth`). Not recoverable from `s_root` alone — its skip does not
    /// record where `p` ended inside it — so detach (which tracks descent depth)
    /// computes and carries it.
    pub(crate) strip_prefix_bits: u16,
}

/// The **direct prefix-locus descent**: walk `p`'s bits against node skips and
/// branch decisions until `p` is exhausted, landing on the single subtree `S`
/// (see [`Locus`]). `Err` only on an absent `p`. This is the part shared between
/// host and verifier (stage 3); both build their detach/splice on it.
#[cfg(test)]
pub(crate) fn locate_prefix_subtree<'a>(
    root: &'a Arc<MrtNodeInner>,
    p: &[u8],
) -> Result<Locus<'a>> {
    let mut cur = root;
    let mut depth = 0u16;
    loop {
        let node = cur.node();
        match classify_prefix_step(node, depth, p)? {
            PrefixStep::Stop { strip_prefix_bits } => {
                return Ok(Locus {
                    s_root: cur,
                    entry_depth: depth,
                    strip_prefix_bits,
                });
            }
            PrefixStep::Descend { side, child_depth } => {
                let MrtNode::Branch { left, right, .. } = node else {
                    unreachable!("Descend is only returned for a branch node");
                };
                cur = if !side { left } else { right };
                depth = child_depth;
            }
        }
    }
}

/// Detach the entire prefix-`p` subtree `S` from `root`, returning the
/// **post-detach** tree (source path collapsed) plus the [`Captured`] `S` for
/// re-skinning at `q`. The host (CoW) collapse mirrors `delete_at`: rebuild the
/// `p`-path on the way up, and where removing `S` leaves its parent branch lonely,
/// re-skip the surviving sibling one level up via [`collapse_survivor`]. `S` may be
/// the **whole tree** (every key has prefix `p`, including a single-key tree), in
/// which case the result is `(None, Captured)` with `entry_depth == 0`. `Err` only
/// on an absent `p` (which includes an empty `root`).
///
/// Test-only thin wrapper over [`detach_prefix_subtree_inner`] with a no-op
/// record hook — the production op (`move_prefix`) threads the real hook; the
/// Stage 1 round-trip gate calls this directly.
#[cfg(test)]
pub(crate) fn detach_prefix_subtree(
    root: Option<Arc<MrtNodeInner>>,
    p: &[u8],
) -> Result<(Option<Arc<MrtNodeInner>>, Captured)> {
    let mut record = |_: &Arc<MrtNodeInner>| {};
    detach_prefix_subtree_inner(root, p, &mut record)
}

/// Shared core behind [`detach_prefix_subtree`] and the tracer's
/// `move_prefix_with_trace`: the `record` hook is handed every pre-state node the
/// detach descent touches — the `p`-path nodes (`S`'s **original** root included)
/// and, on a source collapse, the materialized **survivor** root (`S`'s sibling),
/// which the verifier re-skips onto its grandparent. Installing these by `Arc`
/// identity is exactly what the trace needs to authenticate + replay the detach.
fn detach_prefix_subtree_inner(
    root: Option<Arc<MrtNodeInner>>,
    p: &[u8],
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<(Option<Arc<MrtNodeInner>>, Captured)> {
    let Some(root) = root else {
        return Err(prefix_absent(p));
    };
    detach_at(root, 0, p, record)
}

/// Recursive worker for [`detach_prefix_subtree`]. Returns the rebuilt node for
/// this position (`None` exactly when this node *is* `S`, so the caller collapses)
/// alongside the captured `S`. Every node visited on the descent (and the survivor
/// at a collapse) is handed to `record`.
fn detach_at(
    cur: Arc<MrtNodeInner>,
    depth: u16,
    p: &[u8],
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<(Option<Arc<MrtNodeInner>>, Captured)> {
    record(&cur);
    match classify_prefix_step(cur.node(), depth, p)? {
        // `cur` is `S`: detach it (this position becomes empty) and capture it. It
        // was just recorded above, so the trace reveals `S`'s original root; its
        // interior stays pruned because the descent stops here.
        PrefixStep::Stop { strip_prefix_bits } => Ok((
            None,
            Captured {
                s_root: cur,
                strip_prefix_bits,
            },
        )),
        // Descend, then rebuild — collapsing onto the sibling if the chosen child
        // was `S` (its whole subtree detached → `None`). The surviving sibling root
        // is recorded so the verifier can re-skip it on collapse.
        PrefixStep::Descend { side, child_depth } => {
            let MrtNode::Branch { skip, left, right } = cur.node() else {
                unreachable!("Descend is only returned for a branch node");
            };
            if !side {
                let (new_left, captured) = detach_at(left.clone(), child_depth, p, record)?;
                let rebuilt = match new_left {
                    Some(new_left) => MrtNodeInner::branch(skip.clone(), new_left, right.clone()),
                    None => {
                        record(right);
                        collapse_survivor(skip, true, right.clone())?
                    }
                };
                Ok((Some(rebuilt), captured))
            } else {
                let (new_right, captured) = detach_at(right.clone(), child_depth, p, record)?;
                let rebuilt = match new_right {
                    Some(new_right) => MrtNodeInner::branch(skip.clone(), left.clone(), new_right),
                    None => {
                        record(left);
                        collapse_survivor(skip, false, left.clone())?
                    }
                };
                Ok((Some(rebuilt), captured))
            }
        }
    }
}

/// Delete exactly the keys with byte-prefix `prefix`, via the existing range
/// machinery. Equivalent to
/// `delete_range(prefix, prefix_successor(prefix))` when a successor exists, and
/// the **open-upper-bound** range `[prefix, ∞)` exactly when `prefix_successor` is
/// `None` — i.e. an all-`0xFF` prefix or the empty prefix `[]`, where "every key
/// `≥ prefix`" coincides with "every key with prefix `prefix`". (Do **not** take
/// the open upper bound otherwise: that would delete every key `≥ prefix`, not
/// just the prefix-`prefix` keys.)
pub(crate) fn delete_prefix(
    root: Option<Arc<MrtNodeInner>>,
    prefix: &[u8],
) -> Result<Option<Arc<MrtNodeInner>>> {
    validate_key_len(prefix, "delete_prefix")?;
    let Some(root) = root else {
        return Ok(None);
    };
    let mut record = |_prefix: &RoutePrefix, _arc: &Arc<MrtNodeInner>| {};
    let hi = crate::tracer::prefix_successor(prefix);
    delete_range_at(
        root,
        0,
        &RoutePrefix::root(),
        Some(prefix),
        hi.as_deref(),
        &mut record,
    )
}

// ─── prefix-subtree splice (`move_prefix` subtree relocate, stage 2) ─────────
//
// Stage 2 adds the destination-side host primitives; stage 3 wires them into the
// public `move_prefix` op. `splice_subtree_at` consumes the post-detach tree and
// the captured source subtree, descends `q` on that post-detach tree, proves
// destination emptiness by finding a real divergence, and re-skins only the
// captured root.

/// Re-skin `S`'s root skip by stripping the old source prefix tail and prepending
/// the destination tail. The shared suffix beyond `p` is untouched, so the
/// subtree interior remains hash-invariant; callers rebuild only the root node
/// from this new skip plus its original children/value.
pub(crate) fn reskin_root(
    skip: &RouteBits,
    strip_prefix_bits: u16,
    prepend_q_tail_bits: &RouteBits,
) -> Result<RouteBits> {
    if strip_prefix_bits > skip.bit_len() {
        return Err(Error::Tree(format!(
            "MRT reskin strip length {strip_prefix_bits} exceeds root skip length {}",
            skip.bit_len()
        )));
    }
    let shared_suffix = skip.slice(strip_prefix_bits, skip.bit_len());
    prepend_q_tail_bits.concat(&shared_suffix)
}

fn reskin_captured_root(
    captured: Captured,
    prepend_q_tail_bits: &RouteBits,
) -> Result<Arc<MrtNodeInner>> {
    let strip_prefix_bits = captured.strip_prefix_bits;
    match captured.s_root.node() {
        MrtNode::Leaf { skip, value } => Ok(MrtNodeInner::leaf(
            reskin_root(skip, strip_prefix_bits, prepend_q_tail_bits)?,
            value.clone(),
        )),
        MrtNode::Branch { skip, left, right } => Ok(MrtNodeInner::branch(
            reskin_root(skip, strip_prefix_bits, prepend_q_tail_bits)?,
            left.clone(),
            right.clone(),
        )),
        MrtNode::PrunedHash => Err(Error::PrunedNode("reskin captured subtree root".into())),
    }
}

fn destination_not_empty(q: &[u8]) -> Error {
    Error::Key(format!(
        "MRT move_prefix destination prefix {q:?} is not empty"
    ))
}

/// Splice a captured prefix subtree at byte-prefix `q` in the **post-detach**
/// tree. A non-empty destination tree accepts only a real mismatch before `q`
/// ends; if `q` is exhausted within an existing node, or an existing leaf key is
/// a prefix of `q`, the destination is not an empty prefix-free slot and the
/// splice fails. If the post-detach tree is empty, no connector branch is built:
/// the re-skinned captured root becomes the new root directly.
///
/// Test-only thin wrapper over [`splice_subtree_at_inner`] with a no-op record
/// hook (see [`detach_prefix_subtree`]).
#[cfg(test)]
pub(crate) fn splice_subtree_at(
    root: Option<Arc<MrtNodeInner>>,
    q: &[u8],
    captured: Captured,
) -> Result<Arc<MrtNodeInner>> {
    let mut record = |_: &Arc<MrtNodeInner>| {};
    splice_subtree_at_inner(root, q, captured, &mut record)
}

/// Shared core behind [`splice_subtree_at`] and the tracer's
/// `move_prefix_with_trace`: `record` is handed every node the `q`-descent walks
/// on the **post-detach** tree. Synthetic nodes the detach built (the re-skinned
/// survivor, rebuilt ancestors) are not snapshot `Arc`s, so installing them is a
/// no-op at assembly; the pre-state nodes the splice routes through (incl. the
/// survivor's original children, if the descent dips into them) are revealed.
fn splice_subtree_at_inner(
    root: Option<Arc<MrtNodeInner>>,
    q: &[u8],
    captured: Captured,
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<Arc<MrtNodeInner>> {
    validate_key_len(q, "splice_subtree_at destination prefix")?;
    match root {
        None => {
            let q_tail = RouteBits::from_key_range(q, 0, route_len(q));
            reskin_captured_root(captured, &q_tail)
        }
        Some(root) => splice_at(root, 0, q, captured, record),
    }
}

fn splice_at(
    cur: Arc<MrtNodeInner>,
    depth: u16,
    q: &[u8],
    captured: Captured,
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<Arc<MrtNodeInner>> {
    record(&cur);
    match cur.node() {
        MrtNode::Leaf { skip, value } => match classify_key_vs_skip(skip, q, depth) {
            SkipRelation::Diverge { offset } => {
                let existing =
                    MrtNodeInner::leaf(skip.slice(offset + 1, skip.bit_len()), value.clone());
                build_splice_connector(skip, offset, depth, q, existing, captured)
            }
            SkipRelation::Equal | SkipRelation::KeyIsPrefix => {
                // OVERWRITE semantics: the destination prefix `q` is occupied. The
                // existing subtree at `q` was handed to `record` above (so the
                // verifier authenticates its hash against the start root, then
                // discards it); replace it with the re-skinned moved subtree.
                let q_tail = RouteBits::from_key_range(q, depth, route_len(q));
                reskin_captured_root(captured, &q_tail)
            }
            SkipRelation::SkipIsPrefix => Err(prefix_free_violation(q)),
        },
        MrtNode::Branch { skip, left, right } => match classify_key_vs_skip(skip, q, depth) {
            SkipRelation::Diverge { offset } => {
                let existing = MrtNodeInner::branch(
                    skip.slice(offset + 1, skip.bit_len()),
                    left.clone(),
                    right.clone(),
                );
                build_splice_connector(skip, offset, depth, q, existing, captured)
            }
            SkipRelation::SkipIsPrefix => {
                let branch_depth = checked_branch_depth(depth, skip)?;
                let side = route_bit_at(q, branch_depth);
                let child_depth = checked_child_depth(branch_depth)?;
                if !side {
                    let new_left = splice_at(left.clone(), child_depth, q, captured, record)?;
                    Ok(MrtNodeInner::branch(skip.clone(), new_left, right.clone()))
                } else {
                    let new_right = splice_at(right.clone(), child_depth, q, captured, record)?;
                    Ok(MrtNodeInner::branch(skip.clone(), left.clone(), new_right))
                }
            }
            SkipRelation::Equal | SkipRelation::KeyIsPrefix => {
                // OVERWRITE semantics: the destination prefix `q` is occupied. The
                // existing subtree at `q` was handed to `record` above (so the
                // verifier authenticates its hash against the start root, then
                // discards it); replace it with the re-skinned moved subtree.
                let q_tail = RouteBits::from_key_range(q, depth, route_len(q));
                reskin_captured_root(captured, &q_tail)
            }
        },
        MrtNode::PrunedHash => Err(Error::PrunedNode(format!(
            "splice_subtree_at descent at prefix {q:?}"
        ))),
    }
}

fn build_splice_connector(
    existing_skip: &RouteBits,
    offset: u16,
    depth: u16,
    q: &[u8],
    existing: Arc<MrtNodeInner>,
    captured: Captured,
) -> Result<Arc<MrtNodeInner>> {
    let q_div = checked_add_depth(depth, offset)?;
    let q_bits = route_len(q);
    if q_div >= q_bits {
        return Err(destination_not_empty(q));
    }
    let e_dst = checked_child_depth(q_div)?;
    let q_tail = RouteBits::from_key_range(q, e_dst, q_bits);
    let moved = reskin_captured_root(captured, &q_tail)?;
    let q_side = route_bit_at(q, q_div);
    let existing_side = existing_skip.bit_at(offset);
    debug_assert_ne!(q_side, existing_side);

    let connector_skip = existing_skip.slice(0, offset);
    if !q_side {
        Ok(MrtNodeInner::branch(connector_skip, moved, existing))
    } else {
        Ok(MrtNodeInner::branch(connector_skip, existing, moved))
    }
}

// ─── move_prefix op (`detach` + `splice`, stage 3) ───────────────────────────
//
// `move_prefix(from, to)` relocates the whole prefix-`from` subtree to prefix
// `to` in O(depth) work: `detach_prefix_subtree` removes + captures `S`, then
// `splice_subtree_at` re-skins it at `to`. The same composition runs on the host
// (`Tree::move_prefix`), the tracer (`move_prefix_with_trace`), and the
// verifier replay (`verify_replay::MrtVerifyTree::move_prefix`), so a move is
// byte-identical across all three.

/// The stateless `move_prefix` preconditions, re-checked at **every** mutating
/// entry (host op, tracer apply, verifier replay) before the tree is touched —
/// a `MovePrefix` step is a public value constructible outside the proof builder,
/// so each path must validate identically. Prefixes may have different lengths;
/// the post-detach depth check gates any move that would push a moved key past
/// [`MAX_KEY_LEN`]. A no-op move is still rejected.
pub(crate) fn validate_move_prefix_args(from: &[u8], to: &[u8]) -> Result<()> {
    validate_key_len(from, "move_prefix source prefix")?;
    validate_key_len(to, "move_prefix destination prefix")?;
    if from == to {
        return Err(Error::Key(
            "MRT move_prefix source and destination prefixes must differ".into(),
        ));
    }
    Ok(())
}

/// Authenticated max-key-length gate for a captured `move_prefix` subtree.
///
/// `s_depth` is the captured root's relative `depth_below`; subtracting
/// `strip_prefix_bits` leaves the deepest suffix below `from`. The resulting
/// deepest moved key is `to` plus that suffix, computed widened so malformed
/// tracees and legitimate overflowing moves both become clean `Err`s.
pub(crate) fn validate_move_prefix_result_depth(
    to: &[u8],
    s_depth: u16,
    strip_prefix_bits: u16,
) -> Result<()> {
    let suffix = s_depth.checked_sub(strip_prefix_bits).ok_or_else(|| {
        Error::Tree(format!(
            "MRT move_prefix captured depth_below {s_depth} is smaller than stripped prefix bits {strip_prefix_bits}"
        ))
    })?;
    let to_bits = (to.len() as u32)
        .checked_mul(8)
        .ok_or_else(|| Error::Tree("MRT move_prefix destination bit length overflow".into()))?;
    let new_deep = to_bits
        .checked_add(u32::from(suffix))
        .ok_or_else(|| Error::Tree("MRT move_prefix resulting depth overflow".into()))?;
    if new_deep > u32::from(MAX_ROUTE_BITS) {
        return Err(Error::Key(format!(
            "MRT move_prefix deepest moved key would be {new_deep} bits, exceeding maximum {MAX_ROUTE_BITS}"
        )));
    }
    Ok(())
}

/// Relocate the whole prefix-`from` subtree to prefix `to` (CoW). `Err` on a
/// stateless precondition (`validate_move_prefix_args`), an **absent** `from`
/// (detach diverges, incl. an empty tree), or a non-empty / prefix-violating `to`
/// (splice). On success the moved keys are `to ‖ s` for each original `from ‖ s`,
/// values preserved, and the result is non-empty (a move never removes keys).
pub(crate) fn move_prefix(
    root: Option<Arc<MrtNodeInner>>,
    from: &[u8],
    to: &[u8],
) -> Result<Arc<MrtNodeInner>> {
    let mut record = |_: &Arc<MrtNodeInner>| {};
    move_prefix_with_trace(root, from, to, &mut record)
}

/// [`move_prefix`] threaded with a visit `record` hook for the tracer: detach and
/// splice hand every pre-state node they touch to `record` (the `p`-path, `S`'s
/// original root, the source-collapse survivor, and the `q`-splice path on the
/// post-detach tree). Installing those by `Arc` identity yields the O(depth)
/// trace; synthetic nodes the op builds are dropped at assembly and rebuilt by
/// the verifier at replay.
pub(crate) fn move_prefix_with_trace(
    root: Option<Arc<MrtNodeInner>>,
    from: &[u8],
    to: &[u8],
    record: &mut impl FnMut(&Arc<MrtNodeInner>),
) -> Result<Arc<MrtNodeInner>> {
    validate_move_prefix_args(from, to)?;
    let (root2, captured) = detach_prefix_subtree_inner(root, from, record)?;
    validate_move_prefix_result_depth(
        to,
        captured.s_root.depth_below(),
        captured.strip_prefix_bits,
    )?;
    splice_subtree_at_inner(root2, to, captured, record)
}

#[cfg(test)]
mod hash_vectors_v2_tests {
    use super::*;

    fn sha256_ref(preimage: &[u8]) -> Hash {
        let mut h = Sha256::new();
        h.update(preimage);
        h.finalize().into()
    }

    fn leaf_hash_v2_ref(skip: &RouteBits, value: &[u8]) -> Hash {
        let mut preimage = Vec::new();
        preimage.extend_from_slice(&(skip.bit_len() as u32).to_be_bytes());
        preimage.extend_from_slice(skip.packed_bytes());
        preimage.extend_from_slice(value);
        preimage.push(0x01);
        sha256_ref(&preimage)
    }

    fn branch_hash_v2_ref(
        left_hash: Hash,
        right_hash: Hash,
        left_depth: u16,
        right_depth: u16,
        skip: &RouteBits,
    ) -> Hash {
        let mut preimage = Vec::new();
        preimage.extend_from_slice(&left_hash);
        preimage.extend_from_slice(&right_hash);
        preimage.extend_from_slice(&(left_depth as u32).to_be_bytes());
        preimage.extend_from_slice(&(right_depth as u32).to_be_bytes());
        preimage.extend_from_slice(&(skip.bit_len() as u32).to_be_bytes());
        preimage.extend_from_slice(skip.packed_bytes());
        preimage.push(0x00);
        sha256_ref(&preimage)
    }

    #[test]
    fn hash_vectors_v2_leaf_matches_reference() {
        for (skip, value) in [
            (RouteBits::empty(), b"".as_slice()),
            (
                RouteBits::from_key_range(b"leaf-key", 0, 37),
                b"known-value",
            ),
            (
                RouteBits::from_key_range(b"\x00\xff\x10", 3, 24),
                b"value with length not framed in the hash",
            ),
            (
                RouteBits::from_key_range(b"byte-boundary", 0, 16),
                b"exact multiple of eight skip bits",
            ),
        ] {
            assert_eq!(
                compute_leaf_hash(&skip, value),
                leaf_hash_v2_ref(&skip, value)
            );
        }
    }

    #[test]
    fn hash_vectors_v2_branch_matches_reference() {
        for (skip, left_hash, right_hash, left_depth, right_depth) in [
            (RouteBits::empty(), [0x11; 32], [0x22; 32], 0u16, 0u16),
            (
                RouteBits::from_key_range(b"branch-key", 0, 37),
                [0x33; 32],
                [0x44; 32],
                1,
                65535,
            ),
            (
                RouteBits::from_key_range(b"\x80\x00\x7f", 5, 20),
                [0xa5; 32],
                [0x5a; 32],
                12345,
                256,
            ),
            (
                RouteBits::from_key_range(b"byte-boundary", 0, 16),
                [0xc3; 32],
                [0x3c; 32],
                8,
                32768,
            ),
        ] {
            assert_eq!(
                compute_branch_hash(&left_hash, &right_hash, left_depth, right_depth, &skip),
                branch_hash_v2_ref(left_hash, right_hash, left_depth, right_depth, &skip)
            );
        }
    }

    // ── zkVM buffer-assembly simulation ──
    //
    // The zkVM compute_*_hash code paths can't run natively, but the bug-prone
    // part is the buffer assembly (data layout + FIPS 180-4 padding, including
    // the stack/heap boundary at data_len = 119). These helpers replicate the
    // exact buffer construction from the zkVM functions, hash the result with
    // the Digest trait, and verify it matches the byte-assembled reference.
    // Combined with `*_matches_reference` above (non-zkVM production == ref),
    // this pins the host/guest cross-environment hash-equality invariant.

    fn sha256_padded_buf(buf: &[u8], data_len: usize) -> Hash {
        let padded = (data_len + 9).div_ceil(64) * 64;
        assert!(buf.len() >= padded);
        assert_eq!(buf[data_len], 0x80);
        for (i, &b) in buf[data_len + 1..padded - 8].iter().enumerate() {
            assert_eq!(b, 0, "non-zero padding at byte {}", data_len + 1 + i);
        }
        let expected_bits = (data_len as u64 * 8).to_be_bytes();
        assert_eq!(&buf[padded - 8..padded], &expected_bits);
        sha256_ref(&buf[..data_len])
    }

    // Replicates the zkVM compute_leaf_hash buffer assembly (stack + heap).
    fn leaf_hash_v2_zkvm_sim(skip: &RouteBits, value: &[u8]) -> Hash {
        let bit_len_be = (skip.bit_len() as u32).to_be_bytes();
        let packed = skip.packed_bytes();
        let value_start = 4 + packed.len();
        let tag_pos = value_start + value.len();
        let data_len = tag_pos + 1;
        if data_len <= 119 {
            let padded = (data_len + 9).div_ceil(64) * 64;
            let mut buf = [0xFFu8; 128]; // simulate MaybeUninit with non-zero fill
            buf[..4].copy_from_slice(&bit_len_be);
            buf[4..value_start].copy_from_slice(packed);
            buf[value_start..tag_pos].copy_from_slice(value);
            buf[tag_pos] = 0x01;
            buf[data_len] = 0x80;
            buf[data_len + 1..padded - 8].fill(0);
            buf[padded - 8..padded].copy_from_slice(&((data_len as u64 * 8).to_be_bytes()));
            sha256_padded_buf(&buf, data_len)
        } else {
            let mut tmp = vec![0u8; data_len];
            tmp[..4].copy_from_slice(&bit_len_be);
            tmp[4..value_start].copy_from_slice(packed);
            tmp[value_start..tag_pos].copy_from_slice(value);
            tmp[tag_pos] = 0x01;
            sha256_ref(&tmp)
        }
    }

    // Replicates the zkVM compute_branch_hash buffer assembly (stack + heap).
    fn branch_hash_v2_zkvm_sim(
        left_hash: Hash,
        right_hash: Hash,
        left_depth: u16,
        right_depth: u16,
        skip: &RouteBits,
    ) -> Hash {
        let left_depth_be = (left_depth as u32).to_be_bytes();
        let right_depth_be = (right_depth as u32).to_be_bytes();
        let bit_len_be = (skip.bit_len() as u32).to_be_bytes();
        let packed = skip.packed_bytes();
        // left(32) + right(32) + left_depth(4) + right_depth(4) + bit_len(4)
        let skip_start = 76;
        let tag_pos = skip_start + packed.len();
        let data_len = tag_pos + 1;
        if data_len <= 119 {
            let padded = (data_len + 9).div_ceil(64) * 64;
            let mut buf = [0xFFu8; 128];
            buf[..32].copy_from_slice(&left_hash);
            buf[32..64].copy_from_slice(&right_hash);
            buf[64..68].copy_from_slice(&left_depth_be);
            buf[68..72].copy_from_slice(&right_depth_be);
            buf[72..skip_start].copy_from_slice(&bit_len_be);
            buf[skip_start..tag_pos].copy_from_slice(packed);
            buf[tag_pos] = 0x00;
            buf[data_len] = 0x80;
            buf[data_len + 1..padded - 8].fill(0);
            buf[padded - 8..padded].copy_from_slice(&((data_len as u64 * 8).to_be_bytes()));
            sha256_padded_buf(&buf, data_len)
        } else {
            let mut tmp = vec![0u8; data_len];
            tmp[..32].copy_from_slice(&left_hash);
            tmp[32..64].copy_from_slice(&right_hash);
            tmp[64..68].copy_from_slice(&left_depth_be);
            tmp[68..72].copy_from_slice(&right_depth_be);
            tmp[72..skip_start].copy_from_slice(&bit_len_be);
            tmp[skip_start..tag_pos].copy_from_slice(packed);
            tmp[tag_pos] = 0x00;
            sha256_ref(&tmp)
        }
    }

    #[test]
    fn zkvm_leaf_hash_assembly_sweep() {
        let key = vec![0x5au8; 256];
        // Skip bit-lengths straddling byte boundaries; value sizes straddling
        // the stack/heap boundary (leaf data_len = 5 + packed + value <= 119).
        for &bits in &[0u16, 1, 7, 8, 9, 15, 33, 100, 289, 512] {
            let skip = RouteBits::from_key_range(&key, 0, bits);
            for &val_size in &[0usize, 1, 50, 113, 114, 115, 200] {
                let value = vec![(val_size & 0xFF) as u8; val_size];
                assert_eq!(
                    leaf_hash_v2_zkvm_sim(&skip, &value),
                    leaf_hash_v2_ref(&skip, &value),
                    "leaf zkvm assembly mismatch at bits={}, val_size={}",
                    bits,
                    val_size
                );
            }
        }
    }

    #[test]
    fn zkvm_branch_hash_assembly_sweep() {
        let key = vec![0xa5u8; 256];
        // branch data_len = 77 + packed(skip) (the fixed prefix grew by the two
        // u32 child-depth words); stack/heap boundary at packed = 42 → bit_len in
        // [329, 336] sits at the stack max (data_len 119), 337 spills to heap.
        for &bits in &[0u16, 1, 8, 9, 100, 328, 336, 337, 344, 512] {
            let skip = RouteBits::from_key_range(&key, 0, bits);
            let left = [(bits & 0xFF) as u8; 32];
            let right = [((bits >> 1) & 0xFF) as u8; 32];
            // Vary the committed child depths per case, incl. the u16 ceiling.
            let left_depth = bits.wrapping_mul(3);
            let right_depth = 65535u16.wrapping_sub(bits);
            assert_eq!(
                branch_hash_v2_zkvm_sim(left, right, left_depth, right_depth, &skip),
                branch_hash_v2_ref(left, right, left_depth, right_depth, &skip),
                "branch zkvm assembly mismatch at bits={}",
                bits
            );
        }
    }

    // The materialized-preimage path (lever 1 prototype): `build_branch_preimage`
    // + child slots written into `[0..64]` + `hash_branch_preimage_into` must equal
    // `compute_branch_hash` for the same inputs. On host this pins the host
    // primitive; the preimage layout it checks is shared with the zkVM syscall
    // variant (only the final SHA call is cfg-split), so it also guards that layout.
    #[test]
    fn branch_preimage_path_matches_compute_branch_hash() {
        let key = vec![0xa5u8; 256];
        // Span empty/partial/full skip and the 1-block vs 2-block tail boundary.
        for &bits in &[0u16, 1, 8, 9, 100, 392, 393, 400, 401, 512, 1024] {
            let skip = RouteBits::from_key_range(&key, 0, bits);
            let left = [(bits & 0xFF) as u8; 32];
            let right = [((bits >> 1) & 0xFF) as u8; 32];
            let left_depth = bits.wrapping_mul(3);
            let right_depth = 65535u16.wrapping_sub(bits);

            let mut pre = build_branch_preimage(&skip);
            // Write the child hashes into the [0..32]/[32..64] slots and the two
            // child depth_below words into [64..68]/[68..72], exactly as
            // `rehash_into` does on every dirty rehash (there: hashes in place via
            // the syscall `out_state`, depths rewritten from the children).
            let p = pre.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(left.as_ptr(), p, 32);
                core::ptr::copy_nonoverlapping(right.as_ptr(), p.add(32), 32);
                core::ptr::copy_nonoverlapping(
                    (left_depth as u32).to_be_bytes().as_ptr(),
                    p.add(64),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    (right_depth as u32).to_be_bytes().as_ptr(),
                    p.add(68),
                    4,
                );
            }
            #[repr(align(4))]
            struct Aligned([u8; 32]);
            let mut out = Aligned([0u8; 32]);
            // SAFETY: `out.0` is 32 bytes, 4-aligned, disjoint from `pre`.
            unsafe { hash_branch_preimage_into(out.0.as_mut_ptr(), &pre, &skip) };

            assert_eq!(
                out.0,
                compute_branch_hash(&left, &right, left_depth, right_depth, &skip),
                "preimage path mismatch at bits={bits}"
            );
        }
    }
}

#[cfg(test)]
mod route_bits_tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, RngCore, SeedableRng};

    fn random_key<R: RngCore>(rng: &mut R) -> Vec<u8> {
        let len = rng.gen_range(0..=18);
        let mut key = vec![0u8; len];
        rng.fill_bytes(&mut key);
        key
    }

    #[test]
    fn route_bits_of_has_canonical_length_and_inverts() {
        for key in [
            b"".as_slice(),
            b"a",
            b"ab",
            b"\x00",
            b"\x00\x00",
            b"\xff",
            b"\xff\x00\xff",
            b"key",
        ] {
            let bits = route_bits_of(key);
            assert_eq!(bits.bit_len(), (8 * key.len()) as u16);
            assert_eq!(key_from_route_bits(bits).unwrap(), key);
        }
    }

    #[test]
    fn route_bits_round_trip_random_keys() {
        let mut rng = SmallRng::seed_from_u64(0x5EED_5EED);
        for _ in 0..5000 {
            let key = random_key(&mut rng);
            let bits = route_bits_of(&key);
            assert_eq!(key_from_route_bits(bits).unwrap(), key);
        }
    }

    #[test]
    fn route_bits_distinguishes_prefix_pairs() {
        // A key and a key that extends it must yield different routes and decode
        // back to themselves — the route's bit-length distinguishes a key from one
        // that extends it (raw 8-bit-per-byte route, no terminator).
        let a = route_bits_of(b"ab");
        let ab = route_bits_of(b"abc");
        assert_ne!(a, ab);
        assert_eq!(key_from_route_bits(a).unwrap(), b"ab");
        assert_eq!(key_from_route_bits(ab).unwrap(), b"abc");
    }

    #[test]
    fn key_from_route_bits_rejects_non_canonical_length() {
        // A route's packed bytes *are* the key, so a non-byte-aligned length
        // (bit_len % 8 != 0) has no key and is rejected — here bit_len 2 and 9.
        assert!(key_from_route_bits(RouteBits::from_packed(2, &[0x00]).unwrap()).is_err());
        assert!(key_from_route_bits(RouteBits::from_packed(9, &[0x80, 0x00]).unwrap()).is_err());
    }

    #[test]
    fn key_from_route_bits_rejects_non_byte_aligned_length() {
        // The route's packed bytes *are* the key, so a route that doesn't end on a
        // byte boundary (bit_len % 8 != 0) has no key and is rejected. A 10-bit
        // route ('a' plus two zero bits) is not a whole number of bytes.
        let trailing = RouteBits::from_packed(10, &[b'a', 0x00]).unwrap();
        assert_eq!(trailing.bit_len(), 10);
        assert!(key_from_route_bits(trailing).is_err());
    }

    #[test]
    fn route_prefix_reconstructs_absolute_key_single_branch() {
        let full = route_bits_of(b"hi");
        let p = 5u16;
        let branch_skip = full.slice(0, p);
        let side = full.bit_at(p);
        let leaf_skip = full.slice(p + 1, full.bit_len());

        let prefix = RoutePrefix::root().descend(&branch_skip, side).unwrap();
        assert_eq!(prefix.bits().bit_len(), p + 1);
        assert_eq!(prefix.key_with_suffix(&leaf_skip).unwrap(), b"hi");
    }

    #[test]
    fn route_prefix_reconstructs_absolute_key_multi_branch() {
        let full = route_bits_of(b"hi");
        let (p1, p2) = (3u16, 10u16);
        let b1 = full.slice(0, p1);
        let s1 = full.bit_at(p1);
        let b2 = full.slice(p1 + 1, p2);
        let s2 = full.bit_at(p2);
        let leaf = full.slice(p2 + 1, full.bit_len());

        let key = RoutePrefix::root()
            .descend(&b1, s1)
            .unwrap()
            .descend(&b2, s2)
            .unwrap()
            .key_with_suffix(&leaf)
            .unwrap();
        assert_eq!(key, b"hi");
    }

    #[test]
    fn route_prefix_leaf_at_root() {
        // A leaf directly under the root: the prefix is empty and the suffix is
        // the whole route.
        let key = RoutePrefix::root()
            .key_with_suffix(&route_bits_of(b"x"))
            .unwrap();
        assert_eq!(key, b"x");
    }

    // ── Per-symbol vs per-bit differential ──
    //
    // `from_key_range` and `matches_key_at` walk a route one byte (8-bit group) at
    // a time; these references are the obvious one-bit-at-a-time forms (the pre-
    // optimization code). The per-symbol versions must be bit-identical to them,
    // including the exact first-mismatch `offset` (it drives the insert split)
    // and the untrusted-trace overflow guard.

    fn from_key_range_ref(key: &[u8], start: u16, end: u16) -> RouteBits {
        let bit_len = end - start;
        let mut bytes = vec![0u8; packed_len(bit_len)];
        for offset in 0..bit_len {
            if route_bit_at(key, start + offset) {
                set_packed_bit(&mut bytes, offset, true);
            }
        }
        RouteBits { bit_len, bytes }
    }

    fn matches_key_at_ref(skip: &RouteBits, key: &[u8], depth: u16) -> MatchResult {
        for offset in 0..skip.bit_len {
            let Some(position) = depth.checked_add(offset) else {
                return MatchResult::Mismatch { offset };
            };
            if skip.bit_at(offset) != route_bit_at(key, position) {
                return MatchResult::Mismatch { offset };
            }
        }
        MatchResult::FullMatch
    }

    #[test]
    fn from_key_range_per_symbol_matches_per_bit_reference() {
        // Exhaustive (start, end) over a multi-byte key — every leading-group
        // alignment, full groups, the final partial group, and beyond-route bits.
        let key: Vec<u8> = (0..20u8).collect();
        let total = route_len(&key);
        for start in 0..=total {
            for end in start..=total {
                assert_eq!(
                    RouteBits::from_key_range(&key, start, end),
                    from_key_range_ref(&key, start, end),
                    "from_key_range start={start} end={end}"
                );
            }
        }
        // Random keys (incl. empty and 0x00/0xFF bytes) for data-bit coverage.
        let mut rng = SmallRng::seed_from_u64(0xF00D_CAFE);
        for _ in 0..3000 {
            let key = random_key(&mut rng);
            let total = route_len(&key);
            let start = rng.gen_range(0..=total);
            let end = rng.gen_range(start..=total);
            assert_eq!(
                RouteBits::from_key_range(&key, start, end),
                from_key_range_ref(&key, start, end),
                "from_key_range key={key:?} start={start} end={end}"
            );
        }
    }

    #[test]
    fn matches_key_at_per_symbol_matches_per_bit_reference() {
        let mut rng = SmallRng::seed_from_u64(0xBEEF_F00D);
        for _ in 0..20000 {
            let src = random_key(&mut rng);
            let src_total = route_len(&src);
            let start = rng.gen_range(0..=src_total);
            let end = rng.gen_range(start..=src_total);
            // Build the skip with the reference so the test isolates matches_key_at.
            let skip = from_key_range_ref(&src, start, end);
            // Same key (full/partial match) or a different one (mismatch).
            let query = if rng.gen_bool(0.5) {
                src.clone()
            } else {
                random_key(&mut rng)
            };
            // Usually the natural matching depth `start`; sometimes perturbed to
            // force mismatches at assorted offsets and alignments.
            let depth = if rng.gen_bool(0.7) {
                start
            } else {
                rng.gen_range(0..=src_total)
            };
            assert_eq!(
                skip.matches_key_at(&query, depth),
                matches_key_at_ref(&skip, &query, depth),
                "matches src={src:?} start={start} end={end} query={query:?} depth={depth}"
            );
        }
    }

    #[test]
    fn matches_key_at_overflow_guard_matches_reference() {
        // depth + offset past u16::MAX is only reachable from an untrusted
        // trace. The all-zero skip walks the (also-zero) beyond-route bits up to
        // the overflow point, exercising the guard's exact miss offset; the keyed
        // skip mismatches earlier. Both must agree with the reference.
        let zero = RouteBits::from_packed(10_000, &vec![0u8; packed_len(10_000)]).unwrap();
        let src = vec![0x9Cu8; 1000];
        let keyed = from_key_range_ref(&src, 0, route_len(&src));
        let query = vec![0x9Cu8; 1000];
        for skip in [&zero, &keyed] {
            for depth in [
                u16::MAX,
                u16::MAX - 1,
                u16::MAX - 7,
                u16::MAX - 8,
                65000,
                60000,
                56536,
                56535,
            ] {
                assert_eq!(
                    skip.matches_key_at(&query, depth),
                    matches_key_at_ref(skip, &query, depth),
                    "overflow skip_len={} depth={depth}",
                    skip.bit_len()
                );
            }
        }
    }

    // ── Chunked slice vs per-bit reference (lever 2a) ──
    //
    // `slice`'s unaligned path now copies one byte-group per step; it must stay
    // bit-identical to the obvious one-bit-at-a-time form, across every source
    // alignment (the aligned fast path and the chunked unaligned path).

    fn slice_ref(src: &RouteBits, start: u16, end: u16) -> RouteBits {
        let bit_len = end - start;
        let mut bytes = vec![0u8; packed_len(bit_len)];
        for offset in 0..bit_len {
            if src.bit_at(start + offset) {
                set_packed_bit(&mut bytes, offset, true);
            }
        }
        RouteBits { bit_len, bytes }
    }

    #[test]
    fn slice_chunked_matches_per_bit_reference() {
        // Exhaustive (start, end) over a multi-byte route — every start alignment
        // (incl. the byte-aligned fast path) and every window length.
        let key: Vec<u8> = (0..24u8).collect();
        let src = RouteBits::from_key_range(&key, 0, route_len(&key));
        let total = src.bit_len();
        for start in 0..=total {
            for end in start..=total {
                assert_eq!(
                    src.slice(start, end),
                    slice_ref(&src, start, end),
                    "slice start={start} end={end}"
                );
            }
        }
        // Random routes (incl. empty and 0x00/0xFF bytes) for data-bit coverage.
        let mut rng = SmallRng::seed_from_u64(0x5111_CE5E);
        for _ in 0..3000 {
            let key = random_key(&mut rng);
            let src = RouteBits::from_key_range(&key, 0, route_len(&key));
            let total = src.bit_len();
            let start = rng.gen_range(0..=total);
            let end = rng.gen_range(start..=total);
            assert_eq!(
                src.slice(start, end),
                slice_ref(&src, start, end),
                "slice key={key:?} start={start} end={end}"
            );
        }
    }

    // `blit_bits` (shift-accumulate + byte-aligned fast path) must stay bit-identical
    // to the obvious one-bit-at-a-time OR, across every destination alignment and
    // length — including the partial-first-byte preservation and the last-byte carry
    // that doesn't spill into a new byte.
    fn blit_bits_ref(dst: &mut [u8], dst_off: u16, src: &RouteBits) {
        for offset in 0..src.bit_len {
            if src.bit_at(offset) {
                set_packed_bit(dst, dst_off + offset, true);
            }
        }
    }

    fn blit_case(dst_off: u16, src: &RouteBits) -> (Vec<u8>, Vec<u8>) {
        let len = packed_len(dst_off + src.bit_len());
        // Pre-set the bits before dst_off (must survive); the target range stays zero.
        let mut prefilled = vec![0u8; len];
        for o in 0..dst_off {
            set_packed_bit(&mut prefilled, o, true);
        }
        let mut a = prefilled.clone();
        let mut b = prefilled;
        blit_bits(&mut a, dst_off, src);
        blit_bits_ref(&mut b, dst_off, src);
        (a, b)
    }

    #[test]
    fn blit_bits_chunked_matches_per_bit_reference() {
        // Exhaustive: every destination bit-offset (every `shift`, with/without a
        // leading byte) × every source length carved from a multi-byte route.
        let key: Vec<u8> = (0..24u8).collect();
        let full = RouteBits::from_key_range(&key, 0, route_len(&key));
        for dst_off in 0u16..24 {
            for src_bits in 0u16..=full.bit_len() {
                let src = full.slice(0, src_bits);
                let (a, b) = blit_case(dst_off, &src);
                assert_eq!(a, b, "blit_bits dst_off={dst_off} src_bits={src_bits}");
            }
        }
        // Random routes (incl. empty and 0x00/0xFF bytes) for data-bit coverage.
        let mut rng = SmallRng::seed_from_u64(0xB117_5EED);
        for _ in 0..5000 {
            let key = random_key(&mut rng);
            let route = RouteBits::from_key_range(&key, 0, route_len(&key));
            let src_bits = rng.gen_range(0..=route.bit_len());
            let src = route.slice(0, src_bits);
            let dst_off = rng.gen_range(0u16..24);
            let (a, b) = blit_case(dst_off, &src);
            assert_eq!(
                a, b,
                "blit_bits key={key:?} dst_off={dst_off} src_bits={src_bits}"
            );
        }
    }

    #[test]
    fn split_at_into_matches_split_at() {
        // The buffer-reusing consuming split must return values bit-identical to
        // the borrowing `split_at`, including the prefix's final-byte tail-clear.
        let mut rng = SmallRng::seed_from_u64(0x5917_A7A7);
        for _ in 0..5000 {
            let key = random_key(&mut rng);
            let src = RouteBits::from_key_range(&key, 0, route_len(&key));
            let total = src.bit_len();
            // `split_at`/`split_at_into` require `bit_offset < bit_len`; an empty
            // key now has a zero-length route, so skip it.
            if total == 0 {
                continue;
            }
            let off = rng.gen_range(0..total);
            let (p_ref, b_ref, s_ref) = src.split_at(off);
            let (p_into, b_into, s_into) = src.clone().split_at_into(off);
            assert_eq!(p_ref, p_into, "prefix key={key:?} off={off}");
            assert_eq!(b_ref, b_into, "bit key={key:?} off={off}");
            assert_eq!(s_ref, s_into, "suffix key={key:?} off={off}");
        }
    }

    #[test]
    fn into_suffix_matches_slice() {
        // The buffer-reusing in-place left-shift must be bit-identical to
        // `slice(start, bit_len)`, across every start alignment incl. start==0 and
        // the empty (start==bit_len) suffix, with the final-byte tail cleared.
        let mut rng = SmallRng::seed_from_u64(0x050F_F5E7);
        for _ in 0..6000 {
            let key = random_key(&mut rng);
            let src = RouteBits::from_key_range(&key, 0, route_len(&key));
            let total = src.bit_len();
            let start = rng.gen_range(0..=total);
            assert_eq!(
                src.slice(start, total),
                src.clone().into_suffix(start),
                "key={key:?} start={start}"
            );
        }
    }
}
