use std::convert::TryFrom;
use std::io::{self, Read, Write};
use std::sync::Arc;

use ed::{Decode, Encode};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::tree::{self, MrtNode, MrtNodeInner, RouteBits};
#[cfg(test)]
use super::{apply_batch_to_root, validate_mrt_apply_batch};
use crate::error::{Error, Result};
use crate::hash::{Hash, HASH_LENGTH, NULL_HASH};
#[cfg(test)]
use crate::ops::Batch;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trace(pub(crate) Option<Arc<MrtNodeInner>>);

impl Trace {
    // `Trace` is the trace *wire* type: build it from a host root, (de)serialize
    // it, and check its root. Ordered reads over a (decoded) trace go through the
    // shared `cursor::{point_get, collect_range, ...}` helpers — the host snapshot, the
    // query-proof verify reader, and the verify tree all use those, so there is one
    // read implementation rather than one per tree type.
    pub fn root_hash(&self) -> Hash {
        self.0.as_ref().map_or(NULL_HASH, |node| node.hash())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    pub fn verify_root(&self, expected: Hash) -> Result<()> {
        let actual = self.root_hash();
        if actual == expected {
            Ok(())
        } else {
            Err(Error::HashMismatch(expected, actual))
        }
    }

    pub fn decode_exact(bytes: &[u8]) -> Result<Self> {
        let mut cursor = bytes;
        let trace = <Trace as Decode>::decode(&mut cursor)?;
        let mut probe = [0u8; 1];
        if Read::read(&mut cursor, &mut probe)? != 0 {
            return Err(Error::Ed(ed_invalid(
                "MRT trace decode_exact: trailing bytes after value",
            )));
        }
        Ok(trace)
    }

    /// Encode this trace to its wire form — the inverse of [`Self::decode_exact`],
    /// and the bytes [`crate::mrt::TraceVerifier::decode_trace`] consumes.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        Encode::encode_into(self, &mut bytes)?;
        Ok(bytes)
    }
}

// Test-only read conveniences over a (decoded) trace — production reads go through
// the shared `cursor::*` helpers (snapshot, query-verify, verify tree); these just let
// tests assert host-tree reads against the verify tree without re-deriving the calls.
#[cfg(test)]
impl Trace {
    pub(crate) fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        super::cursor::point_get(self.0.as_ref(), key)
    }

    pub(crate) fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        super::cursor::collect_range(self.0.as_ref(), start, end)
    }

    pub(crate) fn collect_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        super::cursor::collect_prefix(self.0.as_ref(), prefix)
    }

    pub(crate) fn collect_all(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        super::cursor::collect_all(self.0.as_ref())
    }
}

#[derive(Serialize, Deserialize)]
enum MrtTraceWire {
    Empty,
    Leaf {
        skip: RouteBitsWire,
        value: Vec<u8>,
    },
    Branch {
        skip: RouteBitsWire,
        left: Box<MrtTraceWire>,
        right: Box<MrtTraceWire>,
    },
    Pruned {
        hash: Hash,
        // u32-BE to match the hash/wire framing; narrowed + range-checked in
        // `node_from_wire`. Only a pruned stub carries depth on the wire — a
        // materialized child's depth is derived from the child itself.
        depth_below: u32,
    },
}

#[derive(Serialize, Deserialize)]
struct RouteBitsWire {
    // u32-BE to match the hash/wire framing (§3); narrowed to the internal u16
    // in `into_route`.
    bit_len: u32,
    bytes: Vec<u8>,
}

impl MrtTraceWire {
    fn from_trace(trace: &Trace) -> Self {
        match trace.0.as_ref() {
            Some(node) => Self::from_node(node),
            None => MrtTraceWire::Empty,
        }
    }

    fn from_node(node: &Arc<MrtNodeInner>) -> Self {
        match node.node() {
            MrtNode::Leaf { skip, value } => MrtTraceWire::Leaf {
                skip: RouteBitsWire::from_route(skip),
                value: value.clone(),
            },
            MrtNode::Branch { skip, left, right } => MrtTraceWire::Branch {
                skip: RouteBitsWire::from_route(skip),
                left: Box::new(MrtTraceWire::from_node(left)),
                right: Box::new(MrtTraceWire::from_node(right)),
            },
            MrtNode::PrunedHash => MrtTraceWire::Pruned {
                hash: node.hash(),
                depth_below: u32::from(node.depth_below()),
            },
        }
    }
}

impl RouteBitsWire {
    fn from_route(route: &RouteBits) -> Self {
        Self {
            bit_len: u32::from(route.bit_len()),
            bytes: route.packed_bytes().to_vec(),
        }
    }

    fn into_route(self) -> Result<RouteBits> {
        let bit_len = u16::try_from(self.bit_len).map_err(|_| {
            Error::Tree(format!(
                "MRT trace route bit length {} exceeds u16",
                self.bit_len
            ))
        })?;
        RouteBits::from_packed(bit_len, &self.bytes)
            .map_err(|err| Error::Tree(format!("MRT trace route bits invalid: {err}")))
    }
}

fn trace_from_wire(wire: MrtTraceWire) -> Result<Trace> {
    match wire {
        MrtTraceWire::Empty => Ok(Trace(None)),
        // A bare pruned root carries no parent edge to authenticate its depth (and
        // supports no operation), so it is malleable — reject it at the top level,
        // matching the flat `ed` decoder. `Trace` is a public serde surface, so
        // this guard must hold here too, not only in `Decode`.
        MrtTraceWire::Pruned { .. } => Err(Error::Tree(
            "MRT trace root must be a materialized node or empty, not a pruned stub".into(),
        )),
        other => Ok(Trace(Some(node_from_wire(other, 0)?))),
    }
}

fn node_from_wire(wire: MrtTraceWire, depth: usize) -> Result<Arc<MrtNodeInner>> {
    if depth > MAX_MRT_TRACE_DECODE_DEPTH {
        return Err(Error::Tree(format!(
            "MRT trace serde decode depth {depth} exceeds limit {MAX_MRT_TRACE_DECODE_DEPTH}"
        )));
    }

    match wire {
        MrtTraceWire::Empty => Err(Error::Tree(
            "MRT trace empty marker cannot appear below the root".into(),
        )),
        MrtTraceWire::Leaf { skip, value } => Ok(MrtNodeInner::leaf(skip.into_route()?, value)),
        MrtTraceWire::Branch { skip, left, right } => {
            let left = node_from_wire(*left, depth + 1)?;
            let right = node_from_wire(*right, depth + 1)?;
            // `decode_branch` validates the rolled-up depth_below (checked) over the
            // already-decoded children.
            tree::decode_branch(skip.into_route()?, left, right)
        }
        MrtTraceWire::Pruned { hash, depth_below } => {
            let depth_below = u16::try_from(depth_below).map_err(|_| {
                Error::Tree(format!(
                    "MRT trace pruned depth_below {depth_below} exceeds u16"
                ))
            })?;
            tree::decode_pruned(hash, depth_below)
        }
    }
}

impl Serialize for Trace {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MrtTraceWire::from_trace(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Trace {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MrtTraceWire::deserialize(deserializer)?;
        trace_from_wire(wire).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
pub(crate) fn replay(trace: Trace, batch: &Batch) -> Result<Trace> {
    if batch.is_empty() {
        return Ok(trace);
    }
    validate_mrt_apply_batch(batch)?;
    Ok(Trace(apply_batch_to_root(trace.0, batch)?))
}

fn ed_invalid<E: std::fmt::Display>(msg: E) -> ed::Error {
    ed::Error::IOError(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{msg}"),
    ))
}

fn try_u32(len: usize, what: &str) -> ed::Result<u32> {
    u32::try_from(len)
        .map_err(|_| ed_invalid(format!("MRT trace {what} length {len} exceeds u32::MAX")))
}

fn read_exact_capped<R: Read + ?Sized>(input: &mut R, len: usize) -> ed::Result<Vec<u8>> {
    const CHUNK: usize = 4096;
    let mut out = Vec::new();
    let mut remaining = len;
    let mut buf = [0u8; CHUNK];
    while remaining > 0 {
        let take = remaining.min(CHUNK);
        input.read_exact(&mut buf[..take])?;
        out.extend_from_slice(&buf[..take]);
        remaining -= take;
    }
    Ok(out)
}

fn encode_node<W: Write>(node: &Arc<MrtNodeInner>, dest: &mut W) -> ed::Result<()> {
    match node.node() {
        // Leaf: tag ‖ skip.bit_len(u32-BE) ‖ skip.packed ‖ value_len(u32-BE) ‖
        // value. Unlike the *hash* preimage, the wire keeps `value_len` so the
        // decoder can frame the trailing value.
        MrtNode::Leaf { skip, value } => {
            let value_len = try_u32(value.len(), "value")?;
            dest.write_all(&[0x01])?;
            write_skip(skip, dest)?;
            dest.write_all(&value_len.to_be_bytes())?;
            dest.write_all(value)?;
        }
        MrtNode::Branch { skip, left, right } => {
            dest.write_all(&[0x02])?;
            write_skip(skip, dest)?;
            encode_node(left, dest)?;
            encode_node(right, dest)?;
        }
        MrtNode::PrunedHash => {
            // tag ‖ hash(32) ‖ depth_below(u32-BE). A pruned stub is the only wire
            // node carrying depth; a materialized child's depth is derived.
            dest.write_all(&[0x03])?;
            dest.write_all(&node.hash())?;
            dest.write_all(&u32::from(node.depth_below()).to_be_bytes())?;
        }
    }
    Ok(())
}

/// Writes a `RouteBits` as `bit_len(u32-BE) ‖ packed` — the shared skip framing
/// for leaf and branch wire encodings.
fn write_skip<W: Write>(skip: &RouteBits, dest: &mut W) -> ed::Result<()> {
    dest.write_all(&u32::from(skip.bit_len()).to_be_bytes())?;
    dest.write_all(skip.packed_bytes())?;
    Ok(())
}

fn node_encoding_length(node: &Arc<MrtNodeInner>) -> ed::Result<usize> {
    Ok(match node.node() {
        MrtNode::Leaf { skip, value } => {
            try_u32(value.len(), "value")?;
            1 + skip.encoded_len() + 4 + value.len()
        }
        MrtNode::Branch { skip, left, right } => {
            1 + skip.encoded_len() + node_encoding_length(left)? + node_encoding_length(right)?
        }
        MrtNode::PrunedHash => 1 + HASH_LENGTH + 4,
    })
}

impl Encode for Trace {
    fn encode_into<W: Write>(&self, dest: &mut W) -> ed::Result<()> {
        match &self.0 {
            None => dest.write_all(&[0x00])?,
            Some(node) => encode_node(node, dest)?,
        }
        Ok(())
    }

    fn encoding_length(&self) -> ed::Result<usize> {
        match &self.0 {
            None => Ok(1),
            Some(node) => node_encoding_length(node),
        }
    }
}

pub const MAX_MRT_TRACE_DECODE_DEPTH: usize = 1024;

fn decode_node_dyn(input: &mut dyn Read, depth: usize) -> ed::Result<Arc<MrtNodeInner>> {
    let mut tag = [0u8; 1];
    input.read_exact(&mut tag)?;
    decode_node_with_tag_dyn(tag[0], input, depth)
}

/// Reads a `RouteBits` framed as `bit_len(u32-BE) ‖ packed`. The u32 length is
/// narrowed to the internal u16 (numerically `bit_len <= MAX_ROUTE_BITS`), and
/// `from_packed` rejects over-long or non-canonical packing.
fn read_skip_dyn(input: &mut dyn Read) -> ed::Result<RouteBits> {
    let mut bit_len_buf = [0u8; 4];
    input.read_exact(&mut bit_len_buf)?;
    let bit_len_u32 = u32::from_be_bytes(bit_len_buf);
    let bit_len = u16::try_from(bit_len_u32).map_err(|_| {
        ed_invalid(format!(
            "MRT trace route bit length {bit_len_u32} exceeds u16"
        ))
    })?;
    let packed_len = (bit_len as usize).div_ceil(8);
    let packed = read_exact_capped(input, packed_len)?;
    RouteBits::from_packed(bit_len, &packed)
        .map_err(|err| ed_invalid(format!("MRT trace route bits invalid: {err}")))
}

/// Reads a pruned stub's carried `depth_below` (u32-BE), narrowing to the internal
/// u16. The range check (≤ `MAX_ROUTE_BITS`) is applied by `tree::decode_pruned`.
fn read_depth_dyn(input: &mut dyn Read) -> ed::Result<u16> {
    let mut buf = [0u8; 4];
    input.read_exact(&mut buf)?;
    let depth_u32 = u32::from_be_bytes(buf);
    u16::try_from(depth_u32).map_err(|_| {
        ed_invalid(format!(
            "MRT trace pruned depth_below {depth_u32} exceeds u16"
        ))
    })
}

fn decode_node_with_tag_dyn(
    tag: u8,
    input: &mut dyn Read,
    depth: usize,
) -> ed::Result<Arc<MrtNodeInner>> {
    if depth > MAX_MRT_TRACE_DECODE_DEPTH {
        return Err(ed_invalid(format!(
            "MRT trace decode depth {depth} exceeds limit {MAX_MRT_TRACE_DECODE_DEPTH}"
        )));
    }

    match tag {
        // Leaf: skip(bit_len u32-BE ‖ packed) ‖ value_len(u32-BE) ‖ value.
        0x01 => {
            let skip = read_skip_dyn(input)?;
            let mut value_len_buf = [0u8; 4];
            input.read_exact(&mut value_len_buf)?;
            let value_len = u32::from_be_bytes(value_len_buf) as usize;
            let value = read_exact_capped(input, value_len)?;
            Ok(MrtNodeInner::leaf(skip, value))
        }
        0x02 => {
            let skip = read_skip_dyn(input)?;
            let left = decode_node_dyn(input, depth + 1)?;
            let right = decode_node_dyn(input, depth + 1)?;
            // `decode_branch` validates the rolled-up depth_below (checked) over the
            // already-decoded children.
            tree::decode_branch(skip, left, right).map_err(ed_invalid)
        }
        0x03 => {
            let mut hash = [0u8; HASH_LENGTH];
            input.read_exact(&mut hash)?;
            let depth_below = read_depth_dyn(input)?;
            tree::decode_pruned(hash, depth_below).map_err(ed_invalid)
        }
        byte => Err(ed::Error::UnexpectedByte(byte)),
    }
}

impl Decode for Trace {
    fn decode<R: Read>(mut input: R) -> ed::Result<Self> {
        let mut tag = [0u8; 1];
        input.read_exact(&mut tag)?;
        if tag[0] == 0x00 {
            return Ok(Trace(None));
        }
        // A bare pruned root carries no parent edge to authenticate its depth (and
        // supports no operation), so it is malleable — reject it at the top level.
        if tag[0] == 0x03 {
            return Err(ed_invalid(
                "MRT trace root must be a materialized node or empty, not a pruned stub",
            ));
        }
        let node = decode_node_with_tag_dyn(tag[0], &mut input as &mut dyn Read, 0)?;
        Ok(Trace(Some(node)))
    }
}
