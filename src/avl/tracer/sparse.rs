use std::cmp::max;
use std::convert::TryFrom;
use std::io::{self, Read, Write};

use ed::{Decode, Encode, Terminated};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::hash::{kv_hash, node_hash, Hash, Hasher, HASH_LENGTH, NULL_HASH};

const TRACE_TAG_EMPTY: u8 = 0x00;
const TRACE_TAG_PRUNED: u8 = 0x01;
const TRACE_TAG_FULL: u8 = 0x02;
const TRACE_TAG_FULL_STORAGE_HASH: u8 = 0x03;
const TRACE_TAG_FULL_OMITTED: u8 = 0x04;

/// Maximum recursive decode depth accepted for sparse AVL proof nodes.
pub const MAX_TRACE_DECODE_DEPTH: usize = 1024;

/// Maximum decoded byte length for any single length-prefixed trace field.
pub const MAX_TRACE_DECODE_FIELD_BYTES: usize = 64 * 1024 * 1024;

/// Sparse, hash-authenticated AVL proof node.
///
/// `Empty`, `Pruned`, and `Full` preserve the prototype sparse-tree shape. The
/// two hash-only opened variants (`FullStorageHash`, `FullOmitted`) carry a
/// node's position and subtree hash but **not its value**; they differ only in
/// value *provenance*. Both authenticate via `kv_hash`, both hash identically,
/// and both make a value read fail with `Error::ValueOmitted` (the verifier
/// never lists their key as readable). They are therefore **not**
/// interchangeable with `Full`, and the two must not be collapsed into one —
/// the distinction is load-bearing for the read-target soundness check in
/// `write_tracing`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Trace {
    /// Absent subtree.
    Empty,
    /// Elided subtree: only its root key, authenticated `hash`, and child
    /// heights are kept (cannot be descended into or read).
    Pruned {
        key: Vec<u8>,
        hash: Hash,
        child_heights: (u8, u8),
    },
    /// Opened node with its value present — readable.
    Full {
        key: Vec<u8>,
        value: Vec<u8>,
        left: Box<Trace>,
        right: Box<Trace>,
    },
    /// Opened node, value not materialized into the proof — represented by its
    /// storage `kv_hash`. Structurally present, but a value read fails
    /// `ValueOmitted`.
    FullStorageHash {
        key: Vec<u8>,
        kv_hash: Hash,
        left: Box<Trace>,
        right: Box<Trace>,
    },
    /// Opened node whose value was omitted from the proof (and which is *not* a
    /// read target — a read target left omitted is rejected at assembly). A
    /// value read fails `ValueOmitted`.
    FullOmitted {
        key: Vec<u8>,
        kv_hash: Hash,
        left: Box<Trace>,
        right: Box<Trace>,
    },
}

impl Trace {
    pub fn hash(&self) -> Hash {
        self.try_hash()
            .expect("trace key/value lengths and heights should be bounded")
    }

    pub fn try_hash(&self) -> Result<Hash> {
        match self {
            Trace::Empty => Ok(NULL_HASH),
            Trace::Pruned {
                hash,
                child_heights,
                ..
            } => {
                checked_height(*child_heights)?;
                Ok(*hash)
            }
            Trace::Full {
                key,
                value,
                left,
                right,
            } => {
                let kv = kv_hash::<Hasher>(key, value)?;
                Ok(node_hash::<Hasher>(
                    &kv,
                    &left.try_hash()?,
                    &right.try_hash()?,
                ))
            }
            Trace::FullStorageHash {
                kv_hash,
                left,
                right,
                ..
            }
            | Trace::FullOmitted {
                kv_hash,
                left,
                right,
                ..
            } => Ok(node_hash::<Hasher>(
                kv_hash,
                &left.try_hash()?,
                &right.try_hash()?,
            )),
        }
    }

    pub fn height(&self) -> u8 {
        self.try_height()
            .expect("trace heights should already be validated")
    }

    pub fn try_height(&self) -> Result<u8> {
        match self {
            Trace::Empty => Ok(0),
            Trace::Pruned { child_heights, .. } => checked_height(*child_heights),
            Trace::Full { left, right, .. }
            | Trace::FullStorageHash { left, right, .. }
            | Trace::FullOmitted { left, right, .. } => {
                checked_height((left.try_height()?, right.try_height()?))
            }
        }
    }

    pub fn count_full(&self) -> usize {
        match self {
            Trace::Empty | Trace::Pruned { .. } => 0,
            Trace::Full { left, right, .. }
            | Trace::FullStorageHash { left, right, .. }
            | Trace::FullOmitted { left, right, .. } => 1 + left.count_full() + right.count_full(),
        }
    }

    pub fn count_pruned(&self) -> usize {
        match self {
            Trace::Empty => 0,
            Trace::Pruned { .. } => 1,
            Trace::Full { left, right, .. }
            | Trace::FullStorageHash { left, right, .. }
            | Trace::FullOmitted { left, right, .. } => left.count_pruned() + right.count_pruned(),
        }
    }

    pub fn key(&self) -> Option<&[u8]> {
        match self {
            Trace::Empty => None,
            Trace::Pruned { key, .. }
            | Trace::Full { key, .. }
            | Trace::FullStorageHash { key, .. }
            | Trace::FullOmitted { key, .. } => Some(key),
        }
    }

    pub fn get(&self, target: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get_value(target)
            .map(|maybe_value| maybe_value.map(ToOwned::to_owned))
    }

    pub fn get_value(&self, target: &[u8]) -> Result<Option<&[u8]>> {
        match self {
            Trace::Empty => Ok(None),
            Trace::Pruned { key, .. } => Err(Error::PrunedNode(format!(
                "lookup for {target:?} descended into pruned node {key:?}"
            ))),
            Trace::Full {
                key,
                value,
                left,
                right,
            } => {
                if target == key.as_slice() {
                    Ok(Some(value))
                } else if target < key.as_slice() {
                    left.get_value(target)
                } else {
                    right.get_value(target)
                }
            }
            Trace::FullStorageHash {
                key, left, right, ..
            } => {
                if target == key.as_slice() {
                    Err(Error::ValueOmitted(format!(
                        "lookup for {target:?} reached storage hash-only node"
                    )))
                } else if target < key.as_slice() {
                    left.get_value(target)
                } else {
                    right.get_value(target)
                }
            }
            Trace::FullOmitted {
                key, left, right, ..
            } => {
                if target == key.as_slice() {
                    Err(Error::ValueOmitted(format!(
                        "lookup for {target:?} reached proof-omitted node"
                    )))
                } else if target < key.as_slice() {
                    left.get_value(target)
                } else {
                    right.get_value(target)
                }
            }
        }
    }

    pub fn collect_range(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut results = Vec::new();
        self.collect_range_inner(start, end, &mut results)?;
        Ok(results)
    }

    pub fn verify_root(&self, expected_root: Hash) -> Result<()> {
        let actual = self.try_hash()?;
        if actual == expected_root {
            Ok(())
        } else {
            Err(Error::HashMismatch(expected_root, actual))
        }
    }

    pub fn decode_exact(bytes: &[u8]) -> Result<Self> {
        let mut cursor = bytes;
        let trace = <Trace as Decode>::decode(&mut cursor)?;
        let mut probe = [0u8; 1];
        if Read::read(&mut cursor, &mut probe)? != 0 {
            return Err(Error::Ed(ed_invalid(
                "Trace decode_exact: trailing bytes after value",
            )));
        }
        Ok(trace)
    }

    /// Encode this trace to its wire form — the inverse of [`Self::decode_exact`],
    /// and the bytes [`crate::avl::TraceVerifier::decode_trace`] consumes.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        Encode::encode_into(self, &mut bytes)?;
        Ok(bytes)
    }

    pub fn full_or_omitted(
        key: Vec<u8>,
        value: Vec<u8>,
        _read_target: bool,
        left: Trace,
        right: Trace,
    ) -> Result<Self> {
        Ok(Trace::Full {
            key,
            value,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    fn collect_range_inner(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
        results: &mut Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<()> {
        match self {
            Trace::Empty => Ok(()),
            Trace::Pruned { key, .. } => Err(Error::PrunedNode(format!(
                "range traversal descended into pruned node {key:?}"
            ))),
            Trace::Full {
                key,
                value,
                left,
                right,
            } => collect_open_range(
                key,
                Some(value.as_slice()),
                left,
                right,
                start,
                end,
                results,
            ),
            Trace::FullStorageHash {
                key, left, right, ..
            } => collect_open_range(key, None, left, right, start, end, results),
            Trace::FullOmitted {
                key, left, right, ..
            } => collect_open_range(key, None, left, right, start, end, results),
        }
    }
}

fn collect_open_range(
    key: &[u8],
    value: Option<&[u8]>,
    left: &Trace,
    right: &Trace,
    start: &[u8],
    end: Option<&[u8]>,
    results: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<()> {
    if key > start {
        left.collect_range_inner(start, end, results)?;
    }

    if key >= start && end.is_none_or(|upper| key < upper) {
        let value = value.ok_or_else(|| {
            Error::ValueOmitted(format!(
                "range traversal tried to yield proof-omitted node {key:?}"
            ))
        })?;
        results.push((key.to_vec(), value.to_vec()));
    }

    if end.is_none_or(|upper| key < upper) {
        right.collect_range_inner(start, end, results)?;
    }

    Ok(())
}

fn checked_height(child_heights: (u8, u8)) -> Result<u8> {
    1u8.checked_add(max(child_heights.0, child_heights.1))
        .ok_or_else(|| Error::Tree("AVL trace height exceeds u8::MAX".into()))
}

fn ed_invalid<E: std::fmt::Display>(msg: E) -> ed::Error {
    ed::Error::IOError(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{msg}"),
    ))
}

fn try_u32(len: usize, what: &str) -> ed::Result<u32> {
    u32::try_from(len)
        .map_err(|_| ed_invalid(format!("Trace {what} length {len} exceeds u32::MAX")))
}

fn encode_len_prefixed_bytes<W: Write>(dest: &mut W, what: &str, bytes: &[u8]) -> ed::Result<()> {
    let len = try_u32(bytes.len(), what)?;
    dest.write_all(&len.to_be_bytes())?;
    dest.write_all(bytes)?;
    Ok(())
}

fn read_u32<R: Read + ?Sized>(input: &mut R) -> ed::Result<u32> {
    let mut buf = [0u8; 4];
    input.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_exact_capped<R: Read + ?Sized>(
    input: &mut R,
    what: &str,
    len: usize,
) -> ed::Result<Vec<u8>> {
    if len > MAX_TRACE_DECODE_FIELD_BYTES {
        return Err(ed_invalid(format!(
            "Trace {what} length {len} exceeds per-field limit {MAX_TRACE_DECODE_FIELD_BYTES}"
        )));
    }

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

impl Encode for Trace {
    fn encode_into<W: Write>(&self, dest: &mut W) -> ed::Result<()> {
        match self {
            Trace::Empty => dest.write_all(&[TRACE_TAG_EMPTY])?,
            Trace::Pruned {
                key,
                hash,
                child_heights,
            } => {
                checked_height(*child_heights).map_err(ed_invalid)?;
                dest.write_all(&[TRACE_TAG_PRUNED])?;
                encode_len_prefixed_bytes(dest, "pruned key", key)?;
                dest.write_all(hash)?;
                dest.write_all(&[child_heights.0, child_heights.1])?;
            }
            Trace::Full {
                key,
                value,
                left,
                right,
            } => {
                dest.write_all(&[TRACE_TAG_FULL])?;
                encode_len_prefixed_bytes(dest, "key", key)?;
                encode_len_prefixed_bytes(dest, "value", value)?;
                Encode::encode_into(left.as_ref(), dest)?;
                Encode::encode_into(right.as_ref(), dest)?;
            }
            Trace::FullStorageHash {
                key,
                kv_hash,
                left,
                right,
            } => {
                dest.write_all(&[TRACE_TAG_FULL_STORAGE_HASH])?;
                encode_len_prefixed_bytes(dest, "key", key)?;
                dest.write_all(kv_hash)?;
                Encode::encode_into(left.as_ref(), dest)?;
                Encode::encode_into(right.as_ref(), dest)?;
            }
            Trace::FullOmitted {
                key,
                kv_hash,
                left,
                right,
            } => {
                dest.write_all(&[TRACE_TAG_FULL_OMITTED])?;
                encode_len_prefixed_bytes(dest, "key", key)?;
                dest.write_all(kv_hash)?;
                Encode::encode_into(left.as_ref(), dest)?;
                Encode::encode_into(right.as_ref(), dest)?;
            }
        }
        Ok(())
    }

    fn encoding_length(&self) -> ed::Result<usize> {
        Ok(match self {
            Trace::Empty => 1,
            Trace::Pruned { key, .. } => {
                try_u32(key.len(), "pruned key")?;
                1 + 4 + key.len() + HASH_LENGTH + 2
            }
            Trace::Full {
                key,
                value,
                left,
                right,
            } => {
                try_u32(key.len(), "key")?;
                try_u32(value.len(), "value")?;
                1 + 4
                    + key.len()
                    + 4
                    + value.len()
                    + Encode::encoding_length(left.as_ref())?
                    + Encode::encoding_length(right.as_ref())?
            }
            Trace::FullStorageHash {
                key, left, right, ..
            }
            | Trace::FullOmitted {
                key, left, right, ..
            } => {
                try_u32(key.len(), "key")?;
                1 + 4
                    + key.len()
                    + HASH_LENGTH
                    + Encode::encoding_length(left.as_ref())?
                    + Encode::encoding_length(right.as_ref())?
            }
        })
    }
}

impl Decode for Trace {
    fn decode<R: Read>(mut input: R) -> ed::Result<Self> {
        let trace = decode_dyn(&mut input as &mut dyn Read, 0)?;
        trace.try_height().map_err(ed_invalid)?;
        Ok(trace)
    }
}

impl Terminated for Trace {}

fn decode_dyn(input: &mut dyn Read, depth: usize) -> ed::Result<Trace> {
    if depth > MAX_TRACE_DECODE_DEPTH {
        return Err(ed_invalid(format!(
            "Trace decode depth {depth} exceeds limit {MAX_TRACE_DECODE_DEPTH}"
        )));
    }

    let mut tag = [0u8; 1];
    input.read_exact(&mut tag)?;
    match tag[0] {
        TRACE_TAG_EMPTY => Ok(Trace::Empty),
        TRACE_TAG_PRUNED => {
            let key_len = read_u32(input)? as usize;
            let key = read_exact_capped(input, "pruned key", key_len)?;
            let mut hash = [0u8; HASH_LENGTH];
            input.read_exact(&mut hash)?;
            let mut heights = [0u8; 2];
            input.read_exact(&mut heights)?;
            let child_heights = (heights[0], heights[1]);
            checked_height(child_heights).map_err(ed_invalid)?;
            Ok(Trace::Pruned {
                key,
                hash,
                child_heights,
            })
        }
        TRACE_TAG_FULL => {
            let key_len = read_u32(input)? as usize;
            let key = read_exact_capped(input, "key", key_len)?;
            let value_len = read_u32(input)? as usize;
            let value = read_exact_capped(input, "value", value_len)?;
            let left = decode_dyn(input, depth + 1)?;
            let right = decode_dyn(input, depth + 1)?;
            Ok(Trace::Full {
                key,
                value,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        TRACE_TAG_FULL_STORAGE_HASH | TRACE_TAG_FULL_OMITTED => {
            let key_len = read_u32(input)? as usize;
            let key = read_exact_capped(input, "key", key_len)?;
            let mut kv_hash = [0u8; HASH_LENGTH];
            input.read_exact(&mut kv_hash)?;
            let left = Box::new(decode_dyn(input, depth + 1)?);
            let right = Box::new(decode_dyn(input, depth + 1)?);
            if tag[0] == TRACE_TAG_FULL_STORAGE_HASH {
                Ok(Trace::FullStorageHash {
                    key,
                    kv_hash,
                    left,
                    right,
                })
            } else {
                Ok(Trace::FullOmitted {
                    key,
                    kv_hash,
                    left,
                    right,
                })
            }
        }
        byte => Err(ed::Error::UnexpectedByte(byte)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(key: &[u8], value: &[u8]) -> Trace {
        Trace::Full {
            key: key.to_vec(),
            value: value.to_vec(),
            left: Box::new(Trace::Empty),
            right: Box::new(Trace::Empty),
        }
    }

    fn round_trip(trace: &Trace) -> Trace {
        let mut bytes = Vec::new();
        Encode::encode_into(trace, &mut bytes).unwrap();
        Trace::decode_exact(&bytes).unwrap()
    }

    #[test]
    fn hash_only_variants_hash_like_full_but_cannot_be_read() {
        let key = b"k";
        let value = b"value";
        let kv_hash = kv_hash::<Hasher>(key, value).unwrap();
        let full = leaf(key, value);
        let omitted = Trace::FullOmitted {
            key: key.to_vec(),
            kv_hash,
            left: Box::new(Trace::Empty),
            right: Box::new(Trace::Empty),
        };
        let storage = Trace::FullStorageHash {
            key: key.to_vec(),
            kv_hash,
            left: Box::new(Trace::Empty),
            right: Box::new(Trace::Empty),
        };

        assert_eq!(full.hash(), omitted.hash());
        assert_eq!(full.hash(), storage.hash());
        assert!(matches!(omitted.get(key), Err(Error::ValueOmitted(_))));
        assert!(matches!(storage.get(key), Err(Error::ValueOmitted(_))));
    }

    #[test]
    fn range_routes_through_hash_only_but_cannot_yield_it() {
        let omitted_root = Trace::FullOmitted {
            key: b"m".to_vec(),
            kv_hash: kv_hash::<Hasher>(b"m", b"middle").unwrap(),
            left: Box::new(leaf(b"a", b"left")),
            right: Box::new(leaf(b"z", b"right")),
        };
        let storage_root = Trace::FullStorageHash {
            key: b"m".to_vec(),
            kv_hash: kv_hash::<Hasher>(b"m", b"middle").unwrap(),
            left: Box::new(leaf(b"a", b"left")),
            right: Box::new(leaf(b"z", b"right")),
        };

        assert_eq!(
            omitted_root.collect_range(b"a", Some(b"b")).unwrap(),
            vec![(b"a".to_vec(), b"left".to_vec())]
        );
        assert_eq!(
            storage_root.collect_range(b"a", Some(b"b")).unwrap(),
            vec![(b"a".to_vec(), b"left".to_vec())]
        );
        assert!(matches!(
            omitted_root.collect_range(b"a", Some(b"z")),
            Err(Error::ValueOmitted(_))
        ));
        assert!(matches!(
            storage_root.collect_range(b"a", Some(b"z")),
            Err(Error::ValueOmitted(_))
        ));
    }

    #[test]
    fn ed_round_trip_preserves_pruned_and_hash_only_variants() {
        let trace = Trace::FullStorageHash {
            key: b"root".to_vec(),
            kv_hash: [7; HASH_LENGTH],
            left: Box::new(Trace::Pruned {
                key: b"left".to_vec(),
                hash: [1; HASH_LENGTH],
                child_heights: (2, 1),
            }),
            right: Box::new(Trace::FullOmitted {
                key: b"right".to_vec(),
                kv_hash: [8; HASH_LENGTH],
                left: Box::new(Trace::Empty),
                right: Box::new(Trace::Empty),
            }),
        };

        assert_eq!(round_trip(&trace), trace);
    }

    #[test]
    fn decode_exact_rejects_trailing_bytes() {
        let trace = leaf(b"k", b"v");
        let mut bytes = Vec::new();
        Encode::encode_into(&trace, &mut bytes).unwrap();
        bytes.push(0xff);

        assert!(matches!(Trace::decode_exact(&bytes), Err(Error::Ed(_))));
    }

    #[test]
    fn decode_rejects_oversized_length_prefixed_fields() {
        let too_large = u32::try_from(MAX_TRACE_DECODE_FIELD_BYTES + 1).unwrap();

        let mut oversized_key = vec![TRACE_TAG_FULL];
        oversized_key.extend_from_slice(&too_large.to_be_bytes());
        assert!(matches!(
            Trace::decode_exact(&oversized_key),
            Err(Error::Ed(_))
        ));

        let mut oversized_value = vec![TRACE_TAG_FULL];
        oversized_value.extend_from_slice(&1u32.to_be_bytes());
        oversized_value.push(b'k');
        oversized_value.extend_from_slice(&too_large.to_be_bytes());
        assert!(matches!(
            Trace::decode_exact(&oversized_value),
            Err(Error::Ed(_))
        ));
    }

    #[test]
    fn full_or_omitted_keeps_materialized_values_full() {
        let short = Trace::full_or_omitted(
            b"k".to_vec(),
            vec![1; crate::tracer::SMALL_VALUE_INLINE_THRESHOLD],
            false,
            Trace::Empty,
            Trace::Empty,
        )
        .unwrap();
        assert!(matches!(short, Trace::Full { .. }));

        let long = Trace::full_or_omitted(
            b"k".to_vec(),
            vec![1; crate::tracer::SMALL_VALUE_INLINE_THRESHOLD + 1],
            false,
            Trace::Empty,
            Trace::Empty,
        )
        .unwrap();
        assert!(matches!(long, Trace::Full { .. }));
    }
}
