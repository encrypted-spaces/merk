use ed::Encode;

use super::cursor;
use super::trace::Trace;
use super::tracer::{
    assemble_mrt_pruned, record_snapshot_root, trace_get, trace_range, trace_range_inclusive,
    MrtPartialBuilder,
};
use super::Checkpoint;
use crate::error::Result;
use crate::hash::Hash;
use crate::proofs::query::{Query, QueryItem};

trait MrtQuerySurface {
    fn surface_get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn surface_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn surface_range_inclusive(
        &mut self,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
}

struct TracingSurface<'a, 'b> {
    snapshot: &'a Checkpoint,
    builder: &'b mut MrtPartialBuilder,
}

impl MrtQuerySurface for TracingSurface<'_, '_> {
    fn surface_get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        trace_get(self.snapshot, self.builder, key)
    }

    fn surface_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        trace_range(self.snapshot, self.builder, start, end)
    }

    fn surface_range_inclusive(
        &mut self,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        trace_range_inclusive(self.snapshot, self.builder, start, end)
    }
}

struct TraceSurface<'a>(&'a Trace);

impl MrtQuerySurface for TraceSurface<'_> {
    fn surface_get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        cursor::point_get(self.0 .0.as_ref(), key)
    }

    fn surface_range(&mut self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        cursor::collect_range(self.0 .0.as_ref(), start, Some(end))
    }

    fn surface_range_inclusive(
        &mut self,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        cursor::collect_range_inclusive(self.0 .0.as_ref(), start, end)
    }
}

fn collect_item<S: MrtQuerySurface>(
    surface: &mut S,
    item: &QueryItem,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    match item {
        QueryItem::Key(key) => match surface.surface_get(key)? {
            Some(value) => Ok(vec![(key.clone(), value)]),
            None => Ok(Vec::new()),
        },
        QueryItem::Range(range) => surface.surface_range(&range.start, &range.end),
        QueryItem::RangeInclusive(range) => {
            surface.surface_range_inclusive(range.start(), range.end())
        }
    }
}

fn normalize_query<Q, I>(input: I) -> Vec<QueryItem>
where
    Q: Into<QueryItem>,
    I: IntoIterator<Item = Q>,
{
    let mut query = Query::new();
    for item in input.into_iter().map(Into::into) {
        query.insert_item(item);
    }
    query.into_iter().collect()
}

fn normalize_query_ref(query: &Query) -> Vec<QueryItem> {
    query.iter().cloned().collect()
}

fn collect_normalized_query<S: MrtQuerySurface>(
    surface: &mut S,
    items: &[QueryItem],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut out = Vec::new();
    for item in items {
        out.extend(collect_item(surface, item)?);
    }
    Ok(out)
}

pub(crate) fn prove_from_snapshot<Q, I>(snapshot: &Checkpoint, query: I) -> Result<Vec<u8>>
where
    Q: Into<QueryItem>,
    I: IntoIterator<Item = Q>,
{
    let items = normalize_query(query);
    let mut builder = MrtPartialBuilder::new();
    record_snapshot_root(&mut builder, snapshot);
    let _ = collect_normalized_query(
        &mut TracingSurface {
            snapshot,
            builder: &mut builder,
        },
        &items,
    )?;

    let trace = assemble_mrt_pruned(builder);
    let mut bytes = Vec::with_capacity(64);
    Encode::encode_into(&trace, &mut bytes)?;
    Ok(bytes)
}

pub fn verify<Q, I>(bytes: &[u8], query: I, expected_hash: Hash) -> Result<Vec<(Vec<u8>, Vec<u8>)>>
where
    Q: Into<QueryItem>,
    I: IntoIterator<Item = Q>,
{
    let trace = Trace::decode_exact(bytes)?;
    trace.verify_root(expected_hash)?;
    let items = normalize_query(query);
    collect_normalized_query(&mut TraceSurface(&trace), &items)
}

pub fn verify_query(
    bytes: &[u8],
    query: &Query,
    expected_hash: Hash,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let trace = Trace::decode_exact(bytes)?;
    trace.verify_root(expected_hash)?;
    let items = normalize_query_ref(query);
    collect_normalized_query(&mut TraceSurface(&trace), &items)
}
