use std::fmt;

/// An operation to be applied to a key in a sorted storage batch.
#[derive(Clone, PartialEq)]
pub enum Op {
    /// Inserts or updates the key/value entry to the given value.
    Put(Vec<u8>),
    /// Deletes the key/value entry.
    Delete,
    /// Deletes all keys in [batch_entry_key, end). The batch entry key is the range start.
    DeleteRange(Vec<u8>),
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "{}",
            match self {
                Op::Put(value) => format!("Put({value:?})"),
                Op::Delete => "Delete".to_string(),
                Op::DeleteRange(end) => format!("DeleteRange({end:?})"),
            }
        )
    }
}

/// A single `(key, operation)` pair.
pub type BatchEntry = (Vec<u8>, Op);

/// A sorted storage batch.
///
/// Backends validate the detailed ordering rules. Point-operation segments must
/// have sorted, unique keys; `DeleteRange` entries use the entry key as the
/// inclusive start and carry an exclusive end bound.
pub type Batch = [BatchEntry];
