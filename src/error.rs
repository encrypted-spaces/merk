pub use thiserror::Error;

use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedFeature {
    MovePrefix,
}

impl fmt::Display for UnsupportedFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsupportedFeature::MovePrefix => f.write_str("move_prefix"),
        }
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("Attach Error: {0}")]
    Attach(String),
    #[error("Batch Key Error: {0}")]
    BatchKey(String),
    #[error("Bound Error: {0}")]
    Bound(String),
    #[error(transparent)]
    Ed(#[from] ed::Error),
    #[error("Fetch Error: {0}")]
    Fetch(String),
    #[error("Hash Error: {0}")]
    Hash(String),
    #[error("Proof did not match expected hash\n\tExpected: {0:?}\n\tActual: {1:?}")]
    HashMismatch([u8; 32], [u8; 32]),
    #[error("Index OoB Error: {0}")]
    IndexOutOfBounds(String),
    #[error("Integer conversion error: {0}")]
    IntegerConversionError(#[from] std::num::TryFromIntError),
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error("Tried to delete non-existent key {0:?}")]
    KeyDelete(Vec<u8>),
    #[error("Key Error: {0}")]
    Key(String),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("Proof is missing data for query")]
    MissingData,
    #[error("Path Error: {0}")]
    Path(String),
    #[error("Operation on a poisoned trace handle: {0}")]
    Poisoned(String),
    #[error("Proof Error: {0}")]
    Proof(String),
    #[error("Descent hit pruned node: {0}")]
    PrunedNode(String),
    #[error("Stack Underflow")]
    StackUnderflow,
    #[error("Tree Error: {0}")]
    Tree(String),
    #[error("Unsupported feature: {0}")]
    Unsupported(UnsupportedFeature),
    #[error("Unexpected Node Error: {0}")]
    UnexpectedNode(String),
    #[error("Unknown Error")]
    Unknown,
    #[error("Value omitted from proof: {0}")]
    ValueOmitted(String),
    #[error("Version Error: {0}")]
    Version(String),
}

pub type Result<T> = std::result::Result<T, Error>;
