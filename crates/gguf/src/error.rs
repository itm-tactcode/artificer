//! Error types for GGUF loading and parsing.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Result alias for GGUF operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur while loading or inspecting a GGUF file.
#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error while reading `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("I/O error: {0}")]
    IoSimple(#[from] io::Error),

    #[error("invalid GGUF magic: expected 0x{expected:08X} ('GGUF'), got 0x{got:08X}")]
    InvalidMagic { expected: u32, got: u32 },

    #[error("unsupported GGUF version {version} (supported: {supported_min}..={supported_max})")]
    UnsupportedVersion {
        version: u32,
        supported_min: u32,
        supported_max: u32,
    },

    #[error("unexpected end of file: needed {needed} more bytes at offset {offset}")]
    UnexpectedEof { offset: u64, needed: usize },

    #[error("invalid metadata value type id: {0}")]
    InvalidMetadataType(u32),

    #[error("invalid boolean value: expected 0 or 1, got {0}")]
    InvalidBool(u8),

    #[error("invalid UTF-8 in GGUF string at offset {offset}: {source}")]
    InvalidUtf8 {
        offset: u64,
        #[source]
        source: std::str::Utf8Error,
    },

    #[error("metadata key `{key}` is not valid UTF-8 ASCII hierarchical key")]
    InvalidMetadataKey { key: String },

    #[error("unknown ggml tensor type id: {0}")]
    UnknownTensorType(u32),

    #[error("tensor `{name}` has zero dimensions")]
    EmptyTensorShape { name: String },

    #[error("tensor `{name}` has too many dimensions: {n_dims} (max {max})")]
    TooManyDimensions { name: String, n_dims: u32, max: u32 },

    #[error(
        "tensor `{name}` data out of bounds: offset {offset} + size {size} \
         exceeds tensor data region of {region_size} bytes"
    )]
    TensorDataOutOfBounds {
        name: String,
        offset: u64,
        size: u64,
        region_size: u64,
    },

    #[error("tensor `{name}` offset {offset} is not aligned to {alignment}")]
    MisalignedTensor {
        name: String,
        offset: u64,
        alignment: u64,
    },

    #[error("metadata key not found: `{0}`")]
    MetadataKeyNotFound(String),

    #[error("metadata key `{key}` has unexpected type (wanted {expected})")]
    MetadataTypeMismatch { key: String, expected: &'static str },

    #[error("tensor not found: `{0}`")]
    TensorNotFound(String),

    #[error("string length {len} exceeds remaining file data at offset {offset}")]
    StringTooLong { len: u64, offset: u64 },

    #[error("array length {len} is unreasonably large at offset {offset}")]
    ArrayTooLarge { len: u64, offset: u64 },

    #[error("cannot compute byte size for tensor type {ty:?} with shape {shape:?}")]
    UncomputableTensorSize {
        ty: crate::types::TensorType,
        shape: Vec<u64>,
    },

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
