//! Minimal GGUF model loader.
//!
//! Parses the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! binary format (v2/v3, little-endian), exposing metadata and zero-copy tensor
//! views via `mmap` for low-memory CPU inference.
//!
//! # Example
//!
//! ```no_run
//! use gguf::GgufModel;
//!
//! let model = GgufModel::load("model.gguf").expect("load GGUF");
//! println!("{}", model.summary());
//! if let Some(info) = model.find_tensor("token_embd.weight") {
//!     println!("embd: {:?} {:?}", info.shape, info.tensor_type);
//! }
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]

mod error;
mod metadata;
mod model;
mod reader;
mod types;

pub use error::{Error, Result};
pub use metadata::Metadata;
pub use model::{GgufModel, LoadMode, LoadOptions, ModelSummary};
pub use types::{
    align_offset, MetadataArray, MetadataValue, TensorInfo, TensorType, ValueType,
    DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_MAX, GGUF_VERSION_MIN,
};

#[cfg(test)]
mod tests;
