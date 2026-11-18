//! Errors for the inference core.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error(transparent)]
    Gguf(#[from] gguf::Error),

    #[error("unsupported architecture `{0}`")]
    UnsupportedArchitecture(String),

    #[error("missing required metadata key `{0}`")]
    MissingMetadata(String),

    #[error("metadata key `{key}`: {msg}")]
    BadMetadata { key: String, msg: String },

    #[error("missing required tensor `{0}`")]
    MissingTensor(String),

    #[error("tensor `{name}`: {msg}")]
    BadTensor { name: String, msg: String },

    #[error("unsupported tensor type {ty} for tensor `{name}`")]
    UnsupportedTensorType { name: String, ty: String },

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("context length exceeded: need {need}, max {max}")]
    ContextExceeded { need: usize, max: usize },

    #[error("shape mismatch: {0}")]
    Shape(String),

    #[error("backend: {0}")]
    Backend(String),

    #[error("{0}")]
    Other(String),
}
