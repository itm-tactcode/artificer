//! Metadata key-value map with typed accessors for common GGUF keys.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::types::MetadataValue;

/// Ordered metadata store (insertion order preserved for debugging).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    map: HashMap<String, MetadataValue>,
    /// Keys in file order.
    order: Vec<String>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: String, value: MetadataValue) {
        if !self.map.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.map.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.map.get(key)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate key-value pairs in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.order
            .iter()
            .filter_map(|k| self.map.get(k).map(|v| (k.as_str(), v)))
    }

    pub fn require(&self, key: &str) -> Result<&MetadataValue> {
        self.get(key)
            .ok_or_else(|| Error::MetadataKeyNotFound(key.to_owned()))
    }

    pub fn get_str(&self, key: &str) -> Result<Option<&str>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(Some)
                .ok_or_else(|| Error::MetadataTypeMismatch {
                    key: key.to_owned(),
                    expected: "string",
                }),
        }
    }

    pub fn require_str(&self, key: &str) -> Result<&str> {
        self.get_str(key)?
            .ok_or_else(|| Error::MetadataKeyNotFound(key.to_owned()))
    }

    pub fn get_u64(&self, key: &str) -> Result<Option<u64>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| Error::MetadataTypeMismatch {
                    key: key.to_owned(),
                    expected: "unsigned integer",
                }),
        }
    }

    pub fn require_u64(&self, key: &str) -> Result<u64> {
        self.get_u64(key)?
            .ok_or_else(|| Error::MetadataKeyNotFound(key.to_owned()))
    }

    pub fn get_f64(&self, key: &str) -> Result<Option<f64>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| Error::MetadataTypeMismatch {
                    key: key.to_owned(),
                    expected: "float",
                }),
        }
    }

    pub fn get_bool(&self, key: &str) -> Result<Option<bool>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_bool()
                .map(Some)
                .ok_or_else(|| Error::MetadataTypeMismatch {
                    key: key.to_owned(),
                    expected: "bool",
                }),
        }
    }

    // --- Common standardized keys ---

    /// `general.architecture`
    pub fn architecture(&self) -> Result<Option<&str>> {
        self.get_str("general.architecture")
    }

    /// `general.name`
    pub fn name(&self) -> Result<Option<&str>> {
        self.get_str("general.name")
    }

    /// Alignment from `general.alignment`, or default 32.
    pub fn alignment(&self) -> u64 {
        self.get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(crate::types::DEFAULT_ALIGNMENT)
            .max(1)
    }

    /// `general.quantization_version`
    pub fn quantization_version(&self) -> Result<Option<u64>> {
        self.get_u64("general.quantization_version")
    }

    /// Architecture-prefixed context length, e.g. `llama.context_length`.
    pub fn context_length(&self) -> Result<Option<u64>> {
        if let Some(arch) = self.architecture()? {
            let key = format!("{arch}.context_length");
            if let Some(v) = self.get_u64(&key)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// Architecture-prefixed embedding length.
    pub fn embedding_length(&self) -> Result<Option<u64>> {
        if let Some(arch) = self.architecture()? {
            let key = format!("{arch}.embedding_length");
            if let Some(v) = self.get_u64(&key)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// Architecture-prefixed block (layer) count.
    pub fn block_count(&self) -> Result<Option<u64>> {
        if let Some(arch) = self.architecture()? {
            let key = format!("{arch}.block_count");
            if let Some(v) = self.get_u64(&key)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// Architecture-prefixed attention head count.
    pub fn head_count(&self) -> Result<Option<u64>> {
        if let Some(arch) = self.architecture()? {
            let key = format!("{arch}.attention.head_count");
            if let Some(v) = self.get_u64(&key)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// Architecture-prefixed KV head count (GQA); falls back to head_count.
    pub fn head_count_kv(&self) -> Result<Option<u64>> {
        if let Some(arch) = self.architecture()? {
            let key = format!("{arch}.attention.head_count_kv");
            if let Some(v) = self.get_u64(&key)? {
                return Ok(Some(v));
            }
        }
        self.head_count()
    }

    /// Tokenizer model name (`tokenizer.ggml.model`).
    pub fn tokenizer_model(&self) -> Result<Option<&str>> {
        self.get_str("tokenizer.ggml.model")
    }

    /// BOS token id if present.
    pub fn bos_token_id(&self) -> Result<Option<u64>> {
        self.get_u64("tokenizer.ggml.bos_token_id")
    }

    /// EOS token id if present.
    pub fn eos_token_id(&self) -> Result<Option<u64>> {
        self.get_u64("tokenizer.ggml.eos_token_id")
    }
}
