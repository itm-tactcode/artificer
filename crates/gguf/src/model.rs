//! High-level GGUF model loader with memory-mapped tensor data.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::metadata::Metadata;
use crate::reader::Reader;
use crate::types::{
    align_offset, TensorInfo, TensorType, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_MAX,
    GGUF_VERSION_MIN, MAX_TENSOR_DIMS,
};

/// How tensor payload bytes are retained after parsing the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadMode {
    /// Memory-map the entire file (preferred for large models / 16GB RAM hosts).
    #[default]
    Mmap,
    /// Read the full file into a heap buffer (useful for tiny test files / no mmap).
    Buffered,
}

/// Options controlling GGUF loading behaviour.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub mode: LoadMode,
    /// When true, verify each tensor's data region lies within the file.
    pub validate_tensor_bounds: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            mode: LoadMode::Mmap,
            validate_tensor_bounds: true,
        }
    }
}

/// Backing storage for the GGUF file bytes.
enum Storage {
    Mmap(Mmap),
    Buffer(Vec<u8>),
}

impl Storage {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Buffer(b) => b,
        }
    }
}

/// A loaded GGUF model: header metadata + tensor catalog + mapped weights.
///
/// Tensor **weights are not copied** under [`LoadMode::Mmap`]; they remain in the
/// mapped file. This keeps peak RAM close to OS page-cache behaviour — important
/// for ~16GB machines.
pub struct GgufModel {
    path: PathBuf,
    version: u32,
    alignment: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    /// Absolute file offset where the tensor data section begins.
    tensor_data_offset: u64,
    storage: Storage,
}

impl std::fmt::Debug for GgufModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufModel")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("metadata_count", &self.metadata.len())
            .field("tensor_count", &self.tensors.len())
            .field("tensor_data_offset", &self.tensor_data_offset)
            .field("file_size", &self.file_size())
            .finish_non_exhaustive()
    }
}

impl GgufModel {
    /// Load a GGUF file with default options (mmap + bounds checks).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(path, LoadOptions::default())
    }

    /// Load a GGUF file with explicit options.
    pub fn load_with(path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let storage = match opts.mode {
            LoadMode::Mmap => {
                let file = File::open(path).map_err(|e| Error::io(path, e))?;
                // SAFETY: we treat the mapping as read-only and never write to it.
                // Concurrent external writers are undefined behaviour; GGUF models
                // are immutable inputs for inference.
                let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
                Storage::Mmap(mmap)
            }
            LoadMode::Buffered => {
                let buf = std::fs::read(path).map_err(|e| Error::io(path, e))?;
                Storage::Buffer(buf)
            }
        };

        Self::parse(path.to_path_buf(), storage, opts)
    }

    /// Parse an already-buffered GGUF blob (primarily for tests).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(
            PathBuf::from("<memory>"),
            Storage::Buffer(bytes),
            LoadOptions {
                mode: LoadMode::Buffered,
                validate_tensor_bounds: true,
            },
        )
    }

    fn parse(path: PathBuf, storage: Storage, opts: LoadOptions) -> Result<Self> {
        let data = storage.as_slice();
        let mut r = Reader::new(data);

        // --- Header ---
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::InvalidMagic {
                expected: GGUF_MAGIC,
                got: magic,
            });
        }

        let version = r.read_u32()?;
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            return Err(Error::UnsupportedVersion {
                version,
                supported_min: GGUF_VERSION_MIN,
                supported_max: GGUF_VERSION_MAX,
            });
        }

        let tensor_count = r.read_u64()?;
        let metadata_kv_count = r.read_u64()?;

        // --- Metadata ---
        let mut metadata = Metadata::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = r.read_metadata_kv()?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        // --- Tensor infos ---
        let mut tensors = Vec::with_capacity(tensor_count.min(100_000) as usize);
        let mut tensor_index = HashMap::with_capacity(tensors.capacity());

        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32()?;
            if n_dims == 0 {
                return Err(Error::EmptyTensorShape { name });
            }
            if n_dims > MAX_TENSOR_DIMS {
                return Err(Error::TooManyDimensions {
                    name,
                    n_dims,
                    max: MAX_TENSOR_DIMS,
                });
            }
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(r.read_u64()?);
            }
            let type_id = r.read_u32()?;
            let tensor_type =
                TensorType::from_u32(type_id).ok_or(Error::UnknownTensorType(type_id))?;
            let offset = r.read_u64()?;

            if offset % alignment != 0 {
                return Err(Error::MisalignedTensor {
                    name,
                    offset,
                    alignment,
                });
            }

            let info = TensorInfo {
                name: name.clone(),
                shape,
                tensor_type,
                offset,
            };
            tensor_index.insert(name, tensors.len());
            tensors.push(info);
        }

        // Tensor data section starts at the next aligned offset after tensor infos.
        let tensor_data_offset = align_offset(r.position(), alignment);
        let file_len = data.len() as u64;
        let tensor_region_size = file_len.saturating_sub(tensor_data_offset);

        if opts.validate_tensor_bounds {
            for t in &tensors {
                let size = t.nbytes().ok_or_else(|| Error::UncomputableTensorSize {
                    ty: t.tensor_type,
                    shape: t.shape.clone(),
                })?;
                let end =
                    t.offset
                        .checked_add(size)
                        .ok_or_else(|| Error::TensorDataOutOfBounds {
                            name: t.name.clone(),
                            offset: t.offset,
                            size,
                            region_size: tensor_region_size,
                        })?;
                if end > tensor_region_size {
                    return Err(Error::TensorDataOutOfBounds {
                        name: t.name.clone(),
                        offset: t.offset,
                        size,
                        region_size: tensor_region_size,
                    });
                }
            }
        }

        Ok(Self {
            path,
            version,
            alignment,
            metadata,
            tensors,
            tensor_index,
            tensor_data_offset,
            storage,
        })
    }

    // --- Accessors ---

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Total file size in bytes.
    pub fn file_size(&self) -> u64 {
        self.storage.as_slice().len() as u64
    }

    /// Approximate total tensor payload size (sum of individual nbytes).
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(|t| t.nbytes()).sum()
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn require_tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.find_tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))
    }

    /// Absolute file offset of a tensor's data.
    ///
    /// Checked: a corrupt offset that would wrap past `u64::MAX` is reported as
    /// [`Error::TensorDataOutOfBounds`] instead of wrapping.
    pub fn tensor_file_offset(&self, info: &TensorInfo) -> Result<u64> {
        self.tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| Error::TensorDataOutOfBounds {
                name: info.name.clone(),
                offset: info.offset,
                size: info.nbytes().unwrap_or(0),
                region_size: (self.storage.as_slice().len() as u64)
                    .saturating_sub(self.tensor_data_offset),
            })
    }

    /// Borrow the raw bytes for a tensor (zero-copy into the map/buffer).
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self.require_tensor(name)?;
        let size = info.nbytes().ok_or_else(|| Error::UncomputableTensorSize {
            ty: info.tensor_type,
            shape: info.shape.clone(),
        })? as usize;
        let data = self.storage.as_slice();
        let out_of_bounds = || Error::TensorDataOutOfBounds {
            name: name.to_owned(),
            offset: info.offset,
            size: size as u64,
            region_size: (data.len() as u64).saturating_sub(self.tensor_data_offset),
        };
        // Checked: a corrupt offset or size must not wrap past the bounds check below.
        let start = self
            .tensor_data_offset
            .checked_add(info.offset)
            .and_then(|s| usize::try_from(s).ok())
            .ok_or_else(out_of_bounds)?;
        let end = start.checked_add(size).ok_or_else(out_of_bounds)?;
        if end > data.len() {
            return Err(out_of_bounds());
        }
        Ok(&data[start..end])
    }

    /// High-level model summary for logging / CLI.
    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            path: self.path.clone(),
            version: self.version,
            architecture: self
                .metadata
                .architecture()
                .ok()
                .flatten()
                .map(str::to_owned),
            name: self.metadata.name().ok().flatten().map(str::to_owned),
            alignment: self.alignment,
            metadata_count: self.metadata.len(),
            tensor_count: self.tensors.len(),
            file_size: self.file_size(),
            tensor_bytes: self.total_tensor_bytes(),
            context_length: self.metadata.context_length().ok().flatten(),
            embedding_length: self.metadata.embedding_length().ok().flatten(),
            block_count: self.metadata.block_count().ok().flatten(),
            head_count: self.metadata.head_count().ok().flatten(),
        }
    }
}

/// Compact human-readable summary of a loaded model.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub path: PathBuf,
    pub version: u32,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub alignment: u64,
    pub metadata_count: usize,
    pub tensor_count: usize,
    pub file_size: u64,
    pub tensor_bytes: u64,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
}

impl std::fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF model: {}", self.path.display())?;
        writeln!(f, "  version:       {}", self.version)?;
        if let Some(ref n) = self.name {
            writeln!(f, "  name:          {n}")?;
        }
        if let Some(ref a) = self.architecture {
            writeln!(f, "  architecture:  {a}")?;
        }
        writeln!(f, "  alignment:     {}", self.alignment)?;
        writeln!(f, "  metadata kvs:  {}", self.metadata_count)?;
        writeln!(f, "  tensors:       {}", self.tensor_count)?;
        writeln!(
            f,
            "  file size:     {} ({:.2} MiB)",
            self.file_size,
            self.file_size as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  tensor data:   {} ({:.2} MiB)",
            self.tensor_bytes,
            self.tensor_bytes as f64 / (1024.0 * 1024.0)
        )?;
        if let Some(v) = self.context_length {
            writeln!(f, "  context_len:   {v}")?;
        }
        if let Some(v) = self.embedding_length {
            writeln!(f, "  embedding:     {v}")?;
        }
        if let Some(v) = self.block_count {
            writeln!(f, "  blocks:        {v}")?;
        }
        if let Some(v) = self.head_count {
            writeln!(f, "  heads:         {v}")?;
        }
        Ok(())
    }
}
