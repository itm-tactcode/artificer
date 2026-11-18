//! Core GGUF / GGML type definitions.

use std::fmt;

/// GGUF file magic: bytes `G` `G` `U` `F` as little-endian `u32` (`0x46554747`).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default tensor data alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Supported GGUF format versions (inclusive range).
pub const GGUF_VERSION_MIN: u32 = 2;
pub const GGUF_VERSION_MAX: u32 = 3;

/// Maximum tensor rank accepted by this loader (spec currently uses ≤4).
pub const MAX_TENSOR_DIMS: u32 = 8;

/// Maximum metadata string length we accept (defensive; keys are ≤65535).
pub const MAX_STRING_LEN: u64 = 64 * 1024 * 1024; // 64 MiB

/// Maximum array element count we accept when parsing metadata.
pub const MAX_ARRAY_LEN: u64 = 100_000_000;

/// GGML tensor element / block types as stored in GGUF.
///
/// Numeric values match the official GGUF specification.
/// Variant names follow the GGML/GGUF convention (e.g. `Q4_K`), not Rust style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // 31-33 removed
    TQ1_0 = 34,
    TQ2_0 = 35,
    // 36-38 removed
    MXFP4 = 39,
}

impl TensorType {
    /// Parse a raw `u32` type id from a GGUF file.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            _ => return None,
        })
    }

    /// Human-readable name (e.g. `"Q4_K"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
        }
    }

    /// Number of elements packed into one quantization block.
    ///
    /// Non-quantized types return `1`.
    pub fn block_size(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,

            // Classic quants: 32 elements / block
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL => 32,

            // K-quants & most IQ types: 256 elements / super-block
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,

            // MXFP4: 32 elements / block
            Self::MXFP4 => 32,
        }
    }

    /// Size in bytes of one quantization block (or one element for non-quants).
    pub fn type_size(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::I8 => 1,

            // Q4_0: scale(f16) + 16 bytes quants
            Self::Q4_0 => 2 + 16,
            // Q4_1: scale(f16) + min(f16) + 16 bytes
            Self::Q4_1 => 2 + 2 + 16,
            // Q5_0: scale(f16) + 4 high bits + 16 quants
            Self::Q5_0 => 2 + 4 + 16,
            // Q5_1: scale + min + 4 high + 16 quants
            Self::Q5_1 => 2 + 2 + 4 + 16,
            // Q8_0: scale(f16) + 32 bytes
            Self::Q8_0 => 2 + 32,
            // Q8_1: scale(f32) + sum(f32) + 32 bytes  — actually scale f16 + d f16 in some docs;
            // llama.cpp: sizeof(block_q8_1) = 4+4+32 = 40? Wait:
            // Actual llama.cpp: block_q8_1 = { ggml_half d; ggml_half s; int8_t qs[32]; } = 2+2+32 = 36
            // But historically some say 36. Using 36.
            Self::Q8_1 => 2 + 2 + 32,

            // K-quants (from ggml-common.h)
            Self::Q2_K => 2 + 2 + 16 + 64,       // 84
            Self::Q3_K => 2 + 12 + 32 + 64,      // 110
            Self::Q4_K => 2 + 2 + 12 + 128,      // 144
            Self::Q5_K => 2 + 2 + 12 + 32 + 128, // 176
            Self::Q6_K => 2 + 128 + 64 + 16,     // 210
            Self::Q8_K => 4 + 256 + 32,          // 292  (d is float)

            // IQ / ternary / MX types (sizes match ggml-common.h block structs)
            Self::IQ2_XXS => 2 + 64,            // 66
            Self::IQ2_XS => 2 + 64 + 16,        // 82
            Self::IQ3_XXS => 2 + 64 + 32,       // 98
            Self::IQ1_S => 2 + 32 + 16,         // 50
            Self::IQ4_NL => 2 + 16,             // 18
            Self::IQ3_S => 2 + 64 + 8 + 32 + 4, // 110
            Self::IQ2_S => 2 + 64 + 8 + 8,      // 82
            Self::IQ4_XS => 2 + 2 + 4 + 128,    // 136
            Self::IQ1_M => 32 + 16 + 8,         // 56
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
            Self::MXFP4 => 1 + 16,
        }
    }

    /// Whether this type is a quantized encoding (not raw float/int).
    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Bytes required to store `n_elements` of this type.
    ///
    /// For quantized types, `n_elements` must be a multiple of [`block_size`](Self::block_size).
    pub fn nbytes_for_elements(self, n_elements: u64) -> Option<u64> {
        let blck = self.block_size();
        if blck == 0 {
            return None;
        }
        if !n_elements.is_multiple_of(blck) {
            // Still allow computing ceiling for incomplete last block? Spec requires full blocks.
            // Return None for non-multiple to catch corrupt shapes.
            return None;
        }
        let n_blocks = n_elements / blck;
        n_blocks.checked_mul(self.type_size())
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// GGUF metadata value type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }
}

/// A typed metadata value from a GGUF key-value pair.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetadataValue {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::U32(v) => *v as u64,
            Self::U64(v) => *v,
            Self::I8(v) if *v >= 0 => *v as u64,
            Self::I16(v) if *v >= 0 => *v as u64,
            Self::I32(v) if *v >= 0 => *v as u64,
            Self::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }

    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            Self::I8(v) => *v as i64,
            Self::I16(v) => *v as i64,
            Self::I32(v) => *v as i64,
            Self::I64(v) => *v,
            Self::U8(v) => *v as i64,
            Self::U16(v) => *v as i64,
            Self::U32(v) => *v as i64,
            Self::U64(v) if *v <= i64::MAX as u64 => *v as i64,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            Self::F32(v) => *v as f64,
            Self::F64(v) => *v,
            _ => return None,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&MetadataArray> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "\"{v}\""),
            Self::U64(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
            Self::Array(a) => {
                write!(f, "[{}; {} elements]", a.element_type.name(), a.len())
            }
        }
    }
}

/// Homogeneous array stored in GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataArray {
    pub element_type: ValueType,
    pub values: Vec<MetadataValue>,
}

impl MetadataArray {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Try to view as a string list (common for tokenizer tokens / merges).
    pub fn as_string_slice(&self) -> Option<Vec<&str>> {
        if self.element_type != ValueType::String {
            return None;
        }
        let mut out = Vec::with_capacity(self.values.len());
        for v in &self.values {
            out.push(v.as_str()?);
        }
        Some(out)
    }
}

/// Description of a single tensor in a GGUF file (metadata only; data is separate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Dimensions in GGUF order (typically row-major; dim[0] is the innermost).
    pub shape: Vec<u64>,
    /// Element / quant type.
    pub tensor_type: TensorType,
    /// Byte offset of tensor data relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements (product of dimensions).
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().copied().product()
    }

    /// Byte size of this tensor's payload, if computable.
    pub fn nbytes(&self) -> Option<u64> {
        self.tensor_type.nbytes_for_elements(self.n_elements())
    }
}

/// Align `offset` up to the next multiple of `alignment`.
#[inline]
pub fn align_offset(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_gguf_le() {
        let bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"GGUF");
    }

    #[test]
    fn align_offset_works() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn quant_block_sizes() {
        assert_eq!(TensorType::Q4_0.type_size(), 18);
        assert_eq!(TensorType::Q8_0.type_size(), 34);
        assert_eq!(TensorType::Q4_K.type_size(), 144);
        assert_eq!(TensorType::Q5_K.type_size(), 176);
        assert_eq!(TensorType::Q6_K.type_size(), 210);
        assert_eq!(TensorType::Q4_0.nbytes_for_elements(32), Some(18));
        assert_eq!(TensorType::F32.nbytes_for_elements(100), Some(400));
    }
}
