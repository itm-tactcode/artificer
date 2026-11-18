//! Low-level little-endian binary reader over a byte slice.

use crate::error::{Error, Result};
use crate::types::{MetadataArray, MetadataValue, ValueType, MAX_ARRAY_LEN, MAX_STRING_LEN};

/// Cursor over a contiguous byte buffer (typically a memory-mapped GGUF file).
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn position(&self) -> u64 {
        self.pos as u64
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(Error::UnexpectedEof {
                offset: self.pos as u64,
                needed: n,
            });
        }
        Ok(())
    }

    pub fn read_exact(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let start = self.pos;
        self.pos += n;
        Ok(&self.data[start..self.pos])
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let b = self.read_exact(1)?;
        Ok(b[0])
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_exact(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(Error::InvalidBool(v)),
        }
    }

    /// Read a GGUF string: `u64` length + UTF-8 bytes (not null-terminated).
    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_LEN {
            return Err(Error::StringTooLong {
                len,
                offset: self.position(),
            });
        }
        let len = len as usize;
        let offset = self.position();
        let bytes = self.read_exact(len)?;
        let s =
            std::str::from_utf8(bytes).map_err(|source| Error::InvalidUtf8 { offset, source })?;
        Ok(s.to_owned())
    }

    /// Read a metadata value of the given type.
    pub fn read_value(&mut self, ty: ValueType) -> Result<MetadataValue> {
        Ok(match ty {
            ValueType::U8 => MetadataValue::U8(self.read_u8()?),
            ValueType::I8 => MetadataValue::I8(self.read_i8()?),
            ValueType::U16 => MetadataValue::U16(self.read_u16()?),
            ValueType::I16 => MetadataValue::I16(self.read_i16()?),
            ValueType::U32 => MetadataValue::U32(self.read_u32()?),
            ValueType::I32 => MetadataValue::I32(self.read_i32()?),
            ValueType::F32 => MetadataValue::F32(self.read_f32()?),
            ValueType::Bool => MetadataValue::Bool(self.read_bool()?),
            ValueType::String => MetadataValue::String(self.read_string()?),
            ValueType::U64 => MetadataValue::U64(self.read_u64()?),
            ValueType::I64 => MetadataValue::I64(self.read_i64()?),
            ValueType::F64 => MetadataValue::F64(self.read_f64()?),
            ValueType::Array => MetadataValue::Array(self.read_array()?),
        })
    }

    /// Read an array: element type, length, then elements.
    pub fn read_array(&mut self) -> Result<MetadataArray> {
        let type_id = self.read_u32()?;
        let element_type =
            ValueType::from_u32(type_id).ok_or(Error::InvalidMetadataType(type_id))?;
        let len = self.read_u64()?;
        if len > MAX_ARRAY_LEN {
            return Err(Error::ArrayTooLarge {
                len,
                offset: self.position(),
            });
        }
        let mut values = Vec::with_capacity(len.min(1_000_000) as usize);
        for _ in 0..len {
            // Nested arrays are allowed by the spec.
            values.push(self.read_value(element_type)?);
        }
        Ok(MetadataArray {
            element_type,
            values,
        })
    }

    /// Read a typed metadata key-value pair: key, type, value.
    pub fn read_metadata_kv(&mut self) -> Result<(String, MetadataValue)> {
        let key = self.read_string()?;
        let type_id = self.read_u32()?;
        let ty = ValueType::from_u32(type_id).ok_or(Error::InvalidMetadataType(type_id))?;
        let value = self.read_value(ty)?;
        Ok((key, value))
    }
}
