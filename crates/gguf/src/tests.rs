//! Integration-style unit tests using a synthetic GGUF writer.

use std::io::Write;

use crate::types::{
    align_offset, MetadataValue, TensorType, ValueType, DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use crate::{GgufModel, LoadMode, LoadOptions};

/// Minimal little-endian GGUF writer for tests.
struct GgufWriter {
    buf: Vec<u8>,
}

impl GgufWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_i32(&mut self, v: i32) {
        self.write_u32(v as u32);
    }
    fn write_f32(&mut self, v: f32) {
        self.write_u32(v.to_bits());
    }
    fn write_f64(&mut self, v: f64) {
        self.write_u64(v.to_bits());
    }
    fn write_string(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn write_value(&mut self, v: &MetadataValue) {
        match v {
            MetadataValue::U8(x) => self.write_u8(*x),
            MetadataValue::I8(x) => self.write_u8(*x as u8),
            MetadataValue::U16(x) => self.write_u16(*x),
            MetadataValue::I16(x) => self.write_u16(*x as u16),
            MetadataValue::U32(x) => self.write_u32(*x),
            MetadataValue::I32(x) => self.write_i32(*x),
            MetadataValue::F32(x) => self.write_f32(*x),
            MetadataValue::Bool(x) => self.write_u8(u8::from(*x)),
            MetadataValue::String(x) => self.write_string(x),
            MetadataValue::U64(x) => self.write_u64(*x),
            MetadataValue::I64(x) => self.write_u64(*x as u64),
            MetadataValue::F64(x) => self.write_f64(*x),
            MetadataValue::Array(a) => {
                self.write_u32(a.element_type as u32);
                self.write_u64(a.values.len() as u64);
                for el in &a.values {
                    self.write_value(el);
                }
            }
        }
    }

    fn write_kv(&mut self, key: &str, value: &MetadataValue) {
        self.write_string(key);
        self.write_u32(value.value_type() as u32);
        self.write_value(value);
    }

    fn pad_to_alignment(&mut self, alignment: u64) {
        let aligned = align_offset(self.buf.len() as u64, alignment) as usize;
        while self.buf.len() < aligned {
            self.buf.push(0);
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

fn build_sample_gguf() -> Vec<u8> {
    let alignment = DEFAULT_ALIGNMENT;
    let mut w = GgufWriter::new();

    // Header
    w.write_u32(GGUF_MAGIC);
    w.write_u32(3); // version
    w.write_u64(2); // tensor_count
    w.write_u64(8); // metadata_kv_count

    // Metadata
    w.write_kv(
        "general.architecture",
        &MetadataValue::String("llama".into()),
    );
    w.write_kv("general.name", &MetadataValue::String("test-model".into()));
    w.write_kv("general.alignment", &MetadataValue::U32(alignment as u32));
    w.write_kv("general.quantization_version", &MetadataValue::U32(2));
    w.write_kv("llama.context_length", &MetadataValue::U64(2048));
    w.write_kv("llama.embedding_length", &MetadataValue::U64(64));
    w.write_kv("llama.block_count", &MetadataValue::U64(2));
    w.write_kv("llama.attention.head_count", &MetadataValue::U64(4));

    // Tensor 0: token_embd.weight  [64, 32] F32  → 8192 bytes
    // GGUF dims: first dim is typically the leftmost as stored
    let embd_elems = 64u64 * 32;
    let embd_bytes = embd_elems * 4;
    w.write_string("token_embd.weight");
    w.write_u32(2); // n_dims
    w.write_u64(64);
    w.write_u64(32);
    w.write_u32(TensorType::F32 as u32);
    w.write_u64(0); // offset relative to tensor data

    // Tensor 1: blk.0.attn_q.weight  [64, 64] Q4_0
    // n_elements must be multiple of 32
    let q_elems = 64u64 * 64;
    let q_bytes = TensorType::Q4_0.nbytes_for_elements(q_elems).unwrap();
    let q_offset = align_offset(embd_bytes, alignment);
    w.write_string("blk.0.attn_q.weight");
    w.write_u32(2);
    w.write_u64(64);
    w.write_u64(64);
    w.write_u32(TensorType::Q4_0 as u32);
    w.write_u64(q_offset);

    // Align to tensor data section
    w.pad_to_alignment(alignment);
    let tensor_data_start = w.buf.len();
    assert_eq!(tensor_data_start as u64 % alignment, 0);

    // Tensor 0 data
    for i in 0..embd_elems {
        w.write_f32(i as f32 * 0.01);
    }

    // Pad to q_offset
    while (w.buf.len() as u64) < tensor_data_start as u64 + q_offset {
        w.write_u8(0);
    }

    // Tensor 1 data (dummy quant blocks)
    let n_blocks = q_elems / 32;
    for b in 0..n_blocks {
        // scale f16 = 1.0 ≈ 0x3C00
        w.write_u16(0x3C00);
        for j in 0..16u8 {
            w.write_u8(j.wrapping_add(b as u8));
        }
    }
    assert_eq!(
        w.buf.len() as u64,
        tensor_data_start as u64 + q_offset + q_bytes
    );

    w.into_bytes()
}

#[test]
fn load_synthetic_gguf_buffered() {
    let bytes = build_sample_gguf();
    let model = GgufModel::from_bytes(bytes).expect("parse sample GGUF");

    assert_eq!(model.version(), 3);
    assert_eq!(model.alignment(), 32);
    assert_eq!(model.tensor_count(), 2);

    let meta = model.metadata();
    assert_eq!(meta.architecture().unwrap(), Some("llama"));
    assert_eq!(meta.name().unwrap(), Some("test-model"));
    assert_eq!(meta.context_length().unwrap(), Some(2048));
    assert_eq!(meta.embedding_length().unwrap(), Some(64));
    assert_eq!(meta.block_count().unwrap(), Some(2));
    assert_eq!(meta.head_count().unwrap(), Some(4));

    let embd = model.find_tensor("token_embd.weight").expect("embd");
    assert_eq!(embd.shape, vec![64, 32]);
    assert_eq!(embd.tensor_type, TensorType::F32);
    assert_eq!(embd.n_elements(), 2048);
    assert_eq!(embd.nbytes(), Some(8192));

    let data = model.tensor_data("token_embd.weight").unwrap();
    assert_eq!(data.len(), 8192);
    // first f32 should be 0.0
    let first = f32::from_le_bytes(data[0..4].try_into().unwrap());
    assert_eq!(first, 0.0);
    let second = f32::from_le_bytes(data[4..8].try_into().unwrap());
    assert!((second - 0.01).abs() < 1e-6);

    let q = model.find_tensor("blk.0.attn_q.weight").expect("q");
    assert_eq!(q.tensor_type, TensorType::Q4_0);
    let qdata = model.tensor_data("blk.0.attn_q.weight").unwrap();
    assert_eq!(qdata.len() as u64, q.nbytes().unwrap());

    let summary = model.summary();
    assert_eq!(summary.architecture.as_deref(), Some("llama"));
    assert_eq!(summary.tensor_count, 2);
}

#[test]
fn load_synthetic_gguf_mmap() {
    let bytes = build_sample_gguf();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sample.gguf");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    let model = GgufModel::load_with(
        &path,
        LoadOptions {
            mode: LoadMode::Mmap,
            validate_tensor_bounds: true,
        },
    )
    .expect("mmap load");

    assert_eq!(model.tensor_count(), 2);
    assert!(!model.tensor_data("token_embd.weight").unwrap().is_empty());
    println!("{}", model.summary());
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = build_sample_gguf();
    bytes[0] = b'X';
    let err = GgufModel::from_bytes(bytes).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("magic"), "{msg}");
}

#[test]
fn rejects_unsupported_version() {
    let mut w = GgufWriter::new();
    w.write_u32(GGUF_MAGIC);
    w.write_u32(99);
    w.write_u64(0);
    w.write_u64(0);
    let err = GgufModel::from_bytes(w.into_bytes()).unwrap_err();
    assert!(err.to_string().contains("version"));
}

#[test]
fn rejects_missing_tensor() {
    let model = GgufModel::from_bytes(build_sample_gguf()).unwrap();
    assert!(model.tensor_data("nope").is_err());
}

#[test]
fn value_type_roundtrip_ids() {
    for id in 0..=12u32 {
        assert_eq!(ValueType::from_u32(id).unwrap() as u32, id);
    }
}

#[test]
fn metadata_array_strings() {
    let mut w = GgufWriter::new();
    w.write_u32(GGUF_MAGIC);
    w.write_u32(3);
    w.write_u64(0);
    w.write_u64(1);

    use crate::types::MetadataArray;
    let arr = MetadataValue::Array(MetadataArray {
        element_type: ValueType::String,
        values: vec![
            MetadataValue::String("a".into()),
            MetadataValue::String("b".into()),
        ],
    });
    w.write_kv("tokenizer.ggml.tokens", &arr);

    let model = GgufModel::from_bytes(w.into_bytes()).unwrap();
    let v = model.metadata().get("tokenizer.ggml.tokens").unwrap();
    let strings = v.as_array().unwrap().as_string_slice().unwrap();
    assert_eq!(strings, vec!["a", "b"]);
}
