//! Row dequantization into f32.

use gguf::TensorType;

use super::blocks::{
    get_scale_min_k4, BLOCK_Q2_K, BLOCK_Q3_K, BLOCK_Q4_0, BLOCK_Q4_1, BLOCK_Q4_K, BLOCK_Q5_0,
    BLOCK_Q5_1, BLOCK_Q5_K, BLOCK_Q6_K, BLOCK_Q8_0, QK4_0, QK5_0, QK8_0, QK_K,
};
use super::half::f16_to_f32;
use crate::error::{CoreError, Result};

/// Dequantize `n` elements starting at element offset `elem_offset` into `out`.
pub fn dequantize(
    data: &[u8],
    ty: TensorType,
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    if out.len() < n {
        return Err(CoreError::Shape(format!(
            "dequant out len {} < n {n}",
            out.len()
        )));
    }
    match ty {
        TensorType::F32 => dequant_f32(data, elem_offset, n, out),
        TensorType::F16 | TensorType::BF16 => dequant_f16_like(data, ty, elem_offset, n, out),
        TensorType::Q4_0 => dequant_q4_0(data, elem_offset, n, out),
        TensorType::Q4_1 => dequant_q4_1(data, elem_offset, n, out),
        TensorType::Q5_0 => dequant_q5_0(data, elem_offset, n, out),
        TensorType::Q5_1 => dequant_q5_1(data, elem_offset, n, out),
        TensorType::Q8_0 => dequant_q8_0(data, elem_offset, n, out),
        TensorType::Q2_K => dequant_q2_k(data, elem_offset, n, out),
        TensorType::Q3_K => dequant_q3_k(data, elem_offset, n, out),
        TensorType::Q4_K => dequant_q4_k(data, elem_offset, n, out),
        TensorType::Q5_K => dequant_q5_k(data, elem_offset, n, out),
        TensorType::Q6_K => dequant_q6_k(data, elem_offset, n, out),
        other => Err(CoreError::UnsupportedTensorType {
            name: "<weight>".into(),
            ty: other.name().into(),
        }),
    }
}

fn dequant_f32(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    let byte_off = elem_offset
        .checked_mul(4)
        .ok_or_else(|| CoreError::Shape("F32 offset overflow".into()))?;
    let need = n
        .checked_mul(4)
        .ok_or_else(|| CoreError::Shape("F32 size overflow".into()))?;
    if byte_off.saturating_add(need) > data.len() {
        return Err(CoreError::Shape("F32 dequant OOB".into()));
    }
    for i in 0..n {
        let o = byte_off + i * 4;
        out[i] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
    }
    Ok(())
}

fn dequant_f16_like(
    data: &[u8],
    ty: TensorType,
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    let byte_off = elem_offset
        .checked_mul(2)
        .ok_or_else(|| CoreError::Shape("F16 offset overflow".into()))?;
    let need = n
        .checked_mul(2)
        .ok_or_else(|| CoreError::Shape("F16 size overflow".into()))?;
    if byte_off.saturating_add(need) > data.len() {
        return Err(CoreError::Shape("F16 dequant OOB".into()));
    }
    for i in 0..n {
        let o = byte_off + i * 2;
        let h = u16::from_le_bytes([data[o], data[o + 1]]);
        out[i] = if ty == TensorType::BF16 {
            f32::from_bits((h as u32) << 16)
        } else {
            f16_to_f32(h)
        };
    }
    Ok(())
}

pub(super) fn dequant_q4_0(
    data: &[u8],
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    if !elem_offset.is_multiple_of(QK4_0) || !n.is_multiple_of(QK4_0) {
        return Err(CoreError::Shape("Q4_0 offset/n not multiple of 32".into()));
    }
    let start_block = elem_offset / QK4_0;
    let nb = n / QK4_0;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_0;
        if base + BLOCK_Q4_0 > data.len() {
            return Err(CoreError::Shape("Q4_0 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 18];
        let y = &mut out[b * QK4_0..(b + 1) * QK4_0];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i8 - 8;
            let x1 = (qs[j] >> 4) as i8 - 8;
            y[j] = x0 as f32 * d;
            y[j + 16] = x1 as f32 * d;
        }
    }
    Ok(())
}

/// Test-only re-export wrapper.
#[cfg(test)]
pub(super) fn dequant_q4_0_for_test(
    data: &[u8],
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    dequant_q4_0(data, elem_offset, n, out)
}

/// Q4_1: d(f16) + m(f16) + qs[16]; value = q*d + m.
pub(super) fn dequant_q4_1(
    data: &[u8],
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    if !elem_offset.is_multiple_of(QK4_0) || !n.is_multiple_of(QK4_0) {
        return Err(CoreError::Shape("Q4_1 offset/n not multiple of 32".into()));
    }
    let start_block = elem_offset / QK4_0;
    let nb = n / QK4_0;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_1;
        if base + BLOCK_Q4_1 > data.len() {
            return Err(CoreError::Shape("Q4_1 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
        let qs = &data[base + 4..base + 20];
        let y = &mut out[b * QK4_0..(b + 1) * QK4_0];
        for j in 0..16 {
            y[j] = (qs[j] & 0x0F) as f32 * d + m;
            y[j + 16] = (qs[j] >> 4) as f32 * d + m;
        }
    }
    Ok(())
}

/// Q5_1: d + m + qh[4] + qs[16]; value = q*d + m.
pub(super) fn dequant_q5_1(
    data: &[u8],
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    if !elem_offset.is_multiple_of(QK5_0) || !n.is_multiple_of(QK5_0) {
        return Err(CoreError::Shape("Q5_1 offset/n not multiple of 32".into()));
    }
    let start_block = elem_offset / QK5_0;
    let nb = n / QK5_0;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q5_1;
        if base + BLOCK_Q5_1 > data.len() {
            return Err(CoreError::Shape("Q5_1 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
        let qh = u32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);
        let qs = &data[base + 8..base + 24];
        let y = &mut out[b * QK5_0..(b + 1) * QK5_0];
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = (qs[j] & 0x0F) as i32 | xh0;
            let x1 = (qs[j] >> 4) as i32 | xh1;
            y[j] = x0 as f32 * d + m;
            y[j + 16] = x1 as f32 * d + m;
        }
    }
    Ok(())
}

fn dequant_q2_k(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    if !elem_offset.is_multiple_of(QK_K) || !n.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("Q2_K offset/n not multiple of 256".into()));
    }
    let start_block = elem_offset / QK_K;
    let nb = n / QK_K;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q2_K;
        if base + BLOCK_Q2_K > data.len() {
            return Err(CoreError::Shape("Q2_K OOB".into()));
        }
        // block_q2_K: scales[16], qs[64], d(f16), dmin(f16)
        let scales = &data[base..base + 16];
        let mut q = &data[base + 16..base + 80];
        let d = f16_to_f32(u16::from_le_bytes([data[base + 80], data[base + 81]]));
        let min = f16_to_f32(u16::from_le_bytes([data[base + 82], data[base + 83]]));
        let mut y_off = b * QK_K;
        let mut is = 0usize;
        for _ in 0..(QK_K / 128) {
            let mut shift = 0u8;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    out[y_off + l] = dl * ((q[l] >> shift) & 3) as f32 - ml;
                }
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    out[y_off + 16 + l] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                }
                y_off += 32;
                shift += 2;
            }
            q = &q[32..];
        }
    }
    Ok(())
}

fn dequant_q3_k(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    if !elem_offset.is_multiple_of(QK_K) || !n.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("Q3_K offset/n not multiple of 256".into()));
    }
    let start_block = elem_offset / QK_K;
    let nb = n / QK_K;
    // block_q3_K: hmask[32], qs[64], scales[12], d(f16) — size 2+12+32+64=110
    // Actually ggml: uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d;
    // type_size says 2+12+32+64 — d first or last? nbytes order in file is packed struct as declared.
    // llama.cpp: typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; ggml_half d; }
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q3_K;
        if base + BLOCK_Q3_K > data.len() {
            return Err(CoreError::Shape("Q3_K OOB".into()));
        }
        let hm = &data[base..base + 32];
        let mut q = &data[base + 32..base + 96];
        let scales_raw = &data[base + 96..base + 108];
        let d_all = f16_to_f32(u16::from_le_bytes([data[base + 108], data[base + 109]]));

        // unpack scales (from dequantize_row_q3_K)
        let mut aux = [0u32; 4];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes([
                scales_raw[i * 4],
                scales_raw[i * 4 + 1],
                scales_raw[i * 4 + 2],
                scales_raw[i * 4 + 3],
            ]);
        }
        let kmask1 = 0x0303_0303u32;
        let kmask2 = 0x0f0f_0f0fu32;
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scales: &[i8] = bytemuck_i8(&aux);

        let mut y_off = b * QK_K;
        let mut is = 0usize;
        let mut m: u8 = 1;
        for _ in 0..(QK_K / 128) {
            let mut shift = 0u8;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv =
                        ((q[l] >> shift) & 3) as i8 as f32 - if hm[l] & m != 0 { 0.0 } else { 4.0 };
                    out[y_off + l] = dl * qv;
                }
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = ((q[l + 16] >> shift) & 3) as i8 as f32
                        - if hm[l + 16] & m != 0 { 0.0 } else { 4.0 };
                    out[y_off + 16 + l] = dl * qv;
                }
                y_off += 32;
                shift += 2;
                m <<= 1;
            }
            q = &q[32..];
        }
    }
    Ok(())
}

fn bytemuck_i8(aux: &[u32; 4]) -> &[i8] {
    // 16 scale bytes as i8
    // SAFETY: `[u32; 4]` is 16 initialized bytes, i8 has alignment 1, and the slice borrows `aux`.
    unsafe { std::slice::from_raw_parts(aux.as_ptr() as *const i8, 16) }
}

/// Q5_0: 32 values, d(f16) + qh[4] + qs[16]; 5-bit quants centered at 16.
/// Matches llama.cpp `dequantize_row_q5_0`.
pub(super) fn dequant_q5_0(
    data: &[u8],
    elem_offset: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    if !elem_offset.is_multiple_of(QK5_0) || !n.is_multiple_of(QK5_0) {
        return Err(CoreError::Shape("Q5_0 offset/n not multiple of 32".into()));
    }
    let start_block = elem_offset / QK5_0;
    let nb = n / QK5_0;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q5_0;
        if base + BLOCK_Q5_0 > data.len() {
            return Err(CoreError::Shape("Q5_0 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        let qs = &data[base + 6..base + 22];
        let y = &mut out[b * QK5_0..(b + 1) * QK5_0];
        for j in 0..16 {
            // llama.cpp: xh_0 = ((qh >> j) << 4) & 0x10;
            //            xh_1 = (qh >> (j + 12)) & 0x10;
            let xh0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = ((qs[j] & 0x0F) as i32 | xh0) - 16;
            let x1 = ((qs[j] >> 4) as i32 | xh1) - 16;
            y[j] = x0 as f32 * d;
            y[j + 16] = x1 as f32 * d;
        }
    }
    Ok(())
}

/// Q5_K super-block (256 elems). Matches llama.cpp `dequantize_row_q5_K`.
fn dequant_q5_k(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    if !elem_offset.is_multiple_of(QK_K) || !n.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("Q5_K offset/n not multiple of 256".into()));
    }
    let start_block = elem_offset / QK_K;
    let nb = n / QK_K;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q5_K;
        if base + BLOCK_Q5_K > data.len() {
            return Err(CoreError::Shape("Q5_K OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[base + 4..base + 16]);
        let qh = &data[base + 16..base + 48];
        let mut ql = &data[base + 48..base + 176];
        let mut y_off = b * QK_K;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..(QK_K / 64) {
            let (sc, m) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                let n1 = (ql[l] & 0x0F) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                out[y_off + l] = d1 * n1 as f32 - m1;
            }
            for l in 0..32 {
                let n2 = (ql[l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                out[y_off + 32 + l] = d2 * n2 as f32 - m2;
            }
            ql = &ql[32..];
            y_off += 64;
            is += 2;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
    Ok(())
}

fn dequant_q8_0(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    if !elem_offset.is_multiple_of(QK8_0) || !n.is_multiple_of(QK8_0) {
        return Err(CoreError::Shape("Q8_0 offset/n not multiple of 32".into()));
    }
    let start_block = elem_offset / QK8_0;
    let nb = n / QK8_0;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q8_0;
        if base + BLOCK_Q8_0 > data.len() {
            return Err(CoreError::Shape("Q8_0 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let y = &mut out[b * QK8_0..(b + 1) * QK8_0];
        for j in 0..32 {
            y[j] = (data[base + 2 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

fn dequant_q4_k(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    if !elem_offset.is_multiple_of(QK_K) || !n.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("Q4_K offset/n not multiple of 256".into()));
    }
    let start_block = elem_offset / QK_K;
    let nb = n / QK_K;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_K;
        if base + BLOCK_Q4_K > data.len() {
            return Err(CoreError::Shape("Q4_K OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[base + 4..base + 16]);
        let mut q = &data[base + 16..base + 144];
        let mut y_off = b * QK_K;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc, m) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                out[y_off + l] = d1 * (q[l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                out[y_off + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
            }
            q = &q[32..];
            y_off += 64;
            is += 2;
        }
    }
    Ok(())
}

fn dequant_q6_k(data: &[u8], elem_offset: usize, n: usize, out: &mut [f32]) -> Result<()> {
    if !elem_offset.is_multiple_of(QK_K) || !n.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("Q6_K offset/n not multiple of 256".into()));
    }
    let start_block = elem_offset / QK_K;
    let nb = n / QK_K;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q6_K;
        if base + BLOCK_Q6_K > data.len() {
            return Err(CoreError::Shape("Q6_K OOB".into()));
        }
        let ql_all = &data[base..base + 128];
        let qh_all = &data[base + 128..base + 192];
        let scales: [i8; 16] = std::array::from_fn(|i| data[base + 192 + i] as i8);
        let d = f16_to_f32(u16::from_le_bytes([data[base + 208], data[base + 209]]));

        let mut y_base = b * QK_K;
        let mut ql = ql_all;
        let mut qh = qh_all;
        let mut sc_off = 0usize;

        for _ in 0..(QK_K / 128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32) - 32;
                let q2 = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32) - 32;
                let sc = &scales[sc_off..];
                out[y_base + l] = d * sc[is] as f32 * q1 as f32;
                out[y_base + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                out[y_base + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                out[y_base + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
            y_base += 128;
            ql = &ql[64..];
            qh = &qh[32..];
            sc_off += 8;
        }
    }
    Ok(())
}
