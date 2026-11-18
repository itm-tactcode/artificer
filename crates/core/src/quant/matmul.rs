//! Multi-threaded matrix-vector product with fused quant dots.
//!
//! Hot paths: Q4_K/Q6_K→Q8_K and Q5_0/Q8_0→Q8_0 activation quant + integer vec_dot
//! (AVX2 / AVX-512 VNNI when available) via a persistent worker pool.

use gguf::TensorType;

use super::blocks::{
    get_scale_min_k4, BLOCK_Q4_0, BLOCK_Q4_K, BLOCK_Q5_0, BLOCK_Q5_K, BLOCK_Q6_K, BLOCK_Q8_0,
    QK4_0, QK5_0, QK8_0, QK_K,
};
use super::dequant::dequantize;
use super::half::f16_to_f32;
use super::pool::{
    parallel_matmul_q4k_q8k, parallel_matmul_q50_q80, parallel_matmul_q6k_q8k,
    parallel_matmul_q80_q80,
};
use super::simd::{dot_f32_fast, q4k_sum_qx_x};
use crate::error::{CoreError, Result};
use crate::profile;

/// `out[j] = sum_i W[i, j] * x[i]` (GGML `mul_mat` layout).
pub fn matmul_vec(
    weight: &[u8],
    ty: TensorType,
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    matmul_vec_impl(weight, ty, k, n, x, out, true, dev_q6k_q8k())
}

/// [`matmul_vec`] for a sub-slice of a larger tensor (MoE expert slices): identical
/// arithmetic paths, but never consults or fills the Q5_0×4 / Q4_K×8 repack caches,
/// which key on the weight pointer and would otherwise copy every expert into the heap.
pub fn matmul_vec_no_repack(
    weight: &[u8],
    ty: TensorType,
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    matmul_vec_impl(weight, ty, k, n, x, out, false, dev_q6k_q8k())
}

/// DEV-ONLY env flag (diagnostic only; never a user option).
/// `ARTIFICER_DEV_Q6K_Q8K=1` makes every CPU Q6_K matmul quantize its activations to Q8_K
/// (integer Q6_K×Q8_K dots, as ggml's CPU backend does) at every width, instead of only on the
/// k ≥ 8192 && n ≤ 8192 speed heuristic below. Read once per process. Unset (the default and
/// release behaviour): the selection is exactly the heuristic, so outputs are byte-identical.
pub(crate) const DEV_Q6K_Q8K_ENV: &str = "ARTIFICER_DEV_Q6K_Q8K";

static DEV_Q6K_Q8K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn dev_q6k_q8k() -> bool {
    *DEV_Q6K_Q8K.get_or_init(|| {
        let on = dev_q6k_q8k_from(std::env::var(DEV_Q6K_Q8K_ENV).ok().as_deref());
        if on {
            eprintln!(
                "[artificer] {DEV_Q6K_Q8K_ENV}=1: DEV-ONLY Q8_K activations on all Q6_K matmuls"
            );
        }
        on
    })
}

/// Parse the dev flag value: only `1` / `true` / `on` (trimmed, case-insensitive) enable it.
pub(crate) fn dev_q6k_q8k_from(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(t) => t == "1" || t == "true" || t == "on",
        None => false,
    }
}

/// Whether a Q6_K matmul of input width `k` and `n` rows takes the Q8_K-activation path.
/// Default (`dev == false`): the speed heuristic — K fat enough to amortize unpack (Nemotron
/// ffn_down k=12544); mid-size Q6 (tiny-vicuna ffn_down k=5632) stays on fused f32 (int
/// unpack was a net loss there); huge-n LM heads also skip. `dev == true`: every Q6_K
/// matmul whose width is a whole number of Q8_K blocks (all real Q6_K tensors).
#[inline]
fn q6k_uses_q8k(k: usize, n: usize, dev: bool) -> bool {
    k.is_multiple_of(QK_K) && k >= QK_K && (dev || (k >= 8192 && n <= 8192))
}

fn matmul_vec_impl(
    weight: &[u8],
    ty: TensorType,
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
    allow_repack: bool,
    q6k_q8k_dev: bool,
) -> Result<()> {
    if x.len() != k || out.len() != n {
        return Err(CoreError::Shape(format!(
            "matmul_vec: x.len={} k={k} out.len={} n={n}",
            x.len(),
            out.len()
        )));
    }

    // Cap threads: oversubscription on high core-count CPUs hurts tiny mats;
    // large LM-head / SSM projections benefit from more than the old hard cap of 8.
    let n_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(16)
        .min(n)
        .max(1);

    // Q4_K hot path: Q8_K activation quant + integer dots (biggest CPU lever).
    if ty == TensorType::Q4_K && k.is_multiple_of(QK_K) && k >= QK_K {
        use super::q8k::{quantize_row_q8_k, BlockQ8K};
        use std::cell::RefCell;
        thread_local! {
            static Q8_BUF: RefCell<Vec<BlockQ8K>> = const { RefCell::new(Vec::new()) };
        }
        return Q8_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            let nb = k / QK_K;
            if buf.len() < nb {
                buf.resize(
                    nb,
                    BlockQ8K {
                        d: 0.0,
                        qs: [0; QK_K],
                        bsums: [0; QK_K / 16],
                    },
                );
            }
            quantize_row_q8_k(x, &mut buf[..nb]);
            let _t = profile::time_q4k();
            parallel_matmul_q4k_q8k(weight, k, &buf[..nb], out, n_threads, allow_repack)
        });
    }

    // Q5_0 hot path: Q8_0 activation quant + integer dots (Nemotron ssm_in / ffn_up).
    if ty == TensorType::Q5_0 && k.is_multiple_of(QK5_0) && k >= QK5_0 {
        use super::q50::{quantize_row_q8_0, BlockQ80};
        use std::cell::RefCell;
        thread_local! {
            static Q80_BUF: RefCell<Vec<BlockQ80>> = const { RefCell::new(Vec::new()) };
        }
        return Q80_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            let nb = k / QK5_0;
            if buf.len() < nb {
                buf.resize(
                    nb,
                    BlockQ80 {
                        d: 0.0,
                        qs: [0; QK8_0],
                    },
                );
            }
            quantize_row_q8_0(x, &mut buf[..nb]);
            let _t = profile::time_q5();
            parallel_matmul_q50_q80(weight, k, &buf[..nb], out, n_threads, allow_repack)
        });
    }

    // Q6_K hot path: Q8_K act quant + int dots, selected by `q6k_uses_q8k` (speed
    // heuristic by default; every width under the dev-only ARTIFICER_DEV_Q6K_Q8K flag).
    if ty == TensorType::Q6_K && q6k_uses_q8k(k, n, q6k_q8k_dev) {
        use super::q8k::{quantize_row_q8_k, BlockQ8K};
        use std::cell::RefCell;
        thread_local! {
            static Q8_BUF_Q6: RefCell<Vec<BlockQ8K>> = const { RefCell::new(Vec::new()) };
        }
        return Q8_BUF_Q6.with(|cell| {
            let mut buf = cell.borrow_mut();
            let nb = k / QK_K;
            if buf.len() < nb {
                buf.resize(
                    nb,
                    BlockQ8K {
                        d: 0.0,
                        qs: [0; QK_K],
                        bsums: [0; QK_K / 16],
                    },
                );
            }
            quantize_row_q8_k(x, &mut buf[..nb]);
            let _t = profile::time_q6k();
            parallel_matmul_q6k_q8k(weight, k, &buf[..nb], out, n_threads)
        });
    }

    // Q8_0 hot path: Q8_0 activation quant + integer dots (LM head).
    if ty == TensorType::Q8_0 && k.is_multiple_of(QK8_0) && k >= QK8_0 {
        use super::q50::{quantize_row_q8_0, BlockQ80};
        use std::cell::RefCell;
        thread_local! {
            static Q80_BUF_OUT: RefCell<Vec<BlockQ80>> = const { RefCell::new(Vec::new()) };
        }
        return Q80_BUF_OUT.with(|cell| {
            let mut buf = cell.borrow_mut();
            let nb = k / QK8_0;
            if buf.len() < nb {
                buf.resize(
                    nb,
                    BlockQ80 {
                        d: 0.0,
                        qs: [0; QK8_0],
                    },
                );
            }
            quantize_row_q8_0(x, &mut buf[..nb]);
            let _t = profile::time_q80();
            parallel_matmul_q80_q80(weight, k, &buf[..nb], out, n_threads)
        });
    }

    if n_threads == 1 || n < 64 {
        let _t = profile::time_mm_other();
        return matmul_vec_chunk(weight, ty, k, x, out, 0);
    }
    let _t = profile::time_mm_other();

    let chunk = n.div_ceil(n_threads);
    std::thread::scope(|scope| -> Result<()> {
        let mut remaining = out;
        let mut start = 0usize;
        let mut handles = Vec::new();
        while start < n {
            let end = (start + chunk).min(n);
            let (part, rest) = remaining.split_at_mut(end - start);
            let part_start = start;
            handles.push(scope.spawn(move || matmul_vec_chunk(weight, ty, k, x, part, part_start)));
            remaining = rest;
            start = end;
        }
        for h in handles {
            h.join()
                .map_err(|_| CoreError::Other("matmul thread panicked".into()))??;
        }
        Ok(())
    })
}

fn matmul_vec_chunk(
    weight: &[u8],
    ty: TensorType,
    k: usize,
    x: &[f32],
    out_chunk: &mut [f32],
    row_base: usize,
) -> Result<()> {
    match ty {
        TensorType::Q4_K if k.is_multiple_of(QK_K) => {
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = fused_dot_q4_k(weight, (row_base + local_j) * k, k, x)?;
            }
            Ok(())
        }
        TensorType::Q5_K if k.is_multiple_of(QK_K) => {
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = fused_dot_q5_k(weight, (row_base + local_j) * k, k, x)?;
            }
            Ok(())
        }
        TensorType::Q6_K if k.is_multiple_of(QK_K) => {
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = fused_dot_q6_k(weight, (row_base + local_j) * k, k, x)?;
            }
            Ok(())
        }
        TensorType::Q4_0 if k.is_multiple_of(QK4_0) => {
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = fused_dot_q4_0(weight, (row_base + local_j) * k, k, x)?;
            }
            Ok(())
        }
        TensorType::Q4_1 if k.is_multiple_of(QK4_0) => {
            let mut row = vec![0.0f32; k];
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                let j = row_base + local_j;
                dequantize(weight, TensorType::Q4_1, j * k, k, &mut row)?;
                *out_j = dot_f32_fast(&row, x);
            }
            Ok(())
        }
        TensorType::Q5_0 if k.is_multiple_of(QK5_0) => {
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = fused_dot_q5_0(weight, (row_base + local_j) * k, k, x)?;
            }
            Ok(())
        }
        TensorType::Q5_1 if k.is_multiple_of(QK5_0) => {
            let mut row = vec![0.0f32; k];
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                let j = row_base + local_j;
                dequantize(weight, TensorType::Q5_1, j * k, k, &mut row)?;
                *out_j = dot_f32_fast(&row, x);
            }
            Ok(())
        }
        TensorType::Q2_K | TensorType::Q3_K if k.is_multiple_of(QK_K) => {
            let mut row = vec![0.0f32; k];
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                let j = row_base + local_j;
                dequantize(weight, ty, j * k, k, &mut row)?;
                *out_j = dot_f32_fast(&row, x);
            }
            Ok(())
        }
        TensorType::Q8_0 if k.is_multiple_of(QK8_0) => {
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = fused_dot_q8_0(weight, (row_base + local_j) * k, k, x)?;
            }
            Ok(())
        }
        TensorType::F32 => matmul_f32_chunk(weight, k, x, out_chunk, row_base),
        _ => {
            let mut row = vec![0.0f32; k];
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                let j = row_base + local_j;
                dequantize(weight, ty, j * k, k, &mut row)?;
                *out_j = dot_f32_fast(&row, x);
            }
            Ok(())
        }
    }
}

fn matmul_f32_chunk(
    weight: &[u8],
    k: usize,
    x: &[f32],
    out_chunk: &mut [f32],
    row_base: usize,
) -> Result<()> {
    for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
        let base = (row_base + local_j)
            .checked_mul(k)
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| CoreError::Shape("F32 matmul offset overflow".into()))?;
        let need = k
            .checked_mul(4)
            .ok_or_else(|| CoreError::Shape("F32 matmul size overflow".into()))?;
        if base.saturating_add(need) > weight.len() {
            return Err(CoreError::Shape("F32 matmul OOB".into()));
        }
        let row_bytes = &weight[base..base + need];
        // Prefer aligned f32 view when possible for AVX2 dots.
        if base % 4 == 0 {
            // SAFETY: `row_bytes` is k*4 in-bounds bytes. f32 alignment is NOT checked: `base` is
            // always a multiple of 4, so the `if` is always true, and alignment rests on the weight
            // buffer itself starting 4-byte aligned (GGUF tensor data in a page-aligned mmap).
            // FIXME(0.1.x): check `row_bytes.as_ptr()` alignment (or use align_to) instead.
            let row_f32 =
                unsafe { std::slice::from_raw_parts(row_bytes.as_ptr() as *const f32, k) };
            *out_j = dot_f32_fast(row_f32, x);
        } else {
            let mut sum = 0.0f32;
            let mut i = 0;
            while i + 8 <= k {
                let mut a = [0.0f32; 8];
                for t in 0..8 {
                    a[t] = le_f32(row_bytes, (i + t) * 4)?;
                }
                sum += a[0] * x[i]
                    + a[1] * x[i + 1]
                    + a[2] * x[i + 2]
                    + a[3] * x[i + 3]
                    + a[4] * x[i + 4]
                    + a[5] * x[i + 5]
                    + a[6] * x[i + 6]
                    + a[7] * x[i + 7];
                i += 8;
            }
            while i < k {
                sum += le_f32(row_bytes, i * 4)? * x[i];
                i += 1;
            }
            *out_j = sum;
        }
    }
    Ok(())
}

#[inline]
fn le_f32(data: &[u8], o: usize) -> Result<f32> {
    let bytes: [u8; 4] = data
        .get(o..o + 4)
        .ok_or_else(|| CoreError::Shape("F32 read OOB".into()))?
        .try_into()
        .map_err(|_| CoreError::Shape("F32 read".into()))?;
    Ok(f32::from_le_bytes(bytes))
}

#[inline]
fn fused_dot_q4_0(data: &[u8], elem_offset: usize, k: usize, x: &[f32]) -> Result<f32> {
    let start_block = elem_offset / QK4_0;
    let nb = k / QK4_0;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_0;
        if base + BLOCK_Q4_0 > data.len() {
            return Err(CoreError::Shape("Q4_0 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 18];
        let x0 = &x[b * QK4_0..];
        for j in 0..16 {
            let q0 = (qs[j] & 0x0F) as i8 as f32 - 8.0;
            let q1 = (qs[j] >> 4) as i8 as f32 - 8.0;
            sum += q0 * d * x0[j] + q1 * d * x0[j + 16];
        }
    }
    Ok(sum)
}

#[inline]
fn fused_dot_q5_0(data: &[u8], elem_offset: usize, k: usize, x: &[f32]) -> Result<f32> {
    let start_block = elem_offset / QK5_0;
    let nb = k / QK5_0;
    let mut sum = 0.0f32;
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
        let x0 = &x[b * QK5_0..];
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let q0 = ((qs[j] & 0x0F) as i32 | xh0) as f32 - 16.0;
            let q1 = ((qs[j] >> 4) as i32 | xh1) as f32 - 16.0;
            sum += q0 * d * x0[j] + q1 * d * x0[j + 16];
        }
    }
    Ok(sum)
}

#[inline]
fn fused_dot_q8_0(data: &[u8], elem_offset: usize, k: usize, x: &[f32]) -> Result<f32> {
    let start_block = elem_offset / QK8_0;
    let nb = k / QK8_0;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q8_0;
        if base + BLOCK_Q8_0 > data.len() {
            return Err(CoreError::Shape("Q8_0 OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let x0 = &x[b * QK8_0..];
        for j in 0..32 {
            sum += (data[base + 2 + j] as i8 as f32) * d * x0[j];
        }
    }
    Ok(sum)
}

fn fused_dot_q4_k(data: &[u8], elem_offset: usize, k: usize, x: &[f32]) -> Result<f32> {
    let start_block = elem_offset / QK_K;
    let nb = k / QK_K;
    let mut sum = 0.0f32;
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
        let mut x_off = b * QK_K;
        let mut is = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc, m) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            // sum += d * sum(nibble*x) - m * sum(x)  (SIMD when available)
            let (sqx, sx) = q4k_sum_qx_x(q, &x[x_off..x_off + 32], false);
            sum += d1 * sqx - m1 * sx;
            let (sqx, sx) = q4k_sum_qx_x(q, &x[x_off + 32..x_off + 64], true);
            sum += d2 * sqx - m2 * sx;

            q = &q[32..];
            x_off += 64;
            is += 2;
        }
    }
    Ok(sum)
}

fn fused_dot_q5_k(data: &[u8], elem_offset: usize, k: usize, x: &[f32]) -> Result<f32> {
    let start_block = elem_offset / QK_K;
    let nb = k / QK_K;
    let mut sum = 0.0f32;
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
        let mut x_off = b * QK_K;
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
                sum += (d1 * n1 as f32 - m1) * x[x_off + l];
            }
            for l in 0..32 {
                let n2 = (ql[l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                sum += (d2 * n2 as f32 - m2) * x[x_off + 32 + l];
            }
            ql = &ql[32..];
            x_off += 64;
            is += 2;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
    Ok(sum)
}

fn fused_dot_q6_k(data: &[u8], elem_offset: usize, k: usize, x: &[f32]) -> Result<f32> {
    let start_block = elem_offset / QK_K;
    let nb = k / QK_K;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q6_K;
        if base + BLOCK_Q6_K > data.len() {
            return Err(CoreError::Shape("Q6_K OOB".into()));
        }
        let ql_all = &data[base..base + 128];
        let qh_all = &data[base + 128..base + 192];
        let scales: [i8; 16] = std::array::from_fn(|i| data[base + 192 + i] as i8);
        let d = f16_to_f32(u16::from_le_bytes([data[base + 208], data[base + 209]]));

        let mut x_base = b * QK_K;
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
                sum += d * sc[is] as f32 * q1 as f32 * x[x_base + l];
                sum += d * sc[is + 2] as f32 * q2 as f32 * x[x_base + l + 32];
                sum += d * sc[is + 4] as f32 * q3 as f32 * x[x_base + l + 64];
                sum += d * sc[is + 6] as f32 * q4 as f32 * x[x_base + l + 96];
            }
            x_base += 128;
            ql = &ql[64..];
            qh = &qh[32..];
            sc_off += 8;
        }
    }
    Ok(sum)
}

#[cfg(test)]
mod dev_q6k_q8k_tests {
    //! DEV-ONLY flag ARTIFICER_DEV_Q6K_Q8K: default path unchanged, flag path = Q8_K reference.
    use super::*;
    use crate::quant::half::f32_to_f16;
    use crate::quant::q6k::vec_dot_q6_k_q8_k;
    use crate::quant::q8k::{quantize_row_q8_k, BlockQ8K};

    /// Deterministic Q6_K rows (n × k) with a sane per-block scale d ≈ 0.01 (bytes 208..210).
    fn q6k_rows(k: usize, n: usize, seed: u32) -> Vec<u8> {
        let nb = n * (k / QK_K);
        let mut s = seed;
        let mut v: Vec<u8> = (0..nb * BLOCK_Q6_K)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        let d = f32_to_f16(0.01).to_le_bytes();
        for b in 0..nb {
            v[b * BLOCK_Q6_K + 208..b * BLOCK_Q6_K + 210].copy_from_slice(&d);
        }
        v
    }

    fn act(k: usize) -> Vec<f32> {
        (0..k)
            .map(|i| (((i * 37 + 11) % 101) as f32) * 0.02 - 1.0)
            .collect()
    }

    /// Today's fused-f32 Q6_K path, row by row (`fused_dot_q6_k`).
    fn ref_f32(w: &[u8], k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        (0..n)
            .map(|j| fused_dot_q6_k(w, j * k, k, x).unwrap())
            .collect()
    }

    /// The Q8_K-activation path, row by row, from the existing helpers.
    fn ref_q8k(w: &[u8], k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let mut y = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; QK_K / 16]
            };
            k / QK_K
        ];
        quantize_row_q8_k(x, &mut y);
        (0..n)
            .map(|j| vec_dot_q6_k_q8_k(w, j * k, k, &y).unwrap())
            .collect()
    }

    fn run(w: &[u8], k: usize, n: usize, x: &[f32], dev: bool, repack: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        matmul_vec_impl(w, TensorType::Q6_K, k, n, x, &mut out, repack, dev).unwrap();
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    // (k, n): below 8192 (OLMoE attn_v k=2048, tiny-vicuna ffn_down k=5632), at/above 8192 with
    // n ≤ 8192 (Nemotron ffn_down k=12544), and n > 8192 (LM-head-like, excluded by default).
    const SHAPES: [(usize, usize); 5] =
        [(2048, 72), (5632, 40), (8192, 24), (12544, 20), (256, 8200)];

    #[test]
    fn flag_off_selection_is_exactly_the_old_heuristic() {
        for k in (0..=16384).step_by(128) {
            for n in [1usize, 64, 8192, 8193, 50304] {
                let old = k % QK_K == 0 && k >= 8192 && n <= 8192;
                assert_eq!(q6k_uses_q8k(k, n, false), old, "k={k} n={n}");
            }
        }
    }

    #[test]
    fn flag_off_is_byte_identical_to_todays_paths_below_and_above_8192() {
        for (i, &(k, n)) in SHAPES.iter().enumerate() {
            let w = q6k_rows(k, n, 7 + i as u32);
            let x = act(k);
            let want = if k >= 8192 && n <= 8192 {
                ref_q8k(&w, k, n, &x)
            } else {
                ref_f32(&w, k, n, &x)
            };
            for repack in [true, false] {
                assert_eq!(
                    bits(&run(&w, k, n, &x, false, repack)),
                    bits(&want),
                    "k={k} n={n} repack={repack}"
                );
            }
            // Public entry points with the variable unset take the same default path.
            if std::env::var_os(DEV_Q6K_Q8K_ENV).is_none() {
                let mut a = vec![0.0f32; n];
                let mut b = vec![0.0f32; n];
                matmul_vec(&w, TensorType::Q6_K, k, n, &x, &mut a).unwrap();
                matmul_vec_no_repack(&w, TensorType::Q6_K, k, n, &x, &mut b).unwrap();
                assert_eq!(bits(&a), bits(&want), "matmul_vec k={k} n={n}");
                assert_eq!(bits(&b), bits(&want), "matmul_vec_no_repack k={k} n={n}");
            }
        }
    }

    #[test]
    fn flag_on_every_width_matches_q8k_activation_reference() {
        for (i, &(k, n)) in SHAPES.iter().enumerate() {
            let w = q6k_rows(k, n, 7 + i as u32);
            let x = act(k);
            let want = ref_q8k(&w, k, n, &x);
            for repack in [true, false] {
                assert_eq!(
                    bits(&run(&w, k, n, &x, true, repack)),
                    bits(&want),
                    "k={k} n={n} repack={repack}"
                );
            }
        }
        // The small-width case really changes arithmetic vs the default f32 path.
        let (k, n) = (2048, 72);
        let w = q6k_rows(k, n, 7);
        let x = act(k);
        assert_ne!(
            bits(&run(&w, k, n, &x, true, true)),
            bits(&ref_f32(&w, k, n, &x))
        );
    }

    #[test]
    fn flag_env_parsing_and_default_read() {
        assert!(!dev_q6k_q8k_from(None));
        for off in ["", "0", "false", "off", "no", "2", "yes"] {
            assert!(!dev_q6k_q8k_from(Some(off)), "{off:?}");
        }
        for on in ["1", " 1 ", "true", "TRUE", "on", "On"] {
            assert!(dev_q6k_q8k_from(Some(on)), "{on:?}");
        }
        assert_eq!(DEV_Q6K_Q8K_ENV, "ARTIFICER_DEV_Q6K_Q8K");
        // Read-once value follows the process environment (unset in normal test runs).
        let expect = dev_q6k_q8k_from(std::env::var(DEV_Q6K_Q8K_ENV).ok().as_deref());
        assert_eq!(dev_q6k_q8k(), expect);
        // Non-Q6_K widths below one Q8_K block never take the Q8_K path, flag or not.
        assert!(!q6k_uses_q8k(128, 4, true));
    }
}
