//! Q5_0 ×4 weight interleave for decode GEMV (ggml-style layout, MIT-pattern adapt).
//!
//! llama.cpp / ggml CPU repack interleaves N output rows at each K-block so a
//! multi-row GEMV streams weights sequentially instead of striding `nb*22`
//! bytes between rows. For Nemotron Q5 `ssm_in`/`ffn_up` (k=3136 → nb=98) the
//! row stride is ~2.1 KiB — this packs 4 rows' blocks for the same `b` into
//! 88 contiguous bytes.
//!
//! Pattern adapted from ggml-org/llama.cpp `ggml/src/ggml-cpu/repack.cpp`
//! (MIT license): Q4_0x8 / Q5_Kx8 interleave idea; reimplemented for Q5_0×4
//! against Artificer's BlockQ80 act path. No ggml source vendored verbatim.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::blocks::{BLOCK_Q5_0, QK5_0};
use super::q50::BlockQ80;
use super::simd::cpu_features;
use crate::error::{CoreError, Result};

/// Bytes per interleaved K-tile (4 × native Q5_0 blocks).
type CacheKey = (usize, usize, usize, usize); // ptr, len, k, n4

fn cache() -> &'static Mutex<HashMap<CacheKey, Arc<Vec<u8>>>> {
    static C: OnceLock<Mutex<HashMap<CacheKey, Arc<Vec<u8>>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Repack row-major Q5_0 into K-major groups of 4 rows.
/// Layout: for g in 0..n4/4, for b in 0..nb: [row0_blk][row1_blk][row2_blk][row3_blk].
pub(crate) fn repack_q5_0_x4(src: &[u8], k: usize, n4: usize) -> Result<Vec<u8>> {
    if !k.is_multiple_of(QK5_0) || !n4.is_multiple_of(4) {
        return Err(CoreError::Shape("Q5_0x4: k%32!=0 or n4%4!=0".into()));
    }
    let nb = k / QK5_0;
    let need = n4 * nb * BLOCK_Q5_0;
    if src.len() < need {
        return Err(CoreError::Shape(format!(
            "Q5_0x4 repack OOB: src={} need={need}",
            src.len()
        )));
    }
    let mut out = vec![0u8; need];
    let n_groups = n4 / 4;
    for g in 0..n_groups {
        for b in 0..nb {
            for r in 0..4 {
                let src_row = g * 4 + r;
                let src_off = (src_row * nb + b) * BLOCK_Q5_0;
                let dst_off = ((g * nb + b) * 4 + r) * BLOCK_Q5_0;
                out[dst_off..dst_off + BLOCK_Q5_0]
                    .copy_from_slice(&src[src_off..src_off + BLOCK_Q5_0]);
            }
        }
    }
    Ok(out)
}

/// Number of cached repacks whose source weight pointer lies in `[start, end)`.
/// Diagnostic / test hook: proves MoE expert slices never enter the cache.
#[allow(dead_code)]
pub(crate) fn cache_entries_in(start: usize, end: usize) -> usize {
    cache()
        .lock()
        .map(|g| g.keys().filter(|key| key.0 >= start && key.0 < end).count())
        .unwrap_or(0)
}

/// Return cached Q5_0x4 repack for the leading `n4 = n & !3` rows, or None if
/// the mat is too small to amortize (decode GEMV ROI gate).
pub(crate) fn q5_0_x4_cached(weight: &[u8], k: usize, n: usize) -> Option<Arc<Vec<u8>>> {
    // Fat-N Q5 mats only (ssm_in n=17504, ffn_up n=12544). Small mats skip.
    if n < 256 || k < QK5_0 || !k.is_multiple_of(QK5_0) {
        return None;
    }
    let n4 = n & !3usize;
    if n4 == 0 {
        return None;
    }
    let need = n4 * (k / QK5_0) * BLOCK_Q5_0;
    if weight.len() < need {
        return None;
    }
    let key: CacheKey = (weight.as_ptr() as usize, weight.len(), k, n4);
    {
        let guard = cache().lock().ok()?;
        if let Some(v) = guard.get(&key) {
            return Some(Arc::clone(v));
        }
    }
    let packed = repack_q5_0_x4(weight, k, n4).ok()?;
    let arc = Arc::new(packed);
    if let Ok(mut guard) = cache().lock() {
        guard.insert(key, Arc::clone(&arc));
    }
    Some(arc)
}

/// Dot `n_rows` consecutive rows from an x4-repacked buffer starting at output
/// row `row_base` (must be multiple of 4 for the SIMD tile path; scalar OK else).
pub(crate) fn vec_dot_q5_0x4_q8_0_nrows(
    packed: &[u8],
    row_base: usize,
    k: usize,
    y: &[BlockQ80],
    out: &mut [f32],
) -> Result<()> {
    let n_rows = out.len();
    if n_rows == 0 {
        return Ok(());
    }
    let nb = k / QK5_0;
    if y.len() < nb {
        return Err(CoreError::Shape("Q8_0 act too short".into()));
    }
    let feat = cpu_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let use_vnni = feat.avx512_vnni_avx2_fma_f16c();
        if (use_vnni || feat.avx2_fma_f16c()) && row_base.is_multiple_of(4) {
            let mut i = 0usize;
            while i + 4 <= n_rows {
                let g = (row_base + i) / 4;
                // SAFETY: the kernel taken (VNNI if `use_vnni`, else AVX2) runs only when
                // everything it enables was detected at run time by `cpu_features()`: AVX-512
                // F/BW/VL/VNNI, AVX2, FMA and F16C, or AVX2, FMA and F16C. The kernels bounds-check
                // `packed` and `y`.
                let (a, b, c, d) = unsafe {
                    if use_vnni {
                        vec_dot_q5_0x4_4rows_vnni(packed, g, nb, y)?
                    } else {
                        vec_dot_q5_0x4_4rows_avx2(packed, g, nb, y)?
                    }
                };
                out[i] = a;
                out[i + 1] = b;
                out[i + 2] = c;
                out[i + 3] = d;
                i += 4;
            }
            // Tail rows within this chunk (1–3): fall back to scalar from packed.
            while i < n_rows {
                out[i] = vec_dot_q5_0x4_one(packed, row_base + i, nb, y)?;
                i += 1;
            }
            return Ok(());
        }
    }
    let _ = feat;
    for (local, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q5_0x4_one(packed, row_base + local, nb, y)?;
    }
    Ok(())
}

fn vec_dot_q5_0x4_one(packed: &[u8], row: usize, nb: usize, y: &[BlockQ80]) -> Result<f32> {
    // Reconstruct a virtual row-major slice is awkward; use group addressing.
    let g = row / 4;
    let r = row % 4;
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = ((g * nb + b) * 4 + r) * BLOCK_Q5_0;
        if base + BLOCK_Q5_0 > packed.len() {
            return Err(CoreError::Shape("Q5_0x4 OOB".into()));
        }
        // Reuse scalar path via a tiny stack copy into the expected layout call:
        // call existing single-block logic inline.
        let blk = &packed[base..base + BLOCK_Q5_0];
        let d_w = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let mut sumi = 0i32;
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let q0 = ((qs[j] & 0x0F) as i32 | xh0) - 16;
            let q1 = ((qs[j] >> 4) as i32 | xh1) - 16;
            sumi += q0 * y[b].qs[j] as i32;
            sumi += q1 * y[b].qs[j + 16] as i32;
        }
        sum += d_w * y[b].d * sumi as f32;
    }
    Ok(sum)
}

#[inline]
fn half_to_f32(h: u16) -> f32 {
    // Match ggml / Artificer half crate path for scales.
    super::half::f16_to_f32(h)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn vec_dot_q5_0x4_4rows_avx2(
    packed: &[u8],
    group: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32, f32, f32)> {
    use super::q50::{
        f16_act_scale_product, hsum_float_8, mul_sum_i8_pairs_float_avx2, unpack_q5_0_block,
    };
    use std::arch::x86_64::*;
    let end = ((group * nb + nb - 1) * 4 + 4) * BLOCK_Q5_0;
    if nb == 0 || end > packed.len() {
        // Precise bound: last tile
        let last = ((group * nb + (nb.saturating_sub(1))) * 4 + 3) * BLOCK_Q5_0 + BLOCK_Q5_0;
        if last > packed.len() {
            return Err(CoreError::Shape("Q5_0x4 OOB".into()));
        }
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let w = packed.as_ptr();
    for b in 0..nb {
        let tile = (group * nb + b) * 4 * BLOCK_Q5_0;
        if b + 1 < nb {
            let pref = w.add((group * nb + b + 1) * 4 * BLOCK_Q5_0);
            _mm_prefetch(pref as *const i8, _MM_HINT_T0);
        }
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let yd = y[b].d;
        let base0 = tile;
        let base1 = tile + BLOCK_Q5_0;
        let base2 = tile + 2 * BLOCK_Q5_0;
        let base3 = tile + 3 * BLOCK_Q5_0;
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base0], packed[base0 + 1]]),
            yd,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base1], packed[base1 + 1]]),
            yd,
        ));
        let d2 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base2], packed[base2 + 1]]),
            yd,
        ));
        let d3 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base3], packed[base3 + 1]]),
            yd,
        ));
        let qx0 = unpack_q5_0_block(w.add(base0 + 6), w.add(base0 + 2));
        let qx1 = unpack_q5_0_block(w.add(base1 + 6), w.add(base1 + 2));
        let qx2 = unpack_q5_0_block(w.add(base2 + 6), w.add(base2 + 2));
        let qx3 = unpack_q5_0_block(w.add(base3 + 6), w.add(base3 + 2));
        acc0 = _mm256_fmadd_ps(d0, mul_sum_i8_pairs_float_avx2(qx0, qy), acc0);
        acc1 = _mm256_fmadd_ps(d1, mul_sum_i8_pairs_float_avx2(qx1, qy), acc1);
        acc2 = _mm256_fmadd_ps(d2, mul_sum_i8_pairs_float_avx2(qx2, qy), acc2);
        acc3 = _mm256_fmadd_ps(d3, mul_sum_i8_pairs_float_avx2(qx3, qy), acc3);
    }
    Ok((
        hsum_float_8(acc0),
        hsum_float_8(acc1),
        hsum_float_8(acc2),
        hsum_float_8(acc3),
    ))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vec_dot_q5_0x4_4rows_vnni(
    packed: &[u8],
    group: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32, f32, f32)> {
    use super::q50::{
        f16_act_scale_product, hsum_float_8, mul_sum_i8_pairs_float_vnni, unpack_q5_0_block,
    };
    use std::arch::x86_64::*;
    let last = ((group * nb + (nb.saturating_sub(1))) * 4 + 3) * BLOCK_Q5_0 + BLOCK_Q5_0;
    if nb == 0 || last > packed.len() {
        return Err(CoreError::Shape("Q5_0x4 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let w = packed.as_ptr();
    for b in 0..nb {
        let tile = (group * nb + b) * 4 * BLOCK_Q5_0;
        if b + 1 < nb {
            let pref = w.add((group * nb + b + 1) * 4 * BLOCK_Q5_0);
            _mm_prefetch(pref as *const i8, _MM_HINT_T0);
            _mm_prefetch(pref.add(64) as *const i8, _MM_HINT_T0);
        }
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let yd = y[b].d;
        let base0 = tile;
        let base1 = tile + BLOCK_Q5_0;
        let base2 = tile + 2 * BLOCK_Q5_0;
        let base3 = tile + 3 * BLOCK_Q5_0;
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base0], packed[base0 + 1]]),
            yd,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base1], packed[base1 + 1]]),
            yd,
        ));
        let d2 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base2], packed[base2 + 1]]),
            yd,
        ));
        let d3 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([packed[base3], packed[base3 + 1]]),
            yd,
        ));
        let qx0 = unpack_q5_0_block(w.add(base0 + 6), w.add(base0 + 2));
        let qx1 = unpack_q5_0_block(w.add(base1 + 6), w.add(base1 + 2));
        let qx2 = unpack_q5_0_block(w.add(base2 + 6), w.add(base2 + 2));
        let qx3 = unpack_q5_0_block(w.add(base3 + 6), w.add(base3 + 2));
        acc0 = _mm256_fmadd_ps(d0, mul_sum_i8_pairs_float_vnni(qx0, qy), acc0);
        acc1 = _mm256_fmadd_ps(d1, mul_sum_i8_pairs_float_vnni(qx1, qy), acc1);
        acc2 = _mm256_fmadd_ps(d2, mul_sum_i8_pairs_float_vnni(qx2, qy), acc2);
        acc3 = _mm256_fmadd_ps(d3, mul_sum_i8_pairs_float_vnni(qx3, qy), acc3);
    }
    Ok((
        hsum_float_8(acc0),
        hsum_float_8(acc1),
        hsum_float_8(acc2),
        hsum_float_8(acc3),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::q50::{quantize_row_q8_0, vec_dot_q5_0_q8_0_nrows, BlockQ80};

    #[test]
    fn q5_0x4_repack_matches_row_major() {
        let k = 64;
        let n = 8;
        let nb = k / QK5_0;
        let mut weight = vec![0u8; n * nb * BLOCK_Q5_0];
        for r in 0..n {
            for b in 0..nb {
                let base = (r * nb + b) * BLOCK_Q5_0;
                // d = 1.0 f16
                weight[base] = 0x00;
                weight[base + 1] = 0x3c;
                weight[base + 2] = (r + b) as u8;
                weight[base + 3] = (r * 3 + b) as u8;
                weight[base + 4] = (r * 5) as u8;
                weight[base + 5] = (b * 7) as u8;
                for j in 0..16 {
                    weight[base + 6 + j] = ((r * 17 + b * 3 + j) & 0xff) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let mut y = vec![
            BlockQ80 {
                d: 0.0,
                qs: [0; 32]
            };
            nb
        ];
        quantize_row_q8_0(&x, &mut y);

        let packed = repack_q5_0_x4(&weight, k, n).unwrap();
        let mut out_rm = vec![0.0f32; n];
        let mut out_x4 = vec![0.0f32; n];
        vec_dot_q5_0_q8_0_nrows(&weight, 0, k, &y, &mut out_rm).unwrap();
        vec_dot_q5_0x4_q8_0_nrows(&packed, 0, k, &y, &mut out_x4).unwrap();
        for r in 0..n {
            let rel = (out_rm[r] - out_x4[r]).abs() / (out_rm[r].abs() + 1e-6);
            assert!(
                rel < 1e-5,
                "row {r}: rm={} x4={} rel={rel}",
                out_rm[r],
                out_x4[r]
            );
        }
    }
}
