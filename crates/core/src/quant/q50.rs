//! Q5_0 × Q8_0 and Q8_0 × Q8_0 integer vec_dot (ggml-compatible).
//!
//! Q5_0 covers Nemotron SSM `ssm_in` / FFN up / attn projections (~2.3B elems);
//! Q8_0 covers the LM head. ggml-style SIMD unpack + VNNI.
//! F16C scale convert, single OOB check, software-pipelined dual acc.

use super::blocks::{BLOCK_Q5_0, BLOCK_Q8_0, QK5_0, QK8_0};
use super::half::f16_to_f32;
use super::simd::cpu_features;
use crate::error::{CoreError, Result};

/// Activation-side Q8_0 scratch block: d as f32 (avoids per-dot F16 convert) + qs[32].
/// Not GGUF wire layout — weights remain packed Q5_0/Q8_0; only the quantized act uses this.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct BlockQ80 {
    pub d: f32,
    pub qs: [i8; QK8_0],
}

/// Quantize `x` (len multiple of 32) into ggml Q8_0 blocks.
pub(crate) fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ80]) {
    debug_assert_eq!(x.len() % QK8_0, 0);
    debug_assert_eq!(out.len(), x.len() / QK8_0);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpu_features().avx2_fma() {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`.
            unsafe { quantize_row_q8_0_avx2(x, out) };
            return;
        }
    }
    quantize_row_q8_0_scalar(x, out);
}

fn quantize_row_q8_0_scalar(x: &[f32], out: &mut [BlockQ80]) {
    for (bi, block) in out.iter_mut().enumerate() {
        let xb = &x[bi * QK8_0..(bi + 1) * QK8_0];
        let mut amax = 0.0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            continue;
        }
        let d = amax / 127.0;
        let id = 1.0 / d;
        block.d = d;
        for j in 0..QK8_0 {
            let mut v = (xb[j] * id).round() as i32;
            if v > 127 {
                v = 127;
            }
            if v < -128 {
                v = -128;
            }
            block.qs[j] = v as i8;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_q8_0_avx2(x: &[f32], out: &mut [BlockQ80]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let nb = out.len();
    let mut xi = 0usize;
    for bi in 0..nb {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(xi));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(xi + 8));
        let v2 = _mm256_loadu_ps(x.as_ptr().add(xi + 16));
        let v3 = _mm256_loadu_ps(x.as_ptr().add(xi + 24));
        xi += 32;

        let sign = _mm256_set1_ps(-0.0);
        let mut max_abs = _mm256_andnot_ps(sign, v0);
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign, v1));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign, v2));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign, v3));
        let max4 = {
            let hi = _mm256_extractf128_ps(max_abs, 1);
            let lo = _mm256_castps256_ps128(max_abs);
            let mut m = _mm_max_ps(hi, lo);
            m = _mm_max_ps(m, _mm_movehl_ps(m, m));
            m = _mm_max_ss(m, _mm_movehdup_ps(m));
            _mm_cvtss_f32(m)
        };
        let block = &mut out[bi];
        if max4 == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            continue;
        }
        let d = max4 / 127.0;
        block.d = d;
        let id = 127.0 / max4;
        let mul = _mm256_set1_ps(id);
        let r0 = _mm256_round_ps(
            _mm256_mul_ps(v0, mul),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let r1 = _mm256_round_ps(
            _mm256_mul_ps(v1, mul),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let r2 = _mm256_round_ps(
            _mm256_mul_ps(v2, mul),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let r3 = _mm256_round_ps(
            _mm256_mul_ps(v3, mul),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        );
        let i0 = _mm256_cvtps_epi32(r0);
        let i1 = _mm256_cvtps_epi32(r1);
        let i2 = _mm256_cvtps_epi32(r2);
        let i3 = _mm256_cvtps_epi32(r3);
        // pack i32→i16→i8 with AVX2 lane quirks, then permute to linear order
        let p01 = _mm256_packs_epi32(i0, i1);
        let p23 = _mm256_packs_epi32(i2, i3);
        let packed = _mm256_packs_epi16(p01, p23);
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let ordered = _mm256_permutevar8x32_epi32(packed, perm);
        _mm256_storeu_si256(block.qs.as_mut_ptr() as *mut __m256i, ordered);
    }
}

/// Integer Q5_0 × Q8_0 dot for one weight row.
pub(crate) fn vec_dot_q5_0_q8_0(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    let start_block = elem_offset / QK5_0;
    let nb = k / QK5_0;
    if y.len() < nb {
        return Err(CoreError::Shape("Q8_0 act too short".into()));
    }
    let feat = cpu_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if feat.avx512_vnni_avx2_fma_f16c() {
            // SAFETY: AVX-512 F/BW/VL/VNNI, AVX2, FMA and F16C, everything the kernel enables, were
            // detected at run time by `cpu_features()`.
            return unsafe { vec_dot_q5_0_q8_0_avx512_vnni(weight, start_block, nb, y) };
        }
        if feat.avx2_fma_f16c() {
            // SAFETY: AVX2, FMA and F16C, everything the kernel enables, were detected at run time
            // by `cpu_features()`.
            return unsafe { vec_dot_q5_0_q8_0_avx2(weight, start_block, nb, y) };
        }
    }
    let _ = feat;
    vec_dot_q5_0_q8_0_scalar(weight, start_block, nb, y)
}

/// Dot 2 consecutive weight rows sharing the same Q8_0 activation (GEMV tile).
/// Amortizes y loads across rows — main lever for fat-N Q5 mats.
pub(crate) fn vec_dot_q5_0_q8_0_nrows(
    weight: &[u8],
    elem_offset0: usize,
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
    let start0 = elem_offset0 / QK5_0;
    // Contiguous rows: row r starts at start0 + r*nb blocks.
    let feat = cpu_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let use_vnni = feat.avx512_vnni_avx2_fma_f16c();
        if use_vnni || feat.avx2_fma_f16c() {
            let mut i = 0usize;
            while i + 4 <= n_rows {
                // SAFETY: the kernel taken (VNNI if `use_vnni`, else AVX2) runs only when
                // everything it enables was detected at run time by `cpu_features()`: AVX-512
                // F/BW/VL/VNNI, AVX2, FMA and F16C, or AVX2, FMA and F16C. The kernels bounds-check
                // `weight` and `y`.
                let (a, b, c, d) = unsafe {
                    if use_vnni {
                        vec_dot_q5_0_q8_0_4rows_vnni(weight, start0 + i * nb, nb, y)?
                    } else {
                        vec_dot_q5_0_q8_0_4rows_avx2(weight, start0 + i * nb, nb, y)?
                    }
                };
                out[i] = a;
                out[i + 1] = b;
                out[i + 2] = c;
                out[i + 3] = d;
                i += 4;
            }
            while i + 2 <= n_rows {
                // SAFETY: the kernel taken (VNNI if `use_vnni`, else AVX2) runs only when
                // everything it enables was detected at run time by `cpu_features()`: AVX-512
                // F/BW/VL/VNNI, AVX2, FMA and F16C, or AVX2, FMA and F16C. The kernels bounds-check
                // `weight` and `y`.
                let (a, b) = unsafe {
                    if use_vnni {
                        vec_dot_q5_0_q8_0_2rows_vnni(weight, start0 + i * nb, nb, y)?
                    } else {
                        vec_dot_q5_0_q8_0_2rows_avx2(weight, start0 + i * nb, nb, y)?
                    }
                };
                out[i] = a;
                out[i + 1] = b;
                i += 2;
            }
            while i < n_rows {
                out[i] = vec_dot_q5_0_q8_0(weight, elem_offset0 + i * k, k, y)?;
                i += 1;
            }
            return Ok(());
        }
    }
    let _ = feat;
    for (local, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q5_0_q8_0(weight, elem_offset0 + local * k, k, y)?;
    }
    Ok(())
}

/// Dot 2 consecutive Q8_0 weight rows sharing activation (LM head tile).
pub(crate) fn vec_dot_q8_0_q8_0_nrows(
    weight: &[u8],
    elem_offset0: usize,
    k: usize,
    y: &[BlockQ80],
    out: &mut [f32],
) -> Result<()> {
    let n_rows = out.len();
    if n_rows == 0 {
        return Ok(());
    }
    let nb = k / QK8_0;
    if y.len() < nb {
        return Err(CoreError::Shape("Q8_0 act too short".into()));
    }
    let start0 = elem_offset0 / QK8_0;
    let feat = cpu_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let use_vnni = feat.avx512_vnni_avx2_fma_f16c();
        if use_vnni || feat.avx2_fma_f16c() {
            let mut i = 0usize;
            while i + 2 <= n_rows {
                // SAFETY: the kernel taken (VNNI if `use_vnni`, else AVX2) runs only when
                // everything it enables was detected at run time by `cpu_features()`: AVX-512
                // F/BW/VL/VNNI, AVX2, FMA and F16C, or AVX2, FMA and F16C. The kernels bounds-check
                // `weight` and `y`.
                let (a, b) = unsafe {
                    if use_vnni {
                        vec_dot_q8_0_q8_0_2rows_vnni(weight, start0 + i * nb, nb, y)?
                    } else {
                        vec_dot_q8_0_q8_0_2rows_avx2(weight, start0 + i * nb, nb, y)?
                    }
                };
                out[i] = a;
                out[i + 1] = b;
                i += 2;
            }
            while i < n_rows {
                out[i] = vec_dot_q8_0_q8_0(weight, elem_offset0 + i * k, k, y)?;
                i += 1;
            }
            return Ok(());
        }
    }
    let _ = feat;
    for (local, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q8_0_q8_0(weight, elem_offset0 + local * k, k, y)?;
    }
    Ok(())
}

fn vec_dot_q5_0_q8_0_scalar(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q5_0;
        if base + BLOCK_Q5_0 > weight.len() {
            return Err(CoreError::Shape("Q5_0 OOB".into()));
        }
        let d5 = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
        let qh = u32::from_le_bytes([
            weight[base + 2],
            weight[base + 3],
            weight[base + 4],
            weight[base + 5],
        ]);
        let qs = &weight[base + 6..base + 22];
        let yb = &y[b];
        let d8 = yb.d;
        let mut sumi = 0i32;
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
            let q0 = ((qs[j] & 0x0F) as i32 | xh0) - 16;
            let q1 = ((qs[j] >> 4) as i32 | xh1) - 16;
            sumi += q0 * yb.qs[j] as i32;
            sumi += q1 * yb.qs[j + 16] as i32;
        }
        sum += d5 * d8 * sumi as f32;
    }
    Ok(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(super) unsafe fn bytes_from_nibbles_32(qs: *const u8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let tmp = _mm_loadu_si128(qs as *const __m128i);
    let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    let low_mask = _mm256_set1_epi8(0x0F);
    _mm256_and_si256(low_mask, bytes)
}

/// Spread 32 qh bits → 32 bytes {0x00, 0xFF}.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(super) unsafe fn bytes_from_bits_32(qh: *const u8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let mut x32 = 0u32;
    core::ptr::copy_nonoverlapping(qh, &mut x32 as *mut u32 as *mut u8, 4);
    let shuf_mask = _mm256_set_epi64x(
        0x0303_0303_0303_0303,
        0x0202_0202_0202_0202,
        0x0101_0101_0101_0101,
        0x0000_0000_0000_0000,
    );
    let mut bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32 as i32), shuf_mask);
    let bit_mask = _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfeu64 as i64);
    bytes = _mm256_or_si256(bytes, bit_mask);
    _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
}

/// Unpack one Q5_0 block's qs+qh into 32 signed i8 values in [-16, 15].
/// ggml trick: OR 0xF0 into lanes whose high bit is clear → nibble−16 as i8.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(super) unsafe fn unpack_q5_0_block(qs: *const u8, qh: *const u8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let mut qx = bytes_from_nibbles_32(qs);
    let mut bxhi = bytes_from_bits_32(qh);
    bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8(0xF0u8 as i8));
    qx = _mm256_or_si256(qx, bxhi);
    qx
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(super) unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut res = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// Signed i8×i8 pairwise sum → 8×f32 (ggml mul_sum_i8_pairs_float via maddubs).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn mul_sum_i8_pairs_float_avx2(
    x: std::arch::x86_64::__m256i,
    y: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    let ones = _mm256_set1_epi16(1);
    let summed = _mm256_madd_epi16(ones, dot);
    _mm256_cvtepi32_ps(summed)
}

/// Same via AVX-512 VNNI `vpdpbusd` (AVX512VL + AVX512-VNNI CPUs).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
pub(super) unsafe fn mul_sum_i8_pairs_float_vnni(
    x: std::arch::x86_64::__m256i,
    y: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let summed = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
    _mm256_cvtepi32_ps(summed)
}

/// F16C: convert weight f16 scale and multiply by already-f32 activation scale.
/// Caller must be under `target_feature(enable = "...f16c...")`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(super) unsafe fn f16_act_scale_product(d_w_f16: u16, d_act: f32) -> f32 {
    use std::arch::x86_64::*;
    let packed = _mm_cvtsi32_si128(d_w_f16 as i32);
    let dw = _mm_cvtss_f32(_mm_cvtph_ps(packed));
    dw * d_act
}

/// AVX2: SIMD unpack + maddubs; dual float acc + F16C scales; OOB once.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn vec_dot_q5_0_q8_0_avx2(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let end = (start_block + nb) * BLOCK_Q5_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q5_0 OOB".into()));
    }

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut b = 0usize;
    let w = weight.as_ptr();
    while b + 2 <= nb {
        let base0 = (start_block + b) * BLOCK_Q5_0;
        let base1 = (start_block + b + 1) * BLOCK_Q5_0;
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            y[b].d,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            y[b + 1].d,
        ));
        let qx0 = unpack_q5_0_block(w.add(base0 + 6), w.add(base0 + 2));
        let qx1 = unpack_q5_0_block(w.add(base1 + 6), w.add(base1 + 2));
        let qy0 = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let qy1 = _mm256_loadu_si256(y[b + 1].qs.as_ptr() as *const __m256i);
        let q0 = mul_sum_i8_pairs_float_avx2(qx0, qy0);
        let q1 = mul_sum_i8_pairs_float_avx2(qx1, qy1);
        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);
        b += 2;
    }
    while b < nb {
        let base = (start_block + b) * BLOCK_Q5_0;
        let d = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base], weight[base + 1]]),
            y[b].d,
        ));
        let qx = unpack_q5_0_block(w.add(base + 6), w.add(base + 2));
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float_avx2(qx, qy);
        acc0 = _mm256_fmadd_ps(d, q, acc0);
        b += 1;
    }
    Ok(hsum_float_8(_mm256_add_ps(acc0, acc1)))
}

/// AVX-512 VNNI: dual float acc + F16C + hoisted OOB (ggml-class polish).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vec_dot_q5_0_q8_0_avx512_vnni(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let end = (start_block + nb) * BLOCK_Q5_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q5_0 OOB".into()));
    }

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut b = 0usize;
    let w = weight.as_ptr();
    // Dual-acc pairs for ILP on fat rows (k=3136 → nb=98).
    while b + 2 <= nb {
        let base0 = (start_block + b) * BLOCK_Q5_0;
        let base1 = (start_block + b + 1) * BLOCK_Q5_0;
        if b + 4 <= nb {
            let pref = w.add((start_block + b + 2) * BLOCK_Q5_0);
            _mm_prefetch(pref as *const i8, _MM_HINT_T0);
            _mm_prefetch(pref.add(BLOCK_Q5_0) as *const i8, _MM_HINT_T0);
        }
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            y[b].d,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            y[b + 1].d,
        ));
        let qx0 = unpack_q5_0_block(w.add(base0 + 6), w.add(base0 + 2));
        let qx1 = unpack_q5_0_block(w.add(base1 + 6), w.add(base1 + 2));
        let qy0 = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let qy1 = _mm256_loadu_si256(y[b + 1].qs.as_ptr() as *const __m256i);
        let q0 = mul_sum_i8_pairs_float_vnni(qx0, qy0);
        let q1 = mul_sum_i8_pairs_float_vnni(qx1, qy1);
        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);
        b += 2;
    }
    while b < nb {
        let base = (start_block + b) * BLOCK_Q5_0;
        let d = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base], weight[base + 1]]),
            y[b].d,
        ));
        let qx = unpack_q5_0_block(w.add(base + 6), w.add(base + 2));
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float_vnni(qx, qy);
        acc0 = _mm256_fmadd_ps(d, q, acc0);
        b += 1;
    }
    Ok(hsum_float_8(_mm256_add_ps(acc0, acc1)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn vec_dot_q5_0_q8_0_2rows_avx2(
    weight: &[u8],
    start_block0: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32)> {
    use std::arch::x86_64::*;
    let start_block1 = start_block0 + nb;
    let end = (start_block1 + nb) * BLOCK_Q5_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q5_0 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let w = weight.as_ptr();
    for b in 0..nb {
        let base0 = (start_block0 + b) * BLOCK_Q5_0;
        let base1 = (start_block1 + b) * BLOCK_Q5_0;
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            y[b].d,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            y[b].d,
        ));
        let qx0 = unpack_q5_0_block(w.add(base0 + 6), w.add(base0 + 2));
        let qx1 = unpack_q5_0_block(w.add(base1 + 6), w.add(base1 + 2));
        let q0 = mul_sum_i8_pairs_float_avx2(qx0, qy);
        let q1 = mul_sum_i8_pairs_float_avx2(qx1, qy);
        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);
    }
    Ok((hsum_float_8(acc0), hsum_float_8(acc1)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vec_dot_q5_0_q8_0_2rows_vnni(
    weight: &[u8],
    start_block0: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32)> {
    use std::arch::x86_64::*;
    let start_block1 = start_block0 + nb;
    let end = (start_block1 + nb) * BLOCK_Q5_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q5_0 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let w = weight.as_ptr();
    for b in 0..nb {
        let base0 = (start_block0 + b) * BLOCK_Q5_0;
        let base1 = (start_block1 + b) * BLOCK_Q5_0;
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            y[b].d,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            y[b].d,
        ));
        let qx0 = unpack_q5_0_block(w.add(base0 + 6), w.add(base0 + 2));
        let qx1 = unpack_q5_0_block(w.add(base1 + 6), w.add(base1 + 2));
        let q0 = mul_sum_i8_pairs_float_vnni(qx0, qy);
        let q1 = mul_sum_i8_pairs_float_vnni(qx1, qy);
        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);
    }
    Ok((hsum_float_8(acc0), hsum_float_8(acc1)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn vec_dot_q5_0_q8_0_4rows_avx2(
    weight: &[u8],
    start_block0: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32, f32, f32)> {
    use std::arch::x86_64::*;
    let s0 = start_block0;
    let s1 = start_block0 + nb;
    let s2 = start_block0 + 2 * nb;
    let s3 = start_block0 + 3 * nb;
    let end = (s3 + nb) * BLOCK_Q5_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q5_0 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let w = weight.as_ptr();
    for b in 0..nb {
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let yd = y[b].d;
        let base0 = (s0 + b) * BLOCK_Q5_0;
        let base1 = (s1 + b) * BLOCK_Q5_0;
        let base2 = (s2 + b) * BLOCK_Q5_0;
        let base3 = (s3 + b) * BLOCK_Q5_0;
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            yd,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            yd,
        ));
        let d2 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base2], weight[base2 + 1]]),
            yd,
        ));
        let d3 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base3], weight[base3 + 1]]),
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
unsafe fn vec_dot_q5_0_q8_0_4rows_vnni(
    weight: &[u8],
    start_block0: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32, f32, f32)> {
    use std::arch::x86_64::*;
    let s0 = start_block0;
    let s1 = start_block0 + nb;
    let s2 = start_block0 + 2 * nb;
    let s3 = start_block0 + 3 * nb;
    let end = (s3 + nb) * BLOCK_Q5_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q5_0 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let w = weight.as_ptr();
    for b in 0..nb {
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let yd = y[b].d;
        let base0 = (s0 + b) * BLOCK_Q5_0;
        let base1 = (s1 + b) * BLOCK_Q5_0;
        let base2 = (s2 + b) * BLOCK_Q5_0;
        let base3 = (s3 + b) * BLOCK_Q5_0;
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            yd,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            yd,
        ));
        let d2 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base2], weight[base2 + 1]]),
            yd,
        ));
        let d3 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base3], weight[base3 + 1]]),
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn vec_dot_q8_0_q8_0_2rows_avx2(
    weight: &[u8],
    start_block0: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32)> {
    use std::arch::x86_64::*;
    let start_block1 = start_block0 + nb;
    let end = (start_block1 + nb) * BLOCK_Q8_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q8_0 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let w = weight.as_ptr();
    for b in 0..nb {
        let base0 = (start_block0 + b) * BLOCK_Q8_0;
        let base1 = (start_block1 + b) * BLOCK_Q8_0;
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            y[b].d,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            y[b].d,
        ));
        let qx0 = _mm256_loadu_si256(w.add(base0 + 2) as *const __m256i);
        let qx1 = _mm256_loadu_si256(w.add(base1 + 2) as *const __m256i);
        acc0 = _mm256_fmadd_ps(d0, mul_sum_i8_pairs_float_avx2(qx0, qy), acc0);
        acc1 = _mm256_fmadd_ps(d1, mul_sum_i8_pairs_float_avx2(qx1, qy), acc1);
    }
    Ok((hsum_float_8(acc0), hsum_float_8(acc1)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vec_dot_q8_0_q8_0_2rows_vnni(
    weight: &[u8],
    start_block0: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<(f32, f32)> {
    use std::arch::x86_64::*;
    let start_block1 = start_block0 + nb;
    let end = (start_block1 + nb) * BLOCK_Q8_0;
    if end > weight.len() {
        return Err(CoreError::Shape("Q8_0 OOB".into()));
    }
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let w = weight.as_ptr();
    for b in 0..nb {
        let base0 = (start_block0 + b) * BLOCK_Q8_0;
        let base1 = (start_block1 + b) * BLOCK_Q8_0;
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let d0 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base0], weight[base0 + 1]]),
            y[b].d,
        ));
        let d1 = _mm256_set1_ps(f16_act_scale_product(
            u16::from_le_bytes([weight[base1], weight[base1 + 1]]),
            y[b].d,
        ));
        let qx0 = _mm256_loadu_si256(w.add(base0 + 2) as *const __m256i);
        let qx1 = _mm256_loadu_si256(w.add(base1 + 2) as *const __m256i);
        acc0 = _mm256_fmadd_ps(d0, mul_sum_i8_pairs_float_vnni(qx0, qy), acc0);
        acc1 = _mm256_fmadd_ps(d1, mul_sum_i8_pairs_float_vnni(qx1, qy), acc1);
    }
    Ok((hsum_float_8(acc0), hsum_float_8(acc1)))
}

/// Integer Q8_0 × Q8_0 dot for one weight row (LM head / sparse Q8_0 weights).
pub(crate) fn vec_dot_q8_0_q8_0(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    let start_block = elem_offset / QK8_0;
    let nb = k / QK8_0;
    if y.len() < nb {
        return Err(CoreError::Shape("Q8_0 act too short".into()));
    }
    let feat = cpu_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if feat.avx512_vnni_avx2_fma() {
            // SAFETY: AVX-512 F/BW/VL/VNNI, AVX2 and FMA, everything the kernel enables, were
            // detected at run time by `cpu_features()`. It bounds-checks `weight` and `y`.
            return unsafe { vec_dot_q8_0_q8_0_avx512_vnni(weight, start_block, nb, y) };
        }
        if feat.avx2_fma() {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`. It bounds-checks `weight` and `y`.
            return unsafe { vec_dot_q8_0_q8_0_avx2(weight, start_block, nb, y) };
        }
    }
    let _ = feat;
    vec_dot_q8_0_q8_0_scalar(weight, start_block, nb, y)
}

fn vec_dot_q8_0_q8_0_scalar(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q8_0;
        if base + BLOCK_Q8_0 > weight.len() {
            return Err(CoreError::Shape("Q8_0 OOB".into()));
        }
        let d0 = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
        let yb = &y[b];
        let d1 = yb.d;
        let mut sumi = 0i32;
        for j in 0..QK8_0 {
            sumi += (weight[base + 2 + j] as i8 as i32) * (yb.qs[j] as i32);
        }
        sum += d0 * d1 * sumi as f32;
    }
    Ok(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q8_0_q8_0_avx2(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut b = 0usize;
    while b + 2 <= nb {
        for bi in b..b + 2 {
            let base = (start_block + bi) * BLOCK_Q8_0;
            if base + BLOCK_Q8_0 > weight.len() {
                return Err(CoreError::Shape("Q8_0 OOB".into()));
            }
            let d0 = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
            let d1 = y[bi].d;
            let d = _mm256_set1_ps(d0 * d1);
            let qx = _mm256_loadu_si256(weight.as_ptr().add(base + 2) as *const __m256i);
            let qy = _mm256_loadu_si256(y[bi].qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float_avx2(qx, qy);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        b += 2;
    }
    while b < nb {
        let base = (start_block + b) * BLOCK_Q8_0;
        if base + BLOCK_Q8_0 > weight.len() {
            return Err(CoreError::Shape("Q8_0 OOB".into()));
        }
        let d0 = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
        let d1 = y[b].d;
        let d = _mm256_set1_ps(d0 * d1);
        let qx = _mm256_loadu_si256(weight.as_ptr().add(base + 2) as *const __m256i);
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float_avx2(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
        b += 1;
    }
    Ok(hsum_float_8(acc))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vec_dot_q8_0_q8_0_avx512_vnni(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ80],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut b = 0usize;
    while b + 4 <= nb {
        for bi in b..b + 4 {
            let base = (start_block + bi) * BLOCK_Q8_0;
            if base + BLOCK_Q8_0 > weight.len() {
                return Err(CoreError::Shape("Q8_0 OOB".into()));
            }
            let d0 = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
            let d1 = y[bi].d;
            let d = _mm256_set1_ps(d0 * d1);
            let qx = _mm256_loadu_si256(weight.as_ptr().add(base + 2) as *const __m256i);
            let qy = _mm256_loadu_si256(y[bi].qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float_vnni(qx, qy);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        b += 4;
    }
    while b < nb {
        let base = (start_block + b) * BLOCK_Q8_0;
        if base + BLOCK_Q8_0 > weight.len() {
            return Err(CoreError::Shape("Q8_0 OOB".into()));
        }
        let d0 = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
        let d1 = y[b].d;
        let d = _mm256_set1_ps(d0 * d1);
        let qx = _mm256_loadu_si256(weight.as_ptr().add(base + 2) as *const __m256i);
        let qy = _mm256_loadu_si256(y[b].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float_vnni(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
        b += 1;
    }
    Ok(hsum_float_8(acc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    #[test]
    fn q8_0_quant_roundtrip_scale() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let mut blocks = [BlockQ80 {
            d: 0.0,
            qs: [0; 32],
        }; 2];
        quantize_row_q8_0(&x, &mut blocks);
        assert!(blocks[0].d != 0.0);
        let d = blocks[0].d;
        let mut max_err = 0.0f32;
        for j in 0..32 {
            let rec = blocks[0].qs[j] as f32 * d;
            max_err = max_err.max((rec - x[j]).abs());
        }
        assert!(max_err < 0.15, "max_err={max_err}");
    }

    #[test]
    fn q5_int_dot_matches_f32_structure() {
        // Synthetic Q5_0 block: d=1.0, qh=0, qs=0 → all q = -16
        let mut weight = vec![0u8; BLOCK_Q5_0];
        weight[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let x = vec![1.0f32; 32];
        let mut y = [BlockQ80 {
            d: 0.0,
            qs: [0; 32],
        }; 1];
        quantize_row_q8_0(&x, &mut y);
        let dot = vec_dot_q5_0_q8_0(&weight, 0, 32, &y).unwrap();
        assert!(dot < -100.0, "dot={dot}");
    }

    #[test]
    fn q5_int_dot_matches_scalar_reference() {
        // Build a non-trivial Q5_0 row (2 blocks) and compare SIMD vs scalar.
        let mut weight = vec![0u8; BLOCK_Q5_0 * 2];
        for b in 0..2 {
            let base = b * BLOCK_Q5_0;
            weight[base..base + 2]
                .copy_from_slice(&f32_to_f16(0.5 + b as f32 * 0.25).to_le_bytes());
            // qh pattern
            weight[base + 2..base + 6].copy_from_slice(&(0xA5A5_5A5Au32).to_le_bytes());
            for j in 0..16 {
                weight[base + 6 + j] = ((j * 3 + b * 7) & 0xFF) as u8;
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.07 - 2.0).collect();
        let mut y = [BlockQ80 {
            d: 0.0,
            qs: [0; 32],
        }; 2];
        quantize_row_q8_0(&x, &mut y);
        let simd = vec_dot_q5_0_q8_0(&weight, 0, 64, &y).unwrap();
        let scalar = vec_dot_q5_0_q8_0_scalar(&weight, 0, 2, &y).unwrap();
        let rel = (simd - scalar).abs() / scalar.abs().max(1e-3);
        assert!(rel < 1e-5, "simd={simd} scalar={scalar} rel={rel}");
    }

    #[test]
    fn q8_int_dot_matches_scalar_reference() {
        let mut weight = vec![0u8; BLOCK_Q8_0 * 2];
        for b in 0..2 {
            let base = b * BLOCK_Q8_0;
            weight[base..base + 2].copy_from_slice(&f32_to_f16(0.1).to_le_bytes());
            for j in 0..32 {
                weight[base + 2 + j] = ((j as i8).wrapping_mul(3).wrapping_add(b as i8)) as u8;
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.5).collect();
        let mut y = [BlockQ80 {
            d: 0.0,
            qs: [0; 32],
        }; 2];
        quantize_row_q8_0(&x, &mut y);
        let simd = vec_dot_q8_0_q8_0(&weight, 0, 64, &y).unwrap();
        let scalar = vec_dot_q8_0_q8_0_scalar(&weight, 0, 2, &y).unwrap();
        let rel = (simd - scalar).abs() / scalar.abs().max(1e-3);
        assert!(rel < 1e-5, "simd={simd} scalar={scalar} rel={rel}");
    }

    #[test]
    fn q5_nrows_matches_single_row() {
        let mut weight = vec![0u8; BLOCK_Q5_0 * 8]; // 4 rows × 2 blocks
        for r in 0..4 {
            for b in 0..2 {
                let base = (r * 2 + b) * BLOCK_Q5_0;
                weight[base..base + 2].copy_from_slice(
                    &f32_to_f16(0.4 + r as f32 * 0.1 + b as f32 * 0.05).to_le_bytes(),
                );
                weight[base + 2..base + 6]
                    .copy_from_slice(&(0xA5A5_5A5Au32.wrapping_add(r as u32)).to_le_bytes());
                for j in 0..16 {
                    weight[base + 6 + j] = ((j * 3 + r * 11 + b * 7) & 0xFF) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.07 - 2.0).collect();
        let mut y = [BlockQ80 {
            d: 0.0,
            qs: [0; 32],
        }; 2];
        quantize_row_q8_0(&x, &mut y);
        let mut out4 = [0.0f32; 4];
        vec_dot_q5_0_q8_0_nrows(&weight, 0, 64, &y, &mut out4).unwrap();
        for r in 0..4 {
            let single = vec_dot_q5_0_q8_0(&weight, r * 64, 64, &y).unwrap();
            let rel = (out4[r] - single).abs() / single.abs().max(1e-3);
            assert!(
                rel < 1e-5,
                "row {r}: nrows={} single={} rel={rel}",
                out4[r],
                single
            );
        }
    }

    #[test]
    fn q8_nrows_matches_single_row() {
        let mut weight = vec![0u8; BLOCK_Q8_0 * 4]; // 2 rows × 2 blocks
        for r in 0..2 {
            for b in 0..2 {
                let base = (r * 2 + b) * BLOCK_Q8_0;
                weight[base..base + 2]
                    .copy_from_slice(&f32_to_f16(0.1 + r as f32 * 0.05).to_le_bytes());
                for j in 0..32 {
                    weight[base + 2 + j] =
                        ((j as i8).wrapping_mul(3).wrapping_add((r + b) as i8)) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.5).collect();
        let mut y = [BlockQ80 {
            d: 0.0,
            qs: [0; 32],
        }; 2];
        quantize_row_q8_0(&x, &mut y);
        let mut out2 = [0.0f32; 2];
        vec_dot_q8_0_q8_0_nrows(&weight, 0, 64, &y, &mut out2).unwrap();
        for r in 0..2 {
            let single = vec_dot_q8_0_q8_0(&weight, r * 64, 64, &y).unwrap();
            let rel = (out2[r] - single).abs() / single.abs().max(1e-3);
            assert!(
                rel < 1e-5,
                "row {r}: nrows={} single={} rel={rel}",
                out2[r],
                single
            );
        }
    }
}
