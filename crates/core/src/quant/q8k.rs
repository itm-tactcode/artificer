//! Q8_K activation quantization + Q4_K × Q8_K integer vec_dot (ggml-compatible).
//!
//! Quantizing the shared activation once per matvec turns every weight-row dot into
//! an int8×nibble reduction plus scale/min correction — the main CPU lever llama.cpp uses.

use super::blocks::{get_scale_min_k4, BLOCK_Q4_K, BLOCK_Q8_K, QK_K};
use super::half::f16_to_f32;
use super::simd::cpu_features;
use crate::error::{CoreError, Result};

/// One ggml `block_q8_K` (292 bytes): d + qs[256] + bsums[16].
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ8K>() == BLOCK_Q8_K);

/// Quantize `x` (len multiple of 256) into ggml Q8_K blocks.
pub(crate) fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) {
    debug_assert_eq!(x.len() % QK_K, 0);
    debug_assert_eq!(out.len(), x.len() / QK_K);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpu_features().avx2_fma() {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`.
            unsafe { quantize_row_q8_k_avx2(x, out) };
            return;
        }
    }
    quantize_row_q8_k_scalar(x, out);
}

fn quantize_row_q8_k_scalar(x: &[f32], out: &mut [BlockQ8K]) {
    for (bi, block) in out.iter_mut().enumerate() {
        let xb = &x[bi * QK_K..(bi + 1) * QK_K];
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in xb {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            block.bsums.fill(0);
            continue;
        }
        // ggml quantize_row_q8_K_ref: iscale = -127/max; d = 1/iscale
        let iscale = -127.0 / max;
        for j in 0..QK_K {
            let mut v = nearest_int(iscale * xb[j]);
            if v > 127 {
                v = 127;
            }
            if v < -127 {
                v = -127;
            }
            block.qs[j] = v as i8;
        }
        for j in 0..(QK_K / 16) {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += block.qs[j * 16 + ii] as i32;
            }
            block.bsums[j] = sum as i16;
        }
        block.d = 1.0 / iscale;
    }
}

/// AVX2 Q8_K act quant: SIMD amax + pack + bsums (paid once per Q4/Q6 matvec).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_q8_k_avx2(x: &[f32], out: &mut [BlockQ8K]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let sign = _mm256_set1_ps(-0.0);
    for (bi, block) in out.iter_mut().enumerate() {
        let xb = x.as_ptr().add(bi * QK_K);
        // Find signed value with largest abs (ggml q8_K).
        let mut max_abs = 0.0f32;
        let mut max_val = 0.0f32;
        let mut j = 0usize;
        while j < QK_K {
            let v = _mm256_loadu_ps(xb.add(j));
            let ax = _mm256_andnot_ps(sign, v);
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), ax);
            let mut vals = [0.0f32; 8];
            _mm256_storeu_ps(vals.as_mut_ptr(), v);
            for lane in 0..8 {
                if tmp[lane] > max_abs {
                    max_abs = tmp[lane];
                    max_val = vals[lane];
                }
            }
            j += 8;
        }
        if max_abs == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            block.bsums.fill(0);
            continue;
        }
        let iscale = -127.0 / max_val;
        block.d = 1.0 / iscale;
        let mul = _mm256_set1_ps(iscale);
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let mut off = 0usize;
        while off < QK_K {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(xb.add(off)), mul);
            let v1 = _mm256_mul_ps(_mm256_loadu_ps(xb.add(off + 8)), mul);
            let v2 = _mm256_mul_ps(_mm256_loadu_ps(xb.add(off + 16)), mul);
            let v3 = _mm256_mul_ps(_mm256_loadu_ps(xb.add(off + 24)), mul);
            let r0 = _mm256_round_ps(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let r1 = _mm256_round_ps(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let r2 = _mm256_round_ps(v2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let r3 = _mm256_round_ps(v3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            let i0 = _mm256_cvtps_epi32(r0);
            let i1 = _mm256_cvtps_epi32(r1);
            let i2 = _mm256_cvtps_epi32(r2);
            let i3 = _mm256_cvtps_epi32(r3);
            let p01 = _mm256_packs_epi32(i0, i1);
            let p23 = _mm256_packs_epi32(i2, i3);
            let packed = _mm256_packs_epi16(p01, p23);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            // Clamp to [-127, 127] (packs can briefly hit -128).
            let lo = _mm256_set1_epi8(-127i8 as u8 as i8);
            let hi = _mm256_set1_epi8(127);
            let clamped = _mm256_max_epi8(_mm256_min_epi8(ordered, hi), lo);
            _mm256_storeu_si256(block.qs.as_mut_ptr().add(off) as *mut __m256i, clamped);
            off += 32;
        }
        // bsums: 16 groups of 16 i8 → i16
        for g in 0..(QK_K / 16) {
            let v = _mm_loadu_si128(block.qs.as_ptr().add(g * 16) as *const __m128i);
            let lo = _mm256_cvtepi8_epi16(v);
            let mut tmp = [0i16; 16];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, lo);
            let mut sum = 0i32;
            for t in tmp.iter() {
                sum += *t as i32;
            }
            block.bsums[g] = sum as i16;
        }
    }
}

#[inline]
fn nearest_int(f: f32) -> i32 {
    // ggml nearest_int for |f| <= ~4e6
    let i = (f + 12582912.0).to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Allocate + quantize activations for a Q4_K matvec (`k` multiple of 256).
#[allow(dead_code)]
pub(crate) fn quantize_activations_q8_k(x: &[f32]) -> Result<Vec<BlockQ8K>> {
    if !x.len().is_multiple_of(QK_K) {
        return Err(CoreError::Shape(format!(
            "Q8_K quant: k={} not multiple of {QK_K}",
            x.len()
        )));
    }
    let nb = x.len() / QK_K;
    let mut out = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        nb
    ];
    quantize_row_q8_k(x, &mut out);
    Ok(out)
}

/// Integer Q4_K × Q8_K dot for one weight row (`elem_offset` in elements, `k` multiple of 256).
pub(crate) fn vec_dot_q4_k_q8_k(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    let start_block = elem_offset / QK_K;
    let nb = k / QK_K;
    if y.len() < nb {
        return Err(CoreError::Shape("Q8_K act too short".into()));
    }
    let feat = cpu_features();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if feat.avx512_vnni_avx2_fma() {
            // SAFETY: AVX-512 F/BW/VL/VNNI, AVX2 and FMA, everything the kernel enables, were
            // detected at run time by `cpu_features()`. It bounds-checks `weight` and `y`.
            return unsafe { vec_dot_q4_k_q8_k_avx512_vnni(weight, start_block, nb, y) };
        }
        if feat.avx2_fma() {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`. It bounds-checks `weight` and `y`.
            return unsafe { vec_dot_q4_k_q8_k_avx2(weight, start_block, nb, y) };
        }
    }
    let _ = feat;
    vec_dot_q4_k_q8_k_scalar(weight, start_block, nb, y)
}

fn vec_dot_q4_k_q8_k_scalar(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_K;
        if base + BLOCK_Q4_K > weight.len() {
            return Err(CoreError::Shape("Q4_K OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([weight[base + 2], weight[base + 3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&weight[base + 4..base + 16]);
        let q = &weight[base + 16..base + 144];
        let yb = &y[b];
        let d8 = yb.d;

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut x_off = 0usize;
        let mut bs_off = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc, m) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for l in 0..32 {
                let q4 = q[q_off + l];
                sumi1 += (q4 & 0x0F) as i32 * yb.qs[x_off + l] as i32;
                sumi2 += (q4 >> 4) as i32 * yb.qs[x_off + 32 + l] as i32;
            }
            let bsum1 = yb.bsums[bs_off] as i32 + yb.bsums[bs_off + 1] as i32;
            let bsum2 = yb.bsums[bs_off + 2] as i32 + yb.bsums[bs_off + 3] as i32;
            sum += d8 * (d1 * sumi1 as f32 - m1 * bsum1 as f32);
            sum += d8 * (d2 * sumi2 as f32 - m2 * bsum2 as f32);

            q_off += 32;
            x_off += 64;
            bs_off += 4;
            is += 2;
        }
    }
    Ok(sum)
}

/// ggml-style scale shuffle: replicate scale index `i` across 32 i16 lanes as bytes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    static K_SHUFFLE_GGML: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256(K_SHUFFLE_GGML.as_ptr().add(i * 32) as *const __m256i)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_float_8(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut res = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// Unpack Q4_K packed scales into 8 scales + 8 mins (ggml utmp layout).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn unpack_q4k_scales_mins(scales12: &[u8]) -> [u32; 4] {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let mut utmp = [0u32; 4];
    // scales12 is 12 bytes
    utmp[0] = u32::from_le_bytes([scales12[0], scales12[1], scales12[2], scales12[3]]);
    utmp[1] = u32::from_le_bytes([scales12[4], scales12[5], scales12[6], scales12[7]]);
    utmp[2] = u32::from_le_bytes([scales12[8], scales12[9], scales12[10], scales12[11]]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    utmp
}

/// ggml-style Q4_K×Q8_K: integer scale apply + YMM accumulate (no per-32 hsum).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_k_q8_k_avx2(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_K;
        if base + BLOCK_Q4_K > weight.len() {
            return Err(CoreError::Shape("Q4_K OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]]));
        let dmin = -f16_to_f32(u16::from_le_bytes([weight[base + 2], weight[base + 3]]));
        let d = d * y[b].d;
        let dmin = dmin * y[b].d;

        let utmp = unpack_q4k_scales_mins(&weight[base + 4..base + 16]);
        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));
        let q8sums = _mm256_loadu_si256(y[b].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = _mm256_set_m128i(sc128, sc128);

        let mut q4 = weight.as_ptr().add(base + 16);
        let mut q8 = y[b].qs.as_ptr();
        let mut sumi = _mm256_setzero_si256();

        for j in 0..(QK_K / 64) {
            let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8l = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    let mut acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    Ok(hsum_float_8(acc) + _mm_cvtss_f32(acc_m))
}

/// Same ggml accumulate structure; use VNNI `vpdpbusd` for the u8×i8 step when available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn vec_dot_q4_k_q8_k_avx512_vnni(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q4_K;
        if base + BLOCK_Q4_K > weight.len() {
            return Err(CoreError::Shape("Q4_K OOB".into()));
        }
        let d = f16_to_f32(u16::from_le_bytes([weight[base], weight[base + 1]])) * y[b].d;
        let dmin = -f16_to_f32(u16::from_le_bytes([weight[base + 2], weight[base + 3]])) * y[b].d;

        let utmp = unpack_q4k_scales_mins(&weight[base + 4..base + 16]);
        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));
        let q8sums = _mm256_loadu_si256(y[b].bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = _mm256_set_m128i(sc128, sc128);

        let mut q4 = weight.as_ptr().add(base + 16);
        let mut q8 = y[b].qs.as_ptr();
        let mut sumi = _mm256_setzero_si256();

        for j in 0..(QK_K / 64) {
            let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8l = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            // VNNI: dpbusd gives 8×i32 (groups of 4). Need i16 pairs for scale madd.
            // Fall back to maddubs (same as ggml) — VNNI doesn't directly replace
            // maddubs→madd_epi16(scale) chain without regrouping. Use maddubs.
            let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    let mut acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    Ok(hsum_float_8(acc) + _mm_cvtss_f32(acc_m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::simd::q4k_sum_qx_x_scalar;

    #[test]
    fn q8_k_roundtrip_scale() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let mut blocks = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16],
            };
            1
        ];
        quantize_row_q8_k(&x, &mut blocks);
        let mut err = 0.0f32;
        for i in 0..256 {
            let rec = blocks[0].d * blocks[0].qs[i] as f32;
            err += (rec - x[i]).abs();
        }
        assert!(err / 256.0 < 0.02, "mean abs err {}", err / 256.0);
    }

    #[test]
    fn int_dot_matches_f32_path_structure() {
        // Synthetic Q4_K block: d=1, min=0, all scales=1 for first 2 groups, qs = ascending
        let mut weight = vec![0u8; BLOCK_Q4_K];
        // d = 1.0f16 = 0x3C00, min = 0
        weight[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        weight[2..4].copy_from_slice(&0u16.to_le_bytes());
        // scales: first 4 scale bytes = 1, next 4 min bytes = 0
        for i in 0..4 {
            weight[4 + i] = 1;
            weight[8 + i] = 0;
        }
        // remaining scale packing zeros
        for i in 0..128 {
            weight[16 + i] = ((i % 16) | (((i + 3) % 16) << 4)) as u8;
        }
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let y = quantize_activations_q8_k(&x).unwrap();

        // f32 reference via existing fused formula
        let mut sum_f = 0.0f32;
        let d = 1.0f32;
        let min = 0.0f32;
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&weight[4..16]);
        let mut q = &weight[16..144];
        let mut x_off = 0usize;
        let mut is = 0usize;
        for _ in 0..4 {
            let (sc, m) = get_scale_min_k4(is, &scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            let (sqx, sx) = q4k_sum_qx_x_scalar(q, &x[x_off..x_off + 32], false);
            sum_f += d1 * sqx - m1 * sx;
            let (sqx, sx) = q4k_sum_qx_x_scalar(q, &x[x_off + 32..x_off + 64], true);
            sum_f += d2 * sqx - m2 * sx;
            q = &q[32..];
            x_off += 64;
            is += 2;
        }

        let sum_i = vec_dot_q4_k_q8_k(&weight, 0, 256, &y).unwrap();
        let rel = (sum_i - sum_f).abs() / sum_f.abs().max(1e-3);
        assert!(rel < 0.05, "int={sum_i} f32={sum_f} rel={rel}");
    }
}
