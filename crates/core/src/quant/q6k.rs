//! Q6_K × Q8_K integer vec_dot (ggml-compatible) for mid-size Q6_K projections
//! (Nemotron `ffn_down`). Large Q6 LM heads stay on fused f32 (see matmul gate).

use super::blocks::{BLOCK_Q6_K, QK_K};
use super::half::f16_to_f32;
use super::q8k::BlockQ8K;
use super::simd::{cpu_features, CpuFeatures};
use crate::error::{CoreError, Result};

/// Integer Q6_K × Q8_K dot for one weight row (`elem_offset` in elements, `k` multiple of 256).
pub(crate) fn vec_dot_q6_k_q8_k(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    vec_dot_q6_k_q8_k_with(cpu_features(), weight, elem_offset, k, y)
}

/// `vec_dot_q6_k_q8_k` with the feature set passed in (tests force the scalar fallback).
/// `feat` must not claim features the CPU lacks.
#[inline]
fn vec_dot_q6_k_q8_k_with(
    feat: CpuFeatures,
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if feat.avx2_fma() {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()` (callers of `_with` pass that set or a subset of it).
            return unsafe { vec_dot_q6_k_q8_k_avx2(weight, start_block, nb, y) };
        }
    }
    let _ = feat;
    vec_dot_q6_k_q8_k_scalar(weight, start_block, nb, y)
}

fn vec_dot_q6_k_q8_k_scalar(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q6_K;
        if base + BLOCK_Q6_K > weight.len() {
            return Err(CoreError::Shape("Q6_K OOB".into()));
        }
        let ql = &weight[base..base + 128];
        let qh = &weight[base + 128..base + 192];
        let scales: [i8; 16] = std::array::from_fn(|i| weight[base + 192 + i] as i8);
        let d = f16_to_f32(u16::from_le_bytes([weight[base + 208], weight[base + 209]]));
        let yb = &y[b];
        let d8 = yb.d;
        let qs = &yb.qs;

        let mut sumi = [0i32; 16];
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut x_base = 0usize;
        let mut sc_off = 0usize;
        for _ in 0..(QK_K / 128) {
            for l in 0..32 {
                let is = l / 16;
                let qhl = qh[qh_off + l];
                let ql0 = ql[ql_off + l];
                let ql1 = ql[ql_off + l + 32];
                let q1 = ((ql0 & 0xF) | (((qhl >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql1 & 0xF) | (((qhl >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql0 >> 4) | (((qhl >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql1 >> 4) | (((qhl >> 6) & 3) << 4)) as i32 - 32;
                sumi[sc_off + is] += q1 * qs[x_base + l] as i32;
                sumi[sc_off + is + 2] += q2 * qs[x_base + l + 32] as i32;
                sumi[sc_off + is + 4] += q3 * qs[x_base + l + 64] as i32;
                sumi[sc_off + is + 6] += q4 * qs[x_base + l + 96] as i32;
            }
            x_base += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
        let mut acc = 0i32;
        for g in 0..16 {
            acc += scales[g] as i32 * sumi[g];
        }
        sum += d * d8 * acc as f32;
    }
    Ok(sum)
}

/// ggml-style scale shuffle for Q6_K: replicate scale byte across 16 lanes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn get_scale_shuffle_q6(i: usize) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    static K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128(K_SHUFFLE.as_ptr().add(i * 16) as *const __m128i)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_float_8_q6(x: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut res = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

/// AVX2 ggml leftover: signed SIMD unpack + in-register scale shuffle.
/// No QK_K scratch. Applies per-16 scales via shuffle+madd like llama.cpp,
/// but keeps the proven signed (q−32) path (avoids bsums offset footgun).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q6_k_q8_k_avx2(
    weight: &[u8],
    start_block: usize,
    nb: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    use std::arch::x86_64::*;

    let end = (start_block + nb) * BLOCK_Q6_K;
    if end > weight.len() {
        return Err(CoreError::Shape("Q6_K OOB".into()));
    }

    let m3 = _mm256_set1_epi8(3);
    let m15 = _mm256_set1_epi8(15);
    let m32 = _mm256_set1_epi8(32);
    let mut acc = _mm256_setzero_ps();

    for b in 0..nb {
        let base = (start_block + b) * BLOCK_Q6_K;
        let d = f16_to_f32(u16::from_le_bytes([weight[base + 208], weight[base + 209]])) * y[b].d;

        let mut q4 = weight.as_ptr().add(base);
        let mut qh = weight.as_ptr().add(base + 128);
        let mut q8 = y[b].qs.as_ptr();
        let scales = _mm_loadu_si128(weight.as_ptr().add(base + 192) as *const __m128i);

        let mut sumi = _mm256_setzero_si256();
        let mut is = 0usize;

        for _j in 0..(QK_K / 128) {
            let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
            qh = qh.add(32);

            let q4_0 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(q4bits1, m15),
                    _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4),
                ),
                m32,
            );
            let q4_1 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(q4bits2, m15),
                    _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2),
                ),
                m32,
            );
            let q4_2 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15),
                    _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48)),
                ),
                m32,
            );
            let q4_3 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15),
                    _mm256_srli_epi16(
                        _mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64i8 as u8 as i8)),
                        2,
                    ),
                ),
                m32,
            );

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            // 32 signed×signed → 8×i32, then multiply by per-16 scales via shuffle.
            // Widen to i16 first so we can madd_epi16 with replicated scales.
            let p0 = {
                let ax = _mm256_sign_epi8(q4_0, q4_0);
                let sy = _mm256_sign_epi8(q8_0, q4_0);
                _mm256_maddubs_epi16(ax, sy) // 16×i16
            };
            let p1 = {
                let ax = _mm256_sign_epi8(q4_1, q4_1);
                let sy = _mm256_sign_epi8(q8_1, q4_1);
                _mm256_maddubs_epi16(ax, sy)
            };
            let p2 = {
                let ax = _mm256_sign_epi8(q4_2, q4_2);
                let sy = _mm256_sign_epi8(q8_2, q4_2);
                _mm256_maddubs_epi16(ax, sy)
            };
            let p3 = {
                let ax = _mm256_sign_epi8(q4_3, q4_3);
                let sy = _mm256_sign_epi8(q8_3, q4_3);
                _mm256_maddubs_epi16(ax, sy)
            };

            let scale_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is)));
            let scale_1 =
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 1)));
            let scale_2 =
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 2)));
            let scale_3 =
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 3)));
            is += 4;

            let s0 = _mm256_madd_epi16(scale_0, p0);
            let s1 = _mm256_madd_epi16(scale_1, p1);
            let s2 = _mm256_madd_epi16(scale_2, p2);
            let s3 = _mm256_madd_epi16(scale_3, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(s0, s1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(s2, s3));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    Ok(hsum_float_8_q6(acc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;
    use crate::quant::q8k::quantize_row_q8_k;
    use gguf::TensorType;

    #[test]
    fn q6_int_dot_finite_on_synthetic() {
        let mut weight = vec![0u8; BLOCK_Q6_K];
        for i in 0..16 {
            weight[192 + i] = 1u8;
        }
        weight[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let x = vec![1.0f32; 256];
        let mut y = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16],
            };
            1
        ];
        quantize_row_q8_k(&x, &mut y);
        let dot = vec_dot_q6_k_q8_k(&weight, 0, 256, &y).unwrap();
        assert!(dot.is_finite(), "dot={dot}");
        assert!(dot < -100.0, "dot={dot}");
    }

    #[test]
    fn q6_avx2_matches_scalar() {
        let mut weight = vec![0u8; BLOCK_Q6_K];
        for i in 0..128 {
            weight[i] = (i % 256) as u8;
        }
        for i in 0..64 {
            weight[128 + i] = (i * 3) as u8;
        }
        for i in 0..16 {
            weight[192 + i] = (1 + (i % 7)) as u8;
        }
        weight[208..210].copy_from_slice(&f32_to_f16(0.05).to_le_bytes());
        let x: Vec<f32> = (0..256).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let mut y = [BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        }; 1];
        quantize_row_q8_k(&x, &mut y);
        let scalar = vec_dot_q6_k_q8_k_scalar(&weight, 0, 1, &y).unwrap();
        let simd = vec_dot_q6_k_q8_k(&weight, 0, 256, &y).unwrap();
        let rel = (simd - scalar).abs() / scalar.abs().max(1e-3);
        assert!(rel < 1e-5, "simd={simd} scalar={scalar} rel={rel}");
    }

    #[test]
    fn q6_avx2_without_fma_takes_scalar_path() {
        let mut weight = vec![0u8; BLOCK_Q6_K];
        for (i, w) in weight[..192].iter_mut().enumerate() {
            *w = (i * 7 % 256) as u8;
        }
        for (i, w) in weight[192..208].iter_mut().enumerate() {
            *w = (1 + (i % 5)) as u8;
        }
        weight[208..210].copy_from_slice(&f32_to_f16(0.03).to_le_bytes());
        let x: Vec<f32> = (0..256).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
        let mut y = [BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        }; 1];
        quantize_row_q8_k(&x, &mut y);
        // AVX2 without FMA (or nothing at all) must not reach the avx2+fma kernel.
        let avx2_only = CpuFeatures {
            fma: false,
            f16c: false,
            ..cpu_features()
        };
        let scalar = vec_dot_q6_k_q8_k_scalar(&weight, 0, 1, &y).unwrap();
        let got = vec_dot_q6_k_q8_k_with(avx2_only, &weight, 0, 256, &y).unwrap();
        assert_eq!(got.to_bits(), scalar.to_bits(), "got={got} scalar={scalar}");
    }

    #[test]
    fn q6_int_matmul_close_to_fused_f32() {
        let mut weight = vec![0u8; BLOCK_Q6_K];
        for i in 0..128 {
            weight[i] = (i % 256) as u8;
        }
        for i in 0..64 {
            weight[128 + i] = (i * 3) as u8;
        }
        for i in 0..16 {
            weight[192 + i] = (1 + (i % 7)) as u8;
        }
        weight[208..210].copy_from_slice(&f32_to_f16(0.05).to_le_bytes());
        let x: Vec<f32> = (0..256).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let mut y = [BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        }; 1];
        quantize_row_q8_k(&x, &mut y);
        let got = vec_dot_q6_k_q8_k(&weight, 0, 256, &y).unwrap();
        let mut row = vec![0.0f32; 256];
        crate::quant::dequantize(&weight, TensorType::Q6_K, 0, 256, &mut row).unwrap();
        let mut ref_sum = 0.0f32;
        for i in 0..256 {
            ref_sum += row[i] * x[i];
        }
        let err = (got - ref_sum).abs();
        let scale = ref_sum.abs().max(1.0);
        assert!(
            err / scale < 0.05,
            "int={} ref={} rel_err={}",
            got,
            ref_sum,
            err / scale
        );
    }
}
