//! Q4_K ×8 nibble-stripe weight layout for vendor CPU_REPACK gemv.
//!
//! Packs canonical GGUF Q4_K rows into ggml `block_q4_Kx8` (1152 B) so we can
//! call Ollama zen4 `ggml_gemv_q4_K_8x8_q8_K` via FFI. Distinct from the Q4_K vec_dot
//! FFI and from a native AVX2 gemv (which showed no end-to-end gain).
//!
//! Pattern adapted from ggml-org/llama.cpp `ggml/src/ggml-cpu/repack.cpp` (MIT).
//! Env: `ARTIFICER_CPU_REPACK=q4k` (default off). Composes with Q5_0×4.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::blocks::{BLOCK_Q4_K, QK_K};
use crate::error::{CoreError, Result};

/// 8×f16 d + 8×f16 dmin + 96 scales + 1024 qs.
pub(crate) const Q4_KX8_BYTES: usize = 1152;
const NCOLS: usize = 8;

type CacheKey = (usize, usize, usize, usize);

fn cache() -> &'static Mutex<HashMap<CacheKey, Arc<Vec<u8>>>> {
    static C: OnceLock<Mutex<HashMap<CacheKey, Arc<Vec<u8>>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pack_scales_12(s: &[u8; 8], m: &[u8; 8], dst: &mut [u8]) {
    dst[0] = (s[0] & 63) + ((s[4] & 48) << 2);
    dst[1] = (s[1] & 63) + ((s[5] & 48) << 2);
    dst[2] = (s[2] & 63) + ((s[6] & 48) << 2);
    dst[3] = (s[3] & 63) + ((s[7] & 48) << 2);
    dst[4] = (m[0] & 63) + ((m[4] & 48) << 2);
    dst[5] = (m[1] & 63) + ((m[5] & 48) << 2);
    dst[6] = (m[2] & 63) + ((m[6] & 48) << 2);
    dst[7] = (m[3] & 63) + ((m[7] & 48) << 2);
    dst[8] = (s[4] & 15) + ((m[4] & 15) << 4);
    dst[9] = (s[5] & 15) + ((m[5] & 15) << 4);
    dst[10] = (s[6] & 15) + ((m[6] & 15) << 4);
    dst[11] = (s[7] & 15) + ((m[7] & 15) << 4);
}

/// Pack eight canonical Q4_K super-blocks (same K index) into one `block_q4_Kx8`.
fn make_block_q4_kx8(rows: [&[u8]; 8], dst: &mut [u8]) {
    debug_assert!(dst.len() >= Q4_KX8_BYTES);
    for (i, row) in rows.iter().enumerate() {
        dst[i * 2] = row[0];
        dst[i * 2 + 1] = row[1];
        dst[16 + i * 2] = row[2];
        dst[16 + i * 2 + 1] = row[3];
    }
    // 8-byte qs stripes (x86 interleave). 128 B qs × 8 rows.
    let qs_out = &mut dst[128..128 + 1024];
    for i in 0..(1024 / 8) {
        let src_id = i % 8;
        let src_offset = (i / 8) * 8;
        let dst_offset = i * 8;
        let src_qs = &rows[src_id][16..144];
        qs_out[dst_offset..dst_offset + 8].copy_from_slice(&src_qs[src_offset..src_offset + 8]);
    }
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    for i in 0..4 {
        for j in 0..8 {
            let sc = &rows[j][4..16];
            s[j] = sc[i] & 63;
            m[j] = sc[i + 4] & 63;
        }
        pack_scales_12(&s, &m, &mut dst[32 + i * 12..32 + i * 12 + 12]);
    }
    for i in 0..4 {
        for j in 0..8 {
            let sc = &rows[j][4..16];
            s[j] = ((sc[i] & 192) >> 2) | (sc[i + 8] & 15);
            m[j] = ((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4);
        }
        pack_scales_12(&s, &m, &mut dst[32 + 48 + i * 12..32 + 48 + i * 12 + 12]);
    }
}

pub(crate) fn repack_q4_k_x8(src: &[u8], k: usize, n8: usize) -> Result<Vec<u8>> {
    if !k.is_multiple_of(QK_K) || !n8.is_multiple_of(NCOLS) {
        return Err(CoreError::Shape("Q4_Kx8: k%256!=0 or n8%8!=0".into()));
    }
    let nb = k / QK_K;
    let need = n8 * nb * BLOCK_Q4_K;
    if src.len() < need {
        return Err(CoreError::Shape(format!(
            "Q4_Kx8 repack OOB: src={} need={need}",
            src.len()
        )));
    }
    let n_groups = n8 / NCOLS;
    let mut out = vec![0u8; n_groups * nb * Q4_KX8_BYTES];
    for g in 0..n_groups {
        for b in 0..nb {
            let mut rows: [&[u8]; 8] = [&[]; 8];
            for r in 0..8 {
                let src_row = g * 8 + r;
                let off = (src_row * nb + b) * BLOCK_Q4_K;
                rows[r] = &src[off..off + BLOCK_Q4_K];
            }
            let dst_off = (g * nb + b) * Q4_KX8_BYTES;
            make_block_q4_kx8(rows, &mut out[dst_off..dst_off + Q4_KX8_BYTES]);
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

/// Cached nibble-stripe repack for the leading `n & !7` rows.
pub(crate) fn q4_k_x8_cached(weight: &[u8], k: usize, n: usize) -> Option<Arc<Vec<u8>>> {
    if n < 8 || k < QK_K || !k.is_multiple_of(QK_K) {
        return None;
    }
    let n8 = n & !7usize;
    let need = n8 * (k / QK_K) * BLOCK_Q4_K;
    if weight.len() < need {
        return None;
    }
    let key: CacheKey = (weight.as_ptr() as usize, weight.len(), k, n8);
    {
        let guard = cache().lock().ok()?;
        if let Some(v) = guard.get(&key) {
            return Some(Arc::clone(v));
        }
    }
    let packed = repack_q4_k_x8(weight, k, n8).ok()?;
    let arc = Arc::new(packed);
    if let Ok(mut guard) = cache().lock() {
        guard.insert(key, Arc::clone(&arc));
    }
    Some(arc)
}

/// Dot `n_rows` consecutive output rows via vendor `ggml_gemv_q4_K_8x8_q8_K`.
/// `row_base` and `n_rows` must be multiples of 8; packed must cover those groups.
pub(crate) fn gemv_q4_kx8_q8_k_nrows(
    packed: &[u8],
    row_base: usize,
    k: usize,
    y: &[super::q8k::BlockQ8K],
    out: &mut [f32],
) -> Result<()> {
    super::ggml_ffi::gemv_q4_k_8x8_q8_k_ggml(packed, row_base, k, y, out)
}

#[cfg(test)]
mod tests {
    use super::super::q8k::{quantize_row_q8_k, vec_dot_q4_k_q8_k, BlockQ8K};
    use super::*;

    fn fill_row(row: &mut [u8], seed: u32) {
        let mut s = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        row[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        let dmin: u16 = 0x3800;
        row[2..4].copy_from_slice(&dmin.to_le_bytes());
        for b in row.iter_mut().skip(4) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
    }

    #[test]
    fn q4_kx8_vendor_gemv_matches_row_major() {
        // Only runs meaningfully when ARTIFICER_CPU_REPACK is set and libs load.
        if !super::super::ggml_ffi::cpu_repack_q4k_active() {
            return;
        }
        let k = 512;
        let n = 16;
        let nb = k / QK_K;
        let mut weight = vec![0u8; n * nb * BLOCK_Q4_K];
        for r in 0..n {
            for b in 0..nb {
                let off = (r * nb + b) * BLOCK_Q4_K;
                fill_row(
                    &mut weight[off..off + BLOCK_Q4_K],
                    (r * 17 + b * 3) as u32 + 1,
                );
            }
        }
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.4)
            .collect();
        let mut y = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16],
            };
            nb
        ];
        quantize_row_q8_k(&x, &mut y);
        let mut expect = vec![0f32; n];
        for r in 0..n {
            expect[r] = vec_dot_q4_k_q8_k(&weight, r * k, k, &y).unwrap();
        }
        let packed = repack_q4_k_x8(&weight, k, n).unwrap();
        let mut got = vec![0f32; n];
        gemv_q4_kx8_q8_k_nrows(&packed, 0, k, &y, &mut got).unwrap();
        for r in 0..n {
            let denom = expect[r].abs().max(1e-3);
            let rel = (got[r] - expect[r]).abs() / denom;
            assert!(
                rel < 1e-3,
                "vendor gemv row {r}: expect={} got={} rel={rel}",
                expect[r],
                got[r]
            );
        }
    }
}
