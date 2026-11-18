//! Quantization: dequantizers and fused matrix-vector products.
//!
//! GGML block layouts (MIT-compatible algorithms).

mod blocks;
mod dequant;
mod ggml_ffi;
mod half;
mod matmul;
mod pool;
pub(crate) use pool::WaitForJobs;
mod q4k_repack;
mod q50;
mod q50_repack;
mod q6k;
mod q8k;
pub(crate) mod simd;

pub use dequant::dequantize;
#[allow(unused_imports)] // public API for callers / tests
pub use half::{f16_to_f32, f32_to_f16};
pub use matmul::{matmul_vec, matmul_vec_no_repack};
pub use pool::pool_workers;
#[allow(unused_imports)] // re-exported via crate root
pub use simd::{avx2_enabled, avx512_enabled, avx512_vnni_enabled};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequant::dequant_q4_0_for_test;

    #[test]
    fn f16_one() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        assert!((f16_to_f32(0x0000)).abs() < 1e-6);
    }

    #[test]
    fn q4_0_roundtrip_shape() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for j in 0..16 {
            block[2 + j] = (j as u8) | (((j as u8) % 16) << 4);
        }
        let mut out = vec![0.0f32; 32];
        dequant_q4_0_for_test(&block, 0, 32, &mut out).unwrap();
        assert!((out[0] + 8.0).abs() < 1e-5);
    }

    /// MoE repack bypass: expert sub-slices of a fat-N Q5_0 3-D tensor (eligible for the
    /// Q5_0×4 repack) run through `matmul_vec_no_repack` without adding a single cache entry
    /// (Q5_0×4 or Q4_K×8) for any pointer inside the tensor. Control: the default
    /// `matmul_vec` on one slice does add exactly one entry, so the count is sensitive.
    /// Counting is by pointer range, so parallel tests touching other buffers can't race it.
    #[test]
    fn moe_expert_matmul_bypasses_repack_caches() {
        use crate::model::expert_slice;
        use crate::model::moe_slice_tests::fake_blocks;
        use gguf::TensorType;

        let ty = TensorType::Q5_0;
        let (k, n, ne) = (64usize, 256usize, 4usize); // n >= 256: x4 repack would apply
        let data = fake_blocks(ty, k * n * ne / 32, 5);
        let (lo, hi) = (data.as_ptr() as usize, data.as_ptr() as usize + data.len());
        let count = || {
            (
                super::q50_repack::cache_entries_in(lo, hi),
                super::q4k_repack::cache_entries_in(lo, hi),
            )
        };
        let x: Vec<f32> = (0..k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();

        let mut outs = Vec::new();
        for e in 0..ne {
            let s = expert_slice(&data, ty, k, n, ne, e).unwrap();
            let mut out = vec![0.0f32; n];
            matmul_vec_no_repack(s, ty, k, n, &x, &mut out).unwrap();
            // f32 reference from a dequant of the same slice (Q8_0 activation quant error).
            let mut w = vec![0.0f32; k * n];
            dequantize(s, ty, 0, k * n, &mut w).unwrap();
            for j in 0..n {
                let row = &w[j * k..(j + 1) * k];
                let r: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
                let mag: f32 = row.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
                assert!(
                    (out[j] - r).abs() <= 0.02 * mag + 1e-5,
                    "e{e} row {j}: {} vs {r}",
                    out[j]
                );
            }
            outs.push(out);
        }
        assert_eq!(
            count(),
            (0, 0),
            "expert matmuls must not populate repack caches"
        );

        if !super::ggml_ffi::ggml_ffi_active() {
            let s0 = expert_slice(&data, ty, k, n, ne, 0).unwrap();
            let mut out_r = vec![0.0f32; n];
            matmul_vec(s0, ty, k, n, &x, &mut out_r).unwrap();
            assert_eq!(
                count(),
                (1, 0),
                "control: default path caches the x4 repack"
            );
            for j in 0..n {
                let (a, b) = (outs[0][j], out_r[j]);
                let rel = (a - b).abs() / (b.abs() + 1e-6);
                assert!(rel < 1e-5, "row {j}: bypass {a} vs repacked {b}");
            }
        }
    }
}
