//! Optional AVX2/FMA/AVX-512 helpers for fused quant dots + integer Q4_K×Q8_K.
//!
//! Scalar paths remain the portable default; SIMD is selected at runtime when
//! available. Feature flags are cached once — never call CPUID in the matmul inner loop.

use std::sync::OnceLock;

/// Cached CPU feature bits (computed once).
///
/// A SIMD kernel may run only when every feature in its `#[target_feature(enable = ...)]` list
/// is set here; the helper methods name the exact combinations the kernels enable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    /// AVX-512 F, BW and VL all detected.
    pub avx512f: bool,
    /// `avx512f` (F, BW, VL) and AVX-512 VNNI detected.
    pub avx512_vnni: bool,
}

impl CpuFeatures {
    /// AVX2 and FMA.
    #[inline]
    pub fn avx2_fma(self) -> bool {
        self.avx2 && self.fma
    }

    /// AVX2, FMA and F16C.
    #[inline]
    pub fn avx2_fma_f16c(self) -> bool {
        self.avx2_fma() && self.f16c
    }

    /// AVX-512 F, BW, VL and FMA.
    #[inline]
    pub fn avx512f_fma(self) -> bool {
        self.avx512f && self.fma
    }

    /// AVX-512 F, BW, VL, VNNI, plus AVX2 and FMA.
    #[inline]
    pub fn avx512_vnni_avx2_fma(self) -> bool {
        self.avx512_vnni && self.avx2_fma()
    }

    /// AVX-512 F, BW, VL, VNNI, plus AVX2, FMA and F16C.
    #[inline]
    pub fn avx512_vnni_avx2_fma_f16c(self) -> bool {
        self.avx512_vnni_avx2_fma() && self.f16c
    }

    /// Same CPU with FMA and F16C reported absent (`ARTIFICER_DISABLE_FMA=1`).
    fn without_fma_f16c(self) -> Self {
        CpuFeatures {
            fma: false,
            f16c: false,
            ..self
        }
    }
}

/// Runtime CPU features, detected once. Setting `ARTIFICER_DISABLE_FMA=1` (read once, at first use)
/// reports FMA and F16C as absent, so every kernel that enables either falls back to its non-FMA
/// or scalar path; it exists to test those fallbacks on a CPU that has FMA.
pub(crate) fn cpu_features() -> CpuFeatures {
    static FEAT: OnceLock<CpuFeatures> = OnceLock::new();
    *FEAT.get_or_init(|| {
        let feat = detect_cpu_features();
        if std::env::var_os("ARTIFICER_DISABLE_FMA").is_some_and(|v| v == "1") {
            feat.without_fma_f16c()
        } else {
            feat
        }
    })
}

fn detect_cpu_features() -> CpuFeatures {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let avx512f = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl");
        CpuFeatures {
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            f16c: is_x86_feature_detected!("f16c"),
            avx512f,
            avx512_vnni: avx512f && is_x86_feature_detected!("avx512vnni"),
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        CpuFeatures {
            avx2: false,
            fma: false,
            f16c: false,
            avx512f: false,
            avx512_vnni: false,
        }
    }
}

/// Horizontal sum of 8 f32 lanes (portable).
#[inline]
pub(crate) fn hsum8(v: [f32; 8]) -> f32 {
    (v[0] + v[1]) + (v[2] + v[3]) + (v[4] + v[5]) + (v[6] + v[7])
}

/// Dot product of two f32 slices (AVX2/FMA when available).
#[inline]
pub(crate) fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if n >= 32 && feat.avx512f {
            // SAFETY: AVX-512 F/BW/VL (the kernel enables only avx512f) were detected at run time
            // by `cpu_features()`; n = min(a.len(), b.len()).
            return unsafe { dot_f32_avx512(a, b, n) };
        }
        if n >= 16 && feat.avx2 {
            return if feat.fma {
                // SAFETY: the target features this kernel enables were detected at run time by
                // `cpu_features()`; n = min(a.len(), b.len()).
                unsafe { dot_f32_avx2_fma(a, b, n) }
            } else {
                // SAFETY: the target features this kernel enables were detected at run time by
                // `cpu_features()`; n = min(a.len(), b.len()).
                unsafe { dot_f32_avx2(a, b, n) }
            };
        }
    }
    dot_f32_scalar(a, b, n)
}

/// `y[i] += a * x[i]` (AVX2/FMA / AVX-512 when available).
#[inline]
pub(crate) fn axpy_f32_fast(y: &mut [f32], a: f32, x: &[f32]) {
    let n = y.len().min(x.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if n >= 32 && feat.avx512f {
            // SAFETY: AVX-512 F/BW/VL (the kernel enables only avx512f) were detected at run time
            // by `cpu_features()`; n = min(y.len(), x.len()).
            unsafe { axpy_f32_avx512(y, a, x, n) };
            return;
        }
        if n >= 16 && feat.avx2_fma() {
            // SAFETY: AVX2 and FMA were detected at run time by `cpu_features()`; n = min(y.len(),
            // x.len()).
            unsafe { axpy_f32_avx2_fma(y, a, x, n) };
            return;
        }
    }
    for i in 0..n {
        y[i] += a * x[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpy_f32_avx2_fma(y: &mut [f32], a: f32, x: &[f32], n: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let va = _mm256_set1_ps(a);
    let mut i = 0usize;
    while i + 16 <= n {
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        let y1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(va, x0, y0));
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), _mm256_fmadd_ps(va, x1, y1));
        i += 16;
    }
    while i + 8 <= n {
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(va, x0, y0));
        i += 8;
    }
    while i < n {
        *y.get_unchecked_mut(i) += a * *x.get_unchecked(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn axpy_f32_avx512(y: &mut [f32], a: f32, x: &[f32], n: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let va = _mm512_set1_ps(a);
    let mut i = 0usize;
    while i + 32 <= n {
        let y0 = _mm512_loadu_ps(y.as_ptr().add(i));
        let y1 = _mm512_loadu_ps(y.as_ptr().add(i + 16));
        let x0 = _mm512_loadu_ps(x.as_ptr().add(i));
        let x1 = _mm512_loadu_ps(x.as_ptr().add(i + 16));
        _mm512_storeu_ps(y.as_mut_ptr().add(i), _mm512_fmadd_ps(va, x0, y0));
        _mm512_storeu_ps(y.as_mut_ptr().add(i + 16), _mm512_fmadd_ps(va, x1, y1));
        i += 32;
    }
    while i + 16 <= n {
        let y0 = _mm512_loadu_ps(y.as_ptr().add(i));
        let x0 = _mm512_loadu_ps(x.as_ptr().add(i));
        _mm512_storeu_ps(y.as_mut_ptr().add(i), _mm512_fmadd_ps(va, x0, y0));
        i += 16;
    }
    while i < n {
        *y.get_unchecked_mut(i) += a * *x.get_unchecked(i);
        i += 1;
    }
}

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 8 <= n {
        sum += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3]
            + a[i + 4] * b[i + 4]
            + a[i + 5] * b[i + 5]
            + a[i + 6] * b[i + 6]
            + a[i + 7] * b[i + 7];
        i += 8;
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
        i += 16;
    }
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc0);
    let mut sum = hsum8(tmp);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc0);
    let mut sum = hsum8(tmp);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Whether fused Q4_K dots should use the AVX2(+FMA) path.
#[inline]
#[allow(dead_code)]
pub(crate) fn q4k_avx2_enabled() -> bool {
    cpu_features().avx2
}

/// Whether FMA is available (implies AVX2 path uses fmadd).
#[inline]
#[allow(dead_code)]
pub(crate) fn fma_enabled() -> bool {
    cpu_features().fma
}

/// Whether this build/CPU will use AVX2 on the matmul hot path.
pub fn avx2_enabled() -> bool {
    cpu_features().avx2
}

/// Accumulate `sum((q_nibble * x))` and `sum(x)` for 32 low/high nibbles (AVX2+FMA).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn q4k_sum_qx_x_avx2_fma(q: &[u8], x: &[f32], high: bool) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    debug_assert!(q.len() >= 32);
    debug_assert!(x.len() >= 32);

    let mut acc_qx = _mm256_setzero_ps();
    let mut acc_x = _mm256_setzero_ps();
    let mask = _mm256_set1_epi32(0x0F);
    for i in (0..32).step_by(8) {
        // Load 8 quant bytes.
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&q[i..i + 8]);
        let b128 = _mm_loadl_epi64(bytes.as_ptr() as *const __m128i);
        let mut qi = _mm256_cvtepu8_epi32(b128);
        if high {
            qi = _mm256_srli_epi32(qi, 4);
        }
        qi = _mm256_and_si256(qi, mask);
        let qf = _mm256_cvtepi32_ps(qi);
        let xf = _mm256_loadu_ps(x.as_ptr().add(i));
        acc_qx = _mm256_fmadd_ps(qf, xf, acc_qx);
        acc_x = _mm256_add_ps(acc_x, xf);
    }
    let mut tqx = [0.0f32; 8];
    let mut tx = [0.0f32; 8];
    _mm256_storeu_ps(tqx.as_mut_ptr(), acc_qx);
    _mm256_storeu_ps(tx.as_mut_ptr(), acc_x);
    (hsum8(tqx), hsum8(tx))
}

/// AVX2-only variant (no FMA).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q4k_sum_qx_x_avx2(q: &[u8], x: &[f32], high: bool) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    debug_assert!(q.len() >= 32);
    debug_assert!(x.len() >= 32);

    let mut acc_qx = _mm256_setzero_ps();
    let mut acc_x = _mm256_setzero_ps();
    let mask = _mm256_set1_epi32(0x0F);
    for i in (0..32).step_by(8) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&q[i..i + 8]);
        let b128 = _mm_loadl_epi64(bytes.as_ptr() as *const __m128i);
        let mut qi = _mm256_cvtepu8_epi32(b128);
        if high {
            qi = _mm256_srli_epi32(qi, 4);
        }
        qi = _mm256_and_si256(qi, mask);
        let qf = _mm256_cvtepi32_ps(qi);
        let xf = _mm256_loadu_ps(x.as_ptr().add(i));
        acc_qx = _mm256_add_ps(acc_qx, _mm256_mul_ps(qf, xf));
        acc_x = _mm256_add_ps(acc_x, xf);
    }
    let mut tqx = [0.0f32; 8];
    let mut tx = [0.0f32; 8];
    _mm256_storeu_ps(tqx.as_mut_ptr(), acc_qx);
    _mm256_storeu_ps(tx.as_mut_ptr(), acc_x);
    (hsum8(tqx), hsum8(tx))
}

/// Portable: sum(nibble*x), sum(x) over 32 elements.
#[inline]
pub(crate) fn q4k_sum_qx_x_scalar(q: &[u8], x: &[f32], high: bool) -> (f32, f32) {
    let mut sum_qx = 0.0f32;
    let mut sum_x = 0.0f32;
    if high {
        for l in 0..32 {
            sum_qx += (q[l] >> 4) as f32 * x[l];
            sum_x += x[l];
        }
    } else {
        for l in 0..32 {
            sum_qx += (q[l] & 0x0F) as f32 * x[l];
            sum_x += x[l];
        }
    }
    (sum_qx, sum_x)
}

/// Dispatch AVX2/FMA or scalar for one 32-nibble group.
#[inline]
pub(crate) fn q4k_sum_qx_x(q: &[u8], x: &[f32], high: bool) -> (f32, f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if feat.avx2 {
            return if feat.fma {
                // SAFETY: the target features this kernel enables were detected at run time by
                // `cpu_features()`; the kernel reads 32 values of `q` and `x`, which is only
                // debug-asserted.
                // FIXME(0.1.x): check q.len() >= 32 and x.len() >= 32 in release builds.
                unsafe { q4k_sum_qx_x_avx2_fma(q, x, high) }
            } else {
                // SAFETY: the target features this kernel enables were detected at run time by
                // `cpu_features()`; the kernel reads 32 values of `q` and `x`, which is only
                // debug-asserted.
                // FIXME(0.1.x): check q.len() >= 32 and x.len() >= 32 in release builds.
                unsafe { q4k_sum_qx_x_avx2(q, x, high) }
            };
        }
    }
    q4k_sum_qx_x_scalar(q, x, high)
}

/// Whether AVX-512F(+BW+VL) is available at runtime.
pub fn avx512_enabled() -> bool {
    cpu_features().avx512f
}

/// Whether AVX-512 VNNI (integer Q4_K×Q8_K path) is available.
pub fn avx512_vnni_enabled() -> bool {
    cpu_features().avx512_vnni
}

/// Horizontal sum of 16 f32 lanes.
#[inline]
#[allow(dead_code)]
pub(crate) fn hsum16(v: [f32; 16]) -> f32 {
    hsum8([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]])
        + hsum8([v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        acc0 = _mm512_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm512_fmadd_ps(va1, vb1, acc1);
        i += 32;
    }
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    let mut tmp = [0.0f32; 16];
    _mm512_storeu_ps(tmp.as_mut_ptr(), acc0);
    let mut sum = hsum16(tmp);
    // Tail via AVX2 if possible
    if i + 8 <= n && cpu_features().avx2 {
        // can't call avx2 from here without feature; scalar tail is fine for <16
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32 * 0.05).collect();
        let fast = dot_f32_fast(&a, &b);
        let slow = dot_f32_scalar(&a, &b, 64);
        assert!((fast - slow).abs() < 1e-3, "fast={fast} slow={slow}");
    }

    #[test]
    fn feature_tiers_need_every_enabled_feature() {
        let all = CpuFeatures {
            avx2: true,
            fma: true,
            f16c: true,
            avx512f: true,
            avx512_vnni: true,
        };
        assert!(all.avx2_fma_f16c() && all.avx512_vnni_avx2_fma_f16c() && all.avx512f_fma());
        let no_fma = CpuFeatures { fma: false, ..all };
        assert!(!no_fma.avx2_fma() && !no_fma.avx512f_fma() && !no_fma.avx512_vnni_avx2_fma());
        let no_f16c = CpuFeatures { f16c: false, ..all };
        assert!(no_f16c.avx2_fma() && !no_f16c.avx2_fma_f16c());
        assert!(no_f16c.avx512_vnni_avx2_fma() && !no_f16c.avx512_vnni_avx2_fma_f16c());
        let off = all.without_fma_f16c();
        assert!(!off.fma && !off.f16c && off.avx2 && off.avx512_vnni);
        assert!(!off.avx2_fma() && !off.avx512f_fma() && !off.avx512_vnni_avx2_fma());
    }

    #[test]
    fn disable_fma_env_is_honored() {
        let feat = cpu_features();
        if std::env::var_os("ARTIFICER_DISABLE_FMA").is_some_and(|v| v == "1") {
            assert!(
                !feat.fma && !feat.f16c,
                "ARTIFICER_DISABLE_FMA=1 must clear fma and f16c"
            );
        } else {
            assert_eq!(feat, detect_cpu_features());
        }
    }

    #[test]
    fn q4k_sum_matches_scalar() {
        let q: Vec<u8> = (0..32).map(|i| ((i * 3) ^ 0xA5) as u8).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.07 - 0.5).collect();
        for high in [false, true] {
            let (a, b) = q4k_sum_qx_x(&q, &x, high);
            let (c, d) = q4k_sum_qx_x_scalar(&q, &x, high);
            assert!(
                (a - c).abs() < 1e-3 && (b - d).abs() < 1e-3,
                "high={high} simd=({a},{b}) sc=({c},{d})"
            );
        }
    }
}
