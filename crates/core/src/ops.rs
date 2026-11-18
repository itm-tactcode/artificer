//! Elementwise and attention helpers (f32, CPU).

use crate::config::NormKind;

/// RMSNorm: `out[..x.len()] = x / sqrt(mean(x^2) + eps) * weight`. `out` may be a longer
/// scratch buffer (the forward passes its `xb` scratch, sized max(n_embd, n_head * head_dim));
/// only the first `x.len()` elements are written.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert!(
        out.len() >= x.len(),
        "rmsnorm out {} < x {}",
        out.len(),
        x.len()
    );
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * ss * weight[i];
    }
}

/// Gemma RMSNorm: scale by `(1 + weight)`.
pub fn rmsnorm_gemma(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * ss * (1.0 + weight[i]);
    }
}

/// LayerNorm with optional bias.
pub fn layernorm(x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mut mean = 0.0f32;
    for &v in x {
        mean += v;
    }
    mean /= n;
    let mut var = 0.0f32;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    var = (var / n + eps).sqrt().recip();
    for i in 0..x.len() {
        let mut y = (x[i] - mean) * var * weight[i];
        if let Some(b) = bias {
            y += b[i];
        }
        out[i] = y;
    }
}

pub fn normalize(
    kind: NormKind,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    match kind {
        NormKind::Rms => rmsnorm(x, weight, eps, out),
        NormKind::RmsGemma => rmsnorm_gemma(x, weight, eps, out),
        NormKind::Layer => layernorm(x, weight, bias, eps, out),
    }
}

/// SiLU / Swish: `x * sigmoid(x)`
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// Approximate GELU (tanh form).
#[inline]
pub fn gelu(x: f32) -> f32 {
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 x^3)))
    const C: f32 = 0.797_884_6; // sqrt(2/π)
    let x3 = x * x * x;
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x3)).tanh())
}

pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

/// ReLU then square (Nemotron FFN).
pub fn relu_sqr_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let r = v.max(0.0);
        *v = r * r;
    }
}

/// Softmax in-place over `x`.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Classic RoPE on interleaved pairs within each head.
pub fn apply_rope(
    q_or_k: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    freq_base: f32,
    n_rot: usize,
) {
    debug_assert_eq!(q_or_k.len(), n_heads * head_dim);
    let n_rot = n_rot.min(head_dim);
    for h in 0..n_heads {
        let base = h * head_dim;
        let head = &mut q_or_k[base..base + head_dim];
        for i in (0..n_rot).step_by(2) {
            let freq = 1.0 / freq_base.powf((i as f32) / (n_rot as f32));
            let val = pos as f32 * freq;
            let c = val.cos();
            let s = val.sin();
            let x0 = head[i];
            let x1 = head[i + 1];
            head[i] = x0 * c - x1 * s;
            head[i + 1] = x0 * s + x1 * c;
        }
    }
}

/// NEOX RoPE (GPT-NeoX / llama.cpp `LLAMA_ROPE_TYPE_NEOX`): within each head, rotates the
/// half-split pairs `(i, i + n_rot/2)` for `i < n_rot/2` with
/// `theta = pos * freq_base^(-2i / n_rot)`. Dims `n_rot..head_dim` are left untouched
/// (partial rotary, e.g. phi3 / stablelm). Same frequency formula as [`apply_rope`]; only the
/// pairing differs.
pub fn apply_rope_neox(
    q_or_k: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    freq_base: f32,
    n_rot: usize,
) {
    debug_assert_eq!(q_or_k.len(), n_heads * head_dim);
    let n_rot = n_rot.min(head_dim);
    let half = n_rot / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        let head = &mut q_or_k[base..base + head_dim];
        for i in 0..half {
            let freq = 1.0 / freq_base.powf(((2 * i) as f32) / (n_rot as f32));
            let val = pos as f32 * freq;
            let c = val.cos();
            let s = val.sin();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

/// In-place RMSNorm of `x` with `weight` (same math as [`rmsnorm`]).
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    ss = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        x[i] = x[i] * ss * weight[i];
    }
}

/// q/k RMSNorm before RoPE (llama.cpp `attn_q_norm` / `attn_k_norm`, `LLM_NORM_RMS`).
/// The weight shape picks the flavour:
/// - `{head_dim}`: per-head norm (Qwen3, Gemma3), each head normalized on its own;
/// - `{n_heads * head_dim}`: full-width norm over the whole projection (OLMoE, OLMo2).
///
/// `x` is the first `n_heads * head_dim` values of the projection.
pub fn qk_rmsnorm_inplace(
    x: &mut [f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> std::result::Result<(), String> {
    let width = n_heads * head_dim;
    if x.len() < width {
        return Err(format!("q/k norm: projection len {} < {width}", x.len()));
    }
    let x = &mut x[..width];
    if weight.len() == head_dim {
        for h in 0..n_heads {
            rmsnorm_inplace(&mut x[h * head_dim..(h + 1) * head_dim], weight, eps);
        }
        Ok(())
    } else if weight.len() == width {
        rmsnorm_inplace(x, weight, eps);
        Ok(())
    } else {
        Err(format!(
            "q/k norm weight has {} elements; expected head_dim {head_dim} (per-head) or {width} (full-width)",
            weight.len()
        ))
    }
}

/// Soft-cap: `softcap * tanh(x / softcap)`.
pub fn softcap_inplace(x: &mut [f32], softcap: f32) {
    if softcap <= 0.0 {
        return;
    }
    for v in x.iter_mut() {
        *v = softcap * (*v / softcap).tanh();
    }
}

/// `a += b[..a.len()]`. `b` may be a longer scratch buffer (the forward adds its `xb` scratch,
/// sized max(n_embd, n_head * head_dim), into the n_embd residual).
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert!(
        b.len() >= a.len(),
        "add_inplace b {} < a {}",
        b.len(),
        a.len()
    );
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

pub fn scale_inplace(a: &mut [f32], s: f32) {
    for v in a.iter_mut() {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rmsnorm_unit() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0; 2];
        rmsnorm(&x, &w, 1e-5, &mut out);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn gemma_norm_adds_one() {
        let x = [1.0f32, 0.0];
        let w = [0.0f32, 0.0];
        let mut out = [0.0; 2];
        rmsnorm_gemma(&x, &w, 1e-6, &mut out);
        // weight 0 → scale 1, same as unit rms roughly
        assert!(out[0] > 0.0);
    }

    // Inputs / references from an offline numpy (f64) reference script:
    // x[i] = 0.1 * (i + 1) * (-1)^i, 2 heads x head_dim 8, pos 3, base 10000.
    fn ref_x() -> Vec<f32> {
        (0..16)
            .map(|i| 0.1 * (i + 1) as f32 * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect()
    }

    fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((*g as f64 - w).abs() < tol, "{what}[{i}]: got {g} want {w}");
        }
    }

    #[test]
    fn rope_neox_full_rotary_matches_reference() {
        // n_rot == head_dim: pairs (0,4) (1,5) (2,6) (3,7) in each head.
        let mut x = ref_x();
        apply_rope_neox(&mut x, 2, 8, 3, 10000.0, 8);
        let want = [
            -0.169559255,
            -0.013755170,
            0.278868172,
            -0.397598210,
            -0.480884247,
            -0.632305958,
            0.708683662,
            -0.801196410,
            -1.074449227,
            -0.541608207,
            1.054511811,
            -1.195194655,
            -1.159982194,
            -1.632991269,
            1.532320102,
            -1.603592819,
        ];
        assert_close(&x, &want, 1e-5, "neox full");
    }

    #[test]
    fn rope_neox_partial_rotary_matches_reference() {
        // n_rot (4) < head_dim (8): pairs (0,2) (1,3); dims 4..8 untouched.
        let x0 = ref_x();
        let mut x = x0.clone();
        apply_rope_neox(&mut x, 2, 8, 3, 10000.0, 4);
        let want = [
            -0.141335255,
            -0.187911809,
            -0.282885760,
            -0.405819120,
            0.500000000,
            -0.600000024,
            0.699999988,
            -0.800000012,
            -1.046225236,
            -0.963555432,
            -0.961983766,
            -1.229455588,
            1.299999952,
            -1.399999976,
            1.500000000,
            -1.600000024,
        ];
        assert_close(&x, &want, 1e-5, "neox partial");
        for h in 0..2 {
            for d in 4..8 {
                assert_eq!(
                    x[h * 8 + d].to_bits(),
                    x0[h * 8 + d].to_bits(),
                    "dim {d} must not rotate"
                );
            }
        }
    }

    #[test]
    fn rope_norm_unchanged_matches_reference() {
        // Guard for the default path: interleaved pairs (2i, 2i+1).
        let mut x = ref_x();
        apply_rope(&mut x, 2, 8, 3, 10000.0, 8);
        let want = [
            -0.070775249,
            0.212110503,
            0.404809043,
            -0.293478536,
            0.517772318,
            -0.584732294,
            0.702396835,
            -0.797896415,
            -0.749873215,
            1.117000500,
            1.405494423,
            -0.821331598,
            1.341408696,
            -1.360375875,
            1.504793243,
            -1.595492831,
        ];
        assert_close(&x, &want, 1e-5, "norm");
    }

    #[test]
    fn rope_neox_pos0_is_identity() {
        let x0 = ref_x();
        let mut x = x0.clone();
        apply_rope_neox(&mut x, 2, 8, 0, 10000.0, 8);
        assert_eq!(x, x0);
    }

    #[test]
    fn qk_norm_per_head_matches_reference() {
        let mut x = ref_x();
        let w: Vec<f32> = (0..8).map(|i| 0.5 + 0.125 * i as f32).collect();
        qk_rmsnorm_inplace(&mut x, &w, 2, 8, 1e-6).unwrap();
        let want = [
            0.099014561,
            -0.247536401,
            0.445565534,
            -0.693101924,
            0.990145591,
            -1.336696601,
            1.732754754,
            -2.178320332,
            0.354100145,
            -0.491805770,
            0.649183630,
            -0.826233726,
            1.022955963,
            -1.239350518,
            1.475417309,
            -1.731156335,
        ];
        assert_close(&x, &want, 1e-5, "qk per-head");
    }

    #[test]
    fn qk_norm_full_width_matches_reference() {
        // OLMoE-style {n_embd} weight: one RMS over all heads.
        let mut x = ref_x();
        let w: Vec<f32> = (0..16).map(|i| 1.0 - 0.05 * i as f32).collect();
        qk_rmsnorm_inplace(&mut x, &w, 2, 8, 1e-6).unwrap();
        let want = [
            0.103417484,
            -0.196493220,
            0.279227214,
            -0.351619446,
            0.413669930,
            -0.465378690,
            0.506745656,
            -0.537770917,
            0.558454391,
            -0.568796154,
            0.568796166,
            -0.558454428,
            0.537770889,
            -0.506745656,
            0.465378671,
            -0.413669936,
        ];
        assert_close(&x, &want, 1e-5, "qk full-width");
    }

    #[test]
    fn qk_norm_rejects_bad_weight_shape() {
        let mut x = ref_x();
        let w = vec![1.0f32; 5];
        assert!(qk_rmsnorm_inplace(&mut x, &w, 2, 8, 1e-6).is_err());
    }
}
