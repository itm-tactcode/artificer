//! Mixture-of-experts FFN (OLMoE, Qwen1.5/Qwen2-MoE), CPU only.
//!
//! Mirrors llama.cpp `build_moe_ffn` with softmax gating and SiLU-gated experts:
//! `logits = W_router · x`, `p = softmax(logits)` over all experts, pick the top-k
//! experts by `p`, optionally renormalise their weights (neither OLMoE nor Qwen2-MoE does), then
//! `y = Σ_slot w_slot · down_e(silu(gate_e · x) ⊙ (up_e · x))`, summed in slot order
//! (descending probability).
//!
//! Qwen2-MoE (`qwen2moe`) adds a shared expert (llama.cpp `llama_model_qwen2moe` graph): a dense
//! SwiGLU FFN on the same normalised input, scaled by a per-token sigmoid gate and added to the
//! routed sum: `y += sigmoid(w_gate_inp_shexp · x) · down_sh(silu(gate_sh · x) ⊙ (up_sh · x))`.
//! llama.cpp builds that sigmoid as `silu(g) / g`, which equals `sigmoid(g)` for every `g != 0`
//! (and is NaN at exactly 0); we use the true logistic function.
//!
//! Each expert is a contiguous sub-slice of the 3-D `ffn_*_exps` tensors. Expert matmuls go
//! through [`crate::backend::Backend::matmul_vec_expert`], which on CPU never touches the
//! pointer-keyed Q5_0×4 / Q4_K×8 repack caches (one entry per expert per layer would copy
//! the whole expert set into the heap).

use crate::config::MoeConfig;
use crate::error::{CoreError, Result};
use crate::model::{expert_slice, LlamaModel, MoeWeights, SharedExpertWeights, WeightTensor};
use crate::ops::{mul_inplace, silu_inplace, softmax_inplace};
use crate::profile;

use super::scratch::ForwardScratch;
use super::weights::matmul_wt;

/// Softmax over all `logits` (in place, becomes probabilities), then the `k` largest
/// probabilities as `(expert, weight)` in descending order (ties: lower expert index first).
/// With `renorm`, the selected weights are divided by their sum.
pub fn route_topk(logits: &mut [f32], k: usize, renorm: bool, out: &mut Vec<(usize, f32)>) {
    out.clear();
    if logits.is_empty() || k == 0 {
        return;
    }
    softmax_inplace(logits);
    let k = k.min(logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    // Stable sort by descending probability: equal probabilities keep index order.
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.extend(idx[..k].iter().map(|&e| (e, logits[e])));
    if renorm {
        let s: f32 = out.iter().map(|&(_, w)| w).sum();
        if s > 0.0 {
            for (_, w) in out.iter_mut() {
                *w /= s;
            }
        }
    }
}

/// One expert matmul: `out = W_e · x` for expert `e` of 3-D tensor `w`.
fn expert_matmul(
    model: &LlamaModel,
    w: &WeightTensor,
    n_expert: usize,
    e: usize,
    x: &[f32],
    out: &mut [f32],
    backend: &dyn crate::backend::Backend,
) -> Result<()> {
    let data = model.tensor_bytes(w)?;
    let (k, n) = (w.k(), w.n());
    let slice = expert_slice(data, w.ty, k, n, n_expert, e)?;
    if x.len() < k || out.len() < n {
        return Err(CoreError::Shape(format!(
            "expert matmul {}: x.len={} k={k} out.len={} n={n}",
            w.name,
            x.len(),
            out.len()
        )));
    }
    let _role = profile::time_role(&w.name);
    backend.matmul_vec_expert(slice, w.ty, k, n, &x[..k], &mut out[..n])
}

/// MoE FFN on the normalised input `scratch.xb[..n_embd]`; result written to
/// `scratch.xb[..n_embd]` (caller adds the residual).
pub(super) fn moe_ffn(
    model: &LlamaModel,
    moe: &MoeWeights,
    cfg: &MoeConfig,
    scratch: &mut ForwardScratch,
    backend: &dyn crate::backend::Backend,
) -> Result<()> {
    let d = model.config.n_embd as usize;
    let n_expert = cfg.n_expert as usize;
    let ff = moe.n_ff_exp;
    if scratch.moe_logits.len() < n_expert
        || scratch.moe_acc.len() < d
        || scratch.moe_out.len() < d
        || scratch.hb.len() < ff
        || scratch.hb2.len() < ff
    {
        return Err(CoreError::Shape("MoE scratch buffers too small".into()));
    }

    // Router logits (llama.cpp `ffn_moe_logits`).
    matmul_wt(
        model,
        &moe.gate_inp,
        &scratch.xb[..d],
        &mut scratch.moe_logits[..n_expert],
        backend,
    )?;
    let mut sel = std::mem::take(&mut scratch.moe_sel);
    route_topk(
        &mut scratch.moe_logits[..n_expert],
        cfg.n_expert_used as usize,
        cfg.norm_topk_prob,
        &mut sel,
    );

    scratch.moe_acc[..d].fill(0.0);
    for &(e, w) in &sel {
        expert_matmul(
            model,
            &moe.gate_exps,
            n_expert,
            e,
            &scratch.xb[..d],
            &mut scratch.hb[..ff],
            backend,
        )?;
        expert_matmul(
            model,
            &moe.up_exps,
            n_expert,
            e,
            &scratch.xb[..d],
            &mut scratch.hb2[..ff],
            backend,
        )?;
        silu_inplace(&mut scratch.hb[..ff]);
        mul_inplace(&mut scratch.hb[..ff], &scratch.hb2[..ff]);
        expert_matmul(
            model,
            &moe.down_exps,
            n_expert,
            e,
            &scratch.hb[..ff],
            &mut scratch.moe_out[..d],
            backend,
        )?;
        for (a, &o) in scratch.moe_acc[..d].iter_mut().zip(&scratch.moe_out[..d]) {
            *a += o * w;
        }
    }
    scratch.moe_sel = sel;

    // Shared expert (Qwen2-MoE), added after the routed sum like llama.cpp `ffn_out`.
    if let Some(sh) = &moe.shexp {
        let ff_sh = sh.n_ff_shexp;
        if scratch.hb.len() < ff_sh || scratch.hb2.len() < ff_sh {
            return Err(CoreError::Shape(
                "MoE shared-expert scratch buffers too small".into(),
            ));
        }
        shared_expert_add(
            sh,
            &scratch.xb[..d],
            &mut scratch.hb[..ff_sh],
            &mut scratch.hb2[..ff_sh],
            &mut scratch.moe_out[..d],
            &mut scratch.moe_acc[..d],
            |w, x, out| matmul_wt(model, w, x, out, backend),
        )?;
    }

    scratch.xb[..d].copy_from_slice(&scratch.moe_acc[..d]);
    Ok(())
}

/// Logistic sigmoid `1 / (1 + e^-x)`: 0.5 at 0, saturates to 0 / 1 without NaN.
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `acc += sigmoid(gate_logit) · shexp_out` (llama.cpp `ffn_shexp_out`, then `ffn_out`).
pub(crate) fn add_gated_shared_expert(acc: &mut [f32], shexp_out: &[f32], gate_logit: f32) {
    let s = sigmoid(gate_logit);
    for (a, &o) in acc.iter_mut().zip(shexp_out) {
        *a += o * s;
    }
}

/// One token through the shared expert `sh`, added into `acc`:
/// `acc += sigmoid(gate_inp_shexp · x) · down_shexp(silu(gate_shexp · x) ⊙ (up_shexp · x))`.
/// `hb` / `hb2` are `[n_ff_shexp]` work buffers, `out` is `[n_embd]`; `matmul(w, x, out)`
/// computes `out = W · x` (the forward pass passes [`matmul_wt`]).
fn shared_expert_add<M>(
    sh: &SharedExpertWeights,
    x: &[f32],
    hb: &mut [f32],
    hb2: &mut [f32],
    out: &mut [f32],
    acc: &mut [f32],
    mut matmul: M,
) -> Result<()>
where
    M: FnMut(&WeightTensor, &[f32], &mut [f32]) -> Result<()>,
{
    matmul(&sh.gate_shexp, x, &mut *hb)?;
    matmul(&sh.up_shexp, x, &mut *hb2)?;
    silu_inplace(hb);
    mul_inplace(hb, hb2);
    matmul(&sh.down_shexp, &*hb, &mut *out)?;
    let mut gate_logit = [0.0f32; 1];
    matmul(&sh.gate_inp_shexp, x, &mut gate_logit[..])?;
    add_gated_shared_expert(acc, out, gate_logit[0]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{add_gated_shared_expert, route_topk, shared_expert_add, sigmoid};
    use crate::backend::{Backend, CpuBackend};
    use crate::model::{SharedExpertWeights, WeightTensor};
    use gguf::TensorType;

    // Reference from numpy (f64): p = softmax(l); order = argsort(-p, kind="stable").
    const L: [f32; 8] = [0.5, 2.0, -1.0, 1.5, 2.0, 0.0, 3.0, -0.5];
    const P: [f64; 8] = [
        0.03837049, 0.17196461, 0.00856161, 0.10430181, 0.17196461, 0.02327288, 0.46744827,
        0.01411571,
    ];

    #[test]
    fn router_softmax_topk_no_renorm_matches_numpy() {
        let mut l = L;
        let mut sel = Vec::new();
        route_topk(&mut l, 3, false, &mut sel);
        // Softmax over all experts (in place).
        for (g, w) in l.iter().zip(P) {
            assert!((*g as f64 - w).abs() < 1e-6, "p {g} vs {w}");
        }
        // Top-3; experts 1 and 4 tie exactly, lower index first.
        let ids: Vec<usize> = sel.iter().map(|s| s.0).collect();
        assert_eq!(ids, vec![6, 1, 4]);
        for (&(_, w), want) in sel.iter().zip([0.46744827, 0.17196461, 0.17196461]) {
            assert!((w as f64 - want).abs() < 1e-6, "w {w} vs {want}");
        }
    }

    #[test]
    fn router_topk_renorm_matches_numpy() {
        let mut l = L;
        let mut sel = Vec::new();
        route_topk(&mut l, 3, true, &mut sel);
        let ids: Vec<usize> = sel.iter().map(|s| s.0).collect();
        assert_eq!(ids, vec![6, 1, 4]);
        for (&(_, w), want) in sel.iter().zip([0.57611688, 0.21194156, 0.21194156]) {
            assert!((w as f64 - want).abs() < 1e-6, "w {w} vs {want}");
        }
        let s: f32 = sel.iter().map(|s| s.1).sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn router_topk_k_all_and_k_clamped() {
        let mut l = L;
        let mut sel = Vec::new();
        route_topk(&mut l, 99, false, &mut sel);
        assert_eq!(sel.len(), 8);
        let ids: Vec<usize> = sel.iter().map(|s| s.0).collect();
        assert_eq!(ids, vec![6, 1, 4, 3, 0, 5, 7, 2]);
        let s: f32 = sel.iter().map(|s| s.1).sum();
        assert!((s - 1.0).abs() < 1e-6);
        route_topk(&mut [], 2, false, &mut sel);
        assert!(sel.is_empty());
    }

    // --- F2: Qwen2-MoE shared expert ---

    #[test]
    fn sigmoid_values_and_saturation() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(3f32.ln()) - 0.75).abs() < 1e-6);
        assert!((sigmoid(-(3f32.ln())) - 0.25).abs() < 1e-6);
        for x in [-200.0f32, -90.0, 90.0, 200.0, f32::MAX, f32::MIN] {
            let s = sigmoid(x);
            assert!(
                s.is_finite() && (0.0..=1.0).contains(&s),
                "sigmoid({x}) = {s}"
            );
        }
        assert!(sigmoid(-200.0) < 1e-30);
        assert_eq!(sigmoid(200.0), 1.0);
        // llama.cpp's `silu(g) / g` form agrees away from g = 0 (where it is NaN).
        for g in [-6.0f32, -1.5, -0.01, 0.01, 0.7, 4.0] {
            let llama = crate::ops::silu(g) / g;
            assert!(
                (sigmoid(g) - llama).abs() < 1e-6,
                "g={g}: {} vs {llama}",
                sigmoid(g)
            );
        }
    }

    #[test]
    fn gated_shared_expert_adds_sigmoid_scaled_output() {
        let out = [4.0f32, -2.0, 0.5];
        let mut acc = [1.0f32, 2.0, -3.0];
        add_gated_shared_expert(&mut acc, &out, 0.0);
        assert_eq!(acc, [3.0, 1.0, -2.75]);
        let mut acc = [1.0f32, 2.0, -3.0];
        add_gated_shared_expert(&mut acc, &out, 3f32.ln());
        for (a, want) in acc.iter().zip([4.0f32, 0.5, -2.625]) {
            assert!((a - want).abs() < 1e-6, "{a} vs {want}");
        }
        // A strongly negative gate switches the shared expert off.
        let mut acc = [1.0f32, 2.0, -3.0];
        add_gated_shared_expert(&mut acc, &out, -100.0);
        assert_eq!(acc, [1.0, 2.0, -3.0]);
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn lcg_weights(seed: u32, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The whole shared-expert step (SwiGLU + sigmoid gate + add) on tiny F32 tensors through
    /// the CPU matmul, against a naive f64 reference in ggml layout (`W {k, n}`: row j is
    /// `w[j*k .. (j+1)*k]`).
    #[test]
    fn shared_expert_matches_f64_reference_on_tiny_tensors() {
        const D: usize = 4;
        const FF: usize = 3;
        let wt = |name: &str, shape: &[u64]| WeightTensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            ty: TensorType::F32,
        };
        let sh = SharedExpertWeights {
            gate_inp_shexp: wt("inp", &[D as u64]),
            gate_shexp: wt("gate", &[D as u64, FF as u64]),
            up_shexp: wt("up", &[D as u64, FF as u64]),
            down_shexp: wt("down", &[FF as u64, D as u64]),
            n_ff_shexp: FF,
        };
        let wg = lcg_weights(11, D * FF);
        let wu = lcg_weights(22, D * FF);
        let wd = lcg_weights(33, FF * D);
        let wi = lcg_weights(44, D);
        let bytes = |name: &str| -> Vec<u8> {
            match name {
                "inp" => f32_bytes(&wi),
                "gate" => f32_bytes(&wg),
                "up" => f32_bytes(&wu),
                "down" => f32_bytes(&wd),
                _ => unreachable!(),
            }
        };
        let x = [0.5f32, -1.25, 2.0, 0.75];
        let acc0 = [0.1f32, -0.2, 0.3, -0.4];
        let (mut hb, mut hb2, mut out, mut acc) = ([0.0f32; FF], [0.0f32; FF], [0.0f32; D], acc0);
        shared_expert_add(
            &sh,
            &x,
            &mut hb,
            &mut hb2,
            &mut out,
            &mut acc,
            |w, xin, o| CpuBackend.matmul_vec(&bytes(&w.name), w.ty, w.k(), w.n(), xin, o),
        )
        .unwrap();

        let mv = |w: &[f32], k: usize, n: usize, v: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|j| (0..k).map(|i| w[j * k + i] as f64 * v[i]).sum())
                .collect()
        };
        let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let g = mv(&wg, D, FF, &xd);
        let u = mv(&wu, D, FF, &xd);
        let h: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let y = mv(&wd, FF, D, &h);
        let logit: f64 = (0..D).map(|i| wi[i] as f64 * xd[i]).sum();
        let s = 1.0 / (1.0 + (-logit).exp());
        assert!(
            y.iter().any(|v| v.abs() > 1e-3),
            "degenerate reference {y:?}"
        );
        for m in 0..D {
            let want = acc0[m] as f64 + s * y[m];
            assert!(
                (acc[m] as f64 - want).abs() < 1e-5,
                "m={m}: {} vs {want}",
                acc[m]
            );
            // `out` holds the ungated shared-expert output.
            assert!(
                (out[m] as f64 - y[m]).abs() < 1e-5,
                "out[{m}]: {} vs {}",
                out[m],
                y[m]
            );
        }
    }
}
