//! Single-layer attention and FFN blocks.

use crate::config::{FfnActivation, RopeType};
use crate::error::{CoreError, Result};
use crate::kv::KvCache;
use crate::model::{LayerWeights, LlamaModel};
use crate::ops::{
    add_inplace, apply_rope, apply_rope_neox, gelu_inplace, mul_inplace, qk_rmsnorm_inplace,
    relu_sqr_inplace, scale_inplace, silu_inplace, softcap_inplace, softmax_inplace,
};

use super::scratch::ForwardScratch;
use super::weights::{add_bias, apply_norm, load_f32_vec, matmul_wt};
use crate::profile;
use crate::quant::simd::{axpy_f32_fast, dot_f32_fast};

pub(super) fn attention_block(
    model: &LlamaModel,
    layer: &LayerWeights,
    layer_idx: usize,
    pos: usize,
    cache: &mut KvCache,
    scratch: &mut ForwardScratch,
    backend: &dyn crate::backend::Backend,
) -> Result<()> {
    let cfg = &model.config;
    let n_head = cfg.n_head as usize;
    let n_head_kv = cfg
        .layer_n_head_kv
        .get(layer_idx)
        .copied()
        .filter(|&x| x > 0)
        .unwrap_or(cfg.n_head_kv) as usize;
    let head_dim = cfg.head_dim() as usize;
    let n_ctx = cfg.n_ctx as usize;
    let n_rot = cfg.n_rot as usize;

    let attn_q = layer
        .attn_q
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("attn_q".into()))?;
    let attn_k = layer
        .attn_k
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("attn_k".into()))?;
    let attn_v = layer
        .attn_v
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("attn_v".into()))?;
    let attn_output = layer
        .attn_output
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("attn_output".into()))?;

    apply_norm(
        model,
        cfg,
        &layer.attn_norm,
        layer.attn_norm_b.as_ref(),
        &scratch.x,
        &mut scratch.xb,
        &mut scratch.norm_w,
        &mut scratch.norm_b,
    )?;

    matmul_wt(model, attn_q, &scratch.xb, &mut scratch.q, backend)?;
    matmul_wt(model, attn_k, &scratch.xb, &mut scratch.k, backend)?;
    matmul_wt(model, attn_v, &scratch.xb, &mut scratch.v, backend)?;
    if let Some(ref b) = layer.attn_q_b {
        add_bias(model, b, &mut scratch.q)?;
    }
    if let Some(ref b) = layer.attn_k_b {
        add_bias(model, b, &mut scratch.k)?;
    }
    if let Some(ref b) = layer.attn_v_b {
        add_bias(model, b, &mut scratch.v)?;
    }

    // Optional q/k RMSNorm (Qwen3 per-head, OLMoE full-width): after the bias, before RoPE.
    if let Some(ref w) = layer.attn_q_norm {
        qk_norm(model, w, &mut scratch.q, n_head, head_dim, cfg.rms_eps)?;
    }
    if let Some(ref w) = layer.attn_k_norm {
        qk_norm(model, w, &mut scratch.k, n_head_kv, head_dim, cfg.rms_eps)?;
    }

    // Nemotron-H / hybrid: no positional embeddings (paper). Dense arches use RoPE.
    if !cfg.architecture.is_hybrid() && n_rot > 0 && cfg.rope_freq_base > 0.0 {
        let rope = match cfg.rope_type {
            RopeType::Norm => apply_rope,
            RopeType::Neox => apply_rope_neox,
        };
        rope(
            &mut scratch.q,
            n_head,
            head_dim,
            pos,
            cfg.rope_freq_base,
            n_rot,
        );
        rope(
            &mut scratch.k,
            n_head_kv,
            head_dim,
            pos,
            cfg.rope_freq_base,
            n_rot,
        );
    }

    // Gemma: scale Q by 1/√d and use attention scale 1.0.
    let att_scale = if cfg.scale_embeddings {
        scale_inplace(&mut scratch.q, 1.0 / (head_dim as f32).sqrt());
        1.0
    } else {
        1.0 / (head_dim as f32).sqrt()
    };

    cache.write(layer_idx, pos, &scratch.k, &scratch.v);

    // Gemma2: even layers use sliding window; odd layers full context (when configured).
    let attn_start = if cfg.sliding_window > 0 {
        let use_window = if cfg.sliding_window_pattern {
            layer_idx.is_multiple_of(2)
        } else {
            true // all layers windowed when pattern is false but window set (e.g. gemma3-ish)
        };
        if use_window {
            pos.saturating_sub(cfg.sliding_window as usize - 1)
        } else {
            0
        }
    } else {
        0
    };

    {
        let _t = profile::time_attn_non_mm();
        #[cfg(feature = "cuda")]
        if let Some(cuda) = backend.as_cuda() {
            crate::backend::cuda_attention::device_multi_head_attn(
                cuda,
                scratch,
                cache,
                layer_idx,
                pos,
                attn_start,
                n_head,
                n_head_kv,
                head_dim,
                n_ctx,
                att_scale,
                cfg.attn_logit_softcap,
            )?;
        } else {
            multi_head_attn(
                scratch,
                cache,
                layer_idx,
                pos,
                attn_start,
                n_head,
                n_head_kv,
                head_dim,
                n_ctx,
                att_scale,
                cfg.attn_logit_softcap,
            );
        }
        #[cfg(not(feature = "cuda"))]
        multi_head_attn(
            scratch,
            cache,
            layer_idx,
            pos,
            attn_start,
            n_head,
            n_head_kv,
            head_dim,
            n_ctx,
            att_scale,
            cfg.attn_logit_softcap,
        );
    }

    matmul_wt(model, attn_output, &scratch.xb2, &mut scratch.xb, backend)?;
    if let Some(ref b) = layer.attn_output_b {
        add_bias(model, b, &mut scratch.xb)?;
    }
    add_inplace(&mut scratch.x, &scratch.xb);
    Ok(())
}

/// Load a q/k-norm weight and normalize the first `n_heads * head_dim` values of `x`.
fn qk_norm(
    model: &LlamaModel,
    w: &crate::model::WeightTensor,
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    let mut wf = vec![0.0f32; w.n_elements()];
    load_f32_vec(model, w, &mut wf)?;
    qk_rmsnorm_inplace(x, &wf, n_heads, head_dim, eps).map_err(|msg| CoreError::BadTensor {
        name: w.name.clone(),
        msg,
    })
}

pub(crate) fn multi_head_attn(
    scratch: &mut ForwardScratch,
    cache: &KvCache,
    layer_idx: usize,
    pos: usize,
    attn_start: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_ctx: usize,
    att_scale: f32,
    softcap: f32,
) {
    // Zero only the Q-head outputs we write (n_head * head_dim), not the whole xb2 scratch.
    let q_dim = n_head * head_dim;
    scratch.xb2[..q_dim].fill(0.0);
    let kv_mul = n_head / n_head_kv;
    let use_direct = cache.elem == crate::kv::KvElem::F32;
    let start = attn_start.min(pos);
    let att_len = pos + 1 - start;

    // Score pass: outer over time so f16 KV is dequantized once per position
    // (was once per query-head × position — ~n_head× waste on GQA).
    if use_direct {
        for t in start..=pos {
            let k = cache.k_at(layer_idx, t);
            let ti = t - start;
            for h in 0..n_head {
                let h_kv = h / kv_mul;
                let q = &scratch.q[h * head_dim..(h + 1) * head_dim];
                let k_h = &k[h_kv * head_dim..(h_kv + 1) * head_dim];
                scratch.att[h * n_ctx + ti] = dot_f32_fast(q, k_h) * att_scale;
            }
        }
    } else {
        for t in start..=pos {
            cache.copy_k_into(layer_idx, t, &mut scratch.kv_row);
            let ti = t - start;
            for h in 0..n_head {
                let h_kv = h / kv_mul;
                let q = &scratch.q[h * head_dim..(h + 1) * head_dim];
                let k_h = &scratch.kv_row[h_kv * head_dim..(h_kv + 1) * head_dim];
                scratch.att[h * n_ctx + ti] = dot_f32_fast(q, k_h) * att_scale;
            }
        }
    }

    for h in 0..n_head {
        let att = &mut scratch.att[h * n_ctx..h * n_ctx + att_len];
        if softcap > 0.0 {
            softcap_inplace(att, softcap);
        }
        softmax_inplace(att);
    }

    // Value accumulate: same t-outer dequant once per position.
    if use_direct {
        for t in start..=pos {
            let v = cache.v_at(layer_idx, t);
            let ti = t - start;
            for h in 0..n_head {
                let h_kv = h / kv_mul;
                let a = scratch.att[h * n_ctx + ti];
                let v_h = &v[h_kv * head_dim..(h_kv + 1) * head_dim];
                axpy_f32_fast(&mut scratch.xb2[h * head_dim..(h + 1) * head_dim], a, v_h);
            }
        }
    } else {
        for t in start..=pos {
            cache.copy_v_into(layer_idx, t, &mut scratch.kv_row);
            let ti = t - start;
            for h in 0..n_head {
                let h_kv = h / kv_mul;
                let a = scratch.att[h * n_ctx + ti];
                let v_h = &scratch.kv_row[h_kv * head_dim..(h_kv + 1) * head_dim];
                axpy_f32_fast(&mut scratch.xb2[h * head_dim..(h + 1) * head_dim], a, v_h);
            }
        }
    }
}

pub(super) fn ffn_block(
    model: &LlamaModel,
    layer: &LayerWeights,
    scratch: &mut ForwardScratch,
    backend: &dyn crate::backend::Backend,
) -> Result<()> {
    let cfg = &model.config;
    if let Some(ref moe) = layer.moe {
        let ffn_norm = layer
            .ffn_norm
            .as_ref()
            .ok_or_else(|| CoreError::MissingTensor("ffn_norm.weight".into()))?;
        let moe_cfg = cfg
            .moe
            .as_ref()
            .ok_or_else(|| CoreError::Other("MoE layer without MoE config".into()))?;
        apply_norm(
            model,
            cfg,
            ffn_norm,
            layer.ffn_norm_b.as_ref(),
            &scratch.x,
            &mut scratch.xb,
            &mut scratch.norm_w,
            &mut scratch.norm_b,
        )?;
        super::moe::moe_ffn(model, moe, moe_cfg, scratch, backend)?;
        let d = cfg.n_embd as usize;
        add_inplace(&mut scratch.x[..d], &scratch.xb[..d]);
        return Ok(());
    }
    let ffn_up = layer
        .ffn_up
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("ffn_up".into()))?;
    let ffn_down = layer
        .ffn_down
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("ffn_down".into()))?;

    // Dense: separate ffn_norm. Hybrid FFN-only: reuse attn_norm (pre-norm on residual).
    if let Some(ref ffn_norm) = layer.ffn_norm {
        apply_norm(
            model,
            cfg,
            ffn_norm,
            layer.ffn_norm_b.as_ref(),
            &scratch.x,
            &mut scratch.xb,
            &mut scratch.norm_w,
            &mut scratch.norm_b,
        )?;
    } else {
        apply_norm(
            model,
            cfg,
            &layer.attn_norm,
            layer.attn_norm_b.as_ref(),
            &scratch.x,
            &mut scratch.xb,
            &mut scratch.norm_w,
            &mut scratch.norm_b,
        )?;
    }

    match cfg.ffn_activation {
        FfnActivation::SiluGated | FfnActivation::GeluGated => {
            let gate = layer
                .ffn_gate
                .as_ref()
                .ok_or_else(|| CoreError::MissingTensor("ffn_gate.weight".into()))?;
            matmul_wt(model, gate, &scratch.xb, &mut scratch.hb, backend)?;
            matmul_wt(model, ffn_up, &scratch.xb, &mut scratch.hb2, backend)?;
            if let Some(ref b) = layer.ffn_up_b {
                add_bias(model, b, &mut scratch.hb2)?;
            }
            match cfg.ffn_activation {
                FfnActivation::SiluGated => silu_inplace(&mut scratch.hb),
                FfnActivation::GeluGated => gelu_inplace(&mut scratch.hb),
                _ => {}
            }
            mul_inplace(&mut scratch.hb, &scratch.hb2);
            matmul_wt(model, ffn_down, &scratch.hb, &mut scratch.xb, backend)?;
        }
        FfnActivation::ReluSqrSequential => {
            matmul_wt(model, ffn_up, &scratch.xb, &mut scratch.hb, backend)?;
            if let Some(ref b) = layer.ffn_up_b {
                add_bias(model, b, &mut scratch.hb)?;
            }
            relu_sqr_inplace(&mut scratch.hb);
            matmul_wt(model, ffn_down, &scratch.hb, &mut scratch.xb, backend)?;
        }
    }
    if let Some(ref b) = layer.ffn_down_b {
        add_bias(model, b, &mut scratch.xb)?;
    }
    add_inplace(&mut scratch.x, &scratch.xb);
    Ok(())
}
