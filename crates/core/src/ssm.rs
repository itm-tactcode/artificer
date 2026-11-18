//! Mamba-2 SSM state cache and single-token step (Nemotron-H hybrid).
//!
//! CPU hot path (conv1d + selective scan + elementwise) uses AVX-512/AVX2+FMA
//! when available. Conv weights are stored as `[d_conv][conv_dim]` for contiguous
//! channel-wise SIMD. Heads in the selective scan are independent and are farmed
//! across a small persistent worker pool when `n_head` is large.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use crate::backend::Backend;
use crate::config::{ModelConfig, SsmConfig};
use crate::error::{CoreError, Result};
use crate::model::{LayerWeights, LlamaModel, WeightTensor};
use crate::ops::{rmsnorm, silu};
use crate::profile;
use crate::quant::simd::cpu_features;
use crate::quant::WaitForJobs;

/// Softplus used by Mamba (clamped for large x).
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Pre-dequantized static tensors for one SSM layer (loaded once at engine init).
pub(crate) struct SsmLayerStatic {
    /// Causal conv kernel, length `d_conv * conv_dim`, layout **`[d_conv][conv_dim]`**
    /// (repacked from GGUF `[channel, d_conv]` for contiguous SIMD across channels).
    pub(crate) conv_w: Vec<f32>,
    /// Conv bias, length `conv_dim` (zeros if absent).
    pub(crate) conv_b: Vec<f32>,
    /// Per-head A, D, dt bias.
    pub(crate) a: Vec<f32>,
    pub(crate) d: Vec<f32>,
    pub(crate) dt_b: Vec<f32>,
    /// Grouped RMSNorm weights, length `d_inner`.
    pub(crate) norm: Vec<f32>,
    /// Pre-norm on residual stream, length `n_embd`.
    pub(crate) attn_norm: Vec<f32>,
}

/// Per-layer recurrent SSM + conv state for hybrid models.
pub struct SsmCache {
    pub n_ssm_layers: usize,
    /// Maps model layer index → slot in this cache (usize::MAX if not SSM).
    layer_to_slot: Vec<usize>,
    /// conv: [slot][(d_conv-1) * conv_dim]
    conv: Vec<Vec<f32>>,
    /// ssm: [slot][n_head * head_dim * d_state] layout (h, dim, state)
    ssm: Vec<Vec<f32>>,
    /// Static weights per SSM slot (no per-token dequant/alloc).
    statics: Vec<SsmLayerStatic>,
    pub d_conv: usize,
    pub conv_dim: usize,
    pub d_state: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub d_inner: usize,
    pub n_group: usize,
    /// Device-resident conv + SSM state for the CUDA `ssm_mix` hook (cuda builds
    /// only; `None` until the first device step, never touched by the CPU path).
    #[cfg(feature = "cuda")]
    pub(crate) device: Option<crate::backend::cuda_hostops::DeviceSsmState>,
}

impl SsmCache {
    /// Build cache + preload small F32 SSM tensors from the model (once).
    pub fn from_model(model: &LlamaModel) -> Result<Self> {
        let ssm = model
            .config
            .ssm
            .ok_or_else(|| CoreError::Other("SsmCache::from_model without SsmConfig".into()))?;
        let n_layer = model.config.n_layer as usize;
        let is_ssm: Vec<bool> = model
            .config
            .layer_types
            .iter()
            .map(|t| *t == crate::config::LayerType::Ssm)
            .collect();
        Self::new(n_layer, &is_ssm, &ssm, model)
    }

    pub fn new(
        n_layer: usize,
        layer_is_ssm: &[bool],
        ssm: &SsmConfig,
        model: &LlamaModel,
    ) -> Result<Self> {
        let d_conv = ssm.d_conv as usize;
        let conv_dim = ssm.conv_dim() as usize;
        let d_state = ssm.d_state as usize;
        let n_head = ssm.n_head as usize;
        let head_dim = ssm.head_dim() as usize;
        let d_inner = ssm.d_inner as usize;
        let n_group = ssm.n_group as usize;
        let n_embd = model.config.n_embd as usize;

        let mut layer_to_slot = vec![usize::MAX; n_layer];
        let mut conv = Vec::new();
        let mut ssm_states = Vec::new();
        let mut statics = Vec::new();
        let mut slot = 0usize;
        for (i, &is) in layer_is_ssm.iter().enumerate() {
            if is {
                layer_to_slot[i] = slot;
                conv.push(vec![0.0f32; (d_conv - 1) * conv_dim]);
                ssm_states.push(vec![0.0f32; n_head * head_dim * d_state]);

                let layer = model.layers.get(i).ok_or_else(|| {
                    CoreError::Other(format!("missing layer {i} for SSM statics"))
                })?;
                statics.push(load_ssm_statics(
                    model, layer, n_embd, conv_dim, d_conv, n_head, d_inner,
                )?);
                slot += 1;
            }
        }
        Ok(Self {
            n_ssm_layers: slot,
            layer_to_slot,
            conv,
            ssm: ssm_states,
            statics,
            d_conv,
            conv_dim,
            d_state,
            n_head,
            head_dim,
            d_inner,
            n_group,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    pub fn clear(&mut self) {
        for c in &mut self.conv {
            c.fill(0.0);
        }
        for s in &mut self.ssm {
            s.fill(0.0);
        }
        // Device state is zeroed (async, on the compute stream) before its next use.
        #[cfg(feature = "cuda")]
        if let Some(d) = self.device.as_mut() {
            d.request_zero();
        }
    }

    /// Pre-dequantized statics of SSM slot `slot` (the CUDA device path uploads these once).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn statics(&self, slot: usize) -> &SsmLayerStatic {
        &self.statics[slot]
    }

    /// Host conv / SSM state of `slot` (tests compare device state against these).
    #[allow(dead_code)]
    pub(crate) fn host_state(&self, slot: usize) -> (&[f32], &[f32]) {
        (&self.conv[slot], &self.ssm[slot])
    }

    /// Test-only: cache with deterministic synthetic statics (no model needed).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn synthetic_for_test(
        ssm: &SsmConfig,
        n_slots: usize,
        n_embd: usize,
        seed: u32,
    ) -> Self {
        let d_conv = ssm.d_conv as usize;
        let conv_dim = ssm.conv_dim() as usize;
        let d_state = ssm.d_state as usize;
        let n_head = ssm.n_head as usize;
        let head_dim = ssm.head_dim() as usize;
        let d_inner = ssm.d_inner as usize;
        let mut r = seed;
        let mut f = move |scale: f32, off: f32| {
            r = r.wrapping_mul(1664525).wrapping_add(1013904223);
            (((r >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * scale + off
        };
        let mut statics = Vec::new();
        for _ in 0..n_slots {
            statics.push(SsmLayerStatic {
                conv_w: (0..d_conv * conv_dim).map(|_| f(1.0, 0.0)).collect(),
                conv_b: (0..conv_dim).map(|_| f(0.2, 0.0)).collect(),
                a: (0..n_head).map(|_| f(4.0, -2.5)).collect(),
                d: (0..n_head).map(|_| f(1.0, 1.0)).collect(),
                dt_b: (0..n_head).map(|_| f(2.0, -1.0)).collect(),
                norm: (0..d_inner).map(|_| f(0.5, 1.0)).collect(),
                attn_norm: (0..n_embd).map(|_| f(0.5, 1.0)).collect(),
            });
        }
        Self {
            n_ssm_layers: n_slots,
            layer_to_slot: (0..n_slots).collect(),
            conv: vec![vec![0.0f32; (d_conv - 1) * conv_dim]; n_slots],
            ssm: vec![vec![0.0f32; n_head * head_dim * d_state]; n_slots],
            statics,
            d_conv,
            conv_dim,
            d_state,
            n_head,
            head_dim,
            d_inner,
            n_group: ssm.n_group as usize,
            #[cfg(feature = "cuda")]
            device: None,
        }
    }

    pub fn slot(&self, layer: usize) -> Option<usize> {
        let s = self.layer_to_slot.get(layer).copied()?;
        if s == usize::MAX {
            None
        } else {
            Some(s)
        }
    }
}

fn load_ssm_statics(
    model: &LlamaModel,
    layer: &LayerWeights,
    n_embd: usize,
    conv_dim: usize,
    d_conv: usize,
    n_head: usize,
    d_inner: usize,
) -> Result<SsmLayerStatic> {
    let mut attn_norm = vec![0.0f32; n_embd];
    load_f32(model, &layer.attn_norm, &mut attn_norm)?;

    let conv_w_t = layer
        .ssm_conv1d
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("ssm_conv1d".into()))?;
    // GGUF layout is [channel, d_conv]; repack to [d_conv, conv_dim] for SIMD.
    let mut conv_w_ch_major = vec![0.0f32; conv_w_t.n_elements()];
    load_f32(model, conv_w_t, &mut conv_w_ch_major)?;
    if conv_w_ch_major.len() < conv_dim * d_conv {
        return Err(CoreError::Shape(format!(
            "ssm_conv1d len {} < conv_dim {conv_dim} * d_conv {d_conv}",
            conv_w_ch_major.len()
        )));
    }
    let mut conv_w = vec![0.0f32; d_conv * conv_dim];
    for c in 0..conv_dim {
        for k in 0..d_conv {
            conv_w[k * conv_dim + c] = conv_w_ch_major[c * d_conv + k];
        }
    }

    let mut conv_b = vec![0.0f32; conv_dim];
    if let Some(ref b) = layer.ssm_conv1d_b {
        load_f32(model, b, &mut conv_b)?;
    }

    let mut a = vec![0.0f32; n_head];
    let mut d = vec![0.0f32; n_head];
    let mut dt_b = vec![0.0f32; n_head];
    load_f32(
        model,
        layer
            .ssm_a
            .as_ref()
            .ok_or_else(|| CoreError::MissingTensor("ssm_a".into()))?,
        &mut a,
    )?;
    load_f32(
        model,
        layer
            .ssm_d
            .as_ref()
            .ok_or_else(|| CoreError::MissingTensor("ssm_d".into()))?,
        &mut d,
    )?;
    load_f32(
        model,
        layer
            .ssm_dt_b
            .as_ref()
            .ok_or_else(|| CoreError::MissingTensor("ssm_dt.bias".into()))?,
        &mut dt_b,
    )?;

    let mut norm = vec![0.0f32; d_inner];
    load_f32(
        model,
        layer
            .ssm_norm
            .as_ref()
            .ok_or_else(|| CoreError::MissingTensor("ssm_norm".into()))?,
        &mut norm,
    )?;

    Ok(SsmLayerStatic {
        conv_w,
        conv_b,
        a,
        d,
        dt_b,
        norm,
        attn_norm,
    })
}

/// One Mamba-2 decoder step for a single token.
///
/// `x` is residual stream in/out (length n_embd). Uses `scratch` buffers.
/// Static SSM weights are read from `cache.statics` (no per-token alloc/dequant).
pub fn mamba2_step(
    model: &LlamaModel,
    layer: &LayerWeights,
    layer_idx: usize,
    cache: &mut SsmCache,
    x: &mut [f32],
    // workspace
    xb: &mut [f32],
    zx_bcdt: &mut [f32],
    xbc_out: &mut [f32],
    y: &mut [f32],
    tmp: &mut [f32],
    _norm_w: &mut [f32],
    backend: &dyn Backend,
) -> Result<()> {
    let ssm = model
        .config
        .ssm
        .ok_or_else(|| CoreError::Other("mamba2_step without SsmConfig".into()))?;
    let slot = cache
        .slot(layer_idx)
        .ok_or_else(|| CoreError::Other(format!("layer {layer_idx} is not SSM")))?;

    let n_embd = model.config.n_embd as usize;
    let d_inner = ssm.d_inner as usize;
    let n_head = ssm.n_head as usize;
    let head_dim = ssm.head_dim() as usize;
    let d_in = ssm.d_in_proj() as usize;
    let rms_eps = model.config.rms_eps;

    debug_assert_eq!(x.len(), n_embd);
    debug_assert_eq!(d_inner, n_head * head_dim);
    debug_assert!(tmp.len() >= n_head);

    // 1) RMSNorm (cached weight)
    {
        let _t = profile::time_ssm_non_mm();
        rmsnorm(
            x,
            &cache.statics[slot].attn_norm[..n_embd],
            rms_eps,
            &mut xb[..n_embd],
        );
    }

    // 2) in_proj
    let ssm_in = layer
        .ssm_in
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("ssm_in".into()))?;
    matmul(model, ssm_in, &xb[..n_embd], zx_bcdt, backend)?;
    if zx_bcdt.len() < d_in {
        return Err(CoreError::Shape(format!(
            "ssm_in out {} < d_in {d_in}",
            zx_bcdt.len()
        )));
    }

    // 3..8) conv / scan / swiglu / group-norm (non-matmul)
    {
        let _t = profile::time_ssm_non_mm();
        mamba2_mix_host(cache, layer_idx, &model.config, zx_bcdt, xbc_out, y, tmp)?;
    }

    // 9) out proj
    let ssm_out = layer
        .ssm_out
        .as_ref()
        .ok_or_else(|| CoreError::MissingTensor("ssm_out".into()))?;
    matmul(model, ssm_out, y, &mut xb[..n_embd], backend)?;

    // 10) residual
    {
        let _t = profile::time_ssm_non_mm();
        add_inplace_fast(x, &xb[..n_embd]);
    }
    Ok(())
}

/// Mamba-2 mixer, steps 3..8 of [`mamba2_step`] (conv1d + SiLU, dt softplus, selective
/// scan, SwiGLU gate, grouped RMSNorm) for ONE token, factored out unchanged so the CPU
/// path stays bit-identical. Also the reference for the CUDA `ssm_mix` hook.
///
/// `zx` = in_proj output `[z (d_inner) | xBC (conv_dim) | dt (n_head)]`; updates the conv and
/// SSM state of `layer_idx`; writes `y[..d_inner]`. Workspace: `xbc_out >= conv_dim`,
/// `tmp >= n_head`.
pub(crate) fn mamba2_mix_host(
    cache: &mut SsmCache,
    layer_idx: usize,
    cfg: &ModelConfig,
    zx: &[f32],
    xbc_out: &mut [f32],
    y: &mut [f32],
    tmp: &mut [f32],
) -> Result<()> {
    let ssm = cfg
        .ssm
        .ok_or_else(|| CoreError::Other("mamba2_mix_host without SsmConfig".into()))?;
    let slot = cache
        .slot(layer_idx)
        .ok_or_else(|| CoreError::Other(format!("layer {layer_idx} is not SSM")))?;
    let d_inner = ssm.d_inner as usize;
    let d_state = ssm.d_state as usize;
    let n_head = ssm.n_head as usize;
    let head_dim = ssm.head_dim() as usize;
    let n_group = ssm.n_group as usize;
    let d_conv = ssm.d_conv as usize;
    let conv_dim = ssm.conv_dim() as usize;
    let rms_eps = cfg.rms_eps;
    if zx.len() < ssm.d_in_proj() as usize {
        return Err(CoreError::Shape(format!(
            "ssm mix zx {} < d_in {}",
            zx.len(),
            ssm.d_in_proj()
        )));
    }
    debug_assert_eq!(d_inner, n_head * head_dim);
    debug_assert!(tmp.len() >= n_head);

    // Disjoint field borrows: statics + conv/ssm state.
    let SsmCache {
        statics,
        conv: conv_states,
        ssm: ssm_states,
        ..
    } = cache;
    let statics = &statics[slot];

    // Layout: [z | xBC | dt] with sizes d_inner, conv_dim, n_head
    let (z, rest) = zx.split_at(d_inner);
    let (xbc_in, dt) = rest.split_at(conv_dim);

    {
        // 3) Causal depthwise conv over conv_dim channels (cached F32, [d_conv][conv_dim])
        let w_f32 = &statics.conv_w;
        let bias = &statics.conv_b;
        let conv_state = &mut conv_states[slot];

        causal_conv1d_step(d_conv, conv_dim, w_f32, bias, conv_state, xbc_in, xbc_out);

        // shift state left and append current input
        if d_conv > 1 {
            if d_conv == 4 {
                let (s0, rest) = conv_state.split_at_mut(conv_dim);
                let (s1, s2) = rest.split_at_mut(conv_dim);
                s0.copy_from_slice(s1);
                s1.copy_from_slice(s2);
                s2.copy_from_slice(&xbc_in[..conv_dim]);
            } else {
                for c in 0..conv_dim {
                    for k in 0..(d_conv - 2) {
                        conv_state[k * conv_dim + c] = conv_state[(k + 1) * conv_dim + c];
                    }
                    conv_state[(d_conv - 2) * conv_dim + c] = xbc_in[c];
                }
            }
        }

        // 4) Split x, B, C from xbc_out
        let (x_ssm, rest) = xbc_out.split_at(d_inner);
        let (b_all, c_all) = rest.split_at(n_group * d_state);

        // 5) dt bias + softplus into `tmp` (scratch, size >= n_head)
        let dt_sp = &mut tmp[..n_head];
        for h in 0..n_head {
            dt_sp[h] = softplus(dt[h] + statics.dt_b[h]);
        }

        let a = &statics.a;
        let d_vec = &statics.d;

        // 6) SSM scan one step (SIMD + optional parallel heads)
        let ssm_state = &mut ssm_states[slot];
        let heads_per_group = n_head / n_group;
        selective_scan_step(
            ssm_state,
            x_ssm,
            b_all,
            c_all,
            &mut y[..d_inner],
            dt_sp,
            a,
            d_vec,
            n_head,
            head_dim,
            d_state,
            heads_per_group,
        );

        // 7) SwiGLU: silu(z) * y
        swiglu_mul(z, &mut y[..d_inner]);

        // 8) Grouped RMSNorm (cached weight)
        let group_dim = d_inner / n_group;
        let sn = &statics.norm;
        for g in 0..n_group {
            let base = g * group_dim;
            grouped_rmsnorm_inplace(
                &mut y[base..base + group_dim],
                &sn[base..base + group_dim],
                rms_eps,
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Causal conv1d (depthwise) — d_conv==4 fast path with AVX
// ---------------------------------------------------------------------------

fn causal_conv1d_step(
    d_conv: usize,
    conv_dim: usize,
    w: &[f32],
    bias: &[f32],
    conv_state: &[f32],
    xbc_in: &[f32],
    xbc_out: &mut [f32],
) {
    debug_assert_eq!(w.len(), d_conv * conv_dim);
    debug_assert_eq!(bias.len(), conv_dim);
    debug_assert_eq!(xbc_in.len(), conv_dim);
    debug_assert!(xbc_out.len() >= conv_dim);

    if d_conv == 4 {
        let w0 = &w[0..conv_dim];
        let w1 = &w[conv_dim..2 * conv_dim];
        let w2 = &w[2 * conv_dim..3 * conv_dim];
        let w3 = &w[3 * conv_dim..4 * conv_dim];
        let s0 = &conv_state[0..conv_dim];
        let s1 = &conv_state[conv_dim..2 * conv_dim];
        let s2 = &conv_state[2 * conv_dim..3 * conv_dim];

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let feat = cpu_features();
            if feat.avx512f_fma() && conv_dim >= 16 {
                // SAFETY: AVX-512 F/BW/VL and FMA (the kernel enables avx512f and fma) were
                // detected at run time by `cpu_features()`.
                unsafe {
                    causal_conv4_avx512(
                        conv_dim, w0, w1, w2, w3, s0, s1, s2, xbc_in, bias, xbc_out,
                    );
                }
                silu_inplace_slice(&mut xbc_out[..conv_dim]);
                return;
            }
            if feat.avx2_fma() && conv_dim >= 8 {
                // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
                // `cpu_features()`.
                unsafe {
                    causal_conv4_avx2_fma(
                        conv_dim, w0, w1, w2, w3, s0, s1, s2, xbc_in, bias, xbc_out,
                    );
                }
                silu_inplace_slice(&mut xbc_out[..conv_dim]);
                return;
            }
        }

        for c in 0..conv_dim {
            let sum = s0[c] * w0[c] + s1[c] * w1[c] + s2[c] * w2[c] + xbc_in[c] * w3[c] + bias[c];
            xbc_out[c] = silu(sum);
        }
        return;
    }

    // Generic d_conv
    for c in 0..conv_dim {
        let mut sum = 0.0f32;
        for k in 0..(d_conv - 1) {
            sum += conv_state[k * conv_dim + c] * w[k * conv_dim + c];
        }
        sum += xbc_in[c] * w[(d_conv - 1) * conv_dim + c];
        sum += bias[c];
        xbc_out[c] = silu(sum);
    }
}

#[inline]
fn silu_inplace_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn causal_conv4_avx512(
    n: usize,
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    s0: &[f32],
    s1: &[f32],
    s2: &[f32],
    xin: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut c = 0usize;
    while c + 16 <= n {
        let mut acc = _mm512_loadu_ps(bias.as_ptr().add(c));
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(s0.as_ptr().add(c)),
            _mm512_loadu_ps(w0.as_ptr().add(c)),
            acc,
        );
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(s1.as_ptr().add(c)),
            _mm512_loadu_ps(w1.as_ptr().add(c)),
            acc,
        );
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(s2.as_ptr().add(c)),
            _mm512_loadu_ps(w2.as_ptr().add(c)),
            acc,
        );
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(xin.as_ptr().add(c)),
            _mm512_loadu_ps(w3.as_ptr().add(c)),
            acc,
        );
        _mm512_storeu_ps(out.as_mut_ptr().add(c), acc);
        c += 16;
    }
    while c < n {
        out[c] = s0[c] * w0[c] + s1[c] * w1[c] + s2[c] * w2[c] + xin[c] * w3[c] + bias[c];
        c += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn causal_conv4_avx2_fma(
    n: usize,
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    s0: &[f32],
    s1: &[f32],
    s2: &[f32],
    xin: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut c = 0usize;
    while c + 8 <= n {
        let mut acc = _mm256_loadu_ps(bias.as_ptr().add(c));
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(s0.as_ptr().add(c)),
            _mm256_loadu_ps(w0.as_ptr().add(c)),
            acc,
        );
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(s1.as_ptr().add(c)),
            _mm256_loadu_ps(w1.as_ptr().add(c)),
            acc,
        );
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(s2.as_ptr().add(c)),
            _mm256_loadu_ps(w2.as_ptr().add(c)),
            acc,
        );
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(xin.as_ptr().add(c)),
            _mm256_loadu_ps(w3.as_ptr().add(c)),
            acc,
        );
        _mm256_storeu_ps(out.as_mut_ptr().add(c), acc);
        c += 8;
    }
    while c < n {
        out[c] = s0[c] * w0[c] + s1[c] * w1[c] + s2[c] * w2[c] + xin[c] * w3[c] + bias[c];
        c += 1;
    }
}

// ---------------------------------------------------------------------------
// Selective scan — SIMD over d_state, parallel over heads via persistent pool
// ---------------------------------------------------------------------------

fn selective_scan_step(
    ssm_state: &mut [f32],
    x_ssm: &[f32],
    b_all: &[f32],
    c_all: &[f32],
    y: &mut [f32],
    dt_sp: &[f32],
    a: &[f32],
    d_vec: &[f32],
    n_head: usize,
    head_dim: usize,
    d_state: usize,
    heads_per_group: usize,
) {
    // Prefer persistent pool when enough heads to amortize dispatch.
    let n_workers = ssm_pool_workers();
    if n_head >= 8 && n_workers > 1 {
        parallel_scan_heads(
            ssm_state,
            x_ssm,
            b_all,
            c_all,
            y,
            dt_sp,
            a,
            d_vec,
            n_head,
            head_dim,
            d_state,
            heads_per_group,
            n_workers,
        );
    } else {
        for h in 0..n_head {
            scan_one_head(
                h,
                ssm_state,
                x_ssm,
                b_all,
                c_all,
                y,
                dt_sp,
                a,
                d_vec,
                head_dim,
                d_state,
                heads_per_group,
            );
        }
    }
}

fn scan_one_head(
    h: usize,
    ssm_state: &mut [f32],
    x_ssm: &[f32],
    b_all: &[f32],
    c_all: &[f32],
    y: &mut [f32],
    dt_sp: &[f32],
    a: &[f32],
    d_vec: &[f32],
    head_dim: usize,
    d_state: usize,
    heads_per_group: usize,
) {
    let d_a = (dt_sp[h] * a[h]).exp();
    let g = h / heads_per_group;
    let b_g = &b_all[g * d_state..(g + 1) * d_state];
    let c_g = &c_all[g * d_state..(g + 1) * d_state];
    let dt = dt_sp[h];
    let d_skip = d_vec[h];
    let state_off = h * head_dim * d_state;
    let y_off = h * head_dim;
    let state = &mut ssm_state[state_off..state_off + head_dim * d_state];
    let y_h = &mut y[y_off..y_off + head_dim];
    let x_h = &x_ssm[y_off..y_off + head_dim];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if feat.avx512f_fma() && d_state >= 16 {
            // SAFETY: AVX-512 F/BW/VL and FMA (the kernel enables avx512f and fma) were detected at
            // run time by `cpu_features()`.
            unsafe {
                scan_head_avx512(
                    state, x_h, b_g, c_g, y_h, d_a, dt, d_skip, head_dim, d_state,
                );
            }
            return;
        }
        if feat.avx2_fma() && d_state >= 8 {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`.
            unsafe {
                scan_head_avx2_fma(
                    state, x_h, b_g, c_g, y_h, d_a, dt, d_skip, head_dim, d_state,
                );
            }
            return;
        }
    }

    scan_head_scalar(
        state, x_h, b_g, c_g, y_h, d_a, dt, d_skip, head_dim, d_state,
    );
}

fn scan_head_scalar(
    state: &mut [f32],
    x_h: &[f32],
    b_g: &[f32],
    c_g: &[f32],
    y_h: &mut [f32],
    d_a: f32,
    dt: f32,
    d_skip: f32,
    head_dim: usize,
    d_state: usize,
) {
    for i1 in 0..head_dim {
        let x_dt = x_h[i1] * dt;
        let state_base = i1 * d_state;
        let mut sumf = 0.0f32;
        let mut i0 = 0;
        while i0 + 4 <= d_state {
            let s0 = state[state_base + i0] * d_a + b_g[i0] * x_dt;
            let s1 = state[state_base + i0 + 1] * d_a + b_g[i0 + 1] * x_dt;
            let s2 = state[state_base + i0 + 2] * d_a + b_g[i0 + 2] * x_dt;
            let s3 = state[state_base + i0 + 3] * d_a + b_g[i0 + 3] * x_dt;
            state[state_base + i0] = s0;
            state[state_base + i0 + 1] = s1;
            state[state_base + i0 + 2] = s2;
            state[state_base + i0 + 3] = s3;
            sumf += s0 * c_g[i0] + s1 * c_g[i0 + 1] + s2 * c_g[i0 + 2] + s3 * c_g[i0 + 3];
            i0 += 4;
        }
        while i0 < d_state {
            let s = state[state_base + i0] * d_a + b_g[i0] * x_dt;
            sumf += s * c_g[i0];
            state[state_base + i0] = s;
            i0 += 1;
        }
        y_h[i1] = sumf + x_h[i1] * d_skip;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn scan_head_avx512(
    state: &mut [f32],
    x_h: &[f32],
    b_g: &[f32],
    c_g: &[f32],
    y_h: &mut [f32],
    d_a: f32,
    dt: f32,
    d_skip: f32,
    head_dim: usize,
    d_state: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let v_da = _mm512_set1_ps(d_a);
    for i1 in 0..head_dim {
        let x_dt = x_h[i1] * dt;
        let v_xdt = _mm512_set1_ps(x_dt);
        let state_base = i1 * d_state;
        let mut acc = _mm512_setzero_ps();
        let mut i0 = 0usize;
        while i0 + 16 <= d_state {
            let s = _mm512_loadu_ps(state.as_ptr().add(state_base + i0));
            let vb = _mm512_loadu_ps(b_g.as_ptr().add(i0));
            let vc = _mm512_loadu_ps(c_g.as_ptr().add(i0));
            // s_new = d_a * s + x_dt * b
            let s_new = _mm512_fmadd_ps(vb, v_xdt, _mm512_mul_ps(s, v_da));
            _mm512_storeu_ps(state.as_mut_ptr().add(state_base + i0), s_new);
            acc = _mm512_fmadd_ps(s_new, vc, acc);
            i0 += 16;
        }
        let mut sumf = hsum16_avx512(acc);
        while i0 < d_state {
            let s = state[state_base + i0] * d_a + b_g[i0] * x_dt;
            sumf += s * c_g[i0];
            state[state_base + i0] = s;
            i0 += 1;
        }
        y_h[i1] = sumf + x_h[i1] * d_skip;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn hsum16_avx512(v: std::arch::x86_64::__m512) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut buf = [0.0f32; 16];
    _mm512_storeu_ps(buf.as_mut_ptr(), v);
    let mut s = 0.0f32;
    for x in buf {
        s += x;
    }
    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_head_avx2_fma(
    state: &mut [f32],
    x_h: &[f32],
    b_g: &[f32],
    c_g: &[f32],
    y_h: &mut [f32],
    d_a: f32,
    dt: f32,
    d_skip: f32,
    head_dim: usize,
    d_state: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let v_da = _mm256_set1_ps(d_a);
    for i1 in 0..head_dim {
        let x_dt = x_h[i1] * dt;
        let v_xdt = _mm256_set1_ps(x_dt);
        let state_base = i1 * d_state;
        let mut acc = _mm256_setzero_ps();
        let mut i0 = 0usize;
        while i0 + 8 <= d_state {
            let s = _mm256_loadu_ps(state.as_ptr().add(state_base + i0));
            let vb = _mm256_loadu_ps(b_g.as_ptr().add(i0));
            let vc = _mm256_loadu_ps(c_g.as_ptr().add(i0));
            let s_new = _mm256_fmadd_ps(vb, v_xdt, _mm256_mul_ps(s, v_da));
            _mm256_storeu_ps(state.as_mut_ptr().add(state_base + i0), s_new);
            acc = _mm256_fmadd_ps(s_new, vc, acc);
            i0 += 8;
        }
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        let mut sumf =
            (buf[0] + buf[1]) + (buf[2] + buf[3]) + (buf[4] + buf[5]) + (buf[6] + buf[7]);
        while i0 < d_state {
            let s = state[state_base + i0] * d_a + b_g[i0] * x_dt;
            sumf += s * c_g[i0];
            state[state_base + i0] = s;
            i0 += 1;
        }
        y_h[i1] = sumf + x_h[i1] * d_skip;
    }
}

// ---------------------------------------------------------------------------
// Persistent SSM scan worker pool (avoids per-layer thread::scope spawn)
// ---------------------------------------------------------------------------

struct SsmScanJob {
    /// Inclusive head range [h0, h1).
    h0: usize,
    h1: usize,
    ssm_state: *mut f32,
    ssm_state_len: usize,
    x_ssm: *const f32,
    x_ssm_len: usize,
    b_all: *const f32,
    b_all_len: usize,
    c_all: *const f32,
    c_all_len: usize,
    y: *mut f32,
    y_len: usize,
    dt_sp: *const f32,
    dt_sp_len: usize,
    a: *const f32,
    a_len: usize,
    d_vec: *const f32,
    d_vec_len: usize,
    head_dim: usize,
    d_state: usize,
    heads_per_group: usize,
    remaining: *const AtomicUsize,
    /// Set by a worker whose job panicked; the submitter re-panics after the wait.
    panicked: *const AtomicBool,
}

// SAFETY: submitter keeps all buffers alive until remaining hits 0 (a `WaitForJobs` guard also
// covers a panic in its own heads); each job writes a disjoint head range of ssm_state / y,
// though `run_ssm_job` rebuilds full-length `&mut` slices (see the FIXME there).
unsafe impl Send for SsmScanJob {}

struct SsmPoolState {
    jobs: Mutex<Vec<SsmScanJob>>,
    cv: Condvar,
}

struct SsmPool {
    state: &'static SsmPoolState,
    _handles: Vec<JoinHandle<()>>,
    n_workers: usize,
}

fn ssm_global_pool() -> &'static SsmPool {
    static POOL: OnceLock<SsmPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(8) // scan is BW-heavy; 8 is enough for 96 heads
            .max(1);
        let state: &'static SsmPoolState = Box::leak(Box::new(SsmPoolState {
            jobs: Mutex::new(Vec::new()),
            cv: Condvar::new(),
        }));
        let mut handles = Vec::with_capacity(n);
        for id in 0..n {
            handles.push(
                thread::Builder::new()
                    .name(format!("artificer-ssm-{id}"))
                    .spawn(move || ssm_worker_loop(state))
                    .expect("spawn ssm worker"),
            );
        }
        SsmPool {
            state,
            _handles: handles,
            n_workers: n,
        }
    })
}

fn ssm_pool_workers() -> usize {
    ssm_global_pool().n_workers
}

fn ssm_worker_loop(state: &'static SsmPoolState) {
    loop {
        let job = {
            let mut guard = state.jobs.lock().unwrap();
            loop {
                if let Some(j) = guard.pop() {
                    break j;
                }
                guard = state.cv.wait(guard).unwrap();
            }
        };
        let (remaining, panicked) = (job.remaining, job.panicked);
        // Same as the matmul pool: a panicking job must not kill the worker or leave `remaining`
        // above 0. AssertUnwindSafe is fine: the submitter re-panics instead of using the output.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| run_ssm_job(job))) {
            // SAFETY: the submitter keeps both alive until `remaining` reaches 0, which happens
            // only at this decrement, the job's last access (`SsmScanJob` has no Drop); the
            // Release orders the flag store before it. The payload is dropped after.
            unsafe {
                (*panicked).store(true, Ordering::Relaxed);
                (*remaining).fetch_sub(1, Ordering::Release);
            }
            drop(payload);
        }
    }
}

fn run_ssm_job(job: SsmScanJob) {
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0. This job writes
    // only heads h0..h1, but the slice covers the whole buffer while the submitter and other jobs
    // hold overlapping `&mut` slices of it, which Rust's aliasing rules do not allow.
    // FIXME(0.1.x): hand each job only its own head range (or use raw pointers throughout).
    let ssm_state = unsafe { std::slice::from_raw_parts_mut(job.ssm_state, job.ssm_state_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0; the buffer is
    // read-only during the scan.
    let x_ssm = unsafe { std::slice::from_raw_parts(job.x_ssm, job.x_ssm_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0; the buffer is
    // read-only during the scan.
    let b_all = unsafe { std::slice::from_raw_parts(job.b_all, job.b_all_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0; the buffer is
    // read-only during the scan.
    let c_all = unsafe { std::slice::from_raw_parts(job.c_all, job.c_all_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0. This job writes
    // only heads h0..h1, but the slice covers the whole buffer while the submitter and other jobs
    // hold overlapping `&mut` slices of it, which Rust's aliasing rules do not allow.
    // FIXME(0.1.x): hand each job only its own head range (or use raw pointers throughout).
    let y = unsafe { std::slice::from_raw_parts_mut(job.y, job.y_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0; the buffer is
    // read-only during the scan.
    let dt_sp = unsafe { std::slice::from_raw_parts(job.dt_sp, job.dt_sp_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0; the buffer is
    // read-only during the scan.
    let a = unsafe { std::slice::from_raw_parts(job.a, job.a_len) };
    // SAFETY: the submitter keeps the buffers alive until `remaining` reaches 0; the buffer is
    // read-only during the scan.
    let d_vec = unsafe { std::slice::from_raw_parts(job.d_vec, job.d_vec_len) };

    for h in job.h0..job.h1 {
        #[cfg(test)]
        if tests::scan_test_hook(h, job.d_state, false, y) {
            continue;
        }
        scan_one_head(
            h,
            ssm_state,
            x_ssm,
            b_all,
            c_all,
            y,
            dt_sp,
            a,
            d_vec,
            job.head_dim,
            job.d_state,
            job.heads_per_group,
        );
    }
    // SAFETY: `remaining` points at the submitter's counter, which it polls until 0; this is the
    // job's last access.
    unsafe {
        (*job.remaining).fetch_sub(1, Ordering::Release);
    }
}

fn parallel_scan_heads(
    ssm_state: &mut [f32],
    x_ssm: &[f32],
    b_all: &[f32],
    c_all: &[f32],
    y: &mut [f32],
    dt_sp: &[f32],
    a: &[f32],
    d_vec: &[f32],
    n_head: usize,
    head_dim: usize,
    d_state: usize,
    heads_per_group: usize,
    n_workers: usize,
) {
    let pool = ssm_global_pool();
    let n_threads = n_workers.min(n_head).min(pool.n_workers).max(1);
    let chunk = n_head.div_ceil(n_threads);
    let remaining = AtomicUsize::new(0);
    let panicked = AtomicBool::new(false);

    let mut jobs = Vec::new();
    // Farm [chunk..n_head] to pool; run [0..chunk] locally.
    let mut start = chunk.min(n_head);
    while start < n_head {
        let end = (start + chunk).min(n_head);
        jobs.push(SsmScanJob {
            h0: start,
            h1: end,
            ssm_state: ssm_state.as_mut_ptr(),
            ssm_state_len: ssm_state.len(),
            x_ssm: x_ssm.as_ptr(),
            x_ssm_len: x_ssm.len(),
            b_all: b_all.as_ptr(),
            b_all_len: b_all.len(),
            c_all: c_all.as_ptr(),
            c_all_len: c_all.len(),
            y: y.as_mut_ptr(),
            y_len: y.len(),
            dt_sp: dt_sp.as_ptr(),
            dt_sp_len: dt_sp.len(),
            a: a.as_ptr(),
            a_len: a.len(),
            d_vec: d_vec.as_ptr(),
            d_vec_len: d_vec.len(),
            head_dim,
            d_state,
            heads_per_group,
            remaining: &remaining,
            panicked: &panicked,
        });
        start = end;
    }
    remaining.store(jobs.len(), Ordering::Release);
    if !jobs.is_empty() {
        {
            let mut guard = pool.state.jobs.lock().unwrap();
            guard.extend(jobs);
        }
        pool.state.cv.notify_all();
    }
    // Declared after `remaining` and `panicked`, so it drops (and waits) before they do, also when
    // a local `scan_one_head` panics.
    let wait = WaitForJobs(&remaining);

    for h in 0..chunk.min(n_head) {
        #[cfg(test)]
        if tests::scan_test_hook(h, d_state, true, y) {
            continue;
        }
        scan_one_head(
            h,
            ssm_state,
            x_ssm,
            b_all,
            c_all,
            y,
            dt_sp,
            a,
            d_vec,
            head_dim,
            d_state,
            heads_per_group,
        );
    }

    drop(wait);
    // Callers return (); a worker panic is a bug like a local one, so surface it the same way.
    if panicked.load(Ordering::Relaxed) {
        panic!("ssm scan worker panicked");
    }
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------

fn swiglu_mul(z: &[f32], y: &mut [f32]) {
    debug_assert_eq!(z.len(), y.len());
    let n = y.len();
    let mut i = 0;
    // silu is scalar (exp); mul can still be tight.
    while i < n {
        y[i] *= silu(z[i]);
        i += 1;
    }
}

fn grouped_rmsnorm_inplace(slice: &mut [f32], weight: &[f32], eps: f32) {
    let n = slice.len();
    debug_assert_eq!(n, weight.len());
    let ss = sum_sq_fast(slice);
    let inv = (ss / n as f32 + eps).sqrt().recip();
    scale_mul_fast(slice, weight, inv);
}

fn sum_sq_fast(x: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if feat.avx512f_fma() && x.len() >= 16 {
            // SAFETY: AVX-512 F/BW/VL and FMA (the kernel enables avx512f and fma) were detected at
            // run time by `cpu_features()`; the kernel reads only within `x`.
            return unsafe { sum_sq_avx512(x) };
        }
        if feat.avx2_fma() && x.len() >= 8 {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`; the kernel reads only within `x`.
            return unsafe { sum_sq_avx2_fma(x) };
        }
    }
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss
}

fn scale_mul_fast(x: &mut [f32], w: &[f32], scale: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if feat.avx512f_fma() && x.len() >= 16 {
            // SAFETY: AVX-512 F/BW/VL and FMA (the kernel enables avx512f and fma) were detected at
            // run time by `cpu_features()`; the kernel reads `w` up to `x.len()`, and that `w` is
            // at least that long is only debug-asserted by the caller (`grouped_rmsnorm_inplace`).
            // FIXME(0.1.x): check w.len() >= x.len() in release builds before the SIMD call.
            unsafe { scale_mul_avx512(x, w, scale) };
            return;
        }
        if feat.avx2_fma() && x.len() >= 8 {
            // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
            // `cpu_features()`; the kernel reads `w` up to `x.len()`, and that `w` is at least that
            // long is only debug-asserted by the caller (`grouped_rmsnorm_inplace`).
            // FIXME(0.1.x): check w.len() >= x.len() in release builds before the SIMD call.
            unsafe { scale_mul_avx2_fma(x, w, scale) };
            return;
        }
    }
    for i in 0..x.len() {
        x[i] = x[i] * scale * w[i];
    }
}

fn add_inplace_fast(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let feat = cpu_features();
        if feat.avx512f && a.len() >= 16 {
            // SAFETY: the target features this kernel enables were detected at run time by
            // `cpu_features()`; the kernel reads `b` up to `a.len()`; equal lengths are only
            // debug-asserted above.
            // FIXME(0.1.x): check b.len() >= a.len() in release builds before the SIMD call.
            unsafe { add_inplace_avx512(a, b) };
            return;
        }
        if feat.avx2 && a.len() >= 8 {
            // SAFETY: the target features this kernel enables were detected at run time by
            // `cpu_features()`; the kernel reads `b` up to `a.len()`; equal lengths are only
            // debug-asserted above.
            // FIXME(0.1.x): check b.len() >= a.len() in release builds before the SIMD call.
            unsafe { add_inplace_avx2(a, b) };
            return;
        }
    }
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn sum_sq_avx512(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= x.len() {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        acc = _mm512_fmadd_ps(v, v, acc);
        i += 16;
    }
    let mut ss = hsum16_avx512(acc);
    while i < x.len() {
        ss += x[i] * x[i];
        i += 1;
    }
    ss
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_avx2_fma(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc = _mm256_fmadd_ps(v, v, acc);
        i += 8;
    }
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut ss = (buf[0] + buf[1]) + (buf[2] + buf[3]) + (buf[4] + buf[5]) + (buf[6] + buf[7]);
    while i < x.len() {
        ss += x[i] * x[i];
        i += 1;
    }
    ss
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn scale_mul_avx512(x: &mut [f32], w: &[f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let vs = _mm512_set1_ps(scale);
    let mut i = 0;
    while i + 16 <= x.len() {
        let vx = _mm512_loadu_ps(x.as_ptr().add(i));
        let vw = _mm512_loadu_ps(w.as_ptr().add(i));
        let r = _mm512_mul_ps(_mm512_mul_ps(vx, vs), vw);
        _mm512_storeu_ps(x.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < x.len() {
        x[i] = x[i] * scale * w[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_mul_avx2_fma(x: &mut [f32], w: &[f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let vs = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= x.len() {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(w.as_ptr().add(i));
        let r = _mm256_mul_ps(_mm256_mul_ps(vx, vs), vw);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < x.len() {
        x[i] = x[i] * scale * w[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn add_inplace_avx512(a: &mut [f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        _mm512_storeu_ps(a.as_mut_ptr().add(i), _mm512_add_ps(va, vb));
        i += 16;
    }
    while i < a.len() {
        a[i] += b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_inplace_avx2(a: &mut [f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(a.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    while i < a.len() {
        a[i] += b[i];
        i += 1;
    }
}

fn load_f32(model: &LlamaModel, w: &WeightTensor, out: &mut [f32]) -> Result<()> {
    let data = model.tensor_bytes(w)?;
    let n = w.n_elements();
    if out.len() < n {
        return Err(CoreError::Shape(format!(
            "load_f32 buffer {} < {n} for {}",
            out.len(),
            w.name
        )));
    }
    crate::quant::dequantize(data, w.ty, 0, n, &mut out[..n])
}

fn matmul(
    model: &LlamaModel,
    w: &WeightTensor,
    x: &[f32],
    out: &mut [f32],
    backend: &dyn Backend,
) -> Result<()> {
    let data = model.tensor_bytes(w)?;
    let k = w.k();
    let n = w.n();
    if out.len() < n {
        return Err(CoreError::Shape(format!(
            "matmul out {} < n {n} for {}",
            out.len(),
            w.name
        )));
    }
    let _role = profile::time_role(&w.name);
    backend.matmul_vec(data, w.ty, k, n, x, &mut out[..n])
}
/// Test-only naive f64 reference of the Mamba-2 mixer over `n_tok` rows.
/// Written straight from the recurrence, independent of the SIMD / pooled host path and of the
/// device kernels: causal depthwise conv + SiLU, dt = softplus(dt + dt_b), dA = exp(dt·A),
/// h ← h·dA + B·x·dt, y = C·h + D·x, y *= silu(z), grouped RMSNorm. Starts from the conv / SSM
/// state of `slot` in `cache` (not modified). Returns y `[n_tok][d_inner]` and the final conv /
/// SSM state (all rounded to f32 once at the end).
#[cfg(test)]
pub(crate) struct Mamba2RefOut {
    pub(crate) y: Vec<f32>,
    pub(crate) conv: Vec<f32>,
    pub(crate) ssm: Vec<f32>,
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // written as the textbook loops on purpose
pub(crate) fn mamba2_mix_reference_f64(
    cache: &SsmCache,
    slot: usize,
    ssm: &SsmConfig,
    eps: f32,
    zx: &[f32],
    n_tok: usize,
) -> Mamba2RefOut {
    let d_inner = ssm.d_inner as usize;
    let d_state = ssm.d_state as usize;
    let n_head = ssm.n_head as usize;
    let head_dim = ssm.head_dim() as usize;
    let n_group = ssm.n_group as usize;
    let d_conv = ssm.d_conv as usize;
    let conv_dim = ssm.conv_dim() as usize;
    let d_in = ssm.d_in_proj() as usize;
    let dm1 = d_conv - 1;
    assert!(zx.len() >= n_tok * d_in);
    let st = &cache.statics[slot];
    let silu = |x: f64| x / (1.0 + (-x).exp());
    let softplus = |x: f64| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    // conv window: hist[k][c] = input k steps ago... kept as [dm1][conv_dim], oldest first.
    let mut conv: Vec<f64> = cache.conv[slot].iter().map(|&v| v as f64).collect();
    let mut h: Vec<f64> = cache.ssm[slot].iter().map(|&v| v as f64).collect();
    let mut y_all = vec![0.0f32; n_tok * d_inner];
    let mut xbc = vec![0.0f64; conv_dim];
    let mut y = vec![0.0f64; d_inner];
    for t in 0..n_tok {
        let row = &zx[t * d_in..(t + 1) * d_in];
        let z = &row[..d_inner];
        let xin = &row[d_inner..d_inner + conv_dim];
        let dt_raw = &row[d_inner + conv_dim..d_inner + conv_dim + n_head];
        for c in 0..conv_dim {
            let mut acc = st.conv_b[c] as f64;
            for k in 0..dm1 {
                acc += conv[k * conv_dim + c] * st.conv_w[k * conv_dim + c] as f64;
            }
            acc += xin[c] as f64 * st.conv_w[dm1 * conv_dim + c] as f64;
            xbc[c] = silu(acc);
        }
        for c in 0..conv_dim {
            for k in 0..dm1 {
                conv[k * conv_dim + c] = if k + 1 < dm1 {
                    conv[(k + 1) * conv_dim + c]
                } else {
                    xin[c] as f64
                };
            }
        }
        let x = &xbc[..d_inner];
        let b_all = &xbc[d_inner..d_inner + n_group * d_state];
        let c_all = &xbc[d_inner + n_group * d_state..];
        let hpg = n_head / n_group;
        for hd in 0..n_head {
            let dt = softplus(dt_raw[hd] as f64 + st.dt_b[hd] as f64);
            let da = (dt * st.a[hd] as f64).exp();
            let g = hd / hpg;
            for p in 0..head_dim {
                let r = hd * head_dim + p;
                let mut acc = 0.0f64;
                for n in 0..d_state {
                    let s = &mut h[r * d_state + n];
                    *s = *s * da + b_all[g * d_state + n] * x[r] * dt;
                    acc += *s * c_all[g * d_state + n];
                }
                y[r] = acc + x[r] * st.d[hd] as f64;
            }
        }
        for i in 0..d_inner {
            y[i] *= silu(z[i] as f64);
        }
        let gd = d_inner / n_group;
        for g in 0..n_group {
            let seg = &mut y[g * gd..(g + 1) * gd];
            let ms = seg.iter().map(|v| v * v).sum::<f64>() / gd as f64;
            let inv = 1.0 / (ms + eps as f64).sqrt();
            for (i, v) in seg.iter_mut().enumerate() {
                *v *= inv * st.norm[g * gd + i] as f64;
            }
        }
        for i in 0..d_inner {
            y_all[t * d_inner + i] = y[i] as f32;
        }
    }
    Mamba2RefOut {
        y: y_all,
        conv: conv.iter().map(|&v| v as f32).collect(),
        ssm: h.iter().map(|&v| v as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test-only scan modes for `parallel_scan_heads`, selected by a `d_state` no model uses, so
    // they cannot affect other tests running at the same time.
    /// Local heads panic; farmed heads sleep 200 ms, set `y[h] = 1` and bump the counter.
    const HOOK_LOCAL_PANIC: usize = usize::MAX - 1;
    /// Farmed heads panic; local heads do nothing.
    const HOOK_WORKER_PANIC: usize = usize::MAX - 2;
    /// Every head sets `y[h] = 2`.
    const HOOK_FAST: usize = usize::MAX - 3;
    static HOOK_LOCAL_PANIC_DONE: AtomicUsize = AtomicUsize::new(0);

    /// Called for every head before the real scan; `true` means the hook handled the head.
    pub(super) fn scan_test_hook(h: usize, d_state: usize, local: bool, y: &mut [f32]) -> bool {
        match d_state {
            HOOK_LOCAL_PANIC => {
                if local {
                    panic!("injected local scan panic");
                }
                thread::sleep(std::time::Duration::from_millis(200));
                y[h] = 1.0;
                HOOK_LOCAL_PANIC_DONE.fetch_add(1, Ordering::SeqCst);
                true
            }
            HOOK_WORKER_PANIC => {
                if !local {
                    panic!("injected ssm worker panic");
                }
                true
            }
            HOOK_FAST => {
                y[h] = 2.0;
                true
            }
            _ => false,
        }
    }

    fn scan_with_hook(d_state: usize, y: &mut [f32]) {
        let n_head = y.len();
        let mut state = [0.0f32; 1];
        let e: [f32; 0] = [];
        let n_workers = ssm_pool_workers();
        parallel_scan_heads(
            &mut state, &e, &e, &e, y, &e, &e, &e, n_head, 1, d_state, 1, n_workers,
        );
    }

    /// Runs `scan_with_hook` on its own thread; `None` on a 10 s timeout (a hang).
    fn scan_with_timeout(d_state: usize) -> Option<std::thread::Result<Vec<f32>>> {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let mut y = vec![0.0f32; 32];
            let r = catch_unwind(AssertUnwindSafe(|| scan_with_hook(d_state, &mut y)));
            let _ = tx.send(r.map(|()| y));
        });
        rx.recv_timeout(std::time::Duration::from_secs(10)).ok()
    }

    /// A panic in the submitter's own heads must not unwind out of `parallel_scan_heads` while
    /// farmed jobs still point at its buffers and counters.
    #[test]
    fn ssm_local_panic_waits_for_queued_jobs() {
        let n_workers = ssm_pool_workers();
        assert!(n_workers >= 2, "test needs at least 2 ssm workers");
        let n_head: usize = 32;
        let farmed = n_head - n_head.div_ceil(n_workers.min(n_head));
        let mut y = vec![0.0f32; n_head];
        let r = catch_unwind(AssertUnwindSafe(|| {
            scan_with_hook(HOOK_LOCAL_PANIC, &mut y)
        }));
        // Read before anything else: the unwind must already have waited for every farmed head.
        let done_at_return = HOOK_LOCAL_PANIC_DONE.load(Ordering::SeqCst);
        assert!(r.is_err(), "the injected panic must propagate");
        assert!(farmed >= 1);
        assert_eq!(done_at_return, farmed);
        let snapshot = y.clone();
        thread::sleep(std::time::Duration::from_millis(400));
        assert_eq!(y, snapshot, "no late writes");
        assert_eq!(y.iter().filter(|&&v| v == 1.0).count(), farmed);
    }

    /// Live ssm worker threads (Linux: thread names in /proc). A worker sets its name only once it
    /// first runs, so a worker the scheduler has not started yet is not counted: poll until the
    /// count is `want` or 10 s pass, and return the last count.
    #[cfg(target_os = "linux")]
    fn live_ssm_threads(want: usize) -> usize {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let live = std::fs::read_dir("/proc/self/task")
                .unwrap()
                .filter_map(|t| std::fs::read_to_string(t.ok()?.path().join("comm")).ok())
                .filter(|c| c.starts_with("artificer-ssm-"))
                .count();
            if live == want || std::time::Instant::now() >= deadline {
                return live;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// A panicking farmed job must surface as a panic in the submitter (not a hang), every ssm
    /// worker must survive, and the pool must keep working.
    #[test]
    fn ssm_worker_panic_propagates_and_pool_survives() {
        assert!(ssm_pool_workers() >= 2, "test needs at least 2 ssm workers");
        let r = scan_with_timeout(HOOK_WORKER_PANIC).expect("ssm scan hung after a worker panic");
        let payload = r.expect_err("a worker panic must reach the submitter");
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"ssm scan worker panicked")
        );
        #[cfg(target_os = "linux")]
        assert_eq!(live_ssm_threads(ssm_pool_workers()), ssm_pool_workers());
        let y = scan_with_timeout(HOOK_FAST)
            .expect("ssm pool hung after a worker panic")
            .expect("normal scan after the panic must succeed");
        assert!(y.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn softplus_large_is_identity() {
        assert!((softplus(25.0) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn scan_head_simd_matches_scalar() {
        let head_dim = 80;
        let d_state = 128;
        let mut state_a = vec![0.0f32; head_dim * d_state];
        let mut state_b = state_a.clone();
        // deterministic fill
        for (i, v) in state_a.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.01 - 0.05;
        }
        state_b.copy_from_slice(&state_a);
        let x_h: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.001 - 0.02).collect();
        let b_g: Vec<f32> = (0..d_state)
            .map(|i| ((i % 9) as f32) * 0.02 - 0.05)
            .collect();
        let c_g: Vec<f32> = (0..d_state)
            .map(|i| ((i % 11) as f32) * 0.015 - 0.04)
            .collect();
        let mut y_a = vec![0.0f32; head_dim];
        let mut y_b = vec![0.0f32; head_dim];
        let d_a = (-0.3f32).exp();
        let dt = 0.05f32;
        let d_skip = 0.1f32;

        scan_head_scalar(
            &mut state_a,
            &x_h,
            &b_g,
            &c_g,
            &mut y_a,
            d_a,
            dt,
            d_skip,
            head_dim,
            d_state,
        );

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let feat = cpu_features();
            if feat.avx512f_fma() {
                // SAFETY: AVX-512 F/BW/VL and FMA (the kernel enables avx512f and fma) were
                // detected at run time by `cpu_features()`; the state, x, B, C and y buffers are
                // sized head_dim×d_state, head_dim, d_state, d_state and head_dim.
                unsafe {
                    scan_head_avx512(
                        &mut state_b,
                        &x_h,
                        &b_g,
                        &c_g,
                        &mut y_b,
                        d_a,
                        dt,
                        d_skip,
                        head_dim,
                        d_state,
                    );
                }
            } else if feat.avx2_fma() {
                // SAFETY: AVX2 and FMA, everything the kernel enables, were detected at run time by
                // `cpu_features()`; the state, x, B, C and y buffers are sized head_dim×d_state,
                // head_dim, d_state, d_state and head_dim.
                unsafe {
                    scan_head_avx2_fma(
                        &mut state_b,
                        &x_h,
                        &b_g,
                        &c_g,
                        &mut y_b,
                        d_a,
                        dt,
                        d_skip,
                        head_dim,
                        d_state,
                    );
                }
            } else {
                scan_head_scalar(
                    &mut state_b,
                    &x_h,
                    &b_g,
                    &c_g,
                    &mut y_b,
                    d_a,
                    dt,
                    d_skip,
                    head_dim,
                    d_state,
                );
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            scan_head_scalar(
                &mut state_b,
                &x_h,
                &b_g,
                &c_g,
                &mut y_b,
                d_a,
                dt,
                d_skip,
                head_dim,
                d_state,
            );
        }

        for i in 0..head_dim {
            assert!(
                (y_a[i] - y_b[i]).abs() < 1e-4,
                "y mismatch at {i}: {} vs {}",
                y_a[i],
                y_b[i]
            );
        }
        for i in 0..state_a.len() {
            assert!(
                (state_a[i] - state_b[i]).abs() < 1e-4,
                "state mismatch at {i}"
            );
        }
    }

    #[test]
    fn conv4_simd_matches_scalar_layout() {
        let conv_dim = 64;
        let d_conv = 4;
        let mut w = vec![0.0f32; d_conv * conv_dim];
        let mut bias = vec![0.0f32; conv_dim];
        let mut state = vec![0.0f32; (d_conv - 1) * conv_dim];
        let mut xin = vec![0.0f32; conv_dim];
        for i in 0..w.len() {
            w[i] = ((i % 13) as f32) * 0.01 - 0.05;
        }
        for i in 0..conv_dim {
            bias[i] = ((i % 5) as f32) * 0.02;
            xin[i] = ((i % 7) as f32) * 0.03 - 0.1;
        }
        for i in 0..state.len() {
            state[i] = ((i % 11) as f32) * 0.01 - 0.04;
        }

        let mut out_fast = vec![0.0f32; conv_dim];
        let mut out_ref = vec![0.0f32; conv_dim];
        causal_conv1d_step(d_conv, conv_dim, &w, &bias, &state, &xin, &mut out_fast);

        // scalar reference with [d_conv][conv_dim] layout
        let w0 = &w[0..conv_dim];
        let w1 = &w[conv_dim..2 * conv_dim];
        let w2 = &w[2 * conv_dim..3 * conv_dim];
        let w3 = &w[3 * conv_dim..4 * conv_dim];
        let s0 = &state[0..conv_dim];
        let s1 = &state[conv_dim..2 * conv_dim];
        let s2 = &state[2 * conv_dim..3 * conv_dim];
        for c in 0..conv_dim {
            let sum = s0[c] * w0[c] + s1[c] * w1[c] + s2[c] * w2[c] + xin[c] * w3[c] + bias[c];
            out_ref[c] = silu(sum);
        }
        for c in 0..conv_dim {
            assert!(
                (out_fast[c] - out_ref[c]).abs() < 1e-5,
                "conv mismatch at {c}: {} vs {}",
                out_fast[c],
                out_ref[c]
            );
        }
    }

    #[test]
    fn conv_weight_repack_roundtrip_index() {
        // channel-major [c, k] → [k, c]
        let conv_dim = 8;
        let d_conv = 4;
        let mut ch_major = vec![0.0f32; conv_dim * d_conv];
        for c in 0..conv_dim {
            for k in 0..d_conv {
                ch_major[c * d_conv + k] = (c * 10 + k) as f32;
            }
        }
        let mut packed = vec![0.0f32; d_conv * conv_dim];
        for c in 0..conv_dim {
            for k in 0..d_conv {
                packed[k * conv_dim + c] = ch_major[c * d_conv + k];
            }
        }
        assert_eq!(packed[3], 30.0); // k=0,c=3
        assert_eq!(packed[2 * conv_dim + 1], 12.0); // k=2,c=1
    }

    // ---- f64 reference vs the host path, multi-token ----

    fn lcg_vec(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((*seed >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5) * scale
            })
            .collect()
    }

    fn ref_cfg(scfg: SsmConfig) -> ModelConfig {
        ModelConfig {
            architecture: crate::config::Architecture::Llama,
            arch_str: "llama".into(),
            name: Some("ssm-ref-test".into()),
            n_vocab: 32,
            n_ctx: 32,
            n_embd: 64,
            n_layer: 1,
            n_ff: 128,
            n_head: 4,
            n_head_kv: 4,
            n_rot: 16,
            head_dim: 16,
            rms_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_type: crate::config::RopeType::Norm,
            bos_token_id: 1,
            eos_token_id: 2,
            add_bos_token: false,
            add_eos_token: false,
            ffn_activation: crate::config::FfnActivation::SiluGated,
            norm_kind: crate::config::NormKind::Rms,
            scale_embeddings: false,
            logit_scale: 0.0,
            attn_logit_softcap: 0.0,
            sliding_window: 0,
            sliding_window_pattern: false,
            layer_types: vec![],
            layer_n_ff: vec![],
            layer_n_head_kv: vec![],
            ssm: Some(scfg),
            moe: None,
        }
    }

    /// Max |a - b| / max(1, |b|).
    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs() / y.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    const REF_SHAPES: [SsmConfig; 4] = [
        // small, d_state < 32, head_dim 12
        SsmConfig {
            d_conv: 3,
            d_inner: 48,
            d_state: 16,
            n_head: 4,
            n_group: 1,
        },
        SsmConfig {
            d_conv: 4,
            d_inner: 64,
            d_state: 32,
            n_head: 8,
            n_group: 2,
        },
        // d_state 256 (the 8-slot device scan)
        SsmConfig {
            d_conv: 4,
            d_inner: 160,
            d_state: 256,
            n_head: 4,
            n_group: 2,
        },
        // Nemotron3-Nano-4B
        SsmConfig {
            d_conv: 4,
            d_inner: 7680,
            d_state: 128,
            n_head: 96,
            n_group: 8,
        },
    ];

    /// `mamba2_mix_host` stepped token by token vs the naive f64 reference over a multi-token
    /// sequence from a non-zero state: every y row ≤ 2e-5 relative (measured ≤ 1.8e-6 on the
    /// Nemotron shape), conv state bit-exact (pure copy of inputs), SSM state ≤ 2e-5 relative.
    #[test]
    fn mamba2_host_matches_f64_reference_multi_token() {
        for scfg in REF_SHAPES {
            let cfg = ref_cfg(scfg);
            let n_tok = if scfg.d_inner > 1000 { 24 } else { 40 };
            let d_in = scfg.d_in_proj() as usize;
            let d_inner = scfg.d_inner as usize;
            let mut cache = SsmCache::synthetic_for_test(&scfg, 2, 64, 5);
            let mut xbc = vec![0.0f32; scfg.conv_dim() as usize];
            let mut tmp = vec![0.0f32; scfg.n_head as usize];
            let mut y = vec![0.0f32; d_inner];
            let mut seed = 77u32 + scfg.d_state;
            // Non-zero starting state: 3 host steps on layer/slot 1.
            for _ in 0..3 {
                let zx = lcg_vec(&mut seed, d_in, 3.0);
                mamba2_mix_host(&mut cache, 1, &cfg, &zx, &mut xbc, &mut y, &mut tmp).unwrap();
            }
            let zx = lcg_vec(&mut seed, n_tok * d_in, 3.0);
            let r = mamba2_mix_reference_f64(&cache, 1, &scfg, cfg.rms_eps, &zx, n_tok);
            let mut worst = 0.0f32;
            for t in 0..n_tok {
                mamba2_mix_host(
                    &mut cache,
                    1,
                    &cfg,
                    &zx[t * d_in..(t + 1) * d_in],
                    &mut xbc,
                    &mut y,
                    &mut tmp,
                )
                .unwrap();
                worst = worst.max(max_rel(&y, &r.y[t * d_inner..(t + 1) * d_inner]));
            }
            let (hc, hs) = cache.host_state(1);
            assert_eq!(
                hc,
                &r.conv[..],
                "conv state (pure input copy) must be exact"
            );
            let es = max_rel(hs, &r.ssm);
            eprintln!(
                "ssm ref d_inner {} d_state {} d_conv {} n_tok {n_tok}: host vs f64 y {worst:.2e}, state {es:.2e}",
                scfg.d_inner, scfg.d_state, scfg.d_conv
            );
            assert!(worst <= 2e-5, "y rel err {worst}");
            assert!(es <= 2e-5, "ssm state rel err {es}");
        }
    }

    /// Chunk invariance of the reference (what prefill relies on when it splits rows into
    /// GEMM / SSM_CHUNK_ROWS chunks): n tokens at once == n1 then n - n1 with the state carried,
    /// for splits that cross d_conv - 1 and the 8 / 16-row device tiles.
    #[test]
    fn mamba2_reference_chunk_split_invariant() {
        let scfg = REF_SHAPES[1];
        let eps = 1e-5;
        let d_in = scfg.d_in_proj() as usize;
        let d_inner = scfg.d_inner as usize;
        let n = 37;
        let mut seed = 4242u32;
        let zx = lcg_vec(&mut seed, n * d_in, 3.0);
        let mut cache = SsmCache::synthetic_for_test(&scfg, 1, 64, 11);
        cache.ssm[0] = lcg_vec(&mut seed, cache.ssm[0].len(), 0.5);
        cache.conv[0] = lcg_vec(&mut seed, cache.conv[0].len(), 1.0);
        let whole = mamba2_mix_reference_f64(&cache, 0, &scfg, eps, &zx, n);
        for n1 in [1usize, 2, 3, 8, 16, 17, 36] {
            let a = mamba2_mix_reference_f64(&cache, 0, &scfg, eps, &zx, n1);
            let mut c2 = SsmCache::synthetic_for_test(&scfg, 1, 64, 11);
            c2.conv[0] = a.conv.clone();
            c2.ssm[0] = a.ssm.clone();
            let b = mamba2_mix_reference_f64(&c2, 0, &scfg, eps, &zx[n1 * d_in..], n - n1);
            let mut y = a.y.clone();
            y.extend_from_slice(&b.y);
            assert_eq!(y.len(), n * d_inner);
            // State is rounded to f32 at the split, so allow f32-level drift only.
            let e = max_rel(&y, &whole.y);
            assert!(e <= 1e-5, "split {n1}: y rel err {e}");
            assert_eq!(b.conv, whole.conv, "split {n1}: conv state");
            assert!(max_rel(&b.ssm, &whole.ssm) <= 1e-5, "split {n1}: ssm state");
        }
    }
}
