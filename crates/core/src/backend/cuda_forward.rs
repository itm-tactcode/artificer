//! Device-resident CUDA forward (decode and batched prefill).
//!
//! ## Residency
//! - Weights: resident (raw quantized blocks, `CudaBackend::matvec_dev`).
//! - Hidden state / activations: [`DeviceScratch`] on the device across all layers.
//! - KV cache: [`DeviceKvCache`] on the device (attention layers only); K/V projections
//!   write straight into the cache rows (f32), or via `store_f16` (f16, bit-identical
//!   to the host `KvCache` f16 rounding).
//! - Attention: one `attn_rows_*` launch per layer (scores + softmax + AV on device).
//! - One sync per token: logits D2H in [`finish_logits`].
//!
//! ## Device host-ops and their host fallback
//! Norms, residual adds, biases, activations, RoPE, embedding gather, final softcap and
//! the Mamba-2 SSM layer each go through [`DeviceHostOps`]
//! first ([`hostops`] returns the `cuda_hostops` kernels); when a hook returns `Ok(false)` the isolated `host_fallback_*`
//! helper below does D2H → host op → H2D, counted by `CudaBackend::host_fallback_roundtrips`.
//! `ARTIFICER_HOSTOPS=0` disables the device hooks (host fallback everywhere) for A/B runs.
//!
//! The legacy per-matvec path (pre-residency) is still reachable with
//! `ARTIFICER_CUDA_LEGACY=1` for A/B comparisons.

use crate::config::{FfnActivation, LayerType};
use crate::error::{CoreError, Result};
use crate::forward::weights::{add_bias, apply_norm};
use crate::forward::{ForwardScratch, PrefillCancel, PrefillStatus};
use crate::kv::{KvCache, KvElem};
use crate::model::{LlamaModel, WeightTensor};
use crate::ops::{
    add_inplace, apply_rope, gelu_inplace, mul_inplace, relu_sqr_inplace, scale_inplace,
    silu_inplace, softcap_inplace,
};
use crate::quant::dequantize;
use crate::ssm::SsmCache;

use super::cuda::{CudaBackend, DevPtr, DeviceTensor, LaunchArgs, MatvecJob};

// ---------------------------------------------------------------------------
// Hooks for the device host-ops (norm / element-wise / SSM). Defaults = not on device.
// ---------------------------------------------------------------------------

/// Device implementations of the ops that still run on the host. Contract for every
/// method: async on `cb.stream()`, no sync, no host copy; return `Ok(true)` when the op
/// ran on the device, `Ok(false)` to let the host fallback path run instead.
/// `rows` = number of token rows (1 for decode, N for batched prefill).
#[allow(unused_variables, clippy::too_many_arguments)]
pub trait DeviceHostOps: Sync {
    /// `out[r] = norm(x[r]) * w (+ b)` per `model.config.norm_kind` / `rms_eps`.
    fn norm(
        &self,
        cb: &CudaBackend,
        model: &LlamaModel,
        w: &WeightTensor,
        b: Option<&WeightTensor>,
        x: DevPtr,
        out: DevPtr,
        rows: usize,
        dim: usize,
    ) -> Result<bool> {
        Ok(false)
    }
    /// `a += b` (same length).
    fn add_inplace(&self, cb: &CudaBackend, a: DevPtr, b: DevPtr) -> Result<bool> {
        Ok(false)
    }
    /// `out[r] += bias` for each of `rows` rows of `dim`.
    fn add_bias(
        &self,
        cb: &CudaBackend,
        model: &LlamaModel,
        bias: &WeightTensor,
        out: DevPtr,
        rows: usize,
        dim: usize,
    ) -> Result<bool> {
        Ok(false)
    }
    /// Gated: `hb = act(hb) * hb2`; ReluSqrSequential: `hb = relu(hb)^2` (`hb2 = None`).
    fn act(
        &self,
        cb: &CudaBackend,
        act: FfnActivation,
        hb: DevPtr,
        hb2: Option<DevPtr>,
    ) -> Result<bool> {
        Ok(false)
    }
    /// `x *= s`.
    fn scale(&self, cb: &CudaBackend, x: DevPtr, s: f32) -> Result<bool> {
        Ok(false)
    }
    /// `x = cap * tanh(x / cap)` (final logits).
    fn softcap(&self, cb: &CudaBackend, x: DevPtr, cap: f32) -> Result<bool> {
        Ok(false)
    }
    /// RoPE in place on `rows` rows (row stride `row_stride`) at positions `pos0..`.
    fn rope(
        &self,
        cb: &CudaBackend,
        x: DevPtr,
        rows: usize,
        row_stride: usize,
        n_head: usize,
        head_dim: usize,
        pos0: usize,
        freq_base: f32,
        n_rot: usize,
    ) -> Result<bool> {
        Ok(false)
    }
    /// Embedding gather + dequant (+ √d scale when `scale_embeddings`) into `out[rows × n_embd]`.
    fn embed(
        &self,
        cb: &CudaBackend,
        model: &LlamaModel,
        tokens: &[u32],
        out: DevPtr,
    ) -> Result<bool> {
        Ok(false)
    }
    /// `true` once [`Self::ssm_mix`] is implemented (the forward then runs norm + in_proj /
    /// out_proj + residual on the device around it).
    fn has_ssm_mix(&self) -> bool {
        false
    }
    /// Mamba-2 mix (conv + scan + swiglu + grouped RMSNorm) for `n_tok` rows:
    /// reads `zx = [n_tok × d_in_proj]` (= `[z | xBC | dt]`, in_proj output), updates the
    /// conv/SSM state of `layer_idx` (device state kept inside `SsmCache`),
    /// writes `y = [n_tok × d_inner]`.
    fn ssm_mix(
        &self,
        cb: &CudaBackend,
        ssm: &mut SsmCache,
        model: &LlamaModel,
        layer_idx: usize,
        zx: DevPtr,
        y: DevPtr,
        n_tok: usize,
    ) -> Result<bool> {
        Ok(false)
    }
}

/// Default: every op on the host (fallback path).
pub struct HostFallbackOps;
impl DeviceHostOps for HostFallbackOps {}

/// The device host-ops in use: the `cuda_hostops` kernels.
pub(crate) fn hostops() -> &'static dyn DeviceHostOps {
    &super::cuda_hostops::OPS
}

/// `ARTIFICER_CUDA_LEGACY=1` → old per-matvec round-trip path (A/B only).
pub fn legacy_mode() -> bool {
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    FORCE_LEGACY.with(|f| f.get())
        || *LEGACY.get_or_init(|| std::env::var("ARTIFICER_CUDA_LEGACY").is_ok_and(|v| v == "1"))
}

thread_local! {
    /// Per-thread override of [`legacy_mode`] (tests / A-B tools on this thread only).
    static FORCE_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Force the legacy per-matvec CUDA path on the current thread (A/B comparisons).
pub fn set_thread_legacy(on: bool) {
    FORCE_LEGACY.with(|f| f.set(on));
}

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------

/// Device KV cache. Only attention layers get storage. Per layer, K and V are
/// `[n_ctx][kv_dim]` (`kv_dim` = host `KvCache::n_embd_kv`), f32 or f16 like the host.
pub struct DeviceKvCache {
    pub len: usize,
    pub n_ctx: usize,
    pub kv_dim: usize,
    pub elem: KvElem,
    layer_to_slot: Vec<usize>,
    k: Vec<DeviceTensor>,
    v: Vec<DeviceTensor>,
}

impl DeviceKvCache {
    /// Allocate device K/V for layers where `is_attn[layer]`.
    pub fn new(
        cb: &CudaBackend,
        is_attn: &[bool],
        n_ctx: usize,
        kv_dim: usize,
        elem: KvElem,
    ) -> Result<Self> {
        let bytes = n_ctx * kv_dim * elem.size_bytes();
        let n_attn = is_attn.iter().filter(|&&a| a).count();
        cb.check_fits((bytes * 2 * n_attn) as u64, "device KV cache")?;
        let mut layer_to_slot = vec![usize::MAX; is_attn.len()];
        let (mut k, mut v) = (Vec::new(), Vec::new());
        for (l, &a) in is_attn.iter().enumerate() {
            if a {
                layer_to_slot[l] = k.len();
                k.push(cb.alloc_bytes(bytes)?);
                v.push(cb.alloc_bytes(bytes)?);
            }
        }
        Ok(Self {
            len: 0,
            n_ctx,
            kv_dim,
            elem,
            layer_to_slot,
            k,
            v,
        })
    }

    fn slot(&self, layer: usize) -> Result<usize> {
        match self.layer_to_slot.get(layer) {
            Some(&s) if s != usize::MAX => Ok(s),
            _ => Err(CoreError::Other(format!(
                "device KV: layer {layer} has no KV storage"
            ))),
        }
    }

    /// Raw base pointers (K, V) of `layer` (element type = `self.elem`).
    pub fn layer_raw(
        &self,
        layer: usize,
    ) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        let s = self.slot(layer)?;
        Ok((self.k[s].raw(), self.v[s].raw()))
    }

    /// Bytes reserved on the device for K+V.
    pub fn storage_bytes(&self) -> usize {
        self.k.iter().chain(self.v.iter()).map(|t| t.bytes()).sum()
    }

    /// f32 view of K rows `pos0..pos0+n` (contiguous). Only for `elem == F32`.
    pub fn k_rows(&self, layer: usize, pos0: usize, n: usize) -> Result<DevPtr> {
        self.rows_f32(&self.k, layer, pos0, n)
    }

    /// f32 view of V rows `pos0..pos0+n` (contiguous). Only for `elem == F32`.
    pub fn v_rows(&self, layer: usize, pos0: usize, n: usize) -> Result<DevPtr> {
        self.rows_f32(&self.v, layer, pos0, n)
    }

    fn rows_f32(
        &self,
        bufs: &[DeviceTensor],
        layer: usize,
        pos0: usize,
        n: usize,
    ) -> Result<DevPtr> {
        if self.elem != KvElem::F32 {
            return Err(CoreError::Other(
                "device KV f32 row view on f16 cache".into(),
            ));
        }
        if pos0 + n > self.n_ctx {
            return Err(CoreError::ContextExceeded {
                need: pos0 + n,
                max: self.n_ctx,
            });
        }
        Ok(bufs[self.slot(layer)?]
            .ptr()
            .offset(pos0 * self.kv_dim, n * self.kv_dim))
    }

    /// Store `n` f32 rows (`src = [n × kv_dim]`, device) at `pos0..` into K (`is_v = false`) or V.
    /// f32: D2D copy (skipped when `src` already is the cache row); f16: device conversion
    /// with host-identical rounding.
    pub fn store_rows(
        &self,
        cb: &CudaBackend,
        layer: usize,
        pos0: usize,
        n: usize,
        src: DevPtr,
        is_v: bool,
    ) -> Result<()> {
        let bufs = if is_v { &self.v } else { &self.k };
        if pos0 + n > self.n_ctx {
            return Err(CoreError::ContextExceeded {
                need: pos0 + n,
                max: self.n_ctx,
            });
        }
        let elems = n * self.kv_dim;
        match self.elem {
            KvElem::F32 => {
                let dst = self.rows_f32(bufs, layer, pos0, n)?;
                if dst.ptr != src.ptr {
                    cb.copy_d2d(src.prefix(elems), dst)?;
                }
                Ok(())
            }
            KvElem::F16 => {
                let base = bufs[self.slot(layer)?].raw() as u64 + (pos0 * self.kv_dim * 2) as u64;
                let f = cb.kernel("store_f16")?;
                let args = LaunchArgs::new()
                    .ptr(src)
                    .raw(base as *mut std::ffi::c_void)
                    .i32(elems as i32);
                cb.launch(
                    f,
                    ((elems as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    args,
                )
            }
        }
    }

    /// Test/debug: read back `n` rows of K or V at `pos0` as f32 (blocking).
    pub fn read_rows(
        &self,
        cb: &CudaBackend,
        layer: usize,
        pos0: usize,
        n: usize,
        is_v: bool,
    ) -> Result<Vec<f32>> {
        let bufs = if is_v { &self.v } else { &self.k };
        let elems = n * self.kv_dim;
        match self.elem {
            KvElem::F32 => {
                let mut out = vec![0.0f32; elems];
                cb.download_f32(self.rows_f32(bufs, layer, pos0, n)?, &mut out)?;
                Ok(out)
            }
            KvElem::F16 => {
                // Download raw u16 pairs through an f32 view of the byte range.
                let base = bufs[self.slot(layer)?].ptr().ptr + (pos0 * self.kv_dim * 2) as u64;
                let words = elems.div_ceil(2);
                let mut raw = vec![0.0f32; words];
                cb.download_f32(
                    DevPtr {
                        ptr: base,
                        len: words,
                    },
                    &mut raw,
                )?;
                let mut out = Vec::with_capacity(elems);
                for w in raw {
                    let b = w.to_bits();
                    out.push(crate::quant::f16_to_f32((b & 0xFFFF) as u16));
                    out.push(crate::quant::f16_to_f32((b >> 16) as u16));
                }
                out.truncate(elems);
                Ok(out)
            }
        }
    }
}

/// Device activations, row-major `[max_batch × dim]` (decode uses row 0).
pub struct DeviceScratch {
    pub max_batch: usize,
    /// Residual stream / hidden state `[max_batch × n_embd]`.
    pub x: DeviceTensor,
    /// Norm output / matmul output `[max_batch × max(n_embd, q_dim)]`.
    pub xb: DeviceTensor,
    /// Attention context `[max_batch × q_dim]`.
    pub xb2: DeviceTensor,
    pub q: DeviceTensor,
    /// K / V projection output before an f16 KV store `[max_batch × n_embd_kv]`.
    pub k: DeviceTensor,
    pub v: DeviceTensor,
    /// Attention scores `[att_rows × n_head × n_ctx]` (`att_rows <= max_batch`; batched
    /// prefill runs attention in sub-blocks of at most `att_rows` query rows).
    pub att: DeviceTensor,
    /// Query rows the `att` scratch holds.
    pub att_rows: usize,
    /// Split-K decode attention partials ([`super::cuda_attention::fd_part_len`] f32; decode
    /// scratch only, 1 element for prefill scratch).
    pub fd_part: DeviceTensor,
    pub hb: DeviceTensor,
    pub hb2: DeviceTensor,
    /// SSM in_proj output `[max_batch × d_in_proj]` = `[z | xBC | dt]` (empty for dense).
    pub ssm_zx: DeviceTensor,
    /// SSM mix output `[max_batch × d_inner]` (empty for dense).
    pub ssm_y: DeviceTensor,
    /// LM-head output `[n_vocab]` (last row only).
    pub logits: DeviceTensor,
    pub n_embd: usize,
    pub q_dim: usize,
    pub xb_dim: usize,
    pub kv_dim: usize,
    pub n_ff: usize,
    pub d_in_proj: usize,
    pub d_inner: usize,
}

impl DeviceScratch {
    pub fn new(
        cb: &CudaBackend,
        model: &LlamaModel,
        n_ctx: usize,
        kv_dim: usize,
        max_batch: usize,
        att_rows: usize,
    ) -> Result<Self> {
        let cfg = &model.config;
        let n_embd = cfg.n_embd as usize;
        let q_dim = (cfg.n_head as usize * cfg.head_dim as usize).max(n_embd);
        let xb_dim = n_embd.max(q_dim);
        let n_ff = cfg
            .layer_n_ff
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(cfg.n_ff)
            .max(1) as usize;
        let n_head = cfg.n_head.max(1) as usize;
        let (d_in_proj, d_inner) = cfg
            .ssm
            .map(|s| (s.d_in_proj() as usize, s.d_inner as usize))
            .unwrap_or((0, 0));
        let b = max_batch.max(1);
        let att_rows = att_rows.clamp(1, b);
        Ok(Self {
            max_batch: b,
            x: cb.alloc_f32(b * n_embd)?,
            xb: cb.alloc_f32(b * xb_dim)?,
            xb2: cb.alloc_f32(b * q_dim)?,
            q: cb.alloc_f32(b * q_dim)?,
            k: cb.alloc_f32(b * kv_dim)?,
            v: cb.alloc_f32(b * kv_dim)?,
            att: cb.alloc_f32(att_rows * n_head * n_ctx)?,
            att_rows,
            fd_part: cb.alloc_f32(if b == 1 {
                super::cuda_attention::fd_part_len(n_head, cfg.head_dim().max(1) as usize)
            } else {
                1
            })?,
            hb: cb.alloc_f32(b * n_ff)?,
            hb2: cb.alloc_f32(b * n_ff)?,
            ssm_zx: cb.alloc_f32((b * d_in_proj).max(1))?,
            ssm_y: cb.alloc_f32((b * d_inner).max(1))?,
            logits: cb.alloc_f32(cfg.n_vocab as usize)?,
            n_embd,
            q_dim,
            xb_dim,
            kv_dim,
            n_ff,
            d_in_proj,
            d_inner,
        })
    }
}

/// Everything the device forward keeps across tokens (lives in `KvCache::device`).
pub struct CudaForwardState {
    pub scratch: DeviceScratch,
    pub kv: DeviceKvCache,
    /// Batched-prefill activations `[rows × dim]` (lazily sized, see [`prefill_batched`]).
    pub prefill: Option<DeviceScratch>,
}

impl CudaForwardState {
    pub fn new(cb: &CudaBackend, model: &LlamaModel, cache: &KvCache) -> Result<Self> {
        let cfg = &model.config;
        let is_attn: Vec<bool> = if cfg.architecture.is_hybrid() {
            cfg.layer_types
                .iter()
                .map(|t| *t == LayerType::Attention)
                .collect()
        } else {
            vec![true; cfg.n_layer as usize]
        };
        // Device KV element type: f16 by default, `ARTIFICER_KV_F16=0` → f32.
        let elem = crate::kv::device_kv_elem(cache.elem);
        let kv = DeviceKvCache::new(cb, &is_attn, cache.n_ctx, cache.n_embd_kv, elem)?;
        // Decode scratch (1 row); batched prefill sizes its own (see `prefill_rows`).
        let scratch = DeviceScratch::new(cb, model, cache.n_ctx, cache.n_embd_kv, 1, 1)?;
        Ok(Self {
            scratch,
            kv,
            prefill: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Host fallbacks (used when a device hook returns `Ok(false)`)
// ---------------------------------------------------------------------------

/// Host fallback: norm. D2H `x` → `apply_norm` → H2D `out`.
#[allow(clippy::too_many_arguments)]
fn host_fallback_norm(
    cb: &CudaBackend,
    model: &LlamaModel,
    w: &WeightTensor,
    b: Option<&WeightTensor>,
    x: DevPtr,
    out: DevPtr,
    dim: usize,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let ForwardScratch {
        x: hx,
        xb: hxb,
        norm_w,
        norm_b,
        ..
    } = hs;
    cb.host_fallback_download(x.prefix(dim), &mut hx[..dim])?;
    apply_norm(
        model,
        &model.config,
        w,
        b,
        &hx[..dim],
        &mut hxb[..dim],
        norm_w,
        norm_b,
    )?;
    cb.upload_f32(&hxb[..dim], out)
}

/// Host fallback: `a += b`.
fn host_fallback_add(
    cb: &CudaBackend,
    a: DevPtr,
    b: DevPtr,
    n: usize,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let ForwardScratch { x: ha, xb: hb, .. } = hs;
    cb.host_fallback_download(a.prefix(n), &mut ha[..n])?;
    cb.download_f32(b.prefix(n), &mut hb[..n])?;
    add_inplace(&mut ha[..n], &hb[..n]);
    cb.upload_f32(&ha[..n], a)
}

/// Host fallback: bias add.
fn host_fallback_bias(
    cb: &CudaBackend,
    model: &LlamaModel,
    bias: &WeightTensor,
    out: DevPtr,
    n: usize,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let buf = &mut hs.xb[..n];
    cb.host_fallback_download(out.prefix(n), buf)?;
    add_bias(model, bias, buf)?;
    cb.upload_f32(buf, out)
}

/// Host fallback: generic host op `f` in place on `n` device elements (staged through `buf`).
fn host_fallback_map(
    cb: &CudaBackend,
    p: DevPtr,
    n: usize,
    buf: &mut [f32],
    f: impl FnOnce(&mut [f32]),
) -> Result<()> {
    cb.host_fallback_download(p.prefix(n), &mut buf[..n])?;
    f(&mut buf[..n]);
    cb.upload_f32(&buf[..n], p)
}

fn norm_dev_or_host(
    cb: &CudaBackend,
    model: &LlamaModel,
    w: &WeightTensor,
    b: Option<&WeightTensor>,
    x: DevPtr,
    out: DevPtr,
    dim: usize,
    hs: &mut ForwardScratch,
) -> Result<()> {
    if hostops().norm(cb, model, w, b, x, out, 1, dim)? {
        return Ok(());
    }
    host_fallback_norm(cb, model, w, b, x, out, dim, hs)
}

fn add_dev_or_host(
    cb: &CudaBackend,
    a: DevPtr,
    b: DevPtr,
    n: usize,
    hs: &mut ForwardScratch,
) -> Result<()> {
    if hostops().add_inplace(cb, a.prefix(n), b.prefix(n))? {
        return Ok(());
    }
    host_fallback_add(cb, a, b, n, hs)
}

fn bias_dev_or_host(
    cb: &CudaBackend,
    model: &LlamaModel,
    bias: Option<&WeightTensor>,
    out: DevPtr,
    n: usize,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let Some(bias) = bias else { return Ok(()) };
    if hostops().add_bias(cb, model, bias, out.prefix(n), 1, n)? {
        return Ok(());
    }
    host_fallback_bias(cb, model, bias, out, n, hs)
}

fn matvec(
    cb: &CudaBackend,
    model: &LlamaModel,
    w: &WeightTensor,
    x: DevPtr,
    y: DevPtr,
) -> Result<()> {
    let data = model.tensor_bytes(w)?;
    cb.matvec_dev(data, w.ty, w.k(), w.n(), x, y)
}

/// Several matvecs over the same input (Q/K/V, gate/up): the Q8 activation is quantized once.
fn matvec_shared(
    cb: &CudaBackend,
    model: &LlamaModel,
    x: DevPtr,
    outs: &[(&WeightTensor, DevPtr)],
) -> Result<()> {
    let k = outs[0].0.k();
    let mut jobs = Vec::with_capacity(outs.len());
    for (w, y) in outs {
        if w.k() != k {
            return Err(CoreError::Shape(format!(
                "matvec_shared: k mismatch {} vs {k}",
                w.k()
            )));
        }
        jobs.push(MatvecJob {
            weight: model.tensor_bytes(w)?,
            ty: w.ty,
            n: w.n(),
            y: *y,
        });
    }
    cb.matvec_dev_multi(x, k, &jobs)
}

fn req<'a>(w: &'a Option<WeightTensor>, name: &str) -> Result<&'a WeightTensor> {
    w.as_ref()
        .ok_or_else(|| CoreError::MissingTensor(name.into()))
}

// ---------------------------------------------------------------------------
// Device decode step
// ---------------------------------------------------------------------------

/// One token through the device-resident graph. Same contract as
/// [`LlamaModel::forward`]: logits (host) only when `compute_logits`.
#[allow(clippy::too_many_arguments)]
pub fn decode_step<'a>(
    model: &LlamaModel,
    cb: &CudaBackend,
    token: u32,
    pos: usize,
    cache: &mut KvCache,
    ssm: Option<&mut SsmCache>,
    hs: &'a mut ForwardScratch,
    compute_logits: bool,
) -> Result<Option<&'a [f32]>> {
    let mut st = match cache.device.take() {
        Some(s) => s,
        None => Box::new(CudaForwardState::new(cb, model, cache)?),
    };
    let res = decode_inner(model, cb, token, pos, &mut st, ssm, hs, compute_logits);
    if res.is_ok() {
        st.kv.len = st.kv.len.max(pos + 1);
        cache.len = cache.len.max(pos + 1);
    }
    cache.device = Some(st);
    res?;
    Ok(if compute_logits {
        Some(&hs.logits[..])
    } else {
        None
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_inner(
    model: &LlamaModel,
    cb: &CudaBackend,
    token: u32,
    pos: usize,
    st: &mut CudaForwardState,
    mut ssm: Option<&mut SsmCache>,
    hs: &mut ForwardScratch,
    compute_logits: bool,
) -> Result<()> {
    let cfg = &model.config;
    let n_embd = cfg.n_embd as usize;
    let ops = hostops();
    let x = st.scratch.x.ptr();
    let xb = st.scratch.xb.ptr();

    // Embedding (host gather unless the device embed hook runs).
    if !ops.embed(cb, model, &[token], x.prefix(n_embd))? {
        let data = model.tensor_bytes(&model.token_embd)?;
        dequantize(
            data,
            model.token_embd.ty,
            token as usize * n_embd,
            n_embd,
            &mut hs.x[..n_embd],
        )?;
        if cfg.scale_embeddings {
            scale_inplace(&mut hs.x[..n_embd], (n_embd as f32).sqrt());
        }
        cb.upload_f32(&hs.x[..n_embd], x)?;
    }

    let hybrid = cfg.architecture.is_hybrid();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let lt = if hybrid {
            layer.layer_type
        } else {
            LayerType::Dense
        };
        match lt {
            LayerType::Ssm => {
                let ssm = ssm.as_deref_mut().ok_or_else(|| {
                    CoreError::Other("hybrid model requires SsmCache in forward".into())
                })?;
                ssm_layer(model, cb, layer_idx, st, ssm, hs)?;
            }
            LayerType::Attention => attention_layer(model, cb, layer_idx, pos, st, hs)?,
            LayerType::Ffn => ffn_layer(model, cb, layer_idx, st, hs)?,
            LayerType::Dense => {
                attention_layer(model, cb, layer_idx, pos, st, hs)?;
                ffn_layer(model, cb, layer_idx, st, hs)?;
            }
        }
    }

    norm_dev_or_host(
        cb,
        model,
        &model.output_norm,
        model.output_norm_b.as_ref(),
        x,
        xb,
        n_embd,
        hs,
    )?;
    if !compute_logits {
        return Ok(());
    }
    let logits = st.scratch.logits.ptr();
    matvec(cb, model, &model.output, xb, logits)?;
    let mut softcap_on_host = false;
    if cfg.logit_scale > 0.0 && !ops.softcap(cb, logits, cfg.logit_scale)? {
        softcap_on_host = true; // host fallback: applied after the logits download (no extra round trip)
    }
    finish_logits(cb, logits, &mut hs.logits)?;
    if softcap_on_host {
        softcap_inplace(&mut hs.logits, cfg.logit_scale);
    }
    Ok(())
}

/// The one per-token sync: logits D2H + stream sync.
pub fn finish_logits(cb: &CudaBackend, logits: DevPtr, host: &mut [f32]) -> Result<()> {
    let n = host.len().min(logits.len);
    cb.download_f32(logits.prefix(n), &mut host[..n])
}

fn attention_layer(
    model: &LlamaModel,
    cb: &CudaBackend,
    layer_idx: usize,
    pos: usize,
    st: &mut CudaForwardState,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let cfg = &model.config;
    let layer = &model.layers[layer_idx];
    let ops = hostops();
    let n_embd = cfg.n_embd as usize;
    let n_head = cfg.n_head as usize;
    let n_head_kv = cfg
        .layer_n_head_kv
        .get(layer_idx)
        .copied()
        .filter(|&x| x > 0)
        .unwrap_or(cfg.n_head_kv) as usize;
    let head_dim = cfg.head_dim() as usize;
    let n_rot = cfg.n_rot as usize;
    let q_dim = n_head * head_dim;
    let kv_n = n_head_kv * head_dim;
    let sc = &st.scratch;
    let (x, xb, xb2, q) = (sc.x.ptr(), sc.xb.ptr(), sc.xb2.ptr(), sc.q.ptr());

    norm_dev_or_host(
        cb,
        model,
        &layer.attn_norm,
        layer.attn_norm_b.as_ref(),
        x,
        xb,
        n_embd,
        hs,
    )?;

    let attn_q = req(&layer.attn_q, "attn_q")?;
    let attn_k = req(&layer.attn_k, "attn_k")?;
    let attn_v = req(&layer.attn_v, "attn_v")?;
    let attn_o = req(&layer.attn_output, "attn_output")?;

    // K/V: f32 cache → project straight into the cache row (zero copy); f16 → via scratch.
    let direct = st.kv.elem == KvElem::F32;
    let (k_dst, v_dst) = if direct {
        (
            st.kv.k_rows(layer_idx, pos, 1)?,
            st.kv.v_rows(layer_idx, pos, 1)?,
        )
    } else {
        (sc.k.ptr(), sc.v.ptr())
    };
    matvec_shared(
        cb,
        model,
        xb,
        &[(attn_q, q), (attn_k, k_dst), (attn_v, v_dst)],
    )?;
    bias_dev_or_host(cb, model, layer.attn_q_b.as_ref(), q, q_dim, hs)?;
    bias_dev_or_host(cb, model, layer.attn_k_b.as_ref(), k_dst, attn_k.n(), hs)?;
    bias_dev_or_host(cb, model, layer.attn_v_b.as_ref(), v_dst, attn_v.n(), hs)?;

    if !cfg.architecture.is_hybrid() && n_rot > 0 && cfg.rope_freq_base > 0.0 {
        let fb = cfg.rope_freq_base;
        if !ops.rope(cb, q, 1, q_dim, n_head, head_dim, pos, fb, n_rot)? {
            host_fallback_map(cb, q, q_dim, &mut hs.xb2, |b| {
                apply_rope(b, n_head, head_dim, pos, fb, n_rot)
            })?;
        }
        if !ops.rope(cb, k_dst, 1, kv_n, n_head_kv, head_dim, pos, fb, n_rot)? {
            host_fallback_map(cb, k_dst, kv_n, &mut hs.xb2, |b| {
                apply_rope(b, n_head_kv, head_dim, pos, fb, n_rot)
            })?;
        }
    }
    let att_scale = if cfg.scale_embeddings {
        let s = 1.0 / (head_dim as f32).sqrt();
        if !ops.scale(cb, q.prefix(q_dim), s)? {
            host_fallback_map(cb, q, q_dim, &mut hs.xb2, |b| scale_inplace(b, s))?;
        }
        1.0
    } else {
        1.0 / (head_dim as f32).sqrt()
    };
    if !direct {
        st.kv.store_rows(cb, layer_idx, pos, 1, k_dst, false)?;
        st.kv.store_rows(cb, layer_idx, pos, 1, v_dst, true)?;
    }

    let window = if cfg.sliding_window > 0 {
        let use_window = if cfg.sliding_window_pattern {
            layer_idx.is_multiple_of(2)
        } else {
            true
        };
        if use_window {
            cfg.sliding_window as usize
        } else {
            0
        }
    } else {
        0
    };
    // Decode: split-K flash-decode unless `ARTIFICER_DECODE_ATTN=naive`.
    super::cuda_attention::attn_decode_dev(
        cb,
        super::cuda_attention::decode_attn_mode(),
        q,
        &st.kv,
        layer_idx,
        xb2,
        sc.att.ptr(),
        sc.fd_part.ptr(),
        pos,
        n_head,
        n_head_kv,
        head_dim,
        q_dim,
        window,
        att_scale,
        cfg.attn_logit_softcap,
    )?;

    matvec(cb, model, attn_o, xb2, xb)?;
    bias_dev_or_host(cb, model, layer.attn_output_b.as_ref(), xb, attn_o.n(), hs)?;
    add_dev_or_host(cb, x, xb, n_embd, hs)
}

fn ffn_layer(
    model: &LlamaModel,
    cb: &CudaBackend,
    layer_idx: usize,
    st: &mut CudaForwardState,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let cfg = &model.config;
    let layer = &model.layers[layer_idx];
    let ops = hostops();
    let n_embd = cfg.n_embd as usize;
    let sc = &st.scratch;
    let (x, xb, hb, hb2) = (sc.x.ptr(), sc.xb.ptr(), sc.hb.ptr(), sc.hb2.ptr());
    let ffn_up = req(&layer.ffn_up, "ffn_up")?;
    let ffn_down = req(&layer.ffn_down, "ffn_down")?;
    let n_ff = ffn_up.n();

    match layer.ffn_norm.as_ref() {
        Some(nw) => norm_dev_or_host(cb, model, nw, layer.ffn_norm_b.as_ref(), x, xb, n_embd, hs)?,
        None => norm_dev_or_host(
            cb,
            model,
            &layer.attn_norm,
            layer.attn_norm_b.as_ref(),
            x,
            xb,
            n_embd,
            hs,
        )?,
    }

    match cfg.ffn_activation {
        FfnActivation::SiluGated | FfnActivation::GeluGated => {
            let gate = req(&layer.ffn_gate, "ffn_gate.weight")?;
            matvec_shared(cb, model, xb, &[(gate, hb), (ffn_up, hb2)])?;
            bias_dev_or_host(cb, model, layer.ffn_up_b.as_ref(), hb2, n_ff, hs)?;
            if !ops.act(
                cb,
                cfg.ffn_activation,
                hb.prefix(n_ff),
                Some(hb2.prefix(n_ff)),
            )? {
                // Host fallback: activation + gate multiply.
                let ForwardScratch {
                    hb: h1, hb2: h2, ..
                } = hs;
                cb.host_fallback_download(hb.prefix(n_ff), &mut h1[..n_ff])?;
                cb.download_f32(hb2.prefix(n_ff), &mut h2[..n_ff])?;
                match cfg.ffn_activation {
                    FfnActivation::SiluGated => silu_inplace(&mut h1[..n_ff]),
                    _ => gelu_inplace(&mut h1[..n_ff]),
                }
                mul_inplace(&mut h1[..n_ff], &h2[..n_ff]);
                cb.upload_f32(&h1[..n_ff], hb)?;
            }
        }
        FfnActivation::ReluSqrSequential => {
            matvec(cb, model, ffn_up, xb, hb)?;
            bias_dev_or_host(cb, model, layer.ffn_up_b.as_ref(), hb, n_ff, hs)?;
            if !ops.act(cb, cfg.ffn_activation, hb.prefix(n_ff), None)? {
                host_fallback_map(cb, hb, n_ff, &mut hs.hb, relu_sqr_inplace)?;
            }
        }
    }
    matvec(cb, model, ffn_down, hb, xb)?;
    bias_dev_or_host(cb, model, layer.ffn_down_b.as_ref(), xb, ffn_down.n(), hs)?;
    add_dev_or_host(cb, x, xb, n_embd, hs)
}

fn ssm_layer(
    model: &LlamaModel,
    cb: &CudaBackend,
    layer_idx: usize,
    st: &mut CudaForwardState,
    ssm: &mut SsmCache,
    hs: &mut ForwardScratch,
) -> Result<()> {
    let cfg = &model.config;
    let layer = &model.layers[layer_idx];
    let ops = hostops();
    let n_embd = cfg.n_embd as usize;
    let sc = &st.scratch;
    let (x, xb) = (sc.x.ptr(), sc.xb.ptr());
    if ops.has_ssm_mix() {
        // Device path: norm → in_proj → mix → out_proj → residual.
        norm_dev_or_host(cb, model, &layer.attn_norm, None, x, xb, n_embd, hs)?;
        matvec(
            cb,
            model,
            req(&layer.ssm_in, "ssm_in")?,
            xb,
            sc.ssm_zx.ptr(),
        )?;
        if !ops.ssm_mix(
            cb,
            ssm,
            model,
            layer_idx,
            sc.ssm_zx.ptr(),
            sc.ssm_y.ptr(),
            1,
        )? {
            return Err(CoreError::Backend(
                "cuda_hostops: has_ssm_mix() but ssm_mix declined".into(),
            ));
        }
        matvec(
            cb,
            model,
            req(&layer.ssm_out, "ssm_out")?,
            sc.ssm_y.ptr(),
            xb,
        )?;
        return add_dev_or_host(cb, x, xb, n_embd, hs);
    }
    // Host fallback: whole Mamba-2 layer on the host path (state stays in the host SsmCache);
    // its in_proj/out_proj still run on the GPU through `Backend::matmul_vec`.
    let ForwardScratch {
        x: hx,
        xb: hxb,
        ssm_zx,
        ssm_xbc,
        ssm_y,
        ssm_tmp,
        norm_w,
        ..
    } = hs;
    cb.host_fallback_download(x.prefix(n_embd), &mut hx[..n_embd])?;
    crate::ssm::mamba2_step(
        model,
        layer,
        layer_idx,
        ssm,
        &mut hx[..n_embd],
        hxb,
        ssm_zx,
        ssm_xbc,
        ssm_y,
        ssm_tmp,
        norm_w,
        cb,
    )?;
    cb.upload_f32(&hx[..n_embd], x)
}

// ---------------------------------------------------------------------------
// Batched prefill: N prompt rows per layer, GEMM projections
// ---------------------------------------------------------------------------

/// Max prompt rows per batched-prefill GEMM chunk. Every chunk dequantizes
/// each weight to f16 tiles once, so fewer, larger chunks cut the dequant passes (8193 tokens:
/// 51 → 9). Attention inside a chunk runs in sub-blocks sized by [`PREFILL_ATT_SCRATCH_BYTES`].
/// Tuning / A-B: `ARTIFICER_PREFILL_CHUNK=<rows>`.
pub const PREFILL_CHUNK: usize = 1024;

/// Budget for the attention score scratch `att_rows × n_head × n_ctx` f32 (= the old chunk cap:
/// 163 rows for Nemotron at n_ctx 10240). Tuning / A-B: `ARTIFICER_PREFILL_ATT_ROWS=<rows>`.
pub const PREFILL_ATT_SCRATCH_BYTES: usize = 256 << 20;

/// Budget for the other per-row prefill activations (x, xb, xb2, q, k, v, hb, hb2, ssm_zx, ssm_y
/// and the backend's f16 activation copy). Nemotron: ≈ 308 KB/row → 1024 rows ≈ 316 MB.
pub const PREFILL_ACT_SCRATCH_BYTES: usize = 512 << 20;

/// Fewer prompt tokens than this go token by token: the dequant + GEMM pass has a fixed cost
/// of roughly four decode steps (Nemotron / RTX 4050).
pub const PREFILL_MIN_BATCH: usize = 5;

/// Batched prefill needs the device host-ops (row-capable norm / add / bias / act / rope /
/// scale / ssm_mix) and per-layer K/V widths equal to the cache row width. When this is
/// `false`, [`LlamaModel::prefill`] falls back to token-by-token [`decode_step`].
pub fn batched_prefill_supported(model: &LlamaModel, cache: &KvCache) -> bool {
    let cfg = &model.config;
    if legacy_mode() || !hostops().has_ssm_mix() {
        return false;
    }
    let n_embd = cfg.n_embd as usize;
    let head_dim = cfg.head_dim() as usize;
    for (l, layer) in model.layers.iter().enumerate() {
        let attn = !cfg.architecture.is_hybrid() || layer.layer_type == LayerType::Attention;
        if attn {
            let n_head_kv = cfg
                .layer_n_head_kv
                .get(l)
                .copied()
                .filter(|&x| x > 0)
                .unwrap_or(cfg.n_head_kv) as usize;
            if n_head_kv * head_dim != cache.n_embd_kv {
                return false;
            }
        }
        if layer.attn_norm.n_elements() != n_embd {
            return false;
        }
    }
    true
}

thread_local! {
    /// Test-only (gemm_rows, att_rows) override for the calling thread; 0 = default.
    static CHUNK_OVERRIDE: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// Force the batched-prefill chunking on this thread (tests): `gemm_rows` / `att_rows`, 0 = default.
#[cfg(test)]
pub(crate) fn set_thread_prefill_chunking(gemm_rows: usize, att_rows: usize) {
    CHUNK_OVERRIDE.with(|c| c.set((gemm_rows, att_rows)));
}

fn env_rows(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&r| r > 0)
}

/// Bytes per prompt row of the prefill activation scratch (see [`DeviceScratch::new`]) plus
/// the backend's f16 activation copy (`rows × k × 2`, k ≤ max(n_ff, d_inner, q_dim)).
fn prefill_row_bytes(model: &LlamaModel, kv_dim: usize) -> usize {
    let cfg = &model.config;
    let n_embd = cfg.n_embd as usize;
    let q_dim = (cfg.n_head as usize * cfg.head_dim as usize).max(n_embd);
    let n_ff = cfg
        .layer_n_ff
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(cfg.n_ff)
        .max(1) as usize;
    let (d_in_proj, d_inner) = cfg
        .ssm
        .map(|s| (s.d_in_proj() as usize, s.d_inner as usize))
        .unwrap_or((0, 0));
    let f32s = n_embd + 3 * q_dim + 2 * kv_dim + 2 * n_ff + d_in_proj + d_inner;
    f32s * 4 + n_ff.max(d_inner).max(q_dim) * 2 + 4
}

/// Sliding-window length of attention layer `layer_idx` (0 = full causal context).
fn layer_window(cfg: &crate::config::ModelConfig, layer_idx: usize) -> usize {
    if cfg.sliding_window > 0 && (!cfg.sliding_window_pattern || layer_idx.is_multiple_of(2)) {
        cfg.sliding_window as usize
    } else {
        0
    }
}

/// Whether batched prefill attention takes a flash kernel: the mode is
/// [`PrefillAttn::Flash`](super::cuda_attention::PrefillAttn) (tensor-core) or `FlashScalar`
/// and every attention layer's shape is [`fa_supported`](super::cuda_attention::fa_supported).
/// Otherwise all layers use the old `attn_rows` sub-block path.
pub fn flash_prefill_for_model(model: &LlamaModel) -> bool {
    use super::cuda_attention::{fa_supported, prefill_attn_mode};
    if !prefill_attn_mode().is_flash() {
        return false;
    }
    let cfg = &model.config;
    let n_head = cfg.n_head as usize;
    let head_dim = cfg.head_dim() as usize;
    model.layers.iter().enumerate().all(|(l, layer)| {
        let attn = !cfg.architecture.is_hybrid() || layer.layer_type == LayerType::Attention;
        let n_head_kv = cfg
            .layer_n_head_kv
            .get(l)
            .copied()
            .filter(|&x| x > 0)
            .unwrap_or(cfg.n_head_kv) as usize;
        !attn || fa_supported(n_head, n_head_kv, head_dim, layer_window(cfg, l))
    })
}

/// `(gemm_rows, att_rows)`: GEMM chunk rows (≤ [`PREFILL_CHUNK`], within
/// [`PREFILL_ACT_SCRATCH_BYTES`]) and attention sub-block rows (score scratch within
/// [`PREFILL_ATT_SCRATCH_BYTES`]), both ≤ `n_tok`. With flash prefill attention the score
/// scratch is unused, so `att_rows` is 1 (minimal allocation); `gemm_rows` (the chunking) is
/// the same either way.
fn prefill_rows(model: &LlamaModel, n_ctx: usize, kv_dim: usize, n_tok: usize) -> (usize, usize) {
    prefill_rows_for(model, n_ctx, kv_dim, n_tok, flash_prefill_for_model(model))
}

fn prefill_rows_for(
    model: &LlamaModel,
    n_ctx: usize,
    kv_dim: usize,
    n_tok: usize,
    flash: bool,
) -> (usize, usize) {
    let n_head = model.config.n_head.max(1) as usize;
    let (o_gemm, o_att) = CHUNK_OVERRIDE.with(|c| c.get());
    let act_cap = (PREFILL_ACT_SCRATCH_BYTES / prefill_row_bytes(model, kv_dim).max(1)).max(1);
    let gemm = if o_gemm > 0 {
        o_gemm
    } else {
        env_rows("ARTIFICER_PREFILL_CHUNK")
            .unwrap_or(PREFILL_CHUNK)
            .min(act_cap)
    };
    let gemm = n_tok.min(gemm).max(1);
    if flash {
        return (gemm, 1);
    }
    let att_cap = (PREFILL_ATT_SCRATCH_BYTES / (n_head * n_ctx * 4).max(1)).max(1);
    let att = if o_att > 0 {
        o_att
    } else {
        env_rows("ARTIFICER_PREFILL_ATT_ROWS").unwrap_or(att_cap)
    };
    (gemm, att.min(gemm).max(1))
}

/// Batched prefill of `tokens` at positions `pos0..` on the device. Writes the last token's
/// logits into `hs.logits` (one sync at the end). Caller checked
/// [`batched_prefill_supported`]. KV rows for every position are written; SSM state is
/// advanced by the multi-token `ssm_mix`.
///
/// Stop: with `cancel`, the check runs before the first chunk and after every
/// non-last chunk. Kernels are queued asynchronously, so before each between-chunk check
/// the stream is synced: the host can't run ahead of the GPU, and a Stop lands within one
/// chunk of device work. The sync only drains the queue (no numeric change); it costs one
/// launch gap per chunk (≤ 8 at 8k with 1024-row chunks). Without `cancel` nothing is added.
/// On cancel, `cache.len` / device KV length cover whole processed chunks only, the device
/// state is put back in `cache`, and `hs.logits` is left untouched (not sampled).
#[allow(clippy::too_many_arguments)]
pub fn prefill_batched(
    model: &LlamaModel,
    cb: &CudaBackend,
    tokens: &[u32],
    pos0: usize,
    cache: &mut KvCache,
    mut ssm: Option<&mut SsmCache>,
    hs: &mut ForwardScratch,
    mut cancel: PrefillCancel<'_>,
) -> Result<PrefillStatus> {
    let n_ctx = cache.n_ctx;
    if tokens.is_empty() {
        return Err(CoreError::Other("prefill: empty token list".into()));
    }
    if pos0 + tokens.len() > n_ctx {
        return Err(CoreError::ContextExceeded {
            need: pos0 + tokens.len(),
            max: n_ctx,
        });
    }
    let mut st = match cache.device.take() {
        Some(s) => s,
        None => Box::new(CudaForwardState::new(cb, model, cache)?),
    };
    let (rows, att_rows) = prefill_rows(model, n_ctx, cache.n_embd_kv, tokens.len());
    let (have_rows, have_att) = st
        .prefill
        .as_ref()
        .map_or((0, 0), |p| (p.max_batch, p.att_rows));
    if have_rows < rows || have_att < att_rows {
        st.prefill = None; // free the old one first
                           // Keep the largest shape seen so far so alternating prompt sizes don't reallocate.
        st.prefill = Some(DeviceScratch::new(
            cb,
            model,
            n_ctx,
            cache.n_embd_kv,
            rows.max(have_rows),
            att_rows.max(have_att),
        )?);
    }
    if let Some(c) = cancel.as_mut() {
        if c() {
            cache.device = Some(st);
            return Ok(PrefillStatus::Cancelled { processed: 0 });
        }
    }
    let mut res = Ok(PrefillStatus::Complete);
    let mut done = 0usize;
    for chunk in tokens.chunks(rows) {
        let last = done + chunk.len() == tokens.len();
        let r = prefill_chunk(
            model,
            cb,
            chunk,
            pos0 + done,
            &mut st,
            ssm.as_deref_mut(),
            hs,
            last,
        );
        if let Err(e) = r {
            res = Err(e);
            break;
        }
        done += chunk.len();
        st.kv.len = st.kv.len.max(pos0 + done);
        cache.len = cache.len.max(pos0 + done);
        if !last {
            if let Some(c) = cancel.as_mut() {
                if let Err(e) = cb.sync() {
                    res = Err(e);
                    break;
                }
                if c() {
                    res = Ok(PrefillStatus::Cancelled { processed: done });
                    break;
                }
            }
        }
    }
    cache.device = Some(st);
    res
}

#[allow(clippy::too_many_arguments)]
fn prefill_chunk(
    model: &LlamaModel,
    cb: &CudaBackend,
    tokens: &[u32],
    pos0: usize,
    st: &mut CudaForwardState,
    mut ssm: Option<&mut SsmCache>,
    hs: &mut ForwardScratch,
    last: bool,
) -> Result<()> {
    let cfg = &model.config;
    let n = tokens.len();
    let n_embd = cfg.n_embd as usize;
    let ops = hostops();
    let ps = st.prefill.as_ref().expect("prefill scratch allocated");
    let x = ps.x.ptr().prefix(n * n_embd);

    if !ops.embed(cb, model, tokens, x)? {
        // Host gather (table did not fit on the device): one upload for all rows.
        let data = model.tensor_bytes(&model.token_embd)?;
        let mut rows = vec![0.0f32; n * n_embd];
        for (r, &t) in tokens.iter().enumerate() {
            if t as usize >= cfg.n_vocab as usize {
                return Err(CoreError::Other(format!(
                    "token id {t} out of vocab {}",
                    cfg.n_vocab
                )));
            }
            let dst = &mut rows[r * n_embd..(r + 1) * n_embd];
            dequantize(data, model.token_embd.ty, t as usize * n_embd, n_embd, dst)?;
            if cfg.scale_embeddings {
                scale_inplace(dst, (n_embd as f32).sqrt());
            }
        }
        cb.upload_f32(&rows, x)?;
    }

    let hybrid = cfg.architecture.is_hybrid();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let lt = if hybrid {
            layer.layer_type
        } else {
            LayerType::Dense
        };
        match lt {
            LayerType::Ssm => {
                let ssm = ssm.as_deref_mut().ok_or_else(|| {
                    CoreError::Other("hybrid model requires SsmCache in prefill".into())
                })?;
                prefill_ssm(model, cb, layer_idx, st, ssm, n)?;
            }
            LayerType::Attention => prefill_attention(model, cb, layer_idx, pos0, st, n)?,
            LayerType::Ffn => prefill_ffn(model, cb, layer_idx, st, n)?,
            LayerType::Dense => {
                prefill_attention(model, cb, layer_idx, pos0, st, n)?;
                prefill_ffn(model, cb, layer_idx, st, n)?;
            }
        }
    }
    if !last {
        return Ok(());
    }
    // Output norm + LM head on the last row only (decode scratch row 0).
    let ps = st.prefill.as_ref().expect("prefill scratch allocated");
    let xl = ps.x.ptr().row(n - 1, n_embd);
    let xb = st.scratch.xb.ptr();
    dev_norm(
        cb,
        model,
        &model.output_norm,
        model.output_norm_b.as_ref(),
        xl,
        xb.prefix(n_embd),
        1,
        n_embd,
    )?;
    let logits = st.scratch.logits.ptr();
    matvec(cb, model, &model.output, xb, logits)?;
    let mut softcap_on_host = false;
    if cfg.logit_scale > 0.0 && !ops.softcap(cb, logits, cfg.logit_scale)? {
        softcap_on_host = true;
    }
    finish_logits(cb, logits, &mut hs.logits)?;
    if softcap_on_host {
        softcap_inplace(&mut hs.logits, cfg.logit_scale);
    }
    Ok(())
}

/// Batched prefill has no host-fallback path: a declined hook is an error
/// (`batched_prefill_supported` screens the known cases up front).
fn declined(what: &str) -> CoreError {
    CoreError::Backend(format!(
        "batched prefill: device op `{what}` declined (no host fallback for N rows)"
    ))
}

#[allow(clippy::too_many_arguments)]
fn dev_norm(
    cb: &CudaBackend,
    model: &LlamaModel,
    w: &WeightTensor,
    b: Option<&WeightTensor>,
    x: DevPtr,
    out: DevPtr,
    rows: usize,
    dim: usize,
) -> Result<()> {
    if hostops().norm(cb, model, w, b, x, out, rows, dim)? {
        Ok(())
    } else {
        Err(declined("norm"))
    }
}

fn dev_bias(
    cb: &CudaBackend,
    model: &LlamaModel,
    bias: Option<&WeightTensor>,
    out: DevPtr,
    rows: usize,
    dim: usize,
) -> Result<()> {
    let Some(bias) = bias else { return Ok(()) };
    if hostops().add_bias(cb, model, bias, out.prefix(rows * dim), rows, dim)? {
        Ok(())
    } else {
        Err(declined("add_bias"))
    }
}

fn dev_add(cb: &CudaBackend, a: DevPtr, b: DevPtr, n: usize) -> Result<()> {
    if hostops().add_inplace(cb, a.prefix(n), b.prefix(n))? {
        Ok(())
    } else {
        Err(declined("add_inplace"))
    }
}

fn gemm(
    cb: &CudaBackend,
    model: &LlamaModel,
    w: &WeightTensor,
    x: DevPtr,
    rows: usize,
    y: DevPtr,
) -> Result<()> {
    let data = model.tensor_bytes(w)?;
    cb.gemm_dev(data, w.ty, w.k(), w.n(), x, rows, y)
}

fn gemm_shared(
    cb: &CudaBackend,
    model: &LlamaModel,
    x: DevPtr,
    rows: usize,
    outs: &[(&WeightTensor, DevPtr)],
) -> Result<()> {
    let k = outs[0].0.k();
    let mut jobs = Vec::with_capacity(outs.len());
    for (w, y) in outs {
        if w.k() != k {
            return Err(CoreError::Shape(format!(
                "gemm_shared: k mismatch {} vs {k}",
                w.k()
            )));
        }
        jobs.push(MatvecJob {
            weight: model.tensor_bytes(w)?,
            ty: w.ty,
            n: w.n(),
            y: *y,
        });
    }
    cb.gemm_dev_multi(x, k, rows, &jobs)
}

fn prefill_attention(
    model: &LlamaModel,
    cb: &CudaBackend,
    layer_idx: usize,
    pos0: usize,
    st: &mut CudaForwardState,
    n: usize,
) -> Result<()> {
    let cfg = &model.config;
    let layer = &model.layers[layer_idx];
    let ops = hostops();
    let n_embd = cfg.n_embd as usize;
    let n_head = cfg.n_head as usize;
    let n_head_kv = cfg
        .layer_n_head_kv
        .get(layer_idx)
        .copied()
        .filter(|&x| x > 0)
        .unwrap_or(cfg.n_head_kv) as usize;
    let head_dim = cfg.head_dim() as usize;
    let n_rot = cfg.n_rot as usize;
    let q_dim = n_head * head_dim;
    let kv_n = n_head_kv * head_dim;
    let ps = st.prefill.as_ref().expect("prefill scratch allocated");
    let x = ps.x.ptr();
    let xb = ps.xb.ptr();
    let xb2 = ps.xb2.ptr();
    let q = ps.q.ptr();

    dev_norm(
        cb,
        model,
        &layer.attn_norm,
        layer.attn_norm_b.as_ref(),
        x,
        xb,
        n,
        n_embd,
    )?;
    let attn_q = req(&layer.attn_q, "attn_q")?;
    let attn_k = req(&layer.attn_k, "attn_k")?;
    let attn_v = req(&layer.attn_v, "attn_v")?;
    let attn_o = req(&layer.attn_output, "attn_output")?;

    // f32 cache: K/V GEMMs write the N cache rows directly (kv_n == kv_dim checked up front).
    let direct = st.kv.elem == KvElem::F32;
    let (k_dst, v_dst) = if direct {
        (
            st.kv.k_rows(layer_idx, pos0, n)?,
            st.kv.v_rows(layer_idx, pos0, n)?,
        )
    } else {
        (ps.k.ptr(), ps.v.ptr())
    };
    gemm_shared(
        cb,
        model,
        xb,
        n,
        &[(attn_q, q), (attn_k, k_dst), (attn_v, v_dst)],
    )?;
    dev_bias(cb, model, layer.attn_q_b.as_ref(), q, n, q_dim)?;
    dev_bias(cb, model, layer.attn_k_b.as_ref(), k_dst, n, kv_n)?;
    dev_bias(cb, model, layer.attn_v_b.as_ref(), v_dst, n, kv_n)?;

    if !cfg.architecture.is_hybrid() && n_rot > 0 && cfg.rope_freq_base > 0.0 {
        let fb = cfg.rope_freq_base;
        if !ops.rope(cb, q, n, q_dim, n_head, head_dim, pos0, fb, n_rot)? {
            return Err(declined("rope"));
        }
        if !ops.rope(cb, k_dst, n, kv_n, n_head_kv, head_dim, pos0, fb, n_rot)? {
            return Err(declined("rope"));
        }
    }
    let att_scale = if cfg.scale_embeddings {
        let s = 1.0 / (head_dim as f32).sqrt();
        if !ops.scale(cb, q.prefix(n * q_dim), s)? {
            return Err(declined("scale"));
        }
        1.0
    } else {
        1.0 / (head_dim as f32).sqrt()
    };
    if !direct {
        st.kv.store_rows(cb, layer_idx, pos0, n, k_dst, false)?;
        st.kv.store_rows(cb, layer_idx, pos0, n, v_dst, true)?;
    }
    let window = layer_window(cfg, layer_idx);
    // Causal: row r attends positions ..= pos0 + r. All n K/V rows of this chunk are in the
    // cache already. Flash: one launch over all n rows, no score scratch.
    // Old: the query rows run in sub-blocks of `att_rows` (score scratch size) without
    // changing any row's result: same kernel, same per-row math.
    if flash_prefill_for_model(model) {
        super::cuda_attention::attn_prefill_flash_dev(
            cb,
            super::cuda_attention::prefill_attn_mode(),
            q.prefix(n * q_dim),
            &st.kv,
            layer_idx,
            xb2.prefix(n * q_dim),
            n,
            pos0,
            n_head,
            n_head_kv,
            head_dim,
            q_dim,
            att_scale,
            cfg.attn_logit_softcap,
        )?;
    } else {
        let att_rows = ps.att_rows.max(1);
        let mut r0 = 0;
        while r0 < n {
            let m = (n - r0).min(att_rows);
            super::cuda_attention::attn_rows_dev(
                cb,
                q.offset(r0 * q_dim, m * q_dim),
                &st.kv,
                layer_idx,
                xb2.offset(r0 * q_dim, m * q_dim),
                ps.att.ptr(),
                m,
                pos0 + r0,
                n_head,
                n_head_kv,
                head_dim,
                q_dim,
                window,
                att_scale,
                cfg.attn_logit_softcap,
            )?;
            r0 += m;
        }
    }
    gemm(cb, model, attn_o, xb2, n, xb)?;
    dev_bias(cb, model, layer.attn_output_b.as_ref(), xb, n, attn_o.n())?;
    dev_add(cb, x, xb, n * n_embd)
}

fn prefill_ffn(
    model: &LlamaModel,
    cb: &CudaBackend,
    layer_idx: usize,
    st: &mut CudaForwardState,
    n: usize,
) -> Result<()> {
    let cfg = &model.config;
    let layer = &model.layers[layer_idx];
    let ops = hostops();
    let n_embd = cfg.n_embd as usize;
    let ps = st.prefill.as_ref().expect("prefill scratch allocated");
    let (x, xb, hb, hb2) = (ps.x.ptr(), ps.xb.ptr(), ps.hb.ptr(), ps.hb2.ptr());
    let ffn_up = req(&layer.ffn_up, "ffn_up")?;
    let ffn_down = req(&layer.ffn_down, "ffn_down")?;
    let n_ff = ffn_up.n();
    match layer.ffn_norm.as_ref() {
        Some(nw) => dev_norm(cb, model, nw, layer.ffn_norm_b.as_ref(), x, xb, n, n_embd)?,
        None => dev_norm(
            cb,
            model,
            &layer.attn_norm,
            layer.attn_norm_b.as_ref(),
            x,
            xb,
            n,
            n_embd,
        )?,
    }
    match cfg.ffn_activation {
        FfnActivation::SiluGated | FfnActivation::GeluGated => {
            let gate = req(&layer.ffn_gate, "ffn_gate.weight")?;
            gemm_shared(cb, model, xb, n, &[(gate, hb), (ffn_up, hb2)])?;
            dev_bias(cb, model, layer.ffn_up_b.as_ref(), hb2, n, n_ff)?;
            if !ops.act(
                cb,
                cfg.ffn_activation,
                hb.prefix(n * n_ff),
                Some(hb2.prefix(n * n_ff)),
            )? {
                return Err(declined("act"));
            }
        }
        FfnActivation::ReluSqrSequential => {
            gemm(cb, model, ffn_up, xb, n, hb)?;
            dev_bias(cb, model, layer.ffn_up_b.as_ref(), hb, n, n_ff)?;
            if !ops.act(cb, cfg.ffn_activation, hb.prefix(n * n_ff), None)? {
                return Err(declined("act"));
            }
        }
    }
    gemm(cb, model, ffn_down, hb, n, xb)?;
    dev_bias(cb, model, layer.ffn_down_b.as_ref(), xb, n, ffn_down.n())?;
    dev_add(cb, x, xb, n * n_embd)
}

fn prefill_ssm(
    model: &LlamaModel,
    cb: &CudaBackend,
    layer_idx: usize,
    st: &mut CudaForwardState,
    ssm: &mut SsmCache,
    n: usize,
) -> Result<()> {
    let cfg = &model.config;
    let layer = &model.layers[layer_idx];
    let n_embd = cfg.n_embd as usize;
    let ps = st.prefill.as_ref().expect("prefill scratch allocated");
    let (x, xb) = (ps.x.ptr(), ps.xb.ptr());
    let (zx, y) = (ps.ssm_zx.ptr(), ps.ssm_y.ptr());
    dev_norm(cb, model, &layer.attn_norm, None, x, xb, n, n_embd)?;
    gemm(cb, model, req(&layer.ssm_in, "ssm_in")?, xb, n, zx)?;
    // Multi-token conv + scan + gate/group-norm (3 launches), state advanced by n.
    if !hostops().ssm_mix(cb, ssm, model, layer_idx, zx, y, n)? {
        return Err(declined("ssm_mix"));
    }
    gemm(cb, model, req(&layer.ssm_out, "ssm_out")?, y, n, xb)?;
    dev_add(cb, x, xb, n * n_embd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::config::{Architecture, ModelConfig, NormKind};
    use crate::forward::layer::multi_head_attn;

    fn dev() -> Option<CudaBackend> {
        let mut b = CudaBackend::try_new().ok()?;
        b.init_device().ok()?;
        Some(b)
    }

    struct Lcg(u32);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            ((self.0 >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        }
    }

    fn cfg(n_head: u32, n_head_kv: u32, head_dim: u32, n_ctx: u32, n_layer: u32) -> ModelConfig {
        let n_embd = n_head * head_dim;
        ModelConfig {
            architecture: Architecture::Llama,
            arch_str: "llama".into(),
            name: Some("kv-test".into()),
            n_vocab: 32,
            n_ctx,
            n_embd,
            n_layer,
            n_ff: n_embd * 2,
            n_head,
            n_head_kv,
            n_rot: head_dim,
            head_dim,
            rms_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_type: crate::config::RopeType::Norm,
            bos_token_id: 1,
            eos_token_id: 2,
            add_bos_token: false,
            add_eos_token: false,
            ffn_activation: FfnActivation::SiluGated,
            norm_kind: NormKind::Rms,
            scale_embeddings: false,
            logit_scale: 0.0,
            attn_logit_softcap: 0.0,
            sliding_window: 0,
            sliding_window_pattern: false,
            layer_types: vec![],
            layer_n_ff: vec![],
            layer_n_head_kv: vec![],
            ssm: None,
            moe: None,
        }
    }

    fn host_scratch(c: &ModelConfig) -> ForwardScratch {
        let d = c.n_embd as usize;
        let q_dim = (c.n_head * c.head_dim) as usize;
        let n_kv = c.n_embd_kv() as usize;
        ForwardScratch {
            x: vec![0.0; d],
            xb: vec![0.0; d.max(q_dim)],
            xb2: vec![0.0; q_dim],
            q: vec![0.0; q_dim],
            k: vec![0.0; n_kv],
            v: vec![0.0; n_kv],
            att: vec![0.0; c.n_head as usize * c.n_ctx as usize],
            hb: vec![0.0; c.n_ff as usize],
            hb2: vec![0.0; c.n_ff as usize],
            logits: vec![0.0; c.n_vocab as usize],
            kv_row: vec![0.0; n_kv],
            ssm_zx: Vec::new(),
            ssm_xbc: Vec::new(),
            ssm_y: Vec::new(),
            ssm_tmp: Vec::new(),
            norm_w: vec![0.0; d],
            norm_b: vec![0.0; d],
            moe_logits: Vec::new(),
            moe_acc: Vec::new(),
            moe_out: Vec::new(),
            moe_sel: Vec::new(),
        }
    }

    /// Device KV append/read must equal the host KvCache bit-for-bit (f32 and f16).
    #[test]
    fn device_kv_append_read_matches_host() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        for elem in [KvElem::F32, KvElem::F16] {
            let c = cfg(4, 2, 16, 32, 3);
            let mut host = KvCache::with_elem(&c, elem);
            let kv_dim = host.n_embd_kv;
            let kv =
                DeviceKvCache::new(&cb, &[true, false, true], host.n_ctx, kv_dim, elem).unwrap();
            assert!(
                kv.layer_raw(1).is_err(),
                "non-attention layer must have no storage"
            );
            let staging = cb.alloc_f32(kv_dim).unwrap();
            let mut r = Lcg(7);
            for pos in 0..6 {
                for layer in [0usize, 2] {
                    let k: Vec<f32> = (0..kv_dim).map(|_| r.f() * 3.0).collect();
                    let v: Vec<f32> = (0..kv_dim).map(|_| r.f() * 3.0).collect();
                    host.write(layer, pos, &k, &v);
                    cb.upload_f32(&k, staging.ptr()).unwrap();
                    kv.store_rows(&cb, layer, pos, 1, staging.ptr(), false)
                        .unwrap();
                    cb.upload_f32(&v, staging.ptr()).unwrap();
                    kv.store_rows(&cb, layer, pos, 1, staging.ptr(), true)
                        .unwrap();
                }
            }
            let mut row = vec![0.0f32; kv_dim];
            for layer in [0usize, 2] {
                let dk = kv.read_rows(&cb, layer, 0, 6, false).unwrap();
                let dv = kv.read_rows(&cb, layer, 0, 6, true).unwrap();
                for pos in 0..6 {
                    host.copy_k_into(layer, pos, &mut row);
                    assert_eq!(
                        &dk[pos * kv_dim..(pos + 1) * kv_dim],
                        &row[..],
                        "{elem:?} K l{layer} p{pos}"
                    );
                    host.copy_v_into(layer, pos, &mut row);
                    assert_eq!(
                        &dv[pos * kv_dim..(pos + 1) * kv_dim],
                        &row[..],
                        "{elem:?} V l{layer} p{pos}"
                    );
                }
            }
        }
    }

    /// Device attention over the device KV vs host `multi_head_attn` over the host KV,
    /// several decode steps; GQA, sliding window, softcap, f32 + f16.
    #[test]
    fn device_kv_attention_matches_host_over_steps() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let cases = [
            (KvElem::F32, 8usize, 2usize, 32usize, 0usize, 0.0f32),
            (KvElem::F16, 8, 2, 32, 0, 0.0),
            (KvElem::F32, 4, 4, 64, 5, 0.0),
            (KvElem::F32, 6, 3, 16, 0, 30.0),
        ];
        for &(elem, n_head, n_head_kv, head_dim, window, softcap) in &cases {
            let n_ctx = 48usize;
            let c = cfg(
                n_head as u32,
                n_head_kv as u32,
                head_dim as u32,
                n_ctx as u32,
                1,
            );
            let mut host = KvCache::with_elem(&c, elem);
            let kv_dim = host.n_embd_kv;
            let kv = DeviceKvCache::new(&cb, &[true], n_ctx, kv_dim, elem).unwrap();
            let q_dim = n_head * head_dim;
            let mut hs = host_scratch(&c);
            let dq = cb.alloc_f32(q_dim).unwrap();
            let dout = cb.alloc_f32(q_dim).unwrap();
            let datt = cb.alloc_f32(n_head * n_ctx).unwrap();
            let dk = cb.alloc_f32(kv_dim).unwrap();
            let mut r = Lcg(head_dim as u32 * 13 + n_head as u32);
            let scale = 1.0 / (head_dim as f32).sqrt();
            for pos in 0..20 {
                let k: Vec<f32> = (0..kv_dim).map(|_| r.f()).collect();
                let v: Vec<f32> = (0..kv_dim).map(|_| r.f()).collect();
                host.write(0, pos, &k, &v);
                cb.upload_f32(&k, dk.ptr()).unwrap();
                kv.store_rows(&cb, 0, pos, 1, dk.ptr(), false).unwrap();
                cb.upload_f32(&v, dk.ptr()).unwrap();
                kv.store_rows(&cb, 0, pos, 1, dk.ptr(), true).unwrap();
                for qv in hs.q.iter_mut() {
                    *qv = r.f() * 4.0;
                }
                cb.upload_f32(&hs.q, dq.ptr()).unwrap();
                let start = if window > 0 {
                    pos.saturating_sub(window - 1)
                } else {
                    0
                };
                multi_head_attn(
                    &mut hs, &host, 0, pos, start, n_head, n_head_kv, head_dim, n_ctx, scale,
                    softcap,
                );
                super::super::cuda_attention::attn_rows_dev(
                    &cb,
                    dq.ptr(),
                    &kv,
                    0,
                    dout.ptr(),
                    datt.ptr(),
                    1,
                    pos,
                    n_head,
                    n_head_kv,
                    head_dim,
                    q_dim,
                    window,
                    scale,
                    softcap,
                )
                .unwrap();
                let mut got = vec![0.0f32; q_dim];
                cb.download_f32(dout.ptr(), &mut got).unwrap();
                for i in 0..q_dim {
                    let want = hs.xb2[i];
                    assert!(
                        (got[i] - want).abs() <= 1e-5 + 1e-4 * want.abs(),
                        "{elem:?} h{n_head}/{n_head_kv} d{head_dim} w{window} cap{softcap} pos{pos} i{i}: dev {} host {want}",
                        got[i]
                    );
                }
            }
        }
    }

    fn tiny_vicuna() -> Option<std::path::PathBuf> {
        crate::backend::test_model("tiny-vicuna-1b.q4_k_m.gguf")
    }

    fn nemotron() -> Option<std::path::PathBuf> {
        crate::backend::test_model("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf")
    }

    /// Resident decode vs (a) the legacy per-matvec CUDA path and (b) the CPU forward on a
    /// real model for 12 steps (7 prompt tokens + 5 greedy). (a) same device kernels → tight;
    /// (b) CPU quantizes activations to Q8_K/Q8_0, so logits differ by ~0.2 (BOS-only step 0 by
    /// ~1.0, which is never sampled); greedy tokens must match from step 1 on.
    #[test]
    fn resident_decode_matches_legacy_and_cpu_tiny_vicuna() {
        let Some(path) = tiny_vicuna() else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        // Both matvec kernel families: f32-activation `mv_*` (resident == legacy to ~1e-5)
        // and mmvq (Q8 activations: a 1e-6 difference upstream can flip a Q8 rounding,
        // so resident vs legacy is ~0.1 — still well inside the CPU tolerance; argmax equal).
        for mmvq in [false, true] {
            cb.set_mmvq(mmvq);
            let leg_tol = if mmvq { 0.5 } else { 1e-2 };
            let mut model = LlamaModel::load(&path).unwrap();
            model.config.n_ctx = 64;
            let cpu = crate::backend::CpuBackend;
            let mut cache_cpu = KvCache::new(&model.config);
            let mut cache_gpu = KvCache::new(&model.config);
            let mut cache_leg = KvCache::new(&model.config);
            let mut s_cpu = ForwardScratch::new(&model);
            let mut s_gpu = ForwardScratch::new(&model);
            let mut s_leg = ForwardScratch::new(&model);
            let prompt = [1u32, 1724, 2927, 338, 278, 14744, 29973];
            let am = |l: &[f32]| {
                l.iter()
                    .enumerate()
                    .fold(
                        (0usize, f32::MIN),
                        |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                    )
                    .0 as u32
            };
            let mut next = 0u32;
            let rt_before = cb.host_fallback_roundtrips();
            for step in 0..12 {
                let tok = if step < prompt.len() {
                    prompt[step]
                } else {
                    next
                };
                let lc = model
                    .forward(tok, step, &mut cache_cpu, None, &mut s_cpu, true, &cpu)
                    .unwrap()
                    .unwrap()
                    .to_vec();
                let lg = model
                    .forward(tok, step, &mut cache_gpu, None, &mut s_gpu, true, &cb)
                    .unwrap()
                    .unwrap()
                    .to_vec();
                set_thread_legacy(true);
                let ll = model
                    .forward(tok, step, &mut cache_leg, None, &mut s_leg, true, &cb)
                    .unwrap()
                    .unwrap()
                    .to_vec();
                set_thread_legacy(false);
                let maxd = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max)
                };
                let (d_leg, d_cpu) = (maxd(&lg, &ll), maxd(&lg, &lc));
                eprintln!(
                    "step {step}: tok cpu {} res {} legacy {} | max|res-legacy| {d_leg:.2e} max|res-cpu| {d_cpu:.3}",
                    am(&lc), am(&lg), am(&ll)
                );
                assert!(
                    d_leg < leg_tol,
                    "step {step} (mmvq {mmvq}): resident vs legacy CUDA logits differ by {d_leg}"
                );
                assert_eq!(am(&lg), am(&ll), "step {step}: resident vs legacy argmax");
                if step >= 1 {
                    assert!(
                        d_cpu < 1.0,
                        "step {step}: resident vs CPU logits differ by {d_cpu}"
                    );
                    assert_eq!(
                        am(&lg),
                        am(&lc),
                        "step {step}: resident vs CPU greedy token"
                    );
                }
                next = am(&lc);
            }
            assert!(
                cache_leg.device.is_none(),
                "legacy path must not create device state"
            );
            let st = cache_gpu.device.as_ref().expect("device state created");
            assert_eq!(st.kv.len, 12);
            assert_eq!(cache_gpu.len, 12);
            // Host-fallback spots are the only round trips: 22 layers × (2 norms + 2 adds + 1 act) + final norm + embed.
            let per_tok = (cb.host_fallback_roundtrips() - rt_before) as f64 / 12.0;
            eprintln!("host-fallback round trips per token: {per_tok}");
            // Device KV row (layer 0, pos 11) vs host KV row: within matvec numerics.
            let dk = st.kv.read_rows(&cb, 0, 11, 1, false).unwrap();
            let mut hk = vec![0.0f32; cache_cpu.n_embd_kv];
            cache_cpu.copy_k_into(0, 11, &mut hk);
            let kd = hk
                .iter()
                .zip(&dk)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let km = hk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!("device vs host KV row max diff {kd:.4} (max |k| {km:.2})");
            assert!(
                kd < 0.05 * km.max(1.0),
                "device KV row differs from host by {kd}"
            );
            cache_gpu.clear();
            assert_eq!(
                cache_gpu.device.as_ref().unwrap().kv.len,
                0,
                "clear() resets device KV length"
            );
        }
        assert_eq!(cb.kind(), crate::backend::BackendKind::Cuda);
    }

    /// Batched prefill vs token-by-token CUDA on tiny-vicuna: one-shot (12 tokens), a
    /// continuation split (7 + 5 at pos0 = 7), 300 tokens in one chunk, and 300 tokens with
    /// forced chunking (GEMM chunks of 128 rows with attention sub-blocks of 48 rows; GEMM
    /// chunks of 64 with no sub-blocking), covering multi-chunk and multi-sub-block causality.
    /// Last-token logits close + same argmax, then 4 greedy decode steps from each cache agree
    /// (KV rows of every prompt position were written correctly). Runs under every prefill
    /// attention mode (thread override); tiny-vicuna's head_dim 64 is not flash-supported, so
    /// `flash` / `flash_scalar` must fall back to the old path (flash on Nemotron:
    /// `batched_prefill_flash_matches_old_nemotron`).
    #[test]
    fn batched_prefill_matches_token_by_token_tiny_vicuna() {
        use super::super::cuda_attention::{set_thread_prefill_attn, PrefillAttn};
        let Some(path) = tiny_vicuna() else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut model = LlamaModel::load(&path).unwrap();
        model.config.n_ctx = 512;
        assert!(batched_prefill_supported(
            &model,
            &KvCache::new(&model.config)
        ));
        let am = |l: &[f32]| {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0 as u32
        };
        let mut r = Lcg(5);
        let long: Vec<u32> = (0..300)
            .map(|_| 100 + ((r.f() + 0.5) * 20000.0) as u32)
            .collect();
        let short = [
            1u32, 1724, 2927, 338, 278, 14744, 29973, 13, 29909, 29901, 450, 14744,
        ];
        let cases: Vec<(&str, &[u32], Vec<usize>, (usize, usize))> = vec![
            ("one-shot 12", &short, vec![12], (0, 0)),
            ("split 7+5", &short, vec![7, 5], (0, 0)),
            ("long 300", &long, vec![300], (0, 0)),
            ("long 300 gemm128/att48", &long, vec![300], (128, 48)),
            ("long 300 gemm64/att64", &long, vec![300], (64, 64)),
            (
                "split 100+200 gemm128/att48",
                &long,
                vec![100, 200],
                (128, 48),
            ),
        ];
        for (mode, (name, toks, splits, (g_rows, a_rows))) in [
            PrefillAttn::Old,
            PrefillAttn::FlashScalar,
            PrefillAttn::Flash,
        ]
        .into_iter()
        .flat_map(|m| cases.iter().cloned().map(move |c| (m, c)))
        {
            set_thread_prefill_attn(Some(mode));
            assert!(
                !flash_prefill_for_model(&model),
                "tiny-vicuna (d64) must not take flash"
            );
            let name = format!("{mode:?} {name}");
            let mut c_tok = KvCache::new(&model.config);
            let mut s_tok = ForwardScratch::new(&model);
            let lt = model
                .prefill_token_by_token(toks, 0, &mut c_tok, None, &mut s_tok, &cb)
                .unwrap()
                .to_vec();
            let mut c_b = KvCache::new(&model.config);
            let mut s_b = ForwardScratch::new(&model);
            let gemm_before = cb.gemm_ex_launch_count() + cb.mmvq_launch_count();
            let mut p0 = 0;
            let mut lb = Vec::new();
            set_thread_prefill_chunking(g_rows, a_rows);
            for n in splits {
                lb = model
                    .prefill(&toks[p0..p0 + n], p0, &mut c_b, None, &mut s_b, &cb)
                    .unwrap()
                    .to_vec();
                p0 += n;
            }
            set_thread_prefill_chunking(0, 0);
            assert!(cb.gemm_ex_launch_count() + cb.mmvq_launch_count() > gemm_before);
            assert_eq!(c_b.len, toks.len());
            let d = lb
                .iter()
                .zip(&lt)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{name}: max|batched - token| {d:.3e}, argmax {} / {}",
                am(&lb),
                am(&lt)
            );
            assert!(d < 0.5, "{name}: batched vs token logits differ by {d}");
            assert_eq!(am(&lb), am(&lt), "{name}: first token");
            let (mut tb, mut tt) = (am(&lb), am(&lt));
            for i in 0..4 {
                let pos = toks.len() + i;
                let a = model
                    .forward(tb, pos, &mut c_b, None, &mut s_b, true, &cb)
                    .unwrap()
                    .unwrap()
                    .to_vec();
                let b = model
                    .forward(tt, pos, &mut c_tok, None, &mut s_tok, true, &cb)
                    .unwrap()
                    .unwrap()
                    .to_vec();
                tb = am(&a);
                tt = am(&b);
                assert_eq!(tb, tt, "{name}: decode step {i} after prefill");
            }
        }
        set_thread_prefill_attn(None);
    }

    /// Model level: Nemotron3-Nano-4B batched prefill with each flash kernel
    /// (scalar, tensor-core) vs old prefill attention on the same prompt, single chunk (300 tokens) and multi-chunk with continuation
    /// (gemm chunks of 512 rows over 1100 + 37 tokens: chunk boundaries at arbitrary pos0).
    /// Last-token logits close + same argmax, then 4 greedy decode steps agree. f16 KV (default).
    #[test]
    fn batched_prefill_flash_matches_old_nemotron() {
        use super::super::cuda_attention::{
            flash_mma_launch_count, flash_prefill_launch_count, set_thread_prefill_attn,
            PrefillAttn,
        };
        let Some(path) = nemotron() else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut model = LlamaModel::load(path).unwrap();
        model.config.n_ctx = 2048;
        if !batched_prefill_supported(&model, &KvCache::new(&model.config)) {
            eprintln!("skip: batched prefill unsupported for Nemotron in this build");
            return;
        }
        let am = |l: &[f32]| {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0 as u32
        };
        let mut r = Lcg(11);
        let toks: Vec<u32> = (0..1137)
            .map(|_| 1000 + ((r.f() + 0.5) * 30000.0) as u32)
            .collect();
        let cases: [(&str, &[u32], &[usize], usize); 2] = [
            ("300 one chunk", &toks[..300], &[300], 0),
            ("1100+37 gemm512", &toks[..], &[1100, 37], 512),
        ];
        let modes = [
            PrefillAttn::Old,
            PrefillAttn::FlashScalar,
            PrefillAttn::Flash,
        ];
        for (name, toks, splits, g_rows) in cases {
            let mut runs = Vec::new();
            for mode in modes {
                set_thread_prefill_attn(Some(mode));
                assert_eq!(flash_prefill_for_model(&model), mode.is_flash());
                let mut cache = KvCache::new(&model.config);
                let mut ssm = model
                    .config
                    .ssm
                    .map(|_| SsmCache::from_model(&model).unwrap());
                let mut hs = ForwardScratch::new(&model);
                let fa_before = flash_prefill_launch_count();
                let mma_before = flash_mma_launch_count();
                set_thread_prefill_chunking(g_rows, 0);
                let mut p0 = 0;
                let mut logits = Vec::new();
                for &n in splits {
                    logits = model
                        .prefill(
                            &toks[p0..p0 + n],
                            p0,
                            &mut cache,
                            ssm.as_mut(),
                            &mut hs,
                            &cb,
                        )
                        .unwrap()
                        .to_vec();
                    p0 += n;
                }
                set_thread_prefill_chunking(0, 0);
                if mode.is_flash() {
                    assert!(
                        flash_prefill_launch_count() > fa_before,
                        "{name} {mode:?}: flash path not taken"
                    );
                    assert_eq!(
                        flash_mma_launch_count() > mma_before,
                        mode == PrefillAttn::Flash,
                        "{name} {mode:?}: wrong flash kernel"
                    );
                }
                let mut next = vec![am(&logits)];
                for i in 0..4 {
                    let l = model
                        .forward(
                            *next.last().unwrap(),
                            toks.len() + i,
                            &mut cache,
                            ssm.as_mut(),
                            &mut hs,
                            true,
                            &cb,
                        )
                        .unwrap()
                        .unwrap()
                        .to_vec();
                    next.push(am(&l));
                }
                runs.push((logits, next));
            }
            set_thread_prefill_attn(None);
            for (mode, run) in modes.iter().zip(&runs).skip(1) {
                let d = runs[0]
                    .0
                    .iter()
                    .zip(&run.0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "{name}: max|{mode:?} - old| logit {d:.3e}, tokens old {:?} {mode:?} {:?}",
                    runs[0].1, run.1
                );
                assert!(d < 0.5, "{name}: {mode:?} vs old logits differ by {d}");
                assert_eq!(
                    runs[0].1, run.1,
                    "{name} {mode:?}: first token + 4 greedy decode steps"
                );
            }
        }
    }

    /// Model level: Nemotron3-Nano-4B batched prefill with `ARTIFICER_PREFILL_MATMUL=mmq`
    /// vs explicit `old` on the same prompt, single chunk (300 tokens) and multi-chunk with
    /// continuation (512-row gemm chunks over 1100 + 37 tokens). Checks against the acceptance
    /// shape: last-position logits max|Δ| <= 1.0 and the same top-1; then 4 greedy steps where a
    /// mismatch is only accepted on a near tie (old's top-2 gap < 0.5). `old` must not launch MMQ.
    #[test]
    fn batched_prefill_mmq_matches_old_nemotron() {
        use super::super::cuda::{set_thread_prefill_matmul, PrefillMatmul};
        let Some(path) = nemotron() else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        if !cb.mmq_supported() {
            eprintln!("skip: MMQ needs sm_80+");
            return;
        }
        let mut model = LlamaModel::load(path).unwrap();
        model.config.n_ctx = 2048;
        if !batched_prefill_supported(&model, &KvCache::new(&model.config)) {
            eprintln!("skip: batched prefill unsupported for Nemotron in this build");
            return;
        }
        let top2 = |l: &[f32]| {
            let (mut i1, mut v1, mut v2) = (0usize, f32::MIN, f32::MIN);
            for (i, &v) in l.iter().enumerate() {
                if v > v1 {
                    v2 = v1;
                    v1 = v;
                    i1 = i;
                } else if v > v2 {
                    v2 = v;
                }
            }
            (i1 as u32, v1 - v2)
        };
        let mut r = Lcg(11);
        let toks: Vec<u32> = (0..1137)
            .map(|_| 1000 + ((r.f() + 0.5) * 30000.0) as u32)
            .collect();
        let cases: [(&str, &[u32], &[usize], usize); 2] = [
            ("300 one chunk", &toks[..300], &[300], 0),
            ("1100+37 gemm512", &toks[..], &[1100, 37], 512),
        ];
        for (name, toks, splits, g_rows) in cases {
            let mut runs = Vec::new();
            for mode in [PrefillMatmul::Old, PrefillMatmul::Mmq] {
                set_thread_prefill_matmul(Some(mode));
                let mut cache = KvCache::new(&model.config);
                let mut ssm = model
                    .config
                    .ssm
                    .map(|_| SsmCache::from_model(&model).unwrap());
                let mut hs = ForwardScratch::new(&model);
                let mmq_before = cb.mmq_launch_count();
                set_thread_prefill_chunking(g_rows, 0);
                let mut p0 = 0;
                let mut logits = Vec::new();
                for &n in splits {
                    logits = model
                        .prefill(
                            &toks[p0..p0 + n],
                            p0,
                            &mut cache,
                            ssm.as_mut(),
                            &mut hs,
                            &cb,
                        )
                        .unwrap()
                        .to_vec();
                    p0 += n;
                }
                set_thread_prefill_chunking(0, 0);
                assert_eq!(
                    cb.mmq_launch_count() > mmq_before,
                    mode == PrefillMatmul::Mmq,
                    "{name} {mode:?}: MMQ launches"
                );
                let mut steps = vec![top2(&logits)];
                for i in 0..4 {
                    let l = model
                        .forward(
                            steps.last().unwrap().0,
                            toks.len() + i,
                            &mut cache,
                            ssm.as_mut(),
                            &mut hs,
                            true,
                            &cb,
                        )
                        .unwrap()
                        .unwrap()
                        .to_vec();
                    steps.push(top2(&l));
                }
                runs.push((logits, steps));
            }
            set_thread_prefill_matmul(None);
            let d = runs[0]
                .0
                .iter()
                .zip(&runs[1].0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{name}: max|mmq - old| logit {d:.3e}, steps old {:?} mmq {:?}",
                runs[0].1, runs[1].1
            );
            assert!(
                d <= 1.0,
                "{name}: mmq vs old logits differ by {d} (acceptance bound 1.0)"
            );
            for (i, (o, m)) in runs[0].1.iter().zip(&runs[1].1).enumerate() {
                if o.0 != m.0 {
                    assert!(
                        o.1 < 0.5,
                        "{name}: step {i} token {} vs {} with old top-2 gap {}",
                        o.0,
                        m.0,
                        o.1
                    );
                    break; // sequences diverge on a near tie; later steps are not comparable
                }
            }
        }
    }

    /// Stop during batched prefill ; tiny-vicuna with forced 64-row chunks over 300
    /// tokens (5 chunks): the check runs before the first chunk and after each non-last chunk;
    /// a Stop after chunk 1 leaves exactly 64 positions in the host and device KV, and the
    /// cache is reusable (clear + prefill gives bit-identical logits). An untriggered check is
    /// numerically free: same logits bit for bit as `prefill` without a check.
    #[test]
    fn batched_prefill_cancel_between_chunks_tiny_vicuna() {
        let Some(path) = tiny_vicuna() else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut model = LlamaModel::load(&path).unwrap();
        model.config.n_ctx = 512;
        let mut r = Lcg(9);
        let toks: Vec<u32> = (0..300)
            .map(|_| 100 + ((r.f() + 0.5) * 20000.0) as u32)
            .collect();
        let mut cache = KvCache::new(&model.config);
        let mut hs = ForwardScratch::new(&model);
        set_thread_prefill_chunking(64, 64);
        let reference = model
            .prefill(&toks, 0, &mut cache, None, &mut hs, &cb)
            .unwrap()
            .to_vec();
        assert_eq!(cache.len, 300);

        // Untriggered check: 5 polls (before chunk 1, after chunks 1-4), same logits.
        cache.clear();
        let mut polls = 0usize;
        let mut never = || {
            polls += 1;
            false
        };
        let st = model
            .prefill_cancellable(&toks, 0, &mut cache, None, &mut hs, &cb, Some(&mut never))
            .unwrap();
        assert_eq!(st, PrefillStatus::Complete);
        assert_eq!(polls, 5);
        assert_eq!(hs.logits, reference, "untriggered check changed the logits");

        // Stop after the first chunk.
        for (stop_at, want) in [(1usize, 0usize), (2, 64), (4, 192)] {
            cache.clear();
            let mut polls = 0usize;
            let mut stop = || {
                polls += 1;
                polls >= stop_at
            };
            let st = model
                .prefill_cancellable(&toks, 0, &mut cache, None, &mut hs, &cb, Some(&mut stop))
                .unwrap();
            assert_eq!(
                st,
                PrefillStatus::Cancelled { processed: want },
                "stop at poll {stop_at}"
            );
            assert_eq!(polls, stop_at, "no polls after the Stop");
            assert_eq!(cache.len, want);
            assert_eq!(
                cache.device.as_ref().expect("device state restored").kv.len,
                want
            );
        }

        // Reusable: clear + full prefill reproduces the reference bit for bit.
        cache.clear();
        let again = model
            .prefill(&toks, 0, &mut cache, None, &mut hs, &cb)
            .unwrap()
            .to_vec();
        set_thread_prefill_chunking(0, 0);
        assert_eq!(cache.len, 300);
        assert_eq!(again, reference);
    }

    /// Chunk policy (no device): Nemotron-shaped config at n_ctx 10240 gives 1024-row GEMM
    /// chunks with 163-row attention sub-blocks; small prompts use one chunk of n_tok rows.
    #[test]
    fn prefill_rows_policy_nemotron_shape() {
        let Some(path) = nemotron() else {
            return;
        };
        let mut model = LlamaModel::load(path).unwrap();
        model.config.n_ctx = 10240;
        let kv_dim = model.config.n_embd_kv() as usize;
        use super::super::cuda_attention::{set_thread_prefill_attn, PrefillAttn};
        set_thread_prefill_attn(Some(PrefillAttn::Old));
        assert_eq!(prefill_rows(&model, 10240, kv_dim, 8193), (1024, 163));
        assert_eq!(prefill_rows(&model, 10240, kv_dim, 526), (526, 163));
        assert_eq!(prefill_rows(&model, 10240, kv_dim, 100), (100, 100));
        assert_eq!(prefill_rows(&model, 1024, kv_dim, 8193), (1024, 1024));
        let rb = prefill_row_bytes(&model, kv_dim);
        assert!(rb * 1024 <= PREFILL_ACT_SCRATCH_BYTES, "row bytes {rb}");
        set_thread_prefill_chunking(128, 48);
        assert_eq!(prefill_rows(&model, 10240, kv_dim, 300), (128, 48));
        set_thread_prefill_chunking(0, 0);
        // Flash: Nemotron's attention layers are flash-supported; same GEMM
        // chunking (chunk boundaries unchanged), score scratch shrinks to 1 row.
        assert!(!flash_prefill_for_model(&model));
        for mode in [PrefillAttn::Flash, PrefillAttn::FlashScalar] {
            set_thread_prefill_attn(Some(mode));
            assert!(flash_prefill_for_model(&model), "{mode:?}");
            for (n_ctx, n_tok) in [
                (10240usize, 8193usize),
                (10240, 526),
                (10240, 100),
                (1024, 8193),
            ] {
                let old = prefill_rows_for(&model, n_ctx, kv_dim, n_tok, false);
                let fa = prefill_rows(&model, n_ctx, kv_dim, n_tok);
                assert_eq!(fa, prefill_rows_for(&model, n_ctx, kv_dim, n_tok, true));
                assert_eq!(fa, (old.0, 1), "{mode:?} n_ctx {n_ctx} n_tok {n_tok}");
            }
        }
        set_thread_prefill_attn(None);
    }

    /// `layer_window` (no device) keeps the old inline sliding-window rule; any window > 0 makes
    /// `flash_prefill_for_model` false (via `fa_supported`), so such models stay on the old path.
    #[test]
    fn layer_window_rule() {
        let mut c = cfg(8, 2, 128, 256, 2);
        assert_eq!(layer_window(&c, 0), 0);
        c.sliding_window = 64;
        assert_eq!((layer_window(&c, 0), layer_window(&c, 1)), (64, 64));
        c.sliding_window_pattern = true;
        assert_eq!(
            (
                layer_window(&c, 0),
                layer_window(&c, 1),
                layer_window(&c, 2)
            ),
            (64, 0, 64)
        );
        assert!(!super::super::cuda_attention::fa_supported(
            8,
            2,
            128,
            layer_window(&c, 0)
        ));
        assert!(super::super::cuda_attention::fa_supported(
            8,
            2,
            128,
            layer_window(&c, 1)
        ));
    }

    /// Prefill timing (ignored; run manually on a free GPU):
    /// `ARTIFICER_PREFILL_BENCH=sky|long cargo test --release --features cuda -p artificer-core
    ///  --lib prefill_bench -- --ignored --nocapture`. 1 warm-up, median of 3, batched and
    /// token-by-token on the same caches (cleared between runs), first token + logit diff.
    #[test]
    #[ignore = "manual benchmark"]
    fn prefill_bench_nemotron() {
        let Some(path) = nemotron() else {
            return;
        };
        let Some(cb) = dev() else { return };
        let mut model = LlamaModel::load(path).unwrap();
        model.config.n_ctx = 1024;
        let sky = "Explain in two sentences why the sky is blue.";
        let long = "Rayleigh scattering describes how light interacts with particles much smaller than its wavelength. \
            In the atmosphere, nitrogen and oxygen molecules scatter shorter blue wavelengths far more strongly than red ones, \
            which is why the daytime sky looks blue while sunsets, seen through a longer path of air, appear orange and red. ";
        let kind = std::env::var("ARTIFICER_PREFILL_BENCH").unwrap_or_else(|_| "sky".into());
        let user = if kind == "long" {
            format!(
                "{}Summarize the text above in two sentences.",
                long.repeat(5)
            )
        } else {
            sky.to_string()
        };
        let (text, _) = crate::render_user_prompt(&model.tokenizer, &user).unwrap();
        let toks = model.tokenizer.encode(&text, true).unwrap();
        let mut cache = KvCache::new(&model.config);
        let mut ssm = model
            .config
            .ssm
            .map(|_| SsmCache::from_model(&model).unwrap());
        let mut hs = ForwardScratch::new(&model);
        let am = |l: &[f32]| {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0
        };
        let mut run = |batched: bool| -> (f64, Vec<f32>) {
            let mut times = Vec::new();
            let mut last = Vec::new();
            for _ in 0..4 {
                cache.clear();
                if let Some(s) = ssm.as_mut() {
                    s.clear();
                }
                let t0 = std::time::Instant::now();
                let l = if batched {
                    model.prefill(&toks, 0, &mut cache, ssm.as_mut(), &mut hs, &cb)
                } else {
                    model.prefill_token_by_token(&toks, 0, &mut cache, ssm.as_mut(), &mut hs, &cb)
                }
                .unwrap();
                times.push(t0.elapsed().as_secs_f64());
                last = l.to_vec();
            }
            let mut t = times[1..].to_vec();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (t[1], last)
        };
        let (tb, lb) = run(true);
        let (tt, lt) = run(false);
        let d = lb
            .iter()
            .zip(&lt)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "PREFILL_BENCH kind={kind} tokens={} mmvq_max={} batched {:.1} ms ({:.1} tok/s) | token-by-token {:.1} ms ({:.1} tok/s) | first token {} / {} | max|dlogit| {d:.3e}",
            toks.len(),
            cb.prefill_mmvq_max_rows(),
            tb * 1e3,
            toks.len() as f64 / tb,
            tt * 1e3,
            toks.len() as f64 / tt,
            am(&lb),
            am(&lt)
        );
    }
}
