//! Device attention.
//!
//! ## What runs on device
//! - **QKᵀ scores** and **AV context** via cuBLAS Sgemm on device pointers.
//! - **scale + softcap + numerically stable softmax** (max-subtract) via the NVRTC
//!   `softmax_rows` kernel (`cuda_kernels.cu`), between the two GEMMs — scores never
//!   round-trip to the host before softmax.
//!
//! ## What stays on host (honest)
//! - Gathering K/V rows from the CPU-side [`KvCache`] into gemm operands (uploaded per head).
//!
//! Q/K/V/O **projections** are `matmul_vec` / GEMV (not this module).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::Result;
use crate::kv::{KvCache, KvElem};
use crate::ForwardScratch;

use super::cuda::CudaBackend;

/// Successful device attention decode steps (each call that ran gemm score+context).
static ATTENTION_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of [`device_multi_head_attn`] completions (tests / clean-stop).
pub fn attention_launch_count() -> u64 {
    ATTENTION_LAUNCHES.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn reset_attention_launch_count_for_test() {
    ATTENTION_LAUNCHES.store(0, Ordering::Relaxed);
}

/// Decode-step multi-head attention using device GEMM for QKᵀ and AV.
///
/// Writes context into `scratch.xb2[..n_head * head_dim]`. Layout / GQA / sliding
/// window / softcap match host [`crate::forward::layer::multi_head_attn`].
pub fn device_multi_head_attn(
    backend: &CudaBackend,
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
) -> Result<()> {
    let q_dim = n_head * head_dim;
    scratch.xb2[..q_dim].fill(0.0);
    let kv_mul = n_head / n_head_kv;
    let start = attn_start.min(pos);
    let att_len = pos + 1 - start;
    if att_len == 0 || head_dim == 0 || n_head == 0 {
        return Ok(());
    }

    let gemm_before = backend.gemm_launch_count();

    // Scratch for one head's Kᵀ (d×T), V (T×d), and optional KV row dequant.
    let mut k_t = vec![0.0f32; head_dim * att_len];
    let mut v_mat = vec![0.0f32; att_len * head_dim];
    let mut kv_row = vec![0.0f32; n_head_kv * head_dim];
    let mut scores = vec![0.0f32; att_len];
    let mut out_h = vec![0.0f32; head_dim];

    let use_direct = cache.elem == crate::kv::KvElem::F32;

    for h in 0..n_head {
        let h_kv = h / kv_mul;
        // Pack Kᵀ and V for this KV head (row-major operands for gemm_f32).
        for ti in 0..att_len {
            let t = start + ti;
            if use_direct {
                let k = cache.k_at(layer_idx, t);
                let v = cache.v_at(layer_idx, t);
                let k_h = &k[h_kv * head_dim..(h_kv + 1) * head_dim];
                let v_h = &v[h_kv * head_dim..(h_kv + 1) * head_dim];
                for d in 0..head_dim {
                    k_t[d * att_len + ti] = k_h[d];
                    v_mat[ti * head_dim + d] = v_h[d];
                }
            } else {
                cache.copy_k_into(layer_idx, t, &mut kv_row);
                let k_h = &kv_row[h_kv * head_dim..(h_kv + 1) * head_dim];
                for d in 0..head_dim {
                    k_t[d * att_len + ti] = k_h[d];
                }
                cache.copy_v_into(layer_idx, t, &mut kv_row);
                let v_h = &kv_row[h_kv * head_dim..(h_kv + 1) * head_dim];
                for d in 0..head_dim {
                    v_mat[ti * head_dim + d] = v_h[d];
                }
            }
        }

        let q_h = &scratch.q[h * head_dim..(h + 1) * head_dim];
        // Device: scores = q·Kᵀ (cuBLAS) → scale/softcap/stable softmax (NVRTC kernel)
        // → out = p·V (cuBLAS). Probabilities copied back only for scratch.att parity.
        backend.attn_head_device(
            q_h,
            &k_t,
            &v_mat,
            head_dim,
            att_len,
            att_scale,
            softcap,
            &mut scores,
            &mut out_h,
        )?;
        scratch.att[h * n_ctx..h * n_ctx + att_len].copy_from_slice(&scores);

        scratch.xb2[h * head_dim..(h + 1) * head_dim].copy_from_slice(&out_h);
    }

    let gemm_after = backend.gemm_launch_count();
    // 2 gemms per head (score + context)
    debug_assert!(gemm_after >= gemm_before + 2 * n_head as u64);
    let _ = gemm_after;
    ATTENTION_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

/// Device-resident attention over the **device** KV cache: one launch per layer for
/// `n_rows` consecutive query positions `pos0..pos0+n_rows` (decode: `n_rows = 1`).
/// `q`/`out`: `[n_rows × q_stride]`; `att`: `[n_rows × n_head × n_ctx]` scratch.
/// `window == 0` means full causal context; otherwise the host sliding-window rule
/// (`start = pos + 1 - window`). Same math as [`crate::forward::layer::multi_head_attn`].
#[allow(clippy::too_many_arguments)]
pub fn attn_rows_dev(
    backend: &CudaBackend,
    q: super::cuda::DevPtr,
    kv: &super::cuda_forward::DeviceKvCache,
    layer_idx: usize,
    out: super::cuda::DevPtr,
    att: super::cuda::DevPtr,
    n_rows: usize,
    pos0: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    q_stride: usize,
    window: usize,
    att_scale: f32,
    softcap: f32,
) -> Result<()> {
    use super::cuda::LaunchArgs;
    let (k, v) = kv.layer_raw(layer_idx)?;
    let n_ctx = kv.n_ctx;
    if pos0 + n_rows > n_ctx || att.len < n_rows * n_head * n_ctx || q.len < n_rows * q_stride {
        return Err(crate::error::CoreError::Shape(format!(
            "attn_rows_dev: pos0={pos0} rows={n_rows} n_ctx={n_ctx} att={} q={}",
            att.len, q.len
        )));
    }
    let name = match kv.elem {
        KvElem::F32 => "attn_rows_f32",
        KvElem::F16 => "attn_rows_f16",
    };
    let f = backend.kernel(name)?;
    let args = LaunchArgs::new()
        .ptr(q)
        .raw(k)
        .raw(v)
        .ptr(out)
        .ptr(att)
        .i32(head_dim as i32)
        .i32(kv.kv_dim as i32)
        .i32((n_head / n_head_kv.max(1)) as i32)
        .i32(q_stride as i32)
        .i32(n_ctx as i32)
        .i32(pos0 as i32)
        .i32(window as i32)
        .f32(att_scale)
        .f32(softcap);
    backend.launch(f, (n_head as u32, n_rows as u32, 1), (128, 1, 1), 0, args)?;
    backend.note_fused_attention();
    ATTENTION_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

// ---------------------------------------------------------------------------
// Split-K flash-decode (single query row)
// ---------------------------------------------------------------------------

/// Warps per split-K block (must match `FD_WARPS` in `cuda_kernels.cu`); each warp writes one
/// partial, so a call has `n_split * FD_WARPS` partials per query head.
pub const FD_WARPS: usize = 4;
/// Max splits over the KV positions (partial buffer capacity).
pub const FD_MAX_SPLIT: usize = 64;
/// Target keys per split (16 per warp); fewer, longer splits below this.
pub const FD_MIN_KEYS_PER_SPLIT: usize = 64;
/// Kernel limits (`FD_MAXG`, `FD_MAXVEC` in `cuda_kernels.cu`).
const FD_MAXG: usize = 8;
const FD_MAXVEC: usize = 8;

/// Decode attention kernel choice: split-K flash-decode (default) or the naive
/// `attn_rows` kernel (`ARTIFICER_DECODE_ATTN=naive`, A/B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeAttn {
    SplitK,
    Naive,
}

/// Parse `ARTIFICER_DECODE_ATTN` (`naive` | `splitk` | unset → split-K).
pub fn parse_decode_attn(v: Option<&str>) -> DecodeAttn {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s == "naive" || s == "old" || s == "rows" => DecodeAttn::Naive,
        _ => DecodeAttn::SplitK,
    }
}

/// Process-wide decode attention mode (read once from the environment).
pub fn decode_attn_mode() -> DecodeAttn {
    static V: std::sync::OnceLock<DecodeAttn> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_decode_attn(std::env::var("ARTIFICER_DECODE_ATTN").ok().as_deref()))
}

/// `(n_split, chunk)` for `len` attended positions: ≥ [`FD_MIN_KEYS_PER_SPLIT`] keys per split,
/// at most [`FD_MAX_SPLIT`] splits; `n_split * chunk >= len` and every split is non-empty.
pub fn fd_plan(len: usize) -> (usize, usize) {
    fd_plan_with(len, FD_MIN_KEYS_PER_SPLIT)
}

/// [`fd_plan`] with a tunable minimum keys per split (`keys_per_split`, clamped ≥ 1).
pub fn fd_plan_with(len: usize, keys_per_split: usize) -> (usize, usize) {
    let len = len.max(1);
    let n_split = len.div_ceil(keys_per_split.max(1)).clamp(1, FD_MAX_SPLIT);
    let chunk = len.div_ceil(n_split);
    (len.div_ceil(chunk), chunk)
}

/// Split-K short-context policy (not yet benchmarked: the defaults reproduce the measured
/// split-K behaviour; the knobs exist for tuning).
/// * `min_len` (`ARTIFICER_FD_MIN_LEN`, default 0): attended lengths below this use the naive
///   kernel (one launch, no reduce) instead of split-K.
/// * `keys_per_split` (`ARTIFICER_FD_KEYS_PER_SPLIT`, default [`FD_MIN_KEYS_PER_SPLIT`]):
///   larger values mean fewer splits / partials, so a cheaper reduce but less parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdPolicy {
    pub min_len: usize,
    pub keys_per_split: usize,
}

impl Default for FdPolicy {
    fn default() -> Self {
        FdPolicy {
            min_len: 0,
            keys_per_split: FD_MIN_KEYS_PER_SPLIT,
        }
    }
}

impl FdPolicy {
    /// Whether a decode step over `len` attended positions takes the split-K path.
    pub fn use_splitk(&self, len: usize) -> bool {
        len >= self.min_len
    }
}

/// Parse the two policy env values; unset/invalid → defaults, `keys_per_split` clamped to
/// `[4, 8192]`.
pub fn parse_fd_policy(min_len: Option<&str>, keys_per_split: Option<&str>) -> FdPolicy {
    let d = FdPolicy::default();
    let num = |v: Option<&str>| v.and_then(|s| s.trim().parse::<usize>().ok());
    FdPolicy {
        min_len: num(min_len).unwrap_or(d.min_len),
        keys_per_split: num(keys_per_split).map_or(d.keys_per_split, |k| k.clamp(4, 8192)),
    }
}

/// Process-wide split-K policy (read once from the environment).
pub fn fd_policy() -> FdPolicy {
    static V: std::sync::OnceLock<FdPolicy> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        parse_fd_policy(
            std::env::var("ARTIFICER_FD_MIN_LEN").ok().as_deref(),
            std::env::var("ARTIFICER_FD_KEYS_PER_SPLIT").ok().as_deref(),
        )
    })
}

/// Shapes the split-K kernel handles (others use the naive kernel).
pub fn fd_supported(n_head: usize, n_head_kv: usize, head_dim: usize) -> bool {
    n_head_kv > 0
        && n_head.is_multiple_of(n_head_kv)
        && n_head / n_head_kv <= FD_MAXG
        && head_dim.is_multiple_of(32)
        && head_dim / 32 <= FD_MAXVEC
        && head_dim <= 1024
}

/// f32 elements of the split-K partial buffer for `n_head` query heads of `head_dim`.
pub fn fd_part_len(n_head: usize, head_dim: usize) -> usize {
    n_head * FD_MAX_SPLIT * FD_WARPS * (head_dim + 2)
}

/// Split-K kernel generation (`ARTIFICER_FD_KERNEL`).
/// Same partition, key order, math and partial layout in both; `new` is expected bit-identical
/// (same split ranges, per-key op order and reduction tree as `old`) and only changes the
/// launch shape / register use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdKernel {
    /// Original kernels: `fd_partial_{f32,f16}` + `fd_reduce` (grid `n_head`, `head_dim` threads).
    Old,
    /// `fd_partial_f16_g5d128` (kv_mul 5 / head_dim 128 f16 KV, [`fd2_partial_ok`]; other
    /// shapes keep the old partial) + `fd_reduce_smem` (grid `(n_head, ceil(head_dim / 32))`, 32
    /// threads). Default.
    New,
}

/// Kernel generation when `ARTIFICER_FD_KERNEL` is unset, empty or unknown. Flipping the default
/// is this one line; an explicit `old` always forces the original kernels.
pub const FD_KERNEL_DEFAULT: FdKernel = FdKernel::New;

/// Parse `ARTIFICER_FD_KERNEL` (trimmed, case-insensitive): `old` -> Old, `new` -> New; unset or
/// empty -> [`FD_KERNEL_DEFAULT`]; anything else -> the default with `unknown = true` (the caller
/// warns once). Returns `(kernel, unknown)`.
pub fn parse_fd_kernel(v: Option<&str>) -> (FdKernel, bool) {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => (FD_KERNEL_DEFAULT, false),
        Some("old") => (FdKernel::Old, false),
        Some("new") => (FdKernel::New, false),
        Some(_) => (FD_KERNEL_DEFAULT, true),
    }
}

#[cfg(test)]
thread_local! {
    /// Test: override [`fd_kernel`] on this thread (`None` = the process switch).
    static TEST_FD_KERNEL: std::cell::Cell<Option<FdKernel>> = const { std::cell::Cell::new(None) };
}

/// Force the split-K kernel generation on this thread (tests); `None` restores the env value.
#[cfg(test)]
pub(crate) fn set_thread_fd_kernel(k: Option<FdKernel>) {
    TEST_FD_KERNEL.with(|c| c.set(k));
}

/// Process-wide [`FdKernel`] from `ARTIFICER_FD_KERNEL`, read once (one warning if unknown).
pub fn fd_kernel() -> FdKernel {
    #[cfg(test)]
    if let Some(k) = TEST_FD_KERNEL.with(|c| c.get()) {
        return k;
    }
    static V: std::sync::OnceLock<FdKernel> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let raw = std::env::var("ARTIFICER_FD_KERNEL").ok();
        let (k, unknown) = parse_fd_kernel(raw.as_deref());
        if unknown {
            tracing::warn!(
                "ARTIFICER_FD_KERNEL={:?} is not one of old|new; using {:?}",
                raw.unwrap_or_default(),
                FD_KERNEL_DEFAULT
            );
        }
        k
    })
}

/// Query heads per KV head of `fd_partial_f16_g5d128` (`FD2_KVM` in `cuda_kernels.cu`).
pub const FD2_KV_MUL: usize = 5;
/// Head dim of `fd_partial_f16_g5d128` (`FD2_HD`); 4 dims (one 8-byte load) per lane.
pub const FD2_HEAD_DIM: usize = 128;
/// Threads per `fd_reduce_smem` block (`FDR_THREADS`); grid `(n_head, ceil(head_dim / 32))`.
pub const FDR_THREADS: usize = 32;
/// Max partials per head `fd_reduce_smem` holds in smem (`FDR_MAXPART`).
const FDR_MAXPART: usize = FD_MAX_SPLIT * FD_WARPS;
const _: () = assert!(FDR_MAXPART == 256);

/// Whether `fd_partial_f16_g5d128` can run: f16 KV, kv_mul 5, head_dim 128, and 8-byte K/V
/// row loads (`kv_dim % 4 == 0`, both base pointers 8-byte aligned). Anything else keeps the
/// original partial kernel (the reduce is shape-generic).
pub fn fd2_partial_ok(
    elem: KvElem,
    kv_mul: usize,
    head_dim: usize,
    kv_dim: usize,
    k: usize,
    v: usize,
) -> bool {
    elem == KvElem::F16
        && kv_mul == FD2_KV_MUL
        && head_dim == FD2_HEAD_DIM
        && kv_dim.is_multiple_of(4)
        && kv_dim >= FD2_HEAD_DIM
        && k.is_multiple_of(8)
        && v.is_multiple_of(8)
}

/// Split-K kernels for one call: partial name, reduce name and reduce launch
/// `(grid_y, block)`; grid x is `n_head` for the reduce, the partial launch is the same for all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdKernels {
    pub partial: &'static str,
    pub reduce: &'static str,
    pub reduce_grid_y: usize,
    pub reduce_block: usize,
}

/// Pick the split-K kernels for `sel` (see [`FdKernel`]); `partial_ok` = [`fd2_partial_ok`].
/// `Old` is exactly the original launch.
pub fn fd_kernels_pick(
    sel: FdKernel,
    elem: KvElem,
    head_dim: usize,
    partial_ok: bool,
) -> FdKernels {
    let old_partial = match elem {
        KvElem::F32 => "fd_partial_f32",
        KvElem::F16 => "fd_partial_f16",
    };
    match sel {
        FdKernel::Old => FdKernels {
            partial: old_partial,
            reduce: "fd_reduce",
            reduce_grid_y: 1,
            reduce_block: head_dim.max(32),
        },
        FdKernel::New => FdKernels {
            partial: if partial_ok {
                "fd_partial_f16_g5d128"
            } else {
                old_partial
            },
            reduce: "fd_reduce_smem",
            reduce_grid_y: head_dim.div_ceil(FDR_THREADS).max(1),
            reduce_block: FDR_THREADS,
        },
    }
}

/// Host model of the split-K math (CPU tests): online-softmax partials over `splits`
/// consecutive key ranges, merged with the max-rescale rule; returns `Σ softmax(s)·v` for one
/// head. `scores[t]` are post-scale/softcap logits, `v[t]` the value rows.
pub fn fd_reference_merge(
    scores: &[f32],
    v: &[Vec<f32>],
    n_split: usize,
    chunk: usize,
) -> Vec<f32> {
    let d = v.first().map_or(0, |r| r.len());
    let mut parts: Vec<(f32, f32, Vec<f32>)> = Vec::new();
    for s in 0..n_split {
        for w in 0..FD_WARPS {
            let (mut m, mut l, mut o) = (f32::MIN, 0.0f32, vec![0.0f32; d]);
            let t1 = ((s + 1) * chunk).min(scores.len());
            let mut t = s * chunk + w;
            while t < t1 {
                let mn = m.max(scores[t]);
                let (corr, p) = ((m - mn).exp(), (scores[t] - mn).exp());
                l = l * corr + p;
                for (oj, vj) in o.iter_mut().zip(&v[t]) {
                    *oj = *oj * corr + p * vj;
                }
                m = mn;
                t += FD_WARPS;
            }
            parts.push((m, l, o));
        }
    }
    let mm = parts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let (mut big_l, mut acc) = (0.0f32, vec![0.0f32; d]);
    for (m, l, o) in &parts {
        if *l == 0.0 {
            continue;
        }
        let wgt = (m - mm).exp();
        big_l += l * wgt;
        for (a, oj) in acc.iter_mut().zip(o) {
            *a += oj * wgt;
        }
    }
    acc.iter()
        .map(|a| if big_l > 0.0 { a / big_l } else { 0.0 })
        .collect()
}

/// Successful split-K decode attention calls (tests / A-B evidence).
static FD_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of split-K decode attention calls.
pub fn splitk_launch_count() -> u64 {
    FD_LAUNCHES.load(Ordering::Relaxed)
}

/// Split-K calls that ran `fd_partial_f16_g5d128` (tests / A-B evidence).
static FD2_PARTIAL_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of split-K calls on the specialised new partial kernel.
pub fn fd2_partial_launch_count() -> u64 {
    FD2_PARTIAL_LAUNCHES.load(Ordering::Relaxed)
}

/// Decode attention for ONE query row at `pos` over the device KV cache of `layer_idx`.
/// Split-K flash-decode (partial + reduce, kernels per [`fd_kernel`] / [`fd_kernels_pick`]) when
/// [`decode_attn_mode`] is split-K and the shape is [`fd_supported`]; otherwise the naive [`attn_rows_dev`]
/// (`att` = its `[n_head × n_ctx]` score scratch). `part` needs [`fd_part_len`] f32.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_dev(
    backend: &CudaBackend,
    mode: DecodeAttn,
    q: super::cuda::DevPtr,
    kv: &super::cuda_forward::DeviceKvCache,
    layer_idx: usize,
    out: super::cuda::DevPtr,
    att: super::cuda::DevPtr,
    part: super::cuda::DevPtr,
    pos: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    q_stride: usize,
    window: usize,
    att_scale: f32,
    softcap: f32,
) -> Result<()> {
    use super::cuda::LaunchArgs;
    let policy = fd_policy();
    let start = if window > 0 {
        (pos + 1).saturating_sub(window)
    } else {
        0
    };
    let len = (pos + 1).saturating_sub(start);
    if mode == DecodeAttn::Naive
        || !policy.use_splitk(len)
        || !fd_supported(n_head, n_head_kv, head_dim)
        || q_stride != n_head * head_dim
        || part.len < fd_part_len(n_head, head_dim)
    {
        return attn_rows_dev(
            backend, q, kv, layer_idx, out, att, 1, pos, n_head, n_head_kv, head_dim, q_stride,
            window, att_scale, softcap,
        );
    }
    let (k, v) = kv.layer_raw(layer_idx)?;
    if pos >= kv.n_ctx || q.len < q_stride || out.len < q_stride {
        return Err(crate::error::CoreError::Shape(format!(
            "attn_decode_dev: pos={pos} n_ctx={} q={} out={}",
            kv.n_ctx, q.len, out.len
        )));
    }
    let (n_split, chunk) = fd_plan_with(len, policy.keys_per_split);
    let kv_mul = n_head / n_head_kv;
    let ok2 = fd2_partial_ok(kv.elem, kv_mul, head_dim, kv.kv_dim, k as usize, v as usize);
    let ks = fd_kernels_pick(fd_kernel(), kv.elem, head_dim, ok2);
    let args = LaunchArgs::new()
        .ptr(q)
        .raw(k)
        .raw(v)
        .ptr(part)
        .i32(head_dim as i32)
        .i32(kv.kv_dim as i32)
        .i32(kv_mul as i32)
        .i32(start as i32)
        .i32(len as i32)
        .i32(chunk as i32)
        .f32(att_scale)
        .f32(softcap);
    backend.launch(
        backend.kernel(ks.partial)?,
        (n_head_kv as u32, n_split as u32, 1),
        ((32 * FD_WARPS) as u32, 1, 1),
        0,
        args,
    )?;
    let n_part = n_split * FD_WARPS;
    debug_assert!(n_part <= FDR_MAXPART);
    let args = LaunchArgs::new()
        .ptr(part)
        .ptr(out)
        .i32(head_dim as i32)
        .i32(n_part as i32);
    backend.launch(
        backend.kernel(ks.reduce)?,
        (n_head as u32, ks.reduce_grid_y as u32, 1),
        (ks.reduce_block as u32, 1, 1),
        0,
        args,
    )?;
    backend.note_fused_attention();
    ATTENTION_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    FD_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if ks.partial == "fd_partial_f16_g5d128" {
        FD2_PARTIAL_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Flash-attention prefill (scalar-tiled, fp32 math; tensor-core variant below)
// ---------------------------------------------------------------------------

/// Query rows per flash block (`FA_BR` in `cuda_kernels.cu`).
pub const FA_BR: usize = 64;
/// Keys per K/V tile (`FA_BC`).
pub const FA_BC: usize = 64;
/// The only head_dim the flash kernel is compiled for (`FA_D`).
pub const FA_D: usize = 128;
/// Warps per flash block (`FA_WARPS`); each owns `FA_BR / FA_WARPS` query rows.
pub const FA_WARPS: usize = 4;

/// Prefill attention kernel choice (`ARTIFICER_PREFILL_ATTN`): the per-row `attn_rows`
/// path in score-scratch sub-blocks (`old`, the fallback), the tensor-core flash kernel
/// (`flash`, the default, tensor cores: `fa_mma_prefill_*`) or the scalar flash kernel kept
/// for A/B (`flash_scalar`, scalar-tiled: `fa_prefill_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillAttn {
    Old,
    Flash,
    FlashScalar,
}

impl PrefillAttn {
    /// Either flash kernel.
    pub fn is_flash(self) -> bool {
        self != PrefillAttn::Old
    }
}

/// Parse `ARTIFICER_PREFILL_ATTN`: `old` | `naive` | `rows` → old; `flash_scalar` |
/// `fa_scalar` | `scalar` → scalar flash; `flash` | `fa` | `flash_mma` | `mma`, unset or
/// anything else → tensor-core flash (the default).
pub fn parse_prefill_attn(v: Option<&str>) -> PrefillAttn {
    match v
        .map(|s| s.trim().to_ascii_lowercase().replace('-', "_"))
        .as_deref()
    {
        Some("old" | "naive" | "rows") => PrefillAttn::Old,
        Some("flash_scalar" | "fa_scalar" | "scalar") => PrefillAttn::FlashScalar,
        _ => PrefillAttn::Flash,
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only Q-range check override for the calling thread (None = environment / build).
    static FA_QCHECK_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test-only: force the tensor-core kernel's |Q| check on/off for the calling thread.
#[cfg(test)]
pub(crate) fn set_thread_fa_qcheck(on: Option<bool>) {
    FA_QCHECK_OVERRIDE.with(|c| c.set(on));
}

/// Debug check of the tensor-core kernel's f16 Q operand (`ARTIFICER_FA_QCHECK=1`, or any
/// debug build): the kernel flags |Q| ≥ f16 max and the launcher syncs and returns an error.
/// Costs one tiny alloc + a sync per launch, so it is off in release by default.
pub fn fa_qcheck_enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = FA_QCHECK_OVERRIDE.with(|c| c.get()) {
        return on;
    }
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        cfg!(debug_assertions)
            || std::env::var("ARTIFICER_FA_QCHECK")
                .map(|v| v.trim() == "1")
                .unwrap_or(false)
    })
}

thread_local! {
    /// Test-only prefill attention override for the calling thread (None = environment).
    static PREFILL_ATTN_OVERRIDE: std::cell::Cell<Option<PrefillAttn>> = const { std::cell::Cell::new(None) };
}

/// Force the prefill attention mode on this thread (tests); `None` restores the env value.
#[cfg(test)]
pub(crate) fn set_thread_prefill_attn(mode: Option<PrefillAttn>) {
    PREFILL_ATTN_OVERRIDE.with(|c| c.set(mode));
}

/// Prefill attention mode: the thread override (tests) or the process-wide env value (read once).
pub fn prefill_attn_mode() -> PrefillAttn {
    if let Some(m) = PREFILL_ATTN_OVERRIDE.with(|c| c.get()) {
        return m;
    }
    static V: std::sync::OnceLock<PrefillAttn> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_prefill_attn(std::env::var("ARTIFICER_PREFILL_ATTN").ok().as_deref()))
}

/// Shapes the flash prefill kernel handles; everything else stays on the old path.
pub fn fa_supported(n_head: usize, n_head_kv: usize, head_dim: usize, window: usize) -> bool {
    n_head > 0
        && n_head_kv > 0
        && n_head.is_multiple_of(n_head_kv)
        && head_dim == FA_D
        && window == 0
}

/// Tile plan of flash block `qb` for `n_rows` query rows at positions `pos0..`: returns
/// `(n_tiles, first_masked)`. The block visits K/V tiles `0..n_tiles` (keys `t*FA_BC..`), and
/// tiles `>= first_masked` cross the diagonal and are masked (`first_masked == n_tiles` → none).
/// Mirrors `fa_prefill_impl` in `cuda_kernels.cu`.
pub fn fa_block_tiles(pos0: usize, n_rows: usize, qb: usize) -> (usize, usize) {
    let q0 = qb * FA_BR;
    debug_assert!(q0 < n_rows);
    let rows_here = (n_rows - q0).min(FA_BR);
    let qp_min = pos0 + q0;
    let n_tiles = (qp_min + rows_here - 1) / FA_BC + 1;
    // Kernel rule: mask tile t iff t*FA_BC + FA_BC - 1 > qp_min.
    let first_masked = ((qp_min + 1) / FA_BC).min(n_tiles);
    (n_tiles, first_masked)
}

/// Round an f32 to the nearest f16-representable value, ties to even (host model of PTX
/// `cvt.rn.f16.f32`): |x| ≥ 65520 → ±inf, subnormals on the 2^-24 grid.
pub fn f16_rn(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let a = x.abs();
    if a >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    if a < 6.103_515_6e-5 {
        // f16 subnormal range: multiples of 2^-24 (exact in f32).
        let step = 5.960_464_5e-8_f32;
        let q = a / step;
        let f = q.floor();
        let d = q - f;
        let n = if d > 0.5 || (d == 0.5 && (f as u32) % 2 == 1) {
            f + 1.0
        } else {
            f
        };
        return (n * step).copysign(x);
    }
    let bits = x.to_bits();
    let lsb = (bits >> 13) & 1;
    f32::from_bits((bits + 0x0FFF + lsb) & !0x1FFF)
}

/// Host model of the flash prefill math for one head (CPU tests): the kernel's tile plan and
/// fp32 online softmax. `q[r]` is query row r (position `pos0 + r`); `k`/`v` hold key rows
/// `0..pos0 + q.len()`. Returns the attention output rows.
pub fn fa_reference(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    pos0: usize,
    scale: f32,
    softcap: f32,
) -> Vec<Vec<f32>> {
    fa_reference_impl(q, k, v, pos0, scale, softcap, false)
}

/// [`fa_reference`] with the tensor-core kernel's operand rounding: Q, K, V rounded to f16
/// ([`f16_rn`]), P rounded to f16 only as the PV operand (l sums the fp32 P); S, m, l, O fp32.
pub fn fa_reference_mma(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    pos0: usize,
    scale: f32,
    softcap: f32,
) -> Vec<Vec<f32>> {
    fa_reference_impl(q, k, v, pos0, scale, softcap, true)
}

fn fa_reference_impl(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    pos0: usize,
    scale: f32,
    softcap: f32,
    f16_ops: bool,
) -> Vec<Vec<f32>> {
    let rnd = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        rows.iter()
            .map(|r| {
                r.iter()
                    .map(|&x| if f16_ops { f16_rn(x) } else { x })
                    .collect()
            })
            .collect()
    };
    let (q, k, v) = (&rnd(q), &rnd(k), &rnd(v));
    let n_rows = q.len();
    let d = v.first().map_or(0, |r| r.len());
    let kv_end = pos0 + n_rows;
    let mut out = vec![vec![0.0f32; d]; n_rows];
    for qb in 0..n_rows.div_ceil(FA_BR) {
        let (n_tiles, first_masked) = fa_block_tiles(pos0, n_rows, qb);
        let q0 = qb * FA_BR;
        for (r, out_r) in out.iter_mut().enumerate().skip(q0).take(FA_BR) {
            let qp = pos0 + r;
            let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
            let mut o = vec![0.0f32; d];
            for t in 0..n_tiles {
                let mut s = [f32::NEG_INFINITY; FA_BC];
                for (j, sj) in s.iter_mut().enumerate() {
                    let kp = t * FA_BC + j;
                    if kp >= kv_end || (t >= first_masked && kp > qp) {
                        continue;
                    }
                    let dot: f32 = q[r].iter().zip(&k[kp]).map(|(a, b)| a * b).sum();
                    let mut x = dot * scale;
                    if softcap > 0.0 {
                        x = softcap * (x / softcap).tanh();
                    }
                    *sj = x;
                }
                let mn = s.iter().copied().fold(m, f32::max);
                if mn == f32::NEG_INFINITY {
                    continue;
                }
                let corr = (m - mn).exp();
                l *= corr;
                o.iter_mut().for_each(|x| *x *= corr);
                for (j, &sj) in s.iter().enumerate() {
                    let p = (sj - mn).exp();
                    if p > 0.0 {
                        l += p;
                        let pv = if f16_ops { f16_rn(p) } else { p };
                        for (x, vv) in o.iter_mut().zip(&v[t * FA_BC + j]) {
                            *x += pv * vv;
                        }
                    }
                }
                m = mn;
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            for (dst, x) in out_r.iter_mut().zip(&o) {
                *dst = x * inv;
            }
        }
    }
    out
}

/// Flash prefill attention calls (tests / A-B evidence).
static FA_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Tensor-core flash prefill calls.
static FA_MMA_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of flash prefill attention calls (scalar + tensor-core).
pub fn flash_prefill_launch_count() -> u64 {
    FA_LAUNCHES.load(Ordering::Relaxed)
}

/// Process-wide count of tensor-core flash prefill attention calls.
pub fn flash_mma_launch_count() -> u64 {
    FA_MMA_LAUNCHES.load(Ordering::Relaxed)
}

/// Flash prefill attention for `n_rows` query rows at positions `pos0..pos0+n_rows` over the
/// device KV cache of `layer_idx`, full causal context. One launch for all rows, no score
/// scratch. `variant`: [`PrefillAttn::Flash`] = tensor-core `fa_mma_prefill_*`,
/// [`PrefillAttn::FlashScalar`] = scalar `fa_prefill_*` (`Old` is an error).
/// `q`/`out`: `[n_rows × q_stride]` (`q_stride` even). Requires [`fa_supported`] (window 0) and
/// every key `< pos0 + n_rows` already in the cache.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill_flash_dev(
    backend: &CudaBackend,
    variant: PrefillAttn,
    q: super::cuda::DevPtr,
    kv: &super::cuda_forward::DeviceKvCache,
    layer_idx: usize,
    out: super::cuda::DevPtr,
    n_rows: usize,
    pos0: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    q_stride: usize,
    att_scale: f32,
    softcap: f32,
) -> Result<()> {
    use super::cuda::LaunchArgs;
    if n_rows == 0 {
        return Ok(());
    }
    let (k, v) = kv.layer_raw(layer_idx)?;
    let n_ctx = kv.n_ctx;
    if !variant.is_flash()
        || !fa_supported(n_head, n_head_kv, head_dim, 0)
        || kv.kv_dim != n_head_kv * head_dim
        || q_stride < n_head * head_dim
        || !q_stride.is_multiple_of(2)
        || pos0 + n_rows > n_ctx
        || q.len < n_rows * q_stride
        || out.len < n_rows * q_stride
    {
        return Err(crate::error::CoreError::Shape(format!(
            "attn_prefill_flash_dev({variant:?}): heads {n_head}/{n_head_kv} d{head_dim} kv_dim {} q_stride {q_stride} \
             pos0={pos0} rows={n_rows} n_ctx={n_ctx} q={} out={}",
            kv.kv_dim, q.len, out.len
        )));
    }
    let mma = variant == PrefillAttn::Flash;
    let name = match (mma, kv.elem) {
        (true, KvElem::F32) => "fa_mma_prefill_f32kv",
        (true, KvElem::F16) => "fa_mma_prefill_f16",
        (false, KvElem::F32) => "fa_prefill_f32kv",
        (false, KvElem::F16) => "fa_prefill_f16",
    };
    let f = backend.kernel(name)?;
    let mut args = LaunchArgs::new()
        .ptr(q)
        .raw(k)
        .raw(v)
        .ptr(out)
        .i32(n_rows as i32)
        .i32(kv.kv_dim as i32)
        .i32((n_head / n_head_kv) as i32)
        .i32(q_stride as i32)
        .i32(pos0 as i32)
        .f32(att_scale)
        .f32(softcap);
    // Debug |Q| f16-range flag (tensor-core kernel only; null = off).
    let qflag = if mma && fa_qcheck_enabled() {
        Some(backend.alloc_f32(1)?)
    } else {
        None
    };
    if mma {
        args = args.raw(qflag.as_ref().map_or(std::ptr::null_mut(), |t| t.raw()));
    }
    let n_qb = n_rows.div_ceil(FA_BR);
    backend.launch(
        f,
        (n_head as u32, n_qb as u32, 1),
        ((FA_WARPS * 32) as u32, 1, 1),
        0,
        args,
    )?;
    backend.note_fused_attention();
    ATTENTION_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    FA_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if mma {
        FA_MMA_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    if let Some(flag) = qflag {
        let mut bits = [0.0f32; 1];
        backend.download_f32(flag.ptr(), &mut bits)?;
        if bits[0].to_bits() != 0 {
            return Err(crate::error::CoreError::Other(format!(
                "flash prefill (layer {layer_idx}, pos0 {pos0}, rows {n_rows}): |Q| exceeds the f16 range"
            )));
        }
    }
    Ok(())
}

/// CPU-only tests of the split-K planning / merge math (no device needed).
#[cfg(test)]
mod fd_cpu_tests {
    use super::*;

    /// `ARTIFICER_FD_KERNEL` parsing (no device): old / new (trimmed, any case); unset / empty ->
    /// the default silently; unknown -> the default + warn flag. Kernel pick per shape: `old` is
    /// exactly the original launch; `new` takes the specialised partial only for f16, kv_mul 5,
    /// head_dim 128, kv_dim % 4 == 0 and 8-byte aligned K/V, and always the new reduce.
    #[test]
    fn fd_kernel_parse_and_pick() {
        use FdKernel::*;
        assert_eq!(
            FD_KERNEL_DEFAULT, New,
            "default is the new kernels; explicit old forces the original kernels"
        );
        let d = FD_KERNEL_DEFAULT;
        for (v, want) in [
            (None, (d, false)),
            (Some(""), (d, false)),
            (Some("  "), (d, false)),
            (Some("old"), (Old, false)),
            (Some(" OLD\n"), (Old, false)),
            (Some("new"), (New, false)),
            (Some("NEW"), (New, false)),
            (Some(" New "), (New, false)),
            (Some("e1"), (d, true)),
            (Some("naive"), (d, true)),
            (Some("1"), (d, true)),
            (Some("old,new"), (d, true)),
        ] {
            assert_eq!(parse_fd_kernel(v), want, "{v:?}");
        }
        // fd2_partial_ok: Nemotron (kv_dim 1024) yes; any other shape / f32 / misaligned no.
        let (a, b) = (0x7f00_0000_0000usize, 0x7f00_0010_0000usize);
        assert!(fd2_partial_ok(KvElem::F16, 5, 128, 1024, a, b));
        assert!(fd2_partial_ok(KvElem::F16, 5, 128, 256, a, b));
        assert!(!fd2_partial_ok(KvElem::F32, 5, 128, 1024, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 4, 128, 1024, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 8, 128, 1024, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 1, 128, 1024, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 5, 64, 1024, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 5, 256, 1024, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 5, 128, 1026, a, b));
        assert!(!fd2_partial_ok(KvElem::F16, 5, 128, 1024, a + 4, b));
        assert!(!fd2_partial_ok(KvElem::F16, 5, 128, 1024, a, b + 2));
        let old16 = FdKernels {
            partial: "fd_partial_f16",
            reduce: "fd_reduce",
            reduce_grid_y: 1,
            reduce_block: 128,
        };
        assert_eq!(fd_kernels_pick(Old, KvElem::F16, 128, true), old16);
        assert_eq!(fd_kernels_pick(Old, KvElem::F16, 128, false), old16);
        assert_eq!(
            fd_kernels_pick(Old, KvElem::F32, 32, false),
            FdKernels {
                partial: "fd_partial_f32",
                reduce: "fd_reduce",
                reduce_grid_y: 1,
                reduce_block: 32
            }
        );
        assert_eq!(
            fd_kernels_pick(New, KvElem::F16, 128, true),
            FdKernels {
                partial: "fd_partial_f16_g5d128",
                reduce: "fd_reduce_smem",
                reduce_grid_y: 4,
                reduce_block: 32
            }
        );
        assert_eq!(
            fd_kernels_pick(New, KvElem::F16, 64, false),
            FdKernels {
                partial: "fd_partial_f16",
                reduce: "fd_reduce_smem",
                reduce_grid_y: 2,
                reduce_block: 32
            }
        );
        assert_eq!(
            fd_kernels_pick(New, KvElem::F32, 256, false),
            FdKernels {
                partial: "fd_partial_f32",
                reduce: "fd_reduce_smem",
                reduce_grid_y: 8,
                reduce_block: 32
            }
        );
        // The reduce smem holds every partial of a head.
        for len in [1usize, 65, 4097, 8193, 1 << 20] {
            assert!(fd_plan_with(len, 4).0 * FD_WARPS <= FDR_MAXPART);
        }
    }

    #[test]
    fn fd_policy_defaults_match_gated_policy() {
        let d = parse_fd_policy(None, None);
        assert_eq!(d, FdPolicy::default());
        assert_eq!(
            d,
            FdPolicy {
                min_len: 0,
                keys_per_split: FD_MIN_KEYS_PER_SPLIT
            }
        );
        for len in [1usize, 64, 527, 1550, 8193] {
            assert!(d.use_splitk(len));
            assert_eq!(fd_plan_with(len, d.keys_per_split), fd_plan(len));
        }
    }

    #[test]
    fn fd_policy_env_parsing_and_threshold() {
        let p = parse_fd_policy(Some(" 2048 "), Some("256"));
        assert_eq!(
            p,
            FdPolicy {
                min_len: 2048,
                keys_per_split: 256
            }
        );
        assert!(!p.use_splitk(2047) && p.use_splitk(2048));
        assert_eq!(parse_fd_policy(Some("x"), Some("1")).keys_per_split, 4);
        assert_eq!(parse_fd_policy(Some("x"), Some("1")).min_len, 0);
        assert_eq!(parse_fd_policy(None, Some("999999")).keys_per_split, 8192);
    }

    #[test]
    fn fd_plan_with_larger_splits_covers_len() {
        for kps in [4usize, 64, 128, 256, 1024] {
            for len in [1usize, 2, 63, 64, 65, 527, 1550, 2040, 4097, 8193, 10240] {
                let (n, c) = fd_plan_with(len, kps);
                assert!((1..=FD_MAX_SPLIT).contains(&n));
                assert!(
                    n * c >= len && (n - 1) * c < len,
                    "kps {kps} len {len}: {n}x{c}"
                );
                if len <= kps {
                    assert_eq!(n, 1);
                }
            }
        }
        // 526-token prompt + 1024 decode steps: attended lengths 527..=1550.
        assert_eq!(fd_plan(527).0, 9);
        assert_eq!(fd_plan(1550).0, 25);
        assert_eq!(fd_plan_with(1550, 256).0, 7);
    }

    #[test]
    fn fd_plan_covers_len_without_empty_splits() {
        for len in [
            1usize, 2, 31, 63, 64, 65, 127, 128, 129, 526, 2040, 4096, 8193, 10240, 100_000,
        ] {
            let (n, c) = fd_plan(len);
            assert!((1..=FD_MAX_SPLIT).contains(&n), "len {len}: n_split {n}");
            assert!(n * c >= len, "len {len}: {n}×{c} < len");
            assert!((n - 1) * c < len, "len {len}: last split empty ({n}×{c})");
            if len >= FD_MIN_KEYS_PER_SPLIT * FD_MAX_SPLIT {
                assert_eq!(n, FD_MAX_SPLIT);
            }
        }
        assert_eq!(fd_plan(8193), (64, 129));
        assert_eq!(fd_plan(10), (1, 10));
    }

    #[test]
    fn fd_supported_shapes() {
        assert!(fd_supported(40, 8, 128)); // Nemotron: GQA ×5, head_dim 128
        assert!(fd_supported(32, 32, 64)); // tiny-vicuna (MHA)
        assert!(fd_supported(8, 2, 32));
        assert!(!fd_supported(48, 4, 128)); // GQA ×12 > FD_MAXG
        assert!(!fd_supported(8, 2, 80)); // head_dim % 32 != 0
        assert!(!fd_supported(8, 2, 512)); // > 8 dims per lane
        assert!(!fd_supported(6, 4, 64)); // n_head % n_head_kv != 0
    }

    #[test]
    fn parse_decode_attn_env_values() {
        assert_eq!(parse_decode_attn(None), DecodeAttn::SplitK);
        assert_eq!(parse_decode_attn(Some("splitk")), DecodeAttn::SplitK);
        assert_eq!(parse_decode_attn(Some("naive")), DecodeAttn::Naive);
        assert_eq!(parse_decode_attn(Some(" NAIVE ")), DecodeAttn::Naive);
        assert_eq!(parse_decode_attn(Some("old")), DecodeAttn::Naive);
        assert_eq!(parse_decode_attn(Some("")), DecodeAttn::SplitK);
    }

    /// Split-K online-softmax partials + merge equal a direct softmax·V (all plans, incl.
    /// large score ranges that would overflow a non-max-subtracted exp).
    #[test]
    fn fd_merge_matches_direct_softmax() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for &len in &[1usize, 3, 64, 65, 300, 2049, 5000] {
            let d = 16;
            let scores: Vec<f32> = (0..len)
                .map(|i| rnd() * 40.0 + if i % 97 == 0 { 60.0 } else { 0.0 })
                .collect();
            let v: Vec<Vec<f32>> = (0..len).map(|_| (0..d).map(|_| rnd()).collect()).collect();
            let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f64> = scores.iter().map(|&s| ((s - mx) as f64).exp()).collect();
            let z: f64 = e.iter().sum();
            let want: Vec<f64> = (0..d)
                .map(|j| (0..len).map(|t| e[t] * v[t][j] as f64).sum::<f64>() / z)
                .collect();
            let (n, c) = fd_plan(len);
            for &(ns, ch) in &[
                (n, c),
                (1, len),
                (len.min(FD_MAX_SPLIT), len.div_ceil(len.min(FD_MAX_SPLIT))),
            ] {
                let got = fd_reference_merge(&scores, &v, ns, ch);
                for j in 0..d {
                    assert!(
                        (got[j] as f64 - want[j]).abs() < 1e-5,
                        "len {len} plan {ns}×{ch} dim {j}: {} vs {}",
                        got[j],
                        want[j]
                    );
                }
            }
        }
    }
}

/// CPU-only tests of the flash prefill knob, tile plan and math model (no device needed).
#[cfg(test)]
mod fa_cpu_tests {
    use super::*;

    #[test]
    fn parse_prefill_attn_env_values() {
        assert_eq!(parse_prefill_attn(None), PrefillAttn::Flash, "default");
        for v in ["old", "naive", "rows", "OLD", " rows ", "Naive"] {
            assert_eq!(parse_prefill_attn(Some(v)), PrefillAttn::Old, "{v:?}");
        }
        for v in [
            "flash",
            "FLASH",
            " Flash ",
            "fa",
            "flash_mma",
            "mma",
            "",
            "bogus",
            "splitk",
        ] {
            assert_eq!(parse_prefill_attn(Some(v)), PrefillAttn::Flash, "{v:?}");
        }
        for v in [
            "flash_scalar",
            "FLASH_SCALAR",
            "flash-scalar",
            "fa_scalar",
            "scalar",
        ] {
            assert_eq!(
                parse_prefill_attn(Some(v)),
                PrefillAttn::FlashScalar,
                "{v:?}"
            );
        }
        assert!(!PrefillAttn::Old.is_flash());
        assert!(PrefillAttn::Flash.is_flash() && PrefillAttn::FlashScalar.is_flash());
    }

    #[test]
    fn f16_rn_matches_ieee_half() {
        let e = |x: f32| f16_rn(x);
        for x in [
            0.0f32,
            1.0,
            -2.5,
            65504.0,
            -65504.0,
            0.333_251_95,
            6.103_515_6e-5,
        ] {
            assert_eq!(e(x), x, "{x} is f16-exact");
        }
        let u = 2f32.powi(-10); // f16 ulp at 1.0
        assert_eq!(e(1.0 + u / 2.0), 1.0, "tie to even (down)");
        assert_eq!(e(1.0 + 1.5 * u), 1.0 + 2.0 * u, "tie to even (up)");
        assert_eq!(e(1.0 + 0.6 * u), 1.0 + u);
        assert_eq!(e(-(1.0 + 0.4 * u)), -1.0);
        assert_eq!(e(65519.0), 65504.0);
        assert_eq!(e(65520.0), f32::INFINITY);
        assert_eq!(e(-1e6), f32::NEG_INFINITY);
        let s = 2f32.powi(-24); // smallest subnormal
        assert_eq!(e(s / 2.0), 0.0, "tie to even zero");
        assert_eq!(e(1.5 * s), 2.0 * s);
        assert_eq!(e(0.7 * s), s);
        assert_eq!(e(-3.2 * s), -3.0 * s);
        assert!(e(f32::NAN).is_nan());
        // Relative error bound on the normal range.
        let mut x = 1e-4f32;
        while x < 6e4 {
            assert!((e(x) - x).abs() <= x * 2f32.powi(-11), "{x}");
            x *= 1.37;
        }
    }

    #[test]
    fn prefill_attn_thread_override() {
        set_thread_prefill_attn(Some(PrefillAttn::Flash));
        assert_eq!(prefill_attn_mode(), PrefillAttn::Flash);
        set_thread_prefill_attn(Some(PrefillAttn::Old));
        assert_eq!(prefill_attn_mode(), PrefillAttn::Old);
        set_thread_prefill_attn(None);
        let env = parse_prefill_attn(std::env::var("ARTIFICER_PREFILL_ATTN").ok().as_deref());
        assert_eq!(prefill_attn_mode(), env);
    }

    #[test]
    fn fa_supported_shapes() {
        assert!(fa_supported(40, 8, 128, 0)); // Nemotron3-Nano-4B
        assert!(fa_supported(32, 32, 128, 0));
        assert!(fa_supported(8, 1, 128, 0));
        assert!(!fa_supported(32, 32, 64, 0)); // tiny-vicuna head_dim
        assert!(!fa_supported(40, 8, 256, 0));
        assert!(!fa_supported(40, 8, 128, 4096)); // sliding window
        assert!(!fa_supported(40, 6, 128, 0));
        assert!(!fa_supported(40, 0, 128, 0));
        assert!(!fa_supported(0, 8, 128, 0));
    }

    /// Every valid key `<= qp` of every query row is visited exactly once, unmasked tiles hold
    /// only keys every row of the block may see, and no tile starts past the last query.
    #[test]
    fn fa_tile_plan_covers_causal_keys() {
        let rows = [1usize, 2, 15, 63, 64, 65, 127, 128, 163, 300, 1024];
        let pos0s = [0usize, 1, 7, 63, 64, 65, 1000, 1023, 1024, 8192 - 1024];
        for &n_rows in &rows {
            for &pos0 in &pos0s {
                let kv_end = pos0 + n_rows;
                let n_qb = n_rows.div_ceil(FA_BR);
                for qb in 0..n_qb {
                    let (n_tiles, first_masked) = fa_block_tiles(pos0, n_rows, qb);
                    let q0 = qb * FA_BR;
                    let rows_here = (n_rows - q0).min(FA_BR);
                    let (qp_min, qp_max) = (pos0 + q0, pos0 + q0 + rows_here - 1);
                    assert!(first_masked <= n_tiles);
                    assert!((n_tiles - 1) * FA_BC <= qp_max, "tile past last query");
                    assert!(n_tiles * FA_BC > qp_max, "last query's key not covered");
                    for t in 0..first_masked {
                        assert!(
                            t * FA_BC + FA_BC - 1 <= qp_min,
                            "unmasked tile crosses diagonal"
                        );
                        assert!(
                            t * FA_BC + FA_BC <= kv_end,
                            "unmasked tile reads past kv_end"
                        );
                    }
                    for r in q0..q0 + rows_here {
                        let qp = pos0 + r;
                        let mut seen = vec![0u32; qp + 1];
                        for t in 0..n_tiles {
                            for j in 0..FA_BC {
                                let kp = t * FA_BC + j;
                                let masked = t >= first_masked && (kp > qp || kp >= kv_end);
                                if !masked {
                                    assert!(kp <= qp, "rows {n_rows} pos0 {pos0} qb {qb} row {r}: key {kp} > {qp}");
                                    seen[kp] += 1;
                                }
                            }
                        }
                        assert!(
                            seen.iter().all(|&c| c == 1),
                            "rows {n_rows} pos0 {pos0} row {r}: coverage"
                        );
                    }
                }
            }
        }
    }

    /// The tiled online-softmax model equals a direct causal softmax (fp32 tolerance), with and
    /// without softcap, at arbitrary pos0 and tail rows.
    #[test]
    fn fa_reference_matches_direct_softmax() {
        fa_reference_vs_direct(16, false, 1e-5);
    }

    /// Design tolerance of the tensor-core kernel (max|Δ| ≤ 2e-2 abs) is reachable: the host
    /// model with f16 Q/K/V/P operands vs exact f64 softmax at d=128 and the device test's
    /// input ranges stays well inside it.
    #[test]
    fn fa_reference_mma_within_design_tolerance() {
        fa_reference_vs_direct(FA_D, true, 2e-2);
    }

    fn fa_reference_vs_direct(d: usize, mma: bool, tol: f64) {
        let mut seed = 99u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        };
        let mut worst = 0.0f64;
        for &(pos0, n_rows, softcap) in &[
            (0usize, 1usize, 0.0f32),
            (0, 70, 0.0),
            (7, 65, 30.0),
            (100, 130, 0.0),
            (63, 64, 5.0),
        ] {
            let n_keys = pos0 + n_rows;
            let k: Vec<Vec<f32>> = (0..n_keys)
                .map(|_| (0..d).map(|_| rnd()).collect())
                .collect();
            let v: Vec<Vec<f32>> = (0..n_keys)
                .map(|_| (0..d).map(|_| rnd()).collect())
                .collect();
            let q: Vec<Vec<f32>> = (0..n_rows)
                .map(|_| (0..d).map(|_| rnd() * 8.0).collect())
                .collect();
            let scale = 1.0 / (d as f32).sqrt();
            let got = if mma {
                fa_reference_mma(&q, &k, &v, pos0, scale, softcap)
            } else {
                fa_reference(&q, &k, &v, pos0, scale, softcap)
            };
            for r in 0..n_rows {
                let qp = pos0 + r;
                let s: Vec<f64> = (0..=qp)
                    .map(|t| {
                        let dot: f64 = q[r]
                            .iter()
                            .zip(&k[t])
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum();
                        let x = dot * scale as f64;
                        if softcap > 0.0 {
                            softcap as f64 * (x / softcap as f64).tanh()
                        } else {
                            x
                        }
                    })
                    .collect();
                let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let sum: f64 = e.iter().sum();
                for j in 0..d {
                    let want: f64 = e
                        .iter()
                        .enumerate()
                        .map(|(t, p)| p * v[t][j] as f64)
                        .sum::<f64>()
                        / sum;
                    let err = (got[r][j] as f64 - want).abs();
                    worst = worst.max(err);
                    assert!(
                        err < tol,
                        "d{d} mma {mma} pos0 {pos0} rows {n_rows} r {r} j {j}: {} vs {want}",
                        got[r][j]
                    );
                }
            }
        }
        eprintln!("d{d} mma {mma}: max|model - f64 softmax| {worst:.2e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::backend::Backend;
    use crate::config::{Architecture, FfnActivation, ModelConfig, NormKind};
    use crate::forward::layer::multi_head_attn;
    use crate::kv::{KvCache, KvElem};
    use crate::ForwardScratch;

    fn skip_no_device() -> Option<CudaBackend> {
        let mut b = CudaBackend::try_new().ok()?;
        b.init_device().ok()?;
        Some(b)
    }

    fn tiny_cfg(n_head: u32, n_head_kv: u32, head_dim: u32, n_ctx: u32) -> ModelConfig {
        let n_embd = n_head * head_dim;
        ModelConfig {
            architecture: Architecture::Llama,
            arch_str: "llama".into(),
            name: Some("attn-test".into()),
            n_vocab: 32,
            n_ctx,
            n_embd,
            n_layer: 1,
            n_ff: n_embd * 4,
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

    fn make_scratch(cfg: &ModelConfig) -> ForwardScratch {
        let d = cfg.n_embd as usize;
        let n_ff = cfg.n_ff.max(1) as usize;
        let n_vocab = cfg.n_vocab as usize;
        let n_ctx = cfg.n_ctx as usize;
        let n_head = cfg.n_head.max(1) as usize;
        let n_kv = cfg.n_embd_kv().max(1) as usize;
        let q_dim = n_head * cfg.head_dim as usize;
        ForwardScratch {
            x: vec![0.0; d],
            xb: vec![0.0; d.max(q_dim)],
            xb2: vec![0.0; q_dim],
            q: vec![0.0; q_dim],
            k: vec![0.0; n_kv],
            v: vec![0.0; n_kv],
            att: vec![0.0; n_head * n_ctx],
            hb: vec![0.0; n_ff],
            hb2: vec![0.0; n_ff],
            logits: vec![0.0; n_vocab],
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

    fn fill_scratch_q(scratch: &mut ForwardScratch, n_head: usize, head_dim: usize, seed: u32) {
        let mut s = seed;
        for v in scratch.q[..n_head * head_dim].iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 16) as f32 / 65535.0) - 0.5;
        }
    }

    fn write_random_kv(cache: &mut KvCache, layer: usize, pos: usize, n_kv_dim: usize, seed: u32) {
        let mut s = seed;
        let mut k = vec![0.0f32; n_kv_dim];
        let mut v = vec![0.0f32; n_kv_dim];
        for x in k.iter_mut().chain(v.iter_mut()) {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *x = ((s >> 16) as f32 / 65535.0) * 0.5;
        }
        cache.write(layer, pos, &k, &v);
    }

    #[test]
    fn device_attention_matches_cpu_gqa_and_seq_lens() {
        let Some(backend) = skip_no_device() else {
            eprintln!("skip: no usable CUDA device");
            return;
        };
        reset_attention_launch_count_for_test();
        let n_head = 4usize;
        let n_head_kv = 2usize;
        let head_dim = 8usize;
        let n_ctx = 64usize;
        let cfg = tiny_cfg(
            n_head as u32,
            n_head_kv as u32,
            head_dim as u32,
            n_ctx as u32,
        );
        let cases = [(0usize, 0usize), (3, 0), (7, 0), (15, 4)];
        let att_scale = 1.0 / (head_dim as f32).sqrt();
        let softcap = 0.0f32;

        for &(pos, attn_start) in &cases {
            let mut cache = KvCache::with_elem(&cfg, KvElem::F32);
            let n_kv_dim = n_head_kv * head_dim;
            for t in 0..=pos {
                write_random_kv(&mut cache, 0, t, n_kv_dim, 50 + t as u32);
            }
            let mut scratch_cpu = make_scratch(&cfg);
            let mut scratch_gpu = make_scratch(&cfg);
            fill_scratch_q(&mut scratch_cpu, n_head, head_dim, 9);
            scratch_gpu.q[..n_head * head_dim].copy_from_slice(&scratch_cpu.q[..n_head * head_dim]);

            let gemm_before = backend.gemm_launch_count();
            let attn_before = attention_launch_count();

            multi_head_attn(
                &mut scratch_cpu,
                &cache,
                0,
                pos,
                attn_start,
                n_head,
                n_head_kv,
                head_dim,
                n_ctx,
                att_scale,
                softcap,
            );
            device_multi_head_attn(
                &backend,
                &mut scratch_gpu,
                &cache,
                0,
                pos,
                attn_start,
                n_head,
                n_head_kv,
                head_dim,
                n_ctx,
                att_scale,
                softcap,
            )
            .expect("device attention");

            let q_dim = n_head * head_dim;
            for i in 0..q_dim {
                let a = scratch_cpu.xb2[i];
                let b = scratch_gpu.xb2[i];
                let tol = 1e-3 * (1.0 + a.abs());
                assert!(
                    (a - b).abs() < tol,
                    "pos={pos} start={attn_start} i={i}: cpu={a} gpu={b} tol={tol}"
                );
            }
            assert!(
                backend.gemm_launch_count() >= gemm_before + 2 * n_head as u64,
                "expected 2 gemm/head"
            );
            assert!(attention_launch_count() > attn_before);
        }
        eprintln!(
            "CUDA attention ok on {} — attn_launches={} gemm_launches={}",
            backend.device_name(),
            attention_launch_count(),
            backend.gemm_launch_count()
        );
    }

    #[test]
    fn device_attention_with_softcap() {
        let Some(backend) = skip_no_device() else {
            eprintln!("skip: no usable CUDA device");
            return;
        };
        let n_head = 2usize;
        let n_head_kv = 2usize;
        let head_dim = 4usize;
        let n_ctx = 16usize;
        let cfg = tiny_cfg(
            n_head as u32,
            n_head_kv as u32,
            head_dim as u32,
            n_ctx as u32,
        );
        let mut cache = KvCache::with_elem(&cfg, KvElem::F32);
        for t in 0..5 {
            write_random_kv(&mut cache, 0, t, n_head_kv * head_dim, 3 + t as u32);
        }
        let mut scratch_cpu = make_scratch(&cfg);
        let mut scratch_gpu = make_scratch(&cfg);
        fill_scratch_q(&mut scratch_cpu, n_head, head_dim, 11);
        scratch_gpu.q[..n_head * head_dim].copy_from_slice(&scratch_cpu.q[..n_head * head_dim]);
        let softcap = 30.0f32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        multi_head_attn(
            &mut scratch_cpu,
            &cache,
            0,
            4,
            0,
            n_head,
            n_head_kv,
            head_dim,
            n_ctx,
            scale,
            softcap,
        );
        device_multi_head_attn(
            &backend,
            &mut scratch_gpu,
            &cache,
            0,
            4,
            0,
            n_head,
            n_head_kv,
            head_dim,
            n_ctx,
            scale,
            softcap,
        )
        .unwrap();
        for i in 0..n_head * head_dim {
            let tol = 1e-3 * (1.0 + scratch_cpu.xb2[i].abs());
            assert!((scratch_cpu.xb2[i] - scratch_gpu.xb2[i]).abs() < tol);
        }
    }

    /// Split-K flash-decode vs the naive `attn_rows` kernel on the same device KV
    /// (f32 and f16), several context lengths up to 8193, Nemotron GQA ×5 / d128, MHA d64,
    /// softcap and a sliding window.
    #[test]
    fn splitk_decode_matches_naive_over_context_lengths() {
        use super::super::cuda_forward::DeviceKvCache;
        let Some(cb) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut seed = 777u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        };
        // (n_head, n_head_kv, head_dim, window, softcap)
        let shapes = [
            (40usize, 8usize, 128usize, 0usize, 0.0f32),
            (32, 32, 64, 0, 0.0),
            (8, 2, 32, 0, 30.0),
            (8, 2, 64, 100, 0.0),
        ];
        let n_ctx = 8200usize;
        let positions = [0usize, 1, 31, 63, 64, 65, 200, 526, 2039, 4097, 8192];
        for elem in [KvElem::F32, KvElem::F16] {
            for &(n_head, n_head_kv, head_dim, window, softcap) in &shapes {
                let kv_dim = n_head_kv * head_dim;
                let q_dim = n_head * head_dim;
                let kv = DeviceKvCache::new(&cb, &[true], n_ctx, kv_dim, elem).unwrap();
                let rows: Vec<f32> = (0..n_ctx * kv_dim).map(|_| rnd()).collect();
                let src = cb.alloc_f32(n_ctx * kv_dim).unwrap();
                cb.upload_f32(&rows, src.ptr()).unwrap();
                kv.store_rows(&cb, 0, 0, n_ctx, src.ptr(), false).unwrap();
                let rows: Vec<f32> = (0..n_ctx * kv_dim).map(|_| rnd()).collect();
                cb.upload_f32(&rows, src.ptr()).unwrap();
                kv.store_rows(&cb, 0, 0, n_ctx, src.ptr(), true).unwrap();
                let dq = cb.alloc_f32(q_dim).unwrap();
                let (d_naive, d_split) =
                    (cb.alloc_f32(q_dim).unwrap(), cb.alloc_f32(q_dim).unwrap());
                let datt = cb.alloc_f32(n_head * n_ctx).unwrap();
                let part = cb.alloc_f32(fd_part_len(n_head, head_dim)).unwrap();
                let scale = 1.0 / (head_dim as f32).sqrt();
                for &pos in &positions {
                    let q: Vec<f32> = (0..q_dim).map(|_| rnd() * 6.0).collect();
                    cb.upload_f32(&q, dq.ptr()).unwrap();
                    attn_rows_dev(
                        &cb,
                        dq.ptr(),
                        &kv,
                        0,
                        d_naive.ptr(),
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
                    let before = splitk_launch_count();
                    attn_decode_dev(
                        &cb,
                        DecodeAttn::SplitK,
                        dq.ptr(),
                        &kv,
                        0,
                        d_split.ptr(),
                        datt.ptr(),
                        part.ptr(),
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
                    assert!(splitk_launch_count() > before, "split-K path not taken");
                    let (mut a, mut b) = (vec![0.0f32; q_dim], vec![0.0f32; q_dim]);
                    cb.download_f32(d_naive.ptr(), &mut a).unwrap();
                    cb.download_f32(d_split.ptr(), &mut b).unwrap();
                    let worst = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    for i in 0..q_dim {
                        assert!(
                            (a[i] - b[i]).abs() <= 2e-5 + 1e-4 * a[i].abs(),
                            "{elem:?} h{n_head}/{n_head_kv} d{head_dim} w{window} cap{softcap} pos{pos} i{i}: naive {} split {}",
                            a[i], b[i]
                        );
                    }
                    if pos == 8192 {
                        eprintln!("{elem:?} h{n_head}/{n_head_kv} d{head_dim} w{window} cap{softcap}: pos {pos} max|split-naive| {worst:.2e}");
                    }
                }
            }
        }
    }

    /// `ARTIFICER_FD_KERNEL=new` (`fd_partial_f16_g5d128` where
    /// [`fd2_partial_ok`], else the old partial; `fd_reduce_smem`) vs `old` (original
    /// `fd_partial_*` + `fd_reduce`) on the same device KV / q, bit for bit: the output
    /// AND every partial (o, m, l) of the call. Context lengths 1 .. 8193 (single key, empty
    /// warps, lengths not a multiple of the split / chunk, the 64-split cap); Nemotron GQA x5 /
    /// d128 f16 (new partial) with softcap, a sliding window and 2 KV heads; fallback shapes
    /// (kv_mul 1 / 4 / 8 / 6, d64, d32, d256, f32 KV) that run the old partial + new reduce.
    #[test]
    fn fd_kernel_new_matches_old_bitwise() {
        use super::super::cuda_forward::DeviceKvCache;
        let Some(cb) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut seed = 4242u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        };
        // (elem, n_head, n_head_kv, head_dim, window, softcap, expect new partial)
        let shapes = [
            (KvElem::F16, 40usize, 8usize, 128usize, 0usize, 0.0f32, true),
            (KvElem::F16, 40, 8, 128, 0, 30.0, true),
            (KvElem::F16, 40, 8, 128, 100, 0.0, true),
            (KvElem::F16, 10, 2, 128, 0, 0.0, true),
            (KvElem::F32, 40, 8, 128, 0, 0.0, false),
            (KvElem::F16, 32, 32, 64, 0, 0.0, false),
            (KvElem::F16, 8, 2, 32, 0, 30.0, false),
            (KvElem::F16, 16, 2, 128, 0, 0.0, false),
            (KvElem::F16, 20, 4, 64, 0, 0.0, false),
            (KvElem::F16, 6, 1, 256, 0, 0.0, false),
        ];
        let n_ctx = 8200usize;
        let positions = [
            0usize, 1, 2, 3, 4, 6, 30, 63, 64, 65, 100, 255, 256, 257, 526, 1000, 2039, 4097, 8000,
            8191, 8192,
        ];
        for &(elem, n_head, n_head_kv, head_dim, window, softcap, want_new) in &shapes {
            let kv_dim = n_head_kv * head_dim;
            let q_dim = n_head * head_dim;
            let kv = DeviceKvCache::new(&cb, &[true], n_ctx, kv_dim, elem).unwrap();
            let src = cb.alloc_f32(n_ctx * kv_dim).unwrap();
            for is_v in [false, true] {
                let rows: Vec<f32> = (0..n_ctx * kv_dim).map(|_| rnd()).collect();
                cb.upload_f32(&rows, src.ptr()).unwrap();
                kv.store_rows(&cb, 0, 0, n_ctx, src.ptr(), is_v).unwrap();
            }
            let dq = cb.alloc_f32(q_dim).unwrap();
            let datt = cb.alloc_f32(n_head * n_ctx).unwrap();
            let outs = [cb.alloc_f32(q_dim).unwrap(), cb.alloc_f32(q_dim).unwrap()];
            let parts = [
                cb.alloc_f32(fd_part_len(n_head, head_dim)).unwrap(),
                cb.alloc_f32(fd_part_len(n_head, head_dim)).unwrap(),
            ];
            let scale = 1.0 / (head_dim as f32).sqrt();
            let tag = format!("{elem:?} h{n_head}/{n_head_kv} d{head_dim} w{window} cap{softcap}");
            for &pos in &positions {
                let q: Vec<f32> = (0..q_dim).map(|_| rnd() * 6.0).collect();
                cb.upload_f32(&q, dq.ptr()).unwrap();
                let start = if window > 0 {
                    (pos + 1).saturating_sub(window)
                } else {
                    0
                };
                let (n_split, _) = fd_plan_with(pos + 1 - start, fd_policy().keys_per_split);
                let used = n_head * n_split * FD_WARPS * (head_dim + 2);
                let mut got: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
                for (i, sel) in [FdKernel::Old, FdKernel::New].into_iter().enumerate() {
                    set_thread_fd_kernel(Some(sel));
                    cb.memset0(parts[i].ptr()).unwrap();
                    cb.memset0(outs[i].ptr()).unwrap();
                    let (fd_before, new_before) =
                        (splitk_launch_count(), fd2_partial_launch_count());
                    attn_decode_dev(
                        &cb,
                        DecodeAttn::SplitK,
                        dq.ptr(),
                        &kv,
                        0,
                        outs[i].ptr(),
                        datt.ptr(),
                        parts[i].ptr(),
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
                    set_thread_fd_kernel(None);
                    assert!(
                        splitk_launch_count() > fd_before,
                        "{tag} pos{pos}: split-K path not taken"
                    );
                    let new_partial = fd2_partial_launch_count() > new_before;
                    assert_eq!(
                        new_partial,
                        sel == FdKernel::New && want_new,
                        "{tag} pos{pos} {sel:?}: partial kernel"
                    );
                    let (mut o, mut pp) = (vec![0.0f32; q_dim], vec![0.0f32; used]);
                    cb.download_f32(outs[i].ptr(), &mut o).unwrap();
                    cb.download_f32(parts[i].ptr(), &mut pp).unwrap();
                    got.push((o, pp));
                }
                let ((o_old, p_old), (o_new, p_new)) = (&got[0], &got[1]);
                let stride = head_dim + 2;
                for (k, (a, b)) in p_old.iter().zip(p_new).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{tag} pos{pos}: partial head {} part {} elem {}: old {a:e} new {b:e}",
                        k / (n_split * FD_WARPS * stride),
                        (k / stride) % (n_split * FD_WARPS),
                        k % stride
                    );
                }
                for (i, (a, b)) in o_old.iter().zip(o_new).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{tag} pos{pos} out[{i}]: old {a:e} new {b:e}"
                    );
                    assert!(a.is_finite(), "{tag} pos{pos} out[{i}] = {a}");
                }
            }
            eprintln!(
                "{tag}: new == old bitwise over {} positions (new partial: {want_new})",
                positions.len()
            );
        }
    }

    /// Flash prefill (`fa_prefill_*`) vs the old `attn_rows` path (in
    /// 163-row sub-blocks, as in production) on the same device KV, f16 and f32 cache,
    /// Nemotron GQA ×5 / MHA / softcap, arbitrary pos0, tail rows. f16 KV: both read the same
    /// halves (fp32 math, tight tolerance); f32 KV: flash rounds K/V to f16 per tile (≤ 2e-2).
    /// Rows past `n_rows` must stay untouched.
    #[test]
    fn flash_prefill_matches_old() {
        use super::super::cuda_forward::DeviceKvCache;
        let Some(cb) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut seed = 4242u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        };
        // (n_head, n_head_kv, softcap); head_dim 128.
        let shapes = [(40usize, 8usize, 0.0f32), (8, 8, 0.0), (8, 2, 30.0)];
        let head_dim = FA_D;
        let n_ctx = 2048usize;
        let n_rows_list = [1usize, 15, 64, 65, 163, 300, 1024];
        let pos0_list = [0usize, 7, 1000];
        let sub = 163usize;
        const SENTINEL: f32 = 1234.5;
        for elem in [KvElem::F16, KvElem::F32] {
            for &(n_head, n_head_kv, softcap) in &shapes {
                let kv_dim = n_head_kv * head_dim;
                let q_dim = n_head * head_dim;
                let kv = DeviceKvCache::new(&cb, &[true], n_ctx, kv_dim, elem).unwrap();
                let src = cb.alloc_f32(n_ctx * kv_dim).unwrap();
                for is_v in [false, true] {
                    let rows: Vec<f32> = (0..n_ctx * kv_dim).map(|_| rnd()).collect();
                    cb.upload_f32(&rows, src.ptr()).unwrap();
                    kv.store_rows(&cb, 0, 0, n_ctx, src.ptr(), is_v).unwrap();
                }
                let max_rows = 1024usize;
                let dq = cb.alloc_f32(max_rows * q_dim).unwrap();
                let d_old = cb.alloc_f32(max_rows * q_dim).unwrap();
                let d_fa = cb.alloc_f32((max_rows + 1) * q_dim).unwrap();
                let datt = cb.alloc_f32(sub * n_head * n_ctx).unwrap();
                let scale = 1.0 / (head_dim as f32).sqrt();
                // [scalar, mma] worst |flash - old|.
                let mut worst_all = [0.0f32; 2];
                for &pos0 in &pos0_list {
                    for &n_rows in &n_rows_list {
                        let len = n_rows * q_dim;
                        let q: Vec<f32> = (0..len).map(|_| rnd() * 6.0).collect();
                        cb.upload_f32(&q, dq.ptr()).unwrap();
                        let mut r0 = 0;
                        while r0 < n_rows {
                            let m = (n_rows - r0).min(sub);
                            attn_rows_dev(
                                &cb,
                                dq.ptr().offset(r0 * q_dim, m * q_dim),
                                &kv,
                                0,
                                d_old.ptr().offset(r0 * q_dim, m * q_dim),
                                datt.ptr(),
                                m,
                                pos0 + r0,
                                n_head,
                                n_head_kv,
                                head_dim,
                                q_dim,
                                0,
                                scale,
                                softcap,
                            )
                            .unwrap();
                            r0 += m;
                        }
                        let mut a = vec![0.0f32; len];
                        cb.download_f32(d_old.ptr().prefix(len), &mut a).unwrap();
                        for (vi, variant) in [PrefillAttn::FlashScalar, PrefillAttn::Flash]
                            .into_iter()
                            .enumerate()
                        {
                            cb.upload_f32(&vec![SENTINEL; (n_rows + 1) * q_dim], d_fa.ptr())
                                .unwrap();
                            let before = flash_prefill_launch_count();
                            let mma_before = flash_mma_launch_count();
                            attn_prefill_flash_dev(
                                &cb,
                                variant,
                                dq.ptr().prefix(len),
                                &kv,
                                0,
                                d_fa.ptr().prefix(len),
                                n_rows,
                                pos0,
                                n_head,
                                n_head_kv,
                                head_dim,
                                q_dim,
                                scale,
                                softcap,
                            )
                            .unwrap();
                            assert!(
                                flash_prefill_launch_count() > before,
                                "flash path not taken"
                            );
                            assert_eq!(
                                flash_mma_launch_count() > mma_before,
                                variant == PrefillAttn::Flash,
                                "{variant:?}: wrong kernel"
                            );
                            let mut b = vec![0.0f32; (n_rows + 1) * q_dim];
                            cb.download_f32(d_fa.ptr().prefix((n_rows + 1) * q_dim), &mut b)
                                .unwrap();
                            assert!(
                                b[len..].iter().all(|&x| x == SENTINEL),
                                "{variant:?} wrote past n_rows"
                            );
                            let mut worst = 0.0f32;
                            for i in 0..len {
                                let d = (a[i] - b[i]).abs();
                                worst = worst.max(d);
                                // Scalar (M1): f16 smem K/V like the old f16 path. Tensor-core
                                // (M2): design tolerance, 2e-2 abs (f16 Q and P operands).
                                let tol = match (variant, elem) {
                                    (PrefillAttn::Flash, _) => 2e-2,
                                    (_, KvElem::F16) => 1e-4 + 1e-3 * a[i].abs(),
                                    (_, KvElem::F32) => 2e-2,
                                };
                                assert!(
                                    d <= tol && b[i].is_finite(),
                                    "{variant:?} {elem:?} h{n_head}/{n_head_kv} cap{softcap} pos0 {pos0} rows {n_rows} row {} i{}: old {} flash {}",
                                    i / q_dim, i % q_dim, a[i], b[i]
                                );
                            }
                            worst_all[vi] = worst_all[vi].max(worst);
                        }
                    }
                }
                eprintln!(
                    "{elem:?} h{n_head}/{n_head_kv} cap{softcap}: max|scalar-old| {:.2e} max|mma-old| {:.2e}",
                    worst_all[0], worst_all[1]
                );
            }
        }
    }

    /// Tensor-core kernel debug check: with the |Q| check on, a Q element beyond the f16
    /// range makes the launch return an error; in-range Q passes; check off never errors.
    #[test]
    fn flash_mma_qcheck_flags_f16_overflow() {
        use super::super::cuda_forward::DeviceKvCache;
        let Some(cb) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let (n_head, n_head_kv, head_dim, n_ctx, n_rows) =
            (8usize, 8usize, FA_D, 128usize, 64usize);
        let (kv_dim, q_dim) = (n_head_kv * head_dim, n_head * head_dim);
        let kv = DeviceKvCache::new(&cb, &[true], n_ctx, kv_dim, KvElem::F16).unwrap();
        let src = cb.alloc_f32(n_ctx * kv_dim).unwrap();
        cb.upload_f32(&vec![0.01; n_ctx * kv_dim], src.ptr())
            .unwrap();
        for is_v in [false, true] {
            kv.store_rows(&cb, 0, 0, n_ctx, src.ptr(), is_v).unwrap();
        }
        let dq = cb.alloc_f32(n_rows * q_dim).unwrap();
        let dout = cb.alloc_f32(n_rows * q_dim).unwrap();
        let run = |q: &[f32]| {
            cb.upload_f32(q, dq.ptr()).unwrap();
            let r = attn_prefill_flash_dev(
                &cb,
                PrefillAttn::Flash,
                dq.ptr(),
                &kv,
                0,
                dout.ptr(),
                n_rows,
                0,
                n_head,
                n_head_kv,
                head_dim,
                q_dim,
                0.088,
                0.0,
            );
            cb.sync().unwrap();
            r
        };
        let ok = vec![0.5f32; n_rows * q_dim];
        let mut bad = ok.clone();
        bad[37 * q_dim + 5 * head_dim + 77] = 1.0e5;
        set_thread_fa_qcheck(Some(true));
        run(&ok).unwrap();
        let e = run(&bad).expect_err("|Q| = 1e5 must be flagged");
        assert!(e.to_string().contains("f16 range"), "{e}");
        set_thread_fa_qcheck(Some(false));
        run(&bad).unwrap();
        set_thread_fa_qcheck(None);
    }

    /// Manual kernel benchmark (GPU window): Nemotron shape (40/8 heads, d128, f16 KV), one
    /// 1024-row prefill chunk at several pos0: old `attn_rows` in 163-row sub-blocks vs the
    /// scalar and tensor-core flash kernels. Prints ms per chunk-layer.
    #[test]
    #[ignore = "manual benchmark"]
    fn flash_prefill_kernel_bench() {
        use super::super::cuda_forward::DeviceKvCache;
        let Some(cb) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let (n_head, n_head_kv, head_dim, n_ctx, n_rows, sub) =
            (40usize, 8usize, FA_D, 8192usize, 1024usize, 163usize);
        let (kv_dim, q_dim) = (n_head_kv * head_dim, n_head * head_dim);
        let mut seed = 7u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5
        };
        let kv = DeviceKvCache::new(&cb, &[true], n_ctx, kv_dim, KvElem::F16).unwrap();
        let src = cb.alloc_f32(n_ctx * kv_dim).unwrap();
        for is_v in [false, true] {
            let rows: Vec<f32> = (0..n_ctx * kv_dim).map(|_| rnd()).collect();
            cb.upload_f32(&rows, src.ptr()).unwrap();
            kv.store_rows(&cb, 0, 0, n_ctx, src.ptr(), is_v).unwrap();
        }
        drop(src);
        let dq = cb.alloc_f32(n_rows * q_dim).unwrap();
        let q: Vec<f32> = (0..n_rows * q_dim).map(|_| rnd() * 6.0).collect();
        cb.upload_f32(&q, dq.ptr()).unwrap();
        let dout = cb.alloc_f32(n_rows * q_dim).unwrap();
        let datt = cb.alloc_f32(sub * n_head * n_ctx).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (warm, reps) = (2, 5);
        for pos0 in [0usize, 1024, 3072, 7168] {
            let mut line = format!("pos0 {pos0:5} rows {n_rows}:");
            let mut old_ms = 0.0f64;
            for mode in [
                PrefillAttn::Old,
                PrefillAttn::FlashScalar,
                PrefillAttn::Flash,
            ] {
                let once = || {
                    if mode == PrefillAttn::Old {
                        let mut r0 = 0;
                        while r0 < n_rows {
                            let m = (n_rows - r0).min(sub);
                            attn_rows_dev(
                                &cb,
                                dq.ptr().offset(r0 * q_dim, m * q_dim),
                                &kv,
                                0,
                                dout.ptr().offset(r0 * q_dim, m * q_dim),
                                datt.ptr(),
                                m,
                                pos0 + r0,
                                n_head,
                                n_head_kv,
                                head_dim,
                                q_dim,
                                0,
                                scale,
                                0.0,
                            )
                            .unwrap();
                            r0 += m;
                        }
                    } else {
                        attn_prefill_flash_dev(
                            &cb,
                            mode,
                            dq.ptr(),
                            &kv,
                            0,
                            dout.ptr(),
                            n_rows,
                            pos0,
                            n_head,
                            n_head_kv,
                            head_dim,
                            q_dim,
                            scale,
                            0.0,
                        )
                        .unwrap();
                    }
                };
                for _ in 0..warm {
                    once();
                }
                cb.sync().unwrap();
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    once();
                }
                cb.sync().unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
                if mode == PrefillAttn::Old {
                    old_ms = ms;
                    line += &format!(" old {ms:.3} ms");
                } else {
                    line += &format!(" | {mode:?} {ms:.3} ms ({:.2}x)", old_ms / ms);
                }
            }
            eprintln!("{line}");
        }
    }
}
