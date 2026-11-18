//! Device implementations of the "host ops" of the CUDA device-resident forward
//! ([`DeviceHostOps`], called from `cuda_forward`). Kernels are the `ho_*` block at the
//! end of `cuda_kernels.cu` (same NVRTC module as the device-forward kernels).
//!
//! Contract (device-API design): every hook launches async on
//! `cb.stream()`, never syncs, never copies activations to/from the host. Small statics (norm
//! weights, biases) come from [`CudaBackend::resident_f32`] (dequantized + uploaded once, cached
//! for the backend's lifetime; the very first lookup of each tensor does one upload + sync inside
//! that cache, steady-state lookups do not).
//!
//! | hook | device | host reference | numerics vs host |
//! |---|---|---|---|
//! | `norm` (Rms / RmsGemma / Layer) | `ho_rms_norm`, `ho_layer_norm` | `ops::normalize` | parallel reduction: ≤ ~1e-6 rel |
//! | `add_inplace` | `ho_add` | `ops::add_inplace` | bit-identical |
//! | `add_bias` | `ho_add_bias` | `forward::weights::add_bias` | bit-identical |
//! | `act` (SiLU·, GELU·, ReLU²) | `ho_glu`, `ho_relu_sqr` | `ops::{silu,gelu,relu_sqr}_inplace` + mul | ReLU² exact, SiLU/GELU ≤ 1 ulp |
//! | `scale` | `ho_scale` | `ops::scale_inplace` | bit-identical |
//! | `softcap` | `ho_softcap` | `ops::softcap_inplace` | ≤ 1 ulp |
//! | `rope` | `ho_rope` | `ops::apply_rope` | ≤ 1 ulp (pow/cos/sin in f64) |
//! | `embed` (F32/F16/BF16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K) | `ho_embed` | `quant::dequantize` (+ √d) | bit-identical |
//! | `ssm_mix` (Mamba-2 conv+scan+gate+group-norm) | `ho_ssm_conv_tiled`, `ho_ssm_scan_tiled[_ns4]`, `ho_ssm_gate_norm_rows` (prefill); `ho_ssm_conv_rows`, `ho_ssm_scan_rows` (decode / `ARTIFICER_SSM_NAIVE=1`); `ho_ssm_conv`, `ho_ssm_scan`, `ho_ssm_gate_norm` (row loop) | `ssm::mamba2_mix_host` | ≤ ~1e-5 rel (reduction order); all device variants bit-identical to each other |
//!
//! `embed` keeps the raw (quantized) token-embedding table resident on the device (Nemotron
//! Q5_0: 270 MiB, tiny-vicuna Q4_K: 35 MiB) when it fits (`check_fits`), else declines (host path).
//!
//! A/B switches: `ARTIFICER_HOSTOPS=0` (process) or [`set_thread_disabled`] (current thread)
//! make every hook return `Ok(false)` → the host fallback path in `cuda_forward`. Do not flip the switch in
//! the middle of a sequence on a hybrid model: the SSM state then lives on one side only.
//!
//! SSM kernel switches (numerically invisible: every variant is bit-identical):
//! `ARTIFICER_SSM_NAIVE=1` forces the pre-tiling `ho_ssm_conv_rows` / `ho_ssm_scan_rows` for every
//! call (A/B benches from one binary); `ARTIFICER_SSM_TILED_MIN_ROWS=N` (default
//! [`SSM_TILED_MIN_ROWS`]) is the smallest chunk that uses the tiled prefill kernels.
//!
//! SSM prefill numerics switch (NOT numerically invisible): `ARTIFICER_PREFILL_SSM=old|fp32`
//! ([`parse_prefill_ssm`], read once; unset, empty or unknown follow the single constant
//! [`PREFILL_SSM_DEFAULT`], currently `fp32`, the last with one warning). An explicit `old`
//! always forces the original arithmetic (SiLU via the fp64 `ho_expf`). `fp32` makes the tiled
//! prefill chunks (>= [`SSM_TILED_MIN_ROWS`] rows) use `ho_ssm_conv_tiled_f32` +
//! `ho_ssm_gate_norm_rows_f32` (SiLU with fp32 `expf`, <= 2 ulp); the scan, dt / dA and every
//! sub-threshold call (decode, prefill tail chunks) are unchanged.
//! `fp32` also (same math, bit-identical to `ho_ssm_scan_tiled`) runs the scan of those
//! chunks with `ho_ssm_scan_fast_ns4` (d_state 128, 2 channels per warp, cp.async double-buffered
//! B / C tiles; [`ssm_scan_fast_ok`]) and processes up to [`SSM_CHUNK_ROWS_FAST`] rows per
//! 3-launch chunk (larger conv-output scratch, chosen when the device state is created).

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use gguf::TensorType;

use crate::config::{FfnActivation, NormKind, SsmConfig};
use crate::error::{CoreError, Result};
use crate::model::{LlamaModel, WeightTensor};
use crate::ssm::SsmCache;

use super::cuda::{CudaBackend, DevPtr, DeviceTensor, LaunchArgs};
use super::cuda_forward::DeviceHostOps;

/// The CUDA [`DeviceHostOps`] (the instance `cuda_forward::hostops()` returns).
pub struct CudaHostOps;

pub static OPS: CudaHostOps = CudaHostOps;

thread_local! {
    static THREAD_OFF: Cell<bool> = const { Cell::new(false) };
}

/// Disable (or re-enable) every hook on the current thread (A/B tests vs the host-fallback path).
pub fn set_thread_disabled(off: bool) {
    THREAD_OFF.with(|c| c.set(off));
}

/// `false` when `ARTIFICER_HOSTOPS=0` or [`set_thread_disabled`]`(true)` on this thread.
pub fn enabled() -> bool {
    static ENV_OFF: OnceLock<bool> = OnceLock::new();
    !THREAD_OFF.with(|c| c.get())
        && !*ENV_OFF.get_or_init(|| std::env::var("ARTIFICER_HOSTOPS").is_ok_and(|v| v == "0"))
}

const BLOCK: u32 = 256;

/// Grid for the grid-stride element-wise kernels.
fn grid_for(n: usize) -> (u32, u32, u32) {
    ((n.div_ceil(BLOCK as usize)).clamp(1, 65535) as u32, 1, 1)
}

fn need(p: DevPtr, n: usize, what: &str) -> Result<()> {
    if p.len < n {
        return Err(CoreError::Shape(format!(
            "cuda_hostops {what}: device buffer {} < {n}",
            p.len
        )));
    }
    Ok(())
}

thread_local! {
    static LAUNCHES: Cell<u64> = const { Cell::new(0) };
}

/// `ho_*` kernel launches issued by these hooks **on the current thread** (per-thread so
/// concurrent tests / engines do not disturb each other's counts).
pub fn launch_count() -> u64 {
    LAUNCHES.with(|c| c.get())
}

fn go(
    cb: &CudaBackend,
    name: &str,
    grid: (u32, u32, u32),
    block: u32,
    args: LaunchArgs,
) -> Result<()> {
    go_sh(cb, name, grid, block, 0, args)
}

/// [`go`] with `shmem` bytes of dynamic shared memory.
fn go_sh(
    cb: &CudaBackend,
    name: &str,
    grid: (u32, u32, u32),
    block: u32,
    shmem: u32,
    args: LaunchArgs,
) -> Result<()> {
    LAUNCHES.with(|c| c.set(c.get() + 1));
    cb.launch(cb.kernel(name)?, grid, (block, 1, 1), shmem, args)
}

/// `ho_embed` type code (see `ho_deq` in cuda_kernels.cu) and block size in elements.
fn embed_code(ty: TensorType) -> Option<(i32, usize)> {
    Some(match ty {
        TensorType::F32 => (0, 1),
        TensorType::F16 => (1, 1),
        TensorType::BF16 => (2, 1),
        TensorType::Q8_0 => (3, 32),
        TensorType::Q5_0 => (4, 32),
        TensorType::Q4_K => (5, 256),
        TensorType::Q6_K => (6, 256),
        TensorType::Q4_0 => (7, 32),
        _ => return None,
    })
}

/// Raw token-embedding bytes resident on the device (via the backend's statics cache, stored as an
/// "F32" byte copy so no dequant happens). `None` when it does not fit.
fn embed_table(cb: &CudaBackend, model: &LlamaModel) -> Result<Option<DevPtr>> {
    let bytes = model.tensor_bytes(&model.token_embd)?;
    if bytes.len() % 4 != 0 {
        return Ok(None);
    }
    type Decided = Mutex<HashMap<(usize, usize, usize), bool>>;
    static DECIDED: OnceLock<Decided> = OnceLock::new();
    let key = (
        cb as *const CudaBackend as usize,
        bytes.as_ptr() as usize,
        bytes.len(),
    );
    let map = DECIDED.get_or_init(|| Mutex::new(HashMap::new()));
    let decided = map.lock().ok().and_then(|m| m.get(&key).copied());
    let ok = match decided {
        Some(v) => v,
        None => {
            let v = cb
                .check_fits(bytes.len() as u64, "token embedding table")
                .is_ok();
            if let Ok(mut m) = map.lock() {
                m.insert(key, v);
            }
            v
        }
    };
    if !ok {
        return Ok(None);
    }
    Ok(Some(cb.resident_f32_from(
        bytes,
        TensorType::F32,
        bytes.len() / 4,
    )?))
}

impl DeviceHostOps for CudaHostOps {
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
        if !enabled() || w.n_elements() != dim || dim == 0 || rows == 0 {
            return Ok(false);
        }
        need(x, rows * dim, "norm x")?;
        need(out, rows * dim, "norm out")?;
        let cfg = &model.config;
        let wd = cb.resident_f32(model, w)?;
        match cfg.norm_kind {
            NormKind::Rms | NormKind::RmsGemma => {
                let gemma = (cfg.norm_kind == NormKind::RmsGemma) as i32;
                let args = LaunchArgs::new()
                    .ptr(x)
                    .ptr(wd)
                    .ptr(out)
                    .i32(dim as i32)
                    .f32(cfg.rms_eps)
                    .i32(gemma);
                go(cb, "ho_rms_norm", (rows as u32, 1, 1), BLOCK, args)?;
            }
            NormKind::Layer => {
                let bd = match b {
                    Some(bt) if bt.n_elements() >= dim => Some(cb.resident_f32(model, bt)?),
                    Some(_) => return Ok(false),
                    None => None,
                };
                let args = LaunchArgs::new()
                    .ptr(x)
                    .ptr(wd)
                    .ptr(bd.unwrap_or(wd))
                    .ptr(out)
                    .i32(dim as i32)
                    .f32(cfg.rms_eps)
                    .i32(bd.is_some() as i32);
                go(cb, "ho_layer_norm", (rows as u32, 1, 1), BLOCK, args)?;
            }
        }
        Ok(true)
    }

    fn add_inplace(&self, cb: &CudaBackend, a: DevPtr, b: DevPtr) -> Result<bool> {
        if !enabled() {
            return Ok(false);
        }
        let n = a.len;
        need(b, n, "add b")?;
        if n == 0 {
            return Ok(true);
        }
        go(
            cb,
            "ho_add",
            grid_for(n),
            BLOCK,
            LaunchArgs::new().ptr(a).ptr(b).i32(n as i32),
        )?;
        Ok(true)
    }

    fn add_bias(
        &self,
        cb: &CudaBackend,
        model: &LlamaModel,
        bias: &WeightTensor,
        out: DevPtr,
        rows: usize,
        dim: usize,
    ) -> Result<bool> {
        if !enabled() {
            return Ok(false);
        }
        need(out, rows * dim, "bias out")?;
        let nb = bias.n_elements().min(dim);
        if rows == 0 || nb == 0 {
            return Ok(true);
        }
        let bd = cb.resident_f32(model, bias)?;
        let args = LaunchArgs::new()
            .ptr(out)
            .ptr(bd)
            .i32(rows as i32)
            .i32(dim as i32)
            .i32(nb as i32);
        go(cb, "ho_add_bias", grid_for(rows * dim), BLOCK, args)?;
        Ok(true)
    }

    fn act(
        &self,
        cb: &CudaBackend,
        act: FfnActivation,
        hb: DevPtr,
        hb2: Option<DevPtr>,
    ) -> Result<bool> {
        if !enabled() {
            return Ok(false);
        }
        let n = hb.len;
        if n == 0 {
            return Ok(true);
        }
        match (act, hb2) {
            (FfnActivation::SiluGated | FfnActivation::GeluGated, Some(h2)) => {
                need(h2, n, "act hb2")?;
                let kind = (act == FfnActivation::GeluGated) as i32;
                let args = LaunchArgs::new().ptr(hb).ptr(h2).i32(n as i32).i32(kind);
                go(cb, "ho_glu", grid_for(n), BLOCK, args)?;
            }
            (FfnActivation::ReluSqrSequential, None) => {
                go(
                    cb,
                    "ho_relu_sqr",
                    grid_for(n),
                    BLOCK,
                    LaunchArgs::new().ptr(hb).i32(n as i32),
                )?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn scale(&self, cb: &CudaBackend, x: DevPtr, s: f32) -> Result<bool> {
        if !enabled() {
            return Ok(false);
        }
        if x.len > 0 {
            let args = LaunchArgs::new().ptr(x).f32(s).i32(x.len as i32);
            go(cb, "ho_scale", grid_for(x.len), BLOCK, args)?;
        }
        Ok(true)
    }

    fn softcap(&self, cb: &CudaBackend, x: DevPtr, cap: f32) -> Result<bool> {
        if !enabled() {
            return Ok(false);
        }
        if cap > 0.0 && x.len > 0 {
            let args = LaunchArgs::new().ptr(x).f32(cap).i32(x.len as i32);
            go(cb, "ho_softcap", grid_for(x.len), BLOCK, args)?;
        }
        Ok(true)
    }

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
        if !enabled() {
            return Ok(false);
        }
        let n_rot = n_rot.min(head_dim);
        if rows == 0 || n_head == 0 || n_rot == 0 {
            return Ok(true);
        }
        need(x, (rows - 1) * row_stride + n_head * head_dim, "rope x")?;
        let total = rows * n_head * n_rot.div_ceil(2);
        let args = LaunchArgs::new()
            .ptr(x)
            .i32(rows as i32)
            .i32(row_stride as i32)
            .i32(n_head as i32)
            .i32(head_dim as i32)
            .i32(pos0 as i32)
            .f32(freq_base)
            .i32(n_rot as i32);
        go(cb, "ho_rope", grid_for(total), BLOCK, args)?;
        Ok(true)
    }

    fn embed(
        &self,
        cb: &CudaBackend,
        model: &LlamaModel,
        tokens: &[u32],
        out: DevPtr,
    ) -> Result<bool> {
        if !enabled() {
            return Ok(false);
        }
        let cfg = &model.config;
        let n_embd = cfg.n_embd as usize;
        let t = &model.token_embd;
        let Some((code, blk)) = embed_code(t.ty) else {
            return Ok(false);
        };
        if n_embd == 0
            || !n_embd.is_multiple_of(blk)
            || t.n_elements() < cfg.n_vocab as usize * n_embd
        {
            return Ok(false);
        }
        need(out, tokens.len() * n_embd, "embed out")?;
        let Some(table) = embed_table(cb, model)? else {
            return Ok(false);
        };
        let scale = (n_embd as f32).sqrt();
        for (r, &tok) in tokens.iter().enumerate() {
            if tok as usize >= cfg.n_vocab as usize {
                return Err(CoreError::Other(format!(
                    "token id {tok} out of vocab {}",
                    cfg.n_vocab
                )));
            }
            let args = LaunchArgs::new()
                .ptr(table)
                .ptr(out.row(r, n_embd))
                .i32(tok as i32)
                .i32(n_embd as i32)
                .i32(code)
                .f32(scale)
                .i32(cfg.scale_embeddings as i32);
            go(cb, "ho_embed", grid_for(n_embd), BLOCK, args)?;
        }
        Ok(true)
    }

    fn has_ssm_mix(&self) -> bool {
        enabled()
    }

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
        if !enabled() {
            return Ok(false);
        }
        let scfg = model
            .config
            .ssm
            .ok_or_else(|| CoreError::Other("ssm_mix without SsmConfig".into()))?;
        ssm_mix_dev(cb, ssm, scfg, model.config.rms_eps, layer_idx, zx, y, n_tok)
    }
}

// ---------------------------------------------------------------------------
// Device SSM state (lives in `SsmCache::device`)
// ---------------------------------------------------------------------------

struct SsmSlotDev {
    /// Conv state `[(d_conv-1)][conv_dim]` (same layout as the host `SsmCache`).
    conv: DeviceTensor,
    /// SSM state `[n_head][head_dim][d_state]`.
    ssm: DeviceTensor,
    conv_w: DeviceTensor,
    conv_b: DeviceTensor,
    a: DeviceTensor,
    d: DeviceTensor,
    dt_b: DeviceTensor,
    norm: DeviceTensor,
}

/// Rows per multi-token `ssm_mix` chunk (= scratch capacity; matches the proposal's prefill
/// `max_batch`). Larger `n_tok` is processed in chunks of this many rows (3 launches each).
pub(crate) const SSM_CHUNK_ROWS: usize = 256;
/// Rows per chunk when the device SSM state is created in `ARTIFICER_PREFILL_SSM=fp32` mode
/// (256 = [`SSM_CHUNK_ROWS`]: chunk A/B at p8k measured 256 / 512 / 1024 rows at 4.272 /
/// 4.307 / 4.313 s TTFT; 1024-row chunks cost ~34 ms in `ho_ssm_gate_norm_rows_f32`, working
/// set beyond L2). Chunking is numerically invisible (conv window and SSM state carry exactly
/// between chunks).
pub(crate) const SSM_CHUNK_ROWS_FAST: usize = 256;
/// Register-window limits of the multi-row kernels (`HO_CONV_MAX`, `32 * HO_SCAN_NS`); larger
/// shapes use the per-row kernels.
const SSM_ROWS_MAX_D_CONV: usize = 8;
const SSM_ROWS_MAX_D_STATE: usize = 256;
/// Tiled-kernel geometry; must match `HO_CONV_TT`, `HO_SCAN_T`, `HO_SCAN_W` in cuda_kernels.cu.
const SSM_CONV_TT: usize = 8;
const SSM_SCAN_T: usize = 16;
const SSM_SCAN_W: usize = 8;
/// `ho_ssm_scan_fast_ns4` geometry; must match `HO_SCAN2_W`, `HO_SCAN2_C`, `HoScan2<4>`.
const SSM_SCAN2_THREADS: u32 = 32 * 8;
const SSM_SCAN2_CH: usize = 8 * 2;
const SSM_SCAN2_D_STATE: usize = 128;
/// Dynamic smem of `ho_ssm_scan_fast_ns4`: 2 stages of B, C `[16][128]`, x `[16][16]`, dt, dA
/// `[16]`, plus 2 y buffers `[16][16]` (f32) = 37120 bytes (< the 48 KiB default limit).
const SSM_SCAN2_SMEM: u32 = (4
    * (2 * (2 * SSM_SCAN_T * SSM_SCAN2_D_STATE + SSM_SCAN_T * SSM_SCAN2_CH + 2 * SSM_SCAN_T)
        + 2 * SSM_SCAN_T * SSM_SCAN2_CH)) as u32;

#[cfg(test)]
thread_local! {
    /// Test: `TiledFp32` with the old `ho_ssm_scan_tiled[_ns4]` (= the original fp32 mode,
    /// before the fast scan), the reference for the fast scan's bit-identity test.
    static TEST_OLD_SCAN: Cell<bool> = const { Cell::new(false) };
}
#[cfg(test)]
fn test_old_scan() -> bool {
    TEST_OLD_SCAN.with(|c| c.get())
}
#[cfg(not(test))]
#[inline(always)]
fn test_old_scan() -> bool {
    false
}

/// Shapes `ho_ssm_scan_fast_ns4` handles: d_state == 128 and 16-byte aligned B / C rows
/// (`conv_dim`, `d_inner` multiples of 4). Others keep `ho_ssm_scan_tiled[_ns4]` (same results).
pub(crate) fn ssm_scan_fast_ok(scfg: &SsmConfig) -> bool {
    scfg.d_state as usize == SSM_SCAN2_D_STATE
        && scfg.conv_dim().is_multiple_of(4)
        && scfg.d_inner.is_multiple_of(4)
}
/// Smallest chunk (rows) that uses the tiled prefill kernels; smaller calls (decode = 1 row) keep
/// the `*_rows` kernels (identical results). Override: `ARTIFICER_SSM_TILED_MIN_ROWS`.
pub(crate) const SSM_TILED_MIN_ROWS: usize = 8;

/// Which multi-row SSM kernels a chunk of `n` rows uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SsmKernels {
    /// `ho_ssm_conv_rows` + `ho_ssm_scan_rows` (one thread per channel / sequential warps).
    Rows,
    /// `ho_ssm_conv_tiled` + `ho_ssm_scan_tiled[_ns4]` (parallel over positions, smem tiles).
    Tiled,
    /// [`SsmKernels::Tiled`] with the fp32-SiLU conv / gate (`ho_ssm_conv_tiled_f32`,
    /// `ho_ssm_gate_norm_rows_f32`); `ARTIFICER_PREFILL_SSM=fp32`.
    TiledFp32,
}

/// SSM prefill numerics (`ARTIFICER_PREFILL_SSM`), see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillSsm {
    /// Original arithmetic (fp64 `ho_expf` SiLU).
    Old,
    /// fp32 `expf` SiLU in the tiled prefill conv + gate kernels, fast scan. Default.
    Fp32,
}

/// Mode used when `ARTIFICER_PREFILL_SSM` is unset, empty or unknown. Flipping the default is
/// this one line; an explicit `old` always forces the old path.
pub const PREFILL_SSM_DEFAULT: PrefillSsm = PrefillSsm::Fp32;

/// Parse `ARTIFICER_PREFILL_SSM` (trimmed, case-insensitive): `old` -> Old, `fp32` -> Fp32;
/// unset or empty -> [`PREFILL_SSM_DEFAULT`]; anything else -> the default with `unknown = true`
/// (the caller warns once).
/// Returns `(mode, unknown)`.
pub fn parse_prefill_ssm(v: Option<&str>) -> (PrefillSsm, bool) {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => (PREFILL_SSM_DEFAULT, false),
        Some("old") => (PrefillSsm::Old, false),
        Some("fp32") => (PrefillSsm::Fp32, false),
        Some(_) => (PREFILL_SSM_DEFAULT, true),
    }
}

/// Process-wide [`PrefillSsm`] from `ARTIFICER_PREFILL_SSM`, read once (one warning if unknown).
pub fn prefill_ssm() -> PrefillSsm {
    static V: OnceLock<PrefillSsm> = OnceLock::new();
    *V.get_or_init(|| {
        let raw = std::env::var("ARTIFICER_PREFILL_SSM").ok();
        let (mode, unknown) = parse_prefill_ssm(raw.as_deref());
        if unknown {
            tracing::warn!(
                "ARTIFICER_PREFILL_SSM={:?} is not one of old|fp32; using {:?}",
                raw.unwrap_or_default(),
                PREFILL_SSM_DEFAULT
            );
        }
        mode
    })
}

/// Kernel family for a chunk of `n` rows (pure; [`ssm_kernels_for`] feeds it the env switches).
fn ssm_kernels_pick(n: usize, naive: bool, min_rows: usize, mode: PrefillSsm) -> SsmKernels {
    if naive || n < min_rows {
        SsmKernels::Rows
    } else if mode == PrefillSsm::Fp32 {
        SsmKernels::TiledFp32
    } else {
        SsmKernels::Tiled
    }
}

fn ssm_kernels_for(n: usize) -> SsmKernels {
    static CFG: OnceLock<(bool, usize)> = OnceLock::new();
    let (naive, min_rows) = *CFG.get_or_init(|| {
        let naive = std::env::var("ARTIFICER_SSM_NAIVE").is_ok_and(|v| v == "1");
        let min_rows = std::env::var("ARTIFICER_SSM_TILED_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(SSM_TILED_MIN_ROWS);
        (naive, min_rows)
    });
    ssm_kernels_pick(n, naive, min_rows, prefill_ssm())
}

/// Dynamic shared memory of `ho_ssm_scan_tiled` for `d_state` (B, C, x, y, dt, dA tiles).
fn ssm_scan_tiled_smem(d_state: usize) -> u32 {
    (4 * (2 * SSM_SCAN_T * d_state + 2 * SSM_SCAN_T * SSM_SCAN_W + 2 * SSM_SCAN_T)) as u32
}

/// Device copy of an [`SsmCache`]: per-SSM-layer conv + SSM state and the layer statics
/// (uploaded once from the host cache's pre-dequantized statics), plus a conv-output scratch.
pub struct DeviceSsmState {
    slots: Vec<SsmSlotDev>,
    /// Conv output `[SSM_CHUNK_ROWS][conv_dim]` (= `[x | B | C]` after SiLU).
    xbc: DeviceTensor,
    /// Per-(row, head) `[dt | exp(dt·A)]`, `[SSM_CHUNK_ROWS][2 * n_head]`.
    dta: DeviceTensor,
    /// Rows per 3-launch chunk = `xbc` / `dta` capacity ([`SSM_CHUNK_ROWS`], or
    /// [`SSM_CHUNK_ROWS_FAST`] when created under `ARTIFICER_PREFILL_SSM=fp32`).
    chunk_rows: usize,
    zero_pending: bool,
}

impl DeviceSsmState {
    /// Allocate + upload (async; no sync). Initial state = the host cache's current state.
    fn new(cb: &CudaBackend, cache: &SsmCache) -> Result<Self> {
        let n = cache.n_ssm_layers;
        let per_slot = ((cache.d_conv.saturating_sub(1)) * cache.conv_dim
            + cache.n_head * cache.head_dim * cache.d_state
            + cache.d_conv * cache.conv_dim
            + cache.conv_dim
            + 3 * cache.n_head
            + cache.d_inner)
            * 4;
        let chunk_rows = match prefill_ssm() {
            PrefillSsm::Old => SSM_CHUNK_ROWS,
            PrefillSsm::Fp32 => SSM_CHUNK_ROWS_FAST,
        };
        let scratch = chunk_rows * (cache.conv_dim + 2 * cache.n_head) * 4;
        cb.check_fits((per_slot * n + scratch) as u64, "device SSM state")?;
        let up = |v: &[f32]| -> Result<DeviceTensor> {
            let t = cb.alloc_bytes(v.len() * 4)?;
            if v.iter().all(|&x| x.to_bits() == 0) {
                cb.memset0(t.ptr())?;
            } else {
                cb.upload_f32(v, t.ptr())?;
            }
            Ok(t)
        };
        let mut slots = Vec::with_capacity(n);
        for s in 0..n {
            let st = cache.statics(s);
            let (conv, ssm) = cache.host_state(s);
            slots.push(SsmSlotDev {
                conv: up(conv)?,
                ssm: up(ssm)?,
                conv_w: up(&st.conv_w)?,
                conv_b: up(&st.conv_b)?,
                a: up(&st.a)?,
                d: up(&st.d)?,
                dt_b: up(&st.dt_b)?,
                norm: up(&st.norm)?,
            });
        }
        Ok(Self {
            slots,
            xbc: cb.alloc_f32((chunk_rows * cache.conv_dim).max(1))?,
            dta: cb.alloc_f32((chunk_rows * 2 * cache.n_head).max(1))?,
            chunk_rows,
            zero_pending: false,
        })
    }

    /// Test: re-allocate the chunk scratch for `rows` rows per chunk (to cover both chunk sizes
    /// in one process, whatever `ARTIFICER_PREFILL_SSM` says).
    #[cfg(test)]
    pub(crate) fn set_chunk_rows_for_test(&mut self, cb: &CudaBackend, rows: usize) -> Result<()> {
        let conv_dim = self.xbc.ptr().len / self.chunk_rows;
        let two_heads = self.dta.ptr().len / self.chunk_rows;
        self.xbc = cb.alloc_f32(rows * conv_dim)?;
        self.dta = cb.alloc_f32(rows * two_heads)?;
        self.chunk_rows = rows;
        Ok(())
    }

    /// `SsmCache::clear()`: zero conv + SSM state (async) before the next device step.
    pub(crate) fn request_zero(&mut self) {
        self.zero_pending = true;
    }

    /// Test/debug: download conv + SSM state of `slot` (blocking).
    #[allow(dead_code)]
    pub(crate) fn read_state(&self, cb: &CudaBackend, slot: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let s = &self.slots[slot];
        let mut conv = vec![0.0f32; s.conv.ptr().len];
        let mut ssm = vec![0.0f32; s.ssm.ptr().len];
        cb.download_f32(s.conv.ptr(), &mut conv)?;
        cb.download_f32(s.ssm.ptr(), &mut ssm)?;
        Ok((conv, ssm))
    }
}

/// Shared prologue: validate shapes, create the device state on first use, apply a pending
/// `clear()`. `None` = shape not supported on the device.
#[allow(clippy::too_many_arguments)]
fn ssm_prepare(
    cb: &CudaBackend,
    cache: &mut SsmCache,
    scfg: SsmConfig,
    layer_idx: usize,
    zx: DevPtr,
    y: DevPtr,
    n_tok: usize,
) -> Result<Option<usize>> {
    let slot = cache
        .slot(layer_idx)
        .ok_or_else(|| CoreError::Other(format!("layer {layer_idx} is not SSM")))?;
    let d_inner = scfg.d_inner as usize;
    let n_head = scfg.n_head as usize;
    let n_group = scfg.n_group as usize;
    if n_group == 0
        || n_head == 0
        || scfg.d_conv == 0
        || !n_head.is_multiple_of(n_group)
        || !d_inner.is_multiple_of(n_group)
        || d_inner != n_head * scfg.head_dim() as usize
        || scfg.conv_dim() as usize != cache.conv_dim
        || scfg.d_state as usize != cache.d_state
    {
        return Ok(None);
    }
    need(zx, n_tok * scfg.d_in_proj() as usize, "ssm zx")?;
    need(y, n_tok * d_inner, "ssm y")?;
    if cache.device.is_none() {
        let st = DeviceSsmState::new(cb, cache)?;
        cache.device = Some(st);
    }
    let dev = cache.device.as_mut().expect("device SSM state");
    if dev.zero_pending {
        for s in &dev.slots {
            cb.memset0(s.conv.ptr())?;
            cb.memset0(s.ssm.ptr())?;
        }
        dev.zero_pending = false;
    }
    Ok(Some(slot))
}

/// Device Mamba-2 mix for `n_tok` rows. Same math as [`crate::ssm::mamba2_mix_host`] applied
/// row by row: reads `zx[r] = [z | xBC | dt]`, updates the device conv/SSM state of
/// `layer_idx`, writes `y[r][..d_inner]`.
///
/// 3 launches per [`SSM_CHUNK_ROWS`] rows, independent of `n_tok` (decode = 1 row): conv over
/// all rows + dt/dA, scan (walks the rows in-kernel, state in registers), `ho_ssm_gate_norm_rows`.
/// Chunks of ≥ [`SSM_TILED_MIN_ROWS`] rows use the tiled kernels (`ho_ssm_conv_tiled`: parallel
/// over positions × channels; `ho_ssm_scan_tiled`: B/C/x/dt staged per 16-token smem tile, one
/// reduce-scatter per tile), smaller ones `ho_ssm_conv_rows` / `ho_ssm_scan_rows`. All
/// bit-identical to [`ssm_mix_dev_rowloop`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssm_mix_dev(
    cb: &CudaBackend,
    cache: &mut SsmCache,
    scfg: SsmConfig,
    eps: f32,
    layer_idx: usize,
    zx: DevPtr,
    y: DevPtr,
    n_tok: usize,
) -> Result<bool> {
    ssm_mix_dev_with(cb, cache, scfg, eps, layer_idx, zx, y, n_tok, None)
}

/// [`ssm_mix_dev`] with the kernel family forced (`None` = automatic). Tests / microbench.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssm_mix_dev_with(
    cb: &CudaBackend,
    cache: &mut SsmCache,
    scfg: SsmConfig,
    eps: f32,
    layer_idx: usize,
    zx: DevPtr,
    y: DevPtr,
    n_tok: usize,
    force: Option<SsmKernels>,
) -> Result<bool> {
    if scfg.d_conv as usize > SSM_ROWS_MAX_D_CONV || scfg.d_state as usize > SSM_ROWS_MAX_D_STATE {
        return ssm_mix_dev_rowloop(cb, cache, scfg, eps, layer_idx, zx, y, n_tok);
    }
    let Some(slot) = ssm_prepare(cb, cache, scfg, layer_idx, zx, y, n_tok)? else {
        return Ok(false);
    };
    let dev = cache.device.as_ref().expect("device SSM state");
    let s = &dev.slots[slot];
    let d_inner = scfg.d_inner as usize;
    let conv_dim = scfg.conv_dim() as usize;
    let n_head = scfg.n_head as usize;
    let head_dim = scfg.head_dim() as usize;
    let d_state = scfg.d_state as usize;
    let n_group = scfg.n_group as usize;
    let d_conv = scfg.d_conv as usize;
    let d_in = scfg.d_in_proj() as usize;
    let rows = n_head * head_dim;
    let warps_per_block = (BLOCK / 32) as usize;
    let chunk_rows = dev.chunk_rows;
    let mut r0 = 0;
    while r0 < n_tok {
        let n = (n_tok - r0).min(chunk_rows);
        let zc = zx.offset(r0 * d_in, n * d_in);
        let yc = y.offset(r0 * d_inner, n * d_inner);
        let xbc = dev.xbc.ptr().prefix(n * conv_dim);
        let dta = dev.dta.ptr().prefix(n * 2 * n_head);
        let kind = force.unwrap_or_else(|| ssm_kernels_for(n));
        let args = LaunchArgs::new()
            .ptr(zc)
            .i32(d_in as i32)
            .i32(d_inner as i32)
            .i32((d_inner + conv_dim) as i32)
            .ptr(s.conv.ptr())
            .ptr(s.conv_w.ptr())
            .ptr(s.conv_b.ptr())
            .ptr(xbc)
            .ptr(s.a.ptr())
            .ptr(s.dt_b.ptr())
            .ptr(dta)
            .i32(conv_dim as i32)
            .i32(d_conv as i32)
            .i32(n_head as i32)
            .i32(n as i32);
        match kind {
            SsmKernels::Rows => {
                let threads = conv_dim + n * n_head;
                go(
                    cb,
                    "ho_ssm_conv_rows",
                    (threads.div_ceil(BLOCK as usize) as u32, 1, 1),
                    BLOCK,
                    args,
                )?;
            }
            SsmKernels::Tiled | SsmKernels::TiledFp32 => {
                let conv_blocks = conv_dim.div_ceil(BLOCK as usize);
                let dt_blocks = (SSM_CONV_TT * n_head).div_ceil(BLOCK as usize);
                go(
                    cb,
                    if kind == SsmKernels::TiledFp32 {
                        "ho_ssm_conv_tiled_f32"
                    } else {
                        "ho_ssm_conv_tiled"
                    },
                    (
                        (conv_blocks + dt_blocks) as u32,
                        n.div_ceil(SSM_CONV_TT) as u32,
                        1,
                    ),
                    BLOCK,
                    args.i32(conv_blocks as i32),
                )?;
            }
        }
        let args = LaunchArgs::new()
            .ptr(xbc)
            .ptr(dta)
            .ptr(s.d.ptr())
            .ptr(s.ssm.ptr())
            .ptr(yc)
            .i32(n_head as i32)
            .i32(head_dim as i32)
            .i32(d_state as i32)
            .i32(n_group as i32)
            .i32(n as i32);
        match kind {
            SsmKernels::Rows => go(
                cb,
                "ho_ssm_scan_rows",
                (rows.div_ceil(warps_per_block) as u32, 1, 1),
                BLOCK,
                args,
            )?,
            SsmKernels::TiledFp32 if ssm_scan_fast_ok(&scfg) && !test_old_scan() => go_sh(
                cb,
                "ho_ssm_scan_fast_ns4",
                (head_dim.div_ceil(SSM_SCAN2_CH) as u32, n_head as u32, 1),
                SSM_SCAN2_THREADS,
                SSM_SCAN2_SMEM,
                LaunchArgs::new()
                    .ptr(xbc)
                    .ptr(dta)
                    .ptr(s.d.ptr())
                    .ptr(s.ssm.ptr())
                    .ptr(yc)
                    .i32(n_head as i32)
                    .i32(head_dim as i32)
                    .i32(n_group as i32)
                    .i32(n as i32),
            )?,
            SsmKernels::Tiled | SsmKernels::TiledFp32 => go_sh(
                cb,
                if d_state <= 128 {
                    "ho_ssm_scan_tiled_ns4"
                } else {
                    "ho_ssm_scan_tiled"
                },
                (head_dim.div_ceil(SSM_SCAN_W) as u32, n_head as u32, 1),
                (32 * SSM_SCAN_W) as u32,
                ssm_scan_tiled_smem(d_state),
                args,
            )?,
        }
        let args = LaunchArgs::new()
            .ptr(yc)
            .ptr(zc)
            .i32(d_in as i32)
            .ptr(s.norm.ptr())
            .i32((d_inner / n_group) as i32)
            .i32(d_inner as i32)
            .f32(eps);
        go(
            cb,
            if kind == SsmKernels::TiledFp32 {
                "ho_ssm_gate_norm_rows_f32"
            } else {
                "ho_ssm_gate_norm_rows"
            },
            (n_group as u32, n as u32, 1),
            BLOCK,
            args,
        )?;
        r0 += n;
    }
    Ok(true)
}

/// Per-row reference path (3 launches per row; the original decode kernels). Used as the
/// fallback for shapes beyond the multi-row kernels' register windows and by the tests.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssm_mix_dev_rowloop(
    cb: &CudaBackend,
    cache: &mut SsmCache,
    scfg: SsmConfig,
    eps: f32,
    layer_idx: usize,
    zx: DevPtr,
    y: DevPtr,
    n_tok: usize,
) -> Result<bool> {
    let Some(slot) = ssm_prepare(cb, cache, scfg, layer_idx, zx, y, n_tok)? else {
        return Ok(false);
    };
    let dev = cache.device.as_ref().expect("device SSM state");
    let d_inner = scfg.d_inner as usize;
    let conv_dim = scfg.conv_dim() as usize;
    let n_head = scfg.n_head as usize;
    let head_dim = scfg.head_dim() as usize;
    let d_state = scfg.d_state as usize;
    let n_group = scfg.n_group as usize;
    let d_conv = scfg.d_conv as usize;
    let d_in = scfg.d_in_proj() as usize;
    let s = &dev.slots[slot];
    let xbc = dev.xbc.ptr().prefix(conv_dim);
    let rows = n_head * head_dim;
    let warps_per_block = (BLOCK / 32) as usize;
    for r in 0..n_tok {
        let zr = zx.offset(r * d_in, d_in);
        let z = zr.offset(0, d_inner);
        let xbc_in = zr.offset(d_inner, conv_dim);
        let dt = zr.offset(d_inner + conv_dim, n_head);
        let yr = y.offset(r * d_inner, d_inner);
        let args = LaunchArgs::new()
            .ptr(xbc_in)
            .ptr(s.conv.ptr())
            .ptr(s.conv_w.ptr())
            .ptr(s.conv_b.ptr())
            .ptr(xbc)
            .i32(conv_dim as i32)
            .i32(d_conv as i32);
        go(
            cb,
            "ho_ssm_conv",
            (conv_dim.div_ceil(BLOCK as usize) as u32, 1, 1),
            BLOCK,
            args,
        )?;
        let args = LaunchArgs::new()
            .ptr(xbc)
            .ptr(dt)
            .ptr(s.a.ptr())
            .ptr(s.d.ptr())
            .ptr(s.dt_b.ptr())
            .ptr(s.ssm.ptr())
            .ptr(yr)
            .i32(n_head as i32)
            .i32(head_dim as i32)
            .i32(d_state as i32)
            .i32(n_group as i32);
        go(
            cb,
            "ho_ssm_scan",
            (rows.div_ceil(warps_per_block) as u32, 1, 1),
            BLOCK,
            args,
        )?;
        let args = LaunchArgs::new()
            .ptr(yr)
            .ptr(z)
            .ptr(s.norm.ptr())
            .i32((d_inner / n_group) as i32)
            .f32(eps);
        go(cb, "ho_ssm_gate_norm", (n_group as u32, 1, 1), BLOCK, args)?;
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Tests: every hook, device vs host reference. Tolerances:
//   exact (bit-identical): add, bias, scale, relu², embed (all types);
//   |Δ| ≤ 1e-6·max(1,|x|): SiLU·/GELU· gate, softcap, RoPE (transcendentals in f64 → ≤ 1-2 ulp);
//   |Δ| ≤ 1e-5·max(1,|x|): norms (parallel vs sequential sum of squares);
//   SSM mix: |Δy| ≤ 2e-4·max(1,|y|) over 8 steps (FMA + reduction order; state compounds).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::config::{Architecture, ModelConfig};
    use crate::forward::ForwardScratch;
    use crate::kv::KvCache;

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
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 24) as u8
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.f() * scale).collect()
        }
    }

    fn up(cb: &CudaBackend, v: &[f32]) -> DeviceTensor {
        let t = cb.alloc_f32(v.len()).unwrap();
        cb.upload_f32(v, t.ptr()).unwrap();
        t
    }

    fn down(cb: &CudaBackend, p: DevPtr, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        cb.download_f32(p.prefix(n), &mut v).unwrap();
        v
    }

    /// Max |a-b| / max(1,|b|); NaNs must match.
    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                if x.is_nan() || y.is_nan() {
                    assert!(x.is_nan() && y.is_nan(), "NaN mismatch {x} vs {y}");
                    0.0
                } else {
                    (x - y).abs() / y.abs().max(1.0)
                }
            })
            .fold(0.0f32, f32::max)
    }

    fn assert_bits(a: &[f32], b: &[f32], what: &str) {
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "{what}[{i}]: dev {x} host {y}"
            );
        }
    }

    fn base_cfg() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            arch_str: "llama".into(),
            name: Some("hostops-test".into()),
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

    fn model_path(name: &str) -> Option<std::path::PathBuf> {
        crate::backend::test_model(name)
    }

    #[test]
    fn hostops_elementwise_match_host() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let ops = &OPS;
        let mut r = Lcg(11);
        for &n in &[1usize, 255, 3136, 12544] {
            let a = r.vec(n, 8.0);
            let b = r.vec(n, 8.0);
            // add
            let (da, db) = (up(&cb, &a), up(&cb, &b));
            assert!(ops
                .add_inplace(&cb, da.ptr().prefix(n), db.ptr().prefix(n))
                .unwrap());
            let mut h = a.clone();
            crate::ops::add_inplace(&mut h, &b);
            assert_bits(&down(&cb, da.ptr(), n), &h, "add");
            // scale
            let da = up(&cb, &a);
            assert!(ops.scale(&cb, da.ptr().prefix(n), 0.125f32.sqrt()).unwrap());
            let mut h = a.clone();
            crate::ops::scale_inplace(&mut h, 0.125f32.sqrt());
            assert_bits(&down(&cb, da.ptr(), n), &h, "scale");
            // softcap
            let big: Vec<f32> = a.iter().map(|v| v * 10.0).collect();
            let da = up(&cb, &big);
            assert!(ops.softcap(&cb, da.ptr().prefix(n), 30.0).unwrap());
            let mut h = big.clone();
            crate::ops::softcap_inplace(&mut h, 30.0);
            let e = max_rel(&down(&cb, da.ptr(), n), &h);
            assert!(e <= 1e-6, "softcap rel err {e}");
            // relu²
            let da = up(&cb, &a);
            assert!(ops
                .act(
                    &cb,
                    FfnActivation::ReluSqrSequential,
                    da.ptr().prefix(n),
                    None
                )
                .unwrap());
            let mut h = a.clone();
            crate::ops::relu_sqr_inplace(&mut h);
            assert_bits(&down(&cb, da.ptr(), n), &h, "relu2");
            // gated silu / gelu
            for act in [FfnActivation::SiluGated, FfnActivation::GeluGated] {
                let (da, db) = (up(&cb, &a), up(&cb, &b));
                assert!(ops
                    .act(&cb, act, da.ptr().prefix(n), Some(db.ptr().prefix(n)))
                    .unwrap());
                let mut h = a.clone();
                if act == FfnActivation::SiluGated {
                    crate::ops::silu_inplace(&mut h);
                } else {
                    crate::ops::gelu_inplace(&mut h);
                }
                crate::ops::mul_inplace(&mut h, &b);
                let e = max_rel(&down(&cb, da.ptr(), n), &h);
                assert!(e <= 1e-6, "{act:?} rel err {e}");
            }
            // mismatched activation shape → declined
            let da = up(&cb, &a);
            assert!(!ops
                .act(&cb, FfnActivation::SiluGated, da.ptr().prefix(n), None)
                .unwrap());
        }
        // RoPE: GQA-like heads, partial rotary, 2 rows with stride, large positions.
        for &(n_head, head_dim, n_rot, pos0) in &[
            (32usize, 64usize, 64usize, 0usize),
            (4, 128, 64, 1000),
            (8, 16, 16, 7),
        ] {
            let rows = 2;
            let stride = n_head * head_dim + 8;
            let x = r.vec(rows * stride, 4.0);
            let dx = up(&cb, &x);
            assert!(ops
                .rope(
                    &cb,
                    dx.ptr(),
                    rows,
                    stride,
                    n_head,
                    head_dim,
                    pos0,
                    10000.0,
                    n_rot
                )
                .unwrap());
            let got = down(&cb, dx.ptr(), rows * stride);
            let mut want = x.clone();
            for row in 0..rows {
                let s = &mut want[row * stride..row * stride + n_head * head_dim];
                crate::ops::apply_rope(s, n_head, head_dim, pos0 + row, 10000.0, n_rot);
            }
            let e = max_rel(&got, &want);
            eprintln!("rope h{n_head} d{head_dim} rot{n_rot} pos{pos0}: max rel err {e:.2e}");
            assert!(e <= 1e-6, "rope rel err {e}");
            // padding between rows untouched
            assert_bits(
                &got[n_head * head_dim..stride],
                &x[n_head * head_dim..stride],
                "rope pad",
            );
        }
        cb.sync().unwrap();
    }

    /// ho_embed on synthetic tables of every supported type vs `quant::dequantize` (bit-exact).
    #[test]
    fn hostops_embed_kernel_all_types_bit_exact() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let n_embd = 512usize;
        let n_rows = 5usize;
        let types = [
            (TensorType::F32, 4usize, 1usize),
            (TensorType::F16, 2, 1),
            (TensorType::BF16, 2, 1),
            (TensorType::Q8_0, 34, 32),
            (TensorType::Q5_0, 22, 32),
            (TensorType::Q4_K, 144, 256),
            (TensorType::Q6_K, 210, 256),
            (TensorType::Q4_0, 18, 32),
        ];
        let mut r = Lcg(5);
        // The backend's statics cache is keyed by host address: keep every table alive so a freed
        // buffer's address is never reused by the next type (would hit the stale entry).
        let mut keep: Vec<Vec<u8>> = Vec::new();
        for &(ty, bb, be) in &types {
            let bytes_n = n_rows * n_embd / be * bb;
            let mut bytes: Vec<u8> = (0..bytes_n.next_multiple_of(4)).map(|_| r.byte()).collect();
            // Keep f16 scales finite-ish: clear the top exponent bit of every f16 scale field.
            let blocks = n_rows * n_embd / be;
            for b in 0..blocks {
                let base = b * bb;
                let offs: &[usize] = match ty {
                    TensorType::Q8_0 | TensorType::Q5_0 | TensorType::Q4_0 => &[0],
                    TensorType::Q4_K => &[0, 2],
                    TensorType::Q6_K => &[208],
                    _ => &[],
                };
                for &o in offs {
                    bytes[base + o + 1] &= 0x3F;
                }
            }
            let (code, _) = embed_code(ty).unwrap();
            let table = cb
                .resident_f32_from(&bytes, TensorType::F32, bytes.len() / 4)
                .unwrap();
            let out = cb.alloc_f32(n_embd).unwrap();
            for tok in 0..n_rows {
                for do_scale in [0, 1] {
                    let scale = (n_embd as f32).sqrt();
                    let args = LaunchArgs::new()
                        .ptr(table)
                        .ptr(out.ptr())
                        .i32(tok as i32)
                        .i32(n_embd as i32)
                        .i32(code)
                        .f32(scale)
                        .i32(do_scale);
                    go(&cb, "ho_embed", grid_for(n_embd), BLOCK, args).unwrap();
                    let got = down(&cb, out.ptr(), n_embd);
                    let mut want = vec![0.0f32; n_embd];
                    crate::quant::dequantize(&bytes, ty, tok * n_embd, n_embd, &mut want).unwrap();
                    if do_scale == 1 {
                        crate::ops::scale_inplace(&mut want, scale);
                    }
                    assert_bits(
                        &got,
                        &want,
                        &format!("embed {ty:?} row {tok} scale {do_scale}"),
                    );
                }
            }
            keep.push(bytes);
        }
    }

    /// norm (all three kinds, 3 rows), bias and embed hooks on real tiny-vicuna tensors.
    #[test]
    fn hostops_norm_bias_embed_match_host_tiny_vicuna() {
        let Some(path) = model_path("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut model = LlamaModel::load(&path).unwrap();
        let dim = model.config.n_embd as usize;
        let w = model.layers[0].attn_norm.clone();
        let b = model.layers[0].ffn_norm.clone().expect("ffn_norm");
        let rows = 3;
        let mut r = Lcg(3);
        let x = r.vec(rows * dim, 6.0);
        let dx = up(&cb, &x);
        let dout = cb.alloc_f32(rows * dim).unwrap();
        let mut nw = vec![0.0f32; dim];
        let mut nb = vec![0.0f32; dim];
        for kind in [NormKind::Rms, NormKind::RmsGemma, NormKind::Layer] {
            model.config.norm_kind = kind;
            let bias = (kind == NormKind::Layer).then_some(&b);
            assert!(OPS
                .norm(&cb, &model, &w, bias, dx.ptr(), dout.ptr(), rows, dim)
                .unwrap());
            let got = down(&cb, dout.ptr(), rows * dim);
            let mut want = vec![0.0f32; rows * dim];
            for row in 0..rows {
                crate::forward::weights::apply_norm(
                    &model,
                    &model.config,
                    &w,
                    bias,
                    &x[row * dim..(row + 1) * dim],
                    &mut want[row * dim..(row + 1) * dim],
                    &mut nw,
                    &mut nb,
                )
                .unwrap();
            }
            let e = max_rel(&got, &want);
            eprintln!("norm {kind:?}: max rel err {e:.2e}");
            assert!(e <= 1e-5, "norm {kind:?} rel err {e}");
        }
        model.config.norm_kind = NormKind::Rms;
        // bias: per row, bit-exact (uses a norm weight as the "bias" tensor).
        let dx = up(&cb, &x);
        assert!(OPS.add_bias(&cb, &model, &w, dx.ptr(), rows, dim).unwrap());
        let got = down(&cb, dx.ptr(), rows * dim);
        let mut want = x.clone();
        for row in 0..rows {
            crate::forward::weights::add_bias(&model, &w, &mut want[row * dim..(row + 1) * dim])
                .unwrap();
        }
        assert_bits(&got, &want, "bias");
        // embed (Q4_K), with and without √d scaling, several tokens incl. last.
        let toks = [0u32, 1, 1724, 29973, model.config.n_vocab - 1];
        for scale in [false, true] {
            model.config.scale_embeddings = scale;
            let out = cb.alloc_f32(toks.len() * dim).unwrap();
            assert!(OPS.embed(&cb, &model, &toks, out.ptr()).unwrap());
            let got = down(&cb, out.ptr(), toks.len() * dim);
            let data = model.tensor_bytes(&model.token_embd).unwrap();
            for (i, &t) in toks.iter().enumerate() {
                let mut want = vec![0.0f32; dim];
                crate::quant::dequantize(
                    data,
                    model.token_embd.ty,
                    t as usize * dim,
                    dim,
                    &mut want,
                )
                .unwrap();
                if scale {
                    crate::ops::scale_inplace(&mut want, (dim as f32).sqrt());
                }
                assert_bits(&got[i * dim..(i + 1) * dim], &want, "embed");
            }
        }
    }

    fn ssm_model_cfg(scfg: SsmConfig) -> ModelConfig {
        let mut c = base_cfg();
        c.ssm = Some(scfg);
        c
    }

    /// Device Mamba-2 mix vs `mamba2_mix_host` over several decode steps (y + final state),
    /// small and Nemotron-sized shapes; then `clear()`; then a 3-row call vs 3 host steps.
    #[test]
    fn hostops_ssm_mix_matches_host_synthetic() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let shapes = [
            SsmConfig {
                d_conv: 4,
                d_inner: 64,
                d_state: 32,
                n_head: 8,
                n_group: 2,
            },
            SsmConfig {
                d_conv: 3,
                d_inner: 48,
                d_state: 16,
                n_head: 4,
                n_group: 1,
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
        for scfg in shapes {
            let cfg = ssm_model_cfg(scfg);
            let mut host = SsmCache::synthetic_for_test(&scfg, 2, 64, 99);
            let mut devc = SsmCache::synthetic_for_test(&scfg, 2, 64, 99);
            let d_in = scfg.d_in_proj() as usize;
            let d_inner = scfg.d_inner as usize;
            let conv_dim = scfg.conv_dim() as usize;
            let n_head = scfg.n_head as usize;
            let mut xbc = vec![0.0f32; conv_dim];
            let mut tmp = vec![0.0f32; n_head];
            let mut hy = vec![0.0f32; d_inner];
            let dzx = cb.alloc_f32(3 * d_in).unwrap();
            let dy = cb.alloc_f32(3 * d_inner).unwrap();
            let mut r = Lcg(scfg.d_inner + scfg.d_state);
            let mut worst = 0.0f32;
            let mut step = |host: &mut SsmCache,
                            devc: &mut SsmCache,
                            layer: usize,
                            n_tok: usize,
                            r: &mut Lcg| {
                let zx = r.vec(n_tok * d_in, 3.0);
                cb.upload_f32(&zx, dzx.ptr()).unwrap();
                assert!(ssm_mix_dev(
                    &cb,
                    devc,
                    scfg,
                    cfg.rms_eps,
                    layer,
                    dzx.ptr(),
                    dy.ptr(),
                    n_tok
                )
                .unwrap());
                let got = down(&cb, dy.ptr(), n_tok * d_inner);
                let mut e = 0.0f32;
                for t in 0..n_tok {
                    crate::ssm::mamba2_mix_host(
                        host,
                        layer,
                        &cfg,
                        &zx[t * d_in..(t + 1) * d_in],
                        &mut xbc,
                        &mut hy,
                        &mut tmp,
                    )
                    .unwrap();
                    e = e.max(max_rel(&got[t * d_inner..(t + 1) * d_inner], &hy));
                }
                e
            };
            for _ in 0..8 {
                for layer in 0..2 {
                    worst = worst.max(step(&mut host, &mut devc, layer, 1, &mut r));
                }
            }
            // states
            let st = devc.device.as_ref().unwrap();
            let mut st_err = 0.0f32;
            for slot in 0..2 {
                let (dc, ds) = st.read_state(&cb, slot).unwrap();
                let (hc, hs) = host.host_state(slot);
                assert_bits(&dc, hc, "conv state"); // conv state is a pure copy of inputs
                st_err = st_err.max(max_rel(&ds, hs));
            }
            // clear → both zero → continue
            host.clear();
            devc.clear();
            for _ in 0..2 {
                worst = worst.max(step(&mut host, &mut devc, 1, 1, &mut r));
            }
            // n_tok = 3 (sequential rows inside one call) vs 3 host steps
            worst = worst.max(step(&mut host, &mut devc, 0, 3, &mut r));
            eprintln!(
                "ssm mix d_inner {} d_state {} heads {} groups {} d_conv {}: max rel err y {worst:.2e}, ssm state {st_err:.2e}",
                scfg.d_inner, scfg.d_state, scfg.n_head, scfg.n_group, scfg.d_conv
            );
            assert!(worst <= 2e-4, "ssm y rel err {worst}");
            assert!(st_err <= 2e-4, "ssm state rel err {st_err}");
        }
    }

    const NEMOTRON_SSM: SsmConfig = SsmConfig {
        d_conv: 4,
        d_inner: 7680,
        d_state: 128,
        n_head: 96,
        n_group: 8,
    };

    /// Multi-row `ssm_mix` (3 launches), both kernel families — tiled (`ho_ssm_conv_tiled` +
    /// `ho_ssm_scan_tiled[_ns4]`) and naive rows (`ho_ssm_conv_rows` + `ho_ssm_scan_rows`) — vs
    /// looping the single-row device kernels vs host `mamba2_mix_host`, for n_tok ∈ {1, 2, 7, 16,
    /// 17, 29, 64, 163} (+ 300 > one chunk on Nemotron shapes), starting from a non-zero state;
    /// all y rows + final conv/SSM state; then clear() and again. Shapes cover head_dim % 8 != 0,
    /// d_state < 32, d_state 256 (the 8-slot scan), n_tok < d_conv - 1, ragged tiles.
    /// Tiled == naive rows == row loop bit-identical; vs host ≤ 2e-4 relative.
    #[test]
    fn hostops_ssm_mix_rows_match_rowloop_and_host() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let small = SsmConfig {
            d_conv: 3,
            d_inner: 48,
            d_state: 16,
            n_head: 4,
            n_group: 1,
        };
        let wide_state = SsmConfig {
            d_conv: 4,
            d_inner: 160,
            d_state: 256,
            n_head: 4,
            n_group: 2,
        };
        for (scfg, ntoks) in [
            (NEMOTRON_SSM, &[1usize, 2, 7, 16, 17, 29, 64, 163, 300][..]),
            (small, &[1usize, 2, 7, 16, 17, 29, 64][..]),
            (wide_state, &[1usize, 3, 16, 29, 40][..]),
        ] {
            let cfg = ssm_model_cfg(scfg);
            let d_in = scfg.d_in_proj() as usize;
            let d_inner = scfg.d_inner as usize;
            let conv_dim = scfg.conv_dim() as usize;
            let n_head = scfg.n_head as usize;
            let max_n = *ntoks.iter().max().unwrap();
            let dzx = cb.alloc_f32(max_n * d_in).unwrap();
            let dy_rows = cb.alloc_f32(max_n * d_inner).unwrap();
            let dy_naive = cb.alloc_f32(max_n * d_inner).unwrap();
            let dy_loop = cb.alloc_f32(max_n * d_inner).unwrap();
            let mut xbc = vec![0.0f32; conv_dim];
            let mut tmp = vec![0.0f32; n_head];
            let mut hy = vec![0.0f32; d_inner];
            for (pass, &n_tok) in ntoks.iter().chain([7usize].iter()).enumerate() {
                let after_clear = pass == ntoks.len();
                let mut host = SsmCache::synthetic_for_test(&scfg, 2, 64, 7);
                let mut rows = SsmCache::synthetic_for_test(&scfg, 2, 64, 7);
                let mut lp = SsmCache::synthetic_for_test(&scfg, 2, 64, 7);
                let mut nv = SsmCache::synthetic_for_test(&scfg, 2, 64, 7);
                let mut r = Lcg(1000 + n_tok as u32);
                let layer = 1;
                let mut run = |n: usize,
                               host: &mut SsmCache,
                               rows: &mut SsmCache,
                               lp: &mut SsmCache,
                               nv: &mut SsmCache,
                               r: &mut Lcg|
                 -> (f32, f32, u64, u64) {
                    let zx = r.vec(n * d_in, 3.0);
                    cb.upload_f32(&zx, dzx.ptr()).unwrap();
                    // Naive f64 CPU reference (ssm::mamba2_mix_reference_f64), from the host
                    // cache's state before this call.
                    let rf = crate::ssm::mamba2_mix_reference_f64(
                        host,
                        host.slot(layer).unwrap(),
                        &scfg,
                        cfg.rms_eps,
                        &zx,
                        n,
                    );
                    assert!(ssm_mix_dev_with(
                        &cb,
                        nv,
                        scfg,
                        cfg.rms_eps,
                        layer,
                        dzx.ptr(),
                        dy_naive.ptr(),
                        n,
                        Some(SsmKernels::Rows)
                    )
                    .unwrap());
                    let l0 = launch_count();
                    assert!(ssm_mix_dev_with(
                        &cb,
                        rows,
                        scfg,
                        cfg.rms_eps,
                        layer,
                        dzx.ptr(),
                        dy_rows.ptr(),
                        n,
                        Some(SsmKernels::Tiled)
                    )
                    .unwrap());
                    let l1 = launch_count();
                    assert!(ssm_mix_dev_rowloop(
                        &cb,
                        lp,
                        scfg,
                        cfg.rms_eps,
                        layer,
                        dzx.ptr(),
                        dy_loop.ptr(),
                        n
                    )
                    .unwrap());
                    let l2 = launch_count();
                    let yr = down(&cb, dy_rows.ptr(), n * d_inner);
                    let yl = down(&cb, dy_loop.ptr(), n * d_inner);
                    let yn = down(&cb, dy_naive.ptr(), n * d_inner);
                    assert_bits(&yr, &yl, &format!("y tiled vs rowloop n_tok {n}"));
                    assert_bits(&yn, &yl, &format!("y naive rows vs rowloop n_tok {n}"));
                    let e_ref = max_rel(&yr, &rf.y);
                    let mut e = 0.0f32;
                    for t in 0..n {
                        crate::ssm::mamba2_mix_host(
                            host,
                            layer,
                            &cfg,
                            &zx[t * d_in..(t + 1) * d_in],
                            &mut xbc,
                            &mut hy,
                            &mut tmp,
                        )
                        .unwrap();
                        e = e.max(max_rel(&yr[t * d_inner..(t + 1) * d_inner], &hy));
                    }
                    (e, e_ref, l1 - l0, l2 - l1)
                };
                // Non-zero starting state: 3 decode steps (or, after_clear: 3 steps then clear()).
                for _ in 0..3 {
                    run(1, &mut host, &mut rows, &mut lp, &mut nv, &mut r);
                }
                if after_clear {
                    host.clear();
                    rows.clear();
                    lp.clear();
                    nv.clear();
                }
                let (e_y, e_ref, l_rows, l_loop) =
                    run(n_tok, &mut host, &mut rows, &mut lp, &mut nv, &mut r);
                let slot = 1;
                let (rc, rs) = rows.device.as_ref().unwrap().read_state(&cb, slot).unwrap();
                let (lc, ls) = lp.device.as_ref().unwrap().read_state(&cb, slot).unwrap();
                let (nc, ns) = nv.device.as_ref().unwrap().read_state(&cb, slot).unwrap();
                assert_bits(&rc, &lc, "conv state tiled vs rowloop");
                assert_bits(&rs, &ls, "ssm state tiled vs rowloop");
                assert_bits(&nc, &lc, "conv state naive rows vs rowloop");
                assert_bits(&ns, &ls, "ssm state naive rows vs rowloop");
                let (hc, hs) = host.host_state(slot);
                assert_bits(&rc, hc, "conv state vs host");
                let e_s = max_rel(&rs, hs);
                eprintln!(
                    "ssm rows d_inner {} d_state {} d_conv {} n_tok {n_tok}{}: launches tiled {l_rows} / rowloop {l_loop}; y vs host {e_y:.2e}, y vs f64 ref {e_ref:.2e}, ssm state vs host {e_s:.2e}; tiled == naive rows == rowloop bit-exact",
                    scfg.d_inner,
                    scfg.d_state,
                    scfg.d_conv,
                    if after_clear { " (after clear)" } else { "" }
                );
                assert_eq!(l_rows as usize, 3 * n_tok.div_ceil(SSM_CHUNK_ROWS));
                assert_eq!(l_loop as usize, 3 * n_tok);
                assert!(e_y <= 2e-4, "y rel err {e_y}");
                assert!(e_ref <= 2e-4, "y vs f64 reference rel err {e_ref}");
                assert!(e_s <= 2e-4, "ssm state rel err {e_s}");
            }
        }
    }

    /// `ARTIFICER_PREFILL_SSM` parsing (no device): old / fp32 (trimmed, any case); unset /
    /// empty -> the default silently; unknown -> the default + warn flag. An explicit `old`
    /// always forces the old path.
    #[test]
    fn prefill_ssm_parse() {
        use PrefillSsm::*;
        assert_eq!(
            PREFILL_SSM_DEFAULT, Fp32,
            "default is fp32; explicit old forces the old path"
        );
        let d = PREFILL_SSM_DEFAULT;
        for (v, want) in [
            (None, (d, false)),
            (Some(""), (d, false)),
            (Some("  "), (d, false)),
            (Some("old"), (Old, false)),
            (Some(" OLD\n"), (Old, false)),
            (Some("fp32"), (Fp32, false)),
            (Some("FP32"), (Fp32, false)),
            (Some(" Fp32 "), (Fp32, false)),
            (Some("fp16"), (d, true)),
            (Some("f32"), (d, true)),
            (Some("1"), (d, true)),
            (Some("fp32,old"), (d, true)),
        ] {
            assert_eq!(parse_prefill_ssm(v), want, "{v:?}");
        }
    }

    /// The switch only touches tiled (batched-prefill) chunks: decode / tail chunks below the
    /// threshold and ARTIFICER_SSM_NAIVE stay on the old rows kernels in both modes (no device).
    #[test]
    fn prefill_ssm_kernel_pick() {
        use PrefillSsm::*;
        let m = SSM_TILED_MIN_ROWS;
        for mode in [Old, Fp32] {
            for n in [1, 2, m - 1] {
                assert_eq!(
                    ssm_kernels_pick(n, false, m, mode),
                    SsmKernels::Rows,
                    "{mode:?} {n}"
                );
            }
            for n in [m, 256, 1024] {
                assert_eq!(
                    ssm_kernels_pick(n, true, m, mode),
                    SsmKernels::Rows,
                    "naive {mode:?}"
                );
            }
        }
        for n in [m, m + 1, 256, 1024] {
            assert_eq!(ssm_kernels_pick(n, false, m, Old), SsmKernels::Tiled);
            assert_eq!(ssm_kernels_pick(n, false, m, Fp32), SsmKernels::TiledFp32);
        }
    }

    /// ARTIFICER_PREFILL_SSM on the multi-row device mix, from a non-zero state:
    /// * old (forced `Tiled`, and the default env-driven path when the process mode is old) is
    ///   bit-identical to the untouched per-row loop (`ho_ssm_conv` / `_scan` / `_gate_norm`)
    ///   and naive rows kernels, i.e. to the original output (y, conv + SSM state);
    /// * fp32 (`TiledFp32`) vs old: conv state bit-identical (pure copy), y and SSM state within
    ///   2e-5 relative (max(|old|, 1)-scaled; max abs diff printed), both vs the f64 reference
    ///   <= 2e-4, same launch count; a following decode step (rows kernels, both caches) stays
    ///   within the same bound.
    #[test]
    fn hostops_ssm_prefill_fp32_vs_old() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let small = SsmConfig {
            d_conv: 3,
            d_inner: 48,
            d_state: 16,
            n_head: 4,
            n_group: 1,
        };
        let wide_state = SsmConfig {
            d_conv: 4,
            d_inner: 160,
            d_state: 256,
            n_head: 4,
            n_group: 2,
        };
        let m = SSM_TILED_MIN_ROWS;
        let max_abs = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let default_is_old = prefill_ssm() == PrefillSsm::Old;
        let (mut all_abs, mut all_rel) = (0.0f32, 0.0f32);
        for (scfg, ntoks) in [
            (NEMOTRON_SSM, &[m, 16, 255, 256, 257, 300, 600][..]),
            (small, &[m, 29, 300][..]),
            (wide_state, &[m, 40, 257][..]),
        ] {
            let cfg = ssm_model_cfg(scfg);
            let d_in = scfg.d_in_proj() as usize;
            let d_inner = scfg.d_inner as usize;
            let conv_dim = scfg.conv_dim() as usize;
            let max_n = *ntoks.iter().max().unwrap();
            let dzx = cb.alloc_f32(max_n * d_in).unwrap();
            let dy_old = cb.alloc_f32(max_n * d_inner).unwrap();
            let dy_f32 = cb.alloc_f32(max_n * d_inner).unwrap();
            let dy_def = cb.alloc_f32(max_n * d_inner).unwrap();
            let dy_loop = cb.alloc_f32(max_n * d_inner).unwrap();
            let dy_naive = cb.alloc_f32(max_n * d_inner).unwrap();
            let mut xbc = vec![0.0f32; conv_dim];
            let mut hy = vec![0.0f32; d_inner];
            let mut tmp = vec![0.0f32; scfg.n_head as usize];
            for &n_tok in ntoks {
                let mk = || SsmCache::synthetic_for_test(&scfg, 2, 64, 11);
                let mut c_old = mk();
                let mut c_f32 = mk();
                let mut c_def = mk();
                let mut c_lp = mk();
                let mut c_nv = mk();
                let mut c_h = mk();
                let mut r = Lcg(5000 + n_tok as u32);
                let layer = 1;
                let slot = 1;
                let mix = |c: &mut SsmCache, y: DevPtr, n: usize, k: Option<SsmKernels>| {
                    let ok = ssm_mix_dev_with(&cb, c, scfg, cfg.rms_eps, layer, dzx.ptr(), y, n, k);
                    assert!(ok.unwrap());
                };
                // Non-zero starting state: 3 identical decode steps on every cache (+ host twin,
                // so the f64 reference starts from the same state).
                for _ in 0..3 {
                    let zx = r.vec(d_in, 3.0);
                    cb.upload_f32(&zx, dzx.ptr()).unwrap();
                    for c in [&mut c_old, &mut c_f32, &mut c_def, &mut c_lp, &mut c_nv] {
                        mix(c, dy_loop.ptr(), 1, None);
                    }
                    crate::ssm::mamba2_mix_host(
                        &mut c_h, layer, &cfg, &zx, &mut xbc, &mut hy, &mut tmp,
                    )
                    .unwrap();
                }
                let zx = r.vec(n_tok * d_in, 3.0);
                cb.upload_f32(&zx, dzx.ptr()).unwrap();
                let rf = crate::ssm::mamba2_mix_reference_f64(
                    &c_h,
                    c_h.slot(layer).unwrap(),
                    &scfg,
                    cfg.rms_eps,
                    &zx,
                    n_tok,
                );
                let l0 = launch_count();
                mix(&mut c_old, dy_old.ptr(), n_tok, Some(SsmKernels::Tiled));
                let l1 = launch_count();
                mix(&mut c_f32, dy_f32.ptr(), n_tok, Some(SsmKernels::TiledFp32));
                let l2 = launch_count();
                mix(&mut c_def, dy_def.ptr(), n_tok, None);
                mix(&mut c_nv, dy_naive.ptr(), n_tok, Some(SsmKernels::Rows));
                assert!(ssm_mix_dev_rowloop(
                    &cb,
                    &mut c_lp,
                    scfg,
                    cfg.rms_eps,
                    layer,
                    dzx.ptr(),
                    dy_loop.ptr(),
                    n_tok
                )
                .unwrap());
                let n = n_tok * d_inner;
                let yo = down(&cb, dy_old.ptr(), n);
                let yf = down(&cb, dy_f32.ptr(), n);
                let yd = down(&cb, dy_def.ptr(), n);
                let yl = down(&cb, dy_loop.ptr(), n);
                let yn = down(&cb, dy_naive.ptr(), n);
                let st = |c: &SsmCache| c.device.as_ref().unwrap().read_state(&cb, slot).unwrap();
                let (oc, os) = st(&c_old);
                let (fc, fs) = st(&c_f32);
                let (dc, ds) = st(&c_def);
                let (lc, ls) = st(&c_lp);
                let (nc, ns) = st(&c_nv);
                // old == the original kernels (rowloop + naive rows are untouched), bit for bit.
                assert_bits(&yo, &yl, &format!("y old tiled vs rowloop n_tok {n_tok}"));
                assert_bits(&yn, &yl, &format!("y naive rows vs rowloop n_tok {n_tok}"));
                assert_bits(&oc, &lc, "conv state old vs rowloop");
                assert_bits(&os, &ls, "ssm state old vs rowloop");
                assert_bits(&nc, &lc, "conv state naive rows vs rowloop");
                assert_bits(&ns, &ls, "ssm state naive rows vs rowloop");
                if default_is_old {
                    assert_bits(&yd, &yo, &format!("y default vs old n_tok {n_tok}"));
                    assert_bits(&dc, &oc, "conv state default vs old");
                    assert_bits(&ds, &os, "ssm state default vs old");
                }
                // fp32 vs old.
                assert_bits(&fc, &oc, "conv state fp32 vs old");
                assert_eq!(l2 - l1, l1 - l0, "launch count fp32 vs old");
                let (ya, yr) = (max_abs(&yf, &yo), max_rel(&yf, &yo));
                let (sa, sr) = (max_abs(&fs, &os), max_rel(&fs, &os));
                let (eo, ef) = (max_rel(&yo, &rf.y), max_rel(&yf, &rf.y));
                // One decode step after the prefill (rows kernels in both modes).
                let zx1 = r.vec(d_in, 3.0);
                cb.upload_f32(&zx1, dzx.ptr()).unwrap();
                mix(&mut c_old, dy_old.ptr(), 1, None);
                mix(&mut c_f32, dy_f32.ptr(), 1, None);
                let d_o = down(&cb, dy_old.ptr(), d_inner);
                let d_f = down(&cb, dy_f32.ptr(), d_inner);
                let (da, dr) = (max_abs(&d_f, &d_o), max_rel(&d_f, &d_o));
                eprintln!(
                    "ssm prefill fp32 vs old d_inner {} d_state {} n_tok {n_tok}: y max abs {ya:.3e} rel {yr:.3e}; ssm state max abs {sa:.3e} rel {sr:.3e}; next decode y max abs {da:.3e} rel {dr:.3e}; vs f64 ref old {eo:.2e} fp32 {ef:.2e}; old == rowloop bit-exact",
                    scfg.d_inner, scfg.d_state
                );
                all_abs = all_abs.max(ya).max(sa).max(da);
                all_rel = all_rel.max(yr).max(sr).max(dr);
                assert!(yr <= 2e-5, "y fp32 vs old rel {yr}");
                assert!(sr <= 2e-5, "ssm state fp32 vs old rel {sr}");
                assert!(dr <= 2e-5, "next decode y fp32 vs old rel {dr}");
                assert!(eo <= 2e-4 && ef <= 2e-4, "vs f64 ref: old {eo} fp32 {ef}");
            }
        }
        eprintln!("ssm prefill fp32 vs old overall: max abs {all_abs:.3e}, max rel {all_rel:.3e}");
    }

    /// `ho_ssm_scan_fast_ns4` (fp32 mode) vs the old `ho_ssm_scan_tiled_ns4` under the
    /// same fp32 conv / gate, bit for bit: reference = `TiledFp32` with the old scan and
    /// 256-row chunks (= the original fp32 mode, before the fast scan); vs fast scan with
    /// 1024-row chunks and with 256-row chunks. Non-zero start state (3 decode steps), one
    /// prefill of n_tok rows (chunk boundaries 256 / 1024, tile boundaries 16, Rows-kernel
    /// tails < 8 rows), a second prefill of 37 rows (state carried across calls), one decode
    /// step; y of every phase + final conv /
    /// SSM state. Shapes: Nemotron; d_state 128 with a ragged channel block (head_dim 20) and
    /// odd head_dim 13; small (d_state 16) and d_state 256, which use the old scan in both
    /// (`ssm_scan_fast_ok` false) and must match trivially.
    #[test]
    fn hostops_ssm_scan_fast_matches_old_scan() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let ragged = SsmConfig {
            d_conv: 4,
            d_inner: 120,
            d_state: 128,
            n_head: 6,
            n_group: 2,
        };
        let odd = SsmConfig {
            d_conv: 3,
            d_inner: 52,
            d_state: 128,
            n_head: 4,
            n_group: 1,
        };
        let small = SsmConfig {
            d_conv: 3,
            d_inner: 48,
            d_state: 16,
            n_head: 4,
            n_group: 1,
        };
        let wide_state = SsmConfig {
            d_conv: 4,
            d_inner: 160,
            d_state: 256,
            n_head: 4,
            n_group: 2,
        };
        let nemo = &[
            8usize, 15, 16, 17, 255, 256, 257, 300, 1023, 1024, 1025, 2050, 2100,
        ][..];
        for (scfg, ntoks, fast) in [
            (NEMOTRON_SSM, nemo, true),
            (ragged, &[8usize, 17, 300, 1030][..], true),
            (odd, &[9usize, 33, 1100][..], true),
            (small, &[8usize, 300][..], false),
            (wide_state, &[8usize, 257][..], false),
        ] {
            assert_eq!(ssm_scan_fast_ok(&scfg), fast);
            let cfg = ssm_model_cfg(scfg);
            let d_in = scfg.d_in_proj() as usize;
            let d_inner = scfg.d_inner as usize;
            let max_n = *ntoks.iter().max().unwrap();
            let dzx = cb.alloc_f32(max_n * d_in).unwrap();
            let dy = [
                cb.alloc_f32(max_n * d_inner).unwrap(),
                cb.alloc_f32(max_n * d_inner).unwrap(),
                cb.alloc_f32(max_n * d_inner).unwrap(),
            ];
            for &n_tok in ntoks {
                let mk = || SsmCache::synthetic_for_test(&scfg, 2, 64, 23);
                let mut caches = [mk(), mk(), mk()];
                let mut r = Lcg(9000 + n_tok as u32);
                let layer = 1;
                let slot = 1;
                let k = Some(SsmKernels::TiledFp32);
                // (cache, old scan?, chunk rows)
                let setup = [
                    (true, SSM_CHUNK_ROWS),
                    (false, SSM_CHUNK_ROWS_FAST),
                    (false, SSM_CHUNK_ROWS),
                ];
                let phase = |n: usize,
                             kind: Option<SsmKernels>,
                             caches: &mut [SsmCache; 3],
                             r: &mut Lcg| {
                    let zx = r.vec(n * d_in, 3.0);
                    cb.upload_f32(&zx, dzx.ptr()).unwrap();
                    let mut ys = Vec::new();
                    let mut launches = Vec::new();
                    for (i, c) in caches.iter_mut().enumerate() {
                        TEST_OLD_SCAN.with(|f| f.set(setup[i].0));
                        let l0 = launch_count();
                        let ok = ssm_mix_dev_with(
                            &cb,
                            c,
                            scfg,
                            cfg.rms_eps,
                            layer,
                            dzx.ptr(),
                            dy[i].ptr(),
                            n,
                            kind,
                        );
                        TEST_OLD_SCAN.with(|f| f.set(false));
                        assert!(ok.unwrap());
                        launches.push(launch_count() - l0);
                        ys.push(down(&cb, dy[i].ptr(), n * d_inner));
                    }
                    (ys, launches)
                };
                for _ in 0..3 {
                    phase(1, None, &mut caches, &mut r);
                }
                for (i, c) in caches.iter_mut().enumerate() {
                    c.device
                        .as_mut()
                        .unwrap()
                        .set_chunk_rows_for_test(&cb, setup[i].1)
                        .unwrap();
                }
                let (ys, l) = phase(n_tok, k, &mut caches, &mut r);
                for i in 0..3 {
                    assert_eq!(
                        l[i] as usize,
                        3 * n_tok.div_ceil(setup[i].1),
                        "launches cache {i}"
                    );
                }
                assert_bits(
                    &ys[1],
                    &ys[0],
                    &format!("prefill y fast/1024 vs old n_tok {n_tok}"),
                );
                assert_bits(
                    &ys[2],
                    &ys[0],
                    &format!("prefill y fast/256 vs old n_tok {n_tok}"),
                );
                let (ys, _) = phase(37, k, &mut caches, &mut r);
                assert_bits(&ys[1], &ys[0], "2nd prefill (37) y fast/1024 vs old");
                assert_bits(&ys[2], &ys[0], "2nd prefill (37) y fast/256 vs old");
                let (ys, _) = phase(1, None, &mut caches, &mut r);
                assert_bits(&ys[1], &ys[0], "decode y fast/1024 vs old");
                assert_bits(&ys[2], &ys[0], "decode y fast/256 vs old");
                let st: Vec<_> = caches
                    .iter()
                    .map(|c| c.device.as_ref().unwrap().read_state(&cb, slot).unwrap())
                    .collect();
                for i in 1..3 {
                    assert_bits(&st[i].0, &st[0].0, "conv state");
                    assert_bits(&st[i].1, &st[0].1, "ssm state");
                }
                eprintln!(
                    "ssm scan fast vs old d_inner {} d_state {} n_tok {n_tok}: fast kernel {fast}; y (prefill, +37, decode) + conv/ssm state bit-exact; launches old/256 {} fast/1024 {} fast/256 {}",
                    scfg.d_inner, scfg.d_state, l[0], l[1], l[2]
                );
            }
        }
    }

    /// Microbench (run with `--ignored`): one Nemotron-shaped layer, ssm_mix tiled vs naive
    /// multi-row vs per-row loop (loop only up to 64 rows), wall time per call between stream
    /// syncs (30 warm-up calls, then median of 7 × 10 calls). Also picks the tiled threshold.
    #[test]
    #[ignore]
    fn hostops_ssm_mix_rows_microbench() {
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let scfg = NEMOTRON_SSM;
        let d_in = scfg.d_in_proj() as usize;
        let d_inner = scfg.d_inner as usize;
        let mut cache = SsmCache::synthetic_for_test(&scfg, 1, 64, 3);
        let mut r = Lcg(9);
        for &n in &[1usize, 2, 4, 8, 16, 29, 64, 163, 256] {
            let dzx = up(&cb, &r.vec(n * d_in, 3.0));
            let dy = cb.alloc_f32(n * d_inner).unwrap();
            for name in ["rowloop", "rows", "tiled"] {
                if name == "rowloop" && n > 64 {
                    continue;
                }
                let call = |c: &mut SsmCache| match name {
                    "rowloop" => {
                        ssm_mix_dev_rowloop(&cb, c, scfg, 1e-5, 0, dzx.ptr(), dy.ptr(), n).unwrap();
                    }
                    "rows" => {
                        let k = Some(SsmKernels::Rows);
                        ssm_mix_dev_with(&cb, c, scfg, 1e-5, 0, dzx.ptr(), dy.ptr(), n, k).unwrap();
                    }
                    _ => {
                        let k = Some(SsmKernels::Tiled);
                        ssm_mix_dev_with(&cb, c, scfg, 1e-5, 0, dzx.ptr(), dy.ptr(), n, k).unwrap();
                    }
                };
                // Warm-up (also lets the GPU leave its idle clocks).
                for _ in 0..30 {
                    call(&mut cache);
                }
                cb.sync().unwrap();
                let mut ts = Vec::new();
                for _ in 0..7 {
                    let t0 = std::time::Instant::now();
                    for _ in 0..10 {
                        call(&mut cache);
                    }
                    cb.sync().unwrap();
                    ts.push(t0.elapsed().as_secs_f64() * 1e3 / 10.0);
                }
                ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!(
                    "microbench ssm_mix Nemotron layer n_tok {n:>3} {name:>7}: {:.3} ms/call (min {:.3}), {} launches",
                    ts[3],
                    ts[0],
                    if name == "rowloop" { 3 * n } else { 3 }
                );
            }
        }
    }

    /// Whole-model decode: device hooks vs the host fallback path (hooks disabled) vs CPU.
    /// Reports host-fallback round trips + stream syncs per step; with hooks on there must be none
    /// (except the one logits sync).
    fn e2e(name: &str, steps: usize, prompt: &[u32]) {
        let Some(path) = model_path(name) else {
            return;
        };
        let Some(cb) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        // Pin the matvec path: this test isolates hooks on vs off. The mmvq kernels
        // (int8 activations) shift logits ~0.2 vs the f32 matvec, which is
        // covered by the CPU-vs-CUDA greedy tests, not here.
        cb.set_mmvq(false);
        let mut model = LlamaModel::load(&path).unwrap();
        model.config.n_ctx = 64;
        let hybrid = model.config.architecture.is_hybrid();
        let cpu = crate::backend::CpuBackend;
        let mk_ssm = |m: &LlamaModel| m.config.ssm.map(|_| SsmCache::from_model(m).unwrap());
        let (mut kc, mut ko, mut kf) = (
            KvCache::new(&model.config),
            KvCache::new(&model.config),
            KvCache::new(&model.config),
        );
        let (mut sc, mut so, mut sf) = (mk_ssm(&model), mk_ssm(&model), mk_ssm(&model));
        let (mut hc, mut ho, mut hf) = (
            ForwardScratch::new(&model),
            ForwardScratch::new(&model),
            ForwardScratch::new(&model),
        );
        let am = |l: &[f32]| {
            l.iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                )
                .0 as u32
        };
        let maxd = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let use_cpu = !(hybrid && cfg!(debug_assertions));
        let mut next = 0u32;
        let (mut rt_on, mut rt_off, mut sy_on, mut sy_off) = (0u64, 0u64, 0u64, 0u64);
        let mut worst_onoff = 0.0f32;
        let mut cpu_match = 0usize;
        for step in 0..steps {
            let tok = if step < prompt.len() {
                prompt[step]
            } else {
                next
            };
            // The CPU path has debug-only length asserts that trip on hybrid oversized scratch
            // (pre-existing, release-only code path for Nemotron): CPU reference only in release.
            let lc = if use_cpu {
                model
                    .forward(tok, step, &mut kc, sc.as_mut(), &mut hc, true, &cpu)
                    .unwrap()
                    .unwrap()
                    .to_vec()
            } else {
                Vec::new()
            };
            let (r0, s0) = (cb.host_fallback_roundtrips(), cb.stream_sync_count());
            let lo = model
                .forward(tok, step, &mut ko, so.as_mut(), &mut ho, true, &cb)
                .unwrap()
                .unwrap()
                .to_vec();
            let (r1, s1) = (cb.host_fallback_roundtrips(), cb.stream_sync_count());
            set_thread_disabled(true);
            let lf = model
                .forward(tok, step, &mut kf, sf.as_mut(), &mut hf, true, &cb)
                .unwrap()
                .unwrap()
                .to_vec();
            set_thread_disabled(false);
            let (r2, s2) = (cb.host_fallback_roundtrips(), cb.stream_sync_count());
            if step > 0 {
                // step 0 includes one-time statics / table uploads
                rt_on += r1 - r0;
                sy_on += s1 - s0;
                rt_off += r2 - r1;
                sy_off += s2 - s1;
            }
            let d = maxd(&lo, &lf);
            worst_onoff = worst_onoff.max(d);
            let (tc, dc) = if use_cpu {
                (am(&lc), maxd(&lo, &lc))
            } else {
                (u32::MAX, f32::NAN)
            };
            eprintln!(
                "{name} step {step}: tok cpu {tc} on {} off {} | max|on-off| {d:.2e} max|on-cpu| {dc:.3} | host-fallback rt on {} off {}",
                am(&lo), am(&lf), r1 - r0, r2 - r1
            );
            assert_eq!(
                am(&lo),
                am(&lf),
                "{name} step {step}: hooks on vs off argmax"
            );
            if am(&lo) == tc {
                cpu_match += 1;
            }
            next = if use_cpu { tc } else { am(&lo) };
        }
        let n = (steps - 1) as f64;
        eprintln!(
            "{name}: per decode step: host-fallback round trips on {:.1} / off {:.1}; stream syncs on {:.1} / off {:.1}; max|on-off| logit {worst_onoff:.2e}; greedy on==cpu {}",
            rt_on as f64 / n,
            rt_off as f64 / n,
            sy_on as f64 / n,
            sy_off as f64 / n,
            if use_cpu { format!("{cpu_match}/{steps}") } else { "n/a (debug build)".into() }
        );
        assert_eq!(
            rt_on, 0,
            "{name}: hooks on must leave no host-fallback round trips"
        );
        assert!(
            sy_on as f64 / n <= 1.0 + 1e-9,
            "{name}: one sync per step with hooks on"
        );
        assert!(
            worst_onoff < 2e-2,
            "{name}: hooks on vs off logits differ by {worst_onoff}"
        );
        if hybrid {
            assert!(
                so.as_ref().unwrap().device.is_some(),
                "device SSM state created"
            );
            assert!(
                sf.as_ref().unwrap().device.is_none(),
                "host-fallback path keeps SSM on host"
            );
        }
        // clear() → device KV len 0, SSM zero pending; rerun step 0 matches the first pass.
        ko.clear();
        if let Some(s) = so.as_mut() {
            s.clear();
        }
        let l0 = model
            .forward(prompt[0], 0, &mut ko, so.as_mut(), &mut ho, true, &cb)
            .unwrap()
            .unwrap()
            .to_vec();
        kf.clear();
        if let Some(s) = sf.as_mut() {
            s.clear();
        }
        set_thread_disabled(true);
        let f0 = model
            .forward(prompt[0], 0, &mut kf, sf.as_mut(), &mut hf, true, &cb)
            .unwrap()
            .unwrap()
            .to_vec();
        set_thread_disabled(false);
        let d = maxd(&l0, &f0);
        assert!(d < 2e-2, "{name}: after clear(), on vs off differ by {d}");
    }

    #[test]
    fn hostops_e2e_tiny_vicuna_zero_host_fallback() {
        e2e(
            "tiny-vicuna-1b.q4_k_m.gguf",
            12,
            &[1, 1724, 2927, 338, 278, 14744, 29973],
        );
    }

    #[test]
    fn hostops_e2e_nemotron_zero_host_fallback() {
        // Any real token ids exercise all three layer types (SSM / attention / FFN).
        e2e(
            "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
            10,
            &[1, 1061, 4268, 2534, 1317, 4832, 1058],
        );
    }
}
