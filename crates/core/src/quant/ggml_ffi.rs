//! Optional vendor ggml CPU vec_dot FFI (MIT ggml via Ollama-shipped libs).
//!
//! Modes via `ARTIFICER_GGML_FFI`:
//! - `1` / `true` / `q5` — row-major Q5_0 via `ggml_vec_dot_q5_0_q8_0` (f16 act
//!   adapter; skips Artificer Q5_0×4).
//! - `qk` / `q4k` — Q4_K×Q8_K + Q6_K×Q8_K vendor dots; **keeps** Q5_0×4
//!   (composes with the Q5_0×4 interleaved cache).
//! - `q8` — Q8_0×Q8_0 LM-head via `ggml_vec_dot_q8_0_q8_0` (f16 act adapter);
//!   **keeps** Q5_0×4. Main CPU decode lever of the vendor paths.
//! - `all` — Q5 row + Q4/Q6 + Q8 vendor paths.
//!
//! Artificer `BlockQ80` is f32/36 B; ggml wire Q8_0 is f16/34 B (adapter below).
//! Artificer `BlockQ8K` is already ggml wire layout (f32 d, 292 B) — no adapter.
//!
//! License: ggml / llama.cpp MIT (see Ollama `LLAMA_CPP_LICENSE`). Runtime
//! link is dlopen-only — no compile-time dependency. Default path unchanged.
//! The dlopen loader is POSIX-only (`cfg(unix)`); on other platforms `try_load`
//! is a stub returning `None` and the vendor entry points report
//! "ggml FFI backend is not supported on this platform yet" as `CoreError::Backend`.
//!
//! Separate env `ARTIFICER_CPU_REPACK`:
//! - `q4k` / `1` / `true` — Q4_K×8 nibble-stripe + vendor `ggml_gemv_q4_K_8x8_q8_K`
//!   (compose with Q5_0×4). Default off.
//!
//! Library location (only read when one of the env vars above asks for the FFI):
//! `ARTIFICER_GGML_LIB_DIR` names a directory holding `libggml-base.so.0` and
//! `libggml-cpu-zen4.so` (e.g. an Ollama install's `lib/ollama`); it is searched first.
//! The system-wide `/usr/lib/ollama` is the only built-in fallback. If neither loads,
//! a one-line warning goes to stderr and the native kernels are kept.

#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::blocks::{BLOCK_Q4_K, BLOCK_Q5_0, BLOCK_Q6_K, BLOCK_Q8_0, QK5_0, QK8_0, QK_K};
use super::half::f32_to_f16;
use super::q50::BlockQ80;
use super::q8k::BlockQ8K;
use crate::error::{CoreError, Result};

/// ggml wire-format Q8_0 activation block (f16 scale).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BlockQ80Ggml {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ80Ggml>() == 34);
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);

type VecDotFn = unsafe extern "C" fn(
    n: i32,
    s: *mut f32,
    bs: usize,
    vx: *const std::ffi::c_void,
    bx: usize,
    vy: *const std::ffi::c_void,
    by: usize,
    nrc: i32,
);

#[cfg(unix)]
type BackendInitFn = unsafe extern "C" fn() -> *mut std::ffi::c_void;

#[derive(Clone, Copy, PartialEq, Eq)]
enum GgmlMode {
    Off,
    Q5,
    Qk,
    Q8,
    All,
}

type GemvFn = unsafe extern "C" fn(
    n: i32,
    s: *mut f32,
    bs: usize,
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    nr: i32,
    nc: i32,
);

// Never constructed off unix (the stub `try_load` returns None).
#[cfg_attr(not(unix), allow(dead_code))]
struct GgmlCpu {
    _base: *mut std::ffi::c_void,
    _cpu: *mut std::ffi::c_void,
    _backend: *mut std::ffi::c_void,
    vec_dot_q5: VecDotFn,
    vec_dot_q4k: VecDotFn,
    vec_dot_q6k: VecDotFn,
    vec_dot_q8: VecDotFn,
    /// Vendor CPU_REPACK gemv (Q4_K×8); None if symbol missing.
    gemv_q4k_8x8: Option<GemvFn>,
}

// SAFETY: ggml kernels are reentrant for distinct (vx,vy,s); we only call with
// thread-local / job-local buffers. Library handles stay process-lived.
unsafe impl Send for GgmlCpu {}
// SAFETY: as for Send above; shared callers pass their own (vx, vy, s) buffers.
unsafe impl Sync for GgmlCpu {}

static MODE: OnceLock<GgmlMode> = OnceLock::new();
static LIB: OnceLock<Option<GgmlCpu>> = OnceLock::new();
static LOAD_FAIL_LOGGED: AtomicBool = AtomicBool::new(false);

static REPACK_WANTED: OnceLock<bool> = OnceLock::new();
static REPACK_FAIL_LOGGED: AtomicBool = AtomicBool::new(false);

/// Message for the vendor ggml FFI on platforms without POSIX `dlopen`.
#[cfg(not(unix))]
const UNSUPPORTED_MSG: &str = "ggml FFI backend is not supported on this platform yet";

/// Error returned by the vendor entry points when the ggml libs are not loaded.
#[cfg(unix)]
fn not_loaded_error() -> CoreError {
    CoreError::Other("ggml FFI not loaded".into())
}

/// Error returned by the vendor entry points when the ggml libs are not loaded.
#[cfg(not(unix))]
fn not_loaded_error() -> CoreError {
    CoreError::Backend(UNSUPPORTED_MSG.into())
}

fn cpu_repack_wanted() -> bool {
    *REPACK_WANTED.get_or_init(|| match std::env::var("ARTIFICER_CPU_REPACK") {
        Ok(v) => {
            let t = v.trim();
            t == "1"
                || t.eq_ignore_ascii_case("true")
                || t.eq_ignore_ascii_case("q4k")
                || t.eq_ignore_ascii_case("q4")
        }
        Err(_) => false,
    })
}

/// True when Q4_K CPU_REPACK gemv is requested and zen4 gemv symbol loaded.
pub(crate) fn cpu_repack_q4k_active() -> bool {
    if !cpu_repack_wanted() {
        return false;
    }
    LIB.get_or_init(try_load)
        .as_ref()
        .map(|l| l.gemv_q4k_8x8.is_some())
        .unwrap_or(false)
}

fn parse_mode() -> GgmlMode {
    match std::env::var("ARTIFICER_GGML_FFI") {
        Ok(v) => {
            let t = v.trim();
            if t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("q5") {
                GgmlMode::Q5
            } else if t.eq_ignore_ascii_case("qk")
                || t.eq_ignore_ascii_case("q4k")
                || t.eq_ignore_ascii_case("q6k")
            {
                GgmlMode::Qk
            } else if t.eq_ignore_ascii_case("q8") || t.eq_ignore_ascii_case("q80") {
                GgmlMode::Q8
            } else if t.eq_ignore_ascii_case("all") {
                GgmlMode::All
            } else {
                GgmlMode::Off
            }
        }
        Err(_) => GgmlMode::Off,
    }
}

fn mode() -> GgmlMode {
    *MODE.get_or_init(parse_mode)
}

fn env_wants_any() -> bool {
    !matches!(mode(), GgmlMode::Off)
}

/// True when Q5_0 vendor row path is requested and libs loaded.
pub(crate) fn ggml_ffi_active() -> bool {
    matches!(mode(), GgmlMode::Q5 | GgmlMode::All) && LIB.get_or_init(try_load).is_some()
}

/// True when Q4_K/Q6_K vendor dots are requested and libs loaded.
/// Composes with Artificer Q5_0×4 (does not skip the ×4 cache).
pub(crate) fn ggml_ffi_qk_active() -> bool {
    matches!(mode(), GgmlMode::Qk | GgmlMode::All) && LIB.get_or_init(try_load).is_some()
}

/// True when Q8_0 LM-head vendor dots are requested and libs loaded.
/// Composes with Artificer Q5_0×4 (does not skip the ×4 cache).
pub(crate) fn ggml_ffi_q8_active() -> bool {
    matches!(mode(), GgmlMode::Q8 | GgmlMode::All) && LIB.get_or_init(try_load).is_some()
}

/// Env var naming an extra directory to search first for the vendor ggml CPU libs.
#[cfg(unix)]
const LIB_DIR_ENV: &str = "ARTIFICER_GGML_LIB_DIR";

/// System-wide fallback searched after `ARTIFICER_GGML_LIB_DIR` (Ollama's Linux install dir).
#[cfg(unix)]
const SYSTEM_LIB_DIR: &str = "/usr/lib/ollama";

/// Directories `try_load` searches, in order: `ARTIFICER_GGML_LIB_DIR` when set and
/// non-empty, then [`SYSTEM_LIB_DIR`]. There is no user-specific default.
#[cfg(unix)]
fn lib_dirs(env_dir: Option<std::ffi::OsString>) -> Vec<PathBuf> {
    let mut dirs = Vec::with_capacity(2);
    if let Some(d) = env_dir.filter(|d| !d.is_empty()) {
        dirs.push(PathBuf::from(d));
    }
    dirs.push(PathBuf::from(SYSTEM_LIB_DIR));
    dirs
}

#[cfg(unix)]
fn try_load() -> Option<GgmlCpu> {
    let dirs = lib_dirs(std::env::var_os(LIB_DIR_ENV));

    unsafe extern "C" {
        fn dlopen(filename: *const i8, flags: i32) -> *mut std::ffi::c_void;
        fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8) -> *mut std::ffi::c_void;
        fn dlerror() -> *mut i8;
    }
    // RTLD_NOW=2, RTLD_GLOBAL=0x100 on Linux x86_64 (glibc).
    const RTLD_NOW: i32 = 2;
    const RTLD_GLOBAL: i32 = 0x100;

    // SAFETY: dlopen/dlsym/dlerror get NUL-terminated CStrings that outlive each call; handles and
    // symbols are null-checked before use.
    unsafe {
        dlerror();
        for dir_path in &dirs {
            let dir = dir_path.display();
            let base_path = dir_path.join("libggml-base.so.0");
            let cpu_path = dir_path.join("libggml-cpu-zen4.so");
            let (Ok(base_c), Ok(cpu_c)) = (
                CString::new(base_path.as_os_str().as_bytes()),
                CString::new(cpu_path.as_os_str().as_bytes()),
            ) else {
                continue;
            };
            let base = dlopen(base_c.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
            if base.is_null() {
                continue;
            }
            let cpu = dlopen(cpu_c.as_ptr(), RTLD_NOW);
            if cpu.is_null() {
                continue;
            }
            let init_name = CString::new("ggml_backend_cpu_init").ok()?;
            let init_ptr = dlsym(cpu, init_name.as_ptr());
            if init_ptr.is_null() {
                continue;
            }
            let init: BackendInitFn = std::mem::transmute(init_ptr);
            let backend = init();
            if backend.is_null() {
                continue;
            }
            let q5_name = CString::new("ggml_vec_dot_q5_0_q8_0").ok()?;
            let q4_name = CString::new("ggml_vec_dot_q4_K_q8_K").ok()?;
            let q6_name = CString::new("ggml_vec_dot_q6_K_q8_K").ok()?;
            let q8_name = CString::new("ggml_vec_dot_q8_0_q8_0").ok()?;
            let q5_ptr = dlsym(cpu, q5_name.as_ptr());
            let q4_ptr = dlsym(cpu, q4_name.as_ptr());
            let q6_ptr = dlsym(cpu, q6_name.as_ptr());
            let q8_ptr = dlsym(cpu, q8_name.as_ptr());
            if q5_ptr.is_null() || q4_ptr.is_null() || q6_ptr.is_null() || q8_ptr.is_null() {
                continue;
            }
            let gemv_name = CString::new("ggml_gemv_q4_K_8x8_q8_K").ok()?;
            let gemv_ptr = dlsym(cpu, gemv_name.as_ptr());
            let gemv_q4k_8x8: Option<GemvFn> = if gemv_ptr.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut std::ffi::c_void, GemvFn>(
                    gemv_ptr,
                ))
            };
            let vec_dot_q5: VecDotFn = std::mem::transmute(q5_ptr);
            let vec_dot_q4k: VecDotFn = std::mem::transmute(q4_ptr);
            let vec_dot_q6k: VecDotFn = std::mem::transmute(q6_ptr);
            let vec_dot_q8: VecDotFn = std::mem::transmute(q8_ptr);
            let mode_s = match mode() {
                GgmlMode::Q5 => "Q5_0",
                GgmlMode::Qk => "Q4_K+Q6_K",
                GgmlMode::Q8 => "Q8_0",
                GgmlMode::All => "Q5_0+Q4_K+Q6_K+Q8_0",
                GgmlMode::Off => "off",
            };
            if env_wants_any() {
                eprintln!(
                    "[artificer] ARTIFICER_GGML_FFI: loaded ggml CPU {mode_s} vec_dot from {dir} (MIT ggml)"
                );
            }
            if cpu_repack_wanted() {
                if gemv_q4k_8x8.is_some() {
                    eprintln!(
                        "[artificer] ARTIFICER_CPU_REPACK: loaded ggml_gemv_q4_K_8x8_q8_K from {dir} (MIT ggml)"
                    );
                } else {
                    eprintln!(
                        "[artificer] ARTIFICER_CPU_REPACK set but ggml_gemv_q4_K_8x8_q8_K missing — keeping native Q4_K"
                    );
                }
            }
            return Some(GgmlCpu {
                _base: base,
                _cpu: cpu,
                _backend: backend,
                vec_dot_q5,
                vec_dot_q4k,
                vec_dot_q6k,
                vec_dot_q8,
                gemv_q4k_8x8,
            });
        }
        let searched = dirs
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if env_wants_any() && !LOAD_FAIL_LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[artificer] ARTIFICER_GGML_FFI set but failed to dlopen libggml-cpu-zen4.so from {searched} \
                 (set {LIB_DIR_ENV} to the directory with libggml-base.so.0 + libggml-cpu-zen4.so) — keeping native kernels"
            );
        }
        if cpu_repack_wanted() && !REPACK_FAIL_LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[artificer] ARTIFICER_CPU_REPACK set but failed to dlopen libggml-cpu-zen4.so from {searched} \
                 (set {LIB_DIR_ENV}) — keeping native Q4_K"
            );
        }
        None
    }
}

/// Non-unix stub: no POSIX `dlopen`, so the vendor libs are never loaded.
/// Logs once (only when an env var asked for the FFI) and keeps native kernels.
#[cfg(not(unix))]
fn try_load() -> Option<GgmlCpu> {
    if env_wants_any() && !LOAD_FAIL_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[artificer] ARTIFICER_GGML_FFI set but {UNSUPPORTED_MSG} — keeping native kernels"
        );
    }
    if cpu_repack_wanted() && !REPACK_FAIL_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[artificer] ARTIFICER_CPU_REPACK set but {UNSUPPORTED_MSG} — keeping native Q4_K"
        );
    }
    None
}

/// Convert Artificer f32-scale act blocks into ggml f16 wire blocks.
#[inline]
pub(crate) fn adapt_q80_to_ggml(src: &[BlockQ80], dst: &mut [BlockQ80Ggml]) {
    debug_assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        d.d = f32_to_f16(s.d);
        d.qs = s.qs;
    }
}

/// Row-major Q5_0 × ggml-Q8_0 via vendor kernel. `y` must already be f16-adapted.
pub(crate) fn vec_dot_q5_0_q8_0_ggml(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ80Ggml],
) -> Result<f32> {
    if !k.is_multiple_of(QK5_0) {
        return Err(CoreError::Shape("ggml Q5: k%32!=0".into()));
    }
    let nb = k / QK5_0;
    let start = elem_offset / QK5_0;
    if y.len() < nb {
        return Err(CoreError::Shape("ggml Q5: y short".into()));
    }
    let need = (start + nb) * BLOCK_Q5_0;
    if need > weight.len() {
        return Err(CoreError::Shape("ggml Q5: weight OOB".into()));
    }
    let Some(lib) = LIB.get_or_init(try_load).as_ref() else {
        return Err(not_loaded_error());
    };
    let vx = weight[start * BLOCK_Q5_0..need].as_ptr() as *const std::ffi::c_void;
    let vy = y[..nb].as_ptr() as *const std::ffi::c_void;
    let mut s = 0.0f32;
    // SAFETY: vx/vy point at valid packed blocks; nrc=1; backend inited at load.
    unsafe {
        (lib.vec_dot_q5)(k as i32, &mut s, 0, vx, 0, vy, 0, 1);
    }
    Ok(s)
}

/// Multi-row helper: one ggml call per output row (ggml q5_0 nrc==1).
pub(crate) fn vec_dot_q5_0_q8_0_ggml_nrows(
    weight: &[u8],
    elem_offset0: usize,
    k: usize,
    y: &[BlockQ80Ggml],
    out: &mut [f32],
) -> Result<()> {
    for (i, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q5_0_q8_0_ggml(weight, elem_offset0 + i * k, k, y)?;
    }
    Ok(())
}

/// Q4_K × Q8_K via vendor zen4 kernel. `y` is Artificer `BlockQ8K` (== ggml wire).
pub(crate) fn vec_dot_q4_k_q8_k_ggml(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    if !k.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("ggml Q4_K: k%256!=0".into()));
    }
    let nb = k / QK_K;
    let start = elem_offset / QK_K;
    if y.len() < nb {
        return Err(CoreError::Shape("ggml Q4_K: y short".into()));
    }
    let need = (start + nb) * BLOCK_Q4_K;
    if need > weight.len() {
        return Err(CoreError::Shape("ggml Q4_K: weight OOB".into()));
    }
    let Some(lib) = LIB.get_or_init(try_load).as_ref() else {
        return Err(not_loaded_error());
    };
    let vx = weight[start * BLOCK_Q4_K..need].as_ptr() as *const std::ffi::c_void;
    let vy = y[..nb].as_ptr() as *const std::ffi::c_void;
    let mut s = 0.0f32;
    // SAFETY: the symbol has ggml's vec_dot signature; `vx` and `vy` cover the row and activation
    // blocks (bounds checked above) and `s` is a live local.
    unsafe {
        (lib.vec_dot_q4k)(k as i32, &mut s, 0, vx, 0, vy, 0, 1);
    }
    Ok(s)
}

/// Q6_K × Q8_K via vendor zen4 kernel.
pub(crate) fn vec_dot_q6_k_q8_k_ggml(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ8K],
) -> Result<f32> {
    if !k.is_multiple_of(QK_K) {
        return Err(CoreError::Shape("ggml Q6_K: k%256!=0".into()));
    }
    let nb = k / QK_K;
    let start = elem_offset / QK_K;
    if y.len() < nb {
        return Err(CoreError::Shape("ggml Q6_K: y short".into()));
    }
    let need = (start + nb) * BLOCK_Q6_K;
    if need > weight.len() {
        return Err(CoreError::Shape("ggml Q6_K: weight OOB".into()));
    }
    let Some(lib) = LIB.get_or_init(try_load).as_ref() else {
        return Err(not_loaded_error());
    };
    let vx = weight[start * BLOCK_Q6_K..need].as_ptr() as *const std::ffi::c_void;
    let vy = y[..nb].as_ptr() as *const std::ffi::c_void;
    let mut s = 0.0f32;
    // SAFETY: the symbol has ggml's vec_dot signature; `vx` and `vy` cover the row and activation
    // blocks (bounds checked above) and `s` is a live local.
    unsafe {
        (lib.vec_dot_q6k)(k as i32, &mut s, 0, vx, 0, vy, 0, 1);
    }
    Ok(s)
}

/// Q8_0 × Q8_0 via vendor zen4 kernel. `y` must already be f16-adapted.
/// Weight blocks are GGUF/ggml wire (f16 `d`, 34 B) — no weight adapter.
pub(crate) fn vec_dot_q8_0_q8_0_ggml(
    weight: &[u8],
    elem_offset: usize,
    k: usize,
    y: &[BlockQ80Ggml],
) -> Result<f32> {
    if !k.is_multiple_of(QK8_0) {
        return Err(CoreError::Shape("ggml Q8_0: k%32!=0".into()));
    }
    let nb = k / QK8_0;
    let start = elem_offset / QK8_0;
    if y.len() < nb {
        return Err(CoreError::Shape("ggml Q8_0: y short".into()));
    }
    let need = (start + nb) * BLOCK_Q8_0;
    if need > weight.len() {
        return Err(CoreError::Shape("ggml Q8_0: weight OOB".into()));
    }
    let Some(lib) = LIB.get_or_init(try_load).as_ref() else {
        return Err(not_loaded_error());
    };
    let vx = weight[start * BLOCK_Q8_0..need].as_ptr() as *const std::ffi::c_void;
    let vy = y[..nb].as_ptr() as *const std::ffi::c_void;
    let mut s = 0.0f32;
    // SAFETY: the symbol has ggml's vec_dot signature; `vx` and `vy` cover the row and activation
    // blocks (bounds checked above) and `s` is a live local.
    unsafe {
        (lib.vec_dot_q8)(k as i32, &mut s, 0, vx, 0, vy, 0, 1);
    }
    Ok(s)
}

/// Multi-row helper: one ggml call per output row (ggml q8_0 nrc==1).
pub(crate) fn vec_dot_q8_0_q8_0_ggml_nrows(
    weight: &[u8],
    elem_offset0: usize,
    k: usize,
    y: &[BlockQ80Ggml],
    out: &mut [f32],
) -> Result<()> {
    for (i, o) in out.iter_mut().enumerate() {
        *o = vec_dot_q8_0_q8_0_ggml(weight, elem_offset0 + i * k, k, y)?;
    }
    Ok(())
}

/// Vendor CPU_REPACK GEMV for Q4_K×8 packed weights × Q8_K acts.
/// `row_base` and `out.len()` must be multiples of 8.
pub(crate) fn gemv_q4_k_8x8_q8_k_ggml(
    packed: &[u8],
    row_base: usize,
    k: usize,
    y: &[BlockQ8K],
    out: &mut [f32],
) -> Result<()> {
    let n_rows = out.len();
    if n_rows == 0 {
        return Ok(());
    }
    if !k.is_multiple_of(QK_K) || !row_base.is_multiple_of(8) || !n_rows.is_multiple_of(8) {
        return Err(CoreError::Shape("ggml gemv Q4_Kx8 align".into()));
    }
    let nb = k / QK_K;
    if y.len() < nb {
        return Err(CoreError::Shape("ggml gemv Q4_K: y short".into()));
    }
    let g0 = row_base / 8;
    let n_groups = n_rows / 8;
    let need = (g0 + n_groups) * nb * 1152; // Q4_KX8_BYTES
    if need > packed.len() {
        return Err(CoreError::Shape("ggml gemv Q4_K: packed OOB".into()));
    }
    let Some(lib) = LIB.get_or_init(try_load).as_ref() else {
        return Err(not_loaded_error());
    };
    let Some(gemv) = lib.gemv_q4k_8x8 else {
        return Err(CoreError::Other("ggml gemv Q4_K_8x8 missing".into()));
    };
    let vx = packed[g0 * nb * 1152..need].as_ptr() as *const std::ffi::c_void;
    let vy = y[..nb].as_ptr() as *const std::ffi::c_void;
    // SAFETY: packed/y valid; nc multiple of 8; backend inited at load.
    unsafe {
        gemv(k as i32, out.as_mut_ptr(), 0usize, vx, vy, 1, n_rows as i32);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;
    use crate::quant::q50::{quantize_row_q8_0, vec_dot_q5_0_q8_0, vec_dot_q8_0_q8_0, BlockQ80};
    use crate::quant::q6k::vec_dot_q6_k_q8_k;
    use crate::quant::q8k::{quantize_row_q8_k, vec_dot_q4_k_q8_k, BlockQ8K};

    /// `ARTIFICER_GGML_LIB_DIR` goes first; the only built-in dir is system-wide.
    #[cfg(unix)]
    #[test]
    fn lib_dirs_env_first_then_system_only() {
        let sys = vec![PathBuf::from(SYSTEM_LIB_DIR)];
        assert_eq!(lib_dirs(None), sys);
        assert_eq!(lib_dirs(Some("".into())), sys);
        assert_eq!(
            lib_dirs(Some("/opt/ggml/lib".into())),
            vec![
                PathBuf::from("/opt/ggml/lib"),
                PathBuf::from(SYSTEM_LIB_DIR)
            ]
        );
        assert!(lib_dirs(None).iter().all(|d| !d.starts_with("/home")));
    }

    #[test]
    fn ggml_block_size_is_34() {
        assert_eq!(std::mem::size_of::<BlockQ80Ggml>(), 34);
    }

    #[test]
    fn adapter_preserves_qs_and_f16_scale() {
        let mut y = [BlockQ80 {
            d: 0.5,
            qs: [0; 32],
        }];
        y[0].qs[0] = 7;
        y[0].qs[31] = -3;
        let mut g = [BlockQ80Ggml { d: 0, qs: [0; 32] }];
        adapt_q80_to_ggml(&y, &mut g);
        assert_eq!(g[0].d, f32_to_f16(0.5));
        assert_eq!(g[0].qs[0], 7);
        assert_eq!(g[0].qs[31], -3);
    }

    /// Parity vs native — only runs when ARTIFICER_GGML_FFI requests a path and libs present.
    #[test]
    fn ggml_ffi_parity_when_enabled() {
        if !env_wants_any() {
            return;
        }
        if LIB.get_or_init(try_load).is_none() {
            eprintln!("ggml FFI requested but libs missing — skip parity");
            return;
        }

        // Q5_0 tiny row (always exercised when libs load).
        {
            let mut weight = vec![0u8; BLOCK_Q5_0];
            weight[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            for b in weight[6..22].iter_mut() {
                *b = 0x88;
            }
            let x: Vec<f32> = (0..32).map(|_| 1.0f32).collect();
            let mut y_art = [BlockQ80 {
                d: 0.0,
                qs: [0; 32],
            }];
            quantize_row_q8_0(&x, &mut y_art);
            let mut y_g = [BlockQ80Ggml { d: 0, qs: [0; 32] }];
            adapt_q80_to_ggml(&y_art, &mut y_g);
            let native = vec_dot_q5_0_q8_0(&weight, 0, 32, &y_art).unwrap();
            let ggml = vec_dot_q5_0_q8_0_ggml(&weight, 0, 32, &y_g).unwrap();
            let rel = ((native - ggml).abs() / native.abs().max(1e-3)).abs();
            assert!(
                rel < 1e-3,
                "Q5 parity fail native={native} ggml={ggml} rel={rel}"
            );
        }

        // Q4_K: one super-block, scales=1 mins=0, qs ascending nibbles.
        if matches!(mode(), GgmlMode::Qk | GgmlMode::All | GgmlMode::Q5) {
            let mut weight = vec![0u8; BLOCK_Q4_K];
            weight[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
            weight[2..4].copy_from_slice(&f32_to_f16(0.0).to_le_bytes()); // min
                                                                          // 12 scale bytes: low 6 bits = 1 for first groups
            for i in 0..4 {
                weight[4 + i] = 1;
                weight[8 + i] = 0; // mins
            }
            weight[12] = 0;
            weight[13] = 0;
            weight[14] = 0;
            weight[15] = 0;
            for (i, b) in weight[16..144].iter_mut().enumerate() {
                *b = ((i & 0x0f) | (((i.wrapping_add(1)) & 0x0f) << 4)) as u8;
            }
            let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
            let mut y = vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; QK_K],
                    bsums: [0; QK_K / 16],
                };
                1
            ];
            quantize_row_q8_k(&x, &mut y);
            let native = vec_dot_q4_k_q8_k(&weight, 0, 256, &y).unwrap();
            let ggml = vec_dot_q4_k_q8_k_ggml(&weight, 0, 256, &y).unwrap();
            let rel = ((native - ggml).abs() / native.abs().max(1e-3)).abs();
            assert!(
                rel < 2e-3,
                "Q4_K parity fail native={native} ggml={ggml} rel={rel}"
            );
        }

        // Q6_K: zero-ish block still must agree (both ~0).
        if matches!(mode(), GgmlMode::Qk | GgmlMode::All | GgmlMode::Q5) {
            let weight = vec![0u8; BLOCK_Q6_K];
            let x = vec![0.0f32; 256];
            let mut y = vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; QK_K],
                    bsums: [0; QK_K / 16],
                };
                1
            ];
            quantize_row_q8_k(&x, &mut y);
            let native = vec_dot_q6_k_q8_k(&weight, 0, 256, &y).unwrap();
            let ggml = vec_dot_q6_k_q8_k_ggml(&weight, 0, 256, &y).unwrap();
            assert!(
                (native - ggml).abs() < 1e-4,
                "Q6_K zero parity fail native={native} ggml={ggml}"
            );
        }

        // Q8_0: two blocks (k=64), identity-ish.
        if matches!(
            mode(),
            GgmlMode::Q8 | GgmlMode::All | GgmlMode::Q5 | GgmlMode::Qk
        ) {
            let mut weight = vec![0u8; BLOCK_Q8_0 * 2];
            for bi in 0..2 {
                let base = bi * BLOCK_Q8_0;
                weight[base..base + 2].copy_from_slice(&f32_to_f16(0.1).to_le_bytes());
                for j in 0..32 {
                    weight[base + 2 + j] = (j as i8).wrapping_sub(16) as u8;
                }
            }
            let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
            let mut y_art = [BlockQ80 {
                d: 0.0,
                qs: [0; 32],
            }; 2];
            quantize_row_q8_0(&x, &mut y_art);
            let mut y_g = [BlockQ80Ggml { d: 0, qs: [0; 32] }; 2];
            adapt_q80_to_ggml(&y_art, &mut y_g);
            let native = vec_dot_q8_0_q8_0(&weight, 0, 64, &y_art).unwrap();
            let ggml = vec_dot_q8_0_q8_0_ggml(&weight, 0, 64, &y_g).unwrap();
            let rel = ((native - ggml).abs() / native.abs().max(1e-3)).abs();
            assert!(
                rel < 2e-3,
                "Q8_0 parity fail native={native} ggml={ggml} rel={rel}"
            );
        }
    }
}
