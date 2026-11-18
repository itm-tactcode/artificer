//! Optional CUDA backend — real device GEMV/GEMM via runtime `dlopen`.
//!
//! Gated by Cargo feature `cuda`. **No compile-time link** to the CUDA toolkit:
//! default (feature-off) builds never need NVIDIA headers or libs. With the
//! feature on, [`CudaBackend::try_new`] loads `libcudart` + `libcublas` at
//! runtime, probes a device, and fails clearly when the library or a device
//! is missing.
//!
//! ## Kernels
//! - **GEMV / GEMM**: NVIDIA **cuBLAS** (`cublasSgemv_v2` / `cublasSgemm_v2`),
//!   not custom PTX. Matches the existing `dlopen` seam (no `cudarc`).
//!
//! ## Quantization / memory (2026-10-16)
//! Quantized / f16 GGUF weights are uploaded **as raw blocks** and dequantized
//! inside NVRTC-compiled matvec kernels ([`super::cuda_quant`]), so VRAM ≈ GGUF
//! size. Only **F32** tensors are kept as f32 and use cuBLAS Sgemv. Resident
//! buffers are keyed by host mapping pointer + type + shape; free memory is
//! checked ([`CudaBackend::check_fits`]) before every upload.
//!
//! ## Fallback policy
//! **No silent CPU matmul disguised as device work.** If cuBLAS / device
//! buffers / memcpy fail, [`Backend::matmul_vec`] returns [`CoreError::Backend`].
//!
//! Toolkit exercised on the development machine: CUDA **13.4** (`/opt/cuda`), driver
//! runtime via `libcudart.so.13`.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use gguf::TensorType;

use super::cuda_quant::{device_quant_layout, QuantKernels};
use super::{Backend, BackendKind};
use crate::error::{CoreError, Result};
use crate::quant::dequantize;

/// Headroom kept free on the device beyond weights (KV/scratch/cuBLAS workspace).
pub const DEVICE_HEADROOM_BYTES: u64 = 384 * 1024 * 1024;

/// CUDA runtime API status (subset).
type CudaError = c_int;
const CUDA_SUCCESS: CudaError = 0;

/// `cudaMemcpyKind`
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

/// cuBLAS status / ops
type CublasStatus = c_int;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

type FnCudaGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> CudaError;
type FnCudaSetDevice = unsafe extern "C" fn(c_int) -> CudaError;
type FnCudaGetDeviceProperties =
    unsafe extern "C" fn(*mut CudaDevicePropCompat, c_int) -> CudaError;
type FnCudaStreamCreate = unsafe extern "C" fn(*mut *mut c_void) -> CudaError;
type FnCudaStreamDestroy = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaError;
type FnCudaFree = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> CudaError;
type FnCudaMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> CudaError;
type FnCudaDeviceSynchronize = unsafe extern "C" fn() -> CudaError;
type FnCudaGetErrorString = unsafe extern "C" fn(CudaError) -> *const c_char;
type FnCudaEventCreate = unsafe extern "C" fn(*mut *mut c_void) -> CudaError;
type FnCudaEventDestroy = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CudaError;
type FnCudaEventSynchronize = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaEventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> CudaError;
type FnCudaMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CudaError;
type FnCudaStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnCudaMemsetAsync = unsafe extern "C" fn(*mut c_void, c_int, usize, *mut c_void) -> CudaError;

type FnCublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> CublasStatus;
type FnCublasDestroy = unsafe extern "C" fn(*mut c_void) -> CublasStatus;
type FnCublasSgemv = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const c_void,
    c_int,
    *const c_void,
    c_int,
    *const f32,
    *mut c_void,
    c_int,
) -> CublasStatus;
type FnCublasSgemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const c_void,
    c_int,
    *const c_void,
    c_int,
    *const f32,
    *mut c_void,
    c_int,
) -> CublasStatus;
type FnCublasSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CublasStatus;
/// `cublasGemmEx` (mixed precision; batched prefill: f16 A/B, f32 C, f32 compute).
type FnCublasGemmEx = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> CublasStatus;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16F: c_int = 2;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

/// Truncated `cudaDeviceProp` — we only need `name` (first 256 bytes).
#[repr(C)]
struct CudaDevicePropCompat {
    name: [c_char; 256],
    /// Oversized tail so `cudaGetDeviceProperties` cannot clobber the stack
    /// across toolkit versions (we only read `name`).
    _pad: [u8; 3072],
}

struct CudaFns {
    handle: *mut c_void,
    cublas_handle: *mut c_void,
    get_device_count: FnCudaGetDeviceCount,
    set_device: FnCudaSetDevice,
    get_device_properties: FnCudaGetDeviceProperties,
    stream_create: FnCudaStreamCreate,
    stream_destroy: FnCudaStreamDestroy,
    malloc: FnCudaMalloc,
    free: FnCudaFree,
    #[allow(dead_code)]
    memcpy: FnCudaMemcpy,
    memcpy_async: FnCudaMemcpyAsync,
    device_synchronize: FnCudaDeviceSynchronize,
    get_error_string: FnCudaGetErrorString,
    event_create: FnCudaEventCreate,
    event_destroy: FnCudaEventDestroy,
    event_record: FnCudaEventRecord,
    event_synchronize: FnCudaEventSynchronize,
    event_elapsed_time: FnCudaEventElapsedTime,
    mem_get_info: Option<FnCudaMemGetInfo>,
    stream_synchronize: Option<FnCudaStreamSynchronize>,
    memset_async: Option<FnCudaMemsetAsync>,
    cublas_create: FnCublasCreate,
    cublas_destroy: FnCublasDestroy,
    cublas_sgemv: FnCublasSgemv,
    cublas_sgemm: FnCublasSgemm,
    cublas_set_stream: Option<FnCublasSetStream>,
    cublas_gemm_ex: Option<FnCublasGemmEx>,
}

// SAFETY: `CudaFns` holds dlopen handles and plain C function pointers that stay valid for the
// process; nothing in it is thread-affine.
unsafe impl Send for CudaFns {}
// SAFETY: shared access only reads the function pointers; the CUDA runtime API is thread-safe.
unsafe impl Sync for CudaFns {}

impl Drop for CudaFns {
    fn drop(&mut self) {
        // Leave libcudart / libcublas mapped for process lifetime.
        let _ = (self.handle, self.cublas_handle);
    }
}

pub(super) fn dlsym_required(handle: *mut c_void, name: &str) -> Result<*mut c_void> {
    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    }
    let n = CString::new(name).map_err(|e| CoreError::Backend(e.to_string()))?;
    // SAFETY: `handle` is a live dlopen handle and `n` is a NUL-terminated CString that outlives
    // the call.
    let p = unsafe { dlsym(handle, n.as_ptr()) };
    if p.is_null() {
        return Err(CoreError::Backend(format!(
            "CUDA backend: missing symbol `{name}`"
        )));
    }
    Ok(p)
}

pub(super) fn dlopen_lib(candidates: &[&str]) -> Result<*mut c_void> {
    unsafe extern "C" {
        fn dlopen(filename: *const i8, flags: i32) -> *mut c_void;
        fn dlerror() -> *mut i8;
    }
    const RTLD_NOW: i32 = 2;
    // SAFETY: dlopen/dlerror get NUL-terminated CStrings that live across each call; returned
    // handles are null-checked before use.
    unsafe {
        dlerror();
        for path in candidates {
            let c = CString::new(*path).map_err(|e| CoreError::Backend(e.to_string()))?;
            let handle = dlopen(c.as_ptr(), RTLD_NOW);
            if !handle.is_null() {
                return Ok(handle);
            }
        }
    }
    Err(CoreError::Backend(format!(
        "CUDA backend: failed to dlopen any of {candidates:?} (install CUDA toolkit / driver \
         runtime, or use `--backend cpu` / unset ARTIFICER_BACKEND)"
    )))
}

fn dlopen_cuda_stack() -> Result<CudaFns> {
    const CUDART: &[&str] = &[
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11",
        "libcudart.so",
        "/opt/cuda/lib64/libcudart.so.13",
        "/opt/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
    ];
    const CUBLAS: &[&str] = &[
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so",
        "/opt/cuda/lib64/libcublas.so.13",
        "/opt/cuda/lib64/libcublas.so",
        "/usr/local/cuda/lib64/libcublas.so",
    ];

    let handle = dlopen_lib(CUDART)?;
    let cublas_handle = dlopen_lib(CUBLAS).map_err(|e| {
        CoreError::Backend(format!(
            "{e}; device GEMV/GEMM requires libcublas alongside libcudart"
        ))
    })?;

    // SAFETY: each symbol comes from a live handle via `dlsym_required` (null-checked) and is
    // transmuted to the C signature documented for that CUDA/cuBLAS entry point.
    unsafe {
        macro_rules! must {
            ($h:expr, $name:expr) => {{
                let p = dlsym_required($h, $name)?;
                // The target type is the `CudaFns` field the macro result is assigned to.
                #[allow(clippy::missing_transmute_annotations)]
                std::mem::transmute(p)
            }};
        }
        macro_rules! opt {
            ($h:expr, $name:expr) => {{
                unsafe extern "C" {
                    fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
                }
                let n = CString::new($name).unwrap();
                let p = dlsym($h, n.as_ptr());
                if p.is_null() {
                    None
                } else {
                    // The target type is the `CudaFns` field the macro result is assigned to.
                    #[allow(clippy::missing_transmute_annotations)]
                    Some(std::mem::transmute(p))
                }
            }};
        }

        Ok(CudaFns {
            handle,
            cublas_handle,
            get_device_count: must!(handle, "cudaGetDeviceCount"),
            set_device: must!(handle, "cudaSetDevice"),
            get_device_properties: must!(handle, "cudaGetDeviceProperties"),
            stream_create: must!(handle, "cudaStreamCreate"),
            stream_destroy: must!(handle, "cudaStreamDestroy"),
            malloc: must!(handle, "cudaMalloc"),
            free: must!(handle, "cudaFree"),
            memcpy: must!(handle, "cudaMemcpy"),
            memcpy_async: must!(handle, "cudaMemcpyAsync"),
            device_synchronize: must!(handle, "cudaDeviceSynchronize"),
            get_error_string: must!(handle, "cudaGetErrorString"),
            event_create: must!(handle, "cudaEventCreate"),
            event_destroy: must!(handle, "cudaEventDestroy"),
            event_record: must!(handle, "cudaEventRecord"),
            event_synchronize: must!(handle, "cudaEventSynchronize"),
            event_elapsed_time: must!(handle, "cudaEventElapsedTime"),
            mem_get_info: opt!(handle, "cudaMemGetInfo"),
            stream_synchronize: opt!(handle, "cudaStreamSynchronize"),
            memset_async: opt!(handle, "cudaMemsetAsync"),
            cublas_create: must!(cublas_handle, "cublasCreate_v2"),
            cublas_destroy: must!(cublas_handle, "cublasDestroy_v2"),
            cublas_sgemv: must!(cublas_handle, "cublasSgemv_v2"),
            cublas_sgemm: must!(cublas_handle, "cublasSgemm_v2"),
            cublas_set_stream: opt!(cublas_handle, "cublasSetStream_v2"),
            cublas_gemm_ex: opt!(cublas_handle, "cublasGemmEx"),
        })
    }
}

fn cuda_err_msg(fns: &CudaFns, st: CudaError) -> String {
    // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the C
    // signature declared on `CudaFns`; the returned C string is static in the runtime and is
    // null-checked before `CStr::from_ptr`.
    unsafe {
        let p = (fns.get_error_string)(st);
        if !p.is_null() {
            if let Ok(s) = CStr::from_ptr(p).to_str() {
                return format!("cuda error {st}: {s}");
            }
        }
    }
    format!("cuda error {st}")
}

fn cublas_err_msg(st: CublasStatus) -> String {
    format!("cublas status {st}")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct WeightKey {
    ptr: usize,
    len: usize,
    ty: TensorType,
    k: usize,
    n: usize,
}

struct DeviceBuf {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: `DeviceBuf` is a device address plus a size; it is only dereferenced by CUDA calls made
// while holding the backend's locks on the same device.
unsafe impl Send for DeviceBuf {}
// SAFETY: shared references only read the address and size; device access is serialized as for Send
// above.
unsafe impl Sync for DeviceBuf {}

/// CUDA compute backend (feature `cuda`).
///
/// Holds device ordinal, stream, cuBLAS handle, scratch buffers, and a
/// **resident f32 weight cache** for [`Backend::matmul_vec`].
pub struct CudaBackend {
    pub(crate) device_id: i32,
    device_name: String,
    /// Set after [`Backend::init_device`].
    initialized: bool,
    /// cuBLAS handle (`cublasHandle_t` as usize; 0 = unset).
    cublas: Mutex<usize>,
    /// `cudaStream_t` as usize (`0` = unset).
    stream: Mutex<usize>,
    /// Device scratch for activations / GEMM temporaries.
    d_x: Mutex<DeviceBuf>,
    d_out: Mutex<DeviceBuf>,
    d_b: Mutex<DeviceBuf>,
    /// Host→device f32 weight cache (dequantized once).
    weights: Mutex<HashMap<WeightKey, DeviceBuf>>,
    /// Successful device GEMV launches (for tests / verification).
    gemv_launches: AtomicU64,
    /// Successful device GEMM launches.
    gemm_launches: AtomicU64,
    /// Last timed GEMV wall time in milliseconds (CUDA events).
    last_gemv_ms: Mutex<f32>,
    /// Bytes currently held in the weight cache.
    weight_bytes: AtomicU64,
    /// Attention value / context scratch (device softmax path).
    d_v: Mutex<DeviceBuf>,
    d_o: Mutex<DeviceBuf>,
    /// NVRTC-compiled quantized matvec + softmax kernels (set in init_device).
    kernels: Option<QuantKernels>,
    /// Successful quantized (custom-kernel) matvec launches.
    quant_launches: AtomicU64,
    /// Successful device softmax launches.
    softmax_launches: AtomicU64,
    /// Small static f32 tensors (norm weights / biases) resident on device, keyed like weights.
    statics: Mutex<HashMap<WeightKey, DeviceBuf>>,
    /// Device-forward host-fallback round trips (see `cuda_forward`).
    host_fallback_roundtrips: AtomicU64,
    /// Stream syncs issued by the device-forward path.
    stream_syncs: AtomicU64,
    /// Fused device attention launches (`attn_rows_*`: scores + softmax + AV in one kernel).
    fused_attn_launches: AtomicU64,
    /// Q8-quantized activation scratch (int8 values + per-32 (d, d·Σq)).
    d_xq: Mutex<DeviceBuf>,
    d_xds: Mutex<DeviceBuf>,
    /// Use the mmvq (Q8 activation + dp4a) kernels for Q4_K/Q5_0/Q6_K/Q8_0.
    mmvq: AtomicBool,
    mmvq_launches: AtomicU64,
    act_quant_launches: AtomicU64,
    /// Batched prefill scratch: f16 weight tile, f16 activation rows, per-row inverse scales.
    d_wtile: Mutex<DeviceBuf>,
    d_xh: Mutex<DeviceBuf>,
    d_rinv: Mutex<DeviceBuf>,
    gemm_ex_launches: AtomicU64,
    /// Successful int8 MMQ prefill GEMM launches (`ARTIFICER_PREFILL_MATMUL=mmq`).
    mmq_launches: AtomicU64,
    /// Batched-prefill row count up to which the multi-column mmvq kernels are used instead of
    /// dequant + GemmEx (default from `ARTIFICER_PREFILL_MMVQ_MAX`, else 0 = always GEMM).
    prefill_mmvq_max: std::sync::atomic::AtomicUsize,
    fns: CudaFns,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .field("device_name", &self.device_name)
            .field("initialized", &self.initialized)
            .field("gemv_launches", &self.gemv_launches.load(Ordering::Relaxed))
            .field("gemm_launches", &self.gemm_launches.load(Ordering::Relaxed))
            .field("weight_bytes", &self.weight_bytes.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
    /// Probe libcudart + libcublas + device count; pick device 0.
    pub fn try_new() -> Result<Self> {
        let fns = dlopen_cuda_stack()?;
        let mut count: c_int = 0;
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; any out-pointer is a live local.
        let st = unsafe { (fns.get_device_count)(&mut count) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA backend: cudaGetDeviceCount failed ({})",
                cuda_err_msg(&fns, st)
            )));
        }
        if count <= 0 {
            return Err(CoreError::Backend(
                "CUDA backend: no NVIDIA devices visible (cudaGetDeviceCount=0); \
                 use `--backend cpu` or unset ARTIFICER_BACKEND"
                    .into(),
            ));
        }

        let device_id = 0i32;
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; any out-pointer is a live local.
        let st = unsafe { (fns.set_device)(device_id) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA backend: cudaSetDevice({device_id}) failed ({})",
                cuda_err_msg(&fns, st)
            )));
        }

        let mut prop = CudaDevicePropCompat {
            name: [0; 256],
            _pad: [0; 3072],
        };
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; `prop` is a local sized at least as large as the
        // runtime's cudaDeviceProp (padded).
        let st = unsafe { (fns.get_device_properties)(&mut prop, device_id) };
        let device_name = if st == CUDA_SUCCESS {
            // SAFETY: on success the runtime wrote a NUL-terminated device name into `prop.name`
            // (256 bytes, zero-initialized).
            let raw = unsafe { CStr::from_ptr(prop.name.as_ptr()) };
            raw.to_string_lossy().into_owned()
        } else {
            format!("cuda:{device_id}")
        };

        Ok(Self {
            device_id,
            device_name,
            initialized: false,
            cublas: Mutex::new(0),
            stream: Mutex::new(0),
            d_x: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            d_out: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            d_b: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            weights: Mutex::new(HashMap::new()),
            gemv_launches: AtomicU64::new(0),
            gemm_launches: AtomicU64::new(0),
            last_gemv_ms: Mutex::new(0.0),
            weight_bytes: AtomicU64::new(0),
            d_v: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            d_o: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            kernels: None,
            quant_launches: AtomicU64::new(0),
            softmax_launches: AtomicU64::new(0),
            statics: Mutex::new(HashMap::new()),
            host_fallback_roundtrips: AtomicU64::new(0),
            stream_syncs: AtomicU64::new(0),
            fused_attn_launches: AtomicU64::new(0),
            d_xq: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            d_xds: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            // `ARTIFICER_CUDA_MMV=old` selects the f32-activation `mv_*` kernels (A/B).
            mmvq: AtomicBool::new(!std::env::var("ARTIFICER_CUDA_MMV").is_ok_and(|v| v == "old")),
            mmvq_launches: AtomicU64::new(0),
            act_quant_launches: AtomicU64::new(0),
            d_wtile: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            d_xh: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            d_rinv: Mutex::new(DeviceBuf {
                ptr: std::ptr::null_mut(),
                bytes: 0,
            }),
            gemm_ex_launches: AtomicU64::new(0),
            mmq_launches: AtomicU64::new(0),
            prefill_mmvq_max: std::sync::atomic::AtomicUsize::new(prefill_mmvq_max_rows_env()),
            fns,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// True once [`Backend::init_device`] has created stream + cuBLAS.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Number of successful device GEMV launches since construction.
    pub fn gemv_launch_count(&self) -> u64 {
        self.gemv_launches.load(Ordering::Relaxed)
    }

    /// Number of successful device GEMM launches since construction.
    pub fn gemm_launch_count(&self) -> u64 {
        self.gemm_launches.load(Ordering::Relaxed)
    }

    /// Last GEMV CUDA-event elapsed time (milliseconds).
    pub fn last_gemv_elapsed_ms(&self) -> f32 {
        self.last_gemv_ms.lock().map(|g| *g).unwrap_or(0.0)
    }

    /// Successful quantized-weight custom-kernel matvec launches (NVRTC path).
    pub fn quant_matvec_launch_count(&self) -> u64 {
        self.quant_launches.load(Ordering::Relaxed)
    }

    /// Successful device softmax kernel launches (attention).
    pub fn softmax_launch_count(&self) -> u64 {
        self.softmax_launches.load(Ordering::Relaxed)
    }

    /// NVRTC target arch the kernels were compiled for (e.g. `compute_89`).
    pub fn kernel_arch(&self) -> Option<&str> {
        self.kernels.as_ref().map(|k| k.arch.as_str())
    }

    /// Fail loudly if `bytes` more won't fit in free device memory (minus headroom).
    pub fn check_fits(&self, bytes: u64, what: &str) -> Result<()> {
        let (free, total) = self.mem_info()?;
        let need = bytes.saturating_add(DEVICE_HEADROOM_BYTES);
        if need > free as u64 {
            let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
            return Err(CoreError::Backend(format!(
                "CUDA out of device memory: {what} needs {:.1} MiB (+{:.0} MiB headroom) but only                  {:.1} MiB free of {:.1} MiB (already resident: {:.1} MiB) — model does not fit on                  this GPU; use a smaller quant/model or `--backend cpu` (no silent CPU fallback)",
                mib(bytes),
                mib(DEVICE_HEADROOM_BYTES),
                mib(free as u64),
                mib(total as u64),
                mib(self.resident_weight_bytes())
            )));
        }
        Ok(())
    }

    /// Upload raw GGUF bytes (quantized blocks, as-is) once; reuse thereafter.
    fn ensure_raw_resident(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
    ) -> Result<*mut c_void> {
        let key = WeightKey {
            ptr: weight.as_ptr() as usize,
            len: weight.len(),
            ty,
            k,
            n,
        };
        {
            let map = self
                .weights
                .lock()
                .map_err(|_| CoreError::Backend("CUDA weight cache lock poisoned".into()))?;
            if let Some(buf) = map.get(&key) {
                return Ok(buf.ptr);
            }
        }
        let (bb, be) = device_quant_layout(ty)
            .ok_or_else(|| CoreError::Backend(format!("CUDA: {ty:?} has no device kernel")))?;
        if !k.is_multiple_of(be) {
            return Err(CoreError::Shape(format!(
                "CUDA {ty:?} matvec: k={k} not a multiple of block {be}"
            )));
        }
        let bytes = (k / be) * bb * n;
        if weight.len() < bytes {
            return Err(CoreError::Shape(format!(
                "CUDA {ty:?} weight: {} bytes < required {bytes} for {n}x{k}",
                weight.len()
            )));
        }
        self.set_device()?;
        self.check_fits(bytes as u64, &format!("{ty:?} weight {n}x{k}"))?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; `ptr` is a live local out-pointer and the result is
        // null-checked.
        let st = unsafe { (self.fns.malloc)(&mut ptr, bytes) };
        if st != CUDA_SUCCESS || ptr.is_null() {
            return Err(CoreError::Backend(format!(
                "CUDA quantized weight cudaMalloc({bytes} bytes, {ty:?} {n}x{k}) failed ({}) —                  out of device memory? (no silent CPU fallback)",
                cuda_err_msg(&self.fns, st)
            )));
        }
        self.memcpy_h2d(ptr, weight.as_ptr() as *const c_void, bytes)?;
        self.sync_stream()?;
        let mut map = self
            .weights
            .lock()
            .map_err(|_| CoreError::Backend("CUDA weight cache lock poisoned".into()))?;
        if let Some(existing) = map.get(&key) {
            // SAFETY: `ptr` came from our own cudaMalloc just above and is not stored anywhere
            // else, so it is freed exactly once.
            unsafe { (self.fns.free)(ptr) };
            return Ok(existing.ptr);
        }
        map.insert(key, DeviceBuf { ptr, bytes });
        self.weight_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(ptr)
    }

    /// Quantized-resident matvec via the NVRTC kernel for `ty` (raw GGUF blocks on device).
    pub fn quant_matvec(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let kern = self.kernels.as_ref().ok_or_else(|| {
            CoreError::Backend(
                "CUDA quant_matvec: kernels not compiled (init_device not run)".into(),
            )
        })?;
        let d_w = self.ensure_raw_resident(weight, ty, k, n)?;
        let bytes_x = k * 4;
        let bytes_out = n * 4;
        let mut dx = self
            .d_x
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_x lock poisoned".into()))?;
        Self::ensure_buf(&self.fns, &mut dx, bytes_x)?;
        let mut dout = self
            .d_out
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_out lock poisoned".into()))?;
        Self::ensure_buf(&self.fns, &mut dout, bytes_out)?;
        let stream = self.stream_ptr()?;
        self.memcpy_h2d(dx.ptr, x.as_ptr() as *const c_void, bytes_x)?;
        if self.mmvq_applies(ty, k) {
            let (xq, xds) = self.quantize_act_raw(dx.ptr, k)?;
            self.launch_mmvq(ty, d_w, xq, xds, dout.ptr, k, n)?;
        } else {
            kern.launch_matvec(ty, d_w, dx.ptr, dout.ptr, k, n, stream)?;
        }
        self.memcpy_d2h(out.as_mut_ptr() as *mut c_void, dout.ptr, bytes_out)?;
        self.sync_stream()?;
        self.quant_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Row-major `C(m×n) = A(m×k) @ B(k×n)` on device pointers (cuBLAS).
    fn sgemm_dev(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
    ) -> Result<()> {
        let handle = self.cublas_ptr()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // SAFETY: `handle` is the backend's live cuBLAS handle; callers pass device buffers sized
        // for row-major m×k, k×n and m×n f32 matrices.
        let st = unsafe {
            (self.fns.cublas_sgemm)(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha,
                b,
                n as c_int,
                a,
                k as c_int,
                &beta,
                c,
                n as c_int,
            )
        };
        if st != CUBLAS_STATUS_SUCCESS {
            return Err(CoreError::Backend(format!(
                "cuBLAS Sgemm failed ({}) — device attention did not run (no CPU fallback)",
                cublas_err_msg(st)
            )));
        }
        self.gemm_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// One attention head fully on device: scores = q·Kᵀ (cuBLAS), softmax(scale, softcap)
    /// (NVRTC kernel), out = p·V (cuBLAS). Downloads probabilities into `probs` and context into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_head_device(
        &self,
        q: &[f32],
        k_t: &[f32],
        v: &[f32],
        head_dim: usize,
        t: usize,
        scale: f32,
        softcap: f32,
        probs: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        let kern = self.kernels.as_ref().ok_or_else(|| {
            CoreError::Backend("CUDA attention: kernels not compiled (init_device not run)".into())
        })?;
        if q.len() != head_dim
            || k_t.len() != head_dim * t
            || v.len() != t * head_dim
            || probs.len() != t
            || out.len() != head_dim
        {
            return Err(CoreError::Shape(
                "CUDA attn_head_device: bad operand lengths".into(),
            ));
        }
        self.set_device()?;
        let stream = self.stream_ptr()?;
        let mut dx = self
            .d_x
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_x lock poisoned".into()))?;
        let mut db = self
            .d_b
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_b lock poisoned".into()))?;
        let mut dout = self
            .d_out
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_out lock poisoned".into()))?;
        let mut dv = self
            .d_v
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_v lock poisoned".into()))?;
        let mut d_o = self
            .d_o
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_o lock poisoned".into()))?;
        Self::ensure_buf(&self.fns, &mut dx, head_dim * 4)?;
        Self::ensure_buf(&self.fns, &mut db, head_dim * t * 4)?;
        Self::ensure_buf(&self.fns, &mut dout, t * 4)?;
        Self::ensure_buf(&self.fns, &mut dv, t * head_dim * 4)?;
        Self::ensure_buf(&self.fns, &mut d_o, head_dim * 4)?;
        self.memcpy_h2d(dx.ptr, q.as_ptr() as *const c_void, head_dim * 4)?;
        self.memcpy_h2d(db.ptr, k_t.as_ptr() as *const c_void, head_dim * t * 4)?;
        self.memcpy_h2d(dv.ptr, v.as_ptr() as *const c_void, t * head_dim * 4)?;
        self.sgemm_dev(1, t, head_dim, dx.ptr, db.ptr, dout.ptr)?;
        kern.launch_softmax(dout.ptr, 1, t, scale, softcap, stream)?;
        self.softmax_launches.fetch_add(1, Ordering::Relaxed);
        self.sgemm_dev(1, head_dim, t, dout.ptr, dv.ptr, d_o.ptr)?;
        self.memcpy_d2h(probs.as_mut_ptr() as *mut c_void, dout.ptr, t * 4)?;
        self.memcpy_d2h(out.as_mut_ptr() as *mut c_void, d_o.ptr, head_dim * 4)?;
        self.sync_stream()
    }

    /// Device softmax over `rows` rows of `n` (in place): `softmax(softcap(x*scale))`.
    pub fn softmax_device(
        &self,
        data: &mut [f32],
        rows: usize,
        n: usize,
        scale: f32,
        softcap: f32,
    ) -> Result<()> {
        let kern = self
            .kernels
            .as_ref()
            .ok_or_else(|| CoreError::Backend("CUDA softmax: kernels not compiled".into()))?;
        if data.len() != rows * n {
            return Err(CoreError::Shape(
                "CUDA softmax_device: len != rows*n".into(),
            ));
        }
        self.set_device()?;
        let stream = self.stream_ptr()?;
        let mut dout = self
            .d_out
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_out lock poisoned".into()))?;
        Self::ensure_buf(&self.fns, &mut dout, data.len() * 4)?;
        self.memcpy_h2d(dout.ptr, data.as_ptr() as *const c_void, data.len() * 4)?;
        kern.launch_softmax(dout.ptr, rows, n, scale, softcap, stream)?;
        self.memcpy_d2h(data.as_mut_ptr() as *mut c_void, dout.ptr, data.len() * 4)?;
        self.sync_stream()?;
        self.softmax_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Total bytes of resident weights on device (raw quantized blocks, or f32 for F32 tensors).
    pub fn resident_weight_bytes(&self) -> u64 {
        self.weight_bytes.load(Ordering::Relaxed)
    }

    /// Free / total device memory in bytes when `cudaMemGetInfo` is available.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let Some(f) = self.fns.mem_get_info else {
            return Err(CoreError::Backend(
                "CUDA mem_info: cudaMemGetInfo unavailable".into(),
            ));
        };
        let mut free = 0usize;
        let mut total = 0usize;
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; both out-pointers are live locals.
        let st = unsafe { f(&mut free, &mut total) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA mem_info failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok((free, total))
    }

    /// Deprecated seam: real decode attention is [`super::cuda_attention::device_multi_head_attn`].
    pub fn attention_qkv_hook(&self, _n_q: usize, _n_kv: usize, _head_dim: usize) -> Result<()> {
        Ok(())
    }

    /// Deprecated seam: softcap/softmax stay on host; see `cuda_attention` module docs.
    pub fn attention_context_hook(&self, _seq_len: usize, _head_dim: usize) -> Result<()> {
        Ok(())
    }

    fn set_device(&self) -> Result<()> {
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; any out-pointer is a live local.
        let st = unsafe { (self.fns.set_device)(self.device_id) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA cudaSetDevice failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    /// Active `cudaStream_t` for the attention kernels / event sync (`pub(crate)`).
    pub(crate) fn stream_ptr(&self) -> Result<*mut c_void> {
        let s = self
            .stream
            .lock()
            .map_err(|_| CoreError::Backend("CUDA stream lock poisoned".into()))?;
        if *s == 0 {
            return Err(CoreError::Backend(
                "CUDA stream not initialized (call init_device)".into(),
            ));
        }
        Ok(*s as *mut c_void)
    }

    fn cublas_ptr(&self) -> Result<*mut c_void> {
        let h = self
            .cublas
            .lock()
            .map_err(|_| CoreError::Backend("CUDA cublas lock poisoned".into()))?;
        if *h == 0 {
            return Err(CoreError::Backend(
                "cuBLAS handle not initialized (call init_device)".into(),
            ));
        }
        Ok(*h as *mut c_void)
    }

    fn ensure_buf(fns: &CudaFns, slot: &mut DeviceBuf, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if slot.ptr.is_null() || slot.bytes < bytes {
            if !slot.ptr.is_null() {
                // SAFETY: `slot.ptr` is a non-null device allocation owned by this slot; it is
                // nulled right after, so it is freed once.
                let st = unsafe { (fns.free)(slot.ptr) };
                if st != CUDA_SUCCESS {
                    return Err(CoreError::Backend(format!(
                        "CUDA cudaFree failed ({})",
                        cuda_err_msg(fns, st)
                    )));
                }
                slot.ptr = std::ptr::null_mut();
                slot.bytes = 0;
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with
            // the C signature declared on `CudaFns`; `ptr` is a live local out-pointer and the
            // result is null-checked.
            let st = unsafe { (fns.malloc)(&mut ptr, bytes) };
            if st != CUDA_SUCCESS || ptr.is_null() {
                return Err(CoreError::Backend(format!(
                    "CUDA cudaMalloc({bytes} bytes) failed ({}) — device path cannot run \
                     (no silent CPU fallback)",
                    cuda_err_msg(fns, st)
                )));
            }
            slot.ptr = ptr;
            slot.bytes = bytes;
        }
        Ok(())
    }

    fn memcpy_h2d(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        let stream = self.stream_ptr()?;
        // SAFETY: `dst` is a device buffer of at least `bytes` (sized by the caller via
        // ensure_buf/alloc) and `src` is a host slice of `bytes`; a pageable-host H2D copy returns
        // only after the source has been read.
        let st =
            unsafe { (self.fns.memcpy_async)(dst, src, bytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA H2D memcpy failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    fn memcpy_d2h(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        let stream = self.stream_ptr()?;
        // SAFETY: `dst` is a host slice of at least `bytes` and `src` a device buffer of `bytes`;
        // the caller synchronizes the stream before reading `dst`.
        let st =
            unsafe { (self.fns.memcpy_async)(dst, src, bytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA D2H memcpy failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    fn sync_stream(&self) -> Result<()> {
        // Event sync on the stream via device synchronize (portable).
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; the stream (when used) is the backend's live stream.
        let st = unsafe { (self.fns.device_synchronize)() };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA device synchronize failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    /// Dequantize host weight to f32 (n×k row-major) and upload once; reuse thereafter.
    fn ensure_weight_resident(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
    ) -> Result<*mut c_void> {
        let key = WeightKey {
            ptr: weight.as_ptr() as usize,
            len: weight.len(),
            ty,
            k,
            n,
        };
        {
            let map = self
                .weights
                .lock()
                .map_err(|_| CoreError::Backend("CUDA weight cache lock poisoned".into()))?;
            if let Some(buf) = map.get(&key) {
                return Ok(buf.ptr);
            }
        }

        let elems = k
            .checked_mul(n)
            .ok_or_else(|| CoreError::Backend("CUDA weight size overflow".into()))?;
        let mut host = vec![0.0f32; elems];
        // Full-matrix dequant: element layout matches CPU matmul rows (row j at j*k).
        dequantize(weight, ty, 0, elems, &mut host)?;

        let bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CoreError::Backend("CUDA weight byte size overflow".into()))?;
        self.set_device()?;
        self.check_fits(bytes as u64, &format!("f32 weight {n}x{k}"))?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; `ptr` is a live local out-pointer and the result is
        // null-checked.
        let st = unsafe { (self.fns.malloc)(&mut ptr, bytes) };
        if st != CUDA_SUCCESS || ptr.is_null() {
            return Err(CoreError::Backend(format!(
                "CUDA weight upload cudaMalloc({bytes} bytes ≈ {:.1} MiB f32 for {n}×{k} {:?}) \
                 failed ({}) — reduce model size or use `--backend cpu` (no silent CPU fallback)",
                bytes as f64 / (1024.0 * 1024.0),
                ty,
                cuda_err_msg(&self.fns, st)
            )));
        }
        self.memcpy_h2d(ptr, host.as_ptr() as *const c_void, bytes)?;
        self.sync_stream()?;

        let mut map = self
            .weights
            .lock()
            .map_err(|_| CoreError::Backend("CUDA weight cache lock poisoned".into()))?;
        // Another thread may have won the race — free ours and use theirs.
        if let Some(existing) = map.get(&key) {
            // SAFETY: `ptr` came from our own cudaMalloc just above and is not stored anywhere
            // else, so it is freed exactly once.
            let st = unsafe { (self.fns.free)(ptr) };
            if st != CUDA_SUCCESS {
                return Err(CoreError::Backend(format!(
                    "CUDA cudaFree of raced weight failed ({})",
                    cuda_err_msg(&self.fns, st)
                )));
            }
            return Ok(existing.ptr);
        }
        map.insert(key, DeviceBuf { ptr, bytes });
        self.weight_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(ptr)
    }

    /// Device GEMV: `out[j] = sum_i W[j, i] * x[i]` for row-major `W` (n×k), same as CPU.
    ///
    /// Public for tests. Uses cuBLAS `Sgemv` with `CUBLAS_OP_T` on a
    /// column-major view of the row-major matrix (lda = k).
    // `d_w` is a device pointer (a resident weight buffer). The host never
    // dereferences it; it is only passed to cuBLAS, which reads it on the device.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn gemv_f32(
        &self,
        d_w: *mut c_void,
        k: usize,
        n: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if !self.initialized {
            return Err(CoreError::Backend(
                "CUDA gemv_f32: device not initialized".into(),
            ));
        }
        if x.len() != k || out.len() != n {
            return Err(CoreError::Shape(format!(
                "CUDA gemv_f32: x.len={} k={k} out.len={} n={n}",
                x.len(),
                out.len()
            )));
        }
        if d_w.is_null() {
            return Err(CoreError::Backend(
                "CUDA gemv_f32: null weight pointer".into(),
            ));
        }
        self.set_device()?;

        let bytes_x = k * std::mem::size_of::<f32>();
        let bytes_out = n * std::mem::size_of::<f32>();
        {
            let mut dx = self
                .d_x
                .lock()
                .map_err(|_| CoreError::Backend("CUDA d_x lock poisoned".into()))?;
            Self::ensure_buf(&self.fns, &mut dx, bytes_x)?;
            let mut dout = self
                .d_out
                .lock()
                .map_err(|_| CoreError::Backend("CUDA d_out lock poisoned".into()))?;
            Self::ensure_buf(&self.fns, &mut dout, bytes_out)?;

            self.memcpy_h2d(dx.ptr, x.as_ptr() as *const c_void, bytes_x)?;

            let handle = self.cublas_ptr()?;
            let stream = self.stream_ptr()?;
            if let Some(set_stream) = self.fns.cublas_set_stream {
                // SAFETY: `handle` and `stream` are the backend's live cuBLAS handle and CUDA
                // stream.
                let st = unsafe { set_stream(handle, stream) };
                if st != CUBLAS_STATUS_SUCCESS {
                    return Err(CoreError::Backend(format!(
                        "cuBLAS set stream failed ({})",
                        cublas_err_msg(st)
                    )));
                }
            }

            let alpha = 1.0f32;
            let beta = 0.0f32;
            // Events for verification evidence.
            let mut ev_start: *mut c_void = std::ptr::null_mut();
            let mut ev_stop: *mut c_void = std::ptr::null_mut();
            // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with
            // the C signature declared on `CudaFns`; any out-pointer is a live local.
            let st = unsafe { (self.fns.event_create)(&mut ev_start) };
            if st != CUDA_SUCCESS {
                return Err(CoreError::Backend(format!(
                    "cudaEventCreate start failed ({})",
                    cuda_err_msg(&self.fns, st)
                )));
            }
            // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with
            // the C signature declared on `CudaFns`; any out-pointer is a live local.
            let st = unsafe { (self.fns.event_create)(&mut ev_stop) };
            if st != CUDA_SUCCESS {
                // SAFETY: the events were created above, are destroyed once here, and are not used
                // afterwards.
                unsafe {
                    (self.fns.event_destroy)(ev_start);
                }
                return Err(CoreError::Backend(format!(
                    "cudaEventCreate stop failed ({})",
                    cuda_err_msg(&self.fns, st)
                )));
            }
            // SAFETY: the events were created above and `stream` is the backend's live stream.
            unsafe {
                (self.fns.event_record)(ev_start, stream);
            }

            // Row-major W (n×k) ≡ col-major A (k×n); y = A^T x  => CUBLAS_OP_T, m=k, n=n, lda=k.
            // SAFETY: x and out were checked to be k and n long and copied into d_x/d_out sized by
            // ensure_buf; d_w is a resident n×k f32 weight.
            let st = unsafe {
                (self.fns.cublas_sgemv)(
                    handle,
                    CUBLAS_OP_T,
                    k as c_int,
                    n as c_int,
                    &alpha,
                    d_w,
                    k as c_int,
                    dx.ptr,
                    1,
                    &beta,
                    dout.ptr,
                    1,
                )
            };
            if st != CUBLAS_STATUS_SUCCESS {
                // SAFETY: the events were created above, are destroyed once here, and are not used
                // afterwards.
                unsafe {
                    (self.fns.event_destroy)(ev_start);
                    (self.fns.event_destroy)(ev_stop);
                }
                return Err(CoreError::Backend(format!(
                    "cuBLAS Sgemv failed ({}) — device GEMV did not run (no CPU fallback)",
                    cublas_err_msg(st)
                )));
            }

            // SAFETY: the events were created above and `stream` is the backend's live stream.
            unsafe {
                (self.fns.event_record)(ev_stop, stream);
                (self.fns.event_synchronize)(ev_stop);
            }
            let mut ms = 0.0f32;
            // SAFETY: both events were recorded on this stream and ev_stop has been synchronized;
            // `ms` is a live local.
            let st = unsafe { (self.fns.event_elapsed_time)(&mut ms, ev_start, ev_stop) };
            // SAFETY: the events were created above, are destroyed once here, and are not used
            // afterwards.
            unsafe {
                (self.fns.event_destroy)(ev_start);
                (self.fns.event_destroy)(ev_stop);
            }
            if st == CUDA_SUCCESS {
                if let Ok(mut slot) = self.last_gemv_ms.lock() {
                    *slot = ms;
                }
            }

            self.memcpy_d2h(out.as_mut_ptr() as *mut c_void, dout.ptr, bytes_out)?;
            self.sync_stream()?;
        }
        self.gemv_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Device GEMM (row-major f32): `C (m×n) = A (m×k) @ B (k×n)`.
    ///
    /// Uploads A/B, runs cuBLAS `Sgemm`, downloads C. Public for tests.
    /// Does not use the weight cache (ephemeral matrices).
    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<()> {
        if !self.initialized {
            return Err(CoreError::Backend(
                "CUDA gemm_f32: device not initialized".into(),
            ));
        }
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(CoreError::Shape(format!(
                "CUDA gemm_f32: bad lengths a={} (need {}×{}) b={} (need {}×{}) c={} (need {}×{})",
                a.len(),
                m,
                k,
                b.len(),
                k,
                n,
                c.len(),
                m,
                n
            )));
        }
        self.set_device()?;

        let bytes_a = m * k * std::mem::size_of::<f32>();
        let bytes_b = k * n * std::mem::size_of::<f32>();
        let bytes_c = m * n * std::mem::size_of::<f32>();

        let mut dx = self
            .d_x
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_x lock poisoned".into()))?;
        let mut db = self
            .d_b
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_b lock poisoned".into()))?;
        let mut dout = self
            .d_out
            .lock()
            .map_err(|_| CoreError::Backend("CUDA d_out lock poisoned".into()))?;
        Self::ensure_buf(&self.fns, &mut dx, bytes_a)?;
        Self::ensure_buf(&self.fns, &mut db, bytes_b)?;
        Self::ensure_buf(&self.fns, &mut dout, bytes_c)?;

        self.memcpy_h2d(dx.ptr, a.as_ptr() as *const c_void, bytes_a)?;
        self.memcpy_h2d(db.ptr, b.as_ptr() as *const c_void, bytes_b)?;

        let handle = self.cublas_ptr()?;
        let stream = self.stream_ptr()?;
        if let Some(set_stream) = self.fns.cublas_set_stream {
            // SAFETY: `handle` and `stream` are the backend's live cuBLAS handle and CUDA stream.
            let st = unsafe { set_stream(handle, stream) };
            if st != CUBLAS_STATUS_SUCCESS {
                return Err(CoreError::Backend(format!(
                    "cuBLAS set stream failed ({})",
                    cublas_err_msg(st)
                )));
            }
        }

        let alpha = 1.0f32;
        let beta = 0.0f32;
        // Row-major C=A@B via col-major trick: C^T = B^T @ A^T with OP_N,OP_N
        // on (B as n×k col?); standard: treat row-major as col-major transpose:
        // cublasSgemm(N,N, n, m, k, B_rm as (n×k col via ldb=n?)).
        // Common pattern for row-major: Sgemm(OP_N, OP_N, n, m, k, B, n, A, k, C, n)
        // computes C_rm = A_rm @ B_rm.
        // SAFETY: d_x/d_b/d_out were sized by ensure_buf for the m×k, k×n and m×n operands just
        // copied in; `handle` is the live cuBLAS handle.
        let st = unsafe {
            (self.fns.cublas_sgemm)(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha,
                db.ptr,
                n as c_int,
                dx.ptr,
                k as c_int,
                &beta,
                dout.ptr,
                n as c_int,
            )
        };
        if st != CUBLAS_STATUS_SUCCESS {
            return Err(CoreError::Backend(format!(
                "cuBLAS Sgemm failed ({}) — device GEMM did not run (no CPU fallback)",
                cublas_err_msg(st)
            )));
        }

        self.memcpy_d2h(c.as_mut_ptr() as *mut c_void, dout.ptr, bytes_c)?;
        self.sync_stream()?;
        self.gemm_launches.fetch_add(1, Ordering::Relaxed);
        let _ = CUDA_MEMCPY_DEVICE_TO_DEVICE; // keep const used for docs / future
        Ok(())
    }
}

// ===========================================================================
// Device-buffer API — used by `cuda_forward` (device-resident
// decode) and by the `cuda_hostops` kernels. Contract: everything is async on
// the single stream returned by [`CudaBackend::stream`]; hooks never sync.
// ===========================================================================

/// Raw device address + length in **f32 elements** (Copy, non-owning). Kernel args take this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevPtr {
    pub ptr: u64,
    pub len: usize,
}

impl DevPtr {
    pub const NULL: DevPtr = DevPtr { ptr: 0, len: 0 };

    /// Sub-view `[elems, elems + len)` (f32 elements). Panics when out of range.
    pub fn offset(self, elems: usize, len: usize) -> DevPtr {
        assert!(
            elems + len <= self.len,
            "DevPtr::offset {elems}+{len} > {}",
            self.len
        );
        DevPtr {
            ptr: self.ptr + (elems * 4) as u64,
            len,
        }
    }

    /// Row `i` of a row-major `[rows × dim]` buffer.
    pub fn row(self, i: usize, dim: usize) -> DevPtr {
        self.offset(i * dim, dim)
    }

    /// First `len` elements.
    pub fn prefix(self, len: usize) -> DevPtr {
        self.offset(0, len)
    }

    pub fn as_raw(self) -> *mut c_void {
        self.ptr as *mut c_void
    }
}

/// Owned device allocation (`cudaMalloc`), freed on drop.
pub struct DeviceTensor {
    ptr: *mut c_void,
    bytes: usize,
    free: FnCudaFree,
}

// SAFETY: `DeviceTensor` owns a device address plus the cudaFree pointer; all device access goes
// through the backend's single stream.
unsafe impl Send for DeviceTensor {}
// SAFETY: shared references only read the address; device work is ordered on the backend's stream
// as for Send above.
unsafe impl Sync for DeviceTensor {}

impl DeviceTensor {
    /// f32 view of the whole allocation.
    pub fn ptr(&self) -> DevPtr {
        DevPtr {
            ptr: self.ptr as u64,
            len: self.bytes / 4,
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Raw byte address (for non-f32 payloads such as f16 KV rows).
    pub fn raw(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DeviceTensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` is a non-null allocation owned by this tensor; Drop runs once, so
            // it is freed once.
            unsafe {
                (self.free)(self.ptr);
            }
        }
    }
}

impl std::fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceTensor({:#x}, {} B)",
            self.ptr as usize, self.bytes
        )
    }
}

/// The single compute stream (`cudaStream_t`).
#[derive(Clone, Copy, Debug)]
pub struct CuStream(pub(crate) *mut c_void);

impl CuStream {
    /// Raw `cudaStream_t` (for cuBLAS / driver calls in the host-op kernels).
    pub fn as_raw(self) -> *mut c_void {
        self.0
    }
}

/// Kernel handle from the shared NVRTC module (`cuda_kernels.cu`).
#[derive(Clone, Copy, Debug)]
pub struct KernelFn(pub(crate) *mut c_void);

/// Kernel-argument builder for [`CudaBackend::launch`]. Each argument occupies an
/// 8-byte slot holding its little-endian value (pointer / i32 / u32 / f32).
#[derive(Default)]
pub struct LaunchArgs {
    vals: Vec<[u8; 8]>,
}

impl LaunchArgs {
    pub fn new() -> Self {
        Self {
            vals: Vec::with_capacity(16),
        }
    }
    pub fn ptr(mut self, p: DevPtr) -> Self {
        self.vals.push(p.ptr.to_ne_bytes());
        self
    }
    pub fn raw(mut self, p: *mut c_void) -> Self {
        self.vals.push((p as u64).to_ne_bytes());
        self
    }
    pub fn i32(mut self, v: i32) -> Self {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&v.to_ne_bytes());
        self.vals.push(b);
        self
    }
    pub fn u32(mut self, v: u32) -> Self {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&v.to_ne_bytes());
        self.vals.push(b);
        self
    }
    pub fn f32(mut self, v: f32) -> Self {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&v.to_ne_bytes());
        self.vals.push(b);
        self
    }
}

impl CudaBackend {
    /// The one stream every device-forward launch / copy goes to.
    pub fn stream(&self) -> Result<CuStream> {
        Ok(CuStream(self.stream_ptr()?))
    }

    fn kernels_ref(&self) -> Result<&QuantKernels> {
        self.kernels.as_ref().ok_or_else(|| {
            CoreError::Backend("CUDA kernels not compiled (init_device not run)".into())
        })
    }

    /// Look up an `extern "C"` kernel in the shared NVRTC module (cached).
    pub fn kernel(&self, name: &str) -> Result<KernelFn> {
        Ok(KernelFn(self.kernels_ref()?.function(name)?))
    }

    /// Async kernel launch on [`Self::stream`]. Never syncs.
    pub fn launch(
        &self,
        f: KernelFn,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        args: LaunchArgs,
    ) -> Result<()> {
        let mut vals = args.vals;
        let mut params: Vec<*mut c_void> = vals
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        let stream = self.stream_ptr()?;
        self.kernels_ref()?.launch_raw(
            f.0,
            grid,
            block,
            shmem,
            stream,
            &mut params,
            "device-forward kernel",
        )
    }

    /// Allocate `bytes` on the device (fails loudly when it does not fit).
    pub fn alloc_bytes(&self, bytes: usize) -> Result<DeviceTensor> {
        self.set_device()?;
        let bytes = bytes.max(4);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; `ptr` is a live local out-pointer and the result is
        // null-checked.
        let st = unsafe { (self.fns.malloc)(&mut ptr, bytes) };
        if st != CUDA_SUCCESS || ptr.is_null() {
            return Err(CoreError::Backend(format!(
                "CUDA cudaMalloc({bytes} bytes) for device-forward buffer failed ({}) — out of device \
                 memory? (no silent CPU fallback)",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(DeviceTensor {
            ptr,
            bytes,
            free: self.fns.free,
        })
    }

    /// Allocate `elems` f32 on the device, zero-filled (async memset).
    pub fn alloc_f32(&self, elems: usize) -> Result<DeviceTensor> {
        let t = self.alloc_bytes(elems * 4)?;
        self.memset0_raw(t.raw(), t.bytes())?;
        Ok(t)
    }

    fn memset0_raw(&self, p: *mut c_void, bytes: usize) -> Result<()> {
        let stream = self.stream_ptr()?;
        let Some(f) = self.fns.memset_async else {
            return Err(CoreError::Backend(
                "CUDA cudaMemsetAsync unavailable".into(),
            ));
        };
        // SAFETY: `p` is a device allocation of at least `bytes` (from alloc_bytes) and `stream` is
        // the backend's live stream.
        let st = unsafe { f(p, 0, bytes, stream) };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA cudaMemsetAsync failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    /// Async zero-fill of `p`.
    pub fn memset0(&self, p: DevPtr) -> Result<()> {
        self.memset0_raw(p.as_raw(), p.len * 4)
    }

    /// Async H2D of `src` into `dst` (`src.len() <= dst.len`). Pageable `src` is
    /// staged by the driver before return, so the caller may reuse it immediately.
    pub fn upload_f32(&self, src: &[f32], dst: DevPtr) -> Result<()> {
        if src.len() > dst.len {
            return Err(CoreError::Shape(format!(
                "upload_f32: {} > dst {}",
                src.len(),
                dst.len
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        self.memcpy_h2d(dst.as_raw(), src.as_ptr() as *const c_void, src.len() * 4)
    }

    /// Async D2D copy of `src` into `dst` (`src.len <= dst.len`).
    pub fn copy_d2d(&self, src: DevPtr, dst: DevPtr) -> Result<()> {
        if src.len > dst.len {
            return Err(CoreError::Shape(format!(
                "copy_d2d: {} > dst {}",
                src.len, dst.len
            )));
        }
        if src.len == 0 {
            return Ok(());
        }
        let stream = self.stream_ptr()?;
        // SAFETY: `src.len <= dst.len` was checked above; both are device allocations on the
        // backend's stream.
        let st = unsafe {
            (self.fns.memcpy_async)(
                dst.as_raw(),
                src.as_raw() as *const c_void,
                src.len * 4,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                stream,
            )
        };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA D2D memcpy failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    /// D2H of `src[..dst.len()]` followed by a stream sync (blocking).
    pub fn download_f32(&self, src: DevPtr, dst: &mut [f32]) -> Result<()> {
        if dst.len() > src.len {
            return Err(CoreError::Shape(format!(
                "download_f32: {} > src {}",
                dst.len(),
                src.len
            )));
        }
        if !dst.is_empty() {
            self.memcpy_d2h(
                dst.as_mut_ptr() as *mut c_void,
                src.as_raw() as *const c_void,
                dst.len() * 4,
            )?;
        }
        self.sync()
    }

    /// Wait for all work queued on [`Self::stream`] (cudaStreamSynchronize).
    pub fn sync(&self) -> Result<()> {
        self.stream_syncs.fetch_add(1, Ordering::Relaxed);
        let stream = self.stream_ptr()?;
        let st = match self.fns.stream_synchronize {
            // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with
            // the C signature declared on `CudaFns`; the stream (when used) is the backend's live
            // stream.
            Some(f) => unsafe { f(stream) },
            // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with
            // the C signature declared on `CudaFns`; the stream (when used) is the backend's live
            // stream.
            None => unsafe { (self.fns.device_synchronize)() },
        };
        if st != CUDA_SUCCESS {
            return Err(CoreError::Backend(format!(
                "CUDA stream synchronize failed ({}) — an earlier async device op failed",
                cuda_err_msg(&self.fns, st)
            )));
        }
        Ok(())
    }

    /// Host-fallback only: D2H + sync, counted in [`Self::host_fallback_roundtrips`]. Used by the
    /// isolated `host_fallback_*` helpers in `cuda_forward` for ops whose device hook declined.
    pub fn host_fallback_download(&self, src: DevPtr, dst: &mut [f32]) -> Result<()> {
        self.host_fallback_roundtrips
            .fetch_add(1, Ordering::Relaxed);
        self.download_f32(src, dst)
    }

    /// Host round trips taken by host-fallback spots in the device forward.
    pub fn host_fallback_roundtrips(&self) -> u64 {
        self.host_fallback_roundtrips.load(Ordering::Relaxed)
    }

    /// Fused device attention kernel launches (resident path; QKᵀ/softmax/AV in one kernel,
    /// so these do not show up in [`Self::gemm_launch_count`]).
    pub fn fused_attention_launch_count(&self) -> u64 {
        self.fused_attn_launches.load(Ordering::Relaxed)
    }

    /// Record one fused attention launch (its device softmax counts as a softmax launch too).
    pub(crate) fn note_fused_attention(&self) {
        self.fused_attn_launches.fetch_add(1, Ordering::Relaxed);
        self.softmax_launches.fetch_add(1, Ordering::Relaxed);
    }

    /// Stream syncs issued by the device-forward API (`sync` / `download_f32`).
    pub fn stream_sync_count(&self) -> u64 {
        self.stream_syncs.load(Ordering::Relaxed)
    }

    /// Small static tensor (norm weight, bias, SSM A/D/dt) as resident f32 — dequantized once, cached.
    pub fn resident_f32_from(
        &self,
        bytes: &[u8],
        ty: TensorType,
        n_elems: usize,
    ) -> Result<DevPtr> {
        let key = WeightKey {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
            ty,
            k: n_elems,
            n: 0,
        };
        {
            let map = self
                .statics
                .lock()
                .map_err(|_| CoreError::Backend("CUDA statics lock poisoned".into()))?;
            if let Some(b) = map.get(&key) {
                return Ok(DevPtr {
                    ptr: b.ptr as u64,
                    len: n_elems,
                });
            }
        }
        let mut host = vec![0.0f32; n_elems];
        dequantize(bytes, ty, 0, n_elems, &mut host)?;
        let t = self.alloc_bytes(n_elems * 4)?;
        self.upload_f32(&host, t.ptr())?;
        self.sync()?;
        let ptr = t.raw();
        let bytes_len = t.bytes();
        std::mem::forget(t); // ownership moves into the statics map (freed in Drop for CudaBackend)
        let mut map = self
            .statics
            .lock()
            .map_err(|_| CoreError::Backend("CUDA statics lock poisoned".into()))?;
        map.insert(
            key,
            DeviceBuf {
                ptr,
                bytes: bytes_len,
            },
        );
        Ok(DevPtr {
            ptr: ptr as u64,
            len: n_elems,
        })
    }

    /// [`Self::resident_f32_from`] for a model tensor.
    pub fn resident_f32(
        &self,
        model: &crate::model::LlamaModel,
        w: &crate::model::WeightTensor,
    ) -> Result<DevPtr> {
        let bytes = model.tensor_bytes(w)?;
        self.resident_f32_from(bytes, w.ty, w.n_elements())
    }

    /// Async device matvec `y[n] = W(n×k) · x[k]` on resident weights; no host copy, no sync.
    pub fn matvec_dev(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: DevPtr,
        y: DevPtr,
    ) -> Result<()> {
        if x.len < k || y.len < n {
            return Err(CoreError::Shape(format!(
                "matvec_dev: x.len={} < k={k} or y.len={} < n={n}",
                x.len, y.len
            )));
        }
        if ty == TensorType::F32 {
            let d_w = self.ensure_weight_resident(weight, ty, k, n)?;
            let handle = self.cublas_ptr()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            // SAFETY: x.len >= k and y.len >= n were checked above; d_w is the resident n×k f32
            // copy of `weight`.
            let st = unsafe {
                (self.fns.cublas_sgemv)(
                    handle,
                    CUBLAS_OP_T,
                    k as c_int,
                    n as c_int,
                    &alpha,
                    d_w,
                    k as c_int,
                    x.as_raw(),
                    1,
                    &beta,
                    y.as_raw(),
                    1,
                )
            };
            if st != CUBLAS_STATUS_SUCCESS {
                return Err(CoreError::Backend(format!(
                    "cuBLAS Sgemv (device-resident) failed ({})",
                    cublas_err_msg(st)
                )));
            }
            self.gemv_launches.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.matvec_dev_multi(x, k, &[MatvecJob { weight, ty, n, y }])
    }

    /// Several matvecs sharing the same input `x[k]` (Q/K/V, gate/up). The Q8
    /// activation is quantized once and reused by every mmvq-eligible job. Async, no sync.
    pub fn matvec_dev_multi(&self, x: DevPtr, k: usize, jobs: &[MatvecJob<'_>]) -> Result<()> {
        if x.len < k {
            return Err(CoreError::Shape(format!(
                "matvec_dev_multi: x.len={} < k={k}",
                x.len
            )));
        }
        let mut q8: Option<(*mut c_void, *mut c_void)> = None;
        for j in jobs {
            if j.y.len < j.n {
                return Err(CoreError::Shape(format!(
                    "matvec_dev_multi: y.len={} < n={}",
                    j.y.len, j.n
                )));
            }
            if j.ty == TensorType::F32 {
                self.matvec_dev(j.weight, j.ty, k, j.n, x, j.y)?;
                continue;
            }
            if device_quant_layout(j.ty).is_none() {
                return Err(CoreError::Backend(format!(
                    "CUDA matvec_dev: tensor type {:?} has no device kernel (no silent CPU fallback)",
                    j.ty
                )));
            }
            let d_w = self.ensure_raw_resident(j.weight, j.ty, k, j.n)?;
            if self.mmvq_applies(j.ty, k) {
                let (xq, xds) = match q8 {
                    Some(v) => v,
                    None => {
                        let v = self.quantize_act_raw(x.as_raw(), k)?;
                        q8 = Some(v);
                        v
                    }
                };
                self.launch_mmvq(j.ty, d_w, xq, xds, j.y.as_raw(), k, j.n)?;
            } else {
                let stream = self.stream_ptr()?;
                self.kernels_ref()?.launch_matvec(
                    j.ty,
                    d_w,
                    x.as_raw(),
                    j.y.as_raw(),
                    k,
                    j.n,
                    stream,
                )?;
            }
            self.quant_launches.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Enable / disable the mmvq (Q8 activation, dp4a) kernels on this backend (A/B and tests).
    pub fn set_mmvq(&self, on: bool) {
        self.mmvq.store(on, Ordering::Relaxed);
    }

    pub fn mmvq_enabled(&self) -> bool {
        self.mmvq.load(Ordering::Relaxed)
    }

    /// Successful mmvq kernel launches (Q8 activations + dp4a).
    pub fn mmvq_launch_count(&self) -> u64 {
        self.mmvq_launches.load(Ordering::Relaxed)
    }

    /// Activation quantization launches (one per distinct matvec input).
    pub fn act_quant_launch_count(&self) -> u64 {
        self.act_quant_launches.load(Ordering::Relaxed)
    }

    fn mmvq_applies(&self, ty: TensorType, k: usize) -> bool {
        self.mmvq_enabled() && k.is_multiple_of(32) && mmvq_kernel_name(ty).is_some()
    }

    /// Quantize device `x[k]` into the shared Q8 scratch (async). Returns (xq, xds) raw pointers.
    fn quantize_act_raw(&self, x: *mut c_void, k: usize) -> Result<(*mut c_void, *mut c_void)> {
        let nb = k / 32;
        let (xq, xds) = {
            let mut q = self
                .d_xq
                .lock()
                .map_err(|_| CoreError::Backend("CUDA d_xq lock poisoned".into()))?;
            Self::ensure_buf(&self.fns, &mut q, k)?;
            let mut d = self
                .d_xds
                .lock()
                .map_err(|_| CoreError::Backend("CUDA d_xds lock poisoned".into()))?;
            Self::ensure_buf(&self.fns, &mut d, nb * 8)?;
            (q.ptr, d.ptr)
        };
        let f = self.kernel("quant_q8")?;
        let args = LaunchArgs::new().raw(x).raw(xq).raw(xds).i32(k as i32);
        self.launch(f, ((nb as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, args)?;
        self.act_quant_launches.fetch_add(1, Ordering::Relaxed);
        Ok((xq, xds))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_mmvq(
        &self,
        ty: TensorType,
        d_w: *mut c_void,
        xq: *mut c_void,
        xds: *mut c_void,
        y: *mut c_void,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let name = mmvq_kernel_name(ty)
            .ok_or_else(|| CoreError::Backend(format!("CUDA: no mmvq kernel for {ty:?}")))?;
        let f = self.kernel(name)?;
        let args = LaunchArgs::new()
            .raw(d_w)
            .raw(xq)
            .raw(xds)
            .raw(y)
            .i32(k as i32)
            .i32(n as i32);
        self.launch(
            f,
            ((n as u32).div_ceil(MMVQ_ROWS_PER_BLOCK), 1, 1),
            (32 * MMVQ_ROWS_PER_BLOCK, 1, 1),
            0,
            args,
        )?;
        self.mmvq_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl CudaBackend {
    /// `true` when batched-prefill GEMMs can run on tensor cores (`cublasGemmEx` present).
    pub fn has_gemm_ex(&self) -> bool {
        self.fns.cublas_gemm_ex.is_some()
    }

    /// Batched-prefill row count up to which multi-column mmvq replaces dequant + GEMM.
    pub fn prefill_mmvq_max_rows(&self) -> usize {
        self.prefill_mmvq_max.load(Ordering::Relaxed)
    }

    /// Override [`Self::prefill_mmvq_max_rows`] on this backend (tests / tuning).
    pub fn set_prefill_mmvq_max_rows(&self, rows: usize) {
        self.prefill_mmvq_max.store(rows, Ordering::Relaxed);
    }

    /// Successful `cublasGemmEx` calls (batched prefill).
    pub fn gemm_ex_launch_count(&self) -> u64 {
        self.gemm_ex_launches.load(Ordering::Relaxed)
    }

    /// Successful int8 MMQ prefill GEMM launches (one per weight job).
    pub fn mmq_launch_count(&self) -> u64 {
        self.mmq_launches.load(Ordering::Relaxed)
    }

    /// The int8 MMQ kernels use `mma.sync.m16n8k32` s8 (sm_80+). Older devices keep `old`.
    pub fn mmq_supported(&self) -> bool {
        self.kernel_arch()
            .and_then(|a| a.strip_prefix("compute_"))
            .and_then(|v| v.parse::<u32>().ok())
            .is_some_and(|cc| cc >= 80)
    }

    /// Int8 MMQ for the quantized jobs that have a kernel: the activation rows are quantized once
    /// (`quant_q8`, Q8 per 32 + (d, d·Σq), the decode mmvq quantizer) and every job runs
    /// `mmq_<type>` on its resident raw blocks. Returns the jobs without a kernel (old path).
    fn gemm_mmq<'j, 'w>(
        &self,
        x: DevPtr,
        k: usize,
        rows: usize,
        ex: Vec<&'j MatvecJob<'w>>,
    ) -> Result<Vec<&'j MatvecJob<'w>>> {
        let (mq, rest): (Vec<&MatvecJob<'_>>, Vec<&MatvecJob<'_>>) = ex
            .into_iter()
            .partition(|j| mmq_kernel_name(j.ty).is_some());
        if mq.is_empty() {
            return Ok(rest);
        }
        let (xq, xds) = self.quantize_act_raw(x.as_raw(), rows * k)?;
        for j in mq {
            let d_w = self.ensure_raw_resident(j.weight, j.ty, k, j.n)?;
            let f = self.kernel(mmq_kernel_name(j.ty).expect("partitioned"))?;
            let args = LaunchArgs::new()
                .raw(d_w)
                .raw(xq)
                .raw(xds)
                .raw(j.y.as_raw())
                .i32(k as i32)
                .i32(j.n as i32)
                .i32(rows as i32);
            let grid = (
                (rows as u32).div_ceil(MMQ_TILE_ROWS),
                (j.n as u32).div_ceil(MMQ_TILE_N),
                1,
            );
            self.launch(f, grid, (MMQ_THREADS, 1, 1), 0, args)?;
            self.mmq_launches.fetch_add(1, Ordering::Relaxed);
        }
        Ok(rest)
    }

    /// Batched prefill GEMM: `Y[rows × n] = X[rows × k] · Wᵀ` (W = `n × k` GGUF tensor,
    /// row-major X/Y, contiguous). See [`Self::gemm_dev_multi`].
    pub fn gemm_dev(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: DevPtr,
        rows: usize,
        y: DevPtr,
    ) -> Result<()> {
        self.gemm_dev_multi(x, k, rows, &[MatvecJob { weight, ty, n, y }])
    }

    /// Several GEMMs over the same activation rows `X[rows × k]` (Q/K/V, gate/up). Async.
    ///
    /// - F32 weights: cuBLAS Sgemm on the resident f32 copy.
    /// - F16 weights: GemmEx straight on the resident raw bytes.
    /// - Quantized: the resident raw blocks are dequantized tile by tile into an f16 scratch
    ///   (`deq_f16_*`, ≤ [`GEMM_TILE_BYTES`]) and multiplied with GemmEx (f16 in, f32 accumulate).
    ///   X is converted to f16 once per call (`rows_to_f16`, per-row power-of-two scale
    ///   undone on Y by `rows_scale`).
    /// - Anything else (no GemmEx symbol, BF16, k % 32 != 0): per-row [`Self::matvec_dev_multi`].
    pub fn gemm_dev_multi(
        &self,
        x: DevPtr,
        k: usize,
        rows: usize,
        jobs: &[MatvecJob<'_>],
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if x.len < rows * k {
            return Err(CoreError::Shape(format!(
                "gemm_dev_multi: x.len={} < rows {rows} × k {k}",
                x.len
            )));
        }
        for j in jobs {
            if j.y.len < rows * j.n {
                return Err(CoreError::Shape(format!(
                    "gemm_dev_multi: y.len={} < rows {rows} × n {}",
                    j.y.len, j.n
                )));
            }
        }
        let gemm_ok = |ty: TensorType| {
            self.fns.cublas_gemm_ex.is_some()
                && k.is_multiple_of(32)
                && (ty == TensorType::F16 || deq_f16_kernel_name(ty).is_some())
        };
        // Per-row fallback for jobs GemmEx cannot take.
        for j in jobs
            .iter()
            .filter(|j| j.ty != TensorType::F32 && !gemm_ok(j.ty))
        {
            for r in 0..rows {
                self.matvec_dev_multi(
                    x.row(r, k),
                    k,
                    &[MatvecJob {
                        weight: j.weight,
                        ty: j.ty,
                        n: j.n,
                        y: j.y.row(r, j.n),
                    }],
                )?;
            }
        }
        // F32 weights: Sgemm on the resident f32 copy (col-major: Y = Wᵀ X, m = n, n = rows).
        for j in jobs.iter().filter(|j| j.ty == TensorType::F32) {
            let d_w = self.ensure_weight_resident(j.weight, j.ty, k, j.n)?;
            let handle = self.cublas_ptr()?;
            let (alpha, beta) = (1.0f32, 0.0f32);
            // SAFETY: d_w is the resident k×n f32 weight; x.len >= rows*k and y.len >= rows*n were
            // checked at the top of this function.
            let st = unsafe {
                (self.fns.cublas_sgemm)(
                    handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    j.n as c_int,
                    rows as c_int,
                    k as c_int,
                    &alpha,
                    d_w,
                    k as c_int,
                    x.as_raw(),
                    k as c_int,
                    &beta,
                    j.y.as_raw(),
                    j.n as c_int,
                )
            };
            if st != CUBLAS_STATUS_SUCCESS {
                return Err(CoreError::Backend(format!(
                    "cuBLAS Sgemm (prefill) failed ({})",
                    cublas_err_msg(st)
                )));
            }
            self.gemm_launches.fetch_add(1, Ordering::Relaxed);
        }
        let mut ex: Vec<&MatvecJob<'_>> = jobs
            .iter()
            .filter(|j| j.ty != TensorType::F32 && gemm_ok(j.ty))
            .collect();
        // Small batches: multi-column mmvq (raw weights read once per 8 rows) beats dequant+GEMM.
        if rows <= self.prefill_mmvq_max_rows() && self.mmvq_enabled() && k.is_multiple_of(32) {
            let (mc, rest): (Vec<&MatvecJob<'_>>, Vec<&MatvecJob<'_>>) = ex
                .into_iter()
                .partition(|j| mmvq8_kernel_name(j.ty).is_some());
            ex = rest;
            if !mc.is_empty() {
                let (xq, xds) = self.quantize_act_raw(x.as_raw(), rows * k)?;
                for j in mc {
                    let d_w = self.ensure_raw_resident(j.weight, j.ty, k, j.n)?;
                    let f = self.kernel(mmvq8_kernel_name(j.ty).expect("partitioned"))?;
                    let args = LaunchArgs::new()
                        .raw(d_w)
                        .raw(xq)
                        .raw(xds)
                        .raw(j.y.as_raw())
                        .i32(k as i32)
                        .i32(j.n as i32)
                        .i32(rows as i32);
                    let grid = (
                        (j.n as u32).div_ceil(MMVQ_ROWS_PER_BLOCK),
                        (rows as u32).div_ceil(8),
                        1,
                    );
                    self.launch(f, grid, (32 * MMVQ_ROWS_PER_BLOCK, 1, 1), 0, args)?;
                    self.mmvq_launches.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        // Int8 MMQ (`ARTIFICER_PREFILL_MATMUL=mmq`); jobs without a kernel stay below.
        if prefill_matmul_mode() == PrefillMatmul::Mmq
            && k.is_multiple_of(32)
            && self.mmq_supported()
        {
            ex = self.gemm_mmq(x, k, rows, ex)?;
        }
        if ex.is_empty() {
            return Ok(());
        }
        let gemm_ex = self.fns.cublas_gemm_ex.expect("checked by gemm_ok");
        // Activation rows → f16 (+ per-row inverse scale), once for all jobs.
        let (xh, rinv) = {
            let mut xh = self
                .d_xh
                .lock()
                .map_err(|_| CoreError::Backend("CUDA d_xh lock poisoned".into()))?;
            Self::ensure_buf(&self.fns, &mut xh, rows * k * 2)?;
            let mut ri = self
                .d_rinv
                .lock()
                .map_err(|_| CoreError::Backend("CUDA d_rinv lock poisoned".into()))?;
            Self::ensure_buf(&self.fns, &mut ri, rows * 4)?;
            (xh.ptr, ri.ptr)
        };
        let f = self.kernel("rows_to_f16")?;
        self.launch(
            f,
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            LaunchArgs::new()
                .raw(x.as_raw())
                .raw(xh)
                .raw(rinv)
                .i32(k as i32),
        )?;
        let handle = self.cublas_ptr()?;
        let (alpha, beta) = (1.0f32, 0.0f32);
        for j in ex {
            let d_w = self.ensure_raw_resident(j.weight, j.ty, k, j.n)?;
            let tile_rows = if j.ty == TensorType::F16 {
                j.n
            } else {
                (gemm_tile_bytes() / (k * 2)).clamp(1, j.n)
            };
            let wt = if j.ty == TensorType::F16 {
                std::ptr::null_mut()
            } else {
                let mut wt = self
                    .d_wtile
                    .lock()
                    .map_err(|_| CoreError::Backend("CUDA d_wtile lock poisoned".into()))?;
                Self::ensure_buf(&self.fns, &mut wt, tile_rows * k * 2)?;
                wt.ptr
            };
            let mut r0 = 0usize;
            while r0 < j.n {
                let t = tile_rows.min(j.n - r0);
                let a = if j.ty == TensorType::F16 {
                    d_w
                } else {
                    let name = deq_f16_kernel_name(j.ty).expect("checked by gemm_ok");
                    let chunks = (t * (k / 32)) as u32;
                    let args = LaunchArgs::new()
                        .raw(d_w)
                        .raw(wt)
                        .i32(k as i32)
                        .i32(r0 as i32)
                        .i32(t as i32);
                    self.launch(
                        self.kernel(name)?,
                        (chunks.div_ceil(128), 1, 1),
                        (128, 1, 1),
                        0,
                        args,
                    )?;
                    wt
                };
                // Col-major view: C(n × rows, ldc = n) = Aᵀ(t × k) · X(k × rows); C offset by r0.
                let c = (j.y.ptr + (r0 * 4) as u64) as *mut c_void;
                // SAFETY: `a` holds t×k f16 weights (resident or dequantized into wt, sized
                // tile_rows×k), and `c` is offset by r0 rows inside the y tensor of n×rows f32.
                let st = unsafe {
                    gemm_ex(
                        handle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        t as c_int,
                        rows as c_int,
                        k as c_int,
                        &alpha as *const f32 as *const c_void,
                        a,
                        CUDA_R_16F,
                        k as c_int,
                        xh,
                        CUDA_R_16F,
                        k as c_int,
                        &beta as *const f32 as *const c_void,
                        c,
                        CUDA_R_32F,
                        j.n as c_int,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT,
                    )
                };
                if st != CUBLAS_STATUS_SUCCESS {
                    return Err(CoreError::Backend(format!(
                        "cuBLAS GemmEx (prefill) failed ({})",
                        cublas_err_msg(st)
                    )));
                }
                self.gemm_ex_launches.fetch_add(1, Ordering::Relaxed);
                r0 += t;
            }
            let total = (rows * j.n) as u32;
            let f = self.kernel("rows_scale")?;
            let args = LaunchArgs::new()
                .raw(j.y.as_raw())
                .raw(rinv)
                .i32(j.n as i32)
                .i32(rows as i32);
            self.launch(
                f,
                (total.div_ceil(256).min(4096), 1, 1),
                (256, 1, 1),
                0,
                args,
            )?;
        }
        Ok(())
    }
}

/// Default max bytes of one dequantized f16 weight tile (batched prefill). Small enough to stay
/// in L2 between the dequant kernel and the GEMM that consumes it.
pub const GEMM_TILE_BYTES: usize = 16 << 20;

/// f16 tile budget: `ARTIFICER_PREFILL_TILE_MB` (tuning) or [`GEMM_TILE_BYTES`].
pub fn gemm_tile_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ARTIFICER_PREFILL_TILE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .map_or(GEMM_TILE_BYTES, |mb| mb << 20)
    })
}

/// Batched-prefill matmul path, `ARTIFICER_PREFILL_MATMUL=old|mmq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMatmul {
    /// Dequantize weight tiles to f16 + cuBLAS GemmEx (f16 x f16 -> f32), the original batched-prefill path.
    Old,
    /// Int8 MMQ: Q8 activations (per 32) x raw quantized weights on int8 tensor cores
    /// (`mma.sync.m16n8k32` s8); types without a kernel fall back to [`PrefillMatmul::Old`].
    Mmq,
}

/// Mode used when `ARTIFICER_PREFILL_MATMUL` is unset, empty or unknown. Flipping the default
/// to MMQ is this one line; an explicit `old` always forces the old path.
pub const PREFILL_MATMUL_DEFAULT: PrefillMatmul = PrefillMatmul::Mmq;

/// Parse `ARTIFICER_PREFILL_MATMUL` (case-insensitive): `old` / `mmq`, else the default.
pub fn parse_prefill_matmul(v: Option<&str>) -> PrefillMatmul {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("old") => PrefillMatmul::Old,
        Some("mmq") => PrefillMatmul::Mmq,
        _ => PREFILL_MATMUL_DEFAULT,
    }
}

thread_local! {
    /// Test-only prefill matmul override for the calling thread (None = environment).
    static PREFILL_MATMUL_OVERRIDE: std::cell::Cell<Option<PrefillMatmul>> = const { std::cell::Cell::new(None) };
}

/// Force the prefill matmul mode on this thread (tests); `None` restores the env value.
#[cfg(test)]
pub(crate) fn set_thread_prefill_matmul(mode: Option<PrefillMatmul>) {
    PREFILL_MATMUL_OVERRIDE.with(|c| c.set(mode));
}

/// Prefill matmul mode: the thread override (tests) or the process-wide env value (read once;
/// an unknown non-empty value logs one warning and uses [`PREFILL_MATMUL_DEFAULT`]).
pub fn prefill_matmul_mode() -> PrefillMatmul {
    if let Some(m) = PREFILL_MATMUL_OVERRIDE.with(|c| c.get()) {
        return m;
    }
    static V: std::sync::OnceLock<PrefillMatmul> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("ARTIFICER_PREFILL_MATMUL").ok();
        let m = parse_prefill_matmul(v.as_deref());
        if let Some(v) = v.as_deref() {
            let t = v.trim().to_ascii_lowercase();
            if !t.is_empty() && t != "old" && t != "mmq" {
                tracing::warn!("ARTIFICER_PREFILL_MATMUL={v:?} is not old|mmq; using {m:?}");
            }
        }
        m
    })
}

/// Int8 MMQ prefill kernel for `ty`; `None` = that type stays on the old path.
pub fn mmq_kernel_name(ty: TensorType) -> Option<&'static str> {
    match ty {
        TensorType::Q5_0 => Some("mmq_q5_0"),
        TensorType::Q4_K => Some("mmq_q4_k"),
        TensorType::Q6_K => Some("mmq_q6_k"),
        _ => None,
    }
}

/// MMQ tile: activation rows per block (grid.x), weight rows per block (grid.y), threads.
/// Must match `MMQ_BM` / `MMQ_BN` / 8 warps in `cuda_kernels.cu`.
const MMQ_TILE_ROWS: u32 = 64;
const MMQ_TILE_N: u32 = 128;
const MMQ_THREADS: u32 = 256;

/// Default for [`CudaBackend::prefill_mmvq_max_rows`]: `ARTIFICER_PREFILL_MMVQ_MAX`, else 0
/// (always dequant + GEMM; measured faster than multi-column mmvq already at 29 rows on
/// Nemotron / RTX 4050: 66 ms vs 143 ms).
fn prefill_mmvq_max_rows_env() -> usize {
    std::env::var("ARTIFICER_PREFILL_MMVQ_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Multi-column (≤ 8 activation rows per slice) mmvq kernel for `ty`.
pub fn mmvq8_kernel_name(ty: TensorType) -> Option<&'static str> {
    Some(match ty {
        TensorType::Q8_0 => "mmvq8_q8_0",
        TensorType::Q5_0 => "mmvq8_q5_0",
        TensorType::Q4_K => "mmvq8_q4_k",
        TensorType::Q6_K => "mmvq8_q6_k",
        _ => return None,
    })
}

/// f16 dequant kernel for batched-prefill GEMM tiles.
pub fn deq_f16_kernel_name(ty: TensorType) -> Option<&'static str> {
    Some(match ty {
        TensorType::Q4_0 => "deq_f16_q4_0",
        TensorType::Q4_1 => "deq_f16_q4_1",
        TensorType::Q5_0 => "deq_f16_q5_0",
        TensorType::Q5_1 => "deq_f16_q5_1",
        TensorType::Q8_0 => "deq_f16_q8_0",
        TensorType::Q2_K => "deq_f16_q2_k",
        TensorType::Q3_K => "deq_f16_q3_k",
        TensorType::Q4_K => "deq_f16_q4_k",
        TensorType::Q5_K => "deq_f16_q5_k",
        TensorType::Q6_K => "deq_f16_q6_k",
        _ => return None,
    })
}

/// Rows (warps) per thread block for the mmvq kernels.
const MMVQ_ROWS_PER_BLOCK: u32 = 4;

/// mmvq (Q8 activation, dp4a) kernel for `ty`, if any (others keep the f32-activation `mv_*` kernels).
pub fn mmvq_kernel_name(ty: TensorType) -> Option<&'static str> {
    Some(match ty {
        TensorType::Q8_0 => "mmvq_q8_0",
        TensorType::Q5_0 => "mmvq_q5_0",
        TensorType::Q4_K => "mmvq_q4_k",
        TensorType::Q6_K => "mmvq_q6_k",
        _ => return None,
    })
}

/// One matvec of a [`CudaBackend::matvec_dev_multi`] batch: `y[n] = W(n×k) · x`.
pub struct MatvecJob<'a> {
    pub weight: &'a [u8],
    pub ty: TensorType,
    pub n: usize,
    pub y: DevPtr,
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Best-effort teardown; ignore errors during drop.
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; best-effort teardown, the result is ignored.
        let _ = unsafe { (self.fns.set_device)(self.device_id) };
        if let Ok(mut map) = self.weights.lock() {
            for (_, buf) in map.drain() {
                if !buf.ptr.is_null() {
                    // SAFETY: each pointer is a non-null device allocation owned by the backend and
                    // is dropped or nulled here, so it is freed once.
                    unsafe {
                        (self.fns.free)(buf.ptr);
                    }
                }
            }
        }
        if let Ok(mut map) = self.statics.lock() {
            for (_, buf) in map.drain() {
                if !buf.ptr.is_null() {
                    // SAFETY: each pointer is a non-null device allocation owned by the backend and
                    // is dropped or nulled here, so it is freed once.
                    unsafe {
                        (self.fns.free)(buf.ptr);
                    }
                }
            }
        }
        for slot in [
            &self.d_x,
            &self.d_out,
            &self.d_b,
            &self.d_v,
            &self.d_o,
            &self.d_xq,
            &self.d_xds,
            &self.d_wtile,
            &self.d_xh,
            &self.d_rinv,
        ] {
            if let Ok(mut b) = slot.lock() {
                if !b.ptr.is_null() {
                    // SAFETY: each pointer is a non-null device allocation owned by the backend and
                    // is dropped or nulled here, so it is freed once.
                    unsafe {
                        (self.fns.free)(b.ptr);
                    }
                    b.ptr = std::ptr::null_mut();
                    b.bytes = 0;
                }
            }
        }
        if let Ok(mut h) = self.cublas.lock() {
            if *h != 0 {
                // SAFETY: `*h` is the cuBLAS handle created in init_device; it is zeroed right
                // after, so it is destroyed once.
                unsafe {
                    (self.fns.cublas_destroy)(*h as *mut c_void);
                }
                *h = 0;
            }
        }
        if let Ok(mut s) = self.stream.lock() {
            if *s != 0 {
                // SAFETY: `*s` is the stream created in init_device; it is zeroed right after, so
                // it is destroyed once.
                unsafe {
                    (self.fns.stream_destroy)(*s as *mut c_void);
                }
                *s = 0;
            }
        }
    }
}

impl Backend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn as_cuda(&self) -> Option<&CudaBackend> {
        Some(self)
    }

    fn init_device(&mut self) -> Result<()> {
        self.set_device()?;

        let mut stream: *mut c_void = std::ptr::null_mut();
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; any out-pointer is a live local.
        let st = unsafe { (self.fns.stream_create)(&mut stream) };
        if st != CUDA_SUCCESS || stream.is_null() {
            return Err(CoreError::Backend(format!(
                "CUDA init_device: cudaStreamCreate failed ({})",
                cuda_err_msg(&self.fns, st)
            )));
        }
        if let Ok(mut s) = self.stream.lock() {
            *s = stream as usize;
        }

        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: the function pointer was resolved by dlsym from the loaded CUDA runtime with the
        // C signature declared on `CudaFns`; any out-pointer is a live local.
        let st = unsafe { (self.fns.cublas_create)(&mut handle) };
        if st != CUBLAS_STATUS_SUCCESS || handle.is_null() {
            return Err(CoreError::Backend(format!(
                "CUDA init_device: cublasCreate failed ({})",
                cublas_err_msg(st)
            )));
        }
        if let Some(set_stream) = self.fns.cublas_set_stream {
            // SAFETY: `handle` and `stream` are the backend's live cuBLAS handle and CUDA stream.
            let st = unsafe { set_stream(handle, stream) };
            if st != CUBLAS_STATUS_SUCCESS {
                // SAFETY: `handle` was created just above and is not stored yet; it is destroyed
                // once on this error path.
                unsafe {
                    (self.fns.cublas_destroy)(handle);
                }
                return Err(CoreError::Backend(format!(
                    "CUDA init_device: cublasSetStream failed ({})",
                    cublas_err_msg(st)
                )));
            }
        }
        if let Ok(mut h) = self.cublas.lock() {
            *h = handle as usize;
        }

        // NVRTC compile + driver-API load of quantized matvec / softmax kernels.
        self.kernels = Some(QuantKernels::compile_and_load(self.device_id)?);

        self.initialized = true;
        Ok(())
    }

    fn matmul_vec(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if !self.initialized {
            return Err(CoreError::Backend(
                "CUDA matmul_vec: device not initialized (call init_device via create_backend)"
                    .into(),
            ));
        }
        if x.len() != k || out.len() != n {
            return Err(CoreError::Shape(format!(
                "CUDA matmul_vec: x.len={} k={k} out.len={} n={n}",
                x.len(),
                out.len()
            )));
        }
        // Real device path only — explicit error on failure (no host matmul).
        if ty == TensorType::F32 {
            // F32 tensors: cuBLAS Sgemv on resident f32.
            let d_w = self.ensure_weight_resident(weight, ty, k, n)?;
            return self.gemv_f32(d_w, k, n, x, out);
        }
        if device_quant_layout(ty).is_some() {
            // Quantized / f16: raw GGUF blocks resident, dequantized inside the kernel.
            return self.quant_matvec(weight, ty, k, n, x, out);
        }
        Err(CoreError::Backend(format!(
            "CUDA matmul_vec: tensor type {ty:?} has no device kernel (supported: F32, F16, BF16, \
             Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K); use `--backend cpu` (no silent CPU fallback)"
        )))
    }

    fn preflight_weights(&self, bytes: u64) -> Result<()> {
        self.check_fits(bytes, "model weights (quantized, as stored in GGUF)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use gguf::TensorType;

    fn cpu_ref(weight: &[u8], ty: TensorType, k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        CpuBackend
            .matmul_vec(weight, ty, k, n, x, &mut out)
            .unwrap();
        out
    }

    fn make_f32_weight(n: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut w = vec![0.0f32; n * k];
        let mut s = seed;
        for v in w.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 16) as f32 / 65535.0) * 2.0 - 1.0;
        }
        let bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
        (bytes, w)
    }

    fn skip_no_device() -> Option<CudaBackend> {
        let mut b = CudaBackend::try_new().ok()?;
        b.init_device().ok()?;
        Some(b)
    }

    /// Host-only: `ARTIFICER_PREFILL_MATMUL` parsing. Unset / empty / unknown follow the single
    /// default constant; an explicit `old` always forces the old path.
    #[test]
    fn prefill_matmul_switch_parsing() {
        assert_eq!(
            PREFILL_MATMUL_DEFAULT,
            PrefillMatmul::Mmq,
            "default is the int8 MMQ path; explicit old forces the old path"
        );
        assert_eq!(parse_prefill_matmul(None), PREFILL_MATMUL_DEFAULT);
        assert_eq!(parse_prefill_matmul(Some("")), PREFILL_MATMUL_DEFAULT);
        assert_eq!(parse_prefill_matmul(Some("bogus")), PREFILL_MATMUL_DEFAULT);
        for v in ["old", "OLD", " old "] {
            assert_eq!(parse_prefill_matmul(Some(v)), PrefillMatmul::Old);
        }
        for v in ["mmq", "MMQ", " mmq\n"] {
            assert_eq!(parse_prefill_matmul(Some(v)), PrefillMatmul::Mmq);
        }
        set_thread_prefill_matmul(Some(PrefillMatmul::Mmq));
        assert_eq!(prefill_matmul_mode(), PrefillMatmul::Mmq);
        set_thread_prefill_matmul(Some(PrefillMatmul::Old));
        assert_eq!(prefill_matmul_mode(), PrefillMatmul::Old);
        set_thread_prefill_matmul(None);
    }

    // ---- Int8 MMQ kernel tests (GPU) ----

    /// Random raw blocks with sane f16 scale fields (as `cuda_quant` tests) for the MMQ types.
    fn mmq_random_weight(ty: TensorType, k: usize, n: usize, seed: u32) -> Vec<u8> {
        let (bb, be, f16s): (usize, usize, &[usize]) = match ty {
            TensorType::Q5_0 => (22, 32, &[0]),
            TensorType::Q4_K => (144, 256, &[0, 2]),
            TensorType::Q6_K => (210, 256, &[208]),
            _ => unreachable!("not an MMQ test type"),
        };
        let nblk = k / be * n;
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s >> 8
        };
        let mut w: Vec<u8> = (0..nblk * bb).map(|_| (next() & 0xFF) as u8).collect();
        for b in 0..nblk {
            for &off in f16s {
                let h = crate::quant::f32_to_f16(0.002 + (next() % 1000) as f32 * 0.00005);
                w[b * bb + off..b * bb + off + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
        w
    }

    /// Host copy of `quant_q8` (d = amax/127, round-to-nearest-even) returning q*d.
    fn mmq_q8_emulate(x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for (bi, blk) in x.chunks(32).enumerate() {
            let d = blk.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (i, v) in blk.iter().enumerate() {
                out[bi * 32 + i] = if d == 0.0 {
                    0.0
                } else {
                    (v / d).round_ties_even() * d
                };
            }
        }
        out
    }

    /// Run one prefill GEMM with the given matmul mode; returns Y and the (mmq, gemm_ex) launch deltas.
    #[allow(clippy::too_many_arguments)]
    fn mmq_run(
        b: &CudaBackend,
        mode: PrefillMatmul,
        w: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: &[f32],
        rows: usize,
    ) -> (Vec<f32>, u64, u64) {
        set_thread_prefill_matmul(Some(mode));
        let dx = b.alloc_f32(rows * k).unwrap();
        b.upload_f32(x, dx.ptr()).unwrap();
        let dy = b.alloc_f32(rows * n).unwrap();
        let (m0, g0) = (b.mmq_launch_count(), b.gemm_ex_launch_count());
        b.gemm_dev(w, ty, k, n, dx.ptr(), rows, dy.ptr())
            .expect("gemm_dev");
        let mut y = vec![0.0f32; rows * n];
        b.download_f32(dy.ptr(), &mut y).unwrap();
        set_thread_prefill_matmul(None);
        (y, b.mmq_launch_count() - m0, b.gemm_ex_launch_count() - g0)
    }

    /// MMQ vs (1) a host f64 reference over the same Q8-quantized activations (tight: int32 block
    /// sums are exact, only the f32 fold order differs) and (2) the `old` dequant+GemmEx path within
    /// the documented Q8 activation bound (sum |w| * amax/254 per block, the decode mmvq bound) plus
    /// old's f16 slack. Shapes: partial k stage (k % 128 != 0), ragged n (n % 128 != 0), 1 / 7 / 64 /
    /// 65 / 1024 rows, real Nemotron k = 3136 (98 Q5_0 blocks) and 12544.
    fn mmq_check_type(ty: TensorType) {
        let Some(b) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        if !b.mmq_supported() || !b.has_gemm_ex() {
            eprintln!("skip: MMQ needs sm_80+ and GemmEx for the old reference");
            return;
        }
        let kname = mmq_kernel_name(ty).expect("type has an MMQ kernel");
        b.set_prefill_mmvq_max_rows(0);
        let q5 = ty == TensorType::Q5_0;
        let shapes: Vec<(usize, usize, usize)> = if q5 {
            vec![
                (96, 37, 1),
                (3136, 130, 7),
                (3136, 256, 64),
                (3136, 129, 65),
                (512, 200, 1024),
                (12544, 64, 3),
            ]
        } else {
            vec![
                (256, 37, 1),
                (768, 130, 7),
                (3072, 256, 64),
                (7680, 129, 65),
                (512, 200, 1024),
                (12544, 64, 3),
            ]
        };
        let (mut worst_tight, mut worst_old) = (0.0f64, 0.0f64);
        for (si, &(k, n, rows)) in shapes.iter().enumerate() {
            let w = mmq_random_weight(ty, k, n, 5 + si as u32 * 13 + k as u32);
            let mut s = 21u32 + rows as u32 + k as u32;
            let x: Vec<f32> = (0..rows * k)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((s >> 8) & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0) * 3.0
                })
                .collect();
            let (ym, dm, gm) = mmq_run(&b, PrefillMatmul::Mmq, &w, ty, k, n, &x, rows);
            assert_eq!(
                (dm, gm),
                (1, 0),
                "{ty:?} {rows}x{k}->{n}: {kname} must run once and GemmEx not at all"
            );
            let (yo, dm, gm) = mmq_run(&b, PrefillMatmul::Old, &w, ty, k, n, &x, rows);
            assert!(
                dm == 0 && gm > 0,
                "{ty:?}: explicit old must take the old path"
            );
            let mut full = vec![0.0f32; k * n];
            dequantize(&w, ty, 0, k * n, &mut full).unwrap();
            for rr in 0..rows {
                let xr = &x[rr * k..(rr + 1) * k];
                let xe = mmq_q8_emulate(xr);
                for j in 0..n {
                    let wr = &full[j * k..(j + 1) * k];
                    let qref: f64 = wr.iter().zip(&xe).map(|(a, c)| *a as f64 * *c as f64).sum();
                    let qmag: f64 = wr
                        .iter()
                        .zip(&xe)
                        .map(|(a, c)| (*a as f64 * *c as f64).abs())
                        .sum();
                    let mag: f64 = wr
                        .iter()
                        .zip(xr)
                        .map(|(a, c)| (*a as f64 * *c as f64).abs())
                        .sum();
                    let mut q8 = 0.0f64;
                    for (bi, blk) in xr.chunks(32).enumerate() {
                        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
                        q8 += wr[bi * 32..bi * 32 + 32]
                            .iter()
                            .map(|v| v.abs() as f64)
                            .sum::<f64>()
                            * amax
                            / 254.0;
                    }
                    let got = ym[rr * n + j] as f64;
                    let tight = 1e-4 * qmag + 1e-5;
                    let e = (got - qref).abs();
                    worst_tight = worst_tight.max(e / tight);
                    assert!(e <= tight, "{ty:?} {rows}x{k}->{n} row {rr} col {j}: mmq {got} vs Q8 ref {qref} (tol {tight:.3e})");
                    let loose = q8 * 1.01 + 2e-3 * mag + 1e-4;
                    let eo = (got - yo[rr * n + j] as f64).abs();
                    worst_old = worst_old.max(eo / loose);
                    assert!(eo <= loose, "{ty:?} {rows}x{k}->{n} row {rr} col {j}: mmq {got} vs old {} (tol {loose:.3e})", yo[rr * n + j]);
                }
            }
        }
        eprintln!("{ty:?} mmq: worst |err|/tol vs Q8 ref {worst_tight:.3}, vs old {worst_old:.3}");
    }

    #[test]
    fn mmq_matches_reference_q5_0() {
        mmq_check_type(TensorType::Q5_0);
    }

    #[test]
    fn mmq_matches_reference_q4_k() {
        mmq_check_type(TensorType::Q4_K);
    }

    #[test]
    fn mmq_matches_reference_q6_k() {
        mmq_check_type(TensorType::Q6_K);
    }

    /// Several jobs over one activation set (q/k/v style): one quantize, one MMQ launch per
    /// MMQ-type job, and a non-MMQ type (Q8_0) in the same call stays on GemmEx.
    #[test]
    fn mmq_multi_job_mixed_types() {
        let Some(b) = skip_no_device() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        if !b.mmq_supported() || !b.has_gemm_ex() {
            return;
        }
        b.set_prefill_mmvq_max_rows(0);
        let (k, rows) = (3136usize, 33usize);
        let wa = mmq_random_weight(TensorType::Q5_0, k, 160, 3);
        let wb = mmq_random_weight(TensorType::Q5_0, k, 96, 4);
        let mut wc = vec![0u8; k / 32 * 40 * 34];
        for (i, v) in wc.iter_mut().enumerate() {
            *v = (i * 37 % 251) as u8;
        }
        for blk in wc.chunks_mut(34) {
            blk[..2].copy_from_slice(&crate::quant::f32_to_f16(0.003).to_le_bytes());
        }
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let dx = b.alloc_f32(rows * k).unwrap();
        b.upload_f32(&x, dx.ptr()).unwrap();
        let ys: Vec<_> = [160usize, 96, 40]
            .iter()
            .map(|&n| b.alloc_f32(rows * n).unwrap())
            .collect();
        let run = |mode| {
            set_thread_prefill_matmul(Some(mode));
            let jobs = [
                MatvecJob {
                    weight: &wa,
                    ty: TensorType::Q5_0,
                    n: 160,
                    y: ys[0].ptr(),
                },
                MatvecJob {
                    weight: &wb,
                    ty: TensorType::Q5_0,
                    n: 96,
                    y: ys[1].ptr(),
                },
                MatvecJob {
                    weight: &wc,
                    ty: TensorType::Q8_0,
                    n: 40,
                    y: ys[2].ptr(),
                },
            ];
            let (m0, g0, q0) = (
                b.mmq_launch_count(),
                b.gemm_ex_launch_count(),
                b.act_quant_launches.load(Ordering::Relaxed),
            );
            b.gemm_dev_multi(dx.ptr(), k, rows, &jobs).unwrap();
            set_thread_prefill_matmul(None);
            let out: Vec<Vec<f32>> = [160usize, 96, 40]
                .iter()
                .zip(&ys)
                .map(|(&n, y)| {
                    let mut v = vec![0.0f32; rows * n];
                    b.download_f32(y.ptr(), &mut v).unwrap();
                    v
                })
                .collect();
            let q = b.act_quant_launches.load(Ordering::Relaxed) - q0;
            (
                out,
                b.mmq_launch_count() - m0,
                b.gemm_ex_launch_count() - g0,
                q,
            )
        };
        let (om, m, g, q) = run(PrefillMatmul::Mmq);
        assert_eq!(
            (m, q),
            (2, 1),
            "two MMQ launches sharing one activation quantize"
        );
        assert!(g > 0, "Q8_0 job must stay on GemmEx");
        let (oo, m, _, _) = run(PrefillMatmul::Old);
        assert_eq!(m, 0);
        assert_eq!(om[2], oo[2], "the non-MMQ job is the old path bit for bit");
        for (a, c) in om[..2].iter().zip(&oo[..2]) {
            let d = a
                .iter()
                .zip(c)
                .map(|(u, v)| (u - v).abs())
                .fold(0.0f32, f32::max);
            let m = c.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            assert!(
                d <= 0.02 * m + 1e-3,
                "mmq vs old max diff {d} (max |y| {m})"
            );
        }
    }

    #[test]
    fn try_new_probes_or_errors_clearly() {
        match CudaBackend::try_new() {
            Ok(b) => {
                assert_eq!(b.kind(), BackendKind::Cuda);
                assert_eq!(b.name(), "cuda");
                assert!(!b.device_name().is_empty());
                assert!(!b.is_initialized());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("CUDA") || msg.contains("cuda") || msg.contains("libcudart"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn device_gemv_matches_cpu_f32_shapes() {
        let Some(b) = skip_no_device() else {
            return;
        };
        // Include non-multiples of common tile sizes (16/32/128).
        let shapes = [(1, 1), (3, 5), (17, 33), (64, 64), (127, 63), (256, 128)];
        for (k, n) in shapes {
            let (bytes, _) = make_f32_weight(n, k, (k * 1000 + n) as u32);
            let mut x = vec![0.0f32; k];
            let mut s = 7u32;
            for v in x.iter_mut() {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                *v = ((s >> 16) as f32 / 65535.0) - 0.5;
            }
            let expected = cpu_ref(&bytes, TensorType::F32, k, n, &x);
            let mut out = vec![0.0f32; n];
            b.matmul_vec(&bytes, TensorType::F32, k, n, &x, &mut out)
                .expect("device gemv");
            for j in 0..n {
                let tol = 1e-3 * (1.0 + expected[j].abs());
                assert!(
                    (out[j] - expected[j]).abs() < tol,
                    "shape {k}x{n} j={j}: device {} vs cpu {} (tol {tol})",
                    out[j],
                    expected[j]
                );
            }
        }
        assert!(b.gemv_launch_count() >= shapes.len() as u64);
        assert!(b.resident_weight_bytes() > 0);
        assert!(b.last_gemv_elapsed_ms() >= 0.0);
        eprintln!(
            "CUDA GEMV ok on {} — launches={} last_ms={:.4} weight_MiB={:.2}",
            b.device_name(),
            b.gemv_launch_count(),
            b.last_gemv_elapsed_ms(),
            b.resident_weight_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    #[test]
    fn device_gemv_weight_stays_resident() {
        let Some(b) = skip_no_device() else {
            return;
        };
        let (bytes, _) = make_f32_weight(32, 48, 99);
        let x = vec![1.0f32; 48];
        let mut out = vec![0.0f32; 32];
        b.matmul_vec(&bytes, TensorType::F32, 48, 32, &x, &mut out)
            .unwrap();
        let bytes_after_first = b.resident_weight_bytes();
        let launches_after_first = b.gemv_launch_count();
        b.matmul_vec(&bytes, TensorType::F32, 48, 32, &x, &mut out)
            .unwrap();
        assert_eq!(
            b.resident_weight_bytes(),
            bytes_after_first,
            "second call must not re-allocate weight"
        );
        assert_eq!(b.gemv_launch_count(), launches_after_first + 1);
    }

    #[test]
    fn device_gemm_matches_cpu_reference() {
        let Some(b) = skip_no_device() else {
            return;
        };
        // C = A @ B, including awkward dims.
        for &(m, n, k) in &[(2, 3, 4), (7, 5, 9), (16, 16, 16), (31, 17, 13)] {
            let mut a = vec![0.0f32; m * k];
            let mut bb = vec![0.0f32; k * n];
            let mut s = 123u32;
            for v in a.iter_mut().chain(bb.iter_mut()) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = ((s >> 16) as f32 / 65535.0) * 0.5;
            }
            let mut expected = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for t in 0..k {
                        sum += a[i * k + t] * bb[t * n + j];
                    }
                    expected[i * n + j] = sum;
                }
            }
            let mut out = vec![0.0f32; m * n];
            b.gemm_f32(m, n, k, &a, &bb, &mut out).expect("device gemm");
            for i in 0..m * n {
                let tol = 1e-3 * (1.0 + expected[i].abs());
                assert!(
                    (out[i] - expected[i]).abs() < tol,
                    "gemm {m}x{n}x{k} i={i}: {} vs {}",
                    out[i],
                    expected[i]
                );
            }
        }
        assert!(b.gemm_launch_count() >= 4);
        eprintln!(
            "CUDA GEMM ok on {} — launches={}",
            b.device_name(),
            b.gemm_launch_count()
        );
    }

    #[test]
    fn device_gemv_q8_0_dequant_upload() {
        let Some(b) = skip_no_device() else {
            return;
        };
        // Minimal Q8_0: one block (32 elems) as a single output row (n=1, k=32).
        // block: f16 scale + 32 i8 quants. Use crate f16 pack for bit-identical scale.
        let d = 0.1f32;
        let d_bits = crate::quant::f32_to_f16(d);
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&d_bits.to_le_bytes());
        for i in 0..32i8 {
            block.push((i - 16) as u8);
        }
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let expected = cpu_ref(&block, TensorType::Q8_0, 32, 1, &x);
        let mut out = vec![0.0f32; 1];
        b.matmul_vec(&block, TensorType::Q8_0, 32, 1, &x, &mut out)
            .expect("q8 device gemv");
        // Device Q8_0 kernel vs fused CPU Q8 path: allow modest abs tol.
        let tol = 5e-3 * (1.0 + expected[0].abs());
        assert!(
            (out[0] - expected[0]).abs() < tol,
            "q8 device {} vs cpu {} (tol {tol})",
            out[0],
            expected[0]
        );
        // Quantized-resident: raw Q8_0 block (34 bytes), not 128 bytes of f32.
        assert_eq!(b.resident_weight_bytes(), 34);
        assert_eq!(b.quant_matvec_launch_count(), 1);
    }
}
