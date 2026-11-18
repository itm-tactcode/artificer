//! Custom CUDA kernels for **quantized-resident** weights + device softmax.
//!
//! ## Compile approach (chosen: NVRTC at runtime, no nvcc at build time)
//! Kernel source lives in `cuda_kernels.cu` (embedded with `include_str!`). On
//! [`super::Backend::init_device`] we `dlopen` **`libnvrtc.so.13`** and compile it to
//! PTX for the device's compute capability, then `dlopen` the driver
//! **`libcuda.so.1`** and load it with `cuModuleLoadData` / launch with
//! `cuLaunchKernel` on the runtime's stream (primary context). Nothing is
//! linked at build time; the default (non-`cuda`) build never sees any of this.
//!
//! Any NVRTC / driver failure is a [`CoreError::Backend`] with the compiler log —
//! there is **no** CPU fallback.
//!
//! ## Supported on-device weight types
//! Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (raw GGUF blocks,
//! dequantized inside the matvec), F16 / BF16 (raw 2-byte). F32 keeps the cuBLAS path.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};

use gguf::TensorType;

use super::cuda::{dlopen_lib, dlsym_required};
use crate::error::{CoreError, Result};

pub(crate) const KERNEL_SRC: &str = include_str!("cuda_kernels.cu");

/// Warps (rows) per thread block for matvec kernels.
const ROWS_PER_BLOCK: u32 = 4;

type NvrtcResult = c_int;
type CuResult = c_int;

type FnNvrtcCreateProgram = unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
type FnNvrtcCompileProgram =
    unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> NvrtcResult;
type FnNvrtcGetSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> NvrtcResult;
type FnNvrtcGetBuf = unsafe extern "C" fn(*mut c_void, *mut c_char) -> NvrtcResult;
type FnNvrtcDestroyProgram = unsafe extern "C" fn(*mut *mut c_void) -> NvrtcResult;
type FnNvrtcGetErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> CuResult;
type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult;
type FnCuPrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> CuResult;
type FnCuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuResult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type FnCuGetErrorString = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

const CU_ATTR_CC_MAJOR: c_int = 75;
const CU_ATTR_CC_MINOR: c_int = 76;

/// Block layout `(bytes_per_block, elems_per_block)` for device-quantized types.
pub fn device_quant_layout(ty: TensorType) -> Option<(usize, usize)> {
    Some(match ty {
        TensorType::Q4_0 => (18, 32),
        TensorType::Q4_1 => (20, 32),
        TensorType::Q5_0 => (22, 32),
        TensorType::Q5_1 => (24, 32),
        TensorType::Q8_0 => (34, 32),
        TensorType::Q2_K => (84, 256),
        TensorType::Q3_K => (110, 256),
        TensorType::Q4_K => (144, 256),
        TensorType::Q5_K => (176, 256),
        TensorType::Q6_K => (210, 256),
        TensorType::F16 | TensorType::BF16 => (2, 1),
        _ => return None,
    })
}

/// Types whose raw GGUF bytes are kept resident and consumed by a custom kernel.
pub const DEVICE_QUANT_TYPES: &[TensorType] = &[
    TensorType::Q4_0,
    TensorType::Q4_1,
    TensorType::Q5_0,
    TensorType::Q5_1,
    TensorType::Q8_0,
    TensorType::Q2_K,
    TensorType::Q3_K,
    TensorType::Q4_K,
    TensorType::Q5_K,
    TensorType::Q6_K,
    TensorType::F16,
    TensorType::BF16,
];

fn kernel_name(ty: TensorType) -> Option<&'static str> {
    Some(match ty {
        TensorType::Q4_0 => "mv_q4_0",
        TensorType::Q4_1 => "mv_q4_1",
        TensorType::Q5_0 => "mv_q5_0",
        TensorType::Q5_1 => "mv_q5_1",
        TensorType::Q8_0 => "mv_q8_0",
        TensorType::Q2_K => "mv_q2_k",
        TensorType::Q3_K => "mv_q3_k",
        TensorType::Q4_K => "mv_q4_k",
        TensorType::Q5_K => "mv_q5_k",
        TensorType::Q6_K => "mv_q6_k",
        TensorType::F16 | TensorType::BF16 => "mv_f16",
        _ => return None,
    })
}

/// Compiled module + function handles. Lives inside `CudaBackend`.
pub(crate) struct QuantKernels {
    ctx: *mut c_void,
    module: *mut c_void,
    funcs: HashMap<&'static str, usize>,
    softmax: *mut c_void,
    cu_get_fn: FnCuModuleGetFunction,
    /// Lazily resolved kernels by name (device-forward kernels + `ho_*` host-op kernels).
    named: std::sync::Mutex<HashMap<String, usize>>,
    pub(crate) arch: String,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_launch: FnCuLaunchKernel,
    cu_err: Option<FnCuGetErrorString>,
}

// SAFETY: `QuantKernels` holds dlopen'd driver/NVRTC function pointers and CUDA module/function
// handles, which are valid from any thread once the context is bound.
unsafe impl Send for QuantKernels {}
// SAFETY: shared use only reads the pointers; every launch first binds the context with
// cuCtxSetCurrent.
unsafe impl Sync for QuantKernels {}

fn nvrtc_lib() -> Result<*mut c_void> {
    dlopen_lib(&[
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so",
        "/opt/cuda/lib64/libnvrtc.so.13",
        "/opt/cuda/lib64/libnvrtc.so",
        "/usr/local/cuda/lib64/libnvrtc.so",
    ])
    .map_err(|e| {
        CoreError::Backend(format!(
            "{e}; quantized device kernels need NVRTC (libnvrtc)"
        ))
    })
}

fn driver_lib() -> Result<*mut c_void> {
    dlopen_lib(&[
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ])
    .map_err(|e| {
        CoreError::Backend(format!(
            "{e}; quantized device kernels need the driver API (libcuda)"
        ))
    })
}

macro_rules! sym {
    ($h:expr, $name:expr) => {{
        let p = dlsym_required($h, $name)?;
        #[allow(clippy::missing_transmute_annotations)]
        // SAFETY: `p` is a non-null symbol from `dlsym_required`, transmuted to the fn-pointer type
        // the caller annotates for that driver/NVRTC entry point.
        unsafe {
            std::mem::transmute(p)
        }
    }};
}

impl QuantKernels {
    fn cu_check(&self, st: CuResult, what: &str) -> Result<()> {
        cu_check_with(self.cu_err, st, what)
    }

    /// Compile `cuda_kernels.cu` with NVRTC and load it into the primary context of `device`.
    pub(crate) fn compile_and_load(device: i32) -> Result<Self> {
        let drv = driver_lib()?;
        let cu_init: FnCuInit = sym!(drv, "cuInit");
        let cu_device_get: FnCuDeviceGet = sym!(drv, "cuDeviceGet");
        let cu_attr: FnCuDeviceGetAttribute = sym!(drv, "cuDeviceGetAttribute");
        let cu_retain: FnCuPrimaryCtxRetain = sym!(drv, "cuDevicePrimaryCtxRetain");
        let cu_ctx_set_current: FnCuCtxSetCurrent = sym!(drv, "cuCtxSetCurrent");
        let cu_load: FnCuModuleLoadData = sym!(drv, "cuModuleLoadData");
        let cu_get_fn: FnCuModuleGetFunction = sym!(drv, "cuModuleGetFunction");
        let cu_launch: FnCuLaunchKernel = sym!(drv, "cuLaunchKernel");
        let cu_err: Option<FnCuGetErrorString> = dlsym_required(drv, "cuGetErrorString")
            .ok()
            // SAFETY: `p` is the non-null cuGetErrorString symbol, transmuted to its declared C
            // signature.
            .map(|p| unsafe { std::mem::transmute(p) });

        // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
        // signature; out-pointers are live locals.
        cu_check_with(cu_err, unsafe { cu_init(0) }, "cuInit")?;
        let mut dev: c_int = 0;
        cu_check_with(
            cu_err,
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; out-pointers are live locals.
            unsafe { cu_device_get(&mut dev, device) },
            "cuDeviceGet",
        )?;
        let (mut major, mut minor) = (0, 0);
        cu_check_with(
            cu_err,
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; out-pointers are live locals.
            unsafe { cu_attr(&mut major, CU_ATTR_CC_MAJOR, dev) },
            "cuDeviceGetAttribute(major)",
        )?;
        cu_check_with(
            cu_err,
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; out-pointers are live locals.
            unsafe { cu_attr(&mut minor, CU_ATTR_CC_MINOR, dev) },
            "cuDeviceGetAttribute(minor)",
        )?;
        let mut ctx: *mut c_void = std::ptr::null_mut();
        cu_check_with(
            cu_err,
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; out-pointers are live locals.
            unsafe { cu_retain(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        cu_check_with(
            cu_err,
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; out-pointers are live locals.
            unsafe { cu_ctx_set_current(ctx) },
            "cuCtxSetCurrent",
        )?;

        let arch = format!("compute_{major}{minor}");
        let ptx = nvrtc_compile(KERNEL_SRC, &arch)?;

        let mut module: *mut c_void = std::ptr::null_mut();
        cu_check_with(
            cu_err,
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; `ptx` is the NUL-terminated PTX image returned by NVRTC and outlives the
            // call.
            unsafe { cu_load(&mut module, ptx.as_ptr() as *const c_void) },
            "cuModuleLoadData (PTX JIT)",
        )?;
        let get = |name: &str| -> Result<*mut c_void> {
            let c = CString::new(name).unwrap();
            let mut f: *mut c_void = std::ptr::null_mut();
            cu_check_with(
                cu_err,
                // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the
                // declared C signature; `c` is a NUL-terminated CString and the module is loaded.
                unsafe { cu_get_fn(&mut f, module, c.as_ptr()) },
                &format!("cuModuleGetFunction({name})"),
            )?;
            Ok(f)
        };
        let mut funcs = HashMap::new();
        for ty in DEVICE_QUANT_TYPES {
            let name = kernel_name(*ty).unwrap();
            if !funcs.contains_key(name) {
                funcs.insert(name, get(name)? as usize);
            }
        }
        let softmax = get("softmax_rows")?;
        Ok(Self {
            ctx,
            module,
            funcs,
            softmax,
            cu_get_fn,
            named: std::sync::Mutex::new(HashMap::new()),
            arch,
            cu_ctx_set_current,
            cu_launch,
            cu_err,
        })
    }

    /// Resolve any `extern "C"` kernel of the shared NVRTC module by name (cached).
    pub(crate) fn function(&self, name: &str) -> Result<*mut c_void> {
        let mut map = self
            .named
            .lock()
            .map_err(|_| CoreError::Backend("CUDA kernel table lock poisoned".into()))?;
        if let Some(&f) = map.get(name) {
            return Ok(f as *mut c_void);
        }
        let c = CString::new(name).map_err(|e| CoreError::Backend(e.to_string()))?;
        let mut f: *mut c_void = std::ptr::null_mut();
        self.cu_check(
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; `c` is a NUL-terminated CString and the module is loaded.
            unsafe { (self.cu_get_fn)(&mut f, self.module, c.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        map.insert(name.to_string(), f as usize);
        Ok(f)
    }

    /// Generic async launch on `stream` with a prepared kernel-parameter array.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_raw(
        &self,
        f: *mut c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        stream: *mut c_void,
        params: &mut [*mut c_void],
        what: &str,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: `f` is a function from our loaded module, the context was bound just above, and
        // `params` points to one pointer per kernel argument, each to a live local of the matching
        // type.
        let st = unsafe {
            (self.cu_launch)(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shmem,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.cu_check(st, &format!("cuLaunchKernel({what})"))
    }

    fn bind(&self) -> Result<()> {
        self.cu_check(
            // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
            // signature; `self.ctx` is the retained primary context.
            unsafe { (self.cu_ctx_set_current)(self.ctx) },
            "cuCtxSetCurrent",
        )
    }

    /// `y[n] = W(n×k, raw `ty` blocks) · x[k]`, all device pointers, async on `stream`.
    pub(crate) fn launch_matvec(
        &self,
        ty: TensorType,
        d_w: *mut c_void,
        d_x: *mut c_void,
        d_y: *mut c_void,
        k: usize,
        n: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let name = kernel_name(ty).ok_or_else(|| {
            CoreError::Backend(format!("CUDA: no device matvec kernel for {ty:?}"))
        })?;
        let f = *self
            .funcs
            .get(name)
            .ok_or_else(|| CoreError::Backend(format!("CUDA: kernel `{name}` not loaded")))?
            as *mut c_void;
        self.bind()?;
        let mut w = d_w;
        let mut x = d_x;
        let mut y = d_y;
        let mut kk = k as c_int;
        let mut nn = n as c_int;
        let mut bf: c_int = (ty == TensorType::BF16) as c_int;
        let mut params: Vec<*mut c_void> = vec![
            &mut w as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
        ];
        if name == "mv_f16" {
            params.push(&mut bf as *mut _ as *mut c_void);
        }
        let grid = (n as u32).div_ceil(ROWS_PER_BLOCK);
        // SAFETY: the kernel comes from our module, the context is bound, and every entry in the
        // params array points to a live local of the kernel's argument type.
        let st = unsafe {
            (self.cu_launch)(
                f,
                grid,
                1,
                1,
                32 * ROWS_PER_BLOCK,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.cu_check(st, &format!("cuLaunchKernel({name}, n={n}, k={k})"))
    }

    /// In-place stable softmax over `rows` rows of length `n` at `d_s` (after scale / softcap).
    pub(crate) fn launch_softmax(
        &self,
        d_s: *mut c_void,
        rows: usize,
        n: usize,
        scale: f32,
        softcap: f32,
        stream: *mut c_void,
    ) -> Result<()> {
        self.bind()?;
        let mut s = d_s;
        let mut nn = n as c_int;
        let mut sc = scale;
        let mut cap = softcap;
        let mut params: [*mut c_void; 4] = [
            &mut s as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
        ];
        let threads = if n >= 256 {
            256
        } else {
            32 * (n as u32).div_ceil(32).max(1)
        };
        // SAFETY: the kernel comes from our module, the context is bound, and every entry in the
        // params array points to a live local of the kernel's argument type.
        let st = unsafe {
            (self.cu_launch)(
                self.softmax,
                rows as u32,
                1,
                1,
                threads,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.cu_check(
            st,
            &format!("cuLaunchKernel(softmax_rows, rows={rows}, n={n})"),
        )
    }
}

fn cu_check_with(err: Option<FnCuGetErrorString>, st: CuResult, what: &str) -> Result<()> {
    if st == 0 {
        return Ok(());
    }
    let mut msg = String::new();
    if let Some(f) = err {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: the driver-API pointer was resolved by dlsym from libcuda with the declared C
        // signature; the returned message is a static C string, null-checked before reading.
        unsafe {
            f(st, &mut p);
            if !p.is_null() {
                msg = CStr::from_ptr(p).to_string_lossy().into_owned();
            }
        }
    }
    Err(CoreError::Backend(format!(
        "CUDA driver {what} failed (CUresult {st}: {msg}) — no CPU fallback"
    )))
}

fn nvrtc_compile(src: &str, arch: &str) -> Result<Vec<u8>> {
    let h = nvrtc_lib()?;
    let create: FnNvrtcCreateProgram = sym!(h, "nvrtcCreateProgram");
    let compile: FnNvrtcCompileProgram = sym!(h, "nvrtcCompileProgram");
    let log_size: FnNvrtcGetSize = sym!(h, "nvrtcGetProgramLogSize");
    let log: FnNvrtcGetBuf = sym!(h, "nvrtcGetProgramLog");
    let ptx_size: FnNvrtcGetSize = sym!(h, "nvrtcGetPTXSize");
    let ptx: FnNvrtcGetBuf = sym!(h, "nvrtcGetPTX");
    let destroy: FnNvrtcDestroyProgram = sym!(h, "nvrtcDestroyProgram");
    let errstr: FnNvrtcGetErrorString = sym!(h, "nvrtcGetErrorString");
    // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C signature;
    // the returned string is static and null-checked.
    let es = |st: NvrtcResult| unsafe {
        let p = errstr(st);
        if p.is_null() {
            format!("{st}")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };

    let csrc = CString::new(src).map_err(|e| CoreError::Backend(e.to_string()))?;
    let cname = CString::new("artificer_kernels.cu").unwrap();
    let mut prog: *mut c_void = std::ptr::null_mut();
    // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C signature;
    // the source and name CStrings outlive the call and `prog` is a live out-pointer.
    let st = unsafe {
        create(
            &mut prog,
            csrc.as_ptr(),
            cname.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if st != 0 {
        return Err(CoreError::Backend(format!(
            "NVRTC nvrtcCreateProgram failed: {}",
            es(st)
        )));
    }
    let opt_arch = CString::new(format!("--gpu-architecture={arch}")).unwrap();
    let opt_std = CString::new("--std=c++17").unwrap();
    let opts = [opt_arch.as_ptr(), opt_std.as_ptr()];
    // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C signature;
    // `opts` holds NUL-terminated CStrings that outlive the call.
    let st = unsafe { compile(prog, opts.len() as c_int, opts.as_ptr()) };
    if st != 0 {
        let mut n = 0usize;
        let mut text = String::new();
        // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C
        // signature; the log buffer is allocated at the size NVRTC reported.
        unsafe {
            if log_size(prog, &mut n) == 0 && n > 1 {
                let mut buf = vec![0u8; n];
                log(prog, buf.as_mut_ptr() as *mut c_char);
                text = String::from_utf8_lossy(&buf[..n - 1]).into_owned();
            }
            destroy(&mut prog);
        }
        return Err(CoreError::Backend(format!(
            "NVRTC compile of Artificer CUDA kernels failed ({}) for {arch}:\n{text}",
            es(st)
        )));
    }
    let mut n = 0usize;
    // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C signature;
    // `prog` is a live program and the PTX buffer is the size NVRTC reported.
    let st = unsafe { ptx_size(prog, &mut n) };
    if st != 0 {
        // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C
        // signature; `prog` is live and destroyed exactly once on each path.
        unsafe { destroy(&mut prog) };
        return Err(CoreError::Backend(format!(
            "NVRTC nvrtcGetPTXSize failed: {}",
            es(st)
        )));
    }
    let mut buf = vec![0u8; n];
    // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C signature;
    // `prog` is a live program and the PTX buffer is the size NVRTC reported.
    let st = unsafe { ptx(prog, buf.as_mut_ptr() as *mut c_char) };
    // SAFETY: the NVRTC pointer was resolved by dlsym from libnvrtc with the declared C signature;
    // `prog` is live and destroyed exactly once on each path.
    unsafe { destroy(&mut prog) };
    if st != 0 {
        return Err(CoreError::Backend(format!(
            "NVRTC nvrtcGetPTX failed: {}",
            es(st)
        )));
    }
    Ok(buf) // NUL-terminated PTX text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;
    use crate::backend::Backend;
    use crate::quant::{dequantize, f32_to_f16};

    fn dev() -> Option<CudaBackend> {
        let mut b = CudaBackend::try_new().ok()?;
        if let Err(e) = b.init_device() {
            eprintln!("CUDA init_device failed: {e}");
            return None;
        }
        Some(b)
    }

    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0 >> 8
        }
        fn f(&mut self) -> f32 {
            (self.next() & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0
        }
    }

    /// Byte offsets of f16 scale fields inside one block (overwritten with sane values).
    fn f16_fields(ty: TensorType) -> &'static [usize] {
        match ty {
            TensorType::Q4_0 | TensorType::Q5_0 | TensorType::Q8_0 => &[0],
            TensorType::Q4_1 | TensorType::Q5_1 | TensorType::Q4_K | TensorType::Q5_K => &[0, 2],
            TensorType::Q2_K => &[80, 82],
            TensorType::Q3_K => &[108],
            TensorType::Q6_K => &[208],
            _ => &[],
        }
    }

    fn random_weight(ty: TensorType, k: usize, n: usize, seed: u32) -> Vec<u8> {
        let (bb, be) = device_quant_layout(ty).unwrap();
        let nblk = k / be * n;
        let mut r = Lcg(seed);
        let mut w: Vec<u8> = (0..nblk * bb).map(|_| (r.next() & 0xFF) as u8).collect();
        if matches!(ty, TensorType::F16 | TensorType::BF16) {
            for i in 0..k * n {
                let v = r.f();
                let h = if ty == TensorType::F16 {
                    f32_to_f16(v)
                } else {
                    (v.to_bits() >> 16) as u16
                };
                w[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
            }
            return w;
        }
        for b in 0..nblk {
            for &off in f16_fields(ty) {
                let v = 0.002 + (r.next() % 1000) as f32 * 0.00005;
                let h = f32_to_f16(v);
                w[b * bb + off..b * bb + off + 2].copy_from_slice(&h.to_le_bytes());
            }
        }
        w
    }

    fn host_ref(w: &[u8], ty: TensorType, k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let mut full = vec![0.0f32; k * n];
        dequantize(w, ty, 0, k * n, &mut full).unwrap();
        (0..n)
            .map(|j| {
                let mut s = 0.0f64;
                for i in 0..k {
                    s += full[j * k + i] as f64 * x[i] as f64;
                }
                s as f32
            })
            .collect()
    }

    #[test]
    fn device_quant_matvec_matches_host_dequant_all_types() {
        let Some(b) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        // f32-activation matvec kernels (`mv_*`, still selectable via ARTIFICER_CUDA_MMV=old).
        b.set_mmvq(false);
        for (ti, &ty) in DEVICE_QUANT_TYPES.iter().enumerate() {
            let (_, be) = device_quant_layout(ty).unwrap();
            let kb = be.max(32);
            for &(k, n) in &[(kb, 1usize), (kb * 4, 7), (kb * 12, 33)] {
                let w = random_weight(ty, k, n, 17 + ti as u32 * 31 + k as u32);
                let mut r = Lcg(99 + n as u32);
                let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
                let expect = host_ref(&w, ty, k, n, &x);
                let before = b.quant_matvec_launch_count();
                let res_before = b.resident_weight_bytes();
                let mut out = vec![0.0f32; n];
                b.matmul_vec(&w, ty, k, n, &x, &mut out)
                    .expect("device quant matvec");
                assert_eq!(
                    b.quant_matvec_launch_count(),
                    before + 1,
                    "{ty:?}: quant kernel did not run"
                );
                assert_eq!(
                    b.resident_weight_bytes() - res_before,
                    w.len() as u64,
                    "{ty:?}: resident bytes must equal raw quantized size"
                );
                let scale = expect.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                for j in 0..n {
                    let tol = 1e-4 * scale + 1e-4 * (k as f32).sqrt();
                    assert!(
                        (out[j] - expect[j]).abs() <= tol,
                        "{ty:?} {n}x{k} row {j}: device {} vs host {} (tol {tol})",
                        out[j],
                        expect[j]
                    );
                }
            }
        }
    }

    /// Host emulation of the device Q8 activation quantization (per 32: d = amax/127, q = rn(x/d)).
    fn q8_emulate(x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for (bi, blk) in x.chunks(32).enumerate() {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            for (i, v) in blk.iter().enumerate() {
                let q = if d == 0.0 {
                    0.0
                } else {
                    (v / d).round_ties_even()
                };
                out[bi * 32 + i] = q * d;
            }
        }
        out
    }

    #[test]
    fn device_mmvq_matches_cpu_reference_per_type() {
        let Some(b) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        b.set_mmvq(true);
        for (ti, &ty) in [
            TensorType::Q5_0,
            TensorType::Q4_K,
            TensorType::Q8_0,
            TensorType::Q6_K,
        ]
        .iter()
        .enumerate()
        {
            let (_, be) = device_quant_layout(ty).unwrap();
            let kb = be.max(32);
            let mut shapes = vec![(kb, 1usize), (kb * 4, 7), (kb * 12, 33), (kb * 49, 9)];
            if kb == 32 {
                shapes.push((3136, 5)); // 98 sub-blocks: lane tail
            }
            for &(k, n) in &shapes {
                let w = random_weight(ty, k, n, 5 + ti as u32 * 13 + k as u32);
                let mut r = Lcg(41 + n as u32 + k as u32);
                let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
                let before = b.mmvq_launch_count();
                let mut out = vec![0.0f32; n];
                b.matmul_vec(&w, ty, k, n, &x, &mut out)
                    .expect("mmvq matvec");
                assert_eq!(
                    b.mmvq_launch_count(),
                    before + 1,
                    "{ty:?}: mmvq kernel did not run"
                );
                // (1) Tight: device == f64 dot of dequantized W with the emulated Q8 activation.
                let xq = q8_emulate(&x);
                let tight = host_ref(&w, ty, k, n, &xq);
                // (2) Rigorous bound vs the exact f32 activation: |Σ w (x - x')| <= Σ |w| d/2.
                let exact = host_ref(&w, ty, k, n, &x);
                let mut full = vec![0.0f32; k * n];
                dequantize(&w, ty, 0, k * n, &mut full).unwrap();
                // (3) CPU reference dot (what the CPU backend actually computes).
                let mut cpu = vec![0.0f32; n];
                crate::quant::matmul_vec(&w, ty, k, n, &x, &mut cpu).unwrap();
                let scale = exact.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                let mut max_cpu = 0.0f32;
                for j in 0..n {
                    let tol = 1e-4 * scale + 1e-4 * (k as f32).sqrt();
                    assert!(
                        (out[j] - tight[j]).abs() <= tol,
                        "{ty:?} {n}x{k} row {j}: device {} vs Q8-emulated host {} (tol {tol})",
                        out[j],
                        tight[j]
                    );
                    let mut bound = 0.0f64;
                    for (bi, blk) in x.chunks(32).enumerate() {
                        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
                        let wsum: f64 = (0..32)
                            .map(|i| full[j * k + bi * 32 + i].abs() as f64)
                            .sum();
                        bound += wsum * amax / 254.0;
                    }
                    let tol2 = bound as f32 * 1.01 + tol;
                    assert!(
                        (out[j] - exact[j]).abs() <= tol2,
                        "{ty:?} {n}x{k} row {j}: device {} vs exact {} (Q8 bound {tol2})",
                        out[j],
                        exact[j]
                    );
                    let cpu_tol = 2.0 * tol2;
                    assert!(
                        (out[j] - cpu[j]).abs() <= cpu_tol,
                        "{ty:?} {n}x{k} row {j}: device {} vs CPU matmul_vec {} (tol {cpu_tol})",
                        out[j],
                        cpu[j]
                    );
                    max_cpu = max_cpu.max((out[j] - cpu[j]).abs() / scale);
                }
                eprintln!("mmvq {ty:?} {n}x{k}: max |dev-cpu|/scale = {max_cpu:.2e}");
            }
        }
    }

    #[test]
    fn device_matvec_multi_quantizes_shared_input_once() {
        let Some(b) = dev() else {
            return;
        };
        b.set_mmvq(true);
        let k = 512;
        let w1 = random_weight(TensorType::Q5_0, k, 6, 3);
        let w2 = random_weight(TensorType::Q4_K, k, 4, 4);
        let mut r = Lcg(8);
        let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
        let dx = b.alloc_f32(k).unwrap();
        b.upload_f32(&x, dx.ptr()).unwrap();
        let y1 = b.alloc_f32(6).unwrap();
        let y2 = b.alloc_f32(4).unwrap();
        let (aq, mv) = (b.act_quant_launch_count(), b.mmvq_launch_count());
        let jobs = [
            crate::backend::cuda::MatvecJob {
                weight: &w1,
                ty: TensorType::Q5_0,
                n: 6,
                y: y1.ptr(),
            },
            crate::backend::cuda::MatvecJob {
                weight: &w2,
                ty: TensorType::Q4_K,
                n: 4,
                y: y2.ptr(),
            },
        ];
        b.matvec_dev_multi(dx.ptr(), k, &jobs).unwrap();
        assert_eq!(
            b.act_quant_launch_count(),
            aq + 1,
            "shared input must be quantized once"
        );
        assert_eq!(b.mmvq_launch_count(), mv + 2);
        let mut o1 = vec![0.0f32; 6];
        let mut o2 = vec![0.0f32; 4];
        b.download_f32(y1.ptr(), &mut o1).unwrap();
        b.download_f32(y2.ptr(), &mut o2).unwrap();
        let mut e1 = vec![0.0f32; 6];
        let mut e2 = vec![0.0f32; 4];
        b.matmul_vec(&w1, TensorType::Q5_0, k, 6, &x, &mut e1)
            .unwrap();
        b.matmul_vec(&w2, TensorType::Q4_K, k, 4, &x, &mut e2)
            .unwrap();
        assert_eq!(o1, e1);
        assert_eq!(o2, e2);
    }

    /// Batched-prefill GEMM (`gemm_dev`) for every quant type, both small-batch paths
    /// (multi-column mmvq at rows <= 32, f16 dequant tiles + GemmEx above / with mmvq off)
    /// vs an f64 host reference. Bound: Q8 activation error (mmvq) + f16 weight/activation
    /// rounding (GEMM), both <= sum|w| * amax_block / 254 + 2e-3 * sum|w x|.
    #[test]
    fn device_gemm_prefill_matches_host_all_types() {
        let Some(b) = dev() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let mut tys: Vec<TensorType> = DEVICE_QUANT_TYPES.to_vec();
        tys.push(TensorType::F32);
        for (ti, &ty) in tys.iter().enumerate() {
            let kb = if ty == TensorType::F32 {
                32
            } else {
                device_quant_layout(ty).unwrap().1.max(32)
            };
            for &mmvq in &[true, false] {
                b.set_mmvq(mmvq);
                // rows <= 32 -> multi-column mmvq when mmvq is on; else dequant tiles + GemmEx.
                b.set_prefill_mmvq_max_rows(32);
                for &(k, n, rows) in &[(kb * 2, 5usize, 3usize), (kb * 4, 37, 8), (kb * 3, 19, 40)]
                {
                    let w = if ty == TensorType::F32 {
                        let mut r = Lcg(7 + n as u32);
                        (0..k * n)
                            .flat_map(|_| r.f().to_le_bytes())
                            .collect::<Vec<u8>>()
                    } else {
                        random_weight(ty, k, n, 3 + ti as u32 * 17 + k as u32)
                    };
                    let mut r = Lcg(11 + rows as u32 + k as u32);
                    let x: Vec<f32> = (0..rows * k).map(|_| r.f() * 3.0).collect();
                    let dx = b.alloc_f32(rows * k).unwrap();
                    b.upload_f32(&x, dx.ptr()).unwrap();
                    let dy = b.alloc_f32(rows * n).unwrap();
                    b.gemm_dev(&w, ty, k, n, dx.ptr(), rows, dy.ptr())
                        .expect("gemm_dev");
                    let mut y = vec![0.0f32; rows * n];
                    b.download_f32(dy.ptr(), &mut y).unwrap();
                    let mut full = vec![0.0f32; k * n];
                    dequantize(&w, ty, 0, k * n, &mut full).unwrap();
                    for rr in 0..rows {
                        let xr = &x[rr * k..(rr + 1) * k];
                        for j in 0..n {
                            let wr = &full[j * k..(j + 1) * k];
                            let exact: f64 =
                                wr.iter().zip(xr).map(|(a, c)| *a as f64 * *c as f64).sum();
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
                            let tol = (q8 * 1.01 + 2e-3 * mag + 1e-4) as f32;
                            let got = y[rr * n + j];
                            assert!(
                                (got - exact as f32).abs() <= tol,
                                "{ty:?} mmvq={mmvq} {rows}x{k}->{n} row {rr} col {j}: device {got} vs host {exact} (tol {tol})"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            b.gemm_ex_launch_count() > 0 || !b.has_gemm_ex(),
            "GemmEx path never ran"
        );
    }

    /// Large activations (|x| > f16 range) survive the per-row f16 scaling of the GEMM path.
    #[test]
    fn device_gemm_prefill_large_activations() {
        let Some(b) = dev() else {
            return;
        };
        b.set_mmvq(false);
        let (k, n, rows) = (256usize, 16usize, 4usize);
        let w = random_weight(TensorType::Q8_0, k, n, 5);
        let mut r = Lcg(3);
        let x: Vec<f32> = (0..rows * k)
            .map(|i| r.f() * if i / k == 2 { 3.0e5 } else { 1.0 })
            .collect();
        let dx = b.alloc_f32(rows * k).unwrap();
        b.upload_f32(&x, dx.ptr()).unwrap();
        let dy = b.alloc_f32(rows * n).unwrap();
        b.gemm_dev(&w, TensorType::Q8_0, k, n, dx.ptr(), rows, dy.ptr())
            .unwrap();
        let mut y = vec![0.0f32; rows * n];
        b.download_f32(dy.ptr(), &mut y).unwrap();
        let mut full = vec![0.0f32; k * n];
        dequantize(&w, TensorType::Q8_0, 0, k * n, &mut full).unwrap();
        for rr in 0..rows {
            for j in 0..n {
                let (mut e, mut m) = (0.0f64, 0.0f64);
                for i in 0..k {
                    let p = full[j * k + i] as f64 * x[rr * k + i] as f64;
                    e += p;
                    m += p.abs();
                }
                let got = y[rr * n + j] as f64;
                assert!(
                    got.is_finite() && (got - e).abs() <= 2e-3 * m + 1e-4,
                    "row {rr} col {j}: {got} vs {e}"
                );
            }
        }
    }

    #[test]
    fn device_softmax_matches_host() {
        let Some(b) = dev() else {
            return;
        };
        for &(n, scale, cap) in &[
            (1usize, 1.0f32, 0.0f32),
            (7, 0.125, 0.0),
            (300, 0.5, 0.0),
            (1029, 1.0, 30.0),
        ] {
            let mut r = Lcg(n as u32 * 7);
            let raw: Vec<f32> = (0..n).map(|_| r.f() * 40.0).collect();
            let mut host: Vec<f32> = raw.iter().map(|v| v * scale).collect();
            crate::ops::softcap_inplace(&mut host, cap);
            crate::ops::softmax_inplace(&mut host);
            let mut d = raw.clone();
            let before = b.softmax_launch_count();
            b.softmax_device(&mut d, 1, n, scale, cap).unwrap();
            assert_eq!(b.softmax_launch_count(), before + 1);
            let sum: f32 = d.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "n={n} sum {sum}");
            for i in 0..n {
                assert!(
                    (d[i] - host[i]).abs() < 1e-5 + 1e-4 * host[i],
                    "n={n} i={i}: {} vs {}",
                    d[i],
                    host[i]
                );
            }
        }
        // Multi-row + huge logits (stability: no inf/NaN).
        let mut d = vec![1000.0f32, 999.0, -1000.0, 0.0, 0.0, 0.0];
        b.softmax_device(&mut d, 2, 3, 1.0, 0.0).unwrap();
        assert!(d.iter().all(|v| v.is_finite()));
        assert!((d[0] - 0.731_058_6).abs() < 1e-5);
        assert!((d[3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn oom_preflight_errors_clearly() {
        let Some(b) = dev() else {
            return;
        };
        let err = b.preflight_weights(1u64 << 50).unwrap_err().to_string();
        assert!(
            err.contains("out of device memory") && err.contains("does not fit"),
            "{err}"
        );
        b.preflight_weights(1 << 20).unwrap();
    }
}
