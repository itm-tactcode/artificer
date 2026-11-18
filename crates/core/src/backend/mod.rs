//! Pluggable compute backends (CPU default; optional CUDA).
//!
//! Trait + [`CpuBackend`] + selection seam + device GEMV/GEMM in `cuda`
//! (feature `cuda`, cuBLAS via `dlopen`). Device attention lives in `cuda_attention`.
//! Forward/SSM dispatch hot matmul through [`Backend::matmul_vec`] (CPU or CUDA).

use std::fmt;

use gguf::TensorType;

use crate::error::{CoreError, Result};

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_attention;
#[cfg(feature = "cuda")]
pub mod cuda_forward;
#[cfg(feature = "cuda")]
pub mod cuda_hostops;
#[cfg(feature = "cuda")]
pub mod cuda_quant;

/// Test-only: a GGUF for the model-backed unit tests at `$ARTIFICER_TEST_MODELS/<file>`. Prints a
/// skip message and returns `None` when the variable is unset/empty or the file is absent.
#[cfg(test)]
pub(crate) fn test_model(file: &str) -> Option<std::path::PathBuf> {
    const VAR: &str = "ARTIFICER_TEST_MODELS";
    let Some(dir) = std::env::var_os(VAR).filter(|d| !d.is_empty()) else {
        eprintln!("skip: {VAR} is unset or empty; set it to the directory holding {file}");
        return None;
    };
    let path = std::path::Path::new(&dir).join(file);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip: {file} not found in {VAR} ({})", path.display());
        None
    }
}

/// Which compute device to use for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    #[default]
    Cpu,
    Cuda,
}

impl BackendKind {
    /// Parse `cpu` / `cuda` (case-insensitive). Empty → CPU.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(CoreError::Backend(format!(
                "unknown backend `{other}` (expected `cpu` or `cuda`)"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Resolve backend from an optional CLI/API value, else `ARTIFICER_BACKEND`, else CPU.
pub fn resolve_backend_kind(cli: Option<&str>) -> Result<BackendKind> {
    if let Some(s) = cli {
        let t = s.trim();
        if !t.is_empty() {
            return BackendKind::parse(t);
        }
    }
    match std::env::var("ARTIFICER_BACKEND") {
        Ok(v) if !v.trim().is_empty() => BackendKind::parse(&v),
        _ => Ok(BackendKind::Cpu),
    }
}

/// `true` when this build includes the optional `cuda` Cargo feature.
pub const fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

/// Compute backend: matmul (and later dequant / attention) over a device.
pub trait Backend: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> BackendKind;

    fn name(&self) -> &'static str;

    /// `out[j] = sum_i W[i, j] * x[i]` (GGML `mul_mat` layout).
    fn matmul_vec(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()>;

    /// Matmul over a sub-slice of a larger tensor (one MoE expert of a 3-D
    /// `ffn_*_exps` tensor). CPU runs [`crate::quant::matmul_vec_no_repack`] so the
    /// pointer-keyed repack caches never see expert slices. MoE is experimental and CPU-only
    /// (CUDA refuses it with or without `--experimental-moe`), so the default (every other
    /// backend) refuses.
    fn matmul_vec_expert(
        &self,
        _weight: &[u8],
        _ty: TensorType,
        _k: usize,
        _n: usize,
        _x: &[f32],
        _out: &mut [f32],
    ) -> Result<()> {
        Err(CoreError::Backend(format!(
            "MoE expert matmul is CPU-only (MoE is experimental); the {} backend does not \
             support it, with or without --experimental-moe (use --backend cpu --experimental-moe)",
            self.name()
        )))
    }

    /// Device / context setup. CPU is a no-op; CUDA creates stream + cuBLAS.
    fn init_device(&mut self) -> Result<()> {
        Ok(())
    }

    /// Check the device can hold `bytes` of weights before any upload (CPU: no-op).
    fn preflight_weights(&self, _bytes: u64) -> Result<()> {
        Ok(())
    }

    /// Downcast to [`cuda::CudaBackend`] when this is the CUDA backend.
    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&cuda::CudaBackend> {
        None
    }
}

/// Default CPU backend — thin adapter over existing [`crate::quant::matmul_vec`].
/// Does not alter SIMD / repack / pool behavior (CPU kernels unchanged).
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu"
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
        crate::quant::matmul_vec(weight, ty, k, n, x, out)
    }

    fn matmul_vec_expert(
        &self,
        weight: &[u8],
        ty: TensorType,
        k: usize,
        n: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        crate::quant::matmul_vec_no_repack(weight, ty, k, n, x, out)
    }
}

/// Build a backend for `kind`. CUDA requires `--features cuda` and a working device/cuBLAS.
pub fn create_backend(kind: BackendKind) -> Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Cpu => {
            let mut b = CpuBackend;
            b.init_device()?;
            Ok(Box::new(b))
        }
        BackendKind::Cuda => {
            if !cuda_feature_enabled() {
                return Err(CoreError::Backend(
                    "CUDA backend requested but this binary was built without the `cuda` feature; \
                     rebuild with `cargo build -p artificer-cli --features cuda` or use `--backend cpu` / `ARTIFICER_BACKEND=cpu`"
                        .into(),
                ));
            }
            #[cfg(feature = "cuda")]
            {
                let mut b = cuda::CudaBackend::try_new()?;
                b.init_device()?;
                Ok(Box::new(b))
            }
            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("cuda_feature_enabled gated above");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_kind() {
        assert_eq!(BackendKind::parse("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::parse("CPU").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::parse("cuda").unwrap(), BackendKind::Cuda);
        assert_eq!(BackendKind::parse("GPU").unwrap(), BackendKind::Cuda);
        assert!(BackendKind::parse("metal").is_err());
    }

    #[test]
    fn create_cpu_backend() {
        let b = create_backend(BackendKind::Cpu).unwrap();
        assert_eq!(b.kind(), BackendKind::Cpu);
        assert_eq!(b.name(), "cpu");
    }

    #[test]
    fn cuda_without_feature_errors() {
        let res = create_backend(BackendKind::Cuda);
        if !cuda_feature_enabled() {
            assert!(res.is_err(), "cuda must fail without `--features cuda`");
            let msg = res.err().unwrap().to_string();
            assert!(
                msg.contains("cuda") || msg.contains("CUDA"),
                "unexpected: {msg}"
            );
            assert!(
                msg.contains("feature") || msg.contains("rebuild"),
                "unexpected: {msg}"
            );
            return;
        }
        // Feature on: Ok when a device+libcudart are present; Err with a clear CUDA message otherwise.
        match res {
            Ok(b) => {
                assert_eq!(b.kind(), BackendKind::Cuda);
                assert_eq!(b.name(), "cuda");
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("CUDA") || msg.contains("cuda") || msg.contains("libcudart"),
                    "unexpected: {msg}"
                );
            }
        }
    }

    #[test]
    fn cpu_matmul_delegates() {
        // 1x1 F32: W=[2], x=[3] → out=[6]
        let weight = 2.0f32.to_le_bytes();
        let x = [3.0f32];
        let mut out = [0.0f32];
        CpuBackend
            .matmul_vec(&weight, TensorType::F32, 1, 1, &x, &mut out)
            .unwrap();
        assert!((out[0] - 6.0).abs() < 1e-6);
    }
}
