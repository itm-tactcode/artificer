//! Weight access helpers for the forward pass.

use crate::backend::Backend;
use crate::config::ModelConfig;
use crate::error::{CoreError, Result};
use crate::model::{LlamaModel, WeightTensor};
use crate::ops::normalize;
use crate::profile;
use crate::quant::dequantize;

pub(crate) fn apply_norm(
    model: &LlamaModel,
    cfg: &ModelConfig,
    w: &WeightTensor,
    b: Option<&WeightTensor>,
    x: &[f32],
    out: &mut [f32],
    norm_w: &mut [f32],
    norm_b: &mut [f32],
) -> Result<()> {
    load_f32_vec(model, w, norm_w)?;
    let bias = if let Some(bt) = b {
        load_f32_vec(model, bt, norm_b)?;
        Some(&norm_b[..bt.n_elements()])
    } else {
        None
    };
    {
        let _t = profile::time_norm();
        normalize(
            cfg.norm_kind,
            x,
            &norm_w[..w.n_elements()],
            bias,
            cfg.rms_eps,
            out,
        );
    }
    Ok(())
}

pub(crate) fn load_f32_vec(model: &LlamaModel, w: &WeightTensor, out: &mut [f32]) -> Result<()> {
    let data = model.tensor_bytes(w)?;
    let n = w.n_elements();
    if out.len() < n {
        return Err(CoreError::Shape(format!(
            "norm vec {} buffer {}",
            n,
            out.len()
        )));
    }
    dequantize(data, w.ty, 0, n, &mut out[..n])
}

pub(crate) fn add_bias(model: &LlamaModel, b: &WeightTensor, out: &mut [f32]) -> Result<()> {
    let data = model.tensor_bytes(b)?;
    let n = b.n_elements().min(out.len());
    let mut tmp = vec![0.0f32; n];
    dequantize(data, b.ty, 0, n, &mut tmp)?;
    for i in 0..n {
        out[i] += tmp[i];
    }
    Ok(())
}

/// Matrix×vector via the active [`Backend`] (CPU or CUDA).
pub(crate) fn matmul_wt(
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
            "matmul out buffer {} < n {n} for {}",
            out.len(),
            w.name
        )));
    }
    if x.len() < k {
        return Err(CoreError::Shape(format!(
            "matmul {}: x.len={} < k={k}",
            w.name,
            x.len()
        )));
    }
    // Allow oversized scratch buffers (e.g. hybrid Q dim > n_embd).
    let _role = profile::time_role(&w.name);
    backend.matmul_vec(data, w.ty, k, n, &x[..k], &mut out[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use gguf::TensorType;

    /// Smoke: Backend dispatch path used by forward (compiles with cuda feature off).
    #[test]
    fn matmul_wt_dispatches_cpu_backend() {
        // Minimal fake: we only exercise Backend::matmul_vec shape via CpuBackend
        // directly here (matmul_wt needs a LlamaModel). Same contract as matmul_wt.
        let weight = 4.0f32.to_le_bytes();
        let x = [2.5f32];
        let mut out = [0.0f32];
        CpuBackend
            .matmul_vec(&weight, TensorType::F32, 1, 1, &x, &mut out)
            .unwrap();
        assert!((out[0] - 10.0).abs() < 1e-6);
    }
}
