//! Per-layer KV cache for autoregressive decoding.
//!
//! Supports f32 (default) and f16 storage to cut KV RAM roughly in half. The CUDA device KV
//! cache defaults to f16 ([`device_kv_elem`], `ARTIFICER_KV_F16=0` for f32).

use crate::config::LlamaConfig;
use crate::quant::{f16_to_f32, f32_to_f16};

/// Storage element width for the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvElem {
    #[default]
    F32,
    /// IEEE-ish binary16 via our portable half conversion.
    F16,
}

impl KvElem {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Element type of the CUDA **device** KV cache. f16 by default: half the
/// KV bytes per decode step and half the device KV VRAM. `ARTIFICER_KV_F16=0` (or `false` /
/// `off`) keeps f32 for A/B; an explicit f16 request (`--kv-f16`) always gives f16. The host
/// [`KvCache`] keeps the requested element type (CPU path unchanged).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn device_kv_elem(requested: KvElem) -> KvElem {
    resolve_device_kv_elem(requested, std::env::var("ARTIFICER_KV_F16").ok().as_deref())
}

/// [`device_kv_elem`] with the env value passed in (tests).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn resolve_device_kv_elem(requested: KvElem, env: Option<&str>) -> KvElem {
    if requested == KvElem::F16 {
        return KvElem::F16;
    }
    match env.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "0" || v == "false" || v == "off" || v == "no" => KvElem::F32,
        _ => KvElem::F16,
    }
}

/// Layer-wise key/value cache: rows at `pos * n_embd_kv`.
pub struct KvCache {
    pub n_layer: usize,
    pub n_embd_kv: usize,
    pub n_ctx: usize,
    pub elem: KvElem,
    /// Current filled length (number of positions written).
    pub len: usize,
    k_f32: Vec<Vec<f32>>,
    v_f32: Vec<Vec<f32>>,
    k_f16: Vec<Vec<u16>>,
    v_f16: Vec<Vec<u16>>,
    /// Device-resident forward state (device KV + activations), created lazily by
    /// the CUDA forward. Always `None` on the CPU backend.
    #[cfg(feature = "cuda")]
    pub(crate) device: Option<Box<crate::backend::cuda_forward::CudaForwardState>>,
}

impl KvCache {
    pub fn new(cfg: &LlamaConfig) -> Self {
        Self::with_elem(cfg, KvElem::F32)
    }

    pub fn with_elem(cfg: &LlamaConfig, elem: KvElem) -> Self {
        let n_layer = cfg.n_layer as usize;
        let n_embd_kv = cfg.n_embd_kv() as usize;
        let n_ctx = cfg.n_ctx as usize;
        let (k_f32, v_f32, k_f16, v_f16) = match elem {
            KvElem::F32 => {
                let k = (0..n_layer)
                    .map(|_| vec![0.0f32; n_ctx * n_embd_kv])
                    .collect();
                let v = (0..n_layer)
                    .map(|_| vec![0.0f32; n_ctx * n_embd_kv])
                    .collect();
                (k, v, Vec::new(), Vec::new())
            }
            KvElem::F16 => {
                let k = (0..n_layer)
                    .map(|_| vec![0u16; n_ctx * n_embd_kv])
                    .collect();
                let v = (0..n_layer)
                    .map(|_| vec![0u16; n_ctx * n_embd_kv])
                    .collect();
                (Vec::new(), Vec::new(), k, v)
            }
        };
        Self {
            n_layer,
            n_embd_kv,
            n_ctx,
            elem,
            len: 0,
            k_f32,
            v_f32,
            k_f16,
            v_f16,
            #[cfg(feature = "cuda")]
            device: None,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        #[cfg(feature = "cuda")]
        if let Some(d) = self.device.as_mut() {
            d.kv.len = 0;
        }
    }

    /// Approximate heap bytes for K+V storage.
    pub fn storage_bytes(&self) -> usize {
        match self.elem {
            KvElem::F32 => self.n_layer * self.n_ctx * self.n_embd_kv * 2 * 4,
            KvElem::F16 => self.n_layer * self.n_ctx * self.n_embd_kv * 2 * 2,
        }
    }

    pub fn write(&mut self, layer: usize, pos: usize, k_row: &[f32], v_row: &[f32]) {
        debug_assert_eq!(k_row.len(), self.n_embd_kv);
        debug_assert_eq!(v_row.len(), self.n_embd_kv);
        debug_assert!(pos < self.n_ctx);
        let off = pos * self.n_embd_kv;
        match self.elem {
            KvElem::F32 => {
                self.k_f32[layer][off..off + self.n_embd_kv].copy_from_slice(k_row);
                self.v_f32[layer][off..off + self.n_embd_kv].copy_from_slice(v_row);
            }
            KvElem::F16 => {
                for i in 0..self.n_embd_kv {
                    self.k_f16[layer][off + i] = f32_to_f16(k_row[i]);
                    self.v_f16[layer][off + i] = f32_to_f16(v_row[i]);
                }
            }
        }
        if pos + 1 > self.len {
            self.len = pos + 1;
        }
    }

    /// Direct f32 view (only valid when `elem == F32`).
    pub fn k_at(&self, layer: usize, pos: usize) -> &[f32] {
        debug_assert_eq!(self.elem, KvElem::F32);
        let off = pos * self.n_embd_kv;
        &self.k_f32[layer][off..off + self.n_embd_kv]
    }

    /// Direct f32 view (only valid when `elem == F32`).
    pub fn v_at(&self, layer: usize, pos: usize) -> &[f32] {
        debug_assert_eq!(self.elem, KvElem::F32);
        let off = pos * self.n_embd_kv;
        &self.v_f32[layer][off..off + self.n_embd_kv]
    }

    /// Fill `out` with K row at (layer, pos). Works for f32 and f16 storage.
    pub fn copy_k_into(&self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.n_embd_kv);
        let off = pos * self.n_embd_kv;
        match self.elem {
            KvElem::F32 => out.copy_from_slice(&self.k_f32[layer][off..off + self.n_embd_kv]),
            KvElem::F16 => {
                for i in 0..self.n_embd_kv {
                    out[i] = f16_to_f32(self.k_f16[layer][off + i]);
                }
            }
        }
    }

    pub fn copy_v_into(&self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.n_embd_kv);
        let off = pos * self.n_embd_kv;
        match self.elem {
            KvElem::F32 => out.copy_from_slice(&self.v_f32[layer][off..off + self.n_embd_kv]),
            KvElem::F16 => {
                for i in 0..self.n_embd_kv {
                    out[i] = f16_to_f32(self.v_f16[layer][off + i]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn device_kv_elem_default_f16_with_f32_knob() {
        use super::{resolve_device_kv_elem as r, KvElem::*};
        assert_eq!(r(F32, None), F16);
        assert_eq!(r(F32, Some("1")), F16);
        assert_eq!(r(F32, Some("0")), F32);
        assert_eq!(r(F32, Some(" off ")), F32);
        assert_eq!(r(F32, Some("false")), F32);
        assert_eq!(r(F16, Some("0")), F16); // explicit --kv-f16 wins
        assert_eq!(r(F16, None), F16);
    }
}
