//! Bound model weights over a memory-mapped GGUF file.

use gguf::{GgufModel, TensorType};

use crate::config::{LayerType, ModelConfig};
use crate::error::{CoreError, Result};
use crate::tokenizer::Tokenizer;

/// Reference to a tensor payload inside the owned GGUF mapping.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub ty: TensorType,
}

impl WeightTensor {
    pub fn k(&self) -> usize {
        self.shape.first().copied().unwrap_or(0) as usize
    }
    pub fn n(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1) as usize
    }
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
}

/// MoE FFN weights for one layer (llama.cpp `LLM_TENSOR_FFN_*_EXPS`).
///
/// `gate_exps` / `up_exps`: `{n_embd, n_ff_exp, n_expert}`; `down_exps`:
/// `{n_ff_exp, n_embd, n_expert}`; `gate_inp` (router): `{n_embd, n_expert}`. Expert `e` of a
/// 3-D tensor is the contiguous byte range `e * expert_bytes .. (e + 1) * expert_bytes`
/// (see [`expert_slice`]).
#[derive(Debug, Clone)]
pub struct MoeWeights {
    pub gate_inp: WeightTensor,
    pub gate_exps: WeightTensor,
    pub up_exps: WeightTensor,
    pub down_exps: WeightTensor,
    /// Expert FFN width (`n_ff_exp`), from the tensor shape.
    pub n_ff_exp: usize,
    /// Sigmoid-gated shared expert (`qwen2moe`); `None` for OLMoE.
    pub shexp: Option<SharedExpertWeights>,
}

/// Shared expert of one MoE layer (llama.cpp `LLM_TENSOR_FFN_*_SHEXP`, `qwen2moe`): a dense
/// SwiGLU FFN run on every token, scaled by `sigmoid(gate_inp_shexp · x)` and added to the
/// routed-expert sum.
///
/// `gate_shexp` / `up_shexp`: `{n_embd, n_ff_shexp}`; `down_shexp`: `{n_ff_shexp, n_embd}`;
/// `gate_inp_shexp`: `{n_embd}` (one gate logit per token). These are ordinary 2-D matrices,
/// so they go through the normal matmul path (repack caches included), unlike the 3-D experts.
#[derive(Debug, Clone)]
pub struct SharedExpertWeights {
    pub gate_inp_shexp: WeightTensor,
    pub gate_shexp: WeightTensor,
    pub up_shexp: WeightTensor,
    pub down_shexp: WeightTensor,
    /// Shared-expert FFN width (`MoeConfig::n_ff_shexp`; the tensor shapes are checked against it).
    pub n_ff_shexp: usize,
}

/// Bytes of one 2-D `{k, n}` expert matrix of `ty` (rows must be whole quant blocks).
pub fn expert_bytes(ty: TensorType, k: usize, n: usize) -> Result<usize> {
    let bs = ty.block_size() as usize;
    let ts = ty.type_size() as usize;
    if bs == 0 || ts == 0 || !k.is_multiple_of(bs) {
        return Err(CoreError::Shape(format!(
            "expert matrix k={k} is not a whole number of {ty:?} blocks (block {bs})"
        )));
    }
    Ok((k / bs) * ts * n)
}

/// Byte slice of expert `e` inside a 3-D `{k, n, n_expert}` tensor payload `data`.
pub fn expert_slice(
    data: &[u8],
    ty: TensorType,
    k: usize,
    n: usize,
    n_expert: usize,
    e: usize,
) -> Result<&[u8]> {
    let per = expert_bytes(ty, k, n)?;
    if e >= n_expert || data.len() < per * n_expert {
        return Err(CoreError::Shape(format!(
            "expert {e} of {n_expert}: tensor has {} bytes, need {} ({per} per expert)",
            data.len(),
            per * n_expert
        )));
    }
    Ok(&data[e * per..(e + 1) * per])
}

/// Per-layer weights; fields used depend on [`LayerWeights::layer_type`].
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub layer_type: LayerType,
    pub attn_norm: WeightTensor,
    pub attn_norm_b: Option<WeightTensor>,
    // Attention (Dense / Attention)
    pub attn_q: Option<WeightTensor>,
    pub attn_k: Option<WeightTensor>,
    pub attn_v: Option<WeightTensor>,
    pub attn_output: Option<WeightTensor>,
    pub attn_q_b: Option<WeightTensor>,
    pub attn_k_b: Option<WeightTensor>,
    pub attn_v_b: Option<WeightTensor>,
    pub attn_output_b: Option<WeightTensor>,
    /// Optional q/k RMSNorm applied after the bias add, before RoPE (`blk.{i}.attn_q_norm`,
    /// `attn_k_norm`): `{head_dim}` = per-head (Qwen3), `{n_head*head_dim}` = full-width (OLMoE).
    pub attn_q_norm: Option<WeightTensor>,
    pub attn_k_norm: Option<WeightTensor>,
    // FFN (Dense / Ffn)
    pub ffn_norm: Option<WeightTensor>,
    pub ffn_norm_b: Option<WeightTensor>,
    pub ffn_gate: Option<WeightTensor>,
    pub ffn_up: Option<WeightTensor>,
    pub ffn_down: Option<WeightTensor>,
    pub ffn_up_b: Option<WeightTensor>,
    pub ffn_down_b: Option<WeightTensor>,
    /// MoE FFN (OLMoE, Qwen2-MoE): router + 3-D expert tensors (+ shared expert for
    /// Qwen2-MoE). When set, `ffn_gate/up/down` are `None`.
    pub moe: Option<MoeWeights>,
    // Mamba-2 SSM
    pub ssm_in: Option<WeightTensor>,
    pub ssm_conv1d: Option<WeightTensor>,
    pub ssm_conv1d_b: Option<WeightTensor>,
    pub ssm_dt_b: Option<WeightTensor>,
    pub ssm_a: Option<WeightTensor>,
    pub ssm_d: Option<WeightTensor>,
    pub ssm_norm: Option<WeightTensor>,
    pub ssm_out: Option<WeightTensor>,
}

/// Fully loaded model: GGUF storage + config + tokenizer + weight map.
pub struct LlamaModel {
    pub gguf: GgufModel,
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub token_embd: WeightTensor,
    pub output_norm: WeightTensor,
    pub output_norm_b: Option<WeightTensor>,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    /// Why the CUDA path must not run this model (NEOX RoPE, q/k-norm or MoE layers are
    /// CPU-only). `None` = CUDA allowed. Computed once at load.
    pub cuda_refusal: Option<String>,
}

impl LlamaModel {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufModel::load(path)?;
        Self::from_gguf(gguf)
    }

    pub fn from_gguf(gguf: GgufModel) -> Result<Self> {
        let config = ModelConfig::from_gguf(&gguf)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let token_embd = require_tensor(&gguf, "token_embd.weight")?;
        let output_norm = require_tensor(&gguf, "output_norm.weight")?;
        let output_norm_b = optional_tensor(&gguf, "output_norm.bias");
        let output = if gguf.find_tensor("output.weight").is_some() {
            require_tensor(&gguf, "output.weight")?
        } else {
            token_embd.clone()
        };

        let mut layers = Vec::with_capacity(config.n_layer as usize);
        for i in 0..config.n_layer {
            let lt = config
                .layer_types
                .get(i as usize)
                .copied()
                .unwrap_or(LayerType::Dense);
            let attn_norm = require_tensor(&gguf, &format!("blk.{i}.attn_norm.weight"))?;
            let attn_norm_b = optional_tensor(&gguf, &format!("blk.{i}.attn_norm.bias"));

            let layer = match lt {
                LayerType::Dense => {
                    let moe = load_moe_weights(&gguf, &config, i)?;
                    let ffn_gate = optional_tensor(&gguf, &format!("blk.{i}.ffn_gate.weight"));
                    if moe.is_none()
                        && config.ffn_activation != crate::config::FfnActivation::ReluSqrSequential
                        && ffn_gate.is_none()
                    {
                        return Err(CoreError::MissingTensor(format!("blk.{i}.ffn_gate.weight")));
                    }
                    LayerWeights {
                        layer_type: lt,
                        attn_norm,
                        attn_norm_b,
                        attn_q: Some(require_tensor(&gguf, &format!("blk.{i}.attn_q.weight"))?),
                        attn_k: Some(require_tensor(&gguf, &format!("blk.{i}.attn_k.weight"))?),
                        attn_v: Some(require_tensor(&gguf, &format!("blk.{i}.attn_v.weight"))?),
                        attn_output: Some(require_tensor(
                            &gguf,
                            &format!("blk.{i}.attn_output.weight"),
                        )?),
                        attn_q_b: optional_tensor(&gguf, &format!("blk.{i}.attn_q.bias")),
                        attn_k_b: optional_tensor(&gguf, &format!("blk.{i}.attn_k.bias")),
                        attn_v_b: optional_tensor(&gguf, &format!("blk.{i}.attn_v.bias")),
                        attn_output_b: optional_tensor(&gguf, &format!("blk.{i}.attn_output.bias")),
                        attn_q_norm: optional_tensor(&gguf, &format!("blk.{i}.attn_q_norm.weight")),
                        attn_k_norm: optional_tensor(&gguf, &format!("blk.{i}.attn_k_norm.weight")),
                        ffn_norm: Some(require_tensor(&gguf, &format!("blk.{i}.ffn_norm.weight"))?),
                        ffn_norm_b: optional_tensor(&gguf, &format!("blk.{i}.ffn_norm.bias")),
                        ffn_up: if moe.is_some() {
                            None
                        } else {
                            Some(require_tensor(&gguf, &format!("blk.{i}.ffn_up.weight"))?)
                        },
                        ffn_down: if moe.is_some() {
                            None
                        } else {
                            Some(require_tensor(&gguf, &format!("blk.{i}.ffn_down.weight"))?)
                        },
                        ffn_gate,
                        ffn_up_b: optional_tensor(&gguf, &format!("blk.{i}.ffn_up.bias")),
                        ffn_down_b: optional_tensor(&gguf, &format!("blk.{i}.ffn_down.bias")),
                        moe,
                        ssm_in: None,
                        ssm_conv1d: None,
                        ssm_conv1d_b: None,
                        ssm_dt_b: None,
                        ssm_a: None,
                        ssm_d: None,
                        ssm_norm: None,
                        ssm_out: None,
                    }
                }
                LayerType::Ssm => LayerWeights {
                    layer_type: lt,
                    attn_norm,
                    attn_norm_b,
                    attn_q: None,
                    attn_k: None,
                    attn_v: None,
                    attn_output: None,
                    attn_q_b: None,
                    attn_k_b: None,
                    attn_v_b: None,
                    attn_output_b: None,
                    attn_q_norm: None,
                    attn_k_norm: None,
                    ffn_norm: None,
                    ffn_norm_b: None,
                    ffn_gate: None,
                    ffn_up: None,
                    ffn_down: None,
                    ffn_up_b: None,
                    ffn_down_b: None,
                    moe: None,
                    ssm_in: Some(require_tensor(&gguf, &format!("blk.{i}.ssm_in.weight"))?),
                    ssm_conv1d: Some(require_tensor(
                        &gguf,
                        &format!("blk.{i}.ssm_conv1d.weight"),
                    )?),
                    ssm_conv1d_b: optional_tensor(&gguf, &format!("blk.{i}.ssm_conv1d.bias")),
                    ssm_dt_b: Some(require_tensor(&gguf, &format!("blk.{i}.ssm_dt.bias"))?),
                    ssm_a: Some(require_tensor(&gguf, &format!("blk.{i}.ssm_a"))?),
                    ssm_d: Some(require_tensor(&gguf, &format!("blk.{i}.ssm_d"))?),
                    ssm_norm: Some(require_tensor(&gguf, &format!("blk.{i}.ssm_norm.weight"))?),
                    ssm_out: Some(require_tensor(&gguf, &format!("blk.{i}.ssm_out.weight"))?),
                },
                LayerType::Attention => LayerWeights {
                    layer_type: lt,
                    attn_norm,
                    attn_norm_b,
                    attn_q: Some(require_tensor(&gguf, &format!("blk.{i}.attn_q.weight"))?),
                    attn_k: Some(require_tensor(&gguf, &format!("blk.{i}.attn_k.weight"))?),
                    attn_v: Some(require_tensor(&gguf, &format!("blk.{i}.attn_v.weight"))?),
                    attn_output: Some(require_tensor(
                        &gguf,
                        &format!("blk.{i}.attn_output.weight"),
                    )?),
                    attn_q_b: optional_tensor(&gguf, &format!("blk.{i}.attn_q.bias")),
                    attn_k_b: optional_tensor(&gguf, &format!("blk.{i}.attn_k.bias")),
                    attn_v_b: optional_tensor(&gguf, &format!("blk.{i}.attn_v.bias")),
                    attn_output_b: optional_tensor(&gguf, &format!("blk.{i}.attn_output.bias")),
                    attn_q_norm: optional_tensor(&gguf, &format!("blk.{i}.attn_q_norm.weight")),
                    attn_k_norm: optional_tensor(&gguf, &format!("blk.{i}.attn_k_norm.weight")),
                    ffn_norm: None,
                    ffn_norm_b: None,
                    ffn_gate: None,
                    ffn_up: None,
                    ffn_down: None,
                    ffn_up_b: None,
                    ffn_down_b: None,
                    moe: None,
                    ssm_in: None,
                    ssm_conv1d: None,
                    ssm_conv1d_b: None,
                    ssm_dt_b: None,
                    ssm_a: None,
                    ssm_d: None,
                    ssm_norm: None,
                    ssm_out: None,
                },
                LayerType::Ffn => LayerWeights {
                    layer_type: lt,
                    attn_norm,
                    attn_norm_b,
                    attn_q: None,
                    attn_k: None,
                    attn_v: None,
                    attn_output: None,
                    attn_q_b: None,
                    attn_k_b: None,
                    attn_v_b: None,
                    attn_output_b: None,
                    attn_q_norm: None,
                    attn_k_norm: None,
                    // Hybrid FFN uses the block attn_norm only (no ffn_norm tensor).
                    ffn_norm: None,
                    ffn_norm_b: None,
                    ffn_gate: optional_tensor(&gguf, &format!("blk.{i}.ffn_gate.weight")),
                    ffn_up: Some(require_tensor(&gguf, &format!("blk.{i}.ffn_up.weight"))?),
                    ffn_down: Some(require_tensor(&gguf, &format!("blk.{i}.ffn_down.weight"))?),
                    ffn_up_b: optional_tensor(&gguf, &format!("blk.{i}.ffn_up.bias")),
                    ffn_down_b: optional_tensor(&gguf, &format!("blk.{i}.ffn_down.bias")),
                    moe: None,
                    ssm_in: None,
                    ssm_conv1d: None,
                    ssm_conv1d_b: None,
                    ssm_dt_b: None,
                    ssm_a: None,
                    ssm_d: None,
                    ssm_norm: None,
                    ssm_out: None,
                },
            };
            layers.push(layer);
        }

        if token_embd.k() != config.n_embd as usize {
            return Err(CoreError::BadTensor {
                name: token_embd.name.clone(),
                msg: format!(
                    "expected embd dim {}, got {}",
                    config.n_embd,
                    token_embd.k()
                ),
            });
        }

        let cuda_refusal = cuda_refusal_reason(&gguf, &config, &layers);
        Ok(Self {
            gguf,
            config,
            tokenizer,
            token_embd,
            output_norm,
            output_norm_b,
            output,
            layers,
            cuda_refusal,
        })
    }

    /// `Err` with a user-readable message when this model can't use the CUDA path yet.
    pub fn check_cuda_supported(&self) -> Result<()> {
        match &self.cuda_refusal {
            None => Ok(()),
            Some(why) => Err(CoreError::Backend(why.clone())),
        }
    }

    pub fn tensor_bytes(&self, w: &WeightTensor) -> Result<&[u8]> {
        self.gguf.tensor_data(&w.name).map_err(CoreError::from)
    }
}

/// CUDA scope: NEOX RoPE, q/k-norm and MoE run on the CPU
/// only. The device graph implements NORM RoPE and no q/k-norm, so it would be silently
/// wrong for these models; refuse instead.
fn cuda_refusal_reason(
    gguf: &GgufModel,
    config: &ModelConfig,
    layers: &[LayerWeights],
) -> Option<String> {
    let arch = &config.arch_str;
    let mut why = Vec::new();
    let has_rope = !config.architecture.is_hybrid() && config.n_rot > 0;
    if has_rope && config.rope_type == crate::config::RopeType::Neox {
        why.push("it uses NEOX-style RoPE");
    }
    if layers
        .iter()
        .any(|l| l.attn_q_norm.is_some() || l.attn_k_norm.is_some())
    {
        why.push("it has q/k-norm tensors (attn_q_norm / attn_k_norm)");
    }
    let moe = (0..config.n_layer).any(|i| {
        gguf.find_tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
            .is_some()
    });
    if moe {
        why.push("it has MoE layers (ffn_gate_inp)");
    }
    if why.is_empty() {
        return None;
    }
    // MoE is refused on CUDA with or without --experimental-moe; the flag only enables the CPU path.
    let moe_note = if moe {
        " MoE is experimental and CPU-only: run it with --backend cpu --experimental-moe."
    } else {
        ""
    };
    Some(format!(
        "the CUDA backend does not support `{arch}` models yet: {}. NEOX-RoPE architectures and \
         MoE models run on the CPU only; use the CPU backend (--backend cpu).{moe_note}",
        why.join(", ")
    ))
}

/// Load and shape-check the MoE tensors of layer `i`. `Ok(None)` for a dense layer.
/// A model with `config.moe` must have them on every layer; a dense config must have none.
fn load_moe_weights(gguf: &GgufModel, config: &ModelConfig, i: u32) -> Result<Option<MoeWeights>> {
    let inp_name = format!("blk.{i}.ffn_gate_inp.weight");
    let has_router = gguf.find_tensor(&inp_name).is_some();
    let Some(moe_cfg) = config.moe else {
        if has_router {
            return Err(CoreError::UnsupportedArchitecture(format!(
                "{} has MoE tensors ({inp_name}) but no MoE config; only olmoe and qwen2moe MoE \
                 are supported",
                config.arch_str
            )));
        }
        return Ok(None);
    };
    let gate_inp = require_tensor(gguf, &inp_name)?;
    let gate_exps = require_tensor(gguf, &format!("blk.{i}.ffn_gate_exps.weight"))?;
    let up_exps = require_tensor(gguf, &format!("blk.{i}.ffn_up_exps.weight"))?;
    let down_exps = require_tensor(gguf, &format!("blk.{i}.ffn_down_exps.weight"))?;
    let d = config.n_embd as u64;
    let ne = moe_cfg.n_expert as u64;
    let bad = |w: &WeightTensor, want: String| CoreError::BadTensor {
        name: w.name.clone(),
        msg: format!("expected shape {want}, got {:?}", w.shape),
    };
    if gate_inp.shape != [d, ne] {
        return Err(bad(&gate_inp, format!("[{d}, {ne}]")));
    }
    let ff = gate_exps.shape.get(1).copied().unwrap_or(0);
    if ff == 0 || gate_exps.shape != [d, ff, ne] {
        return Err(bad(&gate_exps, format!("[{d}, n_ff_exp, {ne}]")));
    }
    if let Some(want) = moe_cfg.n_ff_exp {
        if ff != want as u64 {
            return Err(bad(
                &gate_exps,
                format!("[{d}, {want}, {ne}] (n_ff_exp from metadata)"),
            ));
        }
    }
    if up_exps.shape != [d, ff, ne] {
        return Err(bad(&up_exps, format!("[{d}, {ff}, {ne}]")));
    }
    if down_exps.shape != [ff, d, ne] {
        return Err(bad(&down_exps, format!("[{ff}, {d}, {ne}]")));
    }
    for w in [&gate_exps, &up_exps, &down_exps] {
        expert_bytes(w.ty, w.k(), w.n())?;
    }
    let shexp_inp_name = format!("blk.{i}.ffn_gate_inp_shexp.weight");
    let shexp = match moe_cfg.n_ff_shexp {
        Some(ff_sh) => {
            let sh = SharedExpertWeights {
                gate_inp_shexp: require_tensor(gguf, &shexp_inp_name)?,
                gate_shexp: require_tensor(gguf, &format!("blk.{i}.ffn_gate_shexp.weight"))?,
                up_shexp: require_tensor(gguf, &format!("blk.{i}.ffn_up_shexp.weight"))?,
                down_shexp: require_tensor(gguf, &format!("blk.{i}.ffn_down_shexp.weight"))?,
                n_ff_shexp: ff_sh as usize,
            };
            check_shared_expert_shapes(d, &sh)?;
            Some(sh)
        }
        None => {
            // Never silently drop a shared expert the config does not know about.
            if gguf.find_tensor(&shexp_inp_name).is_some()
                || gguf
                    .find_tensor(&format!("blk.{i}.ffn_gate_shexp.weight"))
                    .is_some()
            {
                return Err(CoreError::UnsupportedArchitecture(format!(
                    "{} has shared-expert tensors in layer {i} but its MoE config has no shared \
                     expert",
                    config.arch_str
                )));
            }
            None
        }
    };
    Ok(Some(MoeWeights {
        gate_inp,
        gate_exps,
        up_exps,
        down_exps,
        n_ff_exp: ff as usize,
        shexp,
    }))
}

/// Shape check for one layer's shared expert (llama.cpp `llama_model_qwen2moe`):
/// `gate_inp_shexp` `{d}` (a trailing 1 is tolerated), `gate/up_shexp` `{d, ff}`,
/// `down_shexp` `{ff, d}` with `ff = sh.n_ff_shexp` from the config.
fn check_shared_expert_shapes(d: u64, sh: &SharedExpertWeights) -> Result<()> {
    let ff = sh.n_ff_shexp as u64;
    let bad = |w: &WeightTensor, want: String| CoreError::BadTensor {
        name: w.name.clone(),
        msg: format!("expected shape {want}, got {:?}", w.shape),
    };
    if ff == 0 {
        return Err(bad(&sh.gate_shexp, format!("[{d}, n_ff_shexp > 0]")));
    }
    let inp = &sh.gate_inp_shexp.shape;
    if !(*inp == [d] || *inp == [d, 1]) {
        return Err(bad(&sh.gate_inp_shexp, format!("[{d}]")));
    }
    if sh.gate_shexp.shape != [d, ff] {
        return Err(bad(
            &sh.gate_shexp,
            format!("[{d}, {ff}] (n_ff_shexp from metadata)"),
        ));
    }
    if sh.up_shexp.shape != [d, ff] {
        return Err(bad(&sh.up_shexp, format!("[{d}, {ff}]")));
    }
    if sh.down_shexp.shape != [ff, d] {
        return Err(bad(&sh.down_shexp, format!("[{ff}, {d}]")));
    }
    Ok(())
}

fn require_tensor(gguf: &GgufModel, name: &str) -> Result<WeightTensor> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| CoreError::MissingTensor(name.to_owned()))?;
    Ok(WeightTensor {
        name: info.name.clone(),
        shape: info.shape.clone(),
        ty: info.tensor_type,
    })
}

fn optional_tensor(gguf: &GgufModel, name: &str) -> Option<WeightTensor> {
    gguf.find_tensor(name).map(|info| WeightTensor {
        name: info.name.clone(),
        shape: info.shape.clone(),
        ty: info.tensor_type,
    })
}

#[cfg(test)]
pub(crate) mod moe_slice_tests {
    use super::{expert_bytes, expert_slice};
    use crate::quant::dequantize;
    use gguf::TensorType;

    /// Deterministic bytes for `n_blocks` quant blocks with every f16 scale field forced to a
    /// small finite value (random bytes would give NaN / inf scales).
    pub(crate) fn fake_blocks(ty: TensorType, n_blocks: usize, seed: u32) -> Vec<u8> {
        let ts = ty.type_size() as usize;
        let mut s = seed;
        let mut v: Vec<u8> = (0..n_blocks * ts)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        let half = 0x2E66u16.to_le_bytes(); // ≈ 0.1
        for b in 0..n_blocks {
            let blk = &mut v[b * ts..(b + 1) * ts];
            match ty {
                TensorType::Q8_0 | TensorType::Q5_0 | TensorType::Q4_0 => {
                    blk[..2].copy_from_slice(&half)
                }
                TensorType::Q4_K | TensorType::Q5_K => {
                    blk[..2].copy_from_slice(&half);
                    blk[2..4].copy_from_slice(&half);
                }
                TensorType::Q6_K => blk[ts - 2..].copy_from_slice(&half),
                _ => unreachable!("fake_blocks: {ty:?}"),
            }
        }
        v
    }

    /// Expert `e` of a 3-D `{k, n, n_expert}` tensor, sliced by byte offset, dequantises to
    /// exactly the `e`-th `k*n` chunk of a full dequant of the whole 3-D tensor.
    #[test]
    fn expert_slice_offset_equals_full_dequant() {
        for (ty, k, n, ne) in [
            (TensorType::Q4_K, 512usize, 6usize, 5usize),
            (TensorType::Q6_K, 256, 4, 3),
            (TensorType::Q5_0, 64, 7, 4),
            (TensorType::Q8_0, 96, 3, 6),
        ] {
            let bs = ty.block_size() as usize;
            let per_elems = k * n;
            let data = fake_blocks(ty, per_elems * ne / bs, 17 + k as u32);
            assert_eq!(expert_bytes(ty, k, n).unwrap() * ne, data.len());
            let mut full = vec![0.0f32; per_elems * ne];
            dequantize(&data, ty, 0, per_elems * ne, &mut full).unwrap();
            assert!(full.iter().all(|v| v.is_finite()));
            for e in 0..ne {
                let slice = expert_slice(&data, ty, k, n, ne, e).unwrap();
                let mut one = vec![0.0f32; per_elems];
                dequantize(slice, ty, 0, per_elems, &mut one).unwrap();
                assert_eq!(
                    one.as_slice(),
                    &full[e * per_elems..(e + 1) * per_elems],
                    "{ty:?} expert {e}"
                );
            }
            assert!(expert_slice(&data, ty, k, n, ne, ne).is_err());
        }
    }

    #[test]
    fn expert_bytes_rejects_partial_blocks() {
        assert!(expert_bytes(TensorType::Q4_K, 100, 2).is_err());
        assert_eq!(
            expert_bytes(TensorType::Q4_K, 2048, 1024).unwrap(),
            8 * 144 * 1024
        );
    }
}

/// F2: shared-expert (`qwen2moe`) tensor shape checks, model-free.
#[cfg(test)]
mod shexp_shape_tests {
    use super::{check_shared_expert_shapes, SharedExpertWeights, WeightTensor};
    use gguf::TensorType;

    fn wt(name: &str, shape: &[u64]) -> WeightTensor {
        WeightTensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            ty: TensorType::F32,
        }
    }

    /// Qwen1.5-MoE-A2.7B layer shapes: d = 2048, n_ff_shexp = 5632.
    fn a27b(inp: &[u64], gate: &[u64], up: &[u64], down: &[u64], ff: usize) -> SharedExpertWeights {
        SharedExpertWeights {
            gate_inp_shexp: wt("blk.0.ffn_gate_inp_shexp.weight", inp),
            gate_shexp: wt("blk.0.ffn_gate_shexp.weight", gate),
            up_shexp: wt("blk.0.ffn_up_shexp.weight", up),
            down_shexp: wt("blk.0.ffn_down_shexp.weight", down),
            n_ff_shexp: ff,
        }
    }

    #[test]
    fn qwen15_moe_shared_expert_shapes_accepted() {
        let ok = a27b(&[2048], &[2048, 5632], &[2048, 5632], &[5632, 2048], 5632);
        check_shared_expert_shapes(2048, &ok).unwrap();
        // A trailing unit dim on the gate vector is the same tensor.
        let ok1 = a27b(
            &[2048, 1],
            &[2048, 5632],
            &[2048, 5632],
            &[5632, 2048],
            5632,
        );
        check_shared_expert_shapes(2048, &ok1).unwrap();
    }

    #[test]
    fn shared_expert_shape_mismatches_refused_naming_tensor() {
        let g = [2048u64, 5632];
        let dn = [5632u64, 2048];
        for (sh, needle) in [
            (a27b(&[2048, 2], &g, &g, &dn, 5632), "ffn_gate_inp_shexp"),
            (a27b(&[1024], &g, &g, &dn, 5632), "ffn_gate_inp_shexp"),
            // Config width (metadata default n_ff) disagrees with the tensors.
            (a27b(&[2048], &g, &g, &dn, 1408), "ffn_gate_shexp"),
            (a27b(&[2048], &g, &[2048, 1408], &dn, 5632), "ffn_up_shexp"),
            // down transposed.
            (a27b(&[2048], &g, &g, &[2048, 5632], 5632), "ffn_down_shexp"),
            (a27b(&[2048], &g, &g, &dn, 0), "ffn_gate_shexp"),
        ] {
            let e = check_shared_expert_shapes(2048, &sh)
                .unwrap_err()
                .to_string();
            assert!(e.contains(needle), "{needle} not in {e}");
        }
    }
}
