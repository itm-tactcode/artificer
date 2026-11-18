//! Architecture classification and graph flavor flags.

use crate::error::{CoreError, Result};

/// Architectures we can run on the CPU path.
///
/// Dense Llama-like graphs cover most open models; Nemotron uses LayerNorm +
/// ReLU² FFN; Gemma adds embedding scale + GELU. MoE (experimental, CPU only): OLMoE
/// (`olmoe`, Llama-style block + softmax top-k router) and Qwen1.5/Qwen2-MoE (`qwen2moe`,
/// Qwen2 block + softmax top-k router + sigmoid-gated shared expert) are implemented;
/// DeepSeek experts / Qwen3-MoE / other MoE families are not yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Qwen,
    Gemma,
    DeepSeek,
    Nemotron,
    /// Hybrid Mamba-2 + attention + MLP (Nemotron-H / Nano).
    NemotronH,
}

/// Per-layer block type for hybrid graphs (and dense = AttnFfn).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Classic dense: attention then FFN in one layer.
    Dense,
    /// Mamba-2 SSM mixer only.
    Ssm,
    /// Self-attention only (hybrid).
    Attention,
    /// FFN only (hybrid, typically ReLU²).
    Ffn,
}

/// How residual FFN is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    /// SwiGLU: silu(gate) * up → down (Llama, Qwen, DeepSeek dense)
    SiluGated,
    /// GeGLU: gelu(gate) * up → down (Gemma)
    GeluGated,
    /// Sequential: relu(up)² → down (classic Nemotron; no gate)
    ReluSqrSequential,
}

/// RoPE pairing convention, per `llama_model_rope_type` in llama.cpp.
///
/// Table pinned to llama.cpp `8642541b` (v0.8.0), src/llama-model.cpp L3504+:
/// - NEOX: qwen, qwen2, qwen3, qwen2moe, qwen3moe, olmoe, gemma, gemma2, gemma3, gemma3n,
///   gemma4, phi3, stablelm, starcoder2, openelm, exaone, exaone4, nemotron, olmo2,
///   gpt-oss (`LLM_ARCH_OPENAI_MOE`).
/// - NORM: llama, llama4, deci, baichuan, internlm2, minicpm, xverse, olmo, command-r,
///   deepseek, deepseek2.
/// - NONE: nemotron_h (and nemotron_h_moe): no positional embedding (handled by
///   [`Architecture::is_hybrid`]).
///
/// Only qwen2/qwen3 are model-checked against llama.cpp; the other NEOX arches accepted
/// here (gemma*, phi3, stablelm, starcoder2, openelm, exaone, nemotron) are
/// "NEOX per llama.cpp, untested".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Interleaved pairs `(2i, 2i+1)` (GPT-J / original Llama).
    Norm,
    /// Half-split pairs `(i, i + n_rot/2)` (GPT-NeoX).
    Neox,
}

impl RopeType {
    /// RoPE type for a raw `general.architecture` string (pinned table above). Unknown and
    /// NORM arches return `Norm`, which is what every arch used before RoPE type was chosen
    /// per arch.
    pub fn for_arch(arch_str: &str) -> Self {
        match arch_str {
            "qwen" | "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" | "olmoe" | "gemma" | "gemma2"
            | "gemma3" | "gemma3n" | "gemma4" | "phi3" | "stablelm" | "starcoder2" | "openelm"
            | "exaone" | "exaone4" | "nemotron" | "olmo2" | "gpt-oss" => Self::Neox,
            _ => Self::Norm,
        }
    }
}

/// Normalization flavor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// x * rsqrt(mean(x²)+eps) * w
    Rms,
    /// Gemma: x * rsqrt(mean(x²)+eps) * (1+w)
    RmsGemma,
    /// (x-mean)/sqrt(var+eps) * w + b
    Layer,
}

impl Architecture {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "llama" | "llama4" | "deci" => Self::Llama,
            "qwen" | "qwen2" | "qwen3" => Self::Qwen,
            "gemma" | "gemma2" | "gemma3" => Self::Gemma,
            "deepseek" => Self::DeepSeek,
            // Classic dense Nemotron (LayerNorm + ReLU² FFN).
            "nemotron" => Self::Nemotron,
            // Hybrid Mamba-2 + sparse attention + MLP (Nemotron-H / Nano 3).
            "nemotron_h" => Self::NemotronH,
            "baichuan" | "internlm2" | "xverse" | "minicpm" | "olmo" | "stablelm" | "openelm"
            | "phi3" | "starcoder2" | "command-r" | "exaone" => Self::Llama,
            // OLMoE: Llama-style RMS/SwiGLU block with a MoE FFN (see `ModelConfig::moe`).
            "olmoe" => Self::Llama,
            // Qwen1.5-MoE / Qwen2-MoE: Qwen2 block (QKV biases, no q/k norm, NEOX RoPE) with a
            // MoE FFN plus a shared expert (see `ModelConfig::moe`).
            "qwen2moe" => Self::Qwen,
            other => {
                if other.starts_with("nemotron_h") && other != "nemotron_h" {
                    // e.g. nemotron_h_moe
                    return Err(CoreError::UnsupportedArchitecture(format!(
                        "{other} (MoE hybrid not supported; use dense nemotron_h Nano GGUF)"
                    )));
                }
                if other.contains("mamba") || other == "jamba" || other == "falcon-h1" {
                    return Err(CoreError::UnsupportedArchitecture(format!(
                        "{other} (standalone Mamba/Jamba not supported yet; nemotron_h is)"
                    )));
                }
                if other.contains("moe") || other == "deepseek2" || other == "qwen3moe" {
                    return Err(CoreError::UnsupportedArchitecture(format!(
                        "{other} (MoE / MLA not supported yet; use a dense distill GGUF if available)"
                    )));
                }
                return Err(CoreError::UnsupportedArchitecture(other.to_owned()));
            }
        })
    }

    /// Metadata key prefix (`llama`, `qwen2`, …) — may differ from enum name.
    pub fn metadata_prefix(arch_str: &str) -> &str {
        arch_str
    }

    pub fn ffn_activation(self) -> FfnActivation {
        match self {
            Self::Gemma => FfnActivation::GeluGated,
            Self::Nemotron | Self::NemotronH => FfnActivation::ReluSqrSequential,
            _ => FfnActivation::SiluGated,
        }
    }

    pub fn norm_kind(self) -> NormKind {
        match self {
            Self::Gemma => NormKind::RmsGemma,
            Self::Nemotron => NormKind::Layer,
            // Nemotron-H uses RMSNorm on blocks.
            _ => NormKind::Rms,
        }
    }

    pub fn scale_embeddings(self) -> bool {
        matches!(self, Self::Gemma)
    }

    pub fn is_hybrid(self) -> bool {
        matches!(self, Self::NemotronH)
    }
}

#[cfg(test)]
mod tests {
    use super::RopeType;

    #[test]
    fn rope_type_table_follows_pinned_llama_cpp() {
        for a in [
            "qwen",
            "qwen2",
            "qwen3",
            "qwen2moe",
            "qwen3moe",
            "olmoe",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma3n",
            "gemma4",
            "phi3",
            "stablelm",
            "starcoder2",
            "openelm",
            "exaone",
            "exaone4",
            "nemotron",
            "olmo2",
            "gpt-oss",
        ] {
            assert_eq!(RopeType::for_arch(a), RopeType::Neox, "{a}");
        }
        for a in [
            "llama",
            "llama4",
            "deci",
            "baichuan",
            "internlm2",
            "minicpm",
            "xverse",
            "olmo",
            "command-r",
            "deepseek",
            "deepseek2",
        ] {
            assert_eq!(RopeType::for_arch(a), RopeType::Norm, "{a}");
        }
    }
}
