//! Inference core: tokenizer, multi-arch dense transformer, KV cache, sampling.

pub mod backend;

/// Default completion budget (new tokens) when a caller does not set `max_tokens`.
/// One shared value for the server API, MCP tools, CLI `run` and
/// `artificer_runtime::GenerateConfig::default()`.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
/// Hard per-request ceiling on `max_tokens` (server/MCP DoS limit).
pub const MAX_TOKENS_CEILING: u32 = 4096;
mod chat_template;
mod config;
mod error;
mod forward;
mod kv;
mod model;
mod ops;
pub mod perplexity;
mod profile;
mod quant;
mod sample;
mod ssm;
mod sysmem;
pub mod tokenizer;

#[cfg(feature = "cuda")]
pub use backend::cuda::CudaBackend;
#[cfg(feature = "cuda")]
pub use backend::cuda_attention::{attention_launch_count, device_multi_head_attn};
pub use backend::{
    create_backend, cuda_feature_enabled, resolve_backend_kind, Backend, BackendKind, CpuBackend,
};
pub use chat_template::{
    render_chat, render_chat_prompt, render_user_chat_prompt, render_user_prompt, ChatMessage,
    RenderedPrompt, TemplateSource,
};
pub use config::{
    moe_status, Architecture, ContextBudgetReport, FfnActivation, LayerType, LlamaConfig,
    ModelConfig, MoeStatus, NormKind, SsmConfig, EXPERIMENTAL_MOE_FLAG, MOE_FAMILIES,
};
pub use error::{CoreError, Result};
pub use forward::{ForwardScratch, PrefillCancel, PrefillStatus};
pub use kv::{KvCache, KvElem};
pub use model::{LayerWeights, LlamaModel, WeightTensor};
pub use profile::{
    dump_stderr as profile_dump, enabled as profile_enabled, env_enabled as profile_env_enabled,
    note_decode_step, reset as profile_reset, set_active as profile_set_active,
};
pub use quant::pool_workers;
pub use quant::simd::{avx2_enabled, avx512_enabled, avx512_vnni_enabled};
pub use sample::{sample_greedy, sample_token, Rng, SampleParams, XorShiftRng};
pub use ssm::SsmCache;
pub use sysmem::{
    clamp_context_len, current_rss_bytes, kv_bytes_for_ctx, peak_rss_bytes, SystemMemory,
};
pub use tokenizer::{Segment, Tokenizer, TokenizerKind, Utf8StreamDecoder};

/// High-level handle used by the runtime (owns the loaded model).
pub struct ModelHandle {
    pub model: LlamaModel,
}

impl ModelHandle {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            model: LlamaModel::load(path)?,
        })
    }

    pub fn from_gguf(gguf: gguf::GgufModel) -> Result<Self> {
        Ok(Self {
            model: LlamaModel::from_gguf(gguf)?,
        })
    }
}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHandle")
            .field("arch", &self.model.config.arch_str)
            .field("n_layer", &self.model.config.n_layer)
            .field("n_embd", &self.model.config.n_embd)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("arch", &self.config.arch_str)
            .field("name", &self.config.name)
            .field("n_layer", &self.config.n_layer)
            .field("n_embd", &self.config.n_embd)
            .field("n_vocab", &self.config.n_vocab)
            .field("n_ctx", &self.config.n_ctx)
            .finish_non_exhaustive()
    }
}
