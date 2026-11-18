//! Model architecture configuration parsed from GGUF metadata.

mod arch;

pub use arch::{Architecture, FfnActivation, LayerType, NormKind, RopeType};

use gguf::GgufModel;

use crate::error::{CoreError, Result};

/// Hyperparameters for the dense transformer path.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: Architecture,
    /// Raw `general.architecture` string (metadata prefix).
    pub arch_str: String,
    pub name: Option<String>,
    pub n_vocab: u32,
    pub n_ctx: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_rot: u32,
    /// Head dim for K (may differ from V on some models).
    pub head_dim: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// RoPE pairing (NORM interleaved vs NEOX half-split), from the arch string.
    pub rope_type: RopeType,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub ffn_activation: FfnActivation,
    pub norm_kind: NormKind,
    pub scale_embeddings: bool,
    /// Gemma-2 style logit softcap (0 = disabled).
    pub logit_scale: f32,
    pub attn_logit_softcap: f32,
    /// Sliding-window attention size (0 = full causal context). Used by Gemma2 alternating layers.
    pub sliding_window: u32,
    /// If true, even layers use sliding window and odd layers use full attention (Gemma2 pattern).
    pub sliding_window_pattern: bool,
    /// Per-layer block type (dense models: all `Dense`).
    pub layer_types: Vec<LayerType>,
    /// Per-layer FFN width (0 if not FFN / not dense dual-block).
    pub layer_n_ff: Vec<u32>,
    /// Per-layer KV heads (0 for non-attention hybrid layers).
    pub layer_n_head_kv: Vec<u32>,
    /// Mamba-2 / SSM hyperparameters (meaningful when architecture is hybrid).
    pub ssm: Option<SsmConfig>,
    /// Mixture-of-experts FFN routing (OLMoE, Qwen1.5/Qwen2-MoE). `None` = dense FFN.
    pub moe: Option<MoeConfig>,
}

/// MoE FFN hyperparameters (llama.cpp `build_moe_ffn`, softmax gating).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeConfig {
    /// Experts per MoE layer (`{arch}.expert_count`).
    pub n_expert: u32,
    /// Experts selected per token (`{arch}.expert_used_count`).
    pub n_expert_used: u32,
    /// Renormalise the selected top-k probabilities to sum to 1. OLMoE and Qwen2-MoE: `false`
    /// (llama.cpp passes `norm_w = false` for `LLM_ARCH_OLMOE` and `LLM_ARCH_QWEN2MOE`).
    pub norm_topk_prob: bool,
    /// Routed-expert FFN width the loader must find in `ffn_*_exps`. `qwen2moe`:
    /// `{arch}.expert_feed_forward_length`, else llama.cpp's default `n_ff / n_expert_used`.
    /// `None` (OLMoE): taken from the `ffn_gate_exps` shape.
    pub n_ff_exp: Option<u32>,
    /// Shared-expert FFN width. `Some` = every MoE layer also has one always-on shared expert
    /// (`ffn_{gate,up,down}_shexp`) scaled by a sigmoid gate (`ffn_gate_inp_shexp`). `qwen2moe`:
    /// `{arch}.expert_shared_feed_forward_length`, else llama.cpp's default `n_ff`.
    /// `None` = no shared expert (OLMoE).
    pub n_ff_shexp: Option<u32>,
}

impl MoeConfig {
    /// `true` when MoE layers carry a sigmoid-gated shared expert (`qwen2moe`).
    pub fn has_shared_expert(&self) -> bool {
        self.n_ff_shexp.is_some()
    }
}

/// CLI flag that opts in to MoE families whose status is [`MoeStatus::Experimental`].
pub const EXPERIMENTAL_MOE_FLAG: &str = "--experimental-moe";

/// Release status of one MoE family (`general.architecture`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeStatus {
    /// Passed its release gates: loads without any flag.
    Supported,
    /// Loads only with [`EXPERIMENTAL_MOE_FLAG`], and prints a one-line warning when it does.
    Experimental,
}

/// Per-family MoE status table. Re-labelling a family supported is a one-line change here;
/// no flag is then needed for it. An MoE arch that is not listed counts as experimental.
/// Every MoE family is experimental for now.
pub const MOE_FAMILIES: &[(&str, MoeStatus)] = &[
    ("olmoe", MoeStatus::Experimental),
    ("qwen2moe", MoeStatus::Experimental),
];

/// [`MoeStatus`] of MoE family `arch` per [`MOE_FAMILIES`] (unlisted: experimental).
pub fn moe_status(arch: &str) -> MoeStatus {
    moe_status_in(MOE_FAMILIES, arch)
}

fn moe_status_in(table: &[(&str, MoeStatus)], arch: &str) -> MoeStatus {
    table
        .iter()
        .find(|(a, _)| *a == arch)
        .map_or(MoeStatus::Experimental, |&(_, s)| s)
}

/// Experimental-MoE load gate. Dense models and supported MoE families: `Ok(None)`. An
/// experimental MoE family: `Err` naming [`EXPERIMENTAL_MOE_FLAG`] unless `experimental_moe`,
/// then `Ok(Some(one-line warning))` for the caller to print.
fn moe_gate(
    table: &[(&str, MoeStatus)],
    arch: &str,
    is_moe: bool,
    experimental_moe: bool,
) -> Result<Option<String>> {
    if !is_moe {
        return Ok(None);
    }
    match moe_status_in(table, arch) {
        MoeStatus::Supported => Ok(None),
        // Final user-facing strings (quoted in the release README): keep in sync.
        MoeStatus::Experimental if experimental_moe => Ok(Some(format!(
            "experimental MoE: `{arch}` loaded with {EXPERIMENTAL_MOE_FLAG}; output is not validated"
        ))),
        MoeStatus::Experimental => Err(CoreError::Other(format!(
            "`{arch}` is an MoE model; MoE support is experimental: pass {EXPERIMENTAL_MOE_FLAG} to load it"
        ))),
    }
}

/// Mamba-2 SSM dimensions from GGUF metadata.
#[derive(Debug, Clone, Copy)]
pub struct SsmConfig {
    pub d_conv: u32,
    pub d_inner: u32,
    pub d_state: u32,
    /// Number of SSM heads (`time_step_rank` / nheads).
    pub n_head: u32,
    pub n_group: u32,
}

impl SsmConfig {
    pub fn head_dim(self) -> u32 {
        self.d_inner / self.n_head
    }
    /// Channels through the causal conv (x + B + C).
    pub fn conv_dim(self) -> u32 {
        self.d_inner + 2 * self.n_group * self.d_state
    }
    /// `ssm_in` output width: z + xBC + dt.
    pub fn d_in_proj(self) -> u32 {
        2 * self.d_inner + 2 * self.n_group * self.d_state + self.n_head
    }
}

/// Back-compat alias.
pub type LlamaConfig = ModelConfig;

impl ModelConfig {
    pub fn n_embd_kv(&self) -> u32 {
        self.n_head_kv * self.head_dim
    }

    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// `true` when the FFN is mixture-of-experts (see [`MoeConfig`]).
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Experimental-MoE load gate for this model (see [`MOE_FAMILIES`]). `experimental_moe` is
    /// the caller's [`EXPERIMENTAL_MOE_FLAG`]. `Ok(Some(w))`: load, and print warning `w`.
    pub fn check_moe_gate(&self, experimental_moe: bool) -> Result<Option<String>> {
        moe_gate(
            MOE_FAMILIES,
            &self.arch_str,
            self.is_moe(),
            experimental_moe,
        )
    }

    /// Load config from a GGUF model.
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let meta = model.metadata();
        let arch_s = meta
            .architecture()?
            .ok_or_else(|| CoreError::MissingMetadata("general.architecture".into()))?
            .to_owned();
        let architecture = Architecture::parse(&arch_s)?;
        let p = Architecture::metadata_prefix(&arch_s);

        let require_u64 = |key: &str| -> Result<u64> {
            meta.get_u64(key)?
                .ok_or_else(|| CoreError::MissingMetadata(key.to_owned()))
        };

        let n_ctx = require_u64(&format!("{p}.context_length"))? as u32;
        let n_embd = require_u64(&format!("{p}.embedding_length"))? as u32;
        let n_layer = require_u64(&format!("{p}.block_count"))? as u32;

        // feed_forward_length may be scalar (dense) or per-layer array (hybrid).
        let layer_n_ff =
            read_u32_list(meta, &format!("{p}.feed_forward_length"), n_layer as usize)?;
        let n_ff = layer_n_ff.iter().copied().max().unwrap_or(0);

        let n_head = require_u64(&format!("{p}.attention.head_count"))? as u32;
        let layer_n_head_kv = match meta.get(&format!("{p}.attention.head_count_kv")) {
            Some(v) if v.as_array().is_some() => read_u32_list(
                meta,
                &format!("{p}.attention.head_count_kv"),
                n_layer as usize,
            )?,
            _ => {
                let kv = meta
                    .get_u64(&format!("{p}.attention.head_count_kv"))?
                    .unwrap_or(n_head as u64) as u32;
                vec![kv; n_layer as usize]
            }
        };
        let n_head_kv = layer_n_head_kv
            .iter()
            .copied()
            .filter(|&x| x > 0)
            .max()
            .unwrap_or(n_head);

        let head_dim = if let Some(v) = meta.get_u64(&format!("{p}.attention.key_length"))? {
            v as u32
        } else if let Some(v) = meta.get_u64(&format!("{p}.attention.value_length"))? {
            v as u32
        } else if n_head == 0 {
            return Err(CoreError::BadMetadata {
                key: format!("{p}.attention.head_count"),
                msg: "must be non-zero".into(),
            });
        } else {
            n_embd / n_head
        };

        let n_rot = meta
            .get_u64(&format!("{p}.rope.dimension_count"))?
            .unwrap_or(head_dim as u64) as u32;

        let rms_eps =
            if let Some(v) = meta.get_f64(&format!("{p}.attention.layer_norm_rms_epsilon"))? {
                v as f32
            } else if let Some(v) = meta.get_f64(&format!("{p}.attention.layer_norm_epsilon"))? {
                v as f32
            } else {
                1e-5
            };

        let rope_freq_base = meta
            .get_f64(&format!("{p}.rope.freq_base"))?
            .unwrap_or(10_000.0) as f32;

        check_plain_rope(meta, p)?;
        check_no_rope_freq_factors(model.tensors().iter().map(|t| t.name.as_str()))?;

        let moe = read_moe_config(meta, p, &arch_s)?;

        if architecture == Architecture::DeepSeek {
            if let Some(n_exp) = meta.get_u64(&format!("{p}.expert_count"))? {
                if n_exp > 0 {
                    return Err(CoreError::UnsupportedArchitecture(format!(
                        "deepseek with expert_count={n_exp} (MoE not supported; use a dense distill GGUF)"
                    )));
                }
            }
        }

        let n_vocab = read_vocab_size(model, meta)?;

        let bos_token_id = meta.get_u64("tokenizer.ggml.bos_token_id")?.unwrap_or(1) as u32;
        let eos_token_id = meta.get_u64("tokenizer.ggml.eos_token_id")?.unwrap_or(2) as u32;
        let add_bos_token = match meta.get_bool("tokenizer.ggml.add_bos_token")? {
            Some(b) => b,
            // Same key-absent default as the tokenizer (llama.cpp parity for gpt2 + qwen2).
            None => crate::tokenizer::default_add_bos(
                meta.get_str("tokenizer.ggml.model")?,
                meta.get_str("tokenizer.ggml.pre")?,
            ),
        };
        let add_eos_token = meta
            .get_bool("tokenizer.ggml.add_eos_token")?
            .unwrap_or(false);

        let attn_logit_softcap = meta
            .get_f64(&format!("{p}.attn_logit_softcapping"))?
            .or(meta.get_f64(&format!("{p}.attention.logit_softcapping"))?)
            .unwrap_or(0.0) as f32;
        let logit_scale = meta
            .get_f64(&format!("{p}.final_logit_softcapping"))?
            .unwrap_or(0.0) as f32;

        let sliding_window = meta
            .get_u64(&format!("{p}.attention.sliding_window"))?
            .or(meta.get_u64(&format!("{p}.sliding_window"))?)
            .unwrap_or(0) as u32;
        // Gemma2: alternating local/global; gemma3 often all local with SWA.
        let sliding_window_pattern = architecture == Architecture::Gemma
            && (arch_s == "gemma2" || arch_s.starts_with("gemma2"))
            && sliding_window > 0;

        let ssm = if architecture == Architecture::NemotronH {
            Some(SsmConfig {
                d_conv: require_u64(&format!("{p}.ssm.conv_kernel"))? as u32,
                d_inner: require_u64(&format!("{p}.ssm.inner_size"))? as u32,
                d_state: require_u64(&format!("{p}.ssm.state_size"))? as u32,
                n_head: require_u64(&format!("{p}.ssm.time_step_rank"))? as u32,
                n_group: require_u64(&format!("{p}.ssm.group_count"))? as u32,
            })
        } else {
            None
        };

        // Layer schedule: dense = all Dense; hybrid uses n_ff / n_head_kv rules from llama.cpp.
        let layer_types: Vec<LayerType> = if architecture.is_hybrid() {
            (0..n_layer as usize)
                .map(|i| {
                    let ff = layer_n_ff.get(i).copied().unwrap_or(0);
                    let kv = layer_n_head_kv.get(i).copied().unwrap_or(0);
                    if kv == 0 && ff == 0 {
                        LayerType::Ssm
                    } else if ff == 0 {
                        LayerType::Attention
                    } else {
                        LayerType::Ffn
                    }
                })
                .collect()
        } else {
            vec![LayerType::Dense; n_layer as usize]
        };

        let rope_type = RopeType::for_arch(&arch_s);
        Ok(Self {
            architecture,
            arch_str: arch_s,
            name: meta.name()?.map(str::to_owned),
            n_vocab,
            n_ctx,
            n_embd,
            n_layer,
            n_ff,
            n_head,
            n_head_kv,
            n_rot,
            head_dim,
            rms_eps,
            rope_freq_base,
            rope_type,
            bos_token_id,
            eos_token_id,
            add_bos_token,
            add_eos_token,
            ffn_activation: architecture.ffn_activation(),
            norm_kind: architecture.norm_kind(),
            scale_embeddings: architecture.scale_embeddings(),
            logit_scale,
            attn_logit_softcap,
            sliding_window,
            sliding_window_pattern,
            layer_types,
            layer_n_ff,
            layer_n_head_kv,
            ssm,
            moe,
        })
    }

    /// Reduce `n_ctx` so the KV cache fits a memory budget (and optional hard cap).
    ///
    /// `kv_elem_size`: 4 for f32 cache, 2 for f16. Uses ~half of available RAM by default
    /// when `budget_bytes` is `None` and system memory is detectable.
    pub fn apply_context_budget(
        &mut self,
        budget_bytes: Option<u64>,
        kv_elem_size: usize,
        hard_cap: u32,
    ) -> ContextBudgetReport {
        use crate::sysmem::{clamp_context_len, kv_bytes_for_ctx, SystemMemory};

        let requested = self.n_ctx;
        let mem = SystemMemory::detect();
        let budget = budget_bytes
            .or_else(|| mem.budget_bytes(0.45))
            .unwrap_or(u64::MAX);

        let effective = clamp_context_len(
            requested,
            self.n_layer,
            self.n_embd_kv(),
            budget,
            kv_elem_size,
            512,
            hard_cap,
        );
        self.n_ctx = effective;
        ContextBudgetReport {
            requested_ctx: requested,
            effective_ctx: effective,
            budget_bytes: budget,
            kv_bytes: kv_bytes_for_ctx(self.n_layer, self.n_embd_kv(), effective, kv_elem_size),
            total_ram_bytes: mem.total_bytes,
            available_ram_bytes: mem.available_bytes,
            capped: effective < requested,
        }
    }
}

/// Result of [`ModelConfig::apply_context_budget`].
#[derive(Debug, Clone, Copy)]
pub struct ContextBudgetReport {
    pub requested_ctx: u32,
    pub effective_ctx: u32,
    pub budget_bytes: u64,
    pub kv_bytes: u64,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub capped: bool,
}

/// Refuse any RoPE scaling other than plain RoPE (F3 guard). `apply_rope` /
/// `apply_rope_neox` have no `freq_scale`, YaRN, LongRoPE or attention-factor handling, so a
/// scaled model would load and run but be silently wrong at every position.
///
/// Accepted as plain: keys absent; `rope.scaling.type` = `none` (or `linear` with factor 1);
/// `rope.scaling.factor` / legacy `rope.scale_linear` of 0 (llama.cpp's "unset") or 1;
/// `rope.scaling.finetuned` (a flag, no math). Anything else is an error.
pub(crate) fn check_plain_rope(meta: &gguf::Metadata, p: &str) -> Result<()> {
    let refuse = |key: String, what: String| -> Result<()> {
        Err(CoreError::BadMetadata {
            key,
            msg: format!(
                "RoPE scaling is not supported ({what}); this engine implements plain RoPE only \
                 (no linear freq_scale, YaRN or LongRoPE). Refusing to load rather than produce \
                 silently wrong output"
            ),
        })
    };
    let plain_factor = |f: f64| f == 0.0 || f == 1.0;

    let factor_key = format!("{p}.rope.scaling.factor");
    let factor = meta.get_f64(&factor_key)?;
    if let Some(f) = factor {
        if !plain_factor(f) {
            return refuse(factor_key, format!("factor = {f}"));
        }
    }
    let lin_key = format!("{p}.rope.scale_linear");
    if let Some(f) = meta.get_f64(&lin_key)? {
        if !plain_factor(f) {
            return refuse(lin_key, format!("scale_linear = {f}"));
        }
    }
    let type_key = format!("{p}.rope.scaling.type");
    if let Some(t) = meta.get_str(&type_key)? {
        let t = t.trim();
        let ok = t.is_empty() || t.eq_ignore_ascii_case("none") || t.eq_ignore_ascii_case("linear");
        if !ok {
            return refuse(type_key, format!("type = {t:?}"));
        }
    }
    // YaRN / LongRoPE parameters or a non-unit attention factor imply a scaling scheme we
    // don't implement, whatever `type` says. (`original_context_length` and `finetuned`
    // are informational without one of these.)
    let prefix = format!("{p}.rope.scaling.");
    for (key, _) in meta.iter() {
        let Some(rest) = key.strip_prefix(&prefix) else {
            continue;
        };
        if rest.starts_with("yarn_") || rest.starts_with("long_") || rest.starts_with("short_") {
            return refuse(key.to_owned(), format!("`{rest}` is set"));
        }
        if rest == "attn_factor" {
            if let Some(f) = meta.get_f64(key)? {
                if !plain_factor(f) {
                    return refuse(key.to_owned(), format!("attn_factor = {f}"));
                }
            }
        }
    }
    Ok(())
}

/// Refuse per-dimension RoPE frequency factors (F3 guard, companion to [`check_plain_rope`]).
/// llama.cpp passes a `rope_freqs.weight` tensor (llama3-style scaling) — or Phi-3 LongRoPE's
/// `rope_factors_long/short.weight` — to `ggml_rope_ext` as per-dimension freq divisors. Our
/// RoPE ignores them, so such a model would load and be silently wrong at every position.
pub(crate) fn check_no_rope_freq_factors<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    for name in names {
        // `<leaf>.weight`, optionally under `blk.N.`
        let leaf = name.rsplit('.').nth(1).unwrap_or("");
        if matches!(
            leaf,
            "rope_freqs" | "rope_factors_long" | "rope_factors_short"
        ) {
            return Err(CoreError::BadMetadata {
                key: name.to_owned(),
                msg: format!(
                    "RoPE frequency factors are not supported (tensor `{name}` is present); this \
                     engine implements plain RoPE only (no per-dimension freq factors, llama3-style \
                     scaling or LongRoPE). Refusing to load rather than produce silently wrong output"
                ),
            });
        }
    }
    Ok(())
}

/// MoE routing config. `Ok(None)` for dense models (no / zero `expert_count`).
/// `olmoe` and `qwen2moe` are wired; other MoE arches are refused by
/// [`Architecture::parse`] (or, for DeepSeek, right after this).
///
/// `qwen2moe` follows llama.cpp `llama_model_qwen2moe` (pinned 8642541): softmax over all
/// experts, top-k, no renormalisation (`norm_w = false`), no expert weight scale; routed
/// width `expert_feed_forward_length` (absent / 0: `n_ff / n_expert_used`), shared-expert
/// width `expert_shared_feed_forward_length` (absent / 0: `n_ff`).
fn read_moe_config(meta: &gguf::Metadata, p: &str, arch_s: &str) -> Result<Option<MoeConfig>> {
    let n_expert = meta.get_u64(&format!("{p}.expert_count"))?.unwrap_or(0) as u32;
    let shared_expert = match arch_s {
        "olmoe" => false,
        "qwen2moe" => true,
        // DeepSeek dense-distill handling stays where it was.
        _ => return Ok(None),
    };
    if n_expert == 0 {
        return Err(CoreError::BadMetadata {
            key: format!("{p}.expert_count"),
            msg: format!("{arch_s} requires expert_count > 0"),
        });
    }
    let used_key = format!("{p}.expert_used_count");
    let n_expert_used =
        meta.get_u64(&used_key)?
            .ok_or_else(|| CoreError::MissingMetadata(used_key.clone()))? as u32;
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(CoreError::BadMetadata {
            key: used_key,
            msg: format!("expert_used_count {n_expert_used} must be in 1..={n_expert}"),
        });
    }
    if !shared_expert {
        return Ok(Some(MoeConfig {
            n_expert,
            n_expert_used,
            norm_topk_prob: false,
            n_ff_exp: None,
            n_ff_shexp: None,
        }));
    }
    let ff_key = format!("{p}.feed_forward_length");
    let n_ff = read_uniform_u32(meta, &ff_key)?
        .ok_or_else(|| CoreError::MissingMetadata(ff_key.clone()))?;
    let exp_key = format!("{p}.expert_feed_forward_length");
    let n_ff_exp = match read_uniform_u32(meta, &exp_key)? {
        Some(v) if v > 0 => v,
        _ => n_ff / n_expert_used,
    };
    let shexp_key = format!("{p}.expert_shared_feed_forward_length");
    let n_ff_shexp = match meta.get_u64(&shexp_key)? {
        Some(v) if v > 0 => v as u32,
        _ => n_ff,
    };
    if n_ff_exp == 0 {
        return Err(CoreError::BadMetadata {
            key: exp_key,
            msg: format!("{arch_s}: expert FFN width is 0 (feed_forward_length {n_ff})"),
        });
    }
    if n_ff_shexp == 0 {
        return Err(CoreError::BadMetadata {
            key: shexp_key,
            msg: format!("{arch_s}: shared-expert FFN width is 0"),
        });
    }
    Ok(Some(MoeConfig {
        n_expert,
        n_expert_used,
        norm_topk_prob: false,
        n_ff_exp: Some(n_ff_exp),
        n_ff_shexp: Some(n_ff_shexp),
    }))
}

/// A scalar integer, or a per-layer integer array whose elements are all equal. `Ok(None)` when
/// the key is absent. Per-layer arrays with differing values are refused (one MoE width per model).
fn read_uniform_u32(meta: &gguf::Metadata, key: &str) -> Result<Option<u32>> {
    let Some(v) = meta.get(key) else {
        return Ok(None);
    };
    let as_int = |el: &gguf::MetadataValue| el.as_u64().map(|u| u as u32);
    if let Some(u) = as_int(v) {
        return Ok(Some(u));
    }
    let bad = |msg: &str| CoreError::BadMetadata {
        key: key.to_owned(),
        msg: msg.to_owned(),
    };
    let arr = v
        .as_array()
        .ok_or_else(|| bad("expected unsigned integer or integer array"))?;
    let mut vals = arr.values.iter().map(as_int);
    let first = vals
        .next()
        .flatten()
        .ok_or_else(|| bad("expected a non-empty unsigned integer array"))?;
    for x in vals {
        match x {
            Some(x) if x == first => {}
            Some(_) => return Err(bad("per-layer values differ; one width per model only")),
            None => return Err(bad("expected unsigned integer elements")),
        }
    }
    Ok(Some(first))
}

/// Read a scalar u32 or a length-`n` integer array from metadata.
fn read_u32_list(meta: &gguf::Metadata, key: &str, n: usize) -> Result<Vec<u32>> {
    match meta.get(key) {
        None => Err(CoreError::MissingMetadata(key.to_owned())),
        Some(v) => {
            if let Some(u) = v.as_u64() {
                return Ok(vec![u as u32; n]);
            }
            if let Some(i) = v.as_i64() {
                return Ok(vec![i.max(0) as u32; n]);
            }
            let arr = v.as_array().ok_or_else(|| CoreError::BadMetadata {
                key: key.to_owned(),
                msg: "expected integer or integer array".into(),
            })?;
            if arr.values.len() != n {
                return Err(CoreError::BadMetadata {
                    key: key.to_owned(),
                    msg: format!("array len {} != n_layer {n}", arr.values.len()),
                });
            }
            let mut out = Vec::with_capacity(n);
            for el in &arr.values {
                let x = el
                    .as_u64()
                    .or_else(|| el.as_i64().map(|i| i.max(0) as u64))
                    .ok_or_else(|| CoreError::BadMetadata {
                        key: key.to_owned(),
                        msg: "expected integer elements".into(),
                    })?;
                out.push(x as u32);
            }
            Ok(out)
        }
    }
}

fn read_vocab_size(model: &GgufModel, meta: &gguf::Metadata) -> Result<u32> {
    if let Some(v) = meta.get("tokenizer.ggml.tokens") {
        v.as_array()
            .map(|a| a.len() as u32)
            .ok_or_else(|| CoreError::BadMetadata {
                key: "tokenizer.ggml.tokens".into(),
                msg: "expected string array".into(),
            })
    } else if let Some(t) = model.find_tensor("token_embd.weight") {
        Ok(t.shape.get(1).copied().unwrap_or(0) as u32)
    } else {
        Err(CoreError::MissingMetadata(
            "tokenizer.ggml.tokens / token_embd.weight".into(),
        ))
    }
}

#[cfg(test)]
mod moe_rope_tests {
    use super::{
        check_no_rope_freq_factors, check_plain_rope, read_moe_config, Architecture, MoeConfig,
    };
    use gguf::{Metadata, MetadataValue};

    fn meta(kv: &[(&str, MetadataValue)]) -> Metadata {
        let mut m = Metadata::new();
        for (k, v) in kv {
            m.insert((*k).to_owned(), v.clone());
        }
        m
    }

    #[test]
    fn plain_rope_accepted() {
        use MetadataValue::*;
        for m in [
            meta(&[]),
            meta(&[("olmoe.rope.freq_base", F32(10000.0))]),
            meta(&[("olmoe.rope.scaling.type", String("none".into()))]),
            meta(&[
                ("olmoe.rope.scaling.type", String("linear".into())),
                ("olmoe.rope.scaling.factor", F32(1.0)),
            ]),
            meta(&[("olmoe.rope.scaling.factor", F32(0.0))]),
            meta(&[("olmoe.rope.scale_linear", F32(1.0))]),
            // nemotron_h carries this flag only.
            meta(&[("olmoe.rope.scaling.finetuned", Bool(false))]),
            meta(&[("olmoe.rope.scaling.attn_factor", F32(1.0))]),
        ] {
            check_plain_rope(&m, "olmoe").unwrap();
        }
    }

    #[test]
    fn scaled_rope_refused_with_clear_error() {
        use MetadataValue::*;
        for (m, needle) in [
            (
                meta(&[("olmoe.rope.scaling.type", String("yarn".into()))]),
                "type = \"yarn\"",
            ),
            (
                meta(&[("olmoe.rope.scaling.type", String("longrope".into()))]),
                "longrope",
            ),
            (
                meta(&[
                    ("olmoe.rope.scaling.type", String("linear".into())),
                    ("olmoe.rope.scaling.factor", F32(4.0)),
                ]),
                "factor = 4",
            ),
            (
                meta(&[("olmoe.rope.scale_linear", F32(2.0))]),
                "scale_linear = 2",
            ),
            (
                meta(&[("olmoe.rope.scaling.yarn_log_multiplier", F32(0.1))]),
                "yarn_log_multiplier",
            ),
            (
                meta(&[("olmoe.rope.scaling.attn_factor", F32(0.5))]),
                "attn_factor = 0.5",
            ),
        ] {
            let e = check_plain_rope(&m, "olmoe").unwrap_err().to_string();
            assert!(e.contains("RoPE scaling is not supported"), "{e}");
            assert!(e.contains("plain RoPE only"), "{e}");
            assert!(e.contains(needle), "{needle:?} not in {e}");
        }
    }

    #[test]
    fn rope_freq_factor_tensors_refused_with_clear_error() {
        // Plain models: tiny-vicuna / OLMoE / Nemotron-style names all pass.
        check_no_rope_freq_factors([
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "output_norm.weight",
            "output.weight",
        ])
        .unwrap();
        check_no_rope_freq_factors(std::iter::empty()).unwrap();
        for bad in [
            "rope_freqs.weight",
            "blk.3.rope_freqs.weight",
            "rope_factors_long.weight",
            "rope_factors_short.weight",
        ] {
            let e = check_no_rope_freq_factors(["token_embd.weight", bad])
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("RoPE frequency factors are not supported"),
                "{e}"
            );
            assert!(e.contains("plain RoPE only"), "{e}");
            assert!(e.contains(bad), "{bad} not in {e}");
        }
    }

    #[test]
    fn olmoe_moe_config_from_metadata() {
        use MetadataValue::*;
        let m = meta(&[
            ("olmoe.expert_count", U32(64)),
            ("olmoe.expert_used_count", U32(8)),
        ]);
        assert_eq!(
            read_moe_config(&m, "olmoe", "olmoe").unwrap(),
            Some(MoeConfig {
                n_expert: 64,
                n_expert_used: 8,
                norm_topk_prob: false,
                n_ff_exp: None,
                n_ff_shexp: None,
            })
        );
        // Dense arches ignore expert keys here (DeepSeek handled separately).
        assert_eq!(read_moe_config(&m, "llama", "llama").unwrap(), None);
        // olmoe without experts / with a bad k is refused.
        assert!(read_moe_config(&meta(&[]), "olmoe", "olmoe").is_err());
        let bad = meta(&[
            ("olmoe.expert_count", U32(4)),
            ("olmoe.expert_used_count", U32(8)),
        ]);
        assert!(read_moe_config(&bad, "olmoe", "olmoe").is_err());
    }

    #[test]
    fn arch_parse_accepts_olmoe_and_qwen2moe_only() {
        assert_eq!(Architecture::parse("olmoe").unwrap(), Architecture::Llama);
        assert_eq!(Architecture::parse("qwen2moe").unwrap(), Architecture::Qwen);
        for a in [
            "qwen3moe",
            "qwen3vlmoe",
            "deepseek2",
            "nemotron_h_moe",
            "gpt-oss-moe",
            "phimoe",
            "granitemoe",
        ] {
            assert!(Architecture::parse(a).is_err(), "{a}");
        }
    }
}

#[cfg(test)]
mod moe_gate_tests {
    use super::{moe_gate, moe_status, MoeStatus, EXPERIMENTAL_MOE_FLAG, MOE_FAMILIES};

    #[test]
    fn dense_model_loads_with_or_without_flag() {
        for flag in [false, true] {
            assert_eq!(moe_gate(MOE_FAMILIES, "llama", false, flag).unwrap(), None);
            // A dense model of an MoE-family arch string is not gated either.
            assert_eq!(moe_gate(MOE_FAMILIES, "olmoe", false, flag).unwrap(), None);
        }
    }

    #[test]
    fn experimental_moe_refused_without_flag_names_flag() {
        let e = moe_gate(MOE_FAMILIES, "olmoe", true, false)
            .unwrap_err()
            .to_string();
        assert!(e.contains(EXPERIMENTAL_MOE_FLAG), "{e}");
        assert!(e.contains("--experimental-moe"), "{e}");
        assert!(e.contains("olmoe"), "{e}");
        assert_eq!(
            e,
            "`olmoe` is an MoE model; MoE support is experimental: pass --experimental-moe to load it"
        );
    }

    #[test]
    fn experimental_moe_with_flag_warns_one_line() {
        let w = moe_gate(MOE_FAMILIES, "olmoe", true, true)
            .unwrap()
            .expect("warning");
        assert!(!w.contains('\n'), "{w:?}");
        assert!(w.contains("experimental"), "{w}");
        assert!(w.contains("olmoe"), "{w}");
        assert_eq!(
            w,
            "experimental MoE: `olmoe` loaded with --experimental-moe; output is not validated"
        );
    }

    #[test]
    fn supported_family_needs_no_flag() {
        let table = [
            ("olmoe", MoeStatus::Supported),
            ("qwen2moe", MoeStatus::Experimental),
        ];
        for flag in [false, true] {
            assert_eq!(moe_gate(&table, "olmoe", true, flag).unwrap(), None);
        }
        assert!(moe_gate(&table, "qwen2moe", true, false).is_err());
        assert!(moe_gate(&table, "qwen2moe", true, true).unwrap().is_some());
    }

    #[test]
    fn unlisted_moe_family_is_experimental() {
        assert_eq!(moe_status("qwen2moe"), MoeStatus::Experimental);
        assert!(moe_gate(MOE_FAMILIES, "qwen2moe", true, false).is_err());
        assert!(moe_gate(MOE_FAMILIES, "qwen2moe", true, true)
            .unwrap()
            .is_some());
    }

    /// Every MoE family is experimental for now. Re-labelling one
    /// supported changes `MOE_FAMILIES` and this list together.
    #[test]
    fn every_moe_family_is_experimental_for_now() {
        assert_eq!(moe_status("olmoe"), MoeStatus::Experimental);
        for (arch, st) in MOE_FAMILIES {
            assert_eq!(*st, MoeStatus::Experimental, "{arch}");
        }
    }
}

/// F2: Qwen1.5-MoE (`qwen2moe`) config parsing, model-free (synthetic metadata / in-memory GGUF).
#[cfg(test)]
mod qwen2moe_config_tests {
    use super::{
        check_plain_rope, moe_gate, moe_status, read_moe_config, Architecture, ModelConfig,
        MoeConfig, MoeStatus, RopeType, MOE_FAMILIES,
    };
    use gguf::{Metadata, MetadataArray, MetadataValue, ValueType, GGUF_MAGIC};

    fn meta(kv: &[(&str, MetadataValue)]) -> Metadata {
        let mut m = Metadata::new();
        for (k, v) in kv {
            m.insert((*k).to_owned(), v.clone());
        }
        m
    }

    fn u32_arr(v: &[u32]) -> MetadataValue {
        MetadataValue::Array(MetadataArray {
            element_type: ValueType::U32,
            values: v.iter().map(|&x| MetadataValue::U32(x)).collect(),
        })
    }

    /// Qwen1.5-MoE-A2.7B: no `expert_*feed_forward_length` keys in the GGUF.
    fn a27b_moe_meta() -> Vec<(&'static str, MetadataValue)> {
        use MetadataValue::*;
        vec![
            ("qwen2moe.feed_forward_length", U32(5632)),
            ("qwen2moe.expert_count", U32(60)),
            ("qwen2moe.expert_used_count", U32(4)),
        ]
    }

    #[test]
    fn qwen2moe_moe_config_defaults_match_llama_cpp() {
        let m = meta(&a27b_moe_meta());
        let c = read_moe_config(&m, "qwen2moe", "qwen2moe")
            .unwrap()
            .unwrap();
        assert_eq!(
            c,
            MoeConfig {
                n_expert: 60,
                n_expert_used: 4,
                norm_topk_prob: false,
                // llama.cpp: n_ff / n_expert_used = 5632 / 4; shared expert: n_ff.
                n_ff_exp: Some(1408),
                n_ff_shexp: Some(5632),
            }
        );
        assert!(c.has_shared_expert());
        // OLMoE has no shared expert.
        let o = meta(&[
            ("olmoe.expert_count", MetadataValue::U32(64)),
            ("olmoe.expert_used_count", MetadataValue::U32(8)),
        ]);
        assert!(!read_moe_config(&o, "olmoe", "olmoe")
            .unwrap()
            .unwrap()
            .has_shared_expert());
    }

    #[test]
    fn qwen2moe_moe_config_explicit_widths() {
        use MetadataValue::*;
        // Qwen2-57B-A14B style: explicit routed / shared widths win over the defaults.
        let m = meta(&[
            ("qwen2moe.feed_forward_length", U32(18944)),
            ("qwen2moe.expert_count", U32(64)),
            ("qwen2moe.expert_used_count", U32(8)),
            ("qwen2moe.expert_feed_forward_length", U32(2560)),
            ("qwen2moe.expert_shared_feed_forward_length", U32(20480)),
        ]);
        let c = read_moe_config(&m, "qwen2moe", "qwen2moe")
            .unwrap()
            .unwrap();
        assert_eq!((c.n_ff_exp, c.n_ff_shexp), (Some(2560), Some(20480)));
        assert!(!c.norm_topk_prob);

        // A key of 0 means "unset" in llama.cpp: fall back to the defaults.
        let mut kv = a27b_moe_meta();
        kv.push(("qwen2moe.expert_feed_forward_length", U32(0)));
        kv.push(("qwen2moe.expert_shared_feed_forward_length", U32(0)));
        let c = read_moe_config(&meta(&kv), "qwen2moe", "qwen2moe")
            .unwrap()
            .unwrap();
        assert_eq!((c.n_ff_exp, c.n_ff_shexp), (Some(1408), Some(5632)));

        // Per-layer array form: uniform is accepted, mixed widths are refused.
        let mut kv = a27b_moe_meta();
        kv.push((
            "qwen2moe.expert_feed_forward_length",
            u32_arr(&[1408, 1408, 1408]),
        ));
        let c = read_moe_config(&meta(&kv), "qwen2moe", "qwen2moe")
            .unwrap()
            .unwrap();
        assert_eq!(c.n_ff_exp, Some(1408));
        let mut kv = a27b_moe_meta();
        kv.push((
            "qwen2moe.expert_feed_forward_length",
            u32_arr(&[1408, 2816]),
        ));
        let e = read_moe_config(&meta(&kv), "qwen2moe", "qwen2moe")
            .unwrap_err()
            .to_string();
        assert!(e.contains("expert_feed_forward_length"), "{e}");
    }

    #[test]
    fn qwen2moe_bad_expert_metadata_refused() {
        use MetadataValue::*;
        let without = |key: &str| -> Metadata {
            meta(
                &a27b_moe_meta()
                    .into_iter()
                    .filter(|(k, _)| *k != key)
                    .collect::<Vec<_>>(),
            )
        };
        // No experts / no k / no feed_forward_length (needed for the default widths).
        for key in [
            "qwen2moe.expert_count",
            "qwen2moe.expert_used_count",
            "qwen2moe.feed_forward_length",
        ] {
            assert!(
                read_moe_config(&without(key), "qwen2moe", "qwen2moe").is_err(),
                "{key}"
            );
        }
        for (count, used) in [(0u32, 4u32), (60, 0), (4, 8)] {
            let m = meta(&[
                ("qwen2moe.feed_forward_length", U32(5632)),
                ("qwen2moe.expert_count", U32(count)),
                ("qwen2moe.expert_used_count", U32(used)),
            ]);
            assert!(
                read_moe_config(&m, "qwen2moe", "qwen2moe").is_err(),
                "{count}/{used}"
            );
        }
        let e = read_moe_config(&without("qwen2moe.expert_count"), "qwen2moe", "qwen2moe")
            .unwrap_err()
            .to_string();
        assert!(e.contains("qwen2moe requires expert_count > 0"), "{e}");
    }

    #[test]
    fn qwen2moe_scaled_rope_refused() {
        use MetadataValue::*;
        check_plain_rope(&meta(&[("qwen2moe.rope.freq_base", F32(1e6))]), "qwen2moe").unwrap();
        for m in [
            meta(&[("qwen2moe.rope.scaling.type", String("yarn".into()))]),
            meta(&[("qwen2moe.rope.scaling.factor", F32(4.0))]),
            meta(&[("qwen2moe.rope.scaling.yarn_log_multiplier", F32(0.1))]),
        ] {
            let e = check_plain_rope(&m, "qwen2moe").unwrap_err().to_string();
            assert!(e.contains("RoPE scaling is not supported"), "{e}");
        }
    }

    #[test]
    fn qwen2moe_listed_experimental_and_gated() {
        assert!(MOE_FAMILIES.contains(&("qwen2moe", MoeStatus::Experimental)));
        assert_eq!(moe_status("qwen2moe"), MoeStatus::Experimental);
        let e = moe_gate(MOE_FAMILIES, "qwen2moe", true, false)
            .unwrap_err()
            .to_string();
        assert_eq!(
            e,
            "`qwen2moe` is an MoE model; MoE support is experimental: pass --experimental-moe to load it"
        );
        let w = moe_gate(MOE_FAMILIES, "qwen2moe", true, true)
            .unwrap()
            .expect("warning");
        assert_eq!(
            w,
            "experimental MoE: `qwen2moe` loaded with --experimental-moe; output is not validated"
        );
        // A family that is still not listed stays experimental too.
        assert_eq!(moe_status("qwen3moe"), MoeStatus::Experimental);
    }

    // --- Minimal in-memory GGUF writer (metadata only, no tensors). ---
    fn w_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn w_u64(b: &mut Vec<u8>, v: u64) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn w_str(b: &mut Vec<u8>, s: &str) {
        w_u64(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }
    fn w_val(b: &mut Vec<u8>, v: &MetadataValue) {
        match v {
            MetadataValue::String(s) => w_str(b, s),
            MetadataValue::U32(x) => w_u32(b, *x),
            MetadataValue::F32(f) => w_u32(b, f.to_bits()),
            MetadataValue::Bool(x) => b.push(u8::from(*x)),
            MetadataValue::Array(a) => {
                w_u32(b, a.element_type as u32);
                w_u64(b, a.values.len() as u64);
                for el in &a.values {
                    w_val(b, el);
                }
            }
            _ => panic!("test writer: unsupported value"),
        }
    }
    fn gguf_from(kvs: &[(&str, MetadataValue)]) -> gguf::GgufModel {
        let mut b = Vec::new();
        w_u32(&mut b, GGUF_MAGIC);
        w_u32(&mut b, 3);
        w_u64(&mut b, 0); // tensors
        w_u64(&mut b, kvs.len() as u64);
        for (k, v) in kvs {
            w_str(&mut b, k);
            w_u32(&mut b, v.value_type() as u32);
            w_val(&mut b, v);
        }
        gguf::GgufModel::from_bytes(b).unwrap()
    }

    /// The Qwen1.5-MoE-A2.7B-Chat GGUF header (architecture keys as found in the file).
    fn a27b_header() -> Vec<(&'static str, MetadataValue)> {
        use MetadataValue::*;
        let mut kv = vec![
            ("general.architecture", String("qwen2moe".into())),
            ("qwen2moe.block_count", U32(24)),
            ("qwen2moe.context_length", U32(32768)),
            ("qwen2moe.embedding_length", U32(2048)),
            ("qwen2moe.attention.head_count", U32(16)),
            ("qwen2moe.attention.head_count_kv", U32(16)),
            ("qwen2moe.rope.freq_base", F32(1_000_000.0)),
            ("qwen2moe.attention.layer_norm_rms_epsilon", F32(1e-6)),
            ("tokenizer.ggml.model", String("gpt2".into())),
            ("tokenizer.ggml.pre", String("qwen2".into())),
            (
                "tokenizer.ggml.tokens",
                Array(MetadataArray {
                    element_type: ValueType::String,
                    values: ["a", "b", "c"]
                        .iter()
                        .map(|t| String((*t).into()))
                        .collect(),
                }),
            ),
            ("tokenizer.ggml.bos_token_id", U32(151643)),
            ("tokenizer.ggml.eos_token_id", U32(151645)),
        ];
        kv.extend(a27b_moe_meta());
        kv
    }

    #[test]
    fn qwen2moe_model_config_from_synthetic_gguf() {
        let c = ModelConfig::from_gguf(&gguf_from(&a27b_header())).unwrap();
        assert_eq!(c.architecture, Architecture::Qwen);
        assert_eq!(c.arch_str, "qwen2moe");
        assert_eq!(c.rope_type, RopeType::Neox);
        assert_eq!((c.n_embd, c.n_layer, c.n_ff), (2048, 24, 5632));
        assert_eq!(
            (c.n_head, c.n_head_kv, c.head_dim, c.n_rot),
            (16, 16, 128, 128)
        );
        assert_eq!(c.rope_freq_base, 1_000_000.0);
        assert!((c.rms_eps - 1e-6).abs() < 1e-12);
        assert_eq!(
            c.moe,
            Some(MoeConfig {
                n_expert: 60,
                n_expert_used: 4,
                norm_topk_prob: false,
                n_ff_exp: Some(1408),
                n_ff_shexp: Some(5632),
            })
        );
        assert!(c.check_moe_gate(false).is_err());
        assert!(c.check_moe_gate(true).unwrap().is_some());
        // No add_bos_token key + gpt2/qwen2 vocab: no BOS, as in llama.cpp.
        assert!(!c.add_bos_token);

        // Same header with YaRN scaling: refused at config time.
        let mut kv = a27b_header();
        kv.push((
            "qwen2moe.rope.scaling.type",
            MetadataValue::String("yarn".into()),
        ));
        let e = ModelConfig::from_gguf(&gguf_from(&kv))
            .unwrap_err()
            .to_string();
        assert!(e.contains("RoPE scaling is not supported"), "{e}");

        // Experts are required for qwen2moe.
        let kv: Vec<_> = a27b_header()
            .into_iter()
            .filter(|(k, _)| *k != "qwen2moe.expert_count")
            .collect();
        assert!(ModelConfig::from_gguf(&gguf_from(&kv)).is_err());
    }
}
