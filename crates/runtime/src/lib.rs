//! Memory-efficient inference execution loop.

use artificer_core::{
    create_backend, sample_token, ContextBudgetReport, ForwardScratch, KvCache, KvElem, LlamaModel,
    ModelHandle, PrefillStatus, RenderedPrompt, SampleParams, SsmCache, XorShiftRng,
};
use thiserror::Error;

pub use artificer_core::{cuda_feature_enabled, resolve_backend_kind, Backend, BackendKind};

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Core(#[from] artificer_core::CoreError),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub stop_on_eos: bool,
    /// Optional RNG seed; `None` uses entropy.
    pub seed: Option<u64>,
    /// Extra stop token ids (in addition to EOS when `stop_on_eos`).
    pub stop_token_ids: Vec<u32>,
    /// Stop when generated text contains any of these substrings (OpenAI `stop`).
    pub stop_strings: Vec<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: artificer_core::DEFAULT_MAX_TOKENS as usize,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            stop_on_eos: true,
            seed: None,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub prompt_tokens: Vec<u32>,
    pub completion_tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    /// Wall-clock split of this generate call (prefill vs decode). Timing only;
    /// never affects numerics.
    pub timings: GenerateTimings,
    /// Thinking share of `completion_tokens` (counts toward `max_tokens`).
    pub thinking: ThinkingStats,
    /// The prompt ended inside an open `<think>` (template opens thinking).
    pub prompt_opens_think: bool,
}

impl GenerateOutput {
    /// `max_tokens` was hit while the model was still thinking.
    pub fn limit_hit_in_thinking(&self) -> bool {
        self.finish_reason == FinishReason::Length && self.thinking.unterminated
    }
}

/// Per-completion timing. `prefill_tokens` = prompt tokens fed through
/// `LlamaModel::prefill` (batched on CUDA, token by token on CPU);
/// `decode_tokens` = sampled completion tokens (incl. a terminal stop id).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerateTimings {
    pub prefill_tokens: usize,
    pub prefill_ms: f64,
    pub decode_tokens: usize,
    pub decode_ms: f64,
}

impl GenerateTimings {
    /// Tokens per second for `tokens` over `ms`; `None` when either is zero.
    pub fn rate(tokens: usize, ms: f64) -> Option<f64> {
        if tokens == 0 || !(ms > 0.0) {
            None
        } else {
            Some(tokens as f64 * 1000.0 / ms)
        }
    }
    pub fn prefill_tok_s(&self) -> Option<f64> {
        Self::rate(self.prefill_tokens, self.prefill_ms)
    }
    pub fn decode_tok_s(&self) -> Option<f64> {
        Self::rate(self.decode_tokens, self.decode_ms)
    }
}

/// What backend the engine is actually running on, and why (for UIs / `/health`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendStatus {
    /// Backend the caller asked for.
    pub requested: BackendKind,
    /// Backend actually in use.
    pub active: BackendKind,
    /// Device name when `active` is CUDA (e.g. `NVIDIA GeForce RTX 4050 Laptop GPU`).
    pub device: Option<String>,
    /// Set when CUDA was requested but the engine fell back to CPU.
    pub fallback_reason: Option<String>,
}

impl BackendStatus {
    pub fn cpu() -> Self {
        Self {
            requested: BackendKind::Cpu,
            active: BackendKind::Cpu,
            device: None,
            fallback_reason: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// `max_tokens` reached (thinking tokens count toward it) or the context is full.
    Length,
    /// EOS, a stop token id, a stop string, or a chat turn boundary.
    Stop,
    /// The caller's cancel check returned `true` (client disconnect / user stop).
    Cancelled,
}

/// How much of a completion was spent inside a `<think>…</think>` block.
///
/// Tracked by token id (`<think>` / `</think>` vocab entries); models without those
/// tokens always report zero. A prompt that ends inside an open `<think>` (e.g. the
/// Nemotron generation cue `<|im_start|>assistant\n<think>\n`) starts in thinking.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ThinkingStats {
    /// Completion tokens up to and including the last `</think>`, or all of them
    /// when generation ended still inside the think block.
    pub tokens: usize,
    /// Wall time from the first sampled token (prefill excluded) until that
    /// `</think>` (or the end, when unterminated); `None` when there was no thinking.
    pub ms: Option<f64>,
    /// Generation ended (for any reason) inside an unterminated think block.
    pub unterminated: bool,
}

/// `true` when this model's chat template ends the generation prompt inside an open
/// `<think>` (e.g. Nemotron: `<|im_start|>assistant\n<think>\n`), i.e. replies start in
/// thinking. Derived from an actual render + tokenization of a one-turn prompt, with
/// the same rule [`ThinkTracker`] uses for counting.
pub fn template_opens_think(tokenizer: &artificer_core::Tokenizer) -> bool {
    artificer_core::render_user_chat_prompt(tokenizer, "Hi")
        .and_then(|p| p.encode(tokenizer, true))
        .map(|ids| ThinkTracker::new(tokenizer, &ids).starts_inside())
        .unwrap_or(false)
}

/// Incremental `<think>` / `</think>` tracker over sampled token ids.
#[derive(Debug, Clone)]
pub struct ThinkTracker {
    open: Option<u32>,
    close: Option<u32>,
    starts_inside: bool,
    inside: bool,
    seen: bool,
    n: usize,
    tokens: usize,
    close_at: Option<std::time::Instant>,
}

impl ThinkTracker {
    /// `prompt_tokens` decides whether generation starts inside a think block.
    pub fn new(tokenizer: &artificer_core::Tokenizer, prompt_tokens: &[u32]) -> Self {
        Self::with_ids(
            tokenizer.token_id("<think>"),
            tokenizer.token_id("</think>"),
            prompt_tokens,
        )
    }

    pub fn with_ids(open: Option<u32>, close: Option<u32>, prompt_tokens: &[u32]) -> Self {
        let last = |id: Option<u32>| id.and_then(|id| prompt_tokens.iter().rposition(|&t| t == id));
        let inside = match (last(open), last(close)) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        };
        Self {
            open,
            close,
            starts_inside: inside,
            inside,
            seen: inside,
            n: 0,
            tokens: 0,
            close_at: None,
        }
    }

    /// The prompt ended inside an open `<think>`: the reply starts in thinking.
    pub fn starts_inside(&self) -> bool {
        self.starts_inside
    }

    /// Record one sampled completion token.
    pub fn observe(&mut self, id: u32) {
        self.n += 1;
        if Some(id) == self.open {
            self.inside = true;
            self.seen = true;
        } else if Some(id) == self.close {
            self.inside = false;
            self.seen = true;
            self.tokens = self.n;
            self.close_at = Some(std::time::Instant::now());
        }
    }

    /// Final stats; `decode_start` / `end` bound the decode phase.
    pub fn finish(
        &self,
        decode_start: std::time::Instant,
        end: std::time::Instant,
    ) -> ThinkingStats {
        if !self.seen {
            return ThinkingStats::default();
        }
        let (tokens, until) = if self.inside {
            (self.n, Some(end))
        } else {
            (self.tokens, self.close_at)
        };
        ThinkingStats {
            tokens,
            ms: until.map(|t| t.saturating_duration_since(decode_start).as_secs_f64() * 1000.0),
            unterminated: self.inside,
        }
    }
}

/// Event emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// One newly sampled token and its decoded piece (may be empty for specials).
    Token { id: u32, text: String },
}

/// Options applied when loading a model into an [`Engine`].
#[derive(Debug, Clone)]
pub struct EngineLoadOptions {
    /// Cap context using system RAM so the KV cache fits (~45% of available by default).
    pub auto_context: bool,
    /// Absolute max context positions (also used as hard cap for auto mode).
    pub max_context: u32,
    /// Store KV cache as f16 (≈½ RAM, slight quality loss).
    pub kv_f16: bool,
    /// Override auto KV budget in bytes (`None` = detect).
    pub kv_budget_bytes: Option<u64>,
    /// Compute backend (`cpu` default; `cuda` needs `--features cuda` and a CUDA device).
    pub backend: BackendKind,
    /// When `backend` is CUDA and the device/cuBLAS/VRAM preflight fails, run on
    /// CPU instead of erroring and record why in [`Engine::backend_status`].
    /// Default `false` (CUDA failures stay loud).
    pub cpu_fallback: bool,
    /// `--experimental-moe`: allow MoE families marked experimental in
    /// `artificer_core::MOE_FAMILIES`. Without it such a model fails to load with an error
    /// naming the flag. Default `false`. Does not lift the CUDA MoE refusal.
    pub experimental_moe: bool,
}

impl Default for EngineLoadOptions {
    fn default() -> Self {
        Self {
            auto_context: true,
            max_context: 8192,
            kv_f16: false,
            kv_budget_bytes: None,
            backend: BackendKind::Cpu,
            cpu_fallback: false,
            experimental_moe: false,
        }
    }
}

pub struct Engine {
    model: LlamaModel,
    scratch: ForwardScratch,
    cache: KvCache,
    ssm: Option<SsmCache>,
    /// Active compute backend (CPU always works; CUDA needs `--features cuda`).
    backend: Box<dyn Backend>,
    backend_status: BackendStatus,
    pub context_report: Option<ContextBudgetReport>,
    /// One-line warning when an experimental MoE family was loaded with `experimental_moe`.
    moe_warning: Option<String>,
}

impl Engine {
    pub fn from_handle(handle: ModelHandle) -> Result<Self> {
        Self::from_handle_with(handle, EngineLoadOptions::default())
    }

    pub fn from_handle_with(handle: ModelHandle, opts: EngineLoadOptions) -> Result<Self> {
        let mut model = handle.model;
        // CUDA scope: NEOX-RoPE / q/k-norm / MoE models are CPU-only. With CPU fallback
        // on, run them on the CPU and report why; otherwise fail at load with the reason.
        // `experimental_moe` does not lift this: CUDA refuses MoE with or without it.
        let (backend, backend_status) = match (&model.cuda_refusal, opts.backend) {
            (Some(why), BackendKind::Cuda) if opts.cpu_fallback => (
                create_backend(BackendKind::Cpu)?,
                fallback_status(BackendKind::Cuda, why.clone()),
            ),
            (Some(_), BackendKind::Cuda) => {
                return Err(model.check_cuda_supported().unwrap_err().into())
            }
            _ => select_backend(
                opts.backend,
                opts.cpu_fallback,
                model.gguf.total_tensor_bytes(),
            )?,
        };
        // After the CUDA refusal above, so `--backend cuda` refuses MoE with or without the flag.
        let moe_warning = model.config.check_moe_gate(opts.experimental_moe)?;
        let kv_elem = if opts.kv_f16 {
            KvElem::F16
        } else {
            KvElem::F32
        };
        let context_report = if opts.auto_context {
            Some(model.config.apply_context_budget(
                opts.kv_budget_bytes,
                kv_elem.size_bytes(),
                opts.max_context,
            ))
        } else {
            // Still enforce hard max_context.
            if model.config.n_ctx > opts.max_context {
                model.config.n_ctx = opts.max_context;
            }
            None
        };
        let scratch = ForwardScratch::new(&model);
        let cache = KvCache::with_elem(&model.config, kv_elem);
        let ssm = if model.config.ssm.is_some() {
            Some(SsmCache::from_model(&model)?)
        } else {
            None
        };
        Ok(Self {
            model,
            scratch,
            cache,
            ssm,
            backend,
            backend_status,
            context_report,
            moe_warning,
        })
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::load_with(path, EngineLoadOptions::default())
    }

    pub fn load_with(path: impl AsRef<std::path::Path>, opts: EngineLoadOptions) -> Result<Self> {
        Self::from_handle_with(ModelHandle::load(path)?, opts)
    }

    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.backend.kind()
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// One-line experimental-MoE warning to print, when an experimental MoE family was loaded
    /// with `EngineLoadOptions::experimental_moe`; `None` otherwise.
    pub fn moe_warning(&self) -> Option<&str> {
        self.moe_warning.as_deref()
    }

    /// Requested vs active backend, CUDA device name, and any CPU fallback reason.
    pub fn backend_status(&self) -> &BackendStatus {
        &self.backend_status
    }

    pub fn kv_storage_bytes(&self) -> usize {
        self.cache.storage_bytes()
    }

    /// Generate from raw token ids with optional per-token callback (streaming).
    ///
    /// `on_token` is invoked after each non-stop-terminal sample with the new
    /// token id and decoded piece. When a stop string matches, the callback still
    /// receives tokens up to (and including) the piece that completed the match;
    /// final `text` has the stop string stripped.
    ///
    /// Chat control tokens (`<|im_start|>`, `<|im_end|>`, …) are always treated as
    /// stops / stripped from visible text so assistant generation cannot loop on
    /// role headers even when the caller only passed a subset of stop strings.
    /// After one empty leading `<|im_start|>` cycle is swallowed, the BPE prefix that
    /// re-spells that marker is logit-suppressed so the model must emit real content.
    pub fn generate_tokens_with_callback<F>(
        &mut self,
        prompt_tokens: &[u32],
        cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(StreamEvent),
    {
        self.generate_tokens_cancellable(prompt_tokens, cfg, on_token, || false)
    }

    /// Like [`generate_tokens_with_callback`](Self::generate_tokens_with_callback);
    /// `is_cancelled` is polled during prefill (between CUDA prefill chunks, or before
    /// every prompt token on the token-by-token path) and before every sampling step.
    /// When it returns `true`, generation stops with [`FinishReason::Cancelled`]; a
    /// Stop during prefill returns no completion tokens and
    /// `timings.prefill_tokens` = prompt tokens processed before the Stop.
    pub fn generate_tokens_cancellable<F, C>(
        &mut self,
        prompt_tokens: &[u32],
        cfg: &GenerateConfig,
        mut on_token: F,
        mut is_cancelled: C,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(StreamEvent),
        C: FnMut() -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::Other("empty prompt tokens".into()));
        }
        let n_ctx = self.model.config.n_ctx as usize;
        if prompt_tokens.len() >= n_ctx {
            return Err(artificer_core::CoreError::ContextExceeded {
                need: prompt_tokens.len(),
                max: n_ctx,
            }
            .into());
        }

        let t_start = std::time::Instant::now();
        self.cache.clear();
        if let Some(ref mut s) = self.ssm {
            s.clear();
        }
        // Mute profile timers during prefill so decode dumps stay clean.
        if artificer_core::profile_env_enabled() {
            artificer_core::profile_set_active(false);
        }
        // Whole-prompt prefill (`LlamaModel::prefill`): batched GEMM prefill on CUDA
        // when supported; token by token (one `forward` per prompt token, logits only on
        // the last) on CPU, single-token prompts and legacy/unsupported CUDA configs —
        // identical to the previous per-token loop. It leaves the last token's logits in
        // `self.scratch.logits`; the first sampled token below is drawn from there.
        // Stop during prefill: `is_cancelled` is also polled between prefill
        // chunks (CUDA batched chunks, or every token on the token-by-token path), so a
        // cancel ends a long prefill within one chunk with `FinishReason::Cancelled` and no
        // completion tokens. Caches are cleared at the start of every generation, so a
        // partially filled cache is never reused.
        let status = self.model.prefill_cancellable(
            prompt_tokens,
            0,
            &mut self.cache,
            self.ssm.as_mut(),
            &mut self.scratch,
            self.backend.as_ref(),
            Some(&mut is_cancelled),
        )?;
        if artificer_core::profile_env_enabled() {
            artificer_core::profile_set_active(true);
        }
        if let PrefillStatus::Cancelled { processed } = status {
            let t_end = std::time::Instant::now();
            let think = ThinkTracker::new(&self.model.tokenizer, prompt_tokens);
            return Ok(GenerateOutput {
                prompt_tokens: prompt_tokens.to_vec(),
                completion_tokens: Vec::new(),
                text: String::new(),
                finish_reason: FinishReason::Cancelled,
                // Prefill speed over the tokens actually processed before the Stop.
                timings: GenerateTimings {
                    prefill_tokens: processed,
                    prefill_ms: (t_end - t_start).as_secs_f64() * 1000.0,
                    decode_tokens: 0,
                    decode_ms: 0.0,
                },
                thinking: ThinkingStats::default(),
                prompt_opens_think: think.starts_inside(),
            });
        }
        // Prefill timing covers the whole prompt (batched or token by token).
        let t_prefill_done = std::time::Instant::now();

        let mut rng = match cfg.seed {
            Some(s) => XorShiftRng::new(s),
            None => XorShiftRng::from_entropy(),
        };
        let sample_params = SampleParams {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
        };

        let (stop_token_ids, stop_strings) = merge_chat_stops(cfg, &self.model.tokenizer);
        let im_start_id = self.model.tokenizer.token_id("<|im_start|>");
        // Nemotron rarely emits special id 10; it BPE-spells `<|im_start|>` as six
        // normal pieces (`<` `|` `im` `_start` `|` `>`). Resolve those ids so we can
        // suppress a *second* leading marker after the first empty cycle is swallowed.
        let im_start_bpe = resolve_im_start_bpe_pieces(&self.model.tokenizer);

        let mut think = ThinkTracker::new(&self.model.tokenizer, prompt_tokens);
        let prompt_opens_think = think.starts_inside();
        let mut completion = Vec::new();
        let mut text = String::new();
        let mut pos = prompt_tokens.len() - 1;
        let mut finish = FinishReason::Length;
        // After a leading `<|im_start|>` with empty visible text, buffer pieces until
        // we know whether they form a ChatML role header (`assistant\n` / …).
        let mut role_buf: Option<String> = None;
        let mut empty_im_start_count: u32 = 0;
        // A multi-byte character can span tokens (`<0xE4><0xBD><0xA0>`, byte-level BPE
        // pieces): decode token *bytes* incrementally so partial characters are held
        // back instead of each becoming U+FFFD. `utf8_tail_ok` is cleared when `text`
        // is cut at a stop / turn marker, so held bytes past the cut are dropped.
        let mut utf8 = artificer_core::Utf8StreamDecoder::new();
        let mut utf8_tail_ok = true;
        let mut last_piece_id = 0u32;

        // Opt-in debug trace (off by default): ARTIFICER_TRACE_TOP2=1 prints the
        // sampled token and the top-2 logits per step to stderr.
        let trace_top2 = std::env::var("ARTIFICER_TRACE_TOP2").is_ok_and(|v| v == "1");
        // `max_tokens` bounds every sampled token, thinking included.
        for _ in 0..cfg.max_tokens {
            if is_cancelled() {
                finish = FinishReason::Cancelled;
                break;
            }
            if pos + 1 >= n_ctx {
                break;
            }
            // After one swallowed empty `<|im_start|>…` cycle, forbid starting another
            // so the model must emit real content instead of role-header spam.
            apply_im_start_suppression(
                &mut self.scratch.logits,
                &text,
                role_buf.as_deref(),
                empty_im_start_count,
                im_start_id,
                &im_start_bpe,
            );
            let next = sample_token(&self.scratch.logits, sample_params, &mut rng);
            if trace_top2 {
                let (mut t1, mut t2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for &l in self.scratch.logits.iter() {
                    if l > t1 {
                        t2 = t1;
                        t1 = l;
                    } else if l > t2 {
                        t2 = l;
                    }
                }
                eprintln!(
                    "TRACE step={} tok={} top1={:.6} top2={:.6} margin={:.6}",
                    completion.len(),
                    next,
                    t1,
                    t2,
                    t1 - t2
                );
            }

            let is_eos = cfg.stop_on_eos && next == self.model.config.eos_token_id;
            let is_stop_id = stop_token_ids.contains(&next);
            let is_im_start = im_start_id == Some(next);

            completion.push(next);
            think.observe(next);

            if is_eos || is_stop_id {
                // Do not append EOS/stop-id surface to user-visible text.
                finish = FinishReason::Stop;
                break;
            }

            if is_im_start {
                // Mid-content new turn → stop cleanly (do not leak control marker).
                if !text.is_empty() || role_buf.as_ref().is_some_and(|b| !b.is_empty()) {
                    if let Some(buf) = role_buf.take() {
                        // Role header was mid-build; discard it on turn boundary.
                        let _ = buf;
                    }
                    finish = FinishReason::Stop;
                    break;
                }
                empty_im_start_count += 1;
                // Safety only: suppression should prevent further empty cycles. If it
                // somehow fails, bail rather than spinning to max_tokens.
                if empty_im_start_count >= 16 {
                    role_buf = None;
                    finish = FinishReason::Stop;
                    break;
                }
                role_buf = Some(String::new());
                pos += 1;
                self.model.forward(
                    next,
                    pos,
                    &mut self.cache,
                    self.ssm.as_mut(),
                    &mut self.scratch,
                    true,
                    self.backend.as_ref(),
                )?;
                continue;
            }

            let piece = utf8.push(&self.model.tokenizer.decode_token_bytes(next, true));
            last_piece_id = next;

            if let Some(ref mut buf) = role_buf {
                buf.push_str(&piece);
                match role_header_status(buf) {
                    RoleHeaderStatus::Prefix => {
                        pos += 1;
                        self.model.forward(
                            next,
                            pos,
                            &mut self.cache,
                            self.ssm.as_mut(),
                            &mut self.scratch,
                            true,
                            self.backend.as_ref(),
                        )?;
                        continue;
                    }
                    RoleHeaderStatus::Complete => {
                        // Swallowed ChatML role line; stay with empty visible text.
                        role_buf = None;
                        pos += 1;
                        self.model.forward(
                            next,
                            pos,
                            &mut self.cache,
                            self.ssm.as_mut(),
                            &mut self.scratch,
                            true,
                            self.backend.as_ref(),
                        )?;
                        continue;
                    }
                    RoleHeaderStatus::NotRole => {
                        let flushed = role_buf.take().unwrap_or_default();
                        if !flushed.is_empty() {
                            text.push_str(&flushed);
                            on_token(StreamEvent::Token {
                                id: next,
                                text: flushed,
                            });
                        }
                    }
                }
            } else if !piece.is_empty() {
                text.push_str(&piece);
                on_token(StreamEvent::Token {
                    id: next,
                    text: piece,
                });
            }

            // `<|im_start|>` may be emitted as ordinary BPE pieces (not the special
            // token id). Treat it as a turn boundary: keep prior content and stop, or
            // (when leading / empty) swallow it and watch for a role header loop.
            if let Some(idx) = text.find(CHAT_START_MARKER) {
                let before = text[..idx].to_owned();
                let after = text[idx + CHAT_START_MARKER.len()..].to_owned();
                if !before.is_empty() {
                    text = before;
                    utf8_tail_ok = false;
                    finish = FinishReason::Stop;
                    break;
                }
                // Leading marker (or marker with only a role header after it).
                empty_im_start_count += 1;
                // Do not abort on the 2nd empty cycle — logit suppression (above) forces
                // real content after the first swallowed header. Keep a high safety cap.
                if empty_im_start_count >= 16 {
                    text.clear();
                    role_buf = None;
                    utf8_tail_ok = false;
                    finish = FinishReason::Stop;
                    break;
                }
                text.clear();
                match role_header_status(&after) {
                    RoleHeaderStatus::Complete => {
                        role_buf = None;
                    }
                    RoleHeaderStatus::Prefix => {
                        role_buf = Some(after);
                    }
                    RoleHeaderStatus::NotRole => {
                        if !after.is_empty() {
                            text.push_str(&after);
                        }
                        role_buf = None;
                    }
                }
                pos += 1;
                self.model.forward(
                    next,
                    pos,
                    &mut self.cache,
                    self.ssm.as_mut(),
                    &mut self.scratch,
                    true,
                    self.backend.as_ref(),
                )?;
                continue;
            }

            // Hard stops (e.g. <|im_end|> and caller-provided strings). `<|im_start|>`
            // is excluded here and handled above.
            if let Some(stop) = find_stop(&text, &stop_strings) {
                if let Some(idx) = text.rfind(stop) {
                    text.truncate(idx);
                }
                utf8_tail_ok = false;
                finish = FinishReason::Stop;
                break;
            }

            pos += 1;
            self.model.forward(
                next,
                pos,
                &mut self.cache,
                self.ssm.as_mut(),
                &mut self.scratch,
                true,
                self.backend.as_ref(),
            )?;
        }

        // End of stream: bytes still held back form an incomplete character; emit them
        // lossily (U+FFFD), as a one-shot decode of the same tokens would show. Dropped
        // when the text was cut at a stop / marker or a role prefix is still buffered.
        if utf8_tail_ok && role_buf.is_none() {
            let tail = utf8.finish();
            if !tail.is_empty() {
                text.push_str(&tail);
                on_token(StreamEvent::Token {
                    id: last_piece_id,
                    text: tail,
                });
            }
        }
        // If we exit on length while still buffering a role prefix, drop it.
        let _ = role_buf;
        text = strip_control_markers(&text);
        let t_end = std::time::Instant::now();
        let timings = GenerateTimings {
            prefill_tokens: prompt_tokens.len(),
            prefill_ms: (t_prefill_done - t_start).as_secs_f64() * 1000.0,
            decode_tokens: completion.len(),
            decode_ms: (t_end - t_prefill_done).as_secs_f64() * 1000.0,
        };

        Ok(GenerateOutput {
            prompt_tokens: prompt_tokens.to_vec(),
            completion_tokens: completion,
            text,
            finish_reason: finish,
            timings,
            thinking: think.finish(t_prefill_done, t_end),
            prompt_opens_think,
        })
    }

    pub fn generate_tokens(
        &mut self,
        prompt_tokens: &[u32],
        cfg: &GenerateConfig,
    ) -> Result<GenerateOutput> {
        self.generate_tokens_with_callback(prompt_tokens, cfg, |_| {})
    }

    /// Generate from a **trusted raw** prompt string: special-token strings in it
    /// (`<|im_start|>`, …) are parsed as control tokens. For chat input use
    /// [`generate_prompt`](Self::generate_prompt) so user text can't forge specials.
    pub fn generate(&mut self, prompt: &str, cfg: &GenerateConfig) -> Result<GenerateOutput> {
        let prompt_tokens = self.model.tokenizer.encode(prompt, true)?;
        self.generate_tokens(&prompt_tokens, cfg)
    }

    /// Like [`generate`](Self::generate), streaming tokens to `on_token`.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(StreamEvent),
    {
        let prompt_tokens = self.model.tokenizer.encode(prompt, true)?;
        self.generate_tokens_with_callback(&prompt_tokens, cfg, on_token)
    }

    /// Generate from a rendered chat prompt: only template text may produce special
    /// tokens; message text is tokenized as plain text.
    pub fn generate_prompt(
        &mut self,
        prompt: &RenderedPrompt,
        cfg: &GenerateConfig,
    ) -> Result<GenerateOutput> {
        self.generate_prompt_with_callback(prompt, cfg, |_| {})
    }

    /// Like [`generate_prompt`](Self::generate_prompt), streaming tokens to `on_token`.
    pub fn generate_prompt_with_callback<F>(
        &mut self,
        prompt: &RenderedPrompt,
        cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(StreamEvent),
    {
        let prompt_tokens = prompt.encode(&self.model.tokenizer, true)?;
        self.generate_tokens_with_callback(&prompt_tokens, cfg, on_token)
    }

    /// Like [`generate_prompt_with_callback`](Self::generate_prompt_with_callback),
    /// stopping with [`FinishReason::Cancelled`] once `is_cancelled` returns `true`.
    pub fn generate_prompt_cancellable<F, C>(
        &mut self,
        prompt: &RenderedPrompt,
        cfg: &GenerateConfig,
        on_token: F,
        is_cancelled: C,
    ) -> Result<GenerateOutput>
    where
        F: FnMut(StreamEvent),
        C: FnMut() -> bool,
    {
        let prompt_tokens = prompt.encode(&self.model.tokenizer, true)?;
        self.generate_tokens_cancellable(&prompt_tokens, cfg, on_token, is_cancelled)
    }
}

/// Create the requested backend; with `cpu_fallback`, a CUDA failure (missing
/// feature, no device, cuBLAS, VRAM preflight) degrades to CPU with a reason.
fn select_backend(
    requested: BackendKind,
    cpu_fallback: bool,
    weight_bytes: u64,
) -> Result<(Box<dyn Backend>, BackendStatus)> {
    let attempt = create_backend(requested).and_then(|b| {
        // Fail loudly before any upload if quantized weights can't fit on the device.
        b.preflight_weights(weight_bytes)?;
        Ok(b)
    });
    match attempt {
        Ok(b) => {
            let status = BackendStatus {
                requested,
                active: b.kind(),
                device: backend_device_name(b.as_ref()),
                fallback_reason: None,
            };
            Ok((b, status))
        }
        Err(e) if requested == BackendKind::Cuda && cpu_fallback => {
            let reason = e.to_string();
            let cpu = create_backend(BackendKind::Cpu)?;
            Ok((cpu, fallback_status(requested, reason)))
        }
        Err(e) => Err(e.into()),
    }
}

fn fallback_status(requested: BackendKind, reason: String) -> BackendStatus {
    BackendStatus {
        requested,
        active: BackendKind::Cpu,
        device: None,
        fallback_reason: Some(if reason.trim().is_empty() {
            "CUDA backend unavailable".to_owned()
        } else {
            reason
        }),
    }
}

#[allow(unused_variables)]
fn backend_device_name(b: &dyn Backend) -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        if let Some(c) = b.as_cuda() {
            let n = c.device_name().trim();
            return Some(if n.is_empty() {
                "CUDA device".to_owned()
            } else {
                n.to_owned()
            });
        }
    }
    None
}

/// ChatML control markers. End markers are hard stops; the start marker is handled
/// specially in the generate loop (turn boundary / role-header loop guard).
pub const CHAT_CONTROL_STOP_STRINGS: &[&str] = &["<|im_end|>", "<|im_start|>"];
pub const CHAT_END_STOP_STRINGS: &[&str] = &["<|im_end|>"];
const CHAT_START_MARKER: &str = "<|im_start|>";

/// BPE piece ids that spell `<|im_start|>` when the model does not emit the special token.
fn resolve_im_start_bpe_pieces(tokenizer: &artificer_core::Tokenizer) -> Vec<u32> {
    match tokenizer.encode(CHAT_START_MARKER, false) {
        Ok(ids) if !ids.is_empty() => {
            // If encode collapsed to the special token alone, fall back to looking up
            // the common six-piece spelling used by Nemotron-style GPT-2 BPE.
            if ids.len() == 1 && tokenizer.token_id(CHAT_START_MARKER) == Some(ids[0]) {
                ["<", "|", "im", "_start", "|", ">"]
                    .into_iter()
                    .filter_map(|p| tokenizer.token_id(p))
                    .collect()
            } else {
                ids
            }
        }
        _ => ["<", "|", "im", "_start", "|", ">"]
            .into_iter()
            .filter_map(|p| tokenizer.token_id(p))
            .collect(),
    }
}

/// After one empty leading `<|im_start|>` cycle, suppress starting another so generation
/// continues into real content instead of aborting empty or looping role headers.
fn apply_im_start_suppression(
    logits: &mut [f32],
    text: &str,
    role_buf: Option<&str>,
    empty_im_start_count: u32,
    im_start_id: Option<u32>,
    im_start_bpe: &[u32],
) {
    if empty_im_start_count == 0 || role_buf.is_some() {
        return;
    }
    if let Some(id) = im_start_id {
        if (id as usize) < logits.len() {
            logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    // Block starting a new marker only while visible text is still empty. Once real
    // content exists, `<` / `|` must remain available (code, comparisons, etc.).
    if !text.is_empty() {
        return;
    }
    if let Some(&first) = im_start_bpe.first() {
        if (first as usize) < logits.len() {
            logits[first as usize] = f32::NEG_INFINITY;
        }
    }
}

/// Resolve known chat control token ids from the tokenizer vocab (when present).
pub fn resolve_chat_stop_token_ids(tokenizer: &artificer_core::Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    // End / EOS-style controls — hard stops. `<|im_start|>` is handled specially in
    // the generate loop so a leading marker before real content is not fatal.
    for piece in ["<|im_end|>", "</s>"] {
        if let Some(id) = tokenizer.token_id(piece) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn merge_chat_stops(
    cfg: &GenerateConfig,
    tokenizer: &artificer_core::Tokenizer,
) -> (Vec<u32>, Vec<String>) {
    let mut ids = cfg.stop_token_ids.clone();
    for id in resolve_chat_stop_token_ids(tokenizer) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    // Hard string stops: caller stops + end markers. Strip `<|im_start|>` even if the
    // caller passed it — that marker is handled as a turn boundary above.
    let mut strings: Vec<String> = cfg
        .stop_strings
        .iter()
        .filter(|s| !s.is_empty() && s.as_str() != CHAT_START_MARKER)
        .cloned()
        .collect();
    for s in CHAT_END_STOP_STRINGS {
        if !strings.iter().any(|x| x == *s) {
            strings.push((*s).to_owned());
        }
    }
    (ids, strings)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoleHeaderStatus {
    /// `buf` is a proper prefix of assistant/user/system (possibly empty).
    Prefix,
    /// `buf` is a full role line (`assistant\n`, …).
    Complete,
    /// `buf` cannot be a ChatML role header.
    NotRole,
}

fn role_header_status(buf: &str) -> RoleHeaderStatus {
    const ROLES: &[&str] = &["assistant", "user", "system"];
    if buf.is_empty() {
        return RoleHeaderStatus::Prefix;
    }
    for role in ROLES {
        if let Some(rest) = buf.strip_prefix(role) {
            if rest.is_empty() {
                // Wait for the mandatory newline after the role name.
                return RoleHeaderStatus::Prefix;
            }
            if rest.starts_with('\n') {
                return RoleHeaderStatus::Complete;
            }
            return RoleHeaderStatus::NotRole;
        }
        if role.starts_with(buf) {
            return RoleHeaderStatus::Prefix;
        }
    }
    RoleHeaderStatus::NotRole
}

fn find_stop<'a>(text: &str, stops: &'a [String]) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(idx) = text.find(s.as_str()) {
            match best {
                None => best = Some((s.as_str(), idx)),
                Some((_, bi)) if idx < bi => best = Some((s.as_str(), idx)),
                Some((bs, bi)) if idx == bi && s.len() > bs.len() => {
                    best = Some((s.as_str(), idx));
                }
                _ => {}
            }
        }
    }
    best.map(|(s, _)| s)
}

/// Belt-and-suspenders: strip any residual ChatML control markers from visible text.
fn strip_control_markers(text: &str) -> String {
    let mut out = text.to_owned();
    for marker in ["<|im_start|>", "<|im_end|>"] {
        out = out.replace(marker, "");
    }
    out
}

pub fn generate(
    model_path: impl AsRef<std::path::Path>,
    prompt: &str,
    cfg: &GenerateConfig,
) -> Result<GenerateOutput> {
    let mut engine = Engine::load(model_path)?;
    engine.generate(prompt, cfg)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_im_start_suppression, fallback_status, find_stop, role_header_status, select_backend,
        strip_control_markers, BackendKind, GenerateTimings, RoleHeaderStatus,
    };

    #[test]
    fn experimental_moe_is_off_by_default() {
        assert!(!super::EngineLoadOptions::default().experimental_moe);
    }

    #[test]
    fn timings_rates() {
        let t = GenerateTimings {
            prefill_tokens: 31,
            prefill_ms: 100.0,
            decode_tokens: 128,
            decode_ms: 10_322.58,
        };
        assert!((t.prefill_tok_s().unwrap() - 310.0).abs() < 1e-9);
        assert!((t.decode_tok_s().unwrap() - 12.4).abs() < 1e-3);
        assert_eq!(GenerateTimings::rate(0, 10.0), None);
        assert_eq!(GenerateTimings::rate(5, 0.0), None);
    }

    #[test]
    fn cpu_backend_status_has_no_fallback() {
        let (b, st) = select_backend(BackendKind::Cpu, true, 0).unwrap();
        assert_eq!(b.kind(), BackendKind::Cpu);
        assert_eq!(st.requested, BackendKind::Cpu);
        assert_eq!(st.active, BackendKind::Cpu);
        assert!(st.device.is_none() && st.fallback_reason.is_none());
    }

    #[test]
    fn cuda_fallback_records_reason() {
        // Without the feature (default test build) CUDA always fails → fallback path.
        // With the feature and a usable GPU this is Ok(cuda) and has a device name.
        match select_backend(BackendKind::Cuda, true, 0) {
            Ok((b, st)) => {
                assert_eq!(st.requested, BackendKind::Cuda);
                if b.kind() == BackendKind::Cpu {
                    assert_eq!(st.active, BackendKind::Cpu);
                    let r = st.fallback_reason.expect("fallback reason");
                    assert!(r.contains("CUDA") || r.contains("cuda"), "{r}");
                } else {
                    assert!(st.device.is_some());
                    assert!(st.fallback_reason.is_none());
                }
            }
            Err(e) => panic!("cpu_fallback must not error: {e}"),
        }
        if !artificer_core::cuda_feature_enabled() {
            assert!(select_backend(BackendKind::Cuda, false, 0).is_err());
        }
        let st = fallback_status(BackendKind::Cuda, "  ".into());
        assert_eq!(
            st.fallback_reason.as_deref(),
            Some("CUDA backend unavailable")
        );
    }

    #[test]
    fn stop_string_earliest() {
        let stops = vec!["\nUSER:".into(), "ASSISTANT:".into()];
        let t = "Hello\nUSER: more";
        assert_eq!(find_stop(t, &stops), Some("\nUSER:"));
    }

    #[test]
    fn stop_string_chatml_controls() {
        let stops = vec!["<|im_end|>".into(), "<|im_start|>".into()];
        let t = "Hello<|im_start|>assistant\n";
        assert_eq!(find_stop(t, &stops), Some("<|im_start|>"));
        let t2 = "done<|im_end|>";
        assert_eq!(find_stop(t2, &stops), Some("<|im_end|>"));
    }

    #[test]
    fn role_header_assistant_line() {
        assert_eq!(role_header_status(""), RoleHeaderStatus::Prefix);
        assert_eq!(role_header_status("a"), RoleHeaderStatus::Prefix);
        assert_eq!(role_header_status("assist"), RoleHeaderStatus::Prefix);
        assert_eq!(role_header_status("assistant"), RoleHeaderStatus::Prefix);
        assert_eq!(
            role_header_status("assistant\n"),
            RoleHeaderStatus::Complete
        );
        assert_eq!(
            role_header_status("assistant\nmore"),
            RoleHeaderStatus::Complete
        );
        assert_eq!(role_header_status("assistantX"), RoleHeaderStatus::NotRole);
        assert_eq!(role_header_status("408"), RoleHeaderStatus::NotRole);
        assert_eq!(role_header_status("1. Verify"), RoleHeaderStatus::NotRole);
        assert_eq!(role_header_status("user\n"), RoleHeaderStatus::Complete);
        assert_eq!(role_header_status("system\n"), RoleHeaderStatus::Complete);
    }

    #[test]
    fn strip_specials_from_text() {
        assert_eq!(strip_control_markers("<|im_start|>408<|im_end|>"), "408");
        assert_eq!(strip_control_markers("plain"), "plain");
    }

    #[test]
    fn im_start_suppression_after_empty_cycle() {
        let mut logits = vec![0.0f32; 16];
        logits[3] = 5.0; // pretend special id
        logits[7] = 9.0; // pretend first BPE piece
                         // No prior empty cycle → no change.
        apply_im_start_suppression(&mut logits, "", None, 0, Some(3), &[7, 8, 9]);
        assert_eq!(logits[3], 5.0);
        assert_eq!(logits[7], 9.0);
        // After one empty cycle with empty text → ban special + first BPE piece.
        apply_im_start_suppression(&mut logits, "", None, 1, Some(3), &[7, 8, 9]);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[7], f32::NEG_INFINITY);
        assert_eq!(logits[8], 0.0);
        // Non-empty text → do not ban `<`-like first piece (code may need it).
        let mut logits2 = vec![1.0f32; 16];
        apply_im_start_suppression(&mut logits2, "def factorial", None, 1, Some(3), &[7]);
        assert_eq!(logits2[3], f32::NEG_INFINITY); // special still banned
        assert_eq!(logits2[7], 1.0); // first BPE piece not banned
                                     // Mid role-header buffer → leave logits alone (finish swallowing the header).
        let mut logits3 = vec![2.0f32; 16];
        apply_im_start_suppression(&mut logits3, "", Some("assist"), 1, Some(3), &[7]);
        assert_eq!(logits3[3], 2.0);
        assert_eq!(logits3[7], 2.0);
    }
}
