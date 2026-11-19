//! HTTP route handlers (OpenAI chat + health/models).

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use artificer_runtime::{
    BackendStatus, Engine, EngineLoadOptions, FinishReason, GenerateConfig, GenerateTimings,
    StreamEvent,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream;
use tokio::sync::{mpsc, Mutex};

use crate::api_types::*;
use crate::config::RequestLimits;
use crate::error::err_response;
use crate::metrics::ServerMetrics;
use crate::prompt::{messages_to_prompt, total_content_chars};
use crate::tools::{parse_tool_calls, resolve_tools, tools_active, ToolMode};

pub struct AppState {
    pub engine: Mutex<Engine>,
    /// Id reported by `GET /v1/models` (first card). Updated after a successful load.
    pub model_id: tokio::sync::RwLock<String>,
    /// Load options from process start (context cap, KV type). Reused on swap.
    pub load_opts: EngineLoadOptions,
    pub limits: RequestLimits,
    pub metrics: ServerMetrics,
    /// Active backend snapshot (updated on load/swap). Read by `/health` without
    /// taking the engine lock, so health stays responsive mid-generation.
    pub backend: tokio::sync::RwLock<BackendStatus>,
    /// In-flight cancel flag + per-model facts readable without the engine lock.
    pub control: ChatControl,
}

/// Cancellation registry for the (single, engine-serialized) in-flight generation,
/// plus per-model facts that `/v1/models` reads without waiting on the engine lock.
#[derive(Default)]
pub struct ChatControl {
    active: std::sync::Mutex<Option<(String, Arc<AtomicBool>)>>,
    template_opens_think: AtomicBool,
}

impl ChatControl {
    pub fn for_engine(engine: &Engine) -> Self {
        let c = Self::default();
        c.set_model(engine);
        c
    }

    /// Refresh per-model facts (call after load / swap).
    pub fn set_model(&self, engine: &Engine) {
        let opens = artificer_runtime::template_opens_think(&engine.model().tokenizer);
        self.template_opens_think.store(opens, Ordering::SeqCst);
    }

    pub fn template_opens_think(&self) -> bool {
        self.template_opens_think.load(Ordering::SeqCst)
    }

    /// Register `cancel` as the in-flight generation `id` until the guard drops.
    pub(crate) fn begin(&self, id: &str, cancel: Arc<AtomicBool>) -> ActiveGuard<'_> {
        *self.active.lock().unwrap_or_else(|e| e.into_inner()) = Some((id.to_owned(), cancel));
        ActiveGuard {
            control: self,
            id: id.to_owned(),
        }
    }

    /// Cancel the in-flight generation (only if its id matches, when given).
    pub fn cancel(&self, id: Option<&str>) -> Option<String> {
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        match active.as_ref() {
            Some((cur, flag)) if id.is_none_or(|want| want == cur) => {
                flag.store(true, Ordering::SeqCst);
                Some(cur.clone())
            }
            _ => None,
        }
    }
}

pub(crate) struct ActiveGuard<'a> {
    control: &'a ChatControl,
    id: String,
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        let mut active = self
            .control
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if active.as_ref().is_some_and(|(cur, _)| *cur == self.id) {
            *active = None;
        }
    }
}

/// Sets the flag when dropped: a non-streaming handler future that is dropped
/// (client hung up) cancels its blocking generation within one token, or within one
/// prefill chunk while the prompt is still being prefilled.
pub(crate) struct CancelOnDrop(pub(crate) Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// `backend` object on `GET /health` (additive). `active` is `cuda` or `cpu`;
/// `device` is the CUDA device name; `fallback_reason` is set only when CUDA was
/// requested and the server is running on CPU instead.
pub(crate) fn backend_json(st: &BackendStatus) -> serde_json::Value {
    serde_json::json!({
        "requested": st.requested.as_str(),
        "active": st.active.as_str(),
        "device": st.device,
        "fallback_reason": st.fallback_reason,
    })
}

/// Additive `timings` object on the final stream chunk and non-stream response.
/// OpenAI clients ignore unknown top-level keys.
pub(crate) fn timings_json(t: &GenerateTimings) -> serde_json::Value {
    serde_json::json!({
        "prefill_tokens": t.prefill_tokens,
        "prefill_ms": round3(t.prefill_ms),
        "decode_tokens": t.decode_tokens,
        "decode_ms": round3(t.decode_ms),
        "prefill_tok_s": t.prefill_tok_s().map(round3),
        "decode_tok_s": t.decode_tok_s().map(round3),
    })
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

enum StreamMsg {
    Role,
    /// Plain content delta.
    Delta(String),
    /// Full tool_calls payload (emitted once when tools parsed).
    ToolCalls(Vec<ToolCall>),
    /// Final chunk: outcome + server-side counts (the UI never counts itself).
    /// `choice_reason` is the OpenAI `choices[0].finish_reason` (`None` on error).
    Finish {
        choice_reason: Option<&'static str>,
        usage: Option<Usage>,
        timings: Option<serde_json::Value>,
        outcome: Outcome,
    },
    DoneMarker,
}

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "metrics": state.metrics.snapshot(),
        "backend": backend_json(&*state.backend.read().await),
    }))
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let id = state.model_id.read().await.clone();
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id,
            object: "model",
            created: now_secs(),
            owned_by: "local",
            template_opens_think: state.control.template_opens_think(),
        }],
    })
}

#[derive(serde::Deserialize)]
pub(crate) struct LoadModelRequest {
    path: String,
}

/// `POST /v1/models/load` — swap weights in-process.
///
/// Body: `{"path": "<absolute path>"}`. On success the running engine is replaced
/// and `GET /v1/models` reports the new id (file stem). The previous engine is
/// kept if the new file fails to load.
pub async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let path = match checked_model_path(&req.path) {
        Ok(p) => p,
        Err(msg) => return err_response(StatusCode::BAD_REQUEST, msg),
    };

    let model_id = model_id_from_path(&path);
    let opts = state.load_opts.clone();
    let path_for_load = path.clone();

    // Hold the engine lock across the load so a completion cannot start on the
    // old weights and finish after the swap, and so only one swap runs at a time.
    let mut engine = state.engine.lock().await;
    tracing::info!("swapping model to {} (id={})", path.display(), model_id);
    let t0 = std::time::Instant::now();
    let loaded = tokio::task::spawn_blocking(move || Engine::load_with(&path_for_load, opts)).await;

    let new_engine = match loaded {
        Ok(Ok(engine)) => {
            if let Some(w) = engine.moe_warning() {
                tracing::warn!("{w}");
            }
            engine
        }
        // The full error (it can name the path or quote bytes of the file) goes only to the
        // local log; the client gets `public_load_error`.
        Ok(Err(e)) => {
            tracing::error!("model load failed (previous model kept): {e}");
            return err_response(StatusCode::INTERNAL_SERVER_ERROR, public_load_error(&e));
        }
        Err(e) => {
            tracing::error!("model load task failed (previous model kept): {e}");
            return err_response(StatusCode::INTERNAL_SERVER_ERROR, LOAD_FAILED_GENERIC);
        }
    };

    let status = new_engine.backend_status().clone();
    state.control.set_model(&new_engine);
    *engine = new_engine;
    *state.backend.write().await = status.clone();
    {
        let mut id = state.model_id.write().await;
        *id = model_id.clone();
    }
    drop(engine);

    tracing::info!(
        "model swap ready in {:.2}s (id={})",
        t0.elapsed().as_secs_f32(),
        model_id
    );

    Json(serde_json::json!({
        "id": model_id,
        "object": "model",
        "path": path.display().to_string(),
        "backend": backend_json(&status),
    }))
    .into_response()
}

/// `POST /v1/models/load` answer for a path that is missing, not a regular file, or not
/// `.gguf`: one message for all three, with no path, so the endpoint cannot probe the disk.
pub(crate) const LOAD_PATH_ERROR: &str = "model file not found or not a .gguf file";

/// `POST /v1/models/load` answer for any load failure that is not one of our own refusals
/// (GGUF parse and I/O errors quote paths and file bytes; those stay in the server log).
pub(crate) const LOAD_FAILED_GENERIC: &str =
    "This file couldn't be loaded. It may be damaged or not a supported GGUF model.";

/// Architecture names the engine's own refusals spell out (`Architecture::parse`), besides the
/// ones it can load.
const NAMED_REFUSED_ARCHES: [&str; 4] = ["deepseek2", "qwen3moe", "jamba", "falcon-h1"];

/// Validate the requested path: absolute, an existing regular file, `.gguf`. Canonical on success.
pub(crate) fn checked_model_path(raw: &str) -> std::result::Result<PathBuf, &'static str> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("path is required");
    }
    let requested = PathBuf::from(raw);
    if !requested.is_absolute() {
        return Err("path must be an absolute filesystem path");
    }
    let path = match requested.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("model load refused: {} ({e})", requested.display());
            return Err(LOAD_PATH_ERROR);
        }
    };
    if !path.is_file() || !is_gguf(&path) {
        tracing::warn!("model load refused: {} is not a .gguf file", path.display());
        return Err(LOAD_PATH_ERROR);
    }
    Ok(path)
}

/// An architecture name the engine knows (one it loads, or one its refusals name). Only these
/// are echoed: the name is `general.architecture` from the file.
fn known_arch(name: &str) -> bool {
    artificer_core::Architecture::parse(name).is_ok() || NAMED_REFUSED_ARCHES.contains(&name)
}

/// What a client sees for a failed load. Our own fixed refusals pass through (MoE gate, CUDA
/// gate, unsupported architecture — the name only when [`known_arch`]); everything else is
/// [`LOAD_FAILED_GENERIC`].
pub(crate) fn public_load_error(e: &artificer_runtime::RuntimeError) -> String {
    use artificer_core::CoreError;
    use artificer_runtime::RuntimeError;
    let refusal = match e {
        RuntimeError::Core(CoreError::UnsupportedArchitecture(what)) => {
            let name = what.split(' ').next().unwrap_or("");
            Some(if known_arch(name) {
                format!("unsupported architecture `{name}`")
            } else {
                "unsupported model architecture".to_owned()
            })
        }
        RuntimeError::Core(CoreError::Other(msg)) if is_moe_gate(msg) => Some(msg.clone()),
        RuntimeError::Core(CoreError::Backend(msg)) if is_cuda_gate(msg) => Some(msg.clone()),
        _ => None,
    };
    match refusal {
        Some(r) => format!("failed to load model: {r}"),
        None => LOAD_FAILED_GENERIC.to_owned(),
    }
}

/// The experimental-MoE gate (`artificer_core::config::moe_gate`) for a known architecture.
fn is_moe_gate(msg: &str) -> bool {
    let tail = format!(
        "` is an MoE model; MoE support is experimental: pass {} to load it",
        artificer_core::EXPERIMENTAL_MOE_FLAG
    );
    msg.strip_prefix('`')
        .and_then(|m| m.strip_suffix(tail.as_str()))
        .is_some_and(known_arch)
}

/// The CUDA scope refusal (`cuda_refusal_reason` in core `model.rs`) for a known architecture.
/// Everything after the name is fixed text.
fn is_cuda_gate(msg: &str) -> bool {
    msg.strip_prefix("the CUDA backend does not support `")
        .and_then(|m| m.split_once("` models yet: "))
        .is_some_and(|(arch, _)| known_arch(arch))
}

fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

fn model_id_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "local-model".into())
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(msg) = validate_request(&req, &state.limits) {
        return err_response(StatusCode::BAD_REQUEST, msg);
    }

    let (tools, tool_mode) = match resolve_tools(
        &req.tools,
        &req.tool_choice,
        &req.functions,
        &req.function_call,
    ) {
        Ok(x) => x,
        Err(e) => return err_response(StatusCode::BAD_REQUEST, e),
    };

    let model_id = match req.model.clone() {
        Some(m) => m,
        None => state.model_id.read().await.clone(),
    };
    let cfg = build_generate_config(&req, &state.limits);
    let stream = req.stream.unwrap_or(false);
    let created = now_secs();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    if stream {
        return stream_chat(
            state,
            req.messages,
            tools,
            tool_mode,
            cfg,
            model_id,
            id,
            created,
        );
    }

    // Generation runs on a blocking thread so this handler future stays cancellable:
    // if the client hangs up and the future is dropped, `_hangup` sets the flag and
    // decoding stops within one token (prefill within one chunk). `POST /v1/chat/cancel`
    // sets the same flag.
    let cancel = Arc::new(AtomicBool::new(false));
    let _hangup = CancelOnDrop(Arc::clone(&cancel));
    let messages = req.messages;
    let state_bg = Arc::clone(&state);
    let id_bg = id.clone();
    let tool_mode_bg = tool_mode.clone();
    let cfg_bg = cfg.clone();
    let joined = tokio::task::spawn_blocking(move || {
        let guard = state_bg.metrics.begin_request();
        let mut engine = state_bg.engine.blocking_lock();
        let _active = state_bg.control.begin(&id_bg, Arc::clone(&cancel));
        let prompt =
            match messages_to_prompt(&engine.model().tokenizer, &messages, &tools, &tool_mode_bg) {
                Ok((p, src)) => {
                    tracing::debug!(
                        ?src,
                        tools_active = tools_active(&tool_mode_bg),
                        "chat prompt formatted"
                    );
                    p
                }
                Err(e) => {
                    guard.fail();
                    return Err((StatusCode::BAD_REQUEST, e));
                }
            };
        match engine.generate_prompt_cancellable(
            &prompt,
            &cfg_bg,
            |_| {},
            || cancel.load(Ordering::SeqCst),
        ) {
            Ok(out) => {
                guard.success(
                    out.prompt_tokens.len() as u64,
                    out.completion_tokens.len() as u64,
                );
                if out.finish_reason == FinishReason::Cancelled {
                    tracing::info!(tokens = out.completion_tokens.len(), "completion cancelled");
                }
                Ok(out)
            }
            Err(e) => {
                guard.fail();
                Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
            }
        }
    })
    .await;

    let out = match joined {
        Ok(Ok(out)) => out,
        Ok(Err((code, msg))) => return err_response(code, msg),
        Err(e) => return err_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };

    let outcome = Outcome::from_output(&out, cfg.max_tokens);
    let (message, choice_reason) = build_assistant_message(&out.text, &outcome, &tool_mode);

    Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: model_id,
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason: Some(choice_reason),
        }],
        usage: Usage::from_output(&out),
        timings: Some(timings_json(&out.timings)),
        outcome,
    })
    .into_response()
}

#[derive(serde::Deserialize, Default)]
pub(crate) struct CancelRequest {
    #[serde(default)]
    id: Option<String>,
}

/// `POST /v1/chat/cancel` — stop the in-flight completion (stream or not) within one
/// token, or within one prefill chunk while the prompt is still being prefilled. Body (optional): `{"id": "chatcmpl-…"}` to cancel only that completion.
/// The cancelled completion ends normally with top-level `finish_reason: "user"`.
pub async fn cancel_chat(
    State(state): State<Arc<AppState>>,
    body: Option<Json<CancelRequest>>,
) -> impl IntoResponse {
    let req = body.map(|Json(r)| r).unwrap_or_default();
    let cancelled = state.control.cancel(req.id.as_deref());
    if let Some(id) = &cancelled {
        tracing::info!(%id, "completion cancel requested");
    }
    Json(serde_json::json!({ "cancelled": cancelled.is_some(), "id": cancelled }))
}

fn build_assistant_message(
    text: &str,
    outcome: &Outcome,
    tool_mode: &ToolMode,
) -> (ResponseMessage, &'static str) {
    if tools_active(tool_mode) && outcome.finish_reason != "user" {
        if let Some(tool_calls) = parse_tool_calls(text) {
            return (
                ResponseMessage {
                    role: "assistant",
                    content: None,
                    tool_calls: Some(tool_calls),
                },
                "tool_calls",
            );
        }
        // Required / named but no parseable call → still return text with normal finish.
    }
    (
        ResponseMessage {
            role: "assistant",
            content: Some(text.to_owned()),
            tool_calls: None,
        },
        outcome.openai_finish_reason(),
    )
}

fn build_generate_config(req: &ChatCompletionRequest, limits: &RequestLimits) -> GenerateConfig {
    let max_tokens = req.max_tokens.clamp(1, limits.max_tokens);
    let temperature = req.temperature.unwrap_or(0.7).clamp(0.0, 2.0);
    let top_p = req.top_p.unwrap_or(0.95).clamp(0.0, 1.0);
    let top_k = req.top_k.unwrap_or(40).clamp(0, 100_000);

    let mut stop_strings = req
        .stop
        .as_ref()
        .map(|s| match s {
            StopValue::One(x) => vec![x.clone()],
            StopValue::Many(v) => v.clone(),
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|s| !s.is_empty())
        .take(8)
        .collect::<Vec<_>>();

    if stop_strings.is_empty() {
        stop_strings.extend(["\nUSER:".into(), "\nUser:".into(), "\nASSISTANT:".into()]);
    }
    for s in artificer_runtime::CHAT_CONTROL_STOP_STRINGS {
        if !stop_strings.iter().any(|x| x == *s) {
            stop_strings.push((*s).to_owned());
        }
    }

    GenerateConfig {
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop_on_eos: true,
        seed: req.seed,
        stop_token_ids: Vec::new(),
        stop_strings,
    }
}

/// True per-token SSE via blocking generation thread + mpsc.
///
/// When tools are active, tokens are buffered and emitted as either content
/// deltas or a single `tool_calls` delta (OpenAI-compatible), not raw XML mid-stream.
fn stream_chat(
    state: Arc<AppState>,
    messages: Vec<ChatMessage>,
    tools: Vec<Tool>,
    tool_mode: ToolMode,
    cfg: GenerateConfig,
    model_id: String,
    id: String,
    created: u64,
) -> Response {
    let (tx, rx) = mpsc::channel::<StreamMsg>(64);
    let state_bg = Arc::clone(&state);
    let tools_on = tools_active(&tool_mode);
    let id_bg = id.clone();

    tokio::task::spawn_blocking(move || {
        let guard = state_bg.metrics.begin_request();
        let mut engine = state_bg.engine.blocking_lock();
        let cancel = Arc::new(AtomicBool::new(false));
        let _active = state_bg.control.begin(&id_bg, Arc::clone(&cancel));
        let max_tokens = cfg.max_tokens;
        let send_error = |msg: String| {
            let _ = tx.blocking_send(StreamMsg::Finish {
                choice_reason: None,
                usage: None,
                timings: None,
                outcome: Outcome::error(max_tokens, msg),
            });
        };

        let prompt =
            match messages_to_prompt(&engine.model().tokenizer, &messages, &tools, &tool_mode) {
                Ok((p, src)) => {
                    tracing::debug!(
                        ?src,
                        tools_active = tools_on,
                        "chat prompt formatted (stream)"
                    );
                    p
                }
                Err(e) => {
                    send_error(e);
                    let _ = tx.blocking_send(StreamMsg::DoneMarker);
                    guard.fail();
                    return;
                }
            };

        let _ = tx.blocking_send(StreamMsg::Role);

        // A dropped SSE receiver (client disconnect / aborted fetch) or
        // `POST /v1/chat/cancel` stops decoding before the next token, and prefill
        // before the next chunk (no tokens; final chunk says `finish_reason: "user"`).
        let is_cancelled = || tx.is_closed() || cancel.load(Ordering::SeqCst);
        let result = if tools_on {
            // Buffer full generation so we can emit proper tool_calls shape.
            engine.generate_prompt_cancellable(&prompt, &cfg, |_| {}, is_cancelled)
        } else {
            engine.generate_prompt_cancellable(
                &prompt,
                &cfg,
                |ev| {
                    let StreamEvent::Token { text, .. } = ev;
                    if !text.is_empty() {
                        let _ = tx.blocking_send(StreamMsg::Delta(text));
                    }
                },
                is_cancelled,
            )
        };

        match result {
            Ok(out) => {
                guard.success(
                    out.prompt_tokens.len() as u64,
                    out.completion_tokens.len() as u64,
                );
                let outcome = Outcome::from_output(&out, max_tokens);
                if out.finish_reason == FinishReason::Cancelled {
                    tracing::info!(tokens = out.completion_tokens.len(), "stream cancelled");
                }
                let mut choice_reason = outcome.openai_finish_reason();
                if tools_on {
                    match parse_tool_calls(&out.text) {
                        Some(tcs) if outcome.finish_reason != "user" => {
                            let _ = tx.blocking_send(StreamMsg::ToolCalls(tcs));
                            choice_reason = "tool_calls";
                        }
                        _ => {
                            if !out.text.is_empty() {
                                let _ = tx.blocking_send(StreamMsg::Delta(out.text.clone()));
                            }
                        }
                    }
                }
                // Harmless no-op when the client is gone.
                let _ = tx.blocking_send(StreamMsg::Finish {
                    choice_reason: Some(choice_reason),
                    usage: Some(Usage::from_output(&out)),
                    timings: Some(timings_json(&out.timings)),
                    outcome,
                });
            }
            Err(e) => {
                guard.fail();
                tracing::error!("generation failed mid-stream: {e}");
                send_error(e.to_string());
            }
        }
        let _ = tx.blocking_send(StreamMsg::DoneMarker);
    });

    let stream = stream::unfold(
        (rx, id, model_id, created),
        |(mut rx, id, model_id, created)| async move {
            let msg = rx.recv().await?;
            let event = match msg {
                StreamMsg::Role => {
                    let data = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": { "role": "assistant" },
                            "finish_reason": null
                        }]
                    });
                    Event::default().data(data.to_string())
                }
                StreamMsg::Delta(text) => {
                    let data = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": text },
                            "finish_reason": null
                        }]
                    });
                    Event::default().data(data.to_string())
                }
                StreamMsg::ToolCalls(tcs) => {
                    let data = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": tcs.iter().enumerate().map(|(i, tc)| {
                                    serde_json::json!({
                                        "index": i,
                                        "id": tc.id,
                                        "type": tc.type_,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    })
                                }).collect::<Vec<_>>()
                            },
                            "finish_reason": null
                        }]
                    });
                    Event::default().data(data.to_string())
                }
                StreamMsg::Finish {
                    choice_reason,
                    usage,
                    timings,
                    outcome,
                } => {
                    let data = final_chunk_json(
                        &id,
                        created,
                        &model_id,
                        choice_reason,
                        usage.as_ref(),
                        timings,
                        &outcome,
                    );
                    Event::default().data(data.to_string())
                }
                StreamMsg::DoneMarker => Event::default().data("[DONE]"),
            };
            Some((Ok::<Event, Infallible>(event), (rx, id, model_id, created)))
        },
    );

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

/// Last SSE data event before `[DONE]`: OpenAI `choices[0].finish_reason`, `usage`,
/// `timings`, plus the top-level [`Outcome`] fields (`finish_reason`, `max_tokens`,
/// `tokens`, `thinking_ms`, `prefill_ms`, `limit_hit_in_thinking`, `error`).
fn final_chunk_json(
    id: &str,
    created: u64,
    model_id: &str,
    choice_reason: Option<&str>,
    usage: Option<&Usage>,
    timings: Option<serde_json::Value>,
    outcome: &Outcome,
) -> serde_json::Value {
    let mut data = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": choice_reason
        }],
    });
    let obj = data.as_object_mut().expect("object");
    if let Some(u) = usage {
        obj.insert("usage".into(), serde_json::to_value(u).expect("usage"));
    }
    if let Some(t) = timings {
        obj.insert("timings".into(), t);
    }
    if let serde_json::Value::Object(extra) = serde_json::to_value(outcome).expect("outcome") {
        obj.extend(extra);
    }
    data
}

fn validate_request(req: &ChatCompletionRequest, limits: &RequestLimits) -> Result<(), String> {
    if req.messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    if req.messages.len() > limits.max_messages {
        return Err(format!("too many messages (max {})", limits.max_messages));
    }
    let chars = total_content_chars(&req.messages);
    if chars > limits.max_prompt_chars {
        return Err(format!(
            "prompt too large ({} chars, max {})",
            chars, limits.max_prompt_chars
        ));
    }
    for m in &req.messages {
        if m.role == "tool" && m.tool_call_id.as_deref().unwrap_or("").is_empty() {
            // Soft: OpenAI requires tool_call_id; we warn via reject for strictness.
            return Err("messages with role `tool` require tool_call_id".into());
        }
    }
    Ok(())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use artificer_runtime::BackendKind;

    #[test]
    fn backend_json_cuda_and_fallback() {
        let cuda = BackendStatus {
            requested: BackendKind::Cuda,
            active: BackendKind::Cuda,
            device: Some("NVIDIA GeForce RTX 4050 Laptop GPU".into()),
            fallback_reason: None,
        };
        let j = backend_json(&cuda);
        assert_eq!(j["active"], "cuda");
        assert_eq!(j["device"], "NVIDIA GeForce RTX 4050 Laptop GPU");
        assert!(j["fallback_reason"].is_null());

        let fb = BackendStatus {
            requested: BackendKind::Cuda,
            active: BackendKind::Cpu,
            device: None,
            fallback_reason: Some("no CUDA device".into()),
        };
        let j = backend_json(&fb);
        assert_eq!(j["requested"], "cuda");
        assert_eq!(j["active"], "cpu");
        assert_eq!(j["fallback_reason"], "no CUDA device");
        assert_eq!(backend_json(&BackendStatus::cpu())["active"], "cpu");
    }

    #[test]
    fn timings_json_shape() {
        let t = GenerateTimings {
            prefill_tokens: 31,
            prefill_ms: 100.0,
            decode_tokens: 0,
            decode_ms: 0.0,
        };
        let j = timings_json(&t);
        assert_eq!(j["prefill_tokens"], 31);
        assert_eq!(j["prefill_ms"], 100.0);
        assert_eq!(j["prefill_tok_s"], 310.0);
        assert!(j["decode_tok_s"].is_null());
    }

    #[test]
    fn error_outcome_shape() {
        let o = Outcome::error(512, "boom");
        let j = serde_json::to_value(&o).unwrap();
        assert_eq!(j["finish_reason"], "error");
        assert_eq!(j["max_tokens"], 512);
        assert_eq!(
            j["tokens"],
            serde_json::json!({"thinking": 0, "answer": 0, "total": 0})
        );
        assert_eq!(j["error"]["message"], "boom");
        assert_eq!(j["error"]["type"], "server_error");
        let ok = Outcome {
            error: None,
            finish_reason: "user",
            ..o
        };
        assert!(serde_json::to_value(&ok).unwrap().get("error").is_none());
        assert_eq!(ok.openai_finish_reason(), "stop");
    }

    // ── End-to-end handler tests (CPU, `$ARTIFICER_TEST_MODELS` GGUFs; skip when absent) ──

    fn state_for(name: &str) -> Option<Arc<AppState>> {
        let p = crate::test_common::test_model(name)?;
        let engine = Engine::load(&p).unwrap();
        Some(Arc::new(AppState {
            control: ChatControl::for_engine(&engine),
            backend: tokio::sync::RwLock::new(engine.backend_status().clone()),
            engine: Mutex::new(engine),
            model_id: tokio::sync::RwLock::new("test-model".into()),
            load_opts: EngineLoadOptions::default(),
            limits: RequestLimits::default(),
            metrics: ServerMetrics::default(),
        }))
    }

    fn request(prompt: &str, max_tokens: usize, stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1,
            "stream": stream,
        }))
        .unwrap()
    }

    async fn body_bytes(resp: Response) -> Vec<u8> {
        axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        serde_json::from_slice(&body_bytes(resp).await).unwrap()
    }

    /// SSE `data:` payloads in order.
    async fn sse_payloads(resp: Response) -> Vec<String> {
        let body = String::from_utf8(body_bytes(resp).await).unwrap();
        sse_data(&body)
    }

    fn sse_data(body: &str) -> Vec<String> {
        body.lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(|d| d.trim().to_owned())
            .collect()
    }

    fn final_chunk(payloads: &[String]) -> serde_json::Value {
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        serde_json::from_str(&payloads[payloads.len() - 2]).unwrap()
    }

    /// Top-level outcome fields present on both response modes.
    fn assert_outcome(j: &serde_json::Value, reason: &str, max: usize) {
        assert_eq!(j["finish_reason"], reason, "{j}");
        assert_eq!(j["max_tokens"], max);
        let t = &j["tokens"];
        let total = t["total"].as_u64().unwrap();
        assert_eq!(
            t["thinking"].as_u64().unwrap() + t["answer"].as_u64().unwrap(),
            total
        );
        assert_eq!(j["usage"]["completion_tokens"].as_u64().unwrap(), total);
        assert!(j["thinking_ms"].is_number());
        assert!(j["prefill_ms"].as_f64().unwrap() > 0.0);
        assert!(j["limit_hit_in_thinking"].is_boolean());
        assert!(j.get("error").is_none());
    }

    const SKY: &str = "What color is the sky on a clear day? Answer in one short sentence.";
    const LONG: &str = "Write a very long story about a dragon, at least 500 words.";

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_stream_reports_length_and_stop_with_usage() {
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };

        let j =
            body_json(chat_completions(State(state.clone()), Json(request(SKY, 3, false))).await)
                .await;
        assert_outcome(&j, "length", 3);
        assert_eq!(j["choices"][0]["finish_reason"], "length");
        assert_eq!(
            j["tokens"],
            serde_json::json!({"thinking": 0, "answer": 3, "total": 3})
        );
        assert_eq!(j["thinking_ms"], 0.0);
        assert_eq!(j["limit_hit_in_thinking"], false);
        let u = &j["usage"];
        assert_eq!(u["total_tokens"], u["prompt_tokens"].as_u64().unwrap() + 3);

        let j =
            body_json(chat_completions(State(state), Json(request(SKY, 64, false))).await).await;
        eprintln!("non-stream response: {j}");
        assert_outcome(&j, "stop", 64);
        assert_eq!(j["choices"][0]["finish_reason"], "stop");
        assert!(j["tokens"]["total"].as_u64().unwrap() < 64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_final_chunk_carries_finish_and_usage() {
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        for (max, reason) in [(3usize, "length"), (64, "stop")] {
            let resp = chat_completions(State(state.clone()), Json(request(SKY, max, true))).await;
            let payloads = sse_payloads(resp).await;
            let last = final_chunk(&payloads);
            assert_outcome(&last, reason, max);
            assert_eq!(last["choices"][0]["finish_reason"], reason);
            assert_eq!(last["tokens"]["thinking"], 0);
            let n = last["tokens"]["total"].as_u64().unwrap();
            if reason == "length" {
                assert_eq!(n, 3);
            } else {
                assert!(n < 64);
            }
            assert!(last["timings"]["decode_ms"].is_number());
            // Earlier chunks carry no outcome.
            for p in &payloads[..payloads.len() - 2] {
                let c: serde_json::Value = serde_json::from_str(p).unwrap();
                assert!(c["choices"][0]["finish_reason"].is_null());
                assert!(c.get("usage").is_none());
                assert!(c.get("finish_reason").is_none());
                assert!(c.get("tokens").is_none());
            }
        }
    }

    /// Nemotron starts in `<think>`: a 3-token cap ends mid-think, and the final
    /// stream chunk says so.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_cap_mid_think_reports_thinking() {
        let Some(state) = state_for("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf") else {
            return;
        };
        let resp = chat_completions(State(state), Json(request("What is 17 * 24?", 3, true))).await;
        let last = final_chunk(&sse_payloads(resp).await);
        eprintln!("final chunk: {last}");
        assert_outcome(&last, "length", 3);
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(
            last["tokens"],
            serde_json::json!({"thinking": 3, "answer": 0, "total": 3})
        );
        assert_eq!(last["limit_hit_in_thinking"], true);
        assert!(last["thinking_ms"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn models_report_template_opens_think() {
        for (name, opens) in [
            ("tiny-vicuna-1b.q4_k_m.gguf", false),
            ("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf", true),
        ] {
            let Some(state) = state_for(name) else {
                continue;
            };
            let j: serde_json::Value =
                body_json(list_models(State(state)).await.into_response()).await;
            assert_eq!(j["data"][0]["template_opens_think"], opens, "{name}: {j}");
        }
    }

    /// A generation error in a stream ends with a final chunk carrying
    /// `finish_reason: "error"` and an `error` message (here: prompt > context).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_generation_error_reports_error_outcome() {
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        let huge = "hello ".repeat(9000);
        let resp = chat_completions(State(state), Json(request(&huge, 8, true))).await;
        let last = final_chunk(&sse_payloads(resp).await);
        assert_eq!(last["finish_reason"], "error", "{last}");
        assert_eq!(last["max_tokens"], 8);
        assert!(last["choices"][0]["finish_reason"].is_null());
        assert!(!last["error"]["message"].as_str().unwrap().is_empty());
        assert_eq!(last["error"]["type"], "server_error");
        assert!(last.get("usage").is_none());
    }

    /// Dropping the SSE body (client disconnect / user stop) cancels decoding.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_client_disconnect_cancels_generation() {
        use futures_util::StreamExt;
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        let resp = chat_completions(State(state.clone()), Json(request(LONG, 400, true))).await;
        let mut body = resp.into_body().into_data_stream();
        // Role chunk + first content delta, then hang up.
        let mut seen = String::new();
        while !seen.contains("\"content\"") {
            let frame = body.next().await.expect("stream ended early").unwrap();
            seen.push_str(&String::from_utf8_lossy(&frame));
        }
        drop(body);
        // The generation thread holds the engine lock until it stops.
        let t0 = std::time::Instant::now();
        let _engine = state.engine.lock().await;
        let done = state.metrics.snapshot();
        assert!(t0.elapsed() < std::time::Duration::from_secs(60));
        eprintln!("cancelled after {} tokens", done.completion_tokens_total);
        assert_eq!(done.requests_completed, 1);
        assert!(
            done.completion_tokens_total < 50,
            "generation was not cancelled: {} tokens",
            done.completion_tokens_total
        );
    }

    async fn wait_active(state: &AppState) {
        for _ in 0..600 {
            if state.control.active.lock().unwrap().is_some() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("generation never started");
    }

    /// `POST /v1/chat/cancel` stops a live stream; the client still gets the final
    /// chunk with `finish_reason: "user"`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_endpoint_stops_stream_with_user_reason() {
        use futures_util::StreamExt;
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        let resp = chat_completions(State(state.clone()), Json(request(LONG, 400, true))).await;
        let mut body = resp.into_body().into_data_stream();
        let mut seen = String::new();
        while !seen.contains("\"content\"") {
            let frame = body.next().await.expect("stream ended early").unwrap();
            seen.push_str(&String::from_utf8_lossy(&frame));
        }
        // Wrong id: no-op.
        let j = body_json(
            cancel_chat(
                State(state.clone()),
                Some(Json(CancelRequest {
                    id: Some("nope".into()),
                })),
            )
            .await
            .into_response(),
        )
        .await;
        assert_eq!(j["cancelled"], false);
        let j = body_json(
            cancel_chat(State(state.clone()), None)
                .await
                .into_response(),
        )
        .await;
        assert_eq!(j["cancelled"], true);
        assert!(j["id"].as_str().unwrap().starts_with("chatcmpl-"));
        while let Some(frame) = body.next().await {
            seen.push_str(&String::from_utf8_lossy(&frame.unwrap()));
        }
        let last = final_chunk(&sse_data(&seen));
        assert_outcome(&last, "user", 400);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        let n = last["tokens"]["total"].as_u64().unwrap();
        eprintln!("cancel endpoint: stopped after {n} tokens");
        assert!(n < 50, "not cancelled: {n} tokens");
        // Nothing in flight any more.
        let j = body_json(cancel_chat(State(state), None).await.into_response()).await;
        assert_eq!(j["cancelled"], false);
    }

    /// A Stop during a long prefill ends it within one chunk (one token on CPU):
    /// the stream's final chunk says `finish_reason: "user"` with no tokens, only part of
    /// the prompt was prefilled, and the next request on the same engine works.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_during_long_prefill_ends_stream_with_no_tokens() {
        use futures_util::StreamExt;
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        // ≈1500 prompt tokens: tens of seconds of CPU prefill in release, far more in debug.
        let long = "The quick brown fox jumps over the lazy dog. ".repeat(130);
        let resp = chat_completions(State(state.clone()), Json(request(&long, 64, true))).await;
        let mut body = resp.into_body().into_data_stream();
        wait_active(&state).await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let t0 = std::time::Instant::now();
        assert!(state.control.cancel(None).is_some());
        let mut seen = String::new();
        while let Some(frame) = body.next().await {
            seen.push_str(&String::from_utf8_lossy(&frame.unwrap()));
        }
        let stop_s = t0.elapsed().as_secs_f64();
        let last = final_chunk(&sse_data(&seen));
        eprintln!("stop during prefill: final chunk after {stop_s:.2} s: {last}");
        assert_outcome(&last, "user", 64);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            last["tokens"],
            serde_json::json!({"thinking": 0, "answer": 0, "total": 0})
        );
        let prompt_tokens = last["usage"]["prompt_tokens"].as_u64().unwrap();
        let prefilled = last["timings"]["prefill_tokens"].as_u64().unwrap();
        assert!(prompt_tokens > 1000, "prompt only {prompt_tokens} tokens");
        assert!(
            prefilled < prompt_tokens,
            "prefill was not interrupted: {prefilled}/{prompt_tokens}"
        );
        assert!(
            !seen.contains("\"content\""),
            "no text after a Stop during prefill"
        );
        // Nothing in flight; the engine is reusable.
        let j = body_json(
            cancel_chat(State(state.clone()), None)
                .await
                .into_response(),
        )
        .await;
        assert_eq!(j["cancelled"], false);
        let j = body_json(chat_completions(State(state), Json(request(SKY, 3, false))).await).await;
        assert_outcome(&j, "length", 3);
    }

    /// Non-streaming: the cancel endpoint yields `finish_reason: "user"`, and a
    /// dropped handler future (client hang-up) also stops the generation.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_stream_cancel_and_hangup() {
        let Some(state) = state_for("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };

        let task = tokio::spawn(chat_completions(
            State(state.clone()),
            Json(request(LONG, 400, false)),
        ));
        wait_active(&state).await;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(state.control.cancel(None).is_some());
        let j = body_json(task.await.unwrap()).await;
        assert_outcome(&j, "user", 400);
        assert_eq!(j["choices"][0]["finish_reason"], "stop");
        let n = j["tokens"]["total"].as_u64().unwrap();
        assert!(n < 400, "not cancelled: {n}");

        // Hang-up: drop the handler future mid-generation.
        let before = state.metrics.snapshot().completion_tokens_total;
        let fut = chat_completions(State(state.clone()), Json(request(LONG, 400, false)));
        let st = state.clone();
        let dropper = tokio::spawn(async move {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(3600), fut).await;
        });
        wait_active(&st).await;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        dropper.abort();
        let _ = dropper.await;
        let t0 = std::time::Instant::now();
        let _engine = state.engine.lock().await;
        let n = state.metrics.snapshot().completion_tokens_total - before;
        eprintln!("hang-up: stopped after {n} tokens in {:?}", t0.elapsed());
        assert!(n < 400, "hang-up did not cancel: {n} tokens");
        assert!(t0.elapsed() < std::time::Duration::from_secs(30));
    }
}
