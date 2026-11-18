//! Calls the `artificer-client` crate (`crates/client`). No second HTTP client.
//! `GET /` is 404 by design and is never requested. The base URL is the runtime's
//! (`server::Runtime::base_url`): a reused server on `127.0.0.1:8080` or the built-in one.
//!
//! Streaming and the loaded-model id are called only when `crates/client/src/lib.rs`
//! defines them (`chat_completion_stream`, `loaded_model`). This crate does not
//! invent either call.

use std::path::Path;

use std::sync::Mutex;
use std::time::Duration;

use artificer_client::{
    ChatCompletionRequest, ChatCompletionResponse, HealthStatus, McpToolsList, StreamCancel,
};
use artificer_core::{DEFAULT_MAX_TOKENS as CHAT_MAX_TOKENS, MAX_TOKENS_CEILING};
use serde::{Deserialize, Serialize};

/// IPC copy of [`HealthStatus`]. Same fields, no extras. The client type
/// deserializes but does not serialize.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HealthView {
    pub status: String,
    pub metrics: MetricsView,
    /// `GET /health` `backend` object. `None` when the server predates it;
    /// the window then says the backend is unknown instead of showing CPU.
    pub backend: Option<BackendView>,
}

/// IPC copy of [`artificer_client::BackendInfo`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendView {
    pub requested: Option<String>,
    pub active: String,
    pub device: Option<String>,
    pub fallback_reason: Option<String>,
}

/// IPC copy of [`artificer_client::ChatTimings`] (server prefill/decode split).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimingsView {
    pub prefill_tokens: u64,
    pub prefill_ms: f64,
    pub decode_tokens: u64,
    pub decode_ms: f64,
}

/// IPC copy of [`artificer_client::HealthMetrics`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetricsView {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub in_flight: u64,
    pub peak_in_flight: u64,
}

/// IPC copy of [`artificer_client::ChatTokenCounts`] (proposed done-event split).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokensView {
    pub thinking: Option<u64>,
    pub answer: Option<u64>,
    pub total: Option<u64>,
}

/// Assistant text from [`ChatCompletionResponse`], plus the `model` id that response carried.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatTurn {
    pub content: String,
    pub model: String,
    /// Server timings from the final chunk; `None` from older servers.
    pub timings: Option<TimingsView>,
    /// Done-event `finish_reason`, else `choices[0].finish_reason`.
    pub finish_reason: Option<String>,
    /// Done-event `max_tokens`, else the limit this bridge requested.
    pub max_tokens: Option<u64>,
    /// Done-event token split; `None` until the server sends it. The window never counts.
    pub tokens: Option<TokensView>,
    /// Server-measured thinking time (prefill excluded).
    pub thinking_ms: Option<f64>,
    /// Server total for the readout: top-level `completion_tokens` alias, else `usage.completion_tokens`.
    pub completion_tokens: Option<u64>,
    /// Tolerated alias for `tokens.thinking`.
    pub thinking_tokens: Option<u64>,
    /// Prompt prefill wall time.
    pub prefill_ms: Option<f64>,
    /// `finish_reason == "length"` while still thinking.
    pub limit_hit_in_thinking: Option<bool>,
    /// `{message}` when `finish_reason == "error"`.
    pub error: Option<ErrorView>,
    /// Stop's grace period ran out and the bridge cut the stream: no server counts.
    pub connection_dropped: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ErrorView {
    pub message: String,
}

/// What `stop_stream` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StopView {
    /// A stream was running and was asked to stop.
    pub requested: bool,
    /// The server acknowledged `POST /v1/chat/cancel`.
    pub server_ack: bool,
    /// The final chunk did not arrive within the grace period; the connection was cut.
    pub dropped: bool,
}

/// `/v1/models` id plus the optional `template_opens_think` flag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelInfoView {
    pub id: String,
    pub template_opens_think: Option<bool>,
}

/// Compile-time view of the two optional client methods this build may call. JSON is snake_case, matching the other commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClientCaps {
    pub stream: bool,
    pub loaded_model: bool,
}

/// `ARTIFICER_CHAT_MAX_TOKENS` (clamped to 1..=`MAX_TOKENS_CEILING`, the server's hard cap),
/// else the shared `DEFAULT_MAX_TOKENS`. Lets the live Stop shot ask for a reply long enough
/// to press Stop mid-decode. Unparseable or 0 falls back to the default.
fn parse_max_tokens(v: Option<&str>) -> u32 {
    v.and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n >= 1)
        .map(|n| n.min(u64::from(MAX_TOKENS_CEILING)) as u32)
        .unwrap_or(CHAT_MAX_TOKENS)
}

fn chat_max_tokens() -> u32 {
    parse_max_tokens(std::env::var("ARTIFICER_CHAT_MAX_TOKENS").ok().as_deref())
}
/// After Stop, how long to wait for the server's final chunk before cutting the stream.
/// The server stops a prefill within one chunk, well inside this; an older
/// server that only stops after prefill may hit it on a long prompt.
const STOP_GRACE: Duration = Duration::from_secs(5);

/// The running stream's stop handle (one stream at a time, like the window).
static ACTIVE: Mutex<Option<StreamCancel>> = Mutex::new(None);

fn client() -> artificer_client::Client {
    artificer_client::Client::with_base_url(crate::server::app().base_url())
        .unwrap_or_else(|_| artificer_client::Client::new())
}

pub(crate) fn health_view(status: HealthStatus) -> HealthView {
    let m = status.metrics;
    HealthView {
        backend: status.backend.map(|b| BackendView {
            requested: b.requested,
            active: b.active,
            device: b.device,
            fallback_reason: b.fallback_reason,
        }),
        status: status.status,
        metrics: MetricsView {
            requests_started: m.requests_started,
            requests_completed: m.requests_completed,
            requests_failed: m.requests_failed,
            prompt_tokens_total: m.prompt_tokens_total,
            completion_tokens_total: m.completion_tokens_total,
            in_flight: m.in_flight,
            peak_in_flight: m.peak_in_flight,
        },
    }
}

/// One turn of the window's chat history (`messages` of `chat_complete` / `chat_stream`).
/// Assistant turns are the answer only: the window drops the thinking.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    pub content: String,
}

/// What a chat call sends: the window's history, or one user turn (`&str`).
pub trait IntoHistory {
    fn into_history(self) -> Vec<ChatMessageIn>;
}

impl IntoHistory for &str {
    fn into_history(self) -> Vec<ChatMessageIn> {
        vec![ChatMessageIn {
            role: "user".into(),
            content: self.into(),
        }]
    }
}

impl IntoHistory for Vec<ChatMessageIn> {
    fn into_history(self) -> Vec<ChatMessageIn> {
        self
    }
}

/// The history the server may get: roles `user` and `assistant` only, no empty turn, and the
/// user's new message last. Checked before any request is made.
pub(crate) fn check_history(history: &[ChatMessageIn]) -> Result<(), String> {
    if let Some(m) = history
        .iter()
        .find(|m| m.role != "user" && m.role != "assistant")
    {
        return Err(format!("chat history: unsupported role {:?}", m.role));
    }
    if history.iter().any(|m| m.content.trim().is_empty()) {
        return Err("chat history: empty turn".into());
    }
    match history.last() {
        Some(last) if last.role == "user" => Ok(()),
        Some(_) => Err("chat history must end with a user turn".into()),
        None => Err("chat history is empty".into()),
    }
}

/// `history` as the request's `messages`, in order, non-streaming, capped at the default.
pub(crate) fn user_request(history: impl IntoHistory) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: None,
        messages: history
            .into_history()
            .into_iter()
            .map(|m| artificer_client::ChatMessage {
                role: m.role,
                content: Some(m.content),
            })
            .collect(),
        max_tokens: Some(chat_max_tokens()),
        temperature: None,
        stream: false,
    }
}

/// Same turns as [`user_request`], with `stream: true` for
/// `Client::chat_completion_stream` only. Never passed to `chat_completion`.
pub(crate) fn stream_request(history: impl IntoHistory) -> ChatCompletionRequest {
    let mut request = user_request(history);
    request.stream = true;
    request
}

pub(crate) fn assistant_text(response: &ChatCompletionResponse) -> Result<String, String> {
    response
        .choices
        .iter()
        .find_map(|choice| choice.message.content.clone())
        .ok_or_else(|| "chat completion returned no assistant text".to_string())
}

fn turn_from(response: &ChatCompletionResponse) -> Result<ChatTurn, String> {
    let ended_early = response.connection_dropped
        || matches!(
            response.finish_reason.as_deref(),
            Some("user" | "cancelled" | "error")
        );
    let content = match assistant_text(response) {
        Ok(text) => text,
        // A Stop during prefill (or an error) legitimately has no text.
        Err(_) if ended_early => String::new(),
        Err(err) => return Err(err),
    };
    Ok(ChatTurn {
        content,
        model: response.model.clone(),
        timings: response.timings.as_ref().map(|t| TimingsView {
            prefill_tokens: t.prefill_tokens,
            prefill_ms: t.prefill_ms,
            decode_tokens: t.decode_tokens,
            decode_ms: t.decode_ms,
        }),
        finish_reason: response.finish_reason.clone().or_else(|| {
            response
                .choices
                .first()
                .and_then(|c| c.finish_reason.clone())
        }),
        max_tokens: response.max_tokens.or(Some(u64::from(chat_max_tokens()))),
        tokens: response.tokens.as_ref().map(|t| TokensView {
            thinking: t.thinking,
            answer: t.answer,
            total: t.total,
        }),
        thinking_ms: response.thinking_ms,
        // A stream without a usage chunk reads as all-zero usage: treat that as absent.
        completion_tokens: response
            .completion_tokens
            .or((response.usage.total_tokens > 0).then_some(response.usage.completion_tokens)),
        thinking_tokens: response.thinking_tokens,
        prefill_ms: response.prefill_ms,
        limit_hit_in_thinking: response.limit_hit_in_thinking,
        error: response.error.as_ref().map(|e| ErrorView {
            message: e.message.clone(),
        }),
        connection_dropped: response.connection_dropped,
    })
}

/// Stream one turn on `c`, registering a stop handle for [`stop_active`].
pub(crate) fn stream_turn(
    c: &artificer_client::Client,
    history: impl IntoHistory,
    mut on_delta: impl FnMut(&str),
) -> Result<ChatTurn, String> {
    let history = history.into_history();
    check_history(&history)?;
    let handle = StreamCancel::new();
    *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle.clone());
    let request = stream_request(history);
    let result = c.chat_completion_stream_cancellable(&request, &handle, |d| on_delta(d));
    {
        let mut g = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        if g.as_ref().is_some_and(|h| h.is_finished()) {
            *g = None;
        }
    }
    turn_from(&result.map_err(|err| err.to_string())?)
}

/// Stop the running stream: `POST /v1/chat/cancel {id}`, wait up to `grace` for the
/// final chunk (which `stream_turn` returns to the window), else cut the connection.
pub(crate) fn stop_active(c: &artificer_client::Client, grace: Duration) -> StopView {
    let handle = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match handle {
        Some(h) if !h.is_finished() => {
            let o = h.cancel(c, grace);
            StopView {
                requested: true,
                server_ack: o.server_ack,
                dropped: o.dropped,
            }
        }
        _ => StopView {
            requested: false,
            server_ack: false,
            dropped: false,
        },
    }
}

/// Window Stop button. Blocks up to [`STOP_GRACE`]; run it off the main thread.
pub fn stop_stream() -> StopView {
    stop_active(&client(), STOP_GRACE)
}

pub(crate) fn tool_names(list: McpToolsList) -> Vec<String> {
    list.tools.into_iter().map(|tool| tool.name).collect()
}

pub fn client_caps() -> ClientCaps {
    ClientCaps {
        stream: cfg!(artificer_client_has_chat_completion_stream),
        loaded_model: cfg!(artificer_client_has_loaded_model),
    }
}

/// `Client::new().health()`.
pub fn check_health() -> Result<HealthView, String> {
    let status = client().health().map_err(|err| err.to_string())?;
    Ok(health_view(status))
}

/// `Client::chat_completion` with the window's history (see [`user_request`]).
pub fn chat_complete(history: impl IntoHistory) -> Result<ChatTurn, String> {
    let history = history.into_history();
    check_history(&history)?;
    let request = user_request(history);
    let response = client()
        .chat_completion(&request)
        .map_err(|err| err.to_string())?;
    turn_from(&response)
}

/// `Client::chat_completion_stream(&request, on_delta)` with
/// `on_delta: impl FnMut(&str)`. Each delta is sent on `on_delta` as it arrives.
/// Absent from the client, this does not call the server and does not emit chunks.
pub fn chat_stream(
    on_delta: tauri::ipc::Channel<String>,
    messages: Vec<ChatMessageIn>,
) -> Result<ChatTurn, String> {
    #[cfg(artificer_client_has_chat_completion_stream)]
    {
        chat_stream_live(on_delta, messages)
    }
    #[cfg(not(artificer_client_has_chat_completion_stream))]
    {
        let _ = (on_delta, messages);
        Err(
            "missing Client::chat_completion_stream(&self, request: &ChatCompletionRequest, on_delta: impl FnMut(&str)) -> Result<ChatCompletionResponse, ClientError>"
                .into(),
        )
    }
}

#[cfg(artificer_client_has_chat_completion_stream)]
fn chat_stream_live(
    on_delta: tauri::ipc::Channel<String>,
    messages: Vec<ChatMessageIn>,
) -> Result<ChatTurn, String> {
    stream_turn(&client(), messages, |delta| {
        let _ = on_delta.send(delta.to_string());
    })
}

/// `POST /v1/models/load` with an absolute `.gguf` path, then the id from
/// `Client::loaded_model` (`GET /v1/models`, first `data[].id`).
/// The returned string is that id, never the path. Failure does not invent an id.
pub fn load_local_weights(path: &str) -> Result<String, String> {
    let path = checked_weights_path(path.trim())?;
    let c = client();
    c.load_model(Path::new(&path))
        .map_err(|err| err.to_string())?;
    match c.loaded_model().map_err(|err| err.to_string())? {
        Some(id) if !id.is_empty() => Ok(id),
        _ => Err("GET /v1/models returned no model id".into()),
    }
}

/// `Client::loaded_model(&self) -> Result<Option<String>, ClientError>` when present.
pub fn loaded_model() -> Result<Option<String>, String> {
    #[cfg(artificer_client_has_loaded_model)]
    {
        client().loaded_model().map_err(|err| err.to_string())
    }
    #[cfg(not(artificer_client_has_loaded_model))]
    {
        Err("missing Client::loaded_model(&self) -> Result<Option<String>, ClientError>".into())
    }
}

/// `Client::loaded_model_info`: id + `template_opens_think` (None when not reported).
pub fn loaded_model_info() -> Result<Option<ModelInfoView>, String> {
    let info = client()
        .loaded_model_info()
        .map_err(|err| err.to_string())?;
    Ok(info.map(|m| ModelInfoView {
        id: m.id,
        template_opens_think: m.template_opens_think,
    }))
}

/// `Client::mcp_tools_list`, names only.
pub fn mcp_tools_list() -> Result<Vec<String>, String> {
    let list = client().mcp_tools_list().map_err(|err| err.to_string())?;
    Ok(tool_names(list))
}

pub(crate) fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

pub(crate) fn gguf_files_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && is_gguf(&path) {
            if let Some(text) = path.to_str() {
                out.push(text.to_string());
            }
        }
    }
    out.sort();
    out
}

/// Optional weights directory from `ARTIFICER_WEIGHTS_DIR` (unset or empty: none).
fn weights_dir_env() -> Option<std::path::PathBuf> {
    std::env::var_os("ARTIFICER_WEIGHTS_DIR")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
}

/// The one weights directory the desktop app uses: `ARTIFICER_WEIGHTS_DIR` when set, else
/// `~/models` when that directory exists, else none. Never a developer checkout such as
/// `$HOME/Projects/Artificer`. Shared by the `.gguf` list and the file chooser.
fn weights_dir(
    weights_dir: Option<&Path>,
    home: Option<&str>,
    is_dir: impl Fn(&Path) -> bool,
) -> Option<std::path::PathBuf> {
    if let Some(dir) = weights_dir {
        return Some(dir.to_path_buf());
    }
    let models = Path::new(home.filter(|h| !h.is_empty())?).join("models");
    is_dir(&models).then_some(models)
}

/// Non-recursive, sorted `.gguf` files in [`weights_dir`]; empty when there is none.
fn weight_files_in(
    weights_dir_override: Option<&Path>,
    home: Option<&str>,
    is_dir: impl Fn(&Path) -> bool,
) -> Vec<String> {
    weights_dir(weights_dir_override, home, is_dir)
        .map(|dir| gguf_files_in(&dir))
        .unwrap_or_default()
}

/// `.gguf` files already on disk, from the same directory the file chooser opens in.
pub fn local_weight_files() -> Vec<String> {
    weight_files_in(
        weights_dir_env().as_deref(),
        user_home().as_deref(),
        Path::is_dir,
    )
}

/// The user's home directory: `HOME` when set and non-empty; on Windows, where `HOME` is
/// normally unset, `USERPROFILE` instead.
fn home_from(home: Option<&str>, userprofile: Option<&str>, windows: bool) -> Option<String> {
    let non_empty = |v: Option<&str>| v.filter(|v| !v.is_empty()).map(str::to_string);
    non_empty(home).or_else(|| {
        if windows {
            non_empty(userprofile)
        } else {
            None
        }
    })
}

fn user_home() -> Option<String> {
    home_from(
        std::env::var("HOME").ok().as_deref(),
        std::env::var("USERPROFILE").ok().as_deref(),
        cfg!(windows),
    )
}

/// Error text `pick_local_weights` returns when this system has no file picker at all.
/// The window matches on it and shows the paste-a-path hint instead of the error.
pub const PICKER_UNAVAILABLE: &str = "picker-unavailable";

/// A picked or typed path the window may load: absolute, an existing file, `.gguf`.
/// Returns the path unchanged; the error is one short line for the window. Each error starts
/// with a fixed prefix the window matches (`isPathError` in `dist/app-state.js`).
pub(crate) fn checked_weights_path(path: &str) -> Result<String, String> {
    let file = Path::new(path);
    if !file.is_absolute() {
        return Err("model path must be absolute".into());
    }
    let meta = std::fs::metadata(file).map_err(|err| format!("no file at this path ({err})"))?;
    if !meta.is_file() || !is_gguf(file) {
        return Err("only an existing .gguf file can be selected".into());
    }
    Ok(path.to_string())
}

/// What one CHANGE click ended in. rfd (Linux) returns "no file" both for Cancel and when
/// neither the XDG portal nor its `zenity` fallback could open, so a `None` is only a Cancel
/// when some picker was actually there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PickOutcome<T> {
    Picked(T),
    Cancelled,
    Unavailable,
}

/// Pure decision for [`PickOutcome`]: a picked file always wins; otherwise `None` is a Cancel
/// when the portal FileChooser or `zenity` is present, and "no file picker" when neither is.
pub(crate) fn pick_outcome<T>(
    portal_present: bool,
    zenity_present: bool,
    picked: Option<T>,
) -> PickOutcome<T> {
    match picked {
        Some(file) => PickOutcome::Picked(file),
        None if portal_present || zenity_present => PickOutcome::Cancelled,
        None => PickOutcome::Unavailable,
    }
}

/// Interface the portal object must expose for rfd's portal dialog to work.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PORTAL_FILE_CHOOSER: &str = "org.freedesktop.portal.FileChooser";
/// Each probe command gets this long; a hung bus counts as "no portal".
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);

/// True when `gdbus introspect` / `busctl introspect` output lists the FileChooser interface.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn lists_file_chooser(introspection: &str) -> bool {
    introspection.lines().any(|line| {
        line.split_whitespace()
            .take(2)
            .any(|word| word == PORTAL_FILE_CHOOSER)
    })
}

/// Result of running one probe command.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
enum Probe {
    /// The program is not installed (spawn failed).
    Missing,
    /// It ran; stdout when it exited 0 within the deadline, else `None` (failed or killed).
    Ran(Option<String>),
}

/// Spawn `program args`, wait at most `deadline`, and kill only that child if it overruns.
/// stdout is read on a helper thread so a large reply cannot block the child.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn run_with_deadline(program: &str, args: &[&str], deadline: Duration) -> Probe {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    else {
        return Probe::Missing;
    };
    let reader = child.stdout.take().map(|mut out| {
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = out.read_to_string(&mut text);
            text
        })
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() < deadline => {
                std::thread::sleep(Duration::from_millis(10))
            }
            _ => {
                // Our own child only: overran (or try_wait failed), so stop it and reap it.
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let text = reader.and_then(|h| h.join().ok()).unwrap_or_default();
    match status {
        Some(status) if status.success() => Probe::Ran(Some(text)),
        _ => Probe::Ran(None),
    }
}

/// Linux: does the session bus have the portal FileChooser? `gdbus introspect`, else
/// `busctl --user introspect` when gdbus is not installed; neither installed, a failure or a
/// timeout all count as absent.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn portal_file_chooser_present() -> bool {
    portal_file_chooser_present_with("gdbus", "busctl", PROBE_TIMEOUT)
}

/// Arguments for `gdbus` (the first probe).
const GDBUS_ARGS: [&str; 6] = [
    "introspect",
    "--session",
    "--dest",
    "org.freedesktop.portal.Desktop",
    "--object-path",
    "/org/freedesktop/portal/desktop",
];
/// Arguments for `busctl`, used only when `gdbus` is not installed.
const BUSCTL_ARGS: [&str; 4] = [
    "--user",
    "introspect",
    "org.freedesktop.portal.Desktop",
    "/org/freedesktop/portal/desktop",
];

/// [`portal_file_chooser_present`] with the program names and deadline injected (tests run
/// fake programs through the same spawn path).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn portal_file_chooser_present_with(gdbus: &str, busctl: &str, deadline: Duration) -> bool {
    let reply = match run_with_deadline(gdbus, &GDBUS_ARGS, deadline) {
        Probe::Ran(reply) => reply,
        Probe::Missing => match run_with_deadline(busctl, &BUSCTL_ARGS, deadline) {
            Probe::Ran(reply) => reply,
            Probe::Missing => None,
        },
    };
    reply.is_some_and(|text| lists_file_chooser(&text))
}

/// `program` as a file in one of the `PATH` directories.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn on_path(
    program: &str,
    path_var: Option<&std::ffi::OsStr>,
    is_file: impl Fn(&Path) -> bool,
) -> bool {
    path_var.is_some_and(|p| std::env::split_paths(p).any(|dir| is_file(&dir.join(program))))
}

/// (portal FileChooser present, `zenity` on PATH). Not Linux: the native dialog always exists.
fn picker_probes() -> (bool, bool) {
    #[cfg(target_os = "linux")]
    {
        (
            portal_file_chooser_present(),
            on_path("zenity", std::env::var_os("PATH").as_deref(), Path::is_file),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        (true, true)
    }
}

/// Native file picker (tauri-plugin-dialog: Windows/macOS native dialog, Linux XDG portal
/// with rfd's zenity fallback) for one `.gguf` that already exists. Opens in [`weights_dir`]
/// when there is one, else the dialog's own default. Cancel is `Ok(None)`; on Linux with
/// neither the portal FileChooser nor `zenity` it is `Err(PICKER_UNAVAILABLE)`, checked before
/// opening (so the window answers at once) and again after a `None` (the authority).
///
/// Blocks until the dialog closes, so call it only from an `async` command (never the main
/// thread: the plugin runs the dialog there and waits for it).
pub fn pick_local_weights<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<Option<String>, String> {
    let Some(picked) = pick_flow(picker_probes, || open_dialog(app))? else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|err| err.to_string())?;
    let text = path
        .to_str()
        .ok_or_else(|| "the picked path is not valid UTF-8".to_string())?;
    checked_weights_path(text).map(Some)
}

/// The probe-and-pick sequence of [`pick_local_weights`], with the probes and the dialog
/// injected: probe first (no picker → `Err(PICKER_UNAVAILABLE)` without opening anything),
/// open, and after a `None` probe again; that second answer decides Cancel vs unavailable.
pub(crate) fn pick_flow<F>(
    probes: impl Fn() -> (bool, bool),
    open: impl FnOnce() -> Option<F>,
) -> Result<Option<F>, String> {
    let (portal, zenity) = probes();
    if pick_outcome(portal, zenity, None::<()>) == PickOutcome::Unavailable {
        return Err(PICKER_UNAVAILABLE.into());
    }
    let picked = open();
    let (portal, zenity) = if picked.is_some() {
        (portal, zenity)
    } else {
        probes()
    };
    match pick_outcome(portal, zenity, picked) {
        PickOutcome::Picked(file) => Ok(Some(file)),
        PickOutcome::Cancelled => Ok(None),
        PickOutcome::Unavailable => Err(PICKER_UNAVAILABLE.into()),
    }
}

/// Builds and opens the native dialog; blocks until it closes.
fn open_dialog<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Option<tauri_plugin_dialog::FilePath> {
    use tauri::Manager;
    use tauri_plugin_dialog::DialogExt;

    let mut dialog = app
        .dialog()
        .file()
        .set_title("Weights")
        .add_filter("GGUF", &["gguf", "GGUF"]);
    if let Some(dir) = weights_dir(
        weights_dir_env().as_deref(),
        user_home().as_deref(),
        Path::is_dir,
    ) {
        dialog = dialog.set_directory(dir);
    }
    if let Some(window) = app.get_webview_window("main") {
        dialog = dialog.set_parent(&window);
    }
    dialog.blocking_pick_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use artificer_client::{
        BackendInfo, ChatChoice, ChatMessage, ChatResponseMessage, ChatTimings, ChatUsage,
        HealthMetrics, McpTool,
    };

    #[test]
    fn health_view_copies_backend_and_fallback() {
        let metrics = || HealthMetrics {
            requests_started: 0,
            requests_completed: 0,
            requests_failed: 0,
            prompt_tokens_total: 0,
            completion_tokens_total: 0,
            in_flight: 0,
            peak_in_flight: 0,
        };
        let view = health_view(HealthStatus {
            status: "ok".into(),
            metrics: metrics(),
            backend: Some(BackendInfo {
                requested: Some("cuda".into()),
                active: "cpu".into(),
                device: None,
                fallback_reason: Some("no CUDA device".into()),
            }),
        });
        let b = view.backend.expect("backend");
        assert_eq!(b.active, "cpu");
        assert_eq!(b.requested.as_deref(), Some("cuda"));
        assert_eq!(b.fallback_reason.as_deref(), Some("no CUDA device"));
        let json = serde_json::to_value(health_view(HealthStatus {
            status: "ok".into(),
            metrics: metrics(),
            backend: Some(BackendInfo {
                requested: Some("cuda".into()),
                active: "cuda".into(),
                device: Some("NVIDIA GeForce RTX 4050 Laptop GPU".into()),
                fallback_reason: None,
            }),
        }))
        .unwrap();
        assert_eq!(
            json["backend"]["device"],
            "NVIDIA GeForce RTX 4050 Laptop GPU"
        );
        assert!(json["backend"]["fallback_reason"].is_null());
    }
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn user_request_is_one_non_streaming_turn_capped_at_default() {
        let request = user_request(vec![turn("user", "Run a local inference check.")]);
        assert!(!request.stream);
        assert_eq!(request.max_tokens, Some(CHAT_MAX_TOKENS));
        assert!(request.model.is_none());
        assert_eq!(
            request.messages,
            vec![ChatMessage {
                role: "user".into(),
                content: Some("Run a local inference check.".into()),
            }]
        );
    }

    fn turn(role: &str, content: &str) -> ChatMessageIn {
        ChatMessageIn {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn history_becomes_messages_in_order_with_their_roles() {
        let history: Vec<ChatMessageIn> = serde_json::from_str(
            r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"thank you"}]"#,
        )
        .unwrap();
        assert_eq!(check_history(&history), Ok(()));
        let request = user_request(history.clone());
        assert!(!request.stream);
        assert_eq!(request.max_tokens, Some(CHAT_MAX_TOKENS));
        let got: Vec<(&str, Option<&str>)> = request
            .messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_deref()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("user", Some("hi")),
                ("assistant", Some("hello")),
                ("user", Some("thank you")),
            ]
        );
        let streamed = stream_request(history);
        assert!(streamed.stream);
        assert_eq!(streamed.messages, request.messages);
    }

    #[test]
    fn bad_history_is_rejected_before_any_request() {
        assert!(check_history(&[]).is_err());
        assert!(check_history(&[turn("system", "be brief"), turn("user", "hi")]).is_err());
        assert!(check_history(&[turn("tool", "x"), turn("user", "hi")]).is_err());
        assert!(check_history(&[turn("user", " \n ")]).is_err());
        assert!(check_history(&[
            turn("user", "hi"),
            turn("assistant", ""),
            turn("user", "again")
        ])
        .is_err());
        assert!(check_history(&[turn("user", "hi"), turn("assistant", "hello")]).is_err());
        // Refused before the client is built: no server is needed for these.
        let err = chat_complete(vec![turn("user", "hi"), turn("assistant", "hello")]).unwrap_err();
        assert_eq!(err, "chat history must end with a user turn");
        assert!(chat_complete(Vec::new()).is_err());
    }

    #[test]
    fn stream_request_is_the_same_turn_with_stream_true() {
        let request = stream_request("Name the weights file.");
        assert!(request.stream);
        assert_eq!(request.max_tokens, Some(CHAT_MAX_TOKENS));
        assert_eq!(
            request.messages[0].content.as_deref(),
            Some("Name the weights file.")
        );
    }

    #[test]
    fn caps_follow_the_client_cfg() {
        let caps = client_caps();
        let src = include_str!("../../crates/client/src/lib.rs");
        let defines = |name: &str| {
            let needle = format!("fn {name}(");
            src.lines().any(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("//")
                    && !trimmed.starts_with('*')
                    && !trimmed.starts_with("/*")
                    && trimmed.contains(&needle)
            })
        };
        assert_eq!(caps.stream, defines("chat_completion_stream"));
        assert_eq!(caps.loaded_model, defines("loaded_model"));
        assert_eq!(
            caps.stream,
            cfg!(artificer_client_has_chat_completion_stream)
        );
        assert_eq!(caps.loaded_model, cfg!(artificer_client_has_loaded_model));
    }

    #[test]
    fn health_view_copies_real_fields_only() {
        let view = health_view(HealthStatus {
            status: "ok".into(),
            metrics: HealthMetrics {
                requests_started: 2,
                requests_completed: 1,
                requests_failed: 0,
                prompt_tokens_total: 4,
                completion_tokens_total: 3,
                in_flight: 1,
                peak_in_flight: 1,
            },
            backend: None,
        });
        assert!(view.backend.is_none());
        assert_eq!(view.status, "ok");
        assert_eq!(view.metrics.requests_started, 2);
        assert_eq!(view.metrics.requests_completed, 1);
        assert_eq!(view.metrics.requests_failed, 0);
        assert_eq!(view.metrics.prompt_tokens_total, 4);
        assert_eq!(view.metrics.completion_tokens_total, 3);
        assert_eq!(view.metrics.in_flight, 1);
        assert_eq!(view.metrics.peak_in_flight, 1);
    }

    #[test]
    fn assistant_text_and_tool_names_come_from_the_client_types() {
        let response = ChatCompletionResponse {
            id: "c".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "local".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".into(),
                    content: Some("the runtime responded".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: ChatUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
            timings: Some(ChatTimings {
                prefill_tokens: 31,
                prefill_ms: 100.0,
                decode_tokens: 2,
                decode_ms: 161.3,
            }),
            finish_reason: None,
            max_tokens: None,
            tokens: None,
            thinking_ms: None,
            thinking_tokens: None,
            completion_tokens: None,
            prefill_ms: None,
            limit_hit_in_thinking: None,
            error: None,
            connection_dropped: false,
        };
        assert_eq!(
            assistant_text(&response).as_deref(),
            Ok("the runtime responded")
        );
        let turn = turn_from(&response).unwrap();
        assert_eq!(turn.model, "local");
        assert_eq!(turn.content, "the runtime responded");
        let t = turn.timings.expect("timings copied");
        assert_eq!((t.prefill_tokens, t.decode_tokens), (31, 2));
        assert_eq!(t.prefill_ms, 100.0);
        // Today's server: reason from choices[0], limit = what the bridge requested, no split.
        assert_eq!(turn.finish_reason.as_deref(), Some("stop"));
        assert_eq!(turn.max_tokens, Some(u64::from(CHAT_MAX_TOKENS)));
        assert!(turn.tokens.is_none());
        assert_eq!(turn.completion_tokens, Some(2)); // from usage
        assert!(turn.thinking_ms.is_none());
        let mut done = response.clone();
        done.finish_reason = Some("length".into());
        done.max_tokens = Some(1024);
        done.tokens = Some(artificer_client::ChatTokenCounts {
            thinking: Some(418),
            answer: Some(606),
            total: Some(1024),
        });
        done.thinking_ms = Some(11600.0);
        done.prefill_ms = Some(420.0);
        done.limit_hit_in_thinking = Some(false);
        let json = serde_json::to_value(turn_from(&done).unwrap()).unwrap();
        assert_eq!(json["thinking_ms"], 11600.0);
        assert_eq!(json["prefill_ms"], 420.0);
        assert_eq!(json["limit_hit_in_thinking"], false);
        assert!(json["error"].is_null());
        assert_eq!(json["connection_dropped"], false);
        // finish_reason "error": message passes through; no text is fine.
        let mut err = response.clone();
        err.choices[0].message.content = None;
        err.finish_reason = Some("error".into());
        err.error = Some(artificer_client::ChatErrorInfo {
            message: "prompt too long".into(),
            kind: Some("server_error".into()),
        });
        let ej = serde_json::to_value(turn_from(&err).unwrap()).unwrap();
        assert_eq!(ej["error"]["message"], "prompt too long");
        assert_eq!(ej["content"], "");
        // A Stop during prefill: finish "user", no text.
        let mut stopped = response.clone();
        stopped.choices[0].message.content = None;
        stopped.finish_reason = Some("user".into());
        assert_eq!(turn_from(&stopped).unwrap().content, "");
        // A normal reply with no text is still an error.
        let mut empty = response.clone();
        empty.choices[0].message.content = None;
        assert!(turn_from(&empty).is_err());
        assert_eq!(json["finish_reason"], "length");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["tokens"]["thinking"], 418);
        assert_eq!(json["tokens"]["total"], 1024);

        let names = tool_names(McpToolsList {
            tools: vec![
                McpTool {
                    name: "chat".into(),
                    title: None,
                    description: None,
                },
                McpTool {
                    name: "complete".into(),
                    title: None,
                    description: None,
                },
            ],
        });
        assert_eq!(names, vec!["chat".to_string(), "complete".to_string()]);
    }

    #[test]
    fn max_tokens_env_is_bounded() {
        assert_eq!(CHAT_MAX_TOKENS, 1024);
        assert_eq!(MAX_TOKENS_CEILING, 4096);
        assert_eq!(parse_max_tokens(None), CHAT_MAX_TOKENS);
        assert_eq!(parse_max_tokens(Some("128")), 128);
        assert_eq!(parse_max_tokens(Some("2048")), 2048);
        assert_eq!(parse_max_tokens(Some(" 4096 ")), 4096);
        // Over the ceiling: clamped to it, not dropped back to the default.
        assert_eq!(parse_max_tokens(Some("5000")), MAX_TOKENS_CEILING);
        assert_eq!(parse_max_tokens(Some("99999999999")), MAX_TOKENS_CEILING);
        assert_eq!(parse_max_tokens(Some("0")), CHAT_MAX_TOKENS);
        assert_eq!(parse_max_tokens(Some("-5")), CHAT_MAX_TOKENS);
        assert_eq!(parse_max_tokens(Some("lots")), CHAT_MAX_TOKENS);
    }

    #[test]
    fn stop_with_no_stream_is_a_no_op() {
        let _g = BRIDGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        *ACTIVE.lock().unwrap() = None;
        let c = artificer_client::Client::with_base_url("http://127.0.0.1:9").unwrap();
        assert_eq!(
            stop_active(&c, Duration::from_millis(50)),
            StopView {
                requested: false,
                server_ack: false,
                dropped: false
            }
        );
    }

    /// Bridge path end to end against a scripted server: Stop → POST /v1/chat/cancel
    /// with the stream's id → the stream still returns the final chunk, and the window
    /// gets finish_reason "user" plus the server's counts.
    #[test]
    fn stop_stream_returns_server_final_chunk() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let _g = BRIDGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_body = Arc::new(Mutex::new(String::new()));
        let (c2, b2) = (Arc::clone(&cancelled), Arc::clone(&cancel_body));
        std::thread::spawn(move || {
            for sock in listener.incoming() {
                let Ok(mut sock) = sock else { break };
                let (cancelled, body) = (Arc::clone(&c2), Arc::clone(&b2));
                std::thread::spawn(move || {
                    sock.set_read_timeout(Some(Duration::from_millis(300)))
                        .unwrap();
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    while let Ok(n) = sock.read(&mut tmp) {
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                        let t = String::from_utf8_lossy(&buf);
                        if t.contains("\r\n\r\n") && (t.ends_with('}') || t.starts_with("GET")) {
                            break;
                        }
                    }
                    let req = String::from_utf8_lossy(&buf).to_string();
                    let chunk = |s: &mut std::net::TcpStream, p: &str| {
                        let d = format!("data: {p}\n\n");
                        s.write_all(format!("{:X}\r\n{d}\r\n", d.len()).as_bytes())
                            .and_then(|_| s.flush())
                    };
                    if req.starts_with("POST /v1/chat/cancel ") {
                        *body.lock().unwrap() =
                            req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                        cancelled.store(true, Ordering::SeqCst);
                        let b = r#"{"cancelled":true,"id":"chatcmpl-b"}"#;
                        let _ = sock.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}", b.len()).as_bytes());
                        return;
                    }
                    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\ntransfer-encoding: chunked\r\n\r\n");
                    let _ = chunk(
                        &mut sock,
                        r#"{"id":"chatcmpl-b","model":"nm","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
                    );
                    let mut n = 0;
                    while !cancelled.load(Ordering::SeqCst) && n < 400 {
                        if chunk(&mut sock, r#"{"id":"chatcmpl-b","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}"#).is_err() { return; }
                        n += 1;
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    let fin = format!(
                        r#"{{"id":"chatcmpl-b","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":3,"completion_tokens":{n},"total_tokens":{}}},"finish_reason":"user","max_tokens":128,"tokens":{{"thinking":{n},"answer":0,"total":{n}}},"thinking_ms":5.0,"prefill_ms":7.0,"limit_hit_in_thinking":false}}"#,
                        n + 3
                    );
                    let _ = chunk(&mut sock, &fin);
                    let _ = chunk(&mut sock, "[DONE]");
                    let _ = sock.write_all(b"0\r\n\r\n");
                });
            }
        });
        let c =
            artificer_client::Client::with_base_url(format!("http://127.0.0.1:{port}")).unwrap();
        let c2 = c.clone();
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            stop_active(&c2, Duration::from_secs(3))
        });
        let mut deltas = 0;
        let turn = stream_turn(&c, "hi", |_| deltas += 1).expect("turn");
        let stop = stopper.join().unwrap();
        assert_eq!(
            stop,
            StopView {
                requested: true,
                server_ack: true,
                dropped: false
            }
        );
        assert_eq!(turn.finish_reason.as_deref(), Some("user"));
        assert!(!turn.connection_dropped);
        assert!(deltas > 0);
        assert_eq!(
            turn.tokens.as_ref().and_then(|t| t.total),
            Some(deltas as u64)
        );
        assert_eq!(turn.completion_tokens, Some(deltas as u64));
        assert_eq!(*cancel_body.lock().unwrap(), r#"{"id":"chatcmpl-b"}"#);
        assert!(
            ACTIVE.lock().unwrap().is_none(),
            "handle cleared after the stream"
        );
    }

    /// Tests that touch the global ACTIVE handle run one at a time.
    static BRIDGE_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn load_local_weights_rejects_relative_without_calling_server() {
        let err = load_local_weights("tiny-vicuna-1b.q4_k_m.gguf").unwrap_err();
        assert!(err.contains("absolute"));
    }

    #[test]
    fn gguf_listing_keeps_only_gguf_files() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("artificer-weights-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("keep.gguf"), b"").unwrap();
        std::fs::write(dir.join("skip.txt"), b"").unwrap();
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested").join("nope.gguf"), b"").unwrap();
        let listed = gguf_files_in(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(listed.len(), 1);
        assert!(listed[0].ends_with("keep.gguf"));
    }

    #[test]
    fn local_weight_list_follows_picker_order() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("artificer-weight-list-{nanos}"));
        let env_dir = root.join("env");
        let home = root.join("home");
        for dir in [
            &env_dir,
            &home.join("models"),
            &home.join("Projects/Artificer"),
        ] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(env_dir.join("env.gguf"), b"").unwrap();
        std::fs::write(home.join("models").join("models.gguf"), b"").unwrap();
        std::fs::write(home.join("Projects/Artificer").join("checkout.gguf"), b"").unwrap();
        let home_str = home.to_str().unwrap();
        let names = |files: Vec<String>| -> Vec<String> {
            files
                .iter()
                .map(|f| {
                    Path::new(f)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect()
        };

        // ARTIFICER_WEIGHTS_DIR wins: only its files, not ~/models.
        let env_only = weight_files_in(Some(&env_dir), Some(home_str), Path::is_dir);
        // Else ~/models when it exists; $HOME/Projects/Artificer is never scanned.
        let models_only = weight_files_in(None, Some(home_str), Path::is_dir);
        // ~/models missing, or no HOME: empty list.
        let empty_home = root.join("empty-home");
        std::fs::create_dir_all(&empty_home).unwrap();
        let no_models = weight_files_in(None, empty_home.to_str(), Path::is_dir);
        let no_home = weight_files_in(None, None, Path::is_dir);
        let blank_home = weight_files_in(None, Some(""), Path::is_dir);
        let _ = std::fs::remove_dir_all(&root);

        assert_eq!(names(env_only), vec!["env.gguf"]);
        assert_eq!(names(models_only), vec!["models.gguf"]);
        assert!(no_models.is_empty());
        assert!(no_home.is_empty());
        assert!(blank_home.is_empty());
    }

    #[test]
    fn picker_dir_prefers_env_then_models_then_dialog_default() {
        let yes = |_: &Path| true;
        let no = |_: &Path| false;
        // ARTIFICER_WEIGHTS_DIR wins and needs no HOME.
        assert_eq!(
            weights_dir(Some(Path::new("/w")), Some("/h"), yes),
            Some(Path::new("/w").to_path_buf())
        );
        assert_eq!(
            weights_dir(Some(Path::new("/w")), None, no),
            Some(Path::new("/w").to_path_buf())
        );
        // Else ~/models, only when it exists; never $HOME/Projects/Artificer.
        let seen = std::cell::RefCell::new(Vec::new());
        let dir = weights_dir(None, Some("/h"), |p: &Path| {
            seen.borrow_mut().push(p.to_path_buf());
            true
        });
        assert_eq!(dir, Some(Path::new("/h/models").to_path_buf()));
        assert_eq!(
            seen.into_inner(),
            vec![Path::new("/h/models").to_path_buf()]
        );
        // Else no start directory (dialog default), never HOME itself.
        assert_eq!(weights_dir(None, Some("/h"), no), None);
        assert_eq!(weights_dir(None, Some(""), yes), None);
        assert_eq!(weights_dir(None, None, yes), None);
    }

    #[test]
    fn home_falls_back_to_userprofile_on_windows_only() {
        // HOME wins everywhere when set (Git Bash, MSYS on Windows too).
        assert_eq!(
            home_from(Some("/h"), Some(r"C:\Users\you"), true).as_deref(),
            Some("/h")
        );
        assert_eq!(home_from(Some("/h"), None, false).as_deref(), Some("/h"));
        // Windows without HOME: USERPROFILE, so ~/models is found there.
        assert_eq!(
            home_from(None, Some(r"C:\Users\you"), true).as_deref(),
            Some(r"C:\Users\you")
        );
        assert_eq!(
            home_from(Some(""), Some(r"C:\Users\you"), true).as_deref(),
            Some(r"C:\Users\you")
        );
        // Not on Windows: USERPROFILE is ignored.
        assert_eq!(home_from(None, Some("/u"), false), None);
        // Nothing set, or only empty values: no home.
        assert_eq!(home_from(None, None, true), None);
        assert_eq!(home_from(Some(""), Some(""), true), None);
    }

    #[test]
    fn checked_weights_path_accepts_only_existing_absolute_gguf() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("artificer-checked-{nanos}"));
        std::fs::create_dir_all(dir.join("folder.gguf")).unwrap();
        let good = dir.join("model.GGUF");
        std::fs::write(&good, b"").unwrap();
        let txt = dir.join("notes.txt");
        std::fs::write(&txt, b"").unwrap();
        let good_s = good.to_str().unwrap().to_string();
        let ok = checked_weights_path(&good_s);
        let rel = checked_weights_path("model.gguf");
        let missing = checked_weights_path(dir.join("missing.gguf").to_str().unwrap());
        let wrong_ext = checked_weights_path(txt.to_str().unwrap());
        let folder = checked_weights_path(dir.join("folder.gguf").to_str().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(ok, Ok(good_s));
        assert!(rel.unwrap_err().contains("absolute"));
        assert!(missing.unwrap_err().starts_with("no file at this path"));
        assert!(wrong_ext.unwrap_err().contains(".gguf"));
        assert!(folder.unwrap_err().contains(".gguf"));
    }

    #[test]
    fn pick_outcome_covers_portal_and_zenity_combos() {
        // Nothing picked: Cancel only when some picker was there.
        assert_eq!(
            pick_outcome::<&str>(true, true, None),
            PickOutcome::Cancelled
        );
        assert_eq!(
            pick_outcome::<&str>(true, false, None),
            PickOutcome::Cancelled
        );
        assert_eq!(
            pick_outcome::<&str>(false, true, None),
            PickOutcome::Cancelled
        );
        assert_eq!(
            pick_outcome::<&str>(false, false, None),
            PickOutcome::Unavailable
        );
        // A picked file always wins, whatever the probes said.
        for (portal, zenity) in [(true, true), (true, false), (false, true), (false, false)] {
            assert_eq!(
                pick_outcome(portal, zenity, Some("/m/a.gguf")),
                PickOutcome::Picked("/m/a.gguf")
            );
        }
        assert_eq!(PICKER_UNAVAILABLE, "picker-unavailable");
    }

    #[test]
    fn file_chooser_found_in_gdbus_and_busctl_output() {
        let gdbus = "node /org/freedesktop/portal/desktop {\n  interface org.freedesktop.portal.FileChooser {\n    methods:\n";
        let busctl = "NAME                                       TYPE      SIGNATURE RESULT/VALUE FLAGS\norg.freedesktop.portal.FileChooser         interface -                 -            -\n.OpenFile                                  method    ssa{sv}   o            -\n";
        let without = "node /org/freedesktop/portal/desktop {\n  interface org.freedesktop.portal.Settings {\n  interface org.freedesktop.portal.FileChooserX {\n";
        assert!(lists_file_chooser(gdbus));
        assert!(lists_file_chooser(busctl));
        assert!(!lists_file_chooser(without));
        assert!(!lists_file_chooser(""));
    }

    #[test]
    fn zenity_found_only_as_a_file_on_path() {
        let path = std::env::join_paths(["/opt/x/bin", "/usr/bin"]).unwrap();
        assert!(on_path("zenity", Some(&path), |p: &Path| {
            p == Path::new("/usr/bin/zenity")
        }));
        assert!(!on_path("zenity", Some(&path), |_: &Path| false));
        assert!(!on_path("zenity", None, |_: &Path| true));
        assert!(!on_path("zenity", Some("".as_ref()), |_: &Path| false));
    }

    /// Tests that spawn processes run one at a time (a fork while a fake program is still open
    /// for writing would make exec fail with ETXTBSY).
    static SPAWN_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn pick_flow_unavailable_round_trip_is_exactly_picker_unavailable() {
        // Fake bus check says no portal; zenity is not on the (empty) PATH.
        let no_bus = || false;
        let no_zenity = || on_path("zenity", Some("".as_ref()), |_: &Path| false);
        let opened = std::cell::Cell::new(false);
        let result = pick_flow(
            || (no_bus(), no_zenity()),
            || {
                opened.set(true);
                Some("/m/a.gguf")
            },
        );
        assert_eq!(result, Err(PICKER_UNAVAILABLE.to_string()));
        assert_eq!(result.unwrap_err(), "picker-unavailable");
        assert!(!opened.get(), "no dialog is opened when no picker exists");

        // The post-None probe is the authority: present before, gone after -> unavailable.
        let calls = std::cell::Cell::new(0);
        let flaky = || {
            calls.set(calls.get() + 1);
            (calls.get() == 1, false)
        };
        assert_eq!(
            pick_flow(flaky, || None::<&str>),
            Err(PICKER_UNAVAILABLE.to_string())
        );
        // A real Cancel, and a pick.
        assert_eq!(pick_flow(|| (true, false), || None::<&str>), Ok(None));
        assert_eq!(pick_flow(|| (false, true), || None::<&str>), Ok(None));
        assert_eq!(
            pick_flow(|| (true, true), || Some("/m/a.gguf")),
            Ok(Some("/m/a.gguf"))
        );
    }

    #[test]
    fn portal_probe_command_lines_are_the_documented_ones() {
        assert_eq!(
            GDBUS_ARGS.join(" "),
            "introspect --session --dest org.freedesktop.portal.Desktop \
             --object-path /org/freedesktop/portal/desktop"
        );
        assert_eq!(
            BUSCTL_ARGS.join(" "),
            "--user introspect org.freedesktop.portal.Desktop /org/freedesktop/portal/desktop"
        );
        assert_eq!(PROBE_TIMEOUT, Duration::from_millis(1500));
    }

    /// Runs the real spawn path of the portal check against fake `gdbus` / `busctl` scripts
    /// that answer only when called with exactly the documented arguments.
    #[cfg(unix)]
    #[test]
    fn portal_probe_spawn_path_args_fallback_and_timeout() {
        use std::os::unix::fs::PermissionsExt;
        let _guard = SPAWN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("artificer-portal-probe-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let script = |name: &str, body: &str| {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path.to_str().unwrap().to_string()
        };
        let gdbus_ok = script(
            "gdbus-ok",
            &format!(
                "[ \"$*\" = \"{}\" ] || exit 9\necho \"node /org/freedesktop/portal/desktop {{\"\necho \"  interface org.freedesktop.portal.FileChooser {{\"",
                GDBUS_ARGS.join(" ")
            ),
        );
        let gdbus_settings_only = script(
            "gdbus-settings",
            "echo \"  interface org.freedesktop.portal.Settings {\"",
        );
        let gdbus_fails = script("gdbus-fails", "exit 1");
        let gdbus_hangs = script("gdbus-hangs", "exec sleep 30");
        let busctl_ok = script(
            "busctl-ok",
            &format!(
                "[ \"$*\" = \"{}\" ] || exit 9\necho \"org.freedesktop.portal.FileChooser         interface -  -  -\"",
                BUSCTL_ARGS.join(" ")
            ),
        );
        let missing = dir.join("not-installed").to_str().unwrap().to_string();
        let t = PROBE_TIMEOUT;

        let ok = portal_file_chooser_present_with(&gdbus_ok, &missing, t);
        let settings_only = portal_file_chooser_present_with(&gdbus_settings_only, &busctl_ok, t);
        // gdbus installed but failing: busctl is not consulted.
        let fails = portal_file_chooser_present_with(&gdbus_fails, &busctl_ok, t);
        // gdbus missing: busctl with its own arguments.
        let fallback = portal_file_chooser_present_with(&missing, &busctl_ok, t);
        // Neither installed: false, well inside the timeout.
        let started = std::time::Instant::now();
        let neither = portal_file_chooser_present_with(&missing, &missing, t);
        let neither_took = started.elapsed();
        // A hung gdbus: killed at the 1.5 s deadline, false.
        let started = std::time::Instant::now();
        let hung = portal_file_chooser_present_with(&gdbus_hangs, &busctl_ok, t);
        let hung_took = started.elapsed();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(ok, "FileChooser listed with the exact gdbus arguments");
        assert!(!settings_only);
        assert!(!fails);
        assert!(fallback, "busctl fallback with the exact busctl arguments");
        assert!(!neither);
        assert!(
            neither_took < PROBE_TIMEOUT,
            "missing binaries took {neither_took:?}"
        );
        assert!(!hung);
        assert!(
            hung_took >= PROBE_TIMEOUT && hung_took < PROBE_TIMEOUT + Duration::from_secs(1),
            "hung probe took {hung_took:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn probe_runner_reports_missing_output_and_timeout() {
        let _guard = SPAWN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            run_with_deadline("artificer-no-such-program", &[], Duration::from_secs(1)),
            Probe::Missing
        );
        assert_eq!(
            run_with_deadline("sh", &["-c", "echo hi"], Duration::from_secs(5)),
            Probe::Ran(Some("hi\n".to_string()))
        );
        assert_eq!(
            run_with_deadline("sh", &["-c", "exit 3"], Duration::from_secs(5)),
            Probe::Ran(None)
        );
        let started = std::time::Instant::now();
        assert_eq!(
            run_with_deadline("sleep", &["5"], Duration::from_millis(200)),
            Probe::Ran(None)
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "child was not killed"
        );
    }

    /// Live stream proof. No-op unless `ARTIFICER_LIVE=1`. Does not start serve.
    /// Same `chat_completion_stream` call the window command uses.
    #[test]
    fn live_stream_deltas_and_loaded_model() {
        if std::env::var("ARTIFICER_LIVE").ok().as_deref() != Some("1") {
            return;
        }
        let mut c = artificer_client::Client::new();
        c.set_timeout(std::time::Duration::from_secs(180));
        let model = c.loaded_model().expect("loaded_model");
        println!("LOADED_MODEL={model:?}");
        let prompt = "Reply with the single word pong.";
        let request = stream_request(prompt);
        println!("PROMPT={prompt}");
        let mut deltas = Vec::new();
        let response = c
            .chat_completion_stream(&request, |delta| {
                println!("DELTA={delta}");
                deltas.push(delta.to_string());
            })
            .expect("chat_completion_stream");
        let assembled = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();
        println!("ASSEMBLED={assembled}");
        println!("RESPONSE_MODEL={}", response.model);
        println!("DELTA_COUNT={}", deltas.len());
        assert!(model.is_some(), "loaded_model was None");
        assert!(!deltas.is_empty(), "stream produced no deltas");
        assert!(!assembled.is_empty(), "assembled text empty");
    }

    /// Live localhost proof. No-op unless `ARTIFICER_LIVE=1`.
    /// Does not start serve. Calls the same functions the window calls.
    #[test]
    fn live_localhost_health_and_one_chat() {
        if std::env::var("ARTIFICER_LIVE").ok().as_deref() != Some("1") {
            return;
        }
        let health = check_health().expect("check_health");
        println!("HEALTH_STATUS={}", health.status);
        let turn = chat_complete("Say hi in three words").expect("chat_complete");
        println!("CHAT_REPLY={}", turn.content);
        let tools = mcp_tools_list().expect("mcp_tools_list");
        println!("MCP_TOOLS={}", tools.join(","));
        assert!(!health.status.is_empty(), "health status empty");
        assert!(!turn.content.is_empty(), "assistant reply empty");
    }
}
