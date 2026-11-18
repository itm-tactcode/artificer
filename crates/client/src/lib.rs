//! Plain HTTP/1.1 client for a local Artificer `serve` process.
//!
//! Default base URL: `http://127.0.0.1:8080`.
//!
//! `GET /` is **404 by design** (the server does not route `/`). This crate never
//! requests `/`. Call [`Client::health`] (`GET /health`), [`Client::chat_completion`]
//! (`POST /v1/chat/completions`, non-streaming), [`Client::chat_completion_stream`]
//! (`POST /v1/chat/completions` with `"stream": true`, SSE), [`Client::loaded_model`]
//! (`GET /v1/models`), [`Client::load_model`]
//! (`POST /v1/models/load`), and [`Client::mcp_tools_list`]
//! (`POST /mcp` JSON-RPC method `tools/list`).
//! [`Client::chat_completion_stream_cancellable`] + [`StreamCancel`] stop a stream via
//! `POST /v1/chat/cancel` ([`Client::cancel_chat`]).

use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Default Artificer serve base URL. `GET /` on that server is 404 by design.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080";

/// Failure talking to Artificer or decoding its JSON.
#[derive(Debug)]
pub enum ClientError {
    /// Base URL or request could not be formed.
    Invalid(String),
    /// TCP or timeout failure.
    Io(std::io::Error),
    /// HTTP status was not 2xx. Body is truncated for logs.
    Status { status: u16, body: String },
    /// Body was not the expected JSON.
    Json(String),
    /// JSON-RPC `error` object on `POST /mcp`.
    Rpc(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Invalid(m) => write!(f, "{m}"),
            ClientError::Io(e) => write!(f, "{e}"),
            ClientError::Status { status, body } => write!(f, "HTTP {status}: {body}"),
            ClientError::Json(m) => write!(f, "{m}"),
            ClientError::Rpc(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<std::io::Error> for ClientError {
    fn from(value: std::io::Error) -> Self {
        ClientError::Io(value)
    }
}

/// `GET /health` body. `GET /` is not this route and is 404 by design.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub metrics: HealthMetrics,
    /// Active compute backend. Absent on servers that predate the field.
    #[serde(default)]
    pub backend: Option<BackendInfo>,
}

/// `backend` object on `GET /health`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BackendInfo {
    /// Backend the server was asked for (`cpu` / `cuda`). Absent → unknown.
    #[serde(default)]
    pub requested: Option<String>,
    /// Backend actually running (`cpu` / `cuda`).
    pub active: String,
    /// CUDA device name when `active == "cuda"`.
    #[serde(default)]
    pub device: Option<String>,
    /// Set when CUDA was requested and the server fell back to CPU.
    #[serde(default)]
    pub fallback_reason: Option<String>,
}

/// Additive `timings` object on the final stream chunk / non-stream response.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatTimings {
    #[serde(default)]
    pub prefill_tokens: u64,
    #[serde(default)]
    pub prefill_ms: f64,
    #[serde(default)]
    pub decode_tokens: u64,
    #[serde(default)]
    pub decode_ms: f64,
}

/// Done-event token split: `tokens: {thinking, answer, total}`.
/// Every field is optional; a malformed value reads as `None`, never a stream error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct ChatTokenCounts {
    #[serde(default, deserialize_with = "lenient")]
    pub thinking: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    pub answer: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    pub total: Option<u64>,
}

/// Reads an optional extension field; anything that does not parse as `T` is `None`.
fn lenient<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let v = serde_json::Value::deserialize(d)?;
    Ok(serde_json::from_value(v).ok())
}

/// Counters from `ServerMetrics::snapshot` on `GET /health`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct HealthMetrics {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub in_flight: u64,
    pub peak_in_flight: u64,
}

/// One OpenAI-style chat message for `POST /v1/chat/completions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Chat completion request for `POST /v1/chat/completions`.
///
/// [`Client::chat_completion`] sends `"stream": false`.
/// [`Client::chat_completion_stream`] sends `"stream": true` on the same route
/// and reads the server's SSE response. There is no separate stream URL.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatCompletionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
}

impl ChatCompletionRequest {
    /// Single user turn, non-streaming. `max_tokens` is sent so the server does
    /// not fall back to its default (`artificer_core::DEFAULT_MAX_TOKENS`, 1024).
    pub fn user(content: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            model: None,
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(content.into()),
            }],
            max_tokens: Some(max_tokens),
            temperature: None,
            stream: false,
        }
    }
}

/// `POST /v1/chat/completions` JSON body (`object` is `chat.completion`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
    /// Prefill/decode timing from the server, when it sends one.
    #[serde(default)]
    pub timings: Option<ChatTimings>,
    /// Server outcome fields (top level of the final chunk), read defensively.
    /// `'stop' | 'length' | 'user' | 'error'`; `None` from servers that only set
    /// `choices[0].finish_reason`. (`"cancelled"` is tolerated; mapped in the window.)
    #[serde(default, deserialize_with = "lenient")]
    pub finish_reason: Option<String>,
    /// The limit N the server applied (counts thinking + answer tokens).
    #[serde(default, deserialize_with = "lenient")]
    pub max_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    pub tokens: Option<ChatTokenCounts>,
    /// Server-measured thinking time (prefill excluded).
    #[serde(default, deserialize_with = "lenient")]
    pub thinking_ms: Option<f64>,
    /// Tolerated aliases for `tokens.thinking` / `tokens.total` (an earlier server field layout).
    #[serde(default, deserialize_with = "lenient")]
    pub thinking_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    pub completion_tokens: Option<u64>,
    /// Prompt prefill wall time.
    #[serde(default, deserialize_with = "lenient")]
    pub prefill_ms: Option<f64>,
    /// `finish_reason == "length"` while still thinking.
    #[serde(default, deserialize_with = "lenient")]
    pub limit_hit_in_thinking: Option<bool>,
    /// Only with `finish_reason == "error"`: `{"message", "type"}`.
    #[serde(default, deserialize_with = "lenient")]
    pub error: Option<ChatErrorInfo>,
    /// Client-side: the stream was cut by [`StreamCancel`] after its grace period,
    /// so no final chunk (and no server counts) arrived. Never on the wire.
    #[serde(skip)]
    pub connection_dropped: bool,
}

/// `error` object on the final chunk when `finish_reason == "error"`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChatErrorInfo {
    #[serde(default)]
    pub message: String,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
}

/// `POST /v1/chat/cancel` response.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CancelAck {
    #[serde(default)]
    pub cancelled: bool,
    #[serde(default)]
    pub id: Option<String>,
}

/// How [`StreamCancel::cancel`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelOutcome {
    /// The server answered `{"cancelled": true}` at least once.
    pub server_ack: bool,
    /// The final chunk did not arrive within the grace period; the socket was shut down
    /// (the server also stops generating when the stream closes).
    pub dropped: bool,
}

/// Stop handle for one [`Client::chat_completion_stream_cancellable`] call. Clone it to
/// another thread and call [`StreamCancel::cancel`].
#[derive(Debug, Clone, Default)]
pub struct StreamCancel {
    inner: Arc<CancelState>,
}

#[derive(Debug, Default)]
struct CancelState {
    requested: AtomicBool,
    finished: AtomicBool,
    dropped: AtomicBool,
    id: Mutex<Option<String>>,
    socket: Mutex<Option<TcpStream>>,
}

impl StreamCancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_requested(&self) -> bool {
        self.inner.requested.load(Ordering::SeqCst)
    }

    /// The stream call has returned (final chunk, EOF, error or drop).
    pub fn is_finished(&self) -> bool {
        self.inner.finished.load(Ordering::SeqCst)
    }

    /// Completion id from the stream's first chunk, once seen.
    pub fn id(&self) -> Option<String> {
        self.inner
            .id
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set_id(&self, id: &str) {
        let mut g = self.inner.id.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            *g = Some(id.to_owned());
        }
    }

    fn attach(&self, stream: &TcpStream) {
        *self.inner.socket.lock().unwrap_or_else(|e| e.into_inner()) = stream.try_clone().ok();
    }

    fn finish(&self) {
        self.inner.finished.store(true, Ordering::SeqCst);
        self.inner
            .socket
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }

    fn was_dropped(&self) -> bool {
        self.inner.dropped.load(Ordering::SeqCst)
    }

    /// Shut the socket down now; the reading call returns what it has.
    pub fn drop_connection(&self) {
        self.inner.dropped.store(true, Ordering::SeqCst);
        if let Some(s) = self
            .inner
            .socket
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = s.shutdown(Shutdown::Both);
        }
    }

    /// Ask the server to stop (`POST /v1/chat/cancel {"id"}`), then wait for the stream
    /// to deliver its final chunk (`finish_reason: "user"`, real counts). The request is
    /// re-sent every `poll` until the server acknowledges it — a Stop that lands before
    /// the server registered the completion is not lost. If the stream has not finished
    /// within `grace`, the connection is dropped (the server stops within one token when
    /// its SSE channel closes). A Stop during prefill also ends within one
    /// prefill chunk (≤ 1024 prompt tokens on CUDA, one token on CPU); a slower server
    /// (one that only stops after prefill) still ends at the drop. Blocks for at most
    /// about `grace`.
    pub fn cancel(&self, client: &Client, grace: Duration) -> CancelOutcome {
        self.inner.requested.store(true, Ordering::SeqCst);
        let poll = Duration::from_millis(100);
        let deadline = Instant::now() + grace;
        let mut short = client.clone();
        short.set_timeout(
            grace
                .min(Duration::from_secs(2))
                .max(Duration::from_millis(200)),
        );
        let mut acked = false;
        loop {
            if self.is_finished() {
                return CancelOutcome {
                    server_ack: acked,
                    dropped: false,
                };
            }
            if !acked {
                acked = short
                    .cancel_chat(self.id().as_deref())
                    .map(|a| a.cancelled)
                    .unwrap_or(false);
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep(poll.min(deadline - now));
        }
        if self.is_finished() {
            return CancelOutcome {
                server_ack: acked,
                dropped: false,
            };
        }
        self.drop_connection();
        CancelOutcome {
            server_ack: acked,
            dropped: true,
        }
    }
}

struct FinishOnDrop<'a>(&'a StreamCancel);

impl Drop for FinishOnDrop<'_> {
    fn drop(&mut self) {
        self.0.finish();
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// One tool from MCP `tools/list` (`POST /mcp`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Successful `tools/list` result. The server advertises `chat` and `complete`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct McpToolsList {
    pub tools: Vec<McpTool>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcEnvelope {
    #[serde(default)]
    result: Option<McpToolsList>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct Endpoint {
    host: String,
    port: u16,
    host_header: String,
}

/// Blocking client. No async runtime. HTTP/1.1 over `std::net::TcpStream`.
#[derive(Debug, Clone)]
pub struct Client {
    endpoint: Endpoint,
    timeout: Duration,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Client for [`DEFAULT_BASE_URL`] (`http://127.0.0.1:8080`).
    ///
    /// `GET /` on that server is 404 by design; this constructor does not
    /// request it.
    pub fn new() -> Self {
        Self::with_base_url(DEFAULT_BASE_URL).expect("DEFAULT_BASE_URL is valid")
    }

    /// `base_url` must be `http://host[:port]` with no path (the default port is 80).
    /// HTTPS is not supported. `GET /` remains 404 by design on the Artificer server.
    pub fn with_base_url(base_url: impl AsRef<str>) -> Result<Self, ClientError> {
        Ok(Self {
            endpoint: parse_base_url(base_url.as_ref())?,
            timeout: Duration::from_secs(120),
        })
    }

    /// Connect, read, and write timeout. Chat completions can be slow on CPU.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.endpoint.host, self.endpoint.port)
    }

    /// `GET /health`.
    ///
    /// Returns `{ "status": "ok", "metrics": ... }`. This is not `GET /`.
    /// `GET /` is 404 by design because the server does not register that route.
    pub fn health(&self) -> Result<HealthStatus, ClientError> {
        let body = self.roundtrip("GET", "/health", None)?;
        serde_json::from_slice(&body).map_err(|e| ClientError::Json(e.to_string()))
    }

    /// `POST /v1/chat/completions` with `"stream": false`.
    ///
    /// The same route serves SSE when `stream` is true. This method refuses a
    /// request that sets `stream` to true so the caller does not treat an SSE
    /// body as a completion object.
    pub fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ClientError> {
        if request.stream {
            return Err(ClientError::Invalid(
                "chat_completion sends stream=false; the server SSE path is POST /v1/chat/completions with stream=true".into(),
            ));
        }
        if request.messages.is_empty() {
            return Err(ClientError::Invalid("messages must not be empty".into()));
        }
        let payload = serde_json::to_vec(request).map_err(|e| ClientError::Json(e.to_string()))?;
        let body = self.roundtrip("POST", "/v1/chat/completions", Some(&payload))?;
        serde_json::from_slice(&body)
            .map_err(|e| ClientError::Json(format!("{e}; body={}", snippet(&body))))
    }

    /// `POST /mcp` with JSON-RPC `tools/list`.
    ///
    /// Body is `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`. The HTTP route
    /// is `/mcp`, not a separate `/tools/list` path. Notifications are not used;
    /// this call always sends an `id` so the server returns JSON (not 204).
    pub fn mcp_tools_list(&self) -> Result<McpToolsList, ClientError> {
        let payload = br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let body = self.roundtrip("POST", "/mcp", Some(payload))?;
        let env: JsonRpcEnvelope =
            serde_json::from_slice(&body).map_err(|e| ClientError::Json(e.to_string()))?;
        if let Some(err) = env.error {
            return Err(ClientError::Rpc(err.to_string()));
        }
        env.result
            .ok_or_else(|| ClientError::Json("tools/list missing result".into()))
    }

    /// `POST /v1/chat/completions` with `"stream": true`.
    ///
    /// The server responds `content-type: text/event-stream` and
    /// `transfer-encoding: chunked`. Each SSE `data:` payload is a
    /// `chat.completion.chunk`. This method calls `on_delta` once per
    /// `choices[0].delta.content` string as that event is parsed, including
    /// when the event arrives in its own HTTP chunk, and does not split a
    /// non-SSE JSON body into fake deltas.
    ///
    /// The returned [`ChatCompletionResponse`] is assembled from those chunks
    /// (`object` is `chat.completion`, not `chat.completion.chunk`). `id`,
    /// `model`, `created`, concatenated content, `finish_reason`, and `usage`
    /// come from the stream. The caller's `request.stream` flag is not
    /// modified; the wire body always sets `"stream": true`.
    ///
    /// A 2xx body that is not `text/event-stream` is an error and does not
    /// invoke `on_delta`.
    pub fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
        mut on_delta: impl FnMut(&str),
    ) -> Result<ChatCompletionResponse, ClientError> {
        if request.messages.is_empty() {
            return Err(ClientError::Invalid("messages must not be empty".into()));
        }
        let mut wire = request.clone();
        wire.stream = true;
        let payload = serde_json::to_vec(&wire).map_err(|e| ClientError::Json(e.to_string()))?;
        exchange_sse(
            &self.endpoint,
            self.timeout,
            "/v1/chat/completions",
            &payload,
            &mut on_delta,
        )
    }

    /// `GET /v1/models`.
    ///
    /// Returns the first `data[].id`. That id is the model id the serve
    /// process was started with (`ServerConfig.model_id` / CLI `--model-id`,
    /// otherwise the GGUF file stem). `Ok(None)` only when `data` is empty.
    /// This does not download weights. `GET /health` does not include a model
    /// id; this method does not read `/health`.
    pub fn loaded_model(&self) -> Result<Option<String>, ClientError> {
        let body = self.roundtrip("GET", "/v1/models", None)?;
        model_id_from_list(&body)
    }

    /// `POST /v1/chat/cancel` with `{"id": …}` (or `{}` to cancel whatever is running).
    /// The cancelled completion still ends with its final chunk (`finish_reason: "user"`).
    pub fn cancel_chat(&self, id: Option<&str>) -> Result<CancelAck, ClientError> {
        let body = match id {
            Some(id) => serde_json::json!({ "id": id }),
            None => serde_json::json!({}),
        };
        let payload = serde_json::to_vec(&body).map_err(|e| ClientError::Json(e.to_string()))?;
        let resp = self.roundtrip("POST", "/v1/chat/cancel", Some(&payload))?;
        serde_json::from_slice(&resp).map_err(|e| ClientError::Json(e.to_string()))
    }

    /// [`Client::chat_completion_stream`] that another thread can stop with
    /// [`StreamCancel::cancel`]. After a Stop the call keeps reading until the server's
    /// final chunk (`finish_reason: "user"` + counts); if the stream is cut after the
    /// grace period it returns what arrived, with `finish_reason: "user"` and
    /// `connection_dropped: true`.
    pub fn chat_completion_stream_cancellable(
        &self,
        request: &ChatCompletionRequest,
        cancel: &StreamCancel,
        mut on_delta: impl FnMut(&str),
    ) -> Result<ChatCompletionResponse, ClientError> {
        let _done = FinishOnDrop(cancel);
        if request.messages.is_empty() {
            return Err(ClientError::Invalid("messages must not be empty".into()));
        }
        let mut wire = request.clone();
        wire.stream = true;
        let payload = serde_json::to_vec(&wire).map_err(|e| ClientError::Json(e.to_string()))?;
        exchange_sse_with(
            &self.endpoint,
            self.timeout,
            "/v1/chat/completions",
            &payload,
            &mut on_delta,
            Some(cancel),
        )
    }

    /// First `GET /v1/models` entry with the optional `template_opens_think` flag
    /// (`None` when the server does not report it).
    pub fn loaded_model_info(&self) -> Result<Option<LoadedModelInfo>, ClientError> {
        let body = self.roundtrip("GET", "/v1/models", None)?;
        model_info_from_list(&body)
    }

    /// `POST /v1/models/load`.
    ///
    /// Body is `{"path":"<absolute path>"}` with `Content-Type: application/json`.
    /// Any 2xx is success. This method does not read a model id from that body.
    /// The id the server is serving afterward is [`Client::loaded_model`]
    /// (`GET /v1/models`, first `data[].id`).
    pub fn load_model(&self, path: &std::path::Path) -> Result<(), ClientError> {
        let text = path
            .to_str()
            .ok_or_else(|| ClientError::Invalid("model path is not UTF-8".into()))?;
        if text.is_empty() {
            return Err(ClientError::Invalid("model path must not be empty".into()));
        }
        if !path.is_absolute() {
            return Err(ClientError::Invalid("model path must be absolute".into()));
        }
        #[derive(Serialize)]
        struct Body<'a> {
            path: &'a str,
        }
        let payload = serde_json::to_vec(&Body { path: text })
            .map_err(|e| ClientError::Json(e.to_string()))?;
        let _body = self.roundtrip("POST", "/v1/models/load", Some(&payload))?;
        Ok(())
    }

    fn roundtrip(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<Vec<u8>, ClientError> {
        let (status, resp_body) = exchange(&self.endpoint, self.timeout, method, path, body)?;
        if !(200..300).contains(&status) {
            return Err(ClientError::Status {
                status,
                body: snippet(&resp_body),
            });
        }
        Ok(resp_body)
    }
}

fn snippet(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    let mut s: String = text.chars().take(240).collect();
    if text.chars().count() > 240 {
        s.push_str("...");
    }
    s
}

fn parse_base_url(url: &str) -> Result<Endpoint, ClientError> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        ClientError::Invalid(format!("only http:// base URLs are supported, got {url}"))
    })?;
    if rest.is_empty() || rest.contains("://") {
        return Err(ClientError::Invalid(format!("invalid base URL {url}")));
    }
    let (authority, path) = match rest.split_once('/') {
        Some((a, p)) => (a, p),
        None => (rest, ""),
    };
    if !path.is_empty() {
        return Err(ClientError::Invalid(
            "base URL must not include a path; routes are rooted at the origin".into(),
        ));
    }
    let authority = authority.trim_end_matches('/');
    if authority.is_empty() || authority.contains('@') {
        return Err(ClientError::Invalid(format!("invalid authority in {url}")));
    }
    let (host, port) = if let Some(h) = authority.strip_prefix('[') {
        let (host, rest) = h
            .split_once(']')
            .ok_or_else(|| ClientError::Invalid("unclosed IPv6 bracket".into()))?;
        let port = if let Some(p) = rest.strip_prefix(':') {
            parse_port(p)?
        } else if rest.is_empty() {
            80
        } else {
            return Err(ClientError::Invalid(format!("invalid authority in {url}")));
        };
        (host.to_string(), port)
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        if host.is_empty() {
            return Err(ClientError::Invalid(format!("invalid authority in {url}")));
        }
        (host.to_string(), parse_port(port)?)
    } else {
        (authority.to_string(), 80)
    };
    if host.is_empty() {
        return Err(ClientError::Invalid(format!("invalid host in {url}")));
    }
    Ok(Endpoint {
        host_header: format!("{host}:{port}"),
        host,
        port,
    })
}

fn parse_port(port: &str) -> Result<u16, ClientError> {
    port.parse::<u16>()
        .map_err(|_| ClientError::Invalid(format!("invalid port {port}")))
}

pub(crate) fn build_http_request(
    host_header: &str,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Vec<u8> {
    let mut out = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host_header}\r\nAccept: application/json\r\nConnection: close\r\n"
    );
    if let Some(body) = body {
        out.push_str("Content-Type: application/json\r\n");
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    out.push_str("\r\n");
    let mut bytes = out.into_bytes();
    if let Some(body) = body {
        bytes.extend_from_slice(body);
    }
    bytes
}

fn exchange(
    endpoint: &Endpoint,
    timeout: Duration,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> Result<(u16, Vec<u8>), ClientError> {
    let addr = format!("{}:{}", endpoint.host, endpoint.port);
    let mut addrs = addr
        .to_socket_addrs()
        .map_err(|e| ClientError::Invalid(format!("resolve {addr}: {e}")))?;
    let addr = addrs
        .next()
        .ok_or_else(|| ClientError::Invalid(format!("resolve {addr}: no addresses")))?;
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let req = build_http_request(&endpoint.host_header, method, path, body);
    stream.write_all(&req)?;
    stream.flush()?;
    let buf = read_to_end(&mut stream)?;
    parse_http_response(&buf)
}

fn read_to_end(stream: &mut TcpStream) -> Result<Vec<u8>, ClientError> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                if buf.is_empty() {
                    return Err(ClientError::Io(e));
                }
                break;
            }
            Err(e) => return Err(ClientError::Io(e)),
        }
        if buf.len() > 8 * 1024 * 1024 {
            return Err(ClientError::Invalid("response larger than 8 MiB".into()));
        }
    }
    Ok(buf)
}

pub(crate) fn parse_http_response(buf: &[u8]) -> Result<(u16, Vec<u8>), ClientError> {
    let header_end = find_subsequence(buf, b"\r\n\r\n")
        .ok_or_else(|| ClientError::Invalid("HTTP response missing header terminator".into()))?;
    let header = std::str::from_utf8(&buf[..header_end])
        .map_err(|_| ClientError::Invalid("HTTP headers are not UTF-8".into()))?;
    let mut lines = header.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ClientError::Invalid("empty HTTP response".into()))?;
    let mut parts = status_line.split_whitespace();
    let version = parts
        .next()
        .ok_or_else(|| ClientError::Invalid(format!("bad status line {status_line}")))?;
    if !version.starts_with("HTTP/1.") {
        return Err(ClientError::Invalid(format!(
            "unsupported version {version}"
        )));
    }
    let code: u16 = parts
        .next()
        .ok_or_else(|| ClientError::Invalid(format!("bad status line {status_line}")))?
        .parse()
        .map_err(|_| ClientError::Invalid(format!("bad status line {status_line}")))?;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| ClientError::Invalid(format!("bad header {line}")))?;
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = Some(
                value
                    .parse()
                    .map_err(|_| ClientError::Invalid(format!("bad content-length {value}")))?,
            );
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    let raw = &buf[header_end + 4..];
    let body = if chunked {
        decode_chunked(raw)?
    } else if let Some(len) = content_length {
        if raw.len() < len {
            return Err(ClientError::Invalid(format!(
                "truncated body: have {} of {len} bytes",
                raw.len()
            )));
        }
        raw[..len].to_vec()
    } else {
        raw.to_vec()
    };
    Ok((code, body))
}

fn decode_chunked(mut raw: &[u8]) -> Result<Vec<u8>, ClientError> {
    let mut out = Vec::new();
    loop {
        let line_end = find_subsequence(raw, b"\r\n")
            .ok_or_else(|| ClientError::Invalid("truncated chunk size".into()))?;
        let line = std::str::from_utf8(&raw[..line_end])
            .map_err(|_| ClientError::Invalid("chunk size is not UTF-8".into()))?;
        let size_hex = line.split(';').next().unwrap_or(line).trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| ClientError::Invalid(format!("bad chunk size {line}")))?;
        raw = &raw[line_end + 2..];
        if size == 0 {
            break;
        }
        if raw.len() < size + 2 {
            return Err(ClientError::Invalid("truncated chunk".into()));
        }
        out.extend_from_slice(&raw[..size]);
        if &raw[size..size + 2] != b"\r\n" {
            return Err(ClientError::Invalid("chunk missing CRLF".into()));
        }
        raw = &raw[size + 2..];
    }
    Ok(out)
}

fn find_subsequence(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// One `/v1/models` entry: id plus the optional template flag.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct LoadedModelInfo {
    pub id: String,
    /// The chat template ends the prompt with `<think>`, so replies start inside the thinking.
    #[serde(default, deserialize_with = "lenient")]
    pub template_opens_think: Option<bool>,
}

pub(crate) fn model_info_from_list(body: &[u8]) -> Result<Option<LoadedModelInfo>, ClientError> {
    #[derive(Deserialize)]
    struct List {
        data: Vec<LoadedModelInfo>,
    }
    let list: List = serde_json::from_slice(body).map_err(|e| ClientError::Json(e.to_string()))?;
    Ok(list.data.into_iter().next())
}

pub(crate) fn model_id_from_list(body: &[u8]) -> Result<Option<String>, ClientError> {
    #[derive(Deserialize)]
    struct List {
        data: Vec<Card>,
    }
    #[derive(Deserialize)]
    struct Card {
        id: String,
    }
    let list: List = serde_json::from_slice(body).map_err(|e| ClientError::Json(e.to_string()))?;
    Ok(list.data.into_iter().next().map(|c| c.id))
}

#[derive(Default)]
struct StreamAcc {
    id: Option<String>,
    created: u64,
    model: String,
    index: u32,
    role: String,
    content: String,
    saw_content: bool,
    finish_reason: Option<String>,
    usage: Option<ChatUsage>,
    timings: Option<ChatTimings>,
    tool_calls: Option<serde_json::Value>,
    done_finish_reason: Option<String>,
    max_tokens: Option<u64>,
    tokens: Option<ChatTokenCounts>,
    thinking_ms: Option<f64>,
    thinking_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    prefill_ms: Option<f64>,
    limit_hit_in_thinking: Option<bool>,
    error: Option<ChatErrorInfo>,
    connection_dropped: bool,
    cancel: Option<StreamCancel>,
}

impl StreamAcc {
    fn into_response(self) -> Result<ChatCompletionResponse, ClientError> {
        let id = self.id.ok_or_else(|| {
            ClientError::Json("SSE stream ended without a chat completion chunk".into())
        })?;
        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".into(),
            created: self.created,
            model: self.model,
            choices: vec![ChatChoice {
                index: self.index,
                message: ChatResponseMessage {
                    role: if self.role.is_empty() {
                        "assistant".into()
                    } else {
                        self.role
                    },
                    content: if self.saw_content {
                        Some(self.content)
                    } else {
                        None
                    },
                    tool_calls: self.tool_calls,
                },
                finish_reason: self.finish_reason,
            }],
            usage: self.usage.unwrap_or(ChatUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }),
            timings: self.timings,
            finish_reason: self.done_finish_reason,
            max_tokens: self.max_tokens,
            tokens: self.tokens,
            thinking_ms: self.thinking_ms,
            thinking_tokens: self.thinking_tokens,
            completion_tokens: self.completion_tokens,
            prefill_ms: self.prefill_ms,
            limit_hit_in_thinking: self.limit_hit_in_thinking,
            error: self.error,
            connection_dropped: self.connection_dropped,
        })
    }
}

#[derive(Deserialize)]
struct SseChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<SseChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
    #[serde(default)]
    timings: Option<ChatTimings>,
    #[serde(default, deserialize_with = "lenient")]
    finish_reason: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    max_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    tokens: Option<ChatTokenCounts>,
    #[serde(default, deserialize_with = "lenient")]
    thinking_ms: Option<f64>,
    #[serde(default, deserialize_with = "lenient")]
    thinking_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    completion_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient")]
    prefill_ms: Option<f64>,
    #[serde(default, deserialize_with = "lenient")]
    limit_hit_in_thinking: Option<bool>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct SseChoice {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    delta: Option<SseDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct SseDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
}

fn apply_sse_payload(
    payload: &str,
    acc: &mut StreamAcc,
    on_delta: &mut impl FnMut(&str),
) -> Result<(), ClientError> {
    let chunk: SseChunk = serde_json::from_str(payload)
        .map_err(|e| ClientError::Json(format!("{e}; sse={}", snippet(payload.as_bytes()))))?;
    if let Some(err) = chunk.error {
        let message = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("stream error");
        // The final chunk carries finish_reason "error" + error{message}; that is
        // an outcome, not a broken stream. A bare {"error":…} event is still an error.
        if chunk.finish_reason.as_deref() != Some("error") {
            return Err(ClientError::Invalid(format!("stream error: {message}")));
        }
        acc.error = Some(ChatErrorInfo {
            message: message.to_owned(),
            kind: err.get("type").and_then(|t| t.as_str()).map(str::to_owned),
        });
    }
    if let Some(id) = chunk.id {
        if acc.id.is_none() {
            if let Some(c) = &acc.cancel {
                c.set_id(&id);
            }
            acc.id = Some(id);
        }
    }
    if acc.created == 0 {
        if let Some(created) = chunk.created {
            acc.created = created;
        }
    }
    if let Some(model) = chunk.model {
        if !model.is_empty() {
            acc.model = model;
        }
    }
    if let Some(usage) = chunk.usage {
        acc.usage = Some(usage);
    }
    if let Some(timings) = chunk.timings {
        acc.timings = Some(timings);
    }
    if let Some(reason) = chunk.finish_reason {
        acc.done_finish_reason = Some(reason);
    }
    if let Some(max) = chunk.max_tokens {
        acc.max_tokens = Some(max);
    }
    if let Some(tokens) = chunk.tokens {
        acc.tokens = Some(tokens);
    }
    if let Some(ms) = chunk.thinking_ms {
        acc.thinking_ms = Some(ms);
    }
    if let Some(n) = chunk.thinking_tokens {
        acc.thinking_tokens = Some(n);
    }
    if let Some(n) = chunk.completion_tokens {
        acc.completion_tokens = Some(n);
    }
    if let Some(ms) = chunk.prefill_ms {
        acc.prefill_ms = Some(ms);
    }
    if let Some(b) = chunk.limit_hit_in_thinking {
        acc.limit_hit_in_thinking = Some(b);
    }
    if let Some(choice) = chunk.choices.into_iter().next() {
        acc.index = choice.index;
        if let Some(reason) = choice.finish_reason {
            acc.finish_reason = Some(reason);
        }
        if let Some(delta) = choice.delta {
            if let Some(role) = delta.role {
                acc.role = role;
            }
            if let Some(text) = delta.content {
                on_delta(&text);
                acc.content.push_str(&text);
                acc.saw_content = true;
            }
            if delta.tool_calls.is_some() {
                acc.tool_calls = delta.tool_calls;
            }
        }
    }
    Ok(())
}

/// Incremental SSE event splitter. Emits a payload only when a blank line
/// (`\\n\\n` or `\\r\\n\\r\\n`) has been seen. Does not invent events from a
/// partial buffer.
pub(crate) struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub(crate) fn push(&mut self, data: &[u8]) -> Result<Vec<String>, ClientError> {
        self.buf.extend_from_slice(data);
        if self.buf.len() > 8 * 1024 * 1024 {
            return Err(ClientError::Invalid("SSE buffer larger than 8 MiB".into()));
        }
        let mut out = Vec::new();
        while let Some((idx, sep)) = find_sse_event_end(&self.buf) {
            let raw = self.buf[..idx].to_vec();
            self.buf.drain(..idx + sep);
            if let Some(payload) = sse_event_data(&raw)? {
                out.push(payload);
            }
        }
        Ok(out)
    }
}

fn find_sse_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    let crlf = find_subsequence(buf, b"\r\n\r\n");
    let lf = find_subsequence(buf, b"\n\n");
    match (crlf, lf) {
        (Some(a), Some(b)) if a <= b => Some((a, 4)),
        (Some(_), Some(b)) => Some((b, 2)),
        (Some(a), None) => Some((a, 4)),
        (None, Some(b)) => Some((b, 2)),
        (None, None) => None,
    }
}

fn sse_event_data(raw: &[u8]) -> Result<Option<String>, ClientError> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| ClientError::Invalid("SSE event is not UTF-8".into()))?;
    let mut data_lines = Vec::new();
    for line in text.split('\n') {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data_lines.is_empty() {
        Ok(None)
    } else {
        Ok(Some(data_lines.join("\n")))
    }
}

struct RespMeta {
    status: u16,
    content_type: String,
    content_length: Option<usize>,
    chunked: bool,
}

fn parse_header_block(header: &str) -> Result<RespMeta, ClientError> {
    let mut lines = header.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ClientError::Invalid("empty HTTP response".into()))?;
    let mut parts = status_line.split_whitespace();
    let version = parts
        .next()
        .ok_or_else(|| ClientError::Invalid(format!("bad status line {status_line}")))?;
    if !version.starts_with("HTTP/1.") {
        return Err(ClientError::Invalid(format!(
            "unsupported version {version}"
        )));
    }
    let status: u16 = parts
        .next()
        .ok_or_else(|| ClientError::Invalid(format!("bad status line {status_line}")))?
        .parse()
        .map_err(|_| ClientError::Invalid(format!("bad status line {status_line}")))?;
    let mut content_type = String::new();
    let mut content_length = None;
    let mut chunked = false;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| ClientError::Invalid(format!("bad header {line}")))?;
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-type") && content_type.is_empty() {
            content_type = value.to_string();
        } else if name.eq_ignore_ascii_case("content-length") {
            content_length = Some(
                value
                    .parse()
                    .map_err(|_| ClientError::Invalid(format!("bad content-length {value}")))?,
            );
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
    }
    Ok(RespMeta {
        status,
        content_type,
        content_length,
        chunked,
    })
}

struct IoBuf {
    stream: TcpStream,
    buf: Vec<u8>,
    pos: usize,
}

impl IoBuf {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buf: Vec::new(),
            pos: 0,
        }
    }

    fn rest(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        } else if self.pos > 64 * 1024 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
    }

    fn fill(&mut self) -> Result<usize, ClientError> {
        let mut tmp = [0u8; 8192];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => return Ok(0),
                Ok(n) => {
                    self.buf.extend_from_slice(&tmp[..n]);
                    if self.buf.len() > 8 * 1024 * 1024 {
                        return Err(ClientError::Invalid("response larger than 8 MiB".into()));
                    }
                    return Ok(n);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(ClientError::Io(e)),
            }
        }
    }

    fn read_until(&mut self, needle: &[u8]) -> Result<Vec<u8>, ClientError> {
        loop {
            if let Some(i) = find_subsequence(self.rest(), needle) {
                let out = self.rest()[..i].to_vec();
                self.consume(i + needle.len());
                return Ok(out);
            }
            if self.fill()? == 0 {
                return Err(ClientError::Invalid("unexpected EOF".into()));
            }
        }
    }

    fn read_exact(&mut self, n: usize) -> Result<Vec<u8>, ClientError> {
        while self.rest().len() < n {
            if self.fill()? == 0 {
                return Err(ClientError::Invalid(format!(
                    "truncated body: have {} of {n} bytes",
                    self.rest().len()
                )));
            }
        }
        let out = self.rest()[..n].to_vec();
        self.consume(n);
        Ok(out)
    }
}

fn connect_tcp(endpoint: &Endpoint, timeout: Duration) -> Result<TcpStream, ClientError> {
    let addr = format!("{}:{}", endpoint.host, endpoint.port);
    let mut addrs = addr
        .to_socket_addrs()
        .map_err(|e| ClientError::Invalid(format!("resolve {addr}: {e}")))?;
    let addr = addrs
        .next()
        .ok_or_else(|| ClientError::Invalid(format!("resolve {addr}: no addresses")))?;
    let stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.set_nodelay(true)?;
    Ok(stream)
}

fn exchange_sse(
    endpoint: &Endpoint,
    timeout: Duration,
    path: &str,
    body: &[u8],
    on_delta: &mut impl FnMut(&str),
) -> Result<ChatCompletionResponse, ClientError> {
    exchange_sse_with(endpoint, timeout, path, body, on_delta, None)
}

fn exchange_sse_with(
    endpoint: &Endpoint,
    timeout: Duration,
    path: &str,
    body: &[u8],
    on_delta: &mut impl FnMut(&str),
    cancel: Option<&StreamCancel>,
) -> Result<ChatCompletionResponse, ClientError> {
    let mut stream = connect_tcp(endpoint, timeout)?;
    if let Some(c) = cancel {
        c.attach(&stream);
    }
    let req = build_http_request(&endpoint.host_header, "POST", path, Some(body));
    stream.write_all(&req)?;
    stream.flush()?;
    let mut io = IoBuf::new(stream);
    let header = io.read_until(b"\r\n\r\n")?;
    let header = std::str::from_utf8(&header)
        .map_err(|_| ClientError::Invalid("HTTP headers are not UTF-8".into()))?;
    let meta = parse_header_block(header)?;
    if !(200..300).contains(&meta.status) {
        let resp_body = read_rest_body(&mut io, &meta)?;
        return Err(ClientError::Status {
            status: meta.status,
            body: snippet(&resp_body),
        });
    }
    if !meta
        .content_type
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        let resp_body = read_rest_body(&mut io, &meta)?;
        let ct = if meta.content_type.is_empty() {
            "(none)".to_string()
        } else {
            meta.content_type.clone()
        };
        return Err(ClientError::Invalid(format!(
            "POST /v1/chat/completions with stream=true did not return text/event-stream (content-type: {ct}); body={}",
            snippet(&resp_body)
        )));
    }
    let mut parser = SseParser::new();
    let mut acc = StreamAcc {
        cancel: cancel.cloned(),
        ..StreamAcc::default()
    };
    let read = read_sse_body(&mut io, &meta, &mut parser, &mut acc, on_delta);
    if let Some(c) = cancel {
        if c.was_dropped() && !matches!(read, Ok(true)) {
            // Cut by StreamCancel after the grace period: return what arrived.
            acc.connection_dropped = true;
            if acc.done_finish_reason.is_none() {
                acc.done_finish_reason = Some("user".into());
            }
            if acc.id.is_none() {
                acc.id = Some(c.id().unwrap_or_else(|| "cancelled".into()));
            }
            return acc.into_response();
        }
    }
    let done = read?;
    if !done && acc.id.is_none() {
        return Err(ClientError::Json(
            "SSE stream ended without a chat completion chunk".into(),
        ));
    }
    acc.into_response()
}

fn read_sse_body(
    io: &mut IoBuf,
    meta: &RespMeta,
    parser: &mut SseParser,
    acc: &mut StreamAcc,
    on_delta: &mut impl FnMut(&str),
) -> Result<bool, ClientError> {
    if meta.chunked {
        loop {
            let line = io.read_until(b"\r\n")?;
            let line = std::str::from_utf8(&line)
                .map_err(|_| ClientError::Invalid("chunk size is not UTF-8".into()))?;
            let size_hex = line.split(';').next().unwrap_or(line).trim();
            if size_hex.is_empty() {
                continue;
            }
            let size = usize::from_str_radix(size_hex, 16)
                .map_err(|_| ClientError::Invalid(format!("bad chunk size {line}")))?;
            if size == 0 {
                return Ok(false);
            }
            let data = io.read_exact(size)?;
            let crlf = io.read_exact(2)?;
            if crlf.as_slice() != b"\r\n" {
                return Err(ClientError::Invalid("chunk missing CRLF".into()));
            }
            if dispatch_sse(parser, acc, on_delta, &data)? {
                return Ok(true);
            }
        }
    }
    if let Some(mut left) = meta.content_length {
        while left > 0 {
            let want = left.min(8192);
            let data = io.read_exact(want)?;
            left -= data.len();
            if dispatch_sse(parser, acc, on_delta, &data)? {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    loop {
        if !io.rest().is_empty() {
            let data = io.rest().to_vec();
            io.consume(data.len());
            if dispatch_sse(parser, acc, on_delta, &data)? {
                return Ok(true);
            }
            continue;
        }
        if io.fill()? == 0 {
            return Ok(false);
        }
    }
}

fn dispatch_sse(
    parser: &mut SseParser,
    acc: &mut StreamAcc,
    on_delta: &mut impl FnMut(&str),
    data: &[u8],
) -> Result<bool, ClientError> {
    for payload in parser.push(data)? {
        if payload.trim() == "[DONE]" {
            return Ok(true);
        }
        apply_sse_payload(&payload, acc, on_delta)?;
    }
    Ok(false)
}

fn read_rest_body(io: &mut IoBuf, meta: &RespMeta) -> Result<Vec<u8>, ClientError> {
    if meta.chunked {
        let mut out = Vec::new();
        loop {
            let line = io.read_until(b"\r\n")?;
            let line = std::str::from_utf8(&line)
                .map_err(|_| ClientError::Invalid("chunk size is not UTF-8".into()))?;
            let size_hex = line.split(';').next().unwrap_or(line).trim();
            if size_hex.is_empty() {
                continue;
            }
            let size = usize::from_str_radix(size_hex, 16)
                .map_err(|_| ClientError::Invalid(format!("bad chunk size {line}")))?;
            if size == 0 {
                break;
            }
            out.extend_from_slice(&io.read_exact(size)?);
            let crlf = io.read_exact(2)?;
            if crlf.as_slice() != b"\r\n" {
                return Err(ClientError::Invalid("chunk missing CRLF".into()));
            }
        }
        return Ok(out);
    }
    if let Some(len) = meta.content_length {
        return io.read_exact(len);
    }
    let mut out = io.rest().to_vec();
    io.consume(out.len());
    loop {
        let n = io.fill()?;
        if n == 0 {
            out.extend_from_slice(io.rest());
            let n = io.rest().len();
            io.consume(n);
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> &'static str {
        "127.0.0.1:8080"
    }

    // FIXTURE: not a live server. Request lines match what `Client` writes.
    // Response bodies match artificer-server routes.rs / mcp.rs shapes.
    #[test]
    fn fixture_health_request_and_response() {
        let req = build_http_request(host(), "GET", "/health", None);
        let text = String::from_utf8(req).unwrap();
        assert_eq!(
            text,
            "GET /health HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
        );
        let body = r#"{"status":"ok","metrics":{"requests_started":0,"requests_completed":0,"requests_failed":0,"prompt_tokens_total":0,"completion_tokens_total":0,"in_flight":0,"peak_in_flight":0}}"#;
        let raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let (status, parsed) = parse_http_response(raw.as_bytes()).unwrap();
        assert_eq!(status, 200);
        let health: HealthStatus = serde_json::from_slice(&parsed).unwrap();
        assert_eq!(health.status, "ok");
        assert_eq!(health.metrics.requests_started, 0);
    }

    // FIXTURE: additive backend fields on /health; absent on older servers.
    #[test]
    fn fixture_health_backend_fields() {
        let legacy = r#"{"status":"ok","metrics":{"requests_started":0,"requests_completed":0,"requests_failed":0,"prompt_tokens_total":0,"completion_tokens_total":0,"in_flight":0,"peak_in_flight":0}}"#;
        let h: HealthStatus = serde_json::from_str(legacy).unwrap();
        assert!(h.backend.is_none());
        let cuda = r#"{"status":"ok","metrics":{"requests_started":0,"requests_completed":0,"requests_failed":0,"prompt_tokens_total":0,"completion_tokens_total":0,"in_flight":0,"peak_in_flight":0},"backend":{"requested":"cuda","active":"cuda","device":"NVIDIA GeForce RTX 4050 Laptop GPU","fallback_reason":null}}"#;
        let b = serde_json::from_str::<HealthStatus>(cuda)
            .unwrap()
            .backend
            .unwrap();
        assert_eq!(b.active, "cuda");
        assert_eq!(
            b.device.as_deref(),
            Some("NVIDIA GeForce RTX 4050 Laptop GPU")
        );
        assert!(b.fallback_reason.is_none());
        let fb = r#"{"status":"ok","metrics":{"requests_started":0,"requests_completed":0,"requests_failed":0,"prompt_tokens_total":0,"completion_tokens_total":0,"in_flight":0,"peak_in_flight":0},"backend":{"active":"cpu","fallback_reason":"no CUDA device"}}"#;
        let b = serde_json::from_str::<HealthStatus>(fb)
            .unwrap()
            .backend
            .unwrap();
        assert_eq!(b.active, "cpu");
        assert!(b.requested.is_none());
        assert_eq!(b.fallback_reason.as_deref(), Some("no CUDA device"));
    }

    // FIXTURE: final SSE chunk carries additive `timings`; older chunks lack it.
    #[test]
    fn fixture_stream_final_chunk_timings() {
        let mut acc = StreamAcc::default();
        let mut deltas = Vec::new();
        let mut on = |d: &str| deltas.push(d.to_string());
        apply_sse_payload(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#, &mut acc, &mut on).unwrap();
        apply_sse_payload(r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":31,"completion_tokens":2,"total_tokens":33},"timings":{"prefill_tokens":31,"prefill_ms":100.0,"decode_tokens":2,"decode_ms":161.29,"prefill_tok_s":310.0,"decode_tok_s":12.4}}"#, &mut acc, &mut on).unwrap();
        let r = acc.into_response().unwrap();
        let t = r.timings.expect("timings");
        assert_eq!(t.prefill_tokens, 31);
        assert_eq!(t.decode_tokens, 2);
        assert!((t.decode_ms - 161.29).abs() < 1e-9);
        assert_eq!(r.usage.total_tokens, 33);
        assert_eq!(deltas, vec!["Hi".to_string()]);
        let legacy: ChatCompletionResponse = serde_json::from_str(r#"{"id":"x","object":"chat.completion","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#).unwrap();
        assert!(legacy.timings.is_none());
        assert!(
            legacy.finish_reason.is_none()
                && legacy.max_tokens.is_none()
                && legacy.tokens.is_none()
        );
        assert!(r.finish_reason.is_none() && r.max_tokens.is_none() && r.tokens.is_none());
    }

    // FIXTURE: proposed done-event fields (finish_reason, max_tokens, tokens) on the final chunk.
    #[test]
    fn fixture_stream_done_event_token_split() {
        let mut acc = StreamAcc::default();
        let mut on = |_: &str| {};
        apply_sse_payload(r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"plan</think>Hi"},"finish_reason":null}]}"#, &mut acc, &mut on).unwrap();
        apply_sse_payload(r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"finish_reason":"length","max_tokens":1024,"tokens":{"thinking":418,"answer":606,"total":1024}}"#, &mut acc, &mut on).unwrap();
        let r = acc.into_response().unwrap();
        assert_eq!(r.finish_reason.as_deref(), Some("length"));
        assert_eq!(r.max_tokens, Some(1024));
        assert_eq!(
            r.tokens,
            Some(ChatTokenCounts {
                thinking: Some(418),
                answer: Some(606),
                total: Some(1024)
            })
        );
        // Malformed extension values read as None instead of failing the stream.
        let mut acc = StreamAcc::default();
        apply_sse_payload(r#"{"id":"c2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"finish_reason":7,"max_tokens":"lots","tokens":{"thinking":-1,"total":3}}"#, &mut acc, &mut on).unwrap();
        let r = acc.into_response().unwrap();
        assert!(r.finish_reason.is_none() && r.max_tokens.is_none());
        assert_eq!(
            r.tokens,
            Some(ChatTokenCounts {
                thinking: None,
                answer: None,
                total: Some(3)
            })
        );
        assert_eq!(r.choices[0].finish_reason.as_deref(), Some("stop"));
        // thinking_ms and the tolerated top-level aliases.
        let mut acc = StreamAcc::default();
        apply_sse_payload(r#"{"id":"c3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"finish_reason":"cancelled","thinking_ms":11600.5,"thinking_tokens":300,"completion_tokens":410}"#, &mut acc, &mut on).unwrap();
        let r = acc.into_response().unwrap();
        assert_eq!(r.finish_reason.as_deref(), Some("cancelled"));
        assert_eq!(r.thinking_ms, Some(11600.5));
        assert_eq!(
            (r.thinking_tokens, r.completion_tokens),
            (Some(300), Some(410))
        );
    }

    #[test]
    fn fixture_models_template_opens_think() {
        let on = model_info_from_list(br#"{"object":"list","data":[{"id":"nm","object":"model","template_opens_think":true}]}"#).unwrap().unwrap();
        assert_eq!(
            (on.id.as_str(), on.template_opens_think),
            ("nm", Some(true))
        );
        let legacy =
            model_info_from_list(br#"{"object":"list","data":[{"id":"tv","object":"model"}]}"#)
                .unwrap()
                .unwrap();
        assert_eq!(legacy.template_opens_think, None);
        let junk = model_info_from_list(br#"{"data":[{"id":"x","template_opens_think":"yes"}]}"#)
            .unwrap()
            .unwrap();
        assert_eq!(junk.template_opens_think, None);
        assert!(model_info_from_list(br#"{"data":[]}"#).unwrap().is_none());
    }

    // FIXTURE: not a live server.
    #[test]
    fn fixture_chat_completion_request_and_response() {
        let request = ChatCompletionRequest::user("hi", 4);
        let payload = serde_json::to_vec(&request).unwrap();
        let req = build_http_request(host(), "POST", "/v1/chat/completions", Some(&payload));
        let text = String::from_utf8(req).unwrap();
        let expected_head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAccept: application/json\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            payload.len()
        );
        assert!(text.starts_with(&expected_head), "{text}");
        assert!(text.contains(r#""stream":false"#));
        assert!(text.contains(r#""role":"user"#));
        assert!(!text.contains("GET / HTTP"));
        let body = r#"{"id":"chatcmpl-fixture","object":"chat.completion","created":0,"model":"tiny-vicuna-1b.q4_k_m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let (status, parsed) = parse_http_response(raw.as_bytes()).unwrap();
        assert_eq!(status, 200);
        let chat: ChatCompletionResponse = serde_json::from_slice(&parsed).unwrap();
        assert_eq!(chat.object, "chat.completion");
        assert_eq!(chat.choices[0].message.content.as_deref(), Some("ok"));
        assert_eq!(chat.usage.total_tokens, 2);
    }

    // FIXTURE: not a live server. POST /mcp method tools/list, not a /tools/list URL.
    #[test]
    fn fixture_mcp_tools_list_request_and_response() {
        let payload = br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let req = build_http_request(host(), "POST", "/mcp", Some(payload));
        let text = String::from_utf8(req).unwrap();
        let expected = format!(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAccept: application/json\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            payload.len(),
            body = std::str::from_utf8(payload).unwrap()
        );
        assert_eq!(text, expected);
        let body = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"chat","title":"Chat completion","description":"Run a multi-turn chat completion against the local Artificer model. Messages use OpenAI-style roles (system/user/assistant)."},{"name":"complete","title":"Text completion","description":"Generate a completion from a single user prompt (chat-templated)."}]}}"#;
        let raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let (status, parsed) = parse_http_response(raw.as_bytes()).unwrap();
        assert_eq!(status, 200);
        let env: JsonRpcEnvelope = serde_json::from_slice(&parsed).unwrap();
        let tools = env.result.unwrap();
        let names: Vec<_> = tools.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["chat", "complete"]);
    }

    #[test]
    fn default_base_and_root_not_in_health_request() {
        let c = Client::new();
        assert_eq!(c.base_url(), "http://127.0.0.1:8080");
        let req = build_http_request(&c.endpoint.host_header, "GET", "/health", None);
        let text = String::from_utf8(req).unwrap();
        assert!(text.starts_with("GET /health "));
        assert!(!text.contains("GET / "));
    }

    /// Live check against 127.0.0.1:8080. Skipped unless ARTIFICER_LIVE=1.
    #[test]
    fn live_three_calls() {
        if std::env::var("ARTIFICER_LIVE").ok().as_deref() != Some("1") {
            return;
        }
        let mut c = Client::new();
        c.set_timeout(Duration::from_secs(180));
        let health = c.health().expect("health");
        eprintln!(
            "LIVE health status={} metrics_started={}",
            health.status, health.metrics.requests_started
        );
        let chat = c
            .chat_completion(&ChatCompletionRequest::user("Say hi in one word.", 8))
            .expect("chat");
        let content = chat
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        eprintln!(
            "LIVE chat object={} finish={:?} content={}",
            chat.object,
            chat.choices.first().and_then(|c| c.finish_reason.clone()),
            content.chars().take(80).collect::<String>()
        );
        let tools = c.mcp_tools_list().expect("mcp");
        let names: Vec<_> = tools.tools.iter().map(|t| t.name.as_str()).collect();
        eprintln!("LIVE mcp tools={names:?}");
        assert_eq!(health.status, "ok");
        assert_eq!(chat.object, "chat.completion");
        assert!(names.contains(&"chat") && names.contains(&"complete"));
    }

    #[test]
    fn rejects_stream_true() {
        let c = Client::new();
        let mut req = ChatCompletionRequest::user("hi", 1);
        req.stream = true;
        let err = c.chat_completion(&req).unwrap_err();
        assert!(err.to_string().contains("stream=false"));
    }

    // FIXTURE: not a live server. Shape matches GET /v1/models ModelList.
    #[test]
    fn fixture_loaded_model_request_and_body() {
        let req = build_http_request(host(), "GET", "/v1/models", None);
        let text = String::from_utf8(req).unwrap();
        assert_eq!(
            text,
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
        );
        let body = r#"{"object":"list","data":[{"id":"tiny-vicuna-1b.q4_k_m","object":"model","created":0,"owned_by":"local"}]}"#;
        let id = model_id_from_list(body.as_bytes()).unwrap();
        assert_eq!(id.as_deref(), Some("tiny-vicuna-1b.q4_k_m"));
        let empty = r#"{"object":"list","data":[]}"#;
        assert_eq!(model_id_from_list(empty.as_bytes()).unwrap(), None);
    }

    // FIXTURE: not a live server. Partial SSE must not emit a delta.
    #[test]
    fn fixture_sse_parser_waits_for_event_boundary() {
        let mut parser = SseParser::new();
        let partial = br#"data: {"choices":[{"delta":{"content":"A"}}]}"#;
        assert!(parser.push(partial).unwrap().is_empty());
        let rest = b"\n\n";
        let ev = parser.push(rest).unwrap();
        assert_eq!(ev.len(), 1);
        let mut acc = StreamAcc::default();
        let mut got = Vec::new();
        apply_sse_payload(&ev[0], &mut acc, &mut |d| got.push(d.to_string())).unwrap();
        assert_eq!(got, ["A"]);
        assert_eq!(acc.content, "A");
    }

    // FIXTURE: not a live server. One JSON completion is not split into deltas.
    #[test]
    fn fixture_json_body_is_not_fake_streamed() {
        let (port, req_ok, callbacks) = spawn_fixture_server(FixtureMode::JsonBody);
        let client = Client::with_base_url(format!("http://127.0.0.1:{port}")).unwrap();
        let req = ChatCompletionRequest::user("hi", 4);
        let mut deltas = Vec::new();
        let err = client
            .chat_completion_stream(&req, |d| deltas.push(d.to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("text/event-stream"), "{err}");
        assert!(deltas.is_empty());
        assert!(req_ok.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!req.stream);
        assert!(!callbacks.load(std::sync::atomic::Ordering::SeqCst));
    }

    // FIXTURE: not a live server. The second SSE chunk is withheld until the
    // first content delta callback runs, so a buffered reader would deadlock
    // until the server gives up.
    #[test]
    fn fixture_stream_callback_before_rest_of_body() {
        HELD_DELTA.store(false, std::sync::atomic::Ordering::SeqCst);
        let (port, req_ok, early) = spawn_fixture_server(FixtureMode::HeldSse);
        let mut client = Client::with_base_url(format!("http://127.0.0.1:{port}")).unwrap();
        client.set_timeout(Duration::from_secs(3));
        let req = ChatCompletionRequest::user("hi", 4);
        let mut deltas = Vec::new();
        let chat = client
            .chat_completion_stream(&req, |d| {
                deltas.push(d.to_string());
                if d == "A" {
                    HELD_DELTA.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            })
            .expect("sse");
        assert!(req_ok.load(std::sync::atomic::Ordering::SeqCst));
        assert!(
            early.load(std::sync::atomic::Ordering::SeqCst),
            "first content delta was not delivered before the rest of the body"
        );
        assert!(!req.stream);
        assert_eq!(deltas, ["A", "B"]);
        assert_eq!(chat.object, "chat.completion");
        assert_eq!(chat.model, "tiny-vicuna-1b.q4_k_m");
        assert_eq!(chat.choices[0].message.content.as_deref(), Some("AB"));
        assert_eq!(chat.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chat.usage.prompt_tokens, 81);
        assert_eq!(chat.usage.completion_tokens, 4);
        assert_eq!(chat.usage.total_tokens, 85);
    }

    // FIXTURE: not a live server.
    #[test]
    fn fixture_loaded_model_http() {
        let (port, req_ok, _) = spawn_fixture_server(FixtureMode::Models);
        let client = Client::with_base_url(format!("http://127.0.0.1:{port}")).unwrap();
        let id = client.loaded_model().unwrap();
        assert_eq!(id.as_deref(), Some("tiny-vicuna-1b.q4_k_m"));
        assert!(req_ok.load(std::sync::atomic::Ordering::SeqCst));
    }

    // FIXTURE: not a live server. Ephemeral port, not 8080.
    // Body is {"path":"<absolute>"} on POST /v1/models/load.
    #[test]
    fn fixture_load_model_posts_absolute_path() {
        let path = FIXTURE_MODEL_PATH;
        let (port, req_ok, _) = spawn_fixture_server(FixtureMode::Load);
        assert_ne!(port, 8080);
        let client = Client::with_base_url(format!("http://127.0.0.1:{port}")).unwrap();
        client.load_model(std::path::Path::new(path)).unwrap();
        assert!(req_ok.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn load_model_rejects_relative_path_without_connecting() {
        let c = Client::new();
        let err = c
            .load_model(std::path::Path::new("tiny-vicuna-1b.q4_k_m.gguf"))
            .unwrap_err();
        assert!(err.to_string().contains("absolute"));
    }

    #[test]
    fn stream_rejects_empty_messages_without_connecting() {
        let c = Client::new();
        let mut req = ChatCompletionRequest::user("hi", 1);
        req.messages.clear();
        let err = c.chat_completion_stream(&req, |_| {}).unwrap_err();
        assert!(err.to_string().contains("messages must not be empty"));
    }

    /// Live check against 127.0.0.1:8080. Skipped unless ARTIFICER_LIVE=1.
    #[test]
    fn live_stream_and_loaded_model() {
        if std::env::var("ARTIFICER_LIVE").ok().as_deref() != Some("1") {
            return;
        }
        let mut c = Client::new();
        c.set_timeout(Duration::from_secs(180));
        let model = c.loaded_model().expect("loaded_model");
        eprintln!("LIVE loaded_model={model:?}");
        let started = std::time::Instant::now();
        let mut marks: Vec<(Duration, String)> = Vec::new();
        let req = ChatCompletionRequest::user("Say hi", 4);
        let chat = c
            .chat_completion_stream(&req, |d| {
                let t = started.elapsed();
                eprintln!("LIVE delta +{t:?} {d:?}");
                marks.push((t, d.to_string()));
            })
            .expect("stream");
        let done = started.elapsed();
        let content = chat.choices[0].message.content.clone().unwrap_or_default();
        eprintln!(
            "LIVE stream status=ok object={} model={} finish={:?} content={:?} usage={{prompt:{},completion:{},total:{}}} deltas={} done={done:?}",
            chat.object,
            chat.model,
            chat.choices[0].finish_reason,
            content.chars().take(80).collect::<String>(),
            chat.usage.prompt_tokens,
            chat.usage.completion_tokens,
            chat.usage.total_tokens,
            marks.len(),
        );
        if let Some((t0, _)) = marks.first() {
            eprintln!(
                "LIVE first_delta={t0:?} lead_before_return={:?}",
                done.checked_sub(*t0)
            );
        }
        assert_eq!(model.as_deref(), Some("tiny-vicuna-1b.q4_k_m"));
        assert_eq!(chat.object, "chat.completion");
        assert_eq!(chat.model, "tiny-vicuna-1b.q4_k_m");
        assert!(!marks.is_empty());
        let joined: String = marks.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(content, joined);
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum FixtureMode {
    HeldSse,
    JsonBody,
    Models,
    Load,
}

/// Fixture model path for the load test: absolute on the host OS (Windows
/// needs a drive prefix) so the client's absolute-path guard accepts it. The
/// file is never opened; the fixture server only checks the JSON body.
#[cfg(all(test, windows))]
const FIXTURE_MODEL_PATH: &str = r"C:\fixture\tiny-vicuna-1b.q4_k_m.gguf";
#[cfg(all(test, not(windows)))]
const FIXTURE_MODEL_PATH: &str = "/fixture/tiny-vicuna-1b.q4_k_m.gguf";

#[cfg(test)]
fn spawn_fixture_server(
    mode: FixtureMode,
) -> (
    u16,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    let req_ok = Arc::new(AtomicBool::new(false));
    let early = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let req_flag = Arc::clone(&req_ok);
    let early_flag = Arc::clone(&early);
    thread::spawn(move || {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        tx.send(listener.local_addr().unwrap().port()).unwrap();
        let (mut sock, _) = listener.accept().unwrap();
        sock.set_nodelay(true).unwrap();
        sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 2048];
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match sock.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    if find_subsequence(&buf, b"\r\n\r\n").is_some() {
                        let header_end = find_subsequence(&buf, b"\r\n\r\n").unwrap();
                        let header = std::str::from_utf8(&buf[..header_end]).unwrap_or("");
                        let len = header
                            .lines()
                            .find_map(|l| {
                                l.split_once(':')
                                    .filter(|(n, _)| n.eq_ignore_ascii_case("Content-Length"))
                                    .and_then(|(_, v)| v.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if buf.len() >= header_end + 4 + len {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
        let text = String::from_utf8_lossy(&buf);
        let ok = match mode {
            FixtureMode::Models => text.starts_with("GET /v1/models "),
            FixtureMode::Load => {
                text.starts_with("POST /v1/models/load ")
                    && text.contains(&serde_json::json!({ "path": FIXTURE_MODEL_PATH }).to_string())
            }
            FixtureMode::HeldSse | FixtureMode::JsonBody => {
                text.starts_with("POST /v1/chat/completions ") && text.contains("\"stream\":true")
            }
        };
        req_flag.store(ok, Ordering::SeqCst);
        match mode {
            FixtureMode::Load => {
                let body = "{}";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
            FixtureMode::Models => {
                let body = r#"{"object":"list","data":[{"id":"tiny-vicuna-1b.q4_k_m","object":"model","created":0,"owned_by":"local"}]}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
            FixtureMode::JsonBody => {
                let body = r#"{"id":"chatcmpl-fixture","object":"chat.completion","created":0,"model":"tiny-vicuna-1b.q4_k_m","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes());
            }
            FixtureMode::HeldSse => {
                let headers = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\ntransfer-encoding: chunked\r\n\r\n";
                let role = sse_data(
                    r#"{"id":"chatcmpl-fixture","object":"chat.completion.chunk","created":1,"model":"tiny-vicuna-1b.q4_k_m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
                );
                let a = sse_data(
                    r#"{"id":"chatcmpl-fixture","object":"chat.completion.chunk","created":1,"model":"tiny-vicuna-1b.q4_k_m","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}"#,
                );
                let _ = sock.write_all(headers.as_bytes());
                let _ = sock.write_all(&http_chunk(&role));
                let _ = sock.write_all(&http_chunk(&a));
                let _ = sock.flush();
                let start = Instant::now();
                // Next chunks stay unsent until the client callback sees "A".
                while !held_delta_seen() && start.elapsed() < Duration::from_secs(2) {
                    thread::sleep(Duration::from_millis(5));
                }
                early_flag.store(held_delta_seen(), Ordering::SeqCst);
                let b = sse_data(
                    r#"{"id":"chatcmpl-fixture","object":"chat.completion.chunk","created":1,"model":"tiny-vicuna-1b.q4_k_m","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":null}]}"#,
                );
                let fin = sse_data(
                    r#"{"id":"chatcmpl-fixture","object":"chat.completion.chunk","created":1,"model":"tiny-vicuna-1b.q4_k_m","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":81,"completion_tokens":4,"total_tokens":85}}"#,
                );
                let done = sse_data("[DONE]");
                let mut rest = Vec::new();
                rest.extend_from_slice(&http_chunk(&b));
                rest.extend_from_slice(&http_chunk(&fin));
                rest.extend_from_slice(&http_chunk(&done));
                rest.extend_from_slice(b"0\r\n\r\n");
                let _ = sock.write_all(&rest);
                let _ = sock.flush();
            }
        }
    });
    let port = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    (port, req_ok, early)
}

#[cfg(test)]
fn sse_data(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

#[cfg(test)]
fn http_chunk(data: &str) -> Vec<u8> {
    let mut v = format!("{:X}\r\n", data.len()).into_bytes();
    v.extend_from_slice(data.as_bytes());
    v.extend_from_slice(b"\r\n");
    v
}

#[cfg(test)]
fn held_delta_seen() -> bool {
    HELD_DELTA.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(test)]
static HELD_DELTA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Cancel + server outcome fields against a scripted local HTTP server (no engine).
#[cfg(test)]
mod cancel_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[derive(Clone, Copy, PartialEq)]
    enum Cancel {
        /// `{"cancelled":true}`; the stream then sends its final chunk (finish_reason user).
        Honour,
        /// First cancel answers `{"cancelled":false}` (not registered yet), later ones true.
        LateRegister,
        /// Always `{"cancelled":false}` and the stream never ends (forces the drop).
        Ignore,
    }

    struct Mock {
        port: u16,
        cancel_bodies: Arc<Mutex<Vec<String>>>,
        stream_closed: Arc<AtomicBool>,
    }

    fn read_request(sock: &mut std::net::TcpStream) -> String {
        sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            match sock.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
            if let Some(h) = find_subsequence(&buf, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..h]).to_string();
                let len = head
                    .lines()
                    .find_map(|l| {
                        l.split_once(':')
                            .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                            .and_then(|(_, v)| v.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if buf.len() >= h + 4 + len {
                    break;
                }
            }
        }
        String::from_utf8_lossy(&buf).to_string()
    }

    fn chunk(sock: &mut std::net::TcpStream, payload: &str) -> std::io::Result<()> {
        sock.write_all(&http_chunk(&sse_data(payload)))?;
        sock.flush()
    }

    fn spawn(mode: Cancel, prefill: Duration) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let cancel_bodies = Arc::new(Mutex::new(Vec::new()));
        let stream_closed = Arc::new(AtomicBool::new(false));
        let (c2, calls2, bodies2, closed2) = (
            Arc::clone(&cancelled),
            Arc::clone(&cancel_calls),
            Arc::clone(&cancel_bodies),
            Arc::clone(&stream_closed),
        );
        thread::spawn(move || {
            for sock in listener.incoming() {
                let Ok(mut sock) = sock else { break };
                let (cancelled, calls, bodies, closed) = (
                    Arc::clone(&c2),
                    Arc::clone(&calls2),
                    Arc::clone(&bodies2),
                    Arc::clone(&closed2),
                );
                thread::spawn(move || {
                    let req = read_request(&mut sock);
                    if req.starts_with("POST /v1/chat/cancel ") {
                        let body = req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                        bodies.lock().unwrap().push(body);
                        let n = calls.fetch_add(1, Ordering::SeqCst);
                        let ok = match mode {
                            Cancel::Honour => true,
                            Cancel::LateRegister => n > 0,
                            Cancel::Ignore => false,
                        };
                        if ok {
                            cancelled.store(true, Ordering::SeqCst);
                        }
                        let b = format!(
                            r#"{{"cancelled":{ok},"id":{}}}"#,
                            if ok { r#""chatcmpl-x""# } else { "null" }
                        );
                        let _ = sock.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}", b.len()).as_bytes());
                        return;
                    }
                    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\ntransfer-encoding: chunked\r\n\r\n");
                    let _ = chunk(
                        &mut sock,
                        r#"{"id":"chatcmpl-x","model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
                    );
                    thread::sleep(prefill); // prefill is not interruptible
                    let mut sent = 0u64;
                    loop {
                        if cancelled.load(Ordering::SeqCst) {
                            let fin = format!(
                                r#"{{"id":"chatcmpl-x","model":"m","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":5,"completion_tokens":{sent},"total_tokens":{}}},"finish_reason":"user","max_tokens":128,"tokens":{{"thinking":{sent},"answer":0,"total":{sent}}},"thinking_ms":12.5,"prefill_ms":40.0,"limit_hit_in_thinking":false}}"#,
                                5 + sent
                            );
                            let _ = chunk(&mut sock, &fin);
                            let _ = chunk(&mut sock, "[DONE]");
                            let _ = sock.write_all(b"0\r\n\r\n");
                            return;
                        }
                        if chunk(&mut sock, r#"{"id":"chatcmpl-x","choices":[{"index":0,"delta":{"content":"t "},"finish_reason":null}]}"#).is_err() {
                            closed.store(true, Ordering::SeqCst);
                            return;
                        }
                        sent += 1;
                        thread::sleep(Duration::from_millis(15));
                    }
                });
            }
        });
        Mock {
            port,
            cancel_bodies,
            stream_closed,
        }
    }

    fn run(
        mock: &Mock,
        stop_after: Duration,
        grace: Duration,
    ) -> (
        Result<ChatCompletionResponse, ClientError>,
        CancelOutcome,
        usize,
    ) {
        let mut c = Client::with_base_url(format!("http://127.0.0.1:{}", mock.port)).unwrap();
        c.set_timeout(Duration::from_secs(5));
        let handle = StreamCancel::new();
        let (h2, c2) = (handle.clone(), c.clone());
        let stopper = thread::spawn(move || {
            thread::sleep(stop_after);
            h2.cancel(&c2, grace)
        });
        let mut deltas = 0usize;
        let req = ChatCompletionRequest::user("hi", 128);
        let res = c.chat_completion_stream_cancellable(&req, &handle, |_| deltas += 1);
        let outcome = stopper.join().unwrap();
        assert!(handle.is_finished());
        (res, outcome, deltas)
    }

    #[test]
    fn stop_reads_until_final_chunk_with_server_counts() {
        let mock = spawn(Cancel::Honour, Duration::ZERO);
        let (res, outcome, deltas) = run(&mock, Duration::from_millis(120), Duration::from_secs(3));
        let r = res.expect("stream ends normally after cancel");
        assert_eq!(
            outcome,
            CancelOutcome {
                server_ack: true,
                dropped: false
            }
        );
        assert_eq!(r.finish_reason.as_deref(), Some("user"));
        assert!(!r.connection_dropped);
        let t = r.tokens.expect("server counts");
        assert!(
            t.total.unwrap() >= deltas as u64 - 1,
            "server total {:?} vs deltas {deltas}",
            t.total
        );
        assert_eq!(r.max_tokens, Some(128));
        assert_eq!(r.prefill_ms, Some(40.0));
        assert_eq!(r.limit_hit_in_thinking, Some(false));
        // The cancel names the completion it saw on the stream.
        assert_eq!(
            mock.cancel_bodies.lock().unwrap()[0],
            r#"{"id":"chatcmpl-x"}"#
        );
    }

    #[test]
    fn stop_before_server_registers_is_retried() {
        let mock = spawn(Cancel::LateRegister, Duration::ZERO);
        let (res, outcome, _) = run(&mock, Duration::from_millis(60), Duration::from_secs(3));
        assert_eq!(res.unwrap().finish_reason.as_deref(), Some("user"));
        assert_eq!(
            outcome,
            CancelOutcome {
                server_ack: true,
                dropped: false
            }
        );
        assert!(mock.cancel_bodies.lock().unwrap().len() >= 2);
    }

    /// Mock server that only stops after a 400 ms prefill (servers without chunked prefill cancellation): the
    /// client still waits for the final chunk instead of dropping the stream.
    #[test]
    fn stop_during_prefill_ends_after_prefill() {
        let mock = spawn(Cancel::Honour, Duration::from_millis(400));
        let started = Instant::now();
        let (res, outcome, deltas) = run(&mock, Duration::from_millis(50), Duration::from_secs(3));
        let r = res.unwrap();
        assert_eq!(deltas, 0, "no tokens after a Stop during prefill");
        assert_eq!(r.finish_reason.as_deref(), Some("user"));
        assert_eq!(r.tokens.unwrap().total, Some(0));
        assert!(!outcome.dropped);
        assert!(started.elapsed() >= Duration::from_millis(380));
    }

    #[test]
    fn stop_without_final_chunk_drops_connection_after_grace() {
        let mock = spawn(Cancel::Ignore, Duration::ZERO);
        let started = Instant::now();
        let (res, outcome, deltas) =
            run(&mock, Duration::from_millis(80), Duration::from_millis(300));
        let r = res.expect("partial reply, not an error");
        assert_eq!(
            outcome,
            CancelOutcome {
                server_ack: false,
                dropped: true
            }
        );
        assert!(r.connection_dropped);
        assert_eq!(r.finish_reason.as_deref(), Some("user"));
        assert!(
            r.tokens.is_none(),
            "no server counts without the final chunk"
        );
        assert!(deltas > 0);
        assert!(started.elapsed() < Duration::from_secs(2));
        // The server side sees the closed socket on its next write.
        let t = Instant::now();
        while !mock.stream_closed.load(Ordering::SeqCst) && t.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(mock.stream_closed.load(Ordering::SeqCst));
    }

    #[test]
    fn cancel_chat_without_id_sends_empty_object() {
        let mock = spawn(Cancel::Honour, Duration::ZERO);
        let c = Client::with_base_url(format!("http://127.0.0.1:{}", mock.port)).unwrap();
        let ack = c.cancel_chat(None).unwrap();
        assert_eq!(
            ack,
            CancelAck {
                cancelled: true,
                id: Some("chatcmpl-x".into())
            }
        );
        assert_eq!(mock.cancel_bodies.lock().unwrap()[0], "{}");
    }

    #[test]
    fn final_error_chunk_is_an_outcome_bare_error_is_not() {
        let mut acc = StreamAcc::default();
        let mut on = |_: &str| {};
        apply_sse_payload(r#"{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":null}],"finish_reason":"error","max_tokens":8,"tokens":{"thinking":0,"answer":0,"total":0},"thinking_ms":0.0,"prefill_ms":0.0,"limit_hit_in_thinking":false,"error":{"message":"prompt too long: 9000 > 4096","type":"server_error"}}"#, &mut acc, &mut on).unwrap();
        let r = acc.into_response().unwrap();
        assert_eq!(r.finish_reason.as_deref(), Some("error"));
        let e = r.error.expect("error object");
        assert_eq!(e.message, "prompt too long: 9000 > 4096");
        assert_eq!(e.kind.as_deref(), Some("server_error"));
        let mut acc = StreamAcc::default();
        let err = apply_sse_payload(
            r#"{"error":{"message":"boom","type":"server_error"}}"#,
            &mut acc,
            &mut on,
        )
        .unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    #[test]
    fn limit_hit_in_thinking_and_prefill_ms_parse() {
        let mut acc = StreamAcc::default();
        let mut on = |_: &str| {};
        apply_sse_payload(r#"{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":28,"completion_tokens":3,"total_tokens":31},"finish_reason":"length","max_tokens":3,"tokens":{"thinking":3,"answer":0,"total":3},"thinking_ms":230.658,"prefill_ms":2937.546,"limit_hit_in_thinking":true}"#, &mut acc, &mut on).unwrap();
        let r = acc.into_response().unwrap();
        assert_eq!(r.limit_hit_in_thinking, Some(true));
        assert_eq!(r.prefill_ms, Some(2937.546));
        assert_eq!(r.thinking_ms, Some(230.658));
        assert_eq!(r.usage.completion_tokens, 3);
    }
}
