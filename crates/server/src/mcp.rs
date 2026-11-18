//! Model Context Protocol (MCP) server surface for Artificer.
//!
//! Implements JSON-RPC 2.0 over:
//! - **stdio** (`artificer mcp`) — newline-delimited messages (MCP stdio transport)
//! - **HTTP** `POST /mcp` — single JSON-RPC request/response (simple Streamable-HTTP style)
//!
//! Exposed tools (agentic use of the local model):
//! - `chat` — multi-turn chat completion
//! - `complete` — single-prompt completion

use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

use artificer_runtime::{Engine, GenerateConfig};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api_types::ChatMessage;
use crate::error::{err_response, Result, ServerError};
use crate::metrics::ServerMetrics;
use crate::prompt::messages_to_prompt;
use crate::routes::AppState;
use crate::tools::ToolMode;

const PROTOCOL_VERSION: &str = "2025-06-18";
const SERVER_NAME: &str = "artificer";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// ── JSON-RPC types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(default)]
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    fn result(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Notifications have no response body (return None).
    fn is_notification_ack() -> Option<Self> {
        None
    }
}

// ── Shared handler ──────────────────────────────────────────────────────────

/// Process one JSON-RPC message. Returns `None` for notifications (no reply).
pub fn handle_rpc(state: &McpState, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    let id = req.id.clone();
    // Notifications have no id.
    let is_notification = id.is_none();

    match req.method.as_str() {
        "initialize" => Some(JsonRpcResponse::result(id, initialize_result())),
        "notifications/initialized" | "initialized" => JsonRpcResponse::is_notification_ack(),
        "ping" => Some(JsonRpcResponse::result(id, json!({}))),
        "tools/list" => Some(JsonRpcResponse::result(id, tools_list_result())),
        "tools/call" => {
            let params = req.params.unwrap_or(json!({}));
            Some(tools_call(state, id, params))
        }
        "resources/list" => Some(JsonRpcResponse::result(id, json!({ "resources": [] }))),
        "prompts/list" => Some(JsonRpcResponse::result(id, prompts_list_result())),
        "prompts/get" => {
            let params = req.params.unwrap_or(json!({}));
            Some(prompts_get(id, params))
        }
        other => {
            if is_notification {
                // Ignore unknown notifications.
                None
            } else {
                Some(JsonRpcResponse::error(
                    id,
                    -32601,
                    format!("Method not found: {other}"),
                ))
            }
        }
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {},
            "prompts": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        },
        "instructions": "Artificer local LLM. Use the `chat` tool for multi-turn chat completions and `complete` for single-prompt generation against the loaded GGUF model."
    })
}

fn tools_list_result() -> Value {
    json!({
        "tools": [
            {
                "name": "chat",
                "title": "Chat completion",
                "description": "Run a multi-turn chat completion against the local Artificer model. Messages use OpenAI-style roles (system/user/assistant).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "Chat messages",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": { "type": "string", "description": "system | user | assistant" },
                                    "content": { "type": "string" }
                                },
                                "required": ["role", "content"]
                            }
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Max new tokens (default 1024)",
                            "minimum": 1
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (default 0.7)"
                        },
                        "top_p": { "type": "number" },
                        "top_k": { "type": "integer" },
                        "seed": { "type": "integer" }
                    },
                    "required": ["messages"]
                }
            },
            {
                "name": "complete",
                "title": "Text completion",
                "description": "Generate a completion from a single user prompt (chat-templated).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "User prompt text"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Max new tokens (default 1024)",
                            "minimum": 1
                        },
                        "temperature": { "type": "number" },
                        "top_p": { "type": "number" },
                        "top_k": { "type": "integer" },
                        "seed": { "type": "integer" }
                    },
                    "required": ["prompt"]
                }
            }
        ]
    })
}

fn prompts_list_result() -> Value {
    json!({
        "prompts": [
            {
                "name": "chat",
                "title": "Simple chat",
                "description": "A single-user chat turn for the local model",
                "arguments": [
                    {
                        "name": "message",
                        "description": "User message",
                        "required": true
                    }
                ]
            }
        ]
    })
}

fn prompts_get(id: Option<Value>, params: Value) -> JsonRpcResponse {
    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if name != "chat" {
        return JsonRpcResponse::error(id, -32602, format!("Unknown prompt: {name}"));
    }
    let message = params
        .get("arguments")
        .and_then(|a| a.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    JsonRpcResponse::result(
        id,
        json!({
            "description": "Simple chat",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": message
                    }
                }
            ]
        }),
    )
}

fn tools_call(state: &McpState, id: Option<Value>, params: Value) -> JsonRpcResponse {
    let name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return JsonRpcResponse::error(id, -32602, "tools/call requires params.name"),
    };
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = match name {
        "chat" => call_chat(state, &args),
        "complete" => call_complete(state, &args),
        other => {
            return JsonRpcResponse::error(id, -32602, format!("Unknown tool: {other}"));
        }
    };

    match result {
        Ok(text) => JsonRpcResponse::result(
            id,
            json!({
                "content": [
                    { "type": "text", "text": text }
                ],
                "isError": false
            }),
        ),
        Err(e) => JsonRpcResponse::result(
            id,
            json!({
                "content": [
                    { "type": "text", "text": e }
                ],
                "isError": true
            }),
        ),
    }
}

fn call_chat(state: &McpState, args: &Value) -> std::result::Result<String, String> {
    let messages_val = args
        .get("messages")
        .ok_or_else(|| "chat requires arguments.messages".to_string())?;
    let raw: Vec<ChatMessage> = serde_json::from_value(messages_val.clone())
        .map_err(|e| format!("invalid messages: {e}"))?;
    if raw.is_empty() {
        return Err("messages must not be empty".into());
    }
    if raw.len() > state.max_messages {
        return Err(format!("too many messages (max {})", state.max_messages));
    }

    let cfg = gen_cfg_from_args(args, state.max_tokens);
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())?;

    let prompt = messages_to_prompt(&engine.model().tokenizer, &raw, &[], &ToolMode::None)?;

    let guard = state.metrics.begin_request();
    let out = match engine.generate_prompt(&prompt.0, &cfg) {
        Ok(o) => {
            guard.success(
                o.prompt_tokens.len() as u64,
                o.completion_tokens.len() as u64,
            );
            o
        }
        Err(e) => {
            guard.fail();
            return Err(e.to_string());
        }
    };
    Ok(out.text.trim().to_owned())
}

fn call_complete(state: &McpState, args: &Value) -> std::result::Result<String, String> {
    let prompt_text = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "complete requires arguments.prompt".to_string())?;
    if prompt_text.is_empty() {
        return Err("prompt must not be empty".into());
    }
    if prompt_text.len() > state.max_prompt_chars {
        return Err(format!(
            "prompt too large (max {} chars)",
            state.max_prompt_chars
        ));
    }

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: Some(prompt_text.to_owned()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    let cfg = gen_cfg_from_args(args, state.max_tokens);
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| "engine lock poisoned".to_string())?;

    let prompt = messages_to_prompt(&engine.model().tokenizer, &messages, &[], &ToolMode::None)?;

    let guard = state.metrics.begin_request();
    let out = match engine.generate_prompt(&prompt.0, &cfg) {
        Ok(o) => {
            guard.success(
                o.prompt_tokens.len() as u64,
                o.completion_tokens.len() as u64,
            );
            o
        }
        Err(e) => {
            guard.fail();
            return Err(e.to_string());
        }
    };
    Ok(out.text.trim().to_owned())
}

fn gen_cfg_from_args(args: &Value, max_tokens_cap: usize) -> GenerateConfig {
    let max_tokens = args
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(artificer_core::DEFAULT_MAX_TOKENS as usize)
        .clamp(1, max_tokens_cap);
    let temperature = args
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.7)
        .clamp(0.0, 2.0);
    let top_p = args
        .get("top_p")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.95)
        .clamp(0.0, 1.0);
    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_i64())
        .map(|n| n as i32)
        .unwrap_or(40)
        .clamp(0, 100_000);
    let seed = args.get("seed").and_then(|v| v.as_u64());

    GenerateConfig {
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop_on_eos: true,
        seed,
        stop_token_ids: Vec::new(),
        stop_strings: {
            let mut s = vec!["\nUSER:".into(), "\nUser:".into(), "\nASSISTANT:".into()];
            for x in artificer_runtime::CHAT_CONTROL_STOP_STRINGS {
                if !s.iter().any(|y| y == *x) {
                    s.push((*x).to_owned());
                }
            }
            s
        },
    }
}

// ── State ───────────────────────────────────────────────────────────────────

/// MCP-facing engine state (stdio uses std Mutex; HTTP reuses AppState).
pub struct McpState {
    pub engine: std::sync::Mutex<Engine>,
    pub model_id: String,
    pub metrics: ServerMetrics,
    pub max_tokens: usize,
    pub max_messages: usize,
    pub max_prompt_chars: usize,
}

impl McpState {
    pub fn from_engine(engine: Engine, model_id: String) -> Self {
        Self {
            engine: std::sync::Mutex::new(engine),
            model_id,
            metrics: ServerMetrics::default(),
            max_tokens: artificer_core::MAX_TOKENS_CEILING as usize,
            max_messages: 256,
            max_prompt_chars: 200_000,
        }
    }
}

// ── stdio transport ─────────────────────────────────────────────────────────

/// Run MCP over stdin/stdout (newline-delimited JSON-RPC). Logging must go to stderr.
pub fn serve_stdio(state: McpState) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    tracing::info!(
        "MCP stdio server ready (model_id={}, protocol={})",
        state.model_id,
        PROTOCOL_VERSION
    );

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break; // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                write_rpc_line(&mut writer, &resp)?;
                continue;
            }
        };

        if let Some(resp) = handle_rpc(&state, req) {
            write_rpc_line(&mut writer, &resp)?;
        }
    }
    Ok(())
}

fn write_rpc_line(writer: &mut impl Write, resp: &JsonRpcResponse) -> Result<()> {
    serde_json::to_writer(&mut *writer, resp).map_err(|e| ServerError::Other(e.to_string()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

// ── HTTP transport ──────────────────────────────────────────────────────────

/// `POST /mcp` — one JSON-RPC request body → JSON-RPC response (or 204 for notifications).
pub async fn mcp_http(State(state): State<Arc<AppState>>, body: String) -> Response {
    let req: JsonRpcRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(JsonRpcResponse::error(
                    None,
                    -32700,
                    format!("Parse error: {e}"),
                )),
            )
                .into_response();
        }
    };

    let method = req.method.clone();
    let state_bg = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || handle_rpc_from_app(&state_bg, req)).await;

    match result {
        Ok(None) => {
            tracing::debug!(%method, "MCP notification");
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(Some(resp)) => Json(resp).into_response(),
        Err(e) => err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("MCP handler join error: {e}"),
        ),
    }
}

fn handle_rpc_from_app(state: &AppState, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
    // Bridge tokio Mutex → std-style by blocking_lock for the duration of the call.
    // We implement tools/call specially so we can use AppState's engine mutex.
    let id = req.id.clone();
    let is_notification = id.is_none();

    match req.method.as_str() {
        "initialize" => Some(JsonRpcResponse::result(id, initialize_result())),
        "notifications/initialized" | "initialized" => None,
        "ping" => Some(JsonRpcResponse::result(id, json!({}))),
        "tools/list" => Some(JsonRpcResponse::result(id, tools_list_result())),
        "resources/list" => Some(JsonRpcResponse::result(id, json!({ "resources": [] }))),
        "prompts/list" => Some(JsonRpcResponse::result(id, prompts_list_result())),
        "prompts/get" => {
            let params = req.params.unwrap_or(json!({}));
            Some(prompts_get(id, params))
        }
        "tools/call" => {
            let params = req.params.unwrap_or(json!({}));
            Some(tools_call_app(state, id, params))
        }
        other => {
            if is_notification {
                None
            } else {
                Some(JsonRpcResponse::error(
                    id,
                    -32601,
                    format!("Method not found: {other}"),
                ))
            }
        }
    }
}

fn tools_call_app(state: &AppState, id: Option<Value>, params: Value) -> JsonRpcResponse {
    let name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return JsonRpcResponse::error(id, -32602, "tools/call requires params.name"),
    };
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = match name.as_str() {
        "chat" => call_chat_app(state, &args),
        "complete" => call_complete_app(state, &args),
        other => {
            return JsonRpcResponse::error(id, -32602, format!("Unknown tool: {other}"));
        }
    };

    match result {
        Ok(text) => JsonRpcResponse::result(
            id,
            json!({
                "content": [
                    { "type": "text", "text": text }
                ],
                "isError": false
            }),
        ),
        Err(e) => JsonRpcResponse::result(
            id,
            json!({
                "content": [
                    { "type": "text", "text": e }
                ],
                "isError": true
            }),
        ),
    }
}

fn call_chat_app(state: &AppState, args: &Value) -> std::result::Result<String, String> {
    let messages_val = args
        .get("messages")
        .ok_or_else(|| "chat requires arguments.messages".to_string())?;
    let raw: Vec<ChatMessage> = serde_json::from_value(messages_val.clone())
        .map_err(|e| format!("invalid messages: {e}"))?;
    if raw.is_empty() {
        return Err("messages must not be empty".into());
    }
    if raw.len() > state.limits.max_messages {
        return Err(format!(
            "too many messages (max {})",
            state.limits.max_messages
        ));
    }

    let cfg = gen_cfg_from_args(args, state.limits.max_tokens);
    let mut engine = state.engine.blocking_lock();
    let prompt = messages_to_prompt(&engine.model().tokenizer, &raw, &[], &ToolMode::None)?;

    let guard = state.metrics.begin_request();
    match engine.generate_prompt(&prompt.0, &cfg) {
        Ok(o) => {
            guard.success(
                o.prompt_tokens.len() as u64,
                o.completion_tokens.len() as u64,
            );
            Ok(o.text.trim().to_owned())
        }
        Err(e) => {
            guard.fail();
            Err(e.to_string())
        }
    }
}

fn call_complete_app(state: &AppState, args: &Value) -> std::result::Result<String, String> {
    let prompt_text = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "complete requires arguments.prompt".to_string())?;
    if prompt_text.is_empty() {
        return Err("prompt must not be empty".into());
    }
    if prompt_text.len() > state.limits.max_prompt_chars {
        return Err(format!(
            "prompt too large (max {} chars)",
            state.limits.max_prompt_chars
        ));
    }

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: Some(prompt_text.to_owned()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    let cfg = gen_cfg_from_args(args, state.limits.max_tokens);
    let mut engine = state.engine.blocking_lock();
    let prompt = messages_to_prompt(&engine.model().tokenizer, &messages, &[], &ToolMode::None)?;

    let guard = state.metrics.begin_request();
    match engine.generate_prompt(&prompt.0, &cfg) {
        Ok(o) => {
            guard.success(
                o.prompt_tokens.len() as u64,
                o.completion_tokens.len() as u64,
            );
            Ok(o.text.trim().to_owned())
        }
        Err(e) => {
            guard.fail();
            Err(e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_and_tools_list_shape() {
        // No engine needed for these methods — use a dummy state only if tools/call.
        // Test pure result builders.
        let init = initialize_result();
        assert_eq!(init["protocolVersion"], PROTOCOL_VERSION);
        assert!(init["capabilities"]["tools"].is_object());

        let tools = tools_list_result();
        let arr = tools["tools"].as_array().unwrap();
        assert!(arr.iter().any(|t| t["name"] == "chat"));
        assert!(arr.iter().any(|t| t["name"] == "complete"));
    }

    #[test]
    fn max_tokens_defaults_are_the_shared_constants() {
        let def = artificer_core::DEFAULT_MAX_TOKENS as usize;
        let ceil = artificer_core::MAX_TOKENS_CEILING as usize;
        assert_eq!(def, 1024);
        assert_eq!(ceil, 4096);
        // MCP tool args without max_tokens, and over-cap requests.
        assert_eq!(gen_cfg_from_args(&json!({}), ceil).max_tokens, def);
        assert_eq!(
            gen_cfg_from_args(&json!({"max_tokens": 100_000}), ceil).max_tokens,
            ceil
        );
        assert_eq!(crate::config::RequestLimits::default().max_tokens, ceil);
        assert_eq!(artificer_runtime::GenerateConfig::default().max_tokens, def);
        // OpenAI-style chat request without max_tokens.
        let req: crate::api_types::ChatCompletionRequest =
            serde_json::from_value(json!({"messages": [{"role": "user", "content": "hi"}]}))
                .unwrap();
        assert_eq!(req.max_tokens, def);
        // Advertised default in the tool schema matches.
        let tools = tools_list_result().to_string();
        assert!(tools.contains(&format!("(default {def})")));
    }

    #[test]
    fn method_not_found() {
        // Build a minimal McpState without loading a real model by not calling tools.
        // handle_rpc for unknown method with id:
        // We can't easily construct McpState without Engine — test response builder only.
        let resp = JsonRpcResponse::error(Some(json!(1)), -32601, "Method not found: foo");
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn prompts_get_unknown() {
        let resp = prompts_get(Some(json!(1)), json!({"name": "nope"}));
        assert!(resp.error.is_some());
    }

    #[test]
    fn prompts_get_chat() {
        let resp = prompts_get(
            Some(json!(1)),
            json!({"name": "chat", "arguments": {"message": "hello"}}),
        );
        let result = resp.result.unwrap();
        assert_eq!(result["messages"][0]["content"]["text"], "hello");
    }
}
