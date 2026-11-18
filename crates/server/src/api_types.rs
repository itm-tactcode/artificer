//! OpenAI-compatible request/response JSON shapes (including tools / function calling).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Messages ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    /// Present on assistant turns that requested tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present on `role: "tool"` messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional tool/function name (legacy function-calling / tool messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// ── Tools ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema object for parameters (OpenAI `parameters`).
    #[serde(default)]
    pub parameters: Option<Value>,
}

/// OpenAI `tool_choice`: `"none"` | `"auto"` | `"required"` | `{ "type":"function", "function": { "name": "..." } }`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(ToolChoiceNamed),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceNamed {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: ToolChoiceFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments object (OpenAI wire format: string, not object).
    pub arguments: String,
}

// ── Request ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopValue {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopValue>,
    /// OpenAI tools (function definitions).
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Legacy alias: single function (mapped to tools when `tools` is absent).
    #[serde(default)]
    pub functions: Option<Vec<FunctionDef>>,
    #[serde(default)]
    pub function_call: Option<Value>,
}

fn default_max_tokens() -> usize {
    artificer_core::DEFAULT_MAX_TOKENS as usize
}

// ── Response ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    /// Additive Artificer extension: prefill/decode timing (see `routes::timings_json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Value>,
    /// Additive top-level outcome: finish_reason, max_tokens, tokens, thinking_ms, …
    #[serde(flatten)]
    pub outcome: Outcome,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    /// OpenAI sends `null` when the assistant only returned `tool_calls`.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// OpenAI `usage` (unchanged shape).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    /// Every sampled token, thinking included (this is what `max_tokens` limits).
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn from_output(out: &artificer_runtime::GenerateOutput) -> Self {
        Self {
            prompt_tokens: out.prompt_tokens.len(),
            completion_tokens: out.completion_tokens.len(),
            total_tokens: out.prompt_tokens.len() + out.completion_tokens.len(),
        }
    }
}

/// Completion token split. `total == thinking + answer == usage.completion_tokens`.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct TokenCounts {
    pub thinking: usize,
    pub answer: usize,
    pub total: usize,
}

/// Top-level Artificer outcome fields on the non-streaming response and on the final
/// streaming chunk (additive; OpenAI clients ignore them). The UI reads only these.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Outcome {
    /// `"stop"` | `"length"` | `"user"` (cancelled) | `"error"`.
    pub finish_reason: &'static str,
    /// Effective cap for this request; it counts thinking tokens too.
    pub max_tokens: usize,
    pub tokens: TokenCounts,
    /// First generated token → `</think>` (or the end if it never closed), prefill
    /// excluded. `0` when there was no thinking.
    pub thinking_ms: f64,
    /// Prompt prefill wall time.
    pub prefill_ms: f64,
    /// `finish_reason == "length"` while the model was still thinking.
    pub limit_hit_in_thinking: bool,
    /// Present when `finish_reason == "error"`: `{"message": …, "type": "server_error"}`
    /// (OpenAI error shape, so existing clients surface the message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ErrorInfo {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

impl Outcome {
    pub fn from_output(out: &artificer_runtime::GenerateOutput, max_tokens: usize) -> Self {
        use artificer_runtime::FinishReason;
        let total = out.completion_tokens.len();
        let thinking = out.thinking.tokens.min(total);
        Self {
            finish_reason: match out.finish_reason {
                FinishReason::Stop => "stop",
                FinishReason::Length => "length",
                FinishReason::Cancelled => "user",
            },
            max_tokens,
            tokens: TokenCounts {
                thinking,
                answer: total - thinking,
                total,
            },
            thinking_ms: round3(out.thinking.ms.unwrap_or(0.0)),
            prefill_ms: round3(out.timings.prefill_ms),
            limit_hit_in_thinking: out.limit_hit_in_thinking(),
            error: None,
        }
    }

    /// Generation failed; no reliable counts.
    pub fn error(max_tokens: usize, message: impl Into<String>) -> Self {
        Self {
            finish_reason: "error",
            max_tokens,
            tokens: TokenCounts {
                thinking: 0,
                answer: 0,
                total: 0,
            },
            thinking_ms: 0.0,
            prefill_ms: 0.0,
            limit_hit_in_thinking: false,
            error: Some(ErrorInfo {
                message: message.into(),
                kind: "server_error",
            }),
        }
    }

    /// OpenAI `choices[0].finish_reason` for this outcome (`stop` / `length`).
    pub fn openai_finish_reason(&self) -> &'static str {
        if self.finish_reason == "length" {
            "length"
        } else {
            "stop"
        }
    }
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// The chat template's generation prompt ends inside an open `<think>`, so every
    /// reply starts in thinking (Nemotron). Derived from an actual template render.
    pub template_opens_think: bool,
}
