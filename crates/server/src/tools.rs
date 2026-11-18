//! OpenAI tool / function-calling: prompt injection and model-output parsing.
//!
//! Model-facing format is Hermes-style XML (widely used by local instruct models):
//!
//! ```text
//! <tool_call>
//! {"name": "fn", "arguments": {...}}
//! </tool_call>
//! ```
//!
//! Parsed results are converted to OpenAI `tool_calls` wire shapes.

use artificer_core::ChatMessage as ChatTurn;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::api_types::{FunctionCall, FunctionDef, Tool, ToolCall, ToolChoice, ToolChoiceNamed};

/// Max tools accepted per request (DoS guard).
pub const MAX_TOOLS: usize = 32;
/// Max characters for a single tool name.
pub const MAX_TOOL_NAME_LEN: usize = 64;
/// Max characters for tool description text.
pub const MAX_TOOL_DESC_LEN: usize = 4_096;
/// Max serialized size of a parameters JSON schema.
pub const MAX_TOOL_PARAMS_CHARS: usize = 16_384;

/// Effective tool-calling mode after resolving `tool_choice` + available tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolMode {
    /// Do not advertise tools; model should answer normally.
    None,
    /// Model may call tools or answer in natural language.
    Auto,
    /// Model must produce at least one tool call.
    Required,
    /// Model must call this specific function name.
    Named(String),
}

/// Resolve tools list + tool_choice (including legacy `functions` / `function_call`).
pub fn resolve_tools(
    tools: &Option<Vec<Tool>>,
    tool_choice: &Option<ToolChoice>,
    functions: &Option<Vec<FunctionDef>>,
    function_call: &Option<Value>,
) -> Result<(Vec<Tool>, ToolMode), String> {
    let mut list = tools.clone().unwrap_or_default();
    if list.is_empty() {
        if let Some(fns) = functions {
            list = fns
                .iter()
                .map(|f| Tool {
                    type_: "function".into(),
                    function: f.clone(),
                })
                .collect();
        }
    }

    validate_tools(&list)?;

    // Legacy function_call → tool_choice mapping when tool_choice absent.
    let choice = if tool_choice.is_none() {
        function_call
            .as_ref()
            .and_then(legacy_function_call_to_choice)
    } else {
        tool_choice.clone()
    };

    let mode = resolve_mode(&list, choice.as_ref())?;
    Ok((list, mode))
}

fn legacy_function_call_to_choice(v: &Value) -> Option<ToolChoice> {
    match v {
        Value::String(s) => Some(ToolChoice::Mode(s.clone())),
        Value::Object(map) => {
            let name = map.get("name")?.as_str()?.to_string();
            Some(ToolChoice::Named(ToolChoiceNamed {
                type_: "function".into(),
                function: crate::api_types::ToolChoiceFunction { name },
            }))
        }
        _ => None,
    }
}

fn resolve_mode(tools: &[Tool], choice: Option<&ToolChoice>) -> Result<ToolMode, String> {
    if tools.is_empty() {
        return Ok(ToolMode::None);
    }
    match choice {
        None => Ok(ToolMode::Auto),
        Some(ToolChoice::Mode(s)) => match s.to_ascii_lowercase().as_str() {
            "none" => Ok(ToolMode::None),
            "auto" => Ok(ToolMode::Auto),
            "required" | "any" => Ok(ToolMode::Required),
            other => Err(format!("unsupported tool_choice mode: {other}")),
        },
        Some(ToolChoice::Named(n)) => {
            if n.type_ != "function" {
                return Err(format!("unsupported tool_choice type: {}", n.type_));
            }
            if !tools.iter().any(|t| t.function.name == n.function.name) {
                return Err(format!(
                    "tool_choice names unknown function `{}`",
                    n.function.name
                ));
            }
            Ok(ToolMode::Named(n.function.name.clone()))
        }
    }
}

pub fn validate_tools(tools: &[Tool]) -> Result<(), String> {
    if tools.len() > MAX_TOOLS {
        return Err(format!("too many tools (max {MAX_TOOLS})"));
    }
    for t in tools {
        if t.type_ != "function" {
            return Err(format!("unsupported tool type: {}", t.type_));
        }
        let name = t.function.name.trim();
        if name.is_empty() {
            return Err("tool function.name must not be empty".into());
        }
        if name.len() > MAX_TOOL_NAME_LEN {
            return Err(format!(
                "tool name too long (max {MAX_TOOL_NAME_LEN} chars)"
            ));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "tool name `{name}` must be alphanumeric/underscore/hyphen"
            ));
        }
        if let Some(desc) = &t.function.description {
            if desc.len() > MAX_TOOL_DESC_LEN {
                return Err(format!(
                    "tool description too long for `{name}` (max {MAX_TOOL_DESC_LEN})"
                ));
            }
        }
        if let Some(params) = &t.function.parameters {
            let s = params.to_string();
            if s.len() > MAX_TOOL_PARAMS_CHARS {
                return Err(format!(
                    "tool parameters schema too large for `{name}` (max {MAX_TOOL_PARAMS_CHARS} chars)"
                ));
            }
        }
    }
    Ok(())
}

/// Tool-use instructions as a system message (`None` when tools are off).
///
/// Our fixed prose (including the `<tool_call>` example) is trusted markup, so its tags
/// tokenize as the model's special ids; the client-supplied catalog and tool name are
/// plain text and can never forge control tokens.
pub fn tools_system_message(tools: &[Tool], mode: &ToolMode) -> Option<ChatTurn> {
    if matches!(mode, ToolMode::None) || tools.is_empty() {
        return None;
    }

    let catalog: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.function.name,
                "description": t.function.description.as_deref().unwrap_or(""),
                "parameters": t.function.parameters.clone().unwrap_or_else(|| json!({"type":"object","properties":{}})),
            })
        })
        .collect();

    let catalog_str = serde_json::to_string_pretty(&catalog).unwrap_or_else(|_| "[]".into());

    let mut m = ChatTurn::new("system", "");
    m.push_markup(
        "You are a helpful assistant with access to tools (functions). \
When you need to use a tool, respond with one or more tool calls using this exact format — \
and nothing else outside the tags:\n\n\
<tool_call>\n\
{\"name\": \"function_name\", \"arguments\": {\"arg\": \"value\"}}\n\
</tool_call>\n\n\
You may emit multiple <tool_call> blocks if several tools are needed. \
If you can answer without tools, respond with normal assistant text only.\n\n\
Available tools (JSON):\n",
    );
    m.push_text(&catalog_str);

    match mode {
        ToolMode::Required => {
            m.push_markup(
                "\n\nYou MUST call at least one tool. Do not answer with plain text only.",
            );
        }
        ToolMode::Named(name) => {
            m.push_markup("\n\nYou MUST call the tool named `");
            m.push_text(name);
            m.push_markup("` (and only that tool).");
        }
        ToolMode::Auto | ToolMode::None => {}
    }

    Some(m)
}

/// An assistant history turn that included tool_calls, as prompt text. The
/// `<tool_call>` tags are trusted markup; the content and call payloads are plain text.
pub fn assistant_tool_calls_message(content: Option<&str>, tool_calls: &[ToolCall]) -> ChatTurn {
    let mut m = ChatTurn::new("assistant", "");
    if let Some(c) = content {
        let t = c.trim();
        if !t.is_empty() {
            m.push_text(t);
            m.push_text("\n");
        }
    }
    for tc in tool_calls {
        let args = parse_arguments_value(&tc.function.arguments);
        let payload = json!({
            "name": tc.function.name,
            "arguments": args,
        });
        m.push_markup("<tool_call>\n");
        m.push_text(&payload.to_string());
        m.push_markup("\n</tool_call>\n");
    }
    m
}

/// A `role: tool` result as a user turn. The `<tool_response>` tags are trusted
/// markup; the header (name / id) and the tool output are plain text, so a tool result
/// containing e.g. `<|im_end|>` cannot inject control tokens.
pub fn tool_result_message(
    tool_call_id: Option<&str>,
    name: Option<&str>,
    content: &str,
) -> ChatTurn {
    let mut header = String::from("Tool result");
    if let Some(n) = name {
        if !n.is_empty() {
            header.push_str(" from ");
            header.push_str(n);
        }
    }
    if let Some(id) = tool_call_id {
        if !id.is_empty() {
            header.push_str(" (");
            header.push_str(id);
            header.push(')');
        }
    }
    let mut m = ChatTurn::new("user", "");
    m.push_markup("<tool_response>\n");
    m.push_text(&header);
    m.push_text(":\n");
    m.push_text(content);
    m.push_markup("\n</tool_response>");
    m
}

/// Parse model completion text into OpenAI-style tool calls (if any).
pub fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();

    // Primary: <tool_call>...</tool_call>
    let lower = text.to_ascii_lowercase();
    let mut search_from = 0;
    while let Some(rel) = lower[search_from..].find("<tool_call>") {
        let start = search_from + rel + "<tool_call>".len();
        let end_rel = lower[start..].find("</tool_call>")?;
        let body = text[start..start + end_rel].trim();
        if let Some(tc) = tool_call_from_json_body(body) {
            calls.push(tc);
        }
        search_from = start + end_rel + "</tool_call>".len();
    }

    if !calls.is_empty() {
        return Some(calls);
    }

    // Fallback: fenced ```json ... ``` containing name+arguments
    if let Some(body) = extract_json_fence(text) {
        if let Some(tc) = tool_call_from_json_body(body) {
            return Some(vec![tc]);
        }
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(body) {
            let mut multi = Vec::new();
            for item in arr {
                if let Some(tc) = tool_call_from_value(&item) {
                    multi.push(tc);
                }
            }
            if !multi.is_empty() {
                return Some(multi);
            }
        }
    }

    // Fallback: whole text is a single JSON object/array
    let trimmed = text.trim();
    if trimmed.starts_with('{') {
        if let Some(tc) = tool_call_from_json_body(trimmed) {
            return Some(vec![tc]);
        }
    }
    if trimmed.starts_with('[') {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(trimmed) {
            let multi: Vec<_> = arr.iter().filter_map(tool_call_from_value).collect();
            if !multi.is_empty() {
                return Some(multi);
            }
        }
    }

    None
}

fn extract_json_fence(text: &str) -> Option<&str> {
    let lower = text.to_ascii_lowercase();
    let start_key = "```json";
    let idx = lower.find(start_key)?;
    let after = idx + start_key.len();
    let rest = &text[after..];
    let end = rest.find("```")?;
    Some(rest[..end].trim())
}

fn tool_call_from_json_body(body: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(body).ok()?;
    tool_call_from_value(&v)
}

fn tool_call_from_value(v: &Value) -> Option<ToolCall> {
    let obj = v.as_object()?;
    // Accept {name, arguments} or {function: {name, arguments}}
    let (name, args_val) = if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
        let name = func.get("name")?.as_str()?.to_string();
        let args = func.get("arguments").cloned().unwrap_or(json!({}));
        (name, args)
    } else {
        let name = obj.get("name")?.as_str()?.to_string();
        let args = obj
            .get("arguments")
            .or_else(|| obj.get("parameters"))
            .cloned()
            .unwrap_or(json!({}));
        (name, args)
    };

    if name.is_empty() {
        return None;
    }

    let arguments = match args_val {
        Value::String(s) => s,
        other => other.to_string(),
    };

    Some(ToolCall {
        id: new_tool_call_id(),
        type_: "function".into(),
        function: FunctionCall { name, arguments },
    })
}

fn parse_arguments_value(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

pub fn new_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

/// Whether tools should be active for this request (injected + parse for tool_calls).
pub fn tools_active(mode: &ToolMode) -> bool {
    !matches!(mode, ToolMode::None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_types::FunctionDef;

    fn sample_tool(name: &str) -> Tool {
        Tool {
            type_: "function".into(),
            function: FunctionDef {
                name: name.into(),
                description: Some("demo".into()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string" }
                    },
                    "required": ["q"]
                })),
            },
        }
    }

    #[test]
    fn parse_hermes_tool_call() {
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>"#;
        let calls = parse_tool_calls(text).expect("parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("NYC"));
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let text = r#"
<tool_call>
{"name": "a", "arguments": {}}
</tool_call>
some noise
<tool_call>
{"name": "b", "arguments": {"x": 1}}
</tool_call>
"#;
        let calls = parse_tool_calls(text).expect("parsed");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
    }

    #[test]
    fn parse_raw_json_object() {
        let text = r#"{"name":"search","arguments":{"q":"rust"}}"#;
        let calls = parse_tool_calls(text).expect("parsed");
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn no_tool_call_in_plain_text() {
        assert!(parse_tool_calls("The sky is blue.").is_none());
    }

    #[test]
    fn tools_instructions_for_auto() {
        let tools = vec![sample_tool("get_weather")];
        let instr = tools_system_message(&tools, &ToolMode::Auto)
            .unwrap()
            .content;
        assert!(instr.contains("get_weather"));
        assert!(instr.contains("<tool_call>"));
    }

    #[test]
    fn validate_rejects_bad_name() {
        let t = Tool {
            type_: "function".into(),
            function: FunctionDef {
                name: "bad name!".into(),
                description: None,
                parameters: None,
            },
        };
        assert!(validate_tools(&[t]).is_err());
    }

    #[test]
    fn resolve_named_choice() {
        let tools = vec![sample_tool("get_weather")];
        let choice = ToolChoice::Named(ToolChoiceNamed {
            type_: "function".into(),
            function: crate::api_types::ToolChoiceFunction {
                name: "get_weather".into(),
            },
        });
        let (list, mode) = resolve_tools(&Some(tools), &Some(choice), &None, &None).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(mode, ToolMode::Named("get_weather".into()));
    }
}
