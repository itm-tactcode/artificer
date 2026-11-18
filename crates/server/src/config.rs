//! HTTP server configuration.

use std::path::PathBuf;

use artificer_runtime::EngineLoadOptions;

/// Limits applied to untrusted client input (DoS mitigation).
#[derive(Debug, Clone)]
pub struct RequestLimits {
    /// Absolute max completion tokens per request.
    pub max_tokens: usize,
    /// Max JSON body size in bytes.
    pub max_body_bytes: usize,
    /// Max chat messages per request.
    pub max_messages: usize,
    /// Max total characters across all message contents.
    pub max_prompt_chars: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            max_tokens: artificer_core::MAX_TOKENS_CEILING as usize,
            max_body_bytes: 1_048_576, // 1 MiB
            max_messages: 256,
            max_prompt_chars: 200_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: PathBuf,
    pub model_id: String,
    /// When true, enable permissive CORS (browser demos only).
    pub permissive_cors: bool,
    pub limits: RequestLimits,
    /// Engine load options (context budget, KV f16, …).
    pub load_opts: Option<EngineLoadOptions>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            model_path: PathBuf::from("model.gguf"),
            model_id: "local-model".into(),
            permissive_cors: false,
            limits: RequestLimits::default(),
            load_opts: None,
        }
    }
}

impl ServerConfig {
    /// True if bind address is loopback-only (safer default exposure).
    pub fn is_loopback_bind(&self) -> bool {
        matches!(
            self.host.as_str(),
            "127.0.0.1" | "localhost" | "::1" | "[::1]"
        )
    }
}
