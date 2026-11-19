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
    /// Extra `Host` header values accepted on top of the loopback names and a non-loopback
    /// `host` (`--allowed-hosts`). An entry `*` turns the Host check off.
    pub allowed_hosts: Vec<String>,
    /// Extra `Origin` header values accepted on top of `http://{127.0.0.1|localhost|[::1]}:<port>`
    /// (`--allowed-origins`). An entry `*` turns the Origin check off.
    pub allowed_origins: Vec<String>,
    /// When set, every route except `GET /health` needs `Authorization: Bearer <key>`.
    pub api_key: Option<ApiKey>,
    /// Allow a non-loopback `host` with no `api_key` (`--allow-insecure-bind`).
    pub allow_insecure_bind: bool,
}

/// A bearer token for the HTTP API. `Debug` never prints it.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// `None` for an empty or all-whitespace key, and for a key with leading or
    /// trailing whitespace (rejected with a distinct error by [`std::str::FromStr`]).
    pub fn new(key: impl Into<String>) -> Option<Self> {
        parse_api_key(&key.into()).ok()
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl std::str::FromStr for ApiKey {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        parse_api_key(s)
    }
}

fn parse_api_key(s: &str) -> std::result::Result<ApiKey, String> {
    if s.trim().is_empty() {
        Err("the API key must not be empty".to_owned())
    } else if s != s.trim() {
        Err("the API key must not have leading or trailing whitespace".to_owned())
    } else {
        Ok(ApiKey(s.to_owned()))
    }
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
            allowed_hosts: Vec::new(),
            allowed_origins: Vec::new(),
            api_key: None,
            allow_insecure_bind: false,
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

    /// True if `host` is a loopback address: an IP (bracketed IPv6 too) that
    /// [`std::net::IpAddr::is_loopback`] accepts (`127.0.0.0/8`, `::1`), or exactly `localhost`.
    /// Any other hostname, and the unspecified `0.0.0.0` / `::`, is non-loopback.
    pub fn is_loopback_host(&self) -> bool {
        is_loopback_host(&self.host)
    }

    /// Refuse a non-loopback bind with no API key unless `allow_insecure_bind`.
    pub fn check_bind_security(&self) -> std::result::Result<(), String> {
        if self.is_loopback_host() || self.api_key.is_some() || self.allow_insecure_bind {
            return Ok(());
        }
        Err(format!(
            "refusing to listen on non-loopback address {} without an API key: set --api-key \
             (or ARTIFICER_API_KEY), or pass --allow-insecure-bind to serve with no authentication",
            self.host
        ))
    }
}

/// See [`ServerConfig::is_loopback_host`].
pub fn is_loopback_host(host: &str) -> bool {
    let h = host.trim();
    if h == "localhost" {
        return true;
    }
    let bare = h
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .unwrap_or(h);
    bare.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}
