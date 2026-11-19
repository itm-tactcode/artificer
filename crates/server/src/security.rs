//! Request gates in front of every HTTP route: `Host` / `Origin` allowlist (403), optional
//! bearer API key (401) and a JSON `Content-Type` on every POST (415).
//!
//! The `Host` check stops DNS rebinding: a web page whose name resolves to 127.0.0.1 still
//! sends its own name in `Host`. The `Origin` check stops a page on another origin from
//! posting cross-site. The JSON `Content-Type` rule makes every browser POST a CORS
//! preflight. The `CorsLayer` sits outside these gates and only answers that preflight
//! (when `permissive_cors` is set); a foreign `Origin` on the real request still gets 403
//! unless `--allowed-origins` is set, so a page cannot fire a "simple" `text/plain`
//! request at `/mcp` or `/v1/chat/cancel`.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::config::{is_loopback_host, ApiKey, ServerConfig};
use crate::error::err_response;

/// Host names that always pass (with the bound port, or with no port).
const LOOPBACK_NAMES: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];

/// The gates for one server, built once the listener's real port is known.
#[derive(Debug)]
pub(crate) struct Security {
    /// The port the listener is actually bound to (not `cfg.port`, which may be 0).
    port: u16,
    any_host: bool,
    /// A non-loopback `--host`, normalised (lowercase, IPv6 in brackets).
    configured_host: Option<String>,
    /// `--allowed-hosts` entries: host name and an optional fixed port.
    extra_hosts: Vec<(String, Option<u16>)>,
    any_origin: bool,
    extra_origins: Vec<String>,
    api_key: Option<ApiKey>,
}

impl Security {
    pub(crate) fn new(cfg: &ServerConfig, port: u16) -> Result<Self, String> {
        let mut any_host = false;
        let mut extra_hosts = Vec::new();
        for entry in cfg.allowed_hosts.iter().map(|e| e.trim()) {
            if entry.is_empty() {
                continue;
            }
            if entry == "*" {
                any_host = true;
                continue;
            }
            let parsed = parse_authority(entry)
                .ok_or_else(|| format!("invalid --allowed-hosts entry `{entry}`"))?;
            extra_hosts.push(parsed);
        }
        let mut any_origin = false;
        let mut extra_origins = Vec::new();
        for entry in cfg.allowed_origins.iter().map(|e| e.trim()) {
            if entry.is_empty() {
                continue;
            }
            if entry == "*" {
                any_origin = true;
                continue;
            }
            extra_origins.push(normalize_origin(entry));
        }
        Ok(Self {
            port,
            any_host,
            configured_host: configured_host(&cfg.host),
            extra_hosts,
            any_origin,
            extra_origins,
            api_key: cfg.api_key.clone(),
        })
    }

    fn host_allowed(&self, value: Option<&str>) -> bool {
        if self.any_host {
            return true;
        }
        let Some((host, port)) = value.and_then(parse_authority) else {
            return false;
        };
        let bound = self.port;
        if LOOPBACK_NAMES.contains(&host.as_str()) {
            // A port-less loopback Host is what a raw local client sends; with a port it
            // must be ours.
            return port.is_none_or(|p| p == bound);
        }
        if self.configured_host.as_deref() == Some(host.as_str()) {
            // A non-standard loopback bind (e.g. `--host 127.0.0.2`) is accepted with no
            // port, or the bound port, like the usual loopback names. A non-loopback
            // `--host` still requires the bound port (or no port only on port 80).
            if is_loopback_host(&host) {
                return port.is_none_or(|p| p == bound);
            }
            return match port {
                None => bound == 80,
                Some(p) => p == bound,
            };
        }
        self.extra_hosts.iter().any(|(h, fixed)| {
            *h == host
                && match fixed {
                    Some(fixed) => port == Some(*fixed),
                    None => port.is_none_or(|p| p == bound),
                }
        })
    }

    fn origin_allowed(&self, origin: &str) -> bool {
        if self.any_origin {
            return true;
        }
        let o = normalize_origin(origin);
        let bound = self.port;
        let loopback = LOOPBACK_NAMES.iter().any(|name| {
            o == format!("http://{name}:{bound}") || (bound == 80 && o == format!("http://{name}"))
        });
        loopback || self.extra_origins.contains(&o)
    }

    fn key_ok(&self, req: &Request) -> bool {
        let Some(key) = &self.api_key else {
            return true;
        };
        let Some(value) = req.headers().get(header::AUTHORIZATION) else {
            return false;
        };
        let Ok(value) = value.to_str() else {
            return false;
        };
        let Some((scheme, token)) = value.trim().split_once(' ') else {
            return false;
        };
        scheme.eq_ignore_ascii_case("bearer") && ct_eq(token.trim().as_bytes(), key.as_bytes())
    }
}

/// Lowercased `host` and optional port of a `Host` value (`name`, `name:port`, `[v6]`,
/// `[v6]:port`). `None` for anything else (an unbracketed IPv6, a bad port, userinfo, …).
fn parse_authority(value: &str) -> Option<(String, Option<u16>)> {
    let v = value.trim().to_ascii_lowercase();
    if v.is_empty() {
        return None;
    }
    let (host, rest) = if v.starts_with('[') {
        let end = v.find(']')?;
        (v[..=end].to_owned(), &v[end + 1..])
    } else {
        match v.find(':') {
            Some(i) => (v[..i].to_owned(), &v[i..]),
            None => (v.clone(), ""),
        }
    };
    let valid_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_');
    let host_ok = if let Some(inner) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        inner.parse::<std::net::Ipv6Addr>().is_ok()
    } else {
        !host.is_empty() && host.chars().all(valid_char)
    };
    if !host_ok {
        return None;
    }
    let port = if rest.is_empty() {
        None
    } else {
        let digits = rest.strip_prefix(':')?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some(digits.parse::<u16>().ok()?)
    };
    Some((host, port))
}

/// `--host` as a `Host` value when it isn't already one of the always-accepted
/// loopback names (`127.0.0.1`, `localhost`, `[::1]`).
fn configured_host(host: &str) -> Option<String> {
    let h = host.trim();
    if h.is_empty() {
        return None;
    }
    let bare = h
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .unwrap_or(h);
    let normalized = match bare.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V6(v6)) => format!("[{v6}]"),
        Ok(std::net::IpAddr::V4(v4)) => v4.to_string(),
        Err(_) => h.to_ascii_lowercase(),
    };
    if LOOPBACK_NAMES.contains(&normalized.as_str()) {
        return None;
    }
    Some(normalized)
}

fn normalize_origin(origin: &str) -> String {
    origin.trim().to_ascii_lowercase()
}

/// Equality that takes the same time for every pair of inputs of the same lengths.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len().max(b.len());
    let mut diff = (a.len() ^ b.len()) as u64;
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= u64::from(x ^ y);
    }
    std::hint::black_box(diff) == 0
}

/// `application/json`, or any `type/subtype+json`; parameters (`; charset=utf-8`) are fine.
fn is_json_content_type(value: &str) -> bool {
    let media = value
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let Some((ty, sub)) = media.split_once('/') else {
        return false;
    };
    let token = |s: &str| {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
    };
    token(ty)
        && token(sub)
        && (media == "application/json" || (sub.len() > 5 && sub.ends_with("+json")))
}

/// 403 unless `Host` and any `Origin` are allowed.
pub(crate) async fn host_origin_gate(
    State(sec): State<Arc<Security>>,
    req: Request,
    next: Next,
) -> Response {
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok());
    if !sec.host_allowed(host) {
        tracing::warn!(path = %req.uri().path(), "rejected request: Host header not allowed");
        return err_response(StatusCode::FORBIDDEN, "request Host is not allowed");
    }
    if let Some(origin) = req.headers().get(header::ORIGIN) {
        let ok = origin.to_str().is_ok_and(|o| sec.origin_allowed(o));
        if !ok {
            tracing::warn!(path = %req.uri().path(), "rejected request: Origin not allowed");
            return err_response(StatusCode::FORBIDDEN, "request Origin is not allowed");
        }
    }
    next.run(req).await
}

/// 401 unless the API key matches (when one is configured). `GET /health` is always open.
pub(crate) async fn api_key_gate(
    State(sec): State<Arc<Security>>,
    req: Request,
    next: Next,
) -> Response {
    let open = req.method() == Method::GET && req.uri().path() == "/health";
    if open || sec.key_ok(&req) {
        return next.run(req).await;
    }
    tracing::warn!(path = %req.uri().path(), "rejected request: missing or invalid API key");
    let mut resp = err_response(StatusCode::UNAUTHORIZED, "missing or invalid API key");
    resp.headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    resp.into_response()
}

/// 415 unless every POST, including a bodyless cancel, says it is JSON.
pub(crate) async fn json_content_type_gate(req: Request, next: Next) -> Response {
    if req.method() == Method::POST {
        let ok = req
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(is_json_content_type);
        if !ok {
            return err_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
            );
        }
    }
    next.run(req).await
}

/// The three gates around `router`, outermost first: Host/Origin, API key, Content-Type.
pub(crate) fn apply<S>(router: axum::Router<S>, sec: Arc<Security>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(axum::middleware::from_fn(json_content_type_gate))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&sec),
            api_key_gate,
        ))
        .layer(axum::middleware::from_fn_with_state(sec, host_origin_gate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::time::Duration;

    use axum::routing::{get, post};
    use axum::Router;

    fn cfg(host: &str) -> ServerConfig {
        ServerConfig {
            host: host.into(),
            ..ServerConfig::default()
        }
    }

    fn sec(cfg: &ServerConfig, port: u16) -> Security {
        Security::new(cfg, port).unwrap()
    }

    #[test]
    fn authority_parsing() {
        assert_eq!(
            parse_authority("127.0.0.1:8080"),
            Some(("127.0.0.1".into(), Some(8080)))
        );
        assert_eq!(
            parse_authority("LocalHost"),
            Some(("localhost".into(), None))
        );
        assert_eq!(parse_authority("[::1]:9"), Some(("[::1]".into(), Some(9))));
        assert_eq!(parse_authority("[::1]"), Some(("[::1]".into(), None)));
        for bad in [
            "", "::1", "[::1", "a:b", "a:", "a:99999", "u@h:1", "h/x", "[zz]:1", "a:+1",
        ] {
            assert_eq!(parse_authority(bad), None, "{bad}");
        }
    }

    #[test]
    fn host_rules_loopback_names_and_bound_port() {
        let s = sec(&cfg("127.0.0.1"), 41234);
        for ok in [
            "127.0.0.1:41234",
            "localhost:41234",
            "LOCALHOST:41234",
            "[::1]:41234",
            "127.0.0.1",
            "localhost",
            "[::1]",
        ] {
            assert!(s.host_allowed(Some(ok)), "{ok}");
        }
        for bad in [
            "evil.example",
            "evil.example:41234",
            "127.0.0.1:41235",
            "localhost:80",
            "127.0.0.2:41234",
            "0.0.0.0:41234",
            "",
        ] {
            assert!(!s.host_allowed(Some(bad)), "{bad}");
        }
        assert!(!s.host_allowed(None), "missing Host");
    }

    #[test]
    fn host_rules_configured_non_loopback_host() {
        let s = sec(&cfg("192.168.1.20"), 8080);
        assert!(s.host_allowed(Some("192.168.1.20:8080")));
        assert!(
            !s.host_allowed(Some("192.168.1.20")),
            "port-less only on :80"
        );
        assert!(!s.host_allowed(Some("192.168.1.20:8081")));
        assert!(sec(&cfg("192.168.1.20"), 80).host_allowed(Some("192.168.1.20")));
        let s = sec(&cfg("Box.Lan"), 8080);
        assert!(s.host_allowed(Some("box.lan:8080")));
        let s = sec(&cfg("fe80::1"), 8080);
        assert!(s.host_allowed(Some("[fe80::1]:8080")));
        for (wild, host) in [
            ("0.0.0.0", "0.0.0.0:8080"),
            ("::", "[::]:8080"),
            ("[::]", "[::]:8080"),
        ] {
            let s = sec(&cfg(wild), 8080);
            assert!(s.host_allowed(Some(host)), "{wild}");
            assert!(!s.host_allowed(Some("evil.example:8080")), "{wild}");
            assert!(s.host_allowed(Some("127.0.0.1:8080")), "{wild}");
        }
    }

    #[test]
    fn host_rules_configured_loopback_alias() {
        let s = sec(&cfg("127.0.0.2"), 41234);
        assert!(s.host_allowed(Some("127.0.0.2:41234")));
        assert!(
            s.host_allowed(Some("127.0.0.2")),
            "port-less like other loopback names"
        );
        assert!(!s.host_allowed(Some("127.0.0.2:41235")));
        assert!(!s.host_allowed(Some("evil.example:41234")));
        assert!(!s.host_allowed(Some("0.0.0.0:41234")));
        // Standard loopback names still pass; the alias does not replace them.
        assert!(s.host_allowed(Some("127.0.0.1:41234")));
        assert!(s.host_allowed(Some("localhost:41234")));
    }

    #[test]
    fn host_opt_outs() {
        let mut c = cfg("0.0.0.0");
        c.allowed_hosts = vec!["box.lan".into(), "proxy.example:443".into()];
        let s = sec(&c, 8080);
        assert!(s.host_allowed(Some("box.lan")));
        assert!(s.host_allowed(Some("box.lan:8080")));
        assert!(!s.host_allowed(Some("box.lan:9")));
        assert!(s.host_allowed(Some("proxy.example:443")));
        assert!(!s.host_allowed(Some("proxy.example")));
        assert!(!s.host_allowed(Some("evil.example:8080")));
        c.allowed_hosts = vec!["*".into()];
        let s = sec(&c, 8080);
        assert!(s.host_allowed(Some("evil.example")));
        assert!(s.host_allowed(None));
        c.allowed_hosts = vec!["bad host".into()];
        assert!(Security::new(&c, 8080).is_err());
    }

    #[test]
    fn origin_rules() {
        let s = sec(&cfg("127.0.0.1"), 5000);
        for ok in [
            "http://127.0.0.1:5000",
            "http://localhost:5000",
            "http://[::1]:5000",
            "HTTP://LOCALHOST:5000",
        ] {
            assert!(s.origin_allowed(ok), "{ok}");
        }
        for bad in [
            "http://evil.example",
            "http://127.0.0.1:5001",
            "https://127.0.0.1:5000",
            "http://localhost",
            "null",
            "tauri://localhost",
            "http://tauri.localhost",
        ] {
            assert!(!s.origin_allowed(bad), "{bad}");
        }
        let mut c = cfg("127.0.0.1");
        c.allowed_origins = vec!["https://app.example".into()];
        let s = sec(&c, 5000);
        assert!(s.origin_allowed("https://app.example"));
        assert!(!s.origin_allowed("https://other.example"));
        c.allowed_origins = vec!["*".into()];
        assert!(sec(&c, 5000).origin_allowed("http://evil.example"));
    }

    #[test]
    fn constant_time_eq() {
        assert!(ct_eq(b"secret", b"secret"));
        assert!(!ct_eq(b"secret", b"secreT"));
        assert!(!ct_eq(b"secret", b"secret2"));
        assert!(!ct_eq(b"", b"x"));
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn json_content_types() {
        for ok in [
            "application/json",
            "application/json; charset=utf-8",
            "Application/JSON;charset=UTF-8",
            "application/merge-patch+json",
            "application/vnd.api+json; v=1",
        ] {
            assert!(is_json_content_type(ok), "{ok}");
        }
        for bad in [
            "text/plain",
            "",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "json",
            "application/+json",
            "text/json5",
        ] {
            assert!(!is_json_content_type(bad), "{bad}");
        }
    }

    #[test]
    fn api_key_debug_is_redacted() {
        let key = ApiKey::new("hunter2-very-secret").unwrap();
        let mut c = cfg("127.0.0.1");
        c.api_key = Some(key);
        assert!(!format!("{c:?}").contains("hunter2"));
        assert!(!format!("{:?}", sec(&c, 1)).contains("hunter2"));
        assert!(ApiKey::new("  ").is_none());
        assert!("".parse::<ApiKey>().is_err());
    }

    #[test]
    fn api_key_rejects_surrounding_whitespace() {
        for key in [" abc", "abc ", " abc ", "\tabc", "abc\n"] {
            assert!(ApiKey::new(key).is_none(), "new rejects {key:?}");
            let err = key.parse::<ApiKey>().unwrap_err();
            assert!(
                err.contains("leading or trailing whitespace"),
                "{key:?}: {err}"
            );
        }
        // All-whitespace is still the empty-key error, not a stored key.
        assert!(ApiKey::new("  ").is_none());
        let err = "  ".parse::<ApiKey>().unwrap_err();
        assert!(err.contains("must not be empty"), "{err}");
        assert!("abc".parse::<ApiKey>().is_ok());
    }

    // ── The gates on a live socket, around stand-in routes (no model needed) ──

    struct Live {
        port: u16,
        stop: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Drop for Live {
        fn drop(&mut self) {
            if let Some(tx) = self.stop.take() {
                let _ = tx.send(());
            }
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    fn live(cfg: ServerConfig) -> Live {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let sec = Arc::new(Security::new(&cfg, port).unwrap());
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let app = apply(
                    Router::new()
                        .route("/health", get(|| async { "ok" }))
                        .route("/v1/models", get(|| async { "models" }))
                        .route("/mcp", post(|body: String| async move { body }))
                        .route("/v1/chat/cancel", post(|| async { "cancelled" })),
                    sec,
                );
                let l = tokio::net::TcpListener::from_std(listener).unwrap();
                axum::serve(l, app)
                    .with_graceful_shutdown(async {
                        let _ = rx.await;
                    })
                    .await
                    .unwrap();
            });
        });
        Live {
            port,
            stop: Some(tx),
            thread: Some(thread),
        }
    }

    /// Send raw request `head` (lines without the final blank line) plus `body`; the status.
    fn status(port: u16, head: &str, body: &str) -> u16 {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut req = format!("{head}\r\nConnection: close\r\n");
        if !body.is_empty() || head.starts_with("POST") {
            req.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        req.push_str("\r\n");
        req.push_str(body);
        s.write_all(req.as_bytes()).unwrap();
        let mut reply = String::new();
        let _ = s.read_to_string(&mut reply);
        reply
            .split(' ')
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or_else(|| panic!("no status in {reply:?}"))
    }

    const JSON: &str = "Content-Type: application/json";

    #[test]
    fn live_host_and_origin_gate() {
        let srv = live(cfg("127.0.0.1"));
        let p = srv.port;
        let get = |host_line: &str| status(p, &format!("GET /health HTTP/1.1\r\n{host_line}"), "");
        assert_eq!(get(&format!("Host: 127.0.0.1:{p}")), 200);
        assert_eq!(get(&format!("Host: localhost:{p}")), 200);
        assert_eq!(get(&format!("Host: [::1]:{p}")), 200);
        assert_eq!(get("Host: 127.0.0.1"), 200, "port-less loopback Host");
        assert_eq!(get("Host: evil.example"), 403, "port-less foreign Host");
        assert_eq!(
            get(&format!("Host: evil.example:{p}")),
            403,
            "rebinding Host"
        );
        assert_eq!(
            get(&format!("Host: 127.0.0.1:{}", p.wrapping_add(1))),
            403,
            "wrong port"
        );
        assert_eq!(status(p, "GET /health HTTP/1.0", ""), 403, "missing Host");
        // No Origin passes (curl, the Rust client); a foreign one does not.
        let h = format!("Host: 127.0.0.1:{p}");
        assert_eq!(get(&format!("{h}\r\nOrigin: http://evil.example")), 403);
        assert_eq!(get(&format!("{h}\r\nOrigin: http://localhost:{p}")), 200);
        let post = |extra: &str| {
            status(
                p,
                &format!("POST /mcp HTTP/1.1\r\n{h}\r\n{JSON}{extra}"),
                "{}",
            )
        };
        assert_eq!(post("\r\nOrigin: http://evil.example"), 403);
        assert_eq!(post(""), 200);
    }

    #[test]
    fn live_configured_loopback_alias_is_an_allowed_host() {
        let srv = live(cfg("127.0.0.2"));
        let p = srv.port;
        let get = |host_line: &str| status(p, &format!("GET /health HTTP/1.1\r\n{host_line}"), "");
        assert_eq!(get(&format!("Host: 127.0.0.2:{p}")), 200);
        assert_eq!(get("Host: 127.0.0.2"), 200, "port-less configured loopback");
        assert_eq!(get(&format!("Host: evil.example:{p}")), 403);
        assert_eq!(get("Host: evil.example"), 403);
    }

    #[test]
    fn live_opt_outs_pass_everything() {
        let mut c = cfg("127.0.0.1");
        c.allowed_hosts = vec!["*".into()];
        c.allowed_origins = vec!["*".into()];
        let srv = live(c);
        let p = srv.port;
        let r = status(
            p,
            "GET /health HTTP/1.1\r\nHost: evil.example\r\nOrigin: http://evil.example",
            "",
        );
        assert_eq!(r, 200);
        assert_eq!(
            status(p, "GET /health HTTP/1.0", ""),
            200,
            "missing Host with *"
        );
    }

    #[test]
    fn live_json_content_type_gate() {
        let srv = live(cfg("127.0.0.1"));
        let p = srv.port;
        let h = format!("Host: 127.0.0.1:{p}");
        let post = |path: &str, ct: &str, body: &str| {
            let ct = if ct.is_empty() {
                String::new()
            } else {
                format!("\r\nContent-Type: {ct}")
            };
            status(p, &format!("POST {path} HTTP/1.1\r\n{h}{ct}"), body)
        };
        assert_eq!(post("/mcp", "text/plain", "{}"), 415);
        assert_eq!(post("/mcp", "application/x-www-form-urlencoded", "{}"), 415);
        assert_eq!(post("/mcp", "", "{}"), 415);
        assert_eq!(post("/v1/chat/cancel", "", ""), 415, "bodyless cancel");
        assert_eq!(post("/v1/chat/cancel", "text/plain", ""), 415);
        assert_eq!(post("/mcp", "application/json; charset=utf-8", "{}"), 200);
        assert_eq!(post("/v1/chat/cancel", "application/json", "{}"), 200);
        // GETs need no Content-Type.
        assert_eq!(
            status(p, &format!("GET /v1/models HTTP/1.1\r\n{h}"), ""),
            200
        );
    }

    #[test]
    fn live_api_key_gate() {
        let mut c = cfg("127.0.0.1");
        c.api_key = ApiKey::new("s3cret-key");
        let srv = live(c);
        let p = srv.port;
        let h = format!("Host: 127.0.0.1:{p}");
        let models = |auth: &str| status(p, &format!("GET /v1/models HTTP/1.1\r\n{h}{auth}"), "");
        assert_eq!(models(""), 401, "missing key");
        assert_eq!(models("\r\nAuthorization: Bearer wrong"), 401);
        assert_eq!(models("\r\nAuthorization: Bearer s3cret-key2"), 401);
        assert_eq!(models("\r\nAuthorization: Basic s3cret-key"), 401);
        assert_eq!(models("\r\nAuthorization: Bearer s3cret-key"), 200);
        assert_eq!(models("\r\nAuthorization: bearer s3cret-key"), 200);
        // /health stays open without a key (the desktop's reuse probe).
        assert_eq!(status(p, &format!("GET /health HTTP/1.1\r\n{h}"), ""), 200);
        // POSTs need it too; 401 comes before the Content-Type check.
        assert_eq!(
            status(p, &format!("POST /mcp HTTP/1.1\r\n{h}\r\n{JSON}"), "{}"),
            401
        );
        assert_eq!(
            status(
                p,
                &format!("POST /mcp HTTP/1.1\r\n{h}\r\n{JSON}\r\nAuthorization: Bearer s3cret-key"),
                "{}"
            ),
            200
        );
        // The Host gate still runs first.
        assert_eq!(
            status(p, "GET /health HTTP/1.1\r\nHost: evil.example", ""),
            403
        );
    }

    #[test]
    fn live_401_body_does_not_echo_the_key() {
        let mut c = cfg("127.0.0.1");
        c.api_key = ApiKey::new("s3cret-key");
        let srv = live(c);
        let p = srv.port;
        let mut s = std::net::TcpStream::connect(("127.0.0.1", p)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let req = format!(
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nAuthorization: Bearer s3cret-kex\r\nConnection: close\r\n\r\n"
        );
        s.write_all(req.as_bytes()).unwrap();
        let mut reply = String::new();
        let _ = s.read_to_string(&mut reply);
        assert!(reply.starts_with("HTTP/1.1 401"), "{reply}");
        assert!(
            reply
                .to_ascii_lowercase()
                .contains("www-authenticate: bearer"),
            "{reply}"
        );
        assert!(!reply.contains("s3cret"), "{reply}");
    }
}
