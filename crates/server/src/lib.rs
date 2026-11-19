//! OpenAI-compatible HTTP server + MCP surface.
//!
//! Endpoints:
//! - `GET  /health`
//! - `GET  /v1/models`
//! - `POST /v1/models/load` (in-process GGUF swap: `{"path":"<absolute>"}`)
//! - `POST /v1/chat/completions` (non-streaming + SSE; OpenAI tools / tool_calls)
//! - `POST /mcp` (MCP JSON-RPC: initialize, tools/list, tools/call, …)
//!
//! Stdio MCP: [`serve_mcp_stdio`] / CLI `artificer mcp`.

mod api_types;
mod config;
mod error;
#[cfg(test)]
mod hardening_tests;
mod mcp;
mod metrics;
mod prompt;
mod routes;
mod security;
#[cfg(test)]
#[path = "../../runtime/tests/common/mod.rs"]
mod test_common;
mod tools;

use std::sync::Arc;
use std::time::Instant;

use artificer_runtime::Engine;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

pub use config::{is_loopback_host, ApiKey, RequestLimits, ServerConfig};
pub use error::{Result, ServerError};
pub use mcp::{serve_stdio as serve_mcp_stdio, McpState};
use metrics::ServerMetrics;
use routes::{cancel_chat, chat_completions, health, list_models, load_model, AppState};

/// Bind `host:port` for the HTTP server, before any model is loaded.
///
/// `port` 0 asks the OS for a free port; read it back with
/// [`std::net::TcpListener::local_addr`]. The address is resolved exactly as the server always
/// has (`"{host}:{port}"` through tokio's `TcpListener::bind`, so `SO_REUSEADDR` stays on for
/// Unix). The returned listener is non-blocking and ready for [`serve_with_listener`].
///
/// Safe to call from any thread, including from inside another tokio runtime: the bind runs on
/// a short-lived helper thread with its own current-thread runtime.
pub fn bind(host: &str, port: u16) -> std::io::Result<std::net::TcpListener> {
    let addr = format!("{host}:{port}");
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_io()
                    .build()?;
                rt.block_on(async { tokio::net::TcpListener::bind(&addr).await?.into_std() })
            })
            .join()
            .unwrap_or_else(|_| Err(std::io::Error::other("bind thread panicked")))
    })
}

/// Bind, load, and serve until the process is killed (the CLI's `artificer serve`).
///
/// The port is bound first ([`bind`]), so a busy or forbidden port is reported before a
/// multi-GB model is loaded. A non-loopback `host` with no API key is refused before the
/// bind (see [`ServerConfig::check_bind_security`]).
pub async fn serve(cfg: ServerConfig) -> Result<()> {
    cfg.check_bind_security()
        .map_err(ServerError::InsecureBind)?;
    let listener = bind(&cfg.host, cfg.port).map_err(|source| ServerError::Bind {
        addr: format!("{}:{}", cfg.host, cfg.port),
        source,
    })?;
    serve_with_listener(listener, cfg, std::future::pending::<()>()).await
}

/// Load the model in `cfg`, then serve on an already-bound `listener` until `shutdown`
/// completes (graceful: in-flight requests finish first).
///
/// `cfg.host` / `cfg.port` are used only for the log line and the non-loopback warnings; the
/// socket is `listener`. Model load failures return [`ServerError::Runtime`] and nothing is
/// served. A non-loopback `cfg.host` with no API key returns [`ServerError::InsecureBind`]
/// before the model is touched, unless `cfg.allow_insecure_bind`.
pub async fn serve_with_listener<F>(
    listener: std::net::TcpListener,
    cfg: ServerConfig,
    shutdown: F,
) -> Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    cfg.check_bind_security()
        .map_err(ServerError::InsecureBind)?;
    if !cfg.is_loopback_host() && cfg.api_key.is_none() {
        tracing::warn!(
            "binding to non-loopback address {} — this exposes an unauthenticated inference API",
            cfg.host
        );
    }
    if cfg.permissive_cors && !cfg.is_loopback_bind() {
        tracing::warn!("permissive CORS enabled on non-loopback bind");
    }
    listener.set_nonblocking(true)?;
    let local = listener.local_addr()?;
    // Host / Origin are checked against the port actually bound (the desktop binds :0).
    let security =
        Arc::new(security::Security::new(&cfg, local.port()).map_err(ServerError::Other)?);
    if cfg.api_key.is_some() {
        tracing::info!("API key required on every route except GET /health");
    }
    let listener = tokio::net::TcpListener::from_std(listener)?;
    tracing::info!("loading model {} ...", cfg.model_path.display());
    let t0 = Instant::now();
    let load_opts = cfg.load_opts.clone().unwrap_or_default();
    let engine =
        Engine::load_with(&cfg.model_path, load_opts.clone()).map_err(ServerError::from)?;
    if let Some(w) = engine.moe_warning() {
        tracing::warn!("{w}");
    }
    if let Some(r) = engine.context_report {
        if r.capped {
            tracing::info!(
                "context capped {} → {} (KV ~{:.1} MiB)",
                r.requested_ctx,
                r.effective_ctx,
                r.kv_bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }
    tracing::info!(
        "model ready in {:.2}s (arch={}, layers={}, embd={}, n_ctx={})",
        t0.elapsed().as_secs_f32(),
        engine.model().config.arch_str,
        engine.model().config.n_layer,
        engine.model().config.n_embd,
        engine.model().config.n_ctx
    );

    let backend = engine.backend_status().clone();
    if let Some(reason) = &backend.fallback_reason {
        tracing::warn!("CUDA requested but running on CPU: {reason}");
    } else {
        tracing::info!(
            "backend: {}{}",
            backend.active,
            backend
                .device
                .as_deref()
                .map(|d| format!(" ({d})"))
                .unwrap_or_default()
        );
    }
    let control = routes::ChatControl::for_engine(&engine);
    tracing::info!(
        template_opens_think = control.template_opens_think(),
        "chat template probed"
    );
    let state = Arc::new(AppState {
        control,
        backend: tokio::sync::RwLock::new(backend),
        engine: Mutex::new(engine),
        model_id: tokio::sync::RwLock::new(cfg.model_id.clone()),
        load_opts,
        limits: cfg.limits.clone(),
        metrics: ServerMetrics::default(),
    });

    let routes = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/load", post(load_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/cancel", post(cancel_chat))
        .route("/mcp", post(mcp::mcp_http))
        .layer(DefaultBodyLimit::max(cfg.limits.max_body_bytes));
    let mut app = security::apply(routes, security)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    if cfg.permissive_cors {
        app = app.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }

    // The configured address, as before; with port 0 the OS-assigned one.
    let addr = if cfg.port == 0 {
        local.to_string()
    } else {
        format!("{}:{}", cfg.host, cfg.port)
    };
    tracing::info!("OpenAI-compatible server listening on http://{addr}");
    tracing::info!("  GET  /health");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  POST /v1/models/load");
    tracing::info!("  POST /v1/chat/completions  (tools / tool_calls supported)");
    tracing::info!("  POST /v1/chat/cancel       (stop the in-flight completion)");
    tracing::info!("  POST /mcp                  (MCP JSON-RPC)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

/// Blocking entry used by the CLI (`artificer serve`).
pub fn serve_blocking(cfg: ServerConfig) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(serve(cfg))
}

/// Blocking [`serve_with_listener`] on its own multi-thread runtime, for an embedding app that
/// runs the server on a dedicated `std::thread` (the desktop window). Returns when `shutdown`
/// completes and in-flight requests have finished, or on a load/serve error.
pub fn serve_with_listener_blocking<F>(
    listener: std::net::TcpListener,
    cfg: ServerConfig,
    shutdown: F,
) -> Result<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(serve_with_listener(listener, cfg, shutdown))
}

/// Load model and run MCP over stdio (blocking). Used by `artificer mcp`.
pub fn serve_mcp_stdio_blocking(
    model_path: impl AsRef<std::path::Path>,
    model_id: impl Into<String>,
    load_opts: artificer_runtime::EngineLoadOptions,
) -> Result<()> {
    let model_path = model_path.as_ref();
    tracing::info!("loading model {} for MCP ...", model_path.display());
    let t0 = Instant::now();
    let engine = Engine::load_with(model_path, load_opts).map_err(ServerError::from)?;
    if let Some(w) = engine.moe_warning() {
        tracing::warn!("{w}");
    }
    tracing::info!(
        "model ready in {:.2}s (arch={}, layers={}, embd={}, n_ctx={})",
        t0.elapsed().as_secs_f32(),
        engine.model().config.arch_str,
        engine.model().config.n_layer,
        engine.model().config.n_embd,
        engine.model().config.n_ctx
    );
    let state = McpState::from_engine(engine, model_id.into());
    serve_mcp_stdio(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn cfg_for(model: &std::path::Path, port: u16) -> ServerConfig {
        ServerConfig {
            port,
            model_path: model.to_path_buf(),
            ..ServerConfig::default()
        }
    }

    #[test]
    fn bind_port_zero_reads_back_a_real_port() {
        let listener = bind("127.0.0.1", 0).unwrap();
        let addr = listener.local_addr().unwrap();
        assert!(addr.ip().is_loopback());
        assert_ne!(addr.port(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bind_works_inside_a_runtime() {
        let listener = bind("127.0.0.1", 0).unwrap();
        assert_ne!(listener.local_addr().unwrap().port(), 0);
    }

    #[test]
    fn bind_on_a_busy_port_fails_before_any_model_is_touched() {
        let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = held.local_addr().unwrap().port();
        // The model path does not exist: a bind-first server must report the port, not the file.
        let err = serve_blocking(cfg_for(
            std::path::Path::new("/nonexistent/artificer-bind-first.gguf"),
            port,
        ))
        .unwrap_err();
        match err {
            ServerError::Bind { addr, source } => {
                assert_eq!(addr, format!("127.0.0.1:{port}"));
                assert_eq!(source.kind(), std::io::ErrorKind::AddrInUse, "{source}");
            }
            other => panic!("expected a bind error, got {other:?}"),
        }
    }

    #[test]
    fn bad_model_on_a_bound_listener_is_a_load_error() {
        let listener = bind("127.0.0.1", 0).unwrap();
        let err = serve_with_listener_blocking(
            listener,
            cfg_for(
                std::path::Path::new("/nonexistent/artificer-bind-first.gguf"),
                0,
            ),
            std::future::pending::<()>(),
        )
        .unwrap_err();
        assert!(matches!(err, ServerError::Runtime(_)), "{err:?}");
    }

    #[test]
    fn shutdown_signal_stops_a_serving_server() {
        let Some(model) = crate::test_common::test_model("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        let listener = bind("127.0.0.1", 0).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let server = std::thread::spawn(move || {
            serve_with_listener_blocking(listener, cfg_for(&model, 0), async {
                let _ = rx.await;
            })
        });
        // Wait until /health answers on the read-back port.
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let mut up = false;
        while std::time::Instant::now() < deadline {
            if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", port)) {
                use std::io::{Read, Write};
                let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
                let _ = s.write_all(
                    b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
                );
                let mut reply = String::new();
                let _ = s.read_to_string(&mut reply);
                if reply.starts_with("HTTP/1.1 200") && reply.contains("\"status\":\"ok\"") {
                    up = true;
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(up, "server never answered /health on port {port}");
        tx.send(()).unwrap();
        let started = std::time::Instant::now();
        while !server.is_finished() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "server did not stop after the shutdown signal"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        server.join().unwrap().unwrap();
        // The port is free again.
        assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
    }
}
