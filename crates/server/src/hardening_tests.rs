//! End-to-end checks for the 0.1.1 hardening: `POST /v1/models/load` error text, `/health`,
//! MCP prompt / token caps and cancellation, and the non-loopback bind refusal. The
//! model-backed ones read `$ARTIFICER_TEST_MODELS` and skip without it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use artificer_runtime::{Engine, EngineLoadOptions};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::config::{ApiKey, ServerConfig};
use crate::metrics::ServerMetrics;
use crate::routes::{self, AppState, ChatControl};

fn config(host: &str) -> ServerConfig {
    ServerConfig {
        host: host.into(),
        model_path: PathBuf::from("/nonexistent/never-open-security.gguf"),
        ..ServerConfig::default()
    }
}

#[test]
fn bind_security_loopback_ip_and_hostnames() {
    for host in [
        "127.0.0.1",
        "127.0.0.2",
        "127.255.1.9",
        "::1",
        "[::1]",
        "localhost",
    ] {
        assert!(config(host).is_loopback_host(), "{host}");
        assert!(config(host).check_bind_security().is_ok(), "{host}");
    }
    // Don't change the old is_loopback_bind predicate: other callers use it.
    assert!(!config("127.0.0.2").is_loopback_bind());
    for host in [
        "0.0.0.0",
        "::",
        "[::]",
        "box.lan",
        "LOCALHOST",
        "localhost.evil.example",
        "192.168.1.20",
    ] {
        let mut cfg = config(host);
        assert!(!cfg.is_loopback_host(), "{host}");
        let err = cfg.check_bind_security().unwrap_err();
        assert!(
            err.contains("--api-key") && err.contains("--allow-insecure-bind"),
            "{err}"
        );
        cfg.allow_insecure_bind = true;
        assert!(cfg.check_bind_security().is_ok(), "{host}");
        cfg.allow_insecure_bind = false;
        cfg.api_key = ApiKey::new("not-logged");
        assert!(cfg.check_bind_security().is_ok(), "{host}");
    }
}

#[tokio::test]
async fn serve_refuses_non_loopback_before_loading_or_binding() {
    for host in ["0.0.0.0", "::", "unresolvable.invalid"] {
        let cfg = config(host);
        let err = crate::serve(cfg.clone()).await.unwrap_err();
        assert!(
            matches!(err, crate::ServerError::InsecureBind(_)),
            "{host}: {err}"
        );
        // Embedding entry point must enforce it too, even with a separately supplied socket.
        let listener = crate::bind("127.0.0.1", 0).unwrap();
        let err = crate::serve_with_listener(listener, cfg, std::future::pending())
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::ServerError::InsecureBind(_)),
            "{host}: {err}"
        );
        let listener = crate::bind("127.0.0.1", 0).unwrap();
        let mut cfg = config(host);
        cfg.allow_insecure_bind = true;
        let err = crate::serve_with_listener(listener, cfg, std::future::pending())
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::ServerError::Runtime(_)),
            "{host}: got past the gate: {err}"
        );
    }
}

fn state(limits: crate::RequestLimits) -> Option<Arc<AppState>> {
    let model = crate::test_common::test_model("tiny-vicuna-1b.q4_k_m.gguf")?;
    let engine = Engine::load(model).unwrap();
    Some(Arc::new(AppState {
        control: ChatControl::for_engine(&engine),
        backend: tokio::sync::RwLock::new(engine.backend_status().clone()),
        engine: tokio::sync::Mutex::new(engine),
        model_id: tokio::sync::RwLock::new("path-sentinel-secret-model.gguf".into()),
        load_opts: EngineLoadOptions::default(),
        limits,
        metrics: ServerMetrics::default(),
    }))
}

async fn body(resp: Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("artificer-security-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn gguf_with_arch(arch: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensors
    b.extend_from_slice(&1u64.to_le_bytes()); // metadata
    let string = |b: &mut Vec<u8>, v: &str| {
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    };
    string(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes()); // string type
    string(&mut b, arch);
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b
}

fn engine_error(path: &Path) -> String {
    let err = Engine::load_with(path, EngineLoadOptions::default())
        .err()
        .expect("load must fail");
    routes::public_load_error(&err)
}

#[test]
fn load_error_mapping_without_a_model() {
    use artificer_core::CoreError;
    use artificer_runtime::RuntimeError;
    let tmp = TempDir::new();
    let missing = tmp.0.join("never-existed.gguf");
    let dir = tmp.0.join("directory.gguf");
    std::fs::create_dir(&dir).unwrap();
    let not_gguf = tmp.file("private-document.bin", b"file-bytes-sentinel");
    for p in [&missing, &dir, &not_gguf] {
        assert_eq!(
            routes::checked_model_path(&p.to_string_lossy()),
            Err(routes::LOAD_PATH_ERROR),
            "{}",
            p.display()
        );
    }
    assert_eq!(routes::checked_model_path("  "), Err("path is required"));
    assert_eq!(
        routes::checked_model_path("relative/m.gguf"),
        Err("path must be an absolute filesystem path")
    );
    let good = tmp.file("ok.GGUF", b"");
    assert_eq!(
        routes::checked_model_path(&good.to_string_lossy()),
        Ok(good.canonicalize().unwrap())
    );

    // Corrupt files: the engine's own error quotes bytes, the HTTP message does not.
    let not_magic = tmp.file("private-a.gguf", b"PRIVATE-FILE-BYTES-SENTINEL");
    let mut forged_key = gguf_with_arch("llama");
    forged_key[32] = b'!';
    let bad_key = tmp.file("private-b.gguf", &forged_key);
    let truncated = tmp.file("private-c.gguf", &gguf_with_arch("llama")[..30]);
    for p in [&not_magic, &bad_key, &truncated] {
        assert_eq!(
            engine_error(p),
            routes::LOAD_FAILED_GENERIC,
            "{}",
            p.display()
        );
    }
    for junk in [
        "private-file-bytes-sentinel",
        "nemotron_h_private",
        "private_mamba_bytes",
        "private_moe_bytes",
        "",
    ] {
        let p = tmp.file("junk.gguf", &gguf_with_arch(junk));
        let msg = engine_error(&p);
        assert_eq!(
            msg, "failed to load model: unsupported model architecture",
            "{junk:?}"
        );
    }
    let p = tmp.file("junk.gguf", &gguf_with_arch("llama (private-file-bytes)"));
    assert_eq!(
        engine_error(&p),
        "failed to load model: unsupported architecture `llama`"
    );
    for named in ["deepseek2", "qwen3moe", "jamba", "falcon-h1"] {
        let p = tmp.file("named.gguf", &gguf_with_arch(named));
        assert_eq!(
            engine_error(&p),
            format!("failed to load model: unsupported architecture `{named}`")
        );
    }

    // Our fixed refusals pass through, but only for a known architecture name.
    let map = |e: CoreError| routes::public_load_error(&RuntimeError::Core(e));
    let moe = format!(
        "`olmoe` is an MoE model; MoE support is experimental: pass {} to load it",
        artificer_core::EXPERIMENTAL_MOE_FLAG
    );
    assert_eq!(
        map(CoreError::Other(moe.clone())),
        format!("failed to load model: {moe}")
    );
    let forged = moe.replace("olmoe", "private bytes");
    assert_eq!(map(CoreError::Other(forged)), routes::LOAD_FAILED_GENERIC);
    let cuda = "the CUDA backend does not support `qwen2` models yet: it uses NEOX-style RoPE. \
                NEOX-RoPE architectures and MoE models run on the CPU only; use the CPU backend \
                (--backend cpu).";
    assert_eq!(
        map(CoreError::Backend(cuda.into())),
        format!("failed to load model: {cuda}")
    );
    let forged = cuda.replace("qwen2", "private bytes");
    assert_eq!(map(CoreError::Backend(forged)), routes::LOAD_FAILED_GENERIC);
    assert_eq!(
        map(CoreError::Backend("cuInit failed: /private/lib.so".into())),
        routes::LOAD_FAILED_GENERIC
    );
    assert_eq!(
        map(CoreError::UnsupportedArchitecture(
            "deepseek with expert_count=64 (MoE not supported; use a dense distill GGUF)".into()
        )),
        "failed to load model: unsupported architecture `deepseek`"
    );
    for e in [
        CoreError::MissingMetadata("private.key".into()),
        CoreError::BadTensor {
            name: "private".into(),
            msg: "x".into(),
        },
        CoreError::Tokenizer("private".into()),
        CoreError::Other("private".into()),
    ] {
        assert_eq!(map(e), routes::LOAD_FAILED_GENERIC);
    }
    assert_eq!(
        routes::public_load_error(&RuntimeError::Other("private".into())),
        routes::LOAD_FAILED_GENERIC
    );
}

async fn load(state: &Arc<AppState>, path: &Path) -> Value {
    let request = serde_json::from_value(json!({"path": path.to_string_lossy()})).unwrap();
    let resp = routes::load_model(State(Arc::clone(state)), Json(request)).await;
    assert!(!resp.status().is_success());
    body(resp).await
}

#[tokio::test]
async fn http_load_errors_and_health_do_not_expose_paths_or_file_bytes() {
    let Some(state) = state(crate::RequestLimits::default()) else {
        return;
    };
    let tmp = TempDir::new();
    let missing = tmp.0.join("never-existed.gguf");
    let dir = tmp.0.join("directory.gguf");
    std::fs::create_dir(&dir).unwrap();
    let not_gguf = tmp.file("private-document.txt", b"file-bytes-sentinel");
    let mut previous = None;
    for path in [&missing, &dir, &not_gguf] {
        let response = load(&state, path).await;
        assert_eq!(response["error"]["message"], routes::LOAD_PATH_ERROR);
        let s = response.to_string();
        assert!(!s.contains(&*path.to_string_lossy()), "{s}");
        assert!(!s.contains("file-bytes-sentinel"), "{s}");
        if let Some(prev) = previous {
            assert_eq!(response, prev);
        }
        previous = Some(response);
    }
    // A metadata parse error which quotes its input.
    let mut broken = gguf_with_arch("llama");
    // Replace the first key, which must be ASCII hierarchical, with a forged key.
    let key_at = 24 + 8;
    broken[key_at] = b'!';
    let corrupt = tmp.file("private-corrupt.gguf", &broken);
    let response = load(&state, &corrupt).await;
    assert_eq!(response["error"]["message"], routes::LOAD_FAILED_GENERIC);
    let s = response.to_string();
    assert!(
        !s.contains("!eneral.architecture") && !s.contains("private-corrupt"),
        "{s}"
    );
    // Architecture names come from file bytes. Unknown names are not echoed, even
    // when they trigger one of Architecture::parse's substring-based refusal hints.
    for junk in [
        "private-file-bytes-sentinel",
        "nemotron_h_private",
        "private_mamba_bytes",
        "private_moe_bytes",
    ] {
        let path = tmp.file("junk-arch.gguf", &gguf_with_arch(junk));
        let response = load(&state, &path).await;
        assert_eq!(
            response["error"]["message"],
            "failed to load model: unsupported model architecture"
        );
        assert!(!response.to_string().contains(junk), "{response}");
    }
    // A known name is echoed, and only the name: never what follows it in the file.
    let path = tmp.file(
        "junk-arch.gguf",
        &gguf_with_arch("llama (private-file-bytes)"),
    );
    let response = load(&state, &path).await;
    assert_eq!(
        response["error"]["message"],
        "failed to load model: unsupported architecture `llama`"
    );
    let path = tmp.file("junk-arch.gguf", &gguf_with_arch("deepseek2"));
    let response = load(&state, &path).await;
    assert_eq!(
        response["error"]["message"],
        "failed to load model: unsupported architecture `deepseek2`"
    );
    // Health is public: status, metrics, backend — no model id, path or filename.
    let response = body(routes::health(State(state)).await.into_response()).await;
    assert_eq!(response["status"], "ok");
    assert!(response["metrics"].is_object() && response["backend"].is_object());
    assert!(!response.to_string().contains("path-sentinel-secret-model"));
    assert_eq!(response.as_object().unwrap().len(), 3, "{response}");
}

#[tokio::test]
async fn allow_insecure_bind_and_api_key_can_start_server() {
    let Some(model) = crate::test_common::test_model("tiny-vicuna-1b.q4_k_m.gguf") else {
        return;
    };
    // A loopback listener stands in for the wildcard socket, so the test needs no
    // IPv6, DNS or LAN exposure; the gate reads `cfg.host`. Shutdown is immediate.
    for (host, with_key) in [
        ("0.0.0.0", false),
        ("::", false),
        ("hostname.example", false),
        ("0.0.0.0", true),
    ] {
        let listener = crate::bind("127.0.0.1", 0).unwrap();
        let mut cfg = config(host);
        cfg.model_path = model.clone();
        cfg.port = 0;
        cfg.allow_insecure_bind = !with_key;
        if with_key {
            cfg.api_key = ApiKey::new("secret");
        }
        crate::serve_with_listener(listener, cfg, async {})
            .await
            .unwrap_or_else(|e| panic!("{host} key={with_key}: {e}"));
    }
}

async fn rpc(state: &Arc<AppState>, method: &str, id: &str, args: Value) -> Value {
    let request = json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":method,"arguments":args}});
    body(crate::mcp::mcp_http(State(Arc::clone(state)), request.to_string()).await).await
}

async fn wait_active(state: &AppState) {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        // `begin_request` runs after the engine lock and the cancel registration.
        if state.metrics.snapshot().in_flight > 0 {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "MCP never became active"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn cancel_endpoint(state: &Arc<AppState>, id: Option<&str>) -> Value {
    let req = serde_json::from_value(json!({"id": id})).unwrap();
    body(
        routes::cancel_chat(State(Arc::clone(state)), Some(Json(req)))
            .await
            .into_response(),
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_caps_parse_errors_cancel_and_hangup() {
    let limits = crate::RequestLimits {
        max_tokens: 24,
        max_prompt_chars: 180,
        ..Default::default()
    };
    let Some(state) = state(limits) else { return };
    let raw = crate::mcp::mcp_http(State(Arc::clone(&state)), "not-json".into()).await;
    assert_eq!(raw.status(), StatusCode::BAD_REQUEST);
    assert_eq!(body(raw).await["error"]["code"], -32700);
    let raw = crate::mcp::mcp_http(
        State(Arc::clone(&state)),
        json!({"id":1,"method":"tools/list"}).to_string(),
    )
    .await;
    assert!(body(raw).await["result"]["tools"].is_array());
    for method in ["chat", "complete"] {
        let args = if method == "chat" {
            // Each message fits separately; together they exceed max_prompt_chars.
            json!({"messages":[{"role":"user","content":"x".repeat(100)},{"role":"user","content":"y".repeat(100)}]})
        } else {
            json!({"prompt":"x".repeat(181)})
        };
        let resp = rpc(&state, method, "over", args).await;
        assert_eq!(resp["result"]["isError"], true, "{resp}");
        assert!(
            resp["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("prompt too large"),
            "{resp}"
        );
        let before = state.metrics.snapshot().completion_tokens_total;
        let resp = rpc(&state, method, "over-tokens", tool_args(method, 100_000)).await;
        assert_eq!(resp["result"]["isError"], false, "{resp}");
        let decoded = state.metrics.snapshot().completion_tokens_total - before;
        assert!(decoded <= 24, "token cap not applied: {decoded}");
    }
    // Longer cap now, so Stop and a dropped HTTP handler must end it well before the cap.
    let Some(state) = self::state(crate::RequestLimits::default()) else {
        return;
    };
    for (i, method) in ["chat", "complete"].into_iter().enumerate() {
        let id = format!("mcp-cancel-{i}");
        let st = Arc::clone(&state);
        let task_id = id.clone();
        let before = state.metrics.snapshot().completion_tokens_total;
        let task =
            tokio::spawn(async move { rpc(&st, method, &task_id, tool_args(method, 400)).await });
        wait_active(&state).await;
        let wrong = cancel_endpoint(&state, Some("wrong-id")).await;
        assert_eq!(wrong["cancelled"], false);
        let reply = cancel_endpoint(&state, if i == 0 { Some(&id) } else { None }).await;
        assert_eq!(reply["cancelled"], true, "{reply}");
        assert_eq!(reply["id"], id);
        let response = tokio::time::timeout(Duration::from_secs(30), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response["result"]["isError"], false, "{response}");
        assert!(state.metrics.snapshot().completion_tokens_total - before < 400);
        assert_eq!(cancel_endpoint(&state, None).await["cancelled"], false);
    }
    for method in ["chat", "complete"] {
        let st = Arc::clone(&state);
        let before = state.metrics.snapshot().completion_tokens_total;
        let task =
            tokio::spawn(async move { rpc(&st, method, "hangup", tool_args(method, 400)).await });
        wait_active(&state).await;
        task.abort(); // What dropping the HTTP handler future does on a disconnect.
        let _ = task.await;
        let engine = tokio::time::timeout(Duration::from_secs(30), state.engine.lock())
            .await
            .unwrap();
        assert!(state.metrics.snapshot().completion_tokens_total - before < 400);
        drop(engine);
        assert_eq!(cancel_endpoint(&state, None).await["cancelled"], false);
    }
}

fn tool_args(method: &str, max_tokens: usize) -> Value {
    let prompt = "Write a very long story about a dragon, at least 500 words.";
    if method == "chat" {
        json!({"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"seed":1})
    } else {
        json!({"prompt":prompt,"max_tokens":max_tokens,"temperature":0,"seed":1})
    }
}
