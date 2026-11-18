//! The window's runtime: reuse an `artificer serve` that is already answering on
//! `127.0.0.1:8080`, or run the server in this process on a free loopback port.
//!
//! - Reuse only a real Artificer server: `GET /health` must parse as the client's
//!   [`artificer_client::HealthStatus`] with `status == "ok"` and a `backend` object, and
//!   `GET /v1/models` must list a model. A reused server is never stopped.
//! - Built-in: [`artificer_server::bind`] on `127.0.0.1:0` (one retry on a new port), then
//!   [`artificer_server::serve_with_listener_blocking`] on a dedicated thread with the picked
//!   `.gguf`. The thread's result decides the window state: a model load error is "load-failed"
//!   (state 5), a bind error, panic or any other exit is "runtime-failed" (state 4, RETRY).
//! - App exit sends the shutdown signal and joins the thread with a timeout.
//! - CPU only: the server's default load options (`EngineLoadOptions::default()`).
//!
//! The base URL every `client_bridge` call uses comes from [`Runtime::base_url`].

use std::future::Future;
use std::net::TcpListener;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use artificer_server::{ServerConfig, ServerError};
use serde::Serialize;

/// Where a user-started `artificer serve` answers (the client's default address).
pub const REUSE_ADDR: &str = "127.0.0.1:8080";
/// Prefix of an `ensure` error the window shows as state 5 (the runtime could not load the file).
pub const LOAD_FAILED: &str = "load-failed: ";
/// Prefix of an `ensure` error the window shows as state 4 (the runtime did not start; RETRY).
pub const RUNTIME_FAILED: &str = "runtime-failed: ";
/// How long app exit waits for the server thread after the shutdown signal.
pub const EXIT_JOIN: Duration = Duration::from_secs(3);
/// Tauri event name for [`RuntimeEvent`]s (`window.ArtificerRuntime.onEvent`).
pub const EVENT: &str = "artificer://runtime";

const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
const READY_POLL: Duration = Duration::from_millis(100);
const READY_LIMIT: Duration = Duration::from_secs(120);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One-shot shutdown signal (no tokio dependency in this crate).
#[derive(Clone, Default)]
pub struct Shutdown(Arc<Mutex<(bool, Option<Waker>)>>);

impl Shutdown {
    pub fn trigger(&self) {
        let mut g = lock(&self.0);
        g.0 = true;
        if let Some(w) = g.1.take() {
            w.wake();
        }
    }

    pub fn is_triggered(&self) -> bool {
        lock(&self.0).0
    }

    /// A future that completes once [`Shutdown::trigger`] has been called.
    pub fn wait(&self) -> ShutdownWait {
        ShutdownWait(self.clone())
    }
}

/// Future side of [`Shutdown`], handed to the server.
pub struct ShutdownWait(Shutdown);

impl ShutdownWait {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_triggered(&self) -> bool {
        self.0.is_triggered()
    }
}

impl Future for ShutdownWait {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut g = lock(&(self.0).0);
        if g.0 {
            Poll::Ready(())
        } else {
            g.1 = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Pushed to the window (`onEvent`): `starting` / `ready` / `exited`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeEvent {
    pub kind: &'static str,
    pub endpoint: Option<String>,
    pub reused: bool,
    pub error: Option<String>,
}

/// A server the window can use. `loaded`: the built-in server was just started with the
/// requested file, so the window must not load it a second time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Found {
    pub endpoint: String,
    pub reused: bool,
    pub loaded: bool,
    pub model: Option<String>,
}

/// Why the server thread ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Stopped,
    LoadFailed(String),
    Failed(String),
    Panicked(String),
}

impl Outcome {
    fn from_result(result: std::thread::Result<Result<(), ServerError>>) -> Self {
        match result {
            Ok(Ok(())) => Outcome::Stopped,
            // serve_with_listener returns Runtime only for the model load.
            Ok(Err(ServerError::Runtime(e))) => Outcome::LoadFailed(e.to_string()),
            Ok(Err(e)) => Outcome::Failed(e.to_string()),
            Err(payload) => Outcome::Panicked(panic_text(
                payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("panic"),
            )),
        }
    }

    /// The `ensure` error text for an exit before the server answered.
    fn start_error(&self) -> String {
        match self {
            Outcome::LoadFailed(m) => format!("{LOAD_FAILED}{m}"),
            Outcome::Failed(m) => format!("{RUNTIME_FAILED}{m}"),
            Outcome::Panicked(m) => {
                format!("{RUNTIME_FAILED}the runtime stopped unexpectedly ({m})")
            }
            Outcome::Stopped => format!("{RUNTIME_FAILED}the runtime stopped"),
        }
    }

    /// The `exited` event text for an exit after the server was ready.
    fn exit_error(&self) -> String {
        match self {
            Outcome::LoadFailed(m) | Outcome::Failed(m) => m.clone(),
            Outcome::Panicked(m) => format!("the runtime stopped unexpectedly ({m})"),
            Outcome::Stopped => "the runtime stopped".into(),
        }
    }
}

/// Longest panic text the window shows, the `…` included.
const PANIC_TEXT_MAX: usize = 120;

/// A panic message for one window line: its first line only, at most [`PANIC_TEXT_MAX`]
/// characters; `…` marks that something was left out (later lines or the rest of the line).
fn panic_text(message: &str) -> String {
    let mut lines = message.lines();
    let first = lines.next().unwrap_or("").trim_end();
    let more_lines = lines.next().is_some();
    if more_lines || first.chars().count() > PANIC_TEXT_MAX {
        let mut cut: String = first.chars().take(PANIC_TEXT_MAX - 1).collect();
        cut.push('…');
        cut
    } else {
        first.to_string()
    }
}

/// Binds the built-in server's listener (`127.0.0.1:0` in the app).
pub type BindFn = dyn Fn() -> std::io::Result<TcpListener> + Send + Sync;
/// Runs the server until shutdown (`serve_with_listener_blocking` in the app).
pub type ServeFn =
    dyn Fn(TcpListener, ServerConfig, ShutdownWait) -> Result<(), ServerError> + Send + Sync;
/// Delivers a [`RuntimeEvent`] to the window.
pub type EmitFn = dyn Fn(&RuntimeEvent) + Send + Sync;

struct Thread {
    endpoint: String,
    shutdown: Shutdown,
    ready: Arc<AtomicBool>,
    done: Arc<(Mutex<Option<Outcome>>, Condvar)>,
    handle: Option<JoinHandle<()>>,
}

impl Thread {
    fn outcome(&self) -> Option<Outcome> {
        lock(&self.done.0).clone()
    }

    /// Signal, then wait at most `limit` for the thread. True when it has ended.
    fn stop(&mut self, limit: Duration) -> bool {
        self.shutdown.trigger();
        let deadline = Instant::now() + limit;
        while let Some(h) = &self.handle {
            if h.is_finished() {
                let _ = self.handle.take().map(JoinHandle::join);
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        true
    }
}

enum Active {
    None,
    Reused(String),
    Builtin(Thread),
}

struct State {
    active: Active,
    starting: bool,
}

static APP: std::sync::OnceLock<Runtime> = std::sync::OnceLock::new();

/// Install the app's runtime (Tauri setup, with an emitter for the window). First call wins.
pub fn install(runtime: Runtime) {
    let _ = APP.set(runtime);
}

/// The app's runtime ([`install`]ed in setup; without a window it emits nowhere).
pub fn app() -> &'static Runtime {
    APP.get_or_init(|| Runtime::for_app(Arc::new(|_: &RuntimeEvent| {})))
}

/// Reuse-or-start logic. The app holds one; tests make their own with fakes.
pub struct Runtime {
    reuse_addr: String,
    bind: Box<BindFn>,
    serve: Arc<ServeFn>,
    emit: Arc<EmitFn>,
    ready_limit: Duration,
    state: Mutex<State>,
}

impl Runtime {
    pub fn new(
        reuse_addr: impl Into<String>,
        bind: Box<BindFn>,
        serve: Arc<ServeFn>,
        emit: Arc<EmitFn>,
    ) -> Self {
        Self {
            reuse_addr: reuse_addr.into(),
            bind,
            serve,
            emit,
            ready_limit: READY_LIMIT,
            state: Mutex::new(State {
                active: Active::None,
                starting: false,
            }),
        }
    }

    #[cfg(test)]
    fn with_ready_limit(mut self, limit: Duration) -> Self {
        self.ready_limit = limit;
        self
    }

    /// The app's runtime: reuse on [`REUSE_ADDR`], built-in server on `127.0.0.1:0`.
    pub fn for_app(emit: Arc<EmitFn>) -> Self {
        Self::new(
            REUSE_ADDR,
            Box::new(|| artificer_server::bind("127.0.0.1", 0)),
            Arc::new(|listener, cfg, shutdown| {
                artificer_server::serve_with_listener_blocking(listener, cfg, shutdown)
            }),
            emit,
        )
    }

    /// Base URL for `artificer_client`: the reused or built-in server, else the client's
    /// default address (nothing is running yet).
    pub fn base_url(&self) -> String {
        match &lock(&self.state).active {
            Active::Reused(addr) => format!("http://{addr}"),
            Active::Builtin(t) => format!("http://{}", t.endpoint),
            Active::None => format!("http://{REUSE_ADDR}"),
        }
    }

    /// Launch / RETRY: a server that already answers. Never starts one.
    pub fn probe(&self) -> Option<Found> {
        {
            let mut st = lock(&self.state);
            if let Active::Builtin(t) = &st.active {
                if t.outcome().is_none() && t.ready.load(Ordering::SeqCst) {
                    return Some(Found {
                        endpoint: t.endpoint.clone(),
                        reused: false,
                        loaded: false,
                        model: None,
                    });
                }
            } else {
                st.active = Active::None;
            }
        }
        let model = artificer_model_at(&self.reuse_addr)?;
        let mut st = lock(&self.state);
        if matches!(st.active, Active::Builtin(_)) {
            return None; // a start raced in; the window will use that one via ensure
        }
        st.active = Active::Reused(self.reuse_addr.clone());
        Some(Found {
            endpoint: self.reuse_addr.clone(),
            reused: true,
            loaded: false,
            model: Some(model),
        })
    }

    /// Before a load: the server to load into, starting the built-in one with `path` when
    /// there is none. Errors start with [`LOAD_FAILED`] or [`RUNTIME_FAILED`], or are the
    /// bridge's own path-check text.
    pub fn ensure(&self, path: &str) -> Result<Found, String> {
        {
            let mut st = lock(&self.state);
            if st.starting {
                return Err(format!("{RUNTIME_FAILED}the runtime is already starting"));
            }
            match &mut st.active {
                Active::Reused(addr) => {
                    return Ok(Found {
                        endpoint: addr.clone(),
                        reused: true,
                        loaded: false,
                        model: None,
                    })
                }
                Active::Builtin(t) if t.outcome().is_none() && t.ready.load(Ordering::SeqCst) => {
                    return Ok(Found {
                        endpoint: t.endpoint.clone(),
                        reused: false,
                        loaded: false,
                        model: None,
                    })
                }
                Active::Builtin(t) => {
                    // Ended (or never became ready): reap it before starting a new one.
                    t.stop(EXIT_JOIN);
                    st.active = Active::None;
                }
                Active::None => {}
            }
        }
        // A server the user started since launch is still preferred over a second runtime.
        if let Some(found) = self.probe() {
            return Ok(found);
        }
        let path = crate::client_bridge::checked_weights_path(path.trim())?;
        {
            let mut st = lock(&self.state);
            if st.starting {
                return Err(format!("{RUNTIME_FAILED}the runtime is already starting"));
            }
            st.starting = true;
        }
        let result = self.start(&path);
        lock(&self.state).starting = false;
        result
    }

    fn start(&self, path: &str) -> Result<Found, String> {
        let listener = match (self.bind)().or_else(|_| (self.bind)()) {
            Ok(l) => l,
            Err(e) => {
                return Err(format!(
                    "{RUNTIME_FAILED}could not listen on a local port ({e})"
                ))
            }
        };
        let port = listener
            .local_addr()
            .map_err(|e| format!("{RUNTIME_FAILED}{e}"))?
            .port();
        let endpoint = format!("127.0.0.1:{port}");
        let model_id = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("local-model")
            .to_string();
        let cfg = ServerConfig {
            host: "127.0.0.1".into(),
            port,
            model_path: path.into(),
            model_id,
            ..ServerConfig::default()
        };
        let shutdown = Shutdown::default();
        let ready = Arc::new(AtomicBool::new(false));
        let done: Arc<(Mutex<Option<Outcome>>, Condvar)> = Arc::default();
        (self.emit)(&RuntimeEvent {
            kind: "starting",
            endpoint: Some(endpoint.clone()),
            reused: false,
            error: None,
        });
        let handle = {
            let (serve, emit) = (Arc::clone(&self.serve), Arc::clone(&self.emit));
            let (shutdown, ready, done) = (shutdown.clone(), Arc::clone(&ready), Arc::clone(&done));
            let wait = shutdown.wait();
            std::thread::Builder::new()
                .name("artificer-server".into())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        serve(listener, cfg, wait)
                    }));
                    let outcome = Outcome::from_result(result);
                    let after_ready = ready.load(Ordering::SeqCst);
                    let error = outcome.exit_error();
                    *lock(&done.0) = Some(outcome);
                    done.1.notify_all();
                    if after_ready && !shutdown.is_triggered() {
                        emit(&RuntimeEvent {
                            kind: "exited",
                            endpoint: None,
                            reused: false,
                            error: Some(error),
                        });
                    }
                })
                .map_err(|e| format!("{RUNTIME_FAILED}{e}"))?
        };
        lock(&self.state).active = Active::Builtin(Thread {
            endpoint: endpoint.clone(),
            shutdown,
            ready: Arc::clone(&ready),
            done: Arc::clone(&done),
            handle: Some(handle),
        });

        let deadline = Instant::now() + self.ready_limit;
        loop {
            let outcome = {
                let guard = lock(&done.0);
                let (guard, _) = done
                    .1
                    .wait_timeout_while(guard, READY_POLL, |o| o.is_none())
                    .unwrap_or_else(|e| e.into_inner());
                guard.clone()
            };
            if let Some(outcome) = outcome {
                self.reap();
                return Err(outcome.start_error());
            }
            if let Some(model) = artificer_model_at(&endpoint) {
                ready.store(true, Ordering::SeqCst);
                // An exit between the check and the flag is reported by ensure/probe next time.
                (self.emit)(&RuntimeEvent {
                    kind: "ready",
                    endpoint: Some(endpoint.clone()),
                    reused: false,
                    error: None,
                });
                return Ok(Found {
                    endpoint,
                    reused: false,
                    loaded: true,
                    model: Some(model),
                });
            }
            if Instant::now() >= deadline {
                self.reap();
                return Err(format!(
                    "{RUNTIME_FAILED}the runtime did not answer within {} minutes",
                    self.ready_limit.as_secs() / 60
                ));
            }
        }
    }

    /// Stop and forget the built-in thread (start failures).
    fn reap(&self) {
        if let Some(mut t) = self.take_builtin() {
            t.stop(EXIT_JOIN);
        }
    }

    /// The built-in thread, taken out of the state (left `None`) under the lock, so the
    /// caller can signal and join it with the lock released.
    fn take_builtin(&self) -> Option<Thread> {
        let mut st = lock(&self.state);
        match std::mem::replace(&mut st.active, Active::None) {
            Active::Builtin(t) => Some(t),
            other => {
                st.active = other;
                None
            }
        }
    }

    /// App exit: stop the built-in server (never a reused one) and join it for at most
    /// `limit`, with the state lock released while it waits. True when no built-in thread is
    /// left running; a thread that did not end in time is put back for a later call.
    pub fn shutdown(&self, limit: Duration) -> bool {
        let Some(mut t) = self.take_builtin() else {
            return true;
        };
        if t.stop(limit) {
            return true;
        }
        let mut st = lock(&self.state);
        if matches!(st.active, Active::None) {
            st.active = Active::Builtin(t);
        }
        false
    }
}

/// The loaded model id when an Artificer server answers at `addr`: `/health` has the
/// Artificer shape (status ok, metrics, backend) and `/v1/models` lists a model.
fn artificer_model_at(addr: &str) -> Option<String> {
    let mut client = artificer_client::Client::with_base_url(format!("http://{addr}")).ok()?;
    client.set_timeout(PROBE_TIMEOUT);
    let health = client.health().ok()?;
    if health.status != "ok" || health.backend.is_none() {
        return None;
    }
    client
        .loaded_model()
        .ok()
        .flatten()
        .filter(|id| !id.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;

    const ARTIFICER_HEALTH: &str = r#"{"status":"ok","metrics":{"requests_started":0,"requests_completed":0,"requests_failed":0,"prompt_tokens_total":0,"completion_tokens_total":0,"in_flight":0,"peak_in_flight":0},"backend":{"requested":"cpu","active":"cpu"}}"#;
    const MODELS: &str = r#"{"object":"list","data":[{"id":"fake-model","object":"model","created":0,"owned_by":"artificer"}]}"#;
    const NO_MODELS: &str = r#"{"object":"list","data":[]}"#;

    /// Answer one HTTP request with `health` for `/health`, `models` for `/v1/models`.
    fn answer(mut s: std::net::TcpStream, health: &str, models: &str) {
        let _ = s.set_nonblocking(false);
        let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
            match s.read(&mut chunk) {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
        let req = String::from_utf8_lossy(&buf);
        let path = req.split_whitespace().nth(1).unwrap_or("");
        let (status, body) = match path {
            "/health" => ("200 OK", health),
            "/v1/models" => ("200 OK", models),
            _ => ("404 Not Found", "{}"),
        };
        let _ = write!(
            s,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }

    /// Serve `listener` until `stop()` (or `crash` makes it panic).
    fn fake_http(
        listener: TcpListener,
        health: &str,
        models: &str,
        stop: impl Fn() -> bool,
        crash: Option<&AtomicBool>,
    ) -> Result<(), ServerError> {
        listener.set_nonblocking(true)?;
        loop {
            if stop() {
                return Ok(());
            }
            if crash.is_some_and(|c| c.load(Ordering::SeqCst)) {
                panic!("forced server panic");
            }
            match listener.accept() {
                Ok((s, _)) => answer(s, health, models),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// A user-started server on its own port (never stopped by a Runtime).
    struct External {
        addr: String,
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl External {
        fn start(health: &'static str, models: &'static str) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                let _ = fake_http(
                    listener,
                    health,
                    models,
                    || flag.load(Ordering::SeqCst),
                    None,
                );
            });
            Self {
                addr,
                stop,
                handle: Some(handle),
            }
        }

        fn running(&self) -> bool {
            !self.handle.as_ref().unwrap().is_finished()
        }
    }

    impl Drop for External {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn unused_addr() -> String {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .to_string()
    }

    type Events = Arc<Mutex<Vec<RuntimeEvent>>>;

    fn recorder() -> (Events, Arc<EmitFn>) {
        let events: Events = Arc::default();
        let sink = Arc::clone(&events);
        (
            events,
            Arc::new(move |ev: &RuntimeEvent| lock(&sink).push(ev.clone())),
        )
    }

    fn kinds(events: &Events) -> Vec<&'static str> {
        lock(events).iter().map(|e| e.kind).collect()
    }

    fn real_bind() -> Box<BindFn> {
        Box::new(|| artificer_server::bind("127.0.0.1", 0))
    }

    /// An Artificer-shaped fake for the built-in server; `crash` panics it on demand.
    fn fake_serve(crash: Arc<AtomicBool>, calls: Arc<AtomicUsize>) -> Arc<ServeFn> {
        Arc::new(move |listener, _cfg, wait| {
            calls.fetch_add(1, Ordering::SeqCst);
            fake_http(
                listener,
                ARTIFICER_HEALTH,
                MODELS,
                || wait.is_triggered(),
                Some(&crash),
            )
        })
    }

    /// A `.gguf` file in a fresh temp dir (contents as given).
    fn gguf_file(tag: &str, bytes: &[u8]) -> (PathBuf, String) {
        let dir = std::env::temp_dir().join(format!(
            "artificer-desktop-runtime-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("model.gguf");
        std::fs::write(&file, bytes).unwrap();
        (dir, file.to_string_lossy().into_owned())
    }

    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        let end = Instant::now() + Duration::from_secs(10);
        while Instant::now() < end {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn start_serves_on_a_dynamic_port_and_shutdown_joins() {
        let (events, emit) = recorder();
        let calls = Arc::new(AtomicUsize::new(0));
        let rt = Runtime::new(
            unused_addr(),
            real_bind(),
            fake_serve(Arc::default(), Arc::clone(&calls)),
            emit,
        );
        assert_eq!(rt.probe(), None, "nothing is running yet");
        let (dir, path) = gguf_file("start", b"fake");
        let found = rt.ensure(&path).unwrap();
        assert!(!found.reused && found.loaded);
        assert_eq!(found.model.as_deref(), Some("fake-model"));
        assert_ne!(found.endpoint, REUSE_ADDR);
        assert_eq!(rt.base_url(), format!("http://{}", found.endpoint));
        assert_eq!(kinds(&events), ["starting", "ready"]);
        // A second load uses the same server (no second start, no second model load).
        let again = rt.ensure(&path).unwrap();
        assert_eq!(again.endpoint, found.endpoint);
        assert!(!again.loaded);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(rt.probe().map(|f| f.endpoint), Some(found.endpoint.clone()));

        assert!(
            rt.shutdown(EXIT_JOIN),
            "the server thread joins after the signal"
        );
        assert_eq!(
            kinds(&events),
            ["starting", "ready"],
            "a requested stop is not an exit"
        );
        assert!(
            artificer_model_at(&found.endpoint).is_none(),
            "nothing answers after shutdown"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_artificer_server_on_the_reuse_address_is_reused_and_never_stopped() {
        let ext = External::start(ARTIFICER_HEALTH, MODELS);
        let (events, emit) = recorder();
        let calls = Arc::new(AtomicUsize::new(0));
        let rt = Runtime::new(
            ext.addr.clone(),
            real_bind(),
            fake_serve(Arc::default(), Arc::clone(&calls)),
            emit,
        );
        let found = rt.probe().unwrap();
        assert!(found.reused && !found.loaded);
        assert_eq!(found.endpoint, ext.addr);
        assert_eq!(found.model.as_deref(), Some("fake-model"));
        assert_eq!(rt.base_url(), format!("http://{}", ext.addr));
        let (dir, path) = gguf_file("reuse", b"fake");
        let found = rt.ensure(&path).unwrap();
        assert!(
            found.reused && !found.loaded,
            "the window loads into the reused server"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no built-in server");
        assert!(rt.shutdown(EXIT_JOIN));
        assert!(ext.running(), "a reused server is never stopped");
        assert!(artificer_model_at(&ext.addr).is_some());
        assert!(kinds(&events).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_non_artificer_server_or_one_without_a_model_is_not_reused() {
        for (health, models) in [
            (r#"{"status":"ok"}"#, MODELS),
            (r#"{"status":"ok","uptime":3}"#, MODELS),
            (ARTIFICER_HEALTH, NO_MODELS),
            ("<html>ok</html>", "<html></html>"),
        ] {
            let ext = External::start(health, models);
            let calls = Arc::new(AtomicUsize::new(0));
            let (_events, emit) = recorder();
            let rt = Runtime::new(
                ext.addr.clone(),
                real_bind(),
                fake_serve(Arc::default(), Arc::clone(&calls)),
                emit,
            );
            assert_eq!(rt.probe(), None, "{health} / {models}");
            let (dir, path) = gguf_file("not-reused", b"fake");
            let found = rt.ensure(&path).unwrap();
            assert!(!found.reused, "{health}: started its own");
            assert_ne!(found.endpoint, ext.addr);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(rt.shutdown(EXIT_JOIN));
            assert!(ext.running());
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_bad_gguf_is_a_load_failure_state_5() {
        let (events, emit) = recorder();
        // The real server: bind, then the model load fails.
        let rt = Runtime::for_app(emit);
        let rt = Runtime {
            reuse_addr: unused_addr(),
            ..rt
        };
        let (dir, path) = gguf_file("bad", b"this is not a gguf file");
        let err = rt.ensure(&path).unwrap_err();
        assert!(err.starts_with(LOAD_FAILED), "{err}");
        assert_eq!(kinds(&events), ["starting"]);
        assert_eq!(rt.probe(), None);
        assert!(rt.shutdown(EXIT_JOIN));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_path_the_bridge_refuses_never_starts_a_server() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (events, emit) = recorder();
        let rt = Runtime::new(
            unused_addr(),
            real_bind(),
            fake_serve(Arc::default(), Arc::clone(&calls)),
            emit,
        );
        let err = rt.ensure("relative/model.gguf").unwrap_err();
        assert!(err.starts_with("model path must be absolute"), "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(kinds(&events).is_empty());
    }

    #[test]
    fn a_port_collision_is_retried_once_on_a_new_port() {
        let binds = Arc::new(AtomicUsize::new(0));
        let n = Arc::clone(&binds);
        let bind: Box<BindFn> = Box::new(move || {
            if n.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(std::io::Error::from(std::io::ErrorKind::AddrInUse))
            } else {
                artificer_server::bind("127.0.0.1", 0)
            }
        });
        let (_events, emit) = recorder();
        let rt = Runtime::new(
            unused_addr(),
            bind,
            fake_serve(Arc::default(), Arc::default()),
            emit,
        );
        let (dir, path) = gguf_file("collision", b"fake");
        let found = rt.ensure(&path).unwrap();
        assert!(found.loaded);
        assert_eq!(binds.load(Ordering::SeqCst), 2);
        assert!(rt.shutdown(EXIT_JOIN));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn two_bind_failures_are_a_runtime_failure_state_4() {
        let binds = Arc::new(AtomicUsize::new(0));
        let n = Arc::clone(&binds);
        let held = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = held.local_addr().unwrap().port();
        // Always the same busy port: a real collision, twice.
        let bind: Box<BindFn> = Box::new(move || {
            n.fetch_add(1, Ordering::SeqCst);
            artificer_server::bind("127.0.0.1", port)
        });
        let (events, emit) = recorder();
        let rt = Runtime::new(
            unused_addr(),
            bind,
            fake_serve(Arc::default(), Arc::default()),
            emit,
        );
        let (dir, path) = gguf_file("bindfail", b"fake");
        let err = rt.ensure(&path).unwrap_err();
        assert!(err.starts_with(RUNTIME_FAILED), "{err}");
        assert!(err.contains("could not listen"), "{err}");
        assert!(
            !err.contains(&port.to_string()),
            "no port number in window text: {err}"
        );
        assert_eq!(binds.load(Ordering::SeqCst), 2, "one retry");
        assert!(kinds(&events).is_empty());
        drop(held);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_panic_before_ready_is_a_runtime_failure_state_4() {
        let (events, emit) = recorder();
        let serve: Arc<ServeFn> = Arc::new(|_l, _c, _w| panic!("forced server panic"));
        let rt = Runtime::new(unused_addr(), real_bind(), serve, emit);
        let (dir, path) = gguf_file("panic", b"fake");
        let err = rt.ensure(&path).unwrap_err();
        assert!(err.starts_with(RUNTIME_FAILED), "{err}");
        assert!(err.contains("stopped unexpectedly"), "{err}");
        assert!(err.contains("forced server panic"), "{err}");
        assert_eq!(kinds(&events), ["starting"]);
        assert!(rt.shutdown(EXIT_JOIN));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_panic_after_ready_emits_exited_and_retry_starts_again() {
        let (events, emit) = recorder();
        let crash = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let rt = Runtime::new(
            unused_addr(),
            real_bind(),
            fake_serve(Arc::clone(&crash), Arc::clone(&calls)),
            emit,
        );
        let (dir, path) = gguf_file("exit", b"fake");
        rt.ensure(&path).unwrap();
        crash.store(true, Ordering::SeqCst);
        assert!(wait_for(|| kinds(&events).contains(&"exited")));
        let exited = lock(&events).last().cloned().unwrap();
        assert_eq!(exited.kind, "exited");
        assert!(
            exited
                .error
                .as_deref()
                .unwrap_or("")
                .contains("stopped unexpectedly"),
            "{exited:?}"
        );
        assert_eq!(rt.probe(), None, "a dead server is not offered");
        // RETRY: a fresh start.
        crash.store(false, Ordering::SeqCst);
        let found = rt.ensure(&path).unwrap();
        assert!(found.loaded);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(rt.shutdown(EXIT_JOIN));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_start_that_never_answers_is_state_4_and_retry_starts_again() {
        assert_eq!(READY_LIMIT, Duration::from_secs(120));
        assert_eq!(READY_LIMIT.as_secs() / 60, 2, "the window says 2 minutes");
        let (events, emit) = recorder();
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        // First start: bound (connections wait in the backlog) but never answers, like a
        // hung load. Later starts answer like a healthy server.
        let serve: Arc<ServeFn> = Arc::new(move |listener, _c, wait| {
            if c.fetch_add(1, Ordering::SeqCst) == 0 {
                while !wait.is_triggered() {
                    std::thread::sleep(Duration::from_millis(10));
                }
                return Ok(());
            }
            fake_http(
                listener,
                ARTIFICER_HEALTH,
                MODELS,
                || wait.is_triggered(),
                None,
            )
        });
        let rt = Runtime::new(unused_addr(), real_bind(), serve, emit)
            .with_ready_limit(Duration::from_millis(300));
        let (dir, path) = gguf_file("hang", b"fake");
        let t0 = Instant::now();
        let err = rt.ensure(&path).unwrap_err();
        let took = t0.elapsed();
        assert!(err.starts_with(RUNTIME_FAILED), "{err}");
        assert!(err.contains("did not answer within"), "{err}");
        assert!(
            took >= Duration::from_millis(300) && took < Duration::from_secs(20),
            "{took:?}"
        );
        assert_eq!(kinds(&events), ["starting"]);
        // RETRY: the hung thread was stopped and joined; a fresh start becomes ready.
        let found = rt.ensure(&path).unwrap();
        assert!(found.loaded && !found.reused);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(kinds(&events), ["starting", "starting", "ready"]);
        assert!(rt.shutdown(EXIT_JOIN));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_server_that_ignores_shutdown_is_not_waited_on_forever() {
        let (_events, emit) = recorder();
        let release = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&release);
        let serve: Arc<ServeFn> = Arc::new(move |listener, _c, _w| {
            fake_http(
                listener,
                ARTIFICER_HEALTH,
                MODELS,
                || r.load(Ordering::SeqCst),
                None,
            )
        });
        let rt = Runtime::new(unused_addr(), real_bind(), serve, emit);
        let (dir, path) = gguf_file("stuck", b"fake");
        rt.ensure(&path).unwrap();
        let t0 = Instant::now();
        assert!(!rt.shutdown(Duration::from_millis(200)));
        assert!(t0.elapsed() < Duration::from_secs(2));
        release.store(true, Ordering::SeqCst);
        assert!(rt.shutdown(EXIT_JOIN));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn shutdown_waits_with_the_state_lock_released() {
        let (_events, emit) = recorder();
        let release = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&release);
        let serve: Arc<ServeFn> = Arc::new(move |listener, _c, _w| {
            fake_http(
                listener,
                ARTIFICER_HEALTH,
                MODELS,
                || r.load(Ordering::SeqCst),
                None,
            )
        });
        let rt = Runtime::new(unused_addr(), real_bind(), serve, emit);
        let (dir, path) = gguf_file("unlocked", b"fake");
        rt.ensure(&path).unwrap();
        std::thread::scope(|s| {
            let waiting = s.spawn(|| rt.shutdown(Duration::from_millis(1500)));
            std::thread::sleep(Duration::from_millis(200));
            // The state is free while shutdown waits: a call that only takes the state lock
            // does not block behind it. Only that call is timed: probe() also connects to the
            // reuse address, and a refused connect alone can take ~1 s on Windows.
            let t0 = Instant::now();
            let base = rt.base_url();
            let took = t0.elapsed();
            assert!(took < Duration::from_millis(1000), "{took:?}");
            assert_eq!(
                base,
                format!("http://{REUSE_ADDR}"),
                "the thread is out of the state"
            );
            assert!(rt.probe().is_none());
            release.store(true, Ordering::SeqCst);
            assert!(waiting.join().unwrap(), "the thread ends once released");
        });
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn panic_text_is_one_short_line() {
        assert_eq!(panic_text("boom"), "boom");
        assert_eq!(panic_text("first line\nsecond line\nthird"), "first line…");
        assert_eq!(panic_text("trailing  \n"), "trailing");
        let long = "x".repeat(300);
        let cut = panic_text(&long);
        assert_eq!(cut.chars().count(), PANIC_TEXT_MAX);
        assert!(cut.ends_with('…'));
        assert_eq!(
            panic_text(&"y".repeat(PANIC_TEXT_MAX)),
            "y".repeat(PANIC_TEXT_MAX)
        );
        let long_multi = format!("{}\nmore", "é".repeat(200));
        let cut = panic_text(&long_multi);
        assert_eq!(cut.chars().count(), PANIC_TEXT_MAX);
        assert!(cut.starts_with('é') && cut.ends_with('…') && !cut.contains('\n'));
        let payload: Box<dyn std::any::Any + Send> =
            Box::new(format!("bad state\n{}", "z".repeat(500)));
        let err = Outcome::from_result(Err(payload)).start_error();
        assert_eq!(
            err,
            format!("{RUNTIME_FAILED}the runtime stopped unexpectedly (bad state…)")
        );
    }

    /// The real server with a real model (skipped without `ARTIFICER_TEST_MODELS`).
    #[test]
    fn the_real_server_starts_with_a_model_and_shuts_down() {
        let Some(dir) = std::env::var_os("ARTIFICER_TEST_MODELS") else {
            eprintln!("skip: ARTIFICER_TEST_MODELS not set");
            return;
        };
        let model = PathBuf::from(dir).join("tiny-vicuna-1b.q4_k_m.gguf");
        if !model.is_file() {
            eprintln!("skip: {} missing", model.display());
            return;
        }
        let (events, emit) = recorder();
        let rt = Runtime {
            reuse_addr: unused_addr(),
            ..Runtime::for_app(emit)
        };
        let found = rt.ensure(&model.to_string_lossy()).unwrap();
        assert!(found.loaded && !found.reused);
        assert_eq!(found.model.as_deref(), Some("tiny-vicuna-1b.q4_k_m"));
        assert_eq!(kinds(&events), ["starting", "ready"]);
        assert!(rt.shutdown(EXIT_JOIN));
        assert!(artificer_model_at(&found.endpoint).is_none());
        assert_eq!(kinds(&events), ["starting", "ready"]);
    }
}
