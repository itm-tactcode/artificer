//! `artificer serve` binds before it reads the model: the real binary, as a user runs it.

use std::net::TcpListener;
use std::process::Command;

fn serve(model: &str, port: u16) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_artificer"))
        .args([
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .output()
        .expect("run artificer");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn taken_port_and_missing_model_prints_the_bind_line_not_a_load_error() {
    let held = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = held.local_addr().unwrap().port();
    let (ok, err) = serve("no-such-model.gguf", port);
    assert!(!ok, "{err}");
    let line = format!("error: could not listen on 127.0.0.1:{port} (");
    assert!(
        err.lines()
            .any(|l| l.starts_with(&line) && l.ends_with("try --port <N>.")),
        "{err}"
    );
    assert!(!err.contains("server failed"), "{err}");
    assert!(
        !err.contains("no-such-model"),
        "the model was never read: {err}"
    );
    drop(held);
}

#[test]
fn free_port_and_missing_model_prints_the_load_error_and_frees_the_port() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let (ok, err) = serve("no-such-model.gguf", port);
    assert!(!ok, "{err}");
    assert!(err.contains("error: server failed"), "{err}");
    assert!(!err.contains("could not listen"), "{err}");
    TcpListener::bind(("127.0.0.1", port))
        .unwrap_or_else(|e| panic!("port {port} still held after the load error: {e}"));
}
