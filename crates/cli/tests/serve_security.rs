//! `artificer serve` refuses a non-loopback bind without an API key, before it binds or
//! reads the model. Run against the real binary.

use std::net::TcpListener;
use std::process::Command;

fn serve(args: &[&str], env: &[(&str, &str)]) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_artificer"));
    cmd.arg("serve").arg("no-such-model.gguf").args(args);
    for var in [
        "ARTIFICER_API_KEY",
        "ARTIFICER_ALLOWED_HOSTS",
        "ARTIFICER_ALLOWED_ORIGINS",
    ] {
        cmd.env_remove(var);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run artificer");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn non_loopback_host_without_a_key_is_refused_before_bind_and_load() {
    let port = free_port().to_string();
    for host in ["0.0.0.0", "::", "my-box.example"] {
        let (ok, err) = serve(&["--host", host, "--port", &port], &[]);
        assert!(!ok, "{err}");
        assert!(
            err.contains(&format!(
                "refusing to listen on non-loopback address {host} without an API key"
            )),
            "{err}"
        );
        assert!(err.contains("--allow-insecure-bind"), "{err}");
        assert!(!err.contains("could not listen"), "never bound: {err}");
        assert!(!err.contains("no-such-model"), "never loaded: {err}");
    }
}

#[test]
fn a_key_or_the_insecure_flag_lifts_the_refusal() {
    // The refusal is lifted, so the run gets to the (missing) model. A held port makes it
    // stop at the bind, so nothing is ever served on a non-loopback address.
    let held = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = held.local_addr().unwrap().port().to_string();
    type Case<'a> = (&'a [&'a str], &'a [(&'a str, &'a str)]);
    let cases: [Case; 3] = [
        (&["--allow-insecure-bind"], &[]),
        (&["--api-key", "flag-secret-123"], &[]),
        (&[], &[("ARTIFICER_API_KEY", "env-secret-456")]),
    ];
    for (extra, env) in cases {
        let mut args = ["--host", "127.0.0.1", "--port", &port];
        // Check the gate for a non-loopback host with the same options, held loopback port.
        let (ok, err) = serve(&[&args[..], extra].concat(), env);
        assert!(!ok, "{err}");
        assert!(err.contains("could not listen on 127.0.0.1"), "{err}");
        args[1] = "my-box.invalid";
        let (ok, err) = serve(&[&args[..], extra].concat(), env);
        assert!(!ok, "{err}");
        assert!(
            !err.contains("refusing to listen"),
            "{extra:?} {env:?}: {err}"
        );
        assert!(!err.contains("secret"), "the key is never printed: {err}");
    }
    drop(held);
}

#[test]
fn an_empty_api_key_is_an_error_not_no_auth() {
    let port = free_port().to_string();
    let (ok, err) = serve(&["--port", &port, "--api-key", ""], &[]);
    assert!(!ok, "{err}");
    assert!(err.contains("must not be empty"), "{err}");
    let (ok, err) = serve(&["--port", &port], &[("ARTIFICER_API_KEY", "")]);
    assert!(!ok, "{err}");
    assert!(err.contains("ARTIFICER_API_KEY must not be empty"), "{err}");
}

#[test]
fn a_key_with_surrounding_whitespace_is_an_error() {
    let port = free_port().to_string();
    for key in [" abc", "abc ", " abc "] {
        let (ok, err) = serve(&["--port", &port, "--api-key", key], &[]);
        assert!(!ok, "{key:?}: {err}");
        assert!(
            err.contains("leading or trailing whitespace"),
            "{key:?}: {err}"
        );
        assert!(!err.contains("no-such-model"), "never loaded: {err}");
        let (ok, err) = serve(&["--port", &port], &[("ARTIFICER_API_KEY", key)]);
        assert!(!ok, "{key:?}: {err}");
        assert!(
            err.contains("ARTIFICER_API_KEY") && err.contains("leading or trailing whitespace"),
            "{key:?}: {err}"
        );
        assert!(!err.contains("no-such-model"), "never loaded: {err}");
    }
}

#[test]
fn a_whitespace_padded_api_key_flag_does_not_echo_the_key() {
    let port = free_port().to_string();
    let (ok, err) = serve(&["--port", &port, "--api-key", " s3cr3t-xyz "], &[]);
    assert!(!ok, "{err}");
    assert!(err.contains("--api-key"), "{err}");
    assert!(err.contains("leading or trailing whitespace"), "{err}");
    assert!(
        !err.contains("s3cr3t-xyz"),
        "the key is never printed: {err}"
    );
    assert!(!err.contains("no-such-model"), "never loaded: {err}");

    let (ok, err) = serve(&["--port", &port, "--api-key", ""], &[]);
    assert!(!ok, "{err}");
    assert!(err.contains("--api-key"), "{err}");
    assert!(err.contains("must not be empty"), "{err}");
}
