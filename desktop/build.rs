//! Build script for the desktop shell: turns on the `artificer_client_has_*` cfgs when
//! `crates/client/src/lib.rs` defines the matching client calls, then runs `tauri-build` with the
//! app's command list.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(artificer_client_has_chat_completion_stream)");
    println!("cargo::rustc-check-cfg=cfg(artificer_client_has_loaded_model)");
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let client_rs = manifest_dir.join("../crates/client/src/lib.rs");
    println!("cargo:rerun-if-changed={}", client_rs.display());
    let client_src = std::fs::read_to_string(&client_rs).unwrap_or_default();
    if fn_defined(&client_src, "chat_completion_stream") {
        println!("cargo:rustc-cfg=artificer_client_has_chat_completion_stream");
    }
    if fn_defined(&client_src, "loaded_model") {
        println!("cargo:rustc-cfg=artificer_client_has_loaded_model");
    }

    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "check_health",
            "chat_complete",
            "chat_stream",
            "stop_stream",
            "mcp_tools_list",
            "client_caps",
            "loaded_model",
            "loaded_model_info",
            "local_weight_files",
            "pick_local_weights",
            "load_local_weights",
            "smoke_prompt",
            "runtime_probe",
            "runtime_ensure",
        ]),
    ))
    .expect("failed to run tauri-build");
}

/// True when `lib.rs` defines `fn name(`, ignoring comment lines.
/// A mention in a doc comment must not turn the cfg on.
fn fn_defined(src: &str, name: &str) -> bool {
    let needle = format!("fn {name}(");
    src.lines().any(|line| {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with('*') || trimmed.starts_with("/*") {
            return false;
        }
        trimmed.contains(&needle)
    })
}
