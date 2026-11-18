//! Shared helpers for the core integration tests. A test file that needs them declares
//! `mod common;`.
//! Keep `test_model` in sync with its copy in `crates/runtime/tests/common/mod.rs`.

/// A GGUF for the model-backed tests at `$ARTIFICER_TEST_MODELS/<file>`. Prints a skip
/// message and returns `None` when the variable is unset/empty or the file is absent.
pub fn test_model(file: &str) -> Option<std::path::PathBuf> {
    const VAR: &str = "ARTIFICER_TEST_MODELS";
    let Some(dir) = std::env::var_os(VAR).filter(|d| !d.is_empty()) else {
        eprintln!("skip: {VAR} is unset or empty; set it to the directory holding {file}");
        return None;
    };
    let path = std::path::Path::new(&dir).join(file);
    if path.is_file() {
        Some(path)
    } else {
        eprintln!("skip: {file} not found in {VAR} ({})", path.display());
        None
    }
}
