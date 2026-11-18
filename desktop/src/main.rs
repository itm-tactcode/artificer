//! Artificer desktop shell (Tauri 2): registers the window's commands, each a thin wrapper over
//! `client_bridge`, which talks to the runtime through `artificer-client`, plus the native file
//! picker (tauri-plugin-dialog). The runtime (`server`) is a reused `artificer serve` or the
//! built-in server in this process.

// Windows release builds use the GUI subsystem, so no console window opens next to the app window.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod client_bridge;
mod server;

use client_bridge::{ChatMessageIn, ChatTurn, ClientCaps, HealthView};

/// Launch / RETRY: a server that already answers (never starts one). `null` when there is none.
#[tauri::command(async)]
fn runtime_probe() -> Option<server::Found> {
    server::app().probe()
}

/// Before a load: the server to load `path` into, starting the built-in one with `path` when
/// there is none (then `loaded` is true). Errors start with `load-failed: ` or
/// `runtime-failed: `, or are the path check's own line.
#[tauri::command(async)]
fn runtime_ensure(path: String) -> Result<server::Found, String> {
    server::app().ensure(&path)
}

/// `artificer_client::Client::health`.
#[tauri::command]
fn check_health() -> Result<HealthView, String> {
    client_bridge::check_health()
}

/// One non-streaming chat turn: `Client::chat_completion`. `messages` is the chat so far,
/// oldest first, the new user message last.
#[tauri::command]
fn chat_complete(messages: Vec<ChatMessageIn>) -> Result<ChatTurn, String> {
    client_bridge::chat_complete(messages)
}

/// Real chunks via `Client::chat_completion_stream` when that method exists.
/// Until then this returns an error and the window does not pretend to stream.
/// `async` (off the main thread) so `stop_stream` can run while it streams.
/// `messages` as for `chat_complete`.
#[tauri::command(async)]
fn chat_stream(
    on_delta: tauri::ipc::Channel<String>,
    messages: Vec<ChatMessageIn>,
) -> Result<ChatTurn, String> {
    client_bridge::chat_stream(on_delta, messages)
}

/// Stop button: `POST /v1/chat/cancel`, then wait for the stream's final chunk
/// (or cut the stream after a grace period). Off the main thread; may block ~5 s.
#[tauri::command(async)]
fn stop_stream() -> client_bridge::StopView {
    client_bridge::stop_stream()
}

/// MCP `tools/list`: `Client::mcp_tools_list`.
#[tauri::command]
fn mcp_tools_list() -> Result<Vec<String>, String> {
    client_bridge::mcp_tools_list()
}

/// Which optional `artificer-client` methods this build can call. Both are false until `lib.rs` defines them.
#[tauri::command]
fn client_caps() -> ClientCaps {
    client_bridge::client_caps()
}

/// `Client::loaded_model` when that method exists. Does not invent an id.
#[tauri::command]
fn loaded_model() -> Result<Option<String>, String> {
    client_bridge::loaded_model()
}

/// `/v1/models` id + optional `template_opens_think`.
#[tauri::command]
fn loaded_model_info() -> Result<Option<client_bridge::ModelInfoView>, String> {
    client_bridge::loaded_model_info()
}

/// `.gguf` files already on disk. Not a server model list.
#[tauri::command]
fn local_weight_files() -> Vec<String> {
    client_bridge::local_weight_files()
}

/// Native file picker for one `.gguf` that already exists. Cancel is `None`; no picker on this
/// system is `Err("picker-unavailable")`. `async` (off the main thread): the dialog blocks.
#[tauri::command(async)]
fn pick_local_weights(app: tauri::AppHandle) -> Result<Option<String>, String> {
    client_bridge::pick_local_weights(&app)
}

/// Load the chosen local `.gguf`, then return the `GET /v1/models` id.
/// Does not return the path. `async` (off the main thread) so the window keeps painting
/// its loading state while a large file loads.
#[tauri::command(async)]
fn load_local_weights(path: String) -> Result<String, String> {
    client_bridge::load_local_weights(&path)
}

/// Empty unless `ARTIFICER_SMOKE_PROMPT` is set. Normal launches stay idle.
#[tauri::command]
fn smoke_prompt() -> String {
    std::env::var("ARTIFICER_SMOKE_PROMPT").unwrap_or_default()
}

/// Windows title bar and taskbar icons from the exe's own icon resource.
///
/// tauri-build embeds `icons/icon.ico` (16/32/48/256) as icon resource 32512. Tauri 2.12.1 loads one
/// entry of it at `SM_CXICON` and tao sets that only as `ICON_SMALL`, so the title bar shows a
/// downscaled image and `ICON_BIG` is never set. Here each slot gets the frame Windows wants: the
/// small-icon size (16 at 100 % scaling) and the big-icon size (32; the taskbar picks 48 from the
/// same resource where it needs it). Failures leave Tauri's icon in place.
#[cfg(windows)]
fn set_windows_window_icons(window: &tauri::WebviewWindow) {
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, LoadImageW, SendMessageW, ICON_BIG, ICON_SMALL, IMAGE_ICON,
        LR_DEFAULTCOLOR, LR_SHARED, SM_CXICON, SM_CXSMICON, SM_CYICON, SM_CYSMICON, WM_SETICON,
    };

    /// tauri-build's resource id for the app icon (`MAKEINTRESOURCE(32512)`).
    const APP_ICON_ID: usize = 32512;

    let Ok(tauri_hwnd) = window.hwnd() else {
        return;
    };
    // Tauri may link a different `windows` version than this crate; go through the raw pointer value.
    let hwnd = HWND(tauri_hwnd.0 as *mut core::ffi::c_void);
    // SAFETY: GetModuleHandleW(NULL) returns the handle of this process's own exe; nothing is freed.
    let Ok(module) = (unsafe { GetModuleHandleW(PCWSTR::null()) }) else {
        return;
    };
    for (slot, cx, cy) in [
        (ICON_SMALL, SM_CXSMICON, SM_CYSMICON),
        (ICON_BIG, SM_CXICON, SM_CYICON),
    ] {
        // SAFETY: GetSystemMetrics only reads a system value for a valid SM_* index.
        let (w, h) = unsafe { (GetSystemMetrics(cx), GetSystemMetrics(cy)) };
        // SAFETY: `module` is this exe; the name is an integer resource id (MAKEINTRESOURCE), which
        // LoadImageW accepts in place of a string pointer. LR_SHARED makes the system own the icon,
        // so it stays valid for the window's lifetime and must not be destroyed by us.
        let icon = unsafe {
            LoadImageW(
                Some(module.into()),
                PCWSTR(APP_ICON_ID as *const u16),
                IMAGE_ICON,
                w,
                h,
                LR_DEFAULTCOLOR | LR_SHARED,
            )
        };
        let Ok(icon) = icon else {
            continue;
        };
        // SAFETY: `hwnd` is the live main window (we are in setup, on the main thread) and `icon` is a
        // shared HICON; WM_SETICON only stores it.
        unsafe {
            SendMessageW(
                hwnd,
                WM_SETICON,
                Some(WPARAM(slot as usize)),
                Some(LPARAM(icon.0 as isize)),
            );
        }
    }
}

fn main() {
    // Window icon: on Linux and macOS Tauri embeds one PNG, `icons/icon.png` (the 256), as the default
    // window icon. On Windows the 16/32/48/256 frames of `icons/icon.ico` are in the exe resource, and
    // `set_windows_window_icons` gives the title bar and taskbar the matching frames (small and big).
    // FIXME(0.1.x): on X11, also give GTK the 48 (icon list) so small surfaces use the locked 48 pixel
    // icon from logo-final/ instead of a scaled-down 256.
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            #[cfg(windows)]
            {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    set_windows_window_icons(&window);
                }
            }
            let handle = app.handle().clone();
            server::install(server::Runtime::for_app(std::sync::Arc::new(
                move |ev: &server::RuntimeEvent| {
                    use tauri::Emitter;
                    let _ = handle.emit(server::EVENT, ev.clone());
                },
            )));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            check_health,
            chat_complete,
            chat_stream,
            stop_stream,
            mcp_tools_list,
            client_caps,
            loaded_model,
            loaded_model_info,
            local_weight_files,
            pick_local_weights,
            load_local_weights,
            smoke_prompt,
            runtime_probe,
            runtime_ensure
        ])
        .build(tauri::generate_context!())
        .expect("error while building Artificer")
        .run(|_app, event| {
            // Stop the built-in server (never a reused one) and wait for it, briefly.
            if let tauri::RunEvent::Exit = event {
                server::app().shutdown(server::EXIT_JOIN);
            }
        });
}
