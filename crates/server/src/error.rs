//! Server error types and HTTP error responses.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Runtime(#[from] artificer_runtime::RuntimeError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The listening socket could not be opened (`addr` is the requested `host:port`).
    #[error("{}", bind_message(addr, source))]
    Bind {
        addr: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ServerError>;

/// WSAEACCES (10013) on Windows is most often a port in an excluded (reserved) range.
#[cfg(windows)]
const PERMISSION_HINT: &str =
    "The port may be reserved by Windows or need more privileges; try --port <N>.";
#[cfg(not(windows))]
const PERMISSION_HINT: &str = "Permission denied or the port is reserved; try --port <N>.";

/// One plain line for a port that could not be opened: the address, the OS error and what
/// to try next. Windows reports a port in an excluded (reserved) range as WSAEACCES (10013),
/// which Rust maps to `PermissionDenied`.
fn bind_message(addr: &str, source: &std::io::Error) -> String {
    use std::io::ErrorKind;
    let hint = match source.kind() {
        ErrorKind::AddrInUse => "The port is already in use; try --port <N>.",
        ErrorKind::PermissionDenied => PERMISSION_HINT,
        ErrorKind::AddrNotAvailable => "Check --host, or try --port <N>.",
        _ => "The port may be in use or reserved; try --port <N>.",
    };
    format!("could not listen on {addr} ({source}). {hint}")
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    type_: &'static str,
    code: Option<&'static str>,
}

pub fn err_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorDetail {
                message: message.into(),
                type_: "invalid_request_error",
                code: None,
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::ServerError;
    use std::io::{Error, ErrorKind};

    fn bind_err(kind: ErrorKind) -> String {
        ServerError::Bind {
            addr: "127.0.0.1:8080".into(),
            source: Error::from(kind),
        }
        .to_string()
    }

    #[test]
    fn bind_error_is_one_line_with_address_os_error_and_hint() {
        for kind in [
            ErrorKind::AddrInUse,
            ErrorKind::PermissionDenied,
            ErrorKind::AddrNotAvailable,
            ErrorKind::Other,
        ] {
            let msg = bind_err(kind);
            assert!(
                msg.starts_with("could not listen on 127.0.0.1:8080 ("),
                "{msg}"
            );
            assert!(msg.ends_with("try --port <N>."), "{msg}");
            assert!(!msg.contains('\n'), "{msg}");
        }
    }

    #[test]
    fn permission_denied_hint_names_windows_only_on_windows() {
        let msg = bind_err(ErrorKind::PermissionDenied);
        let os = Error::from(ErrorKind::PermissionDenied);
        #[cfg(windows)]
        let hint = "The port may be reserved by Windows or need more privileges; try --port <N>.";
        #[cfg(not(windows))]
        let hint = "Permission denied or the port is reserved; try --port <N>.";
        assert_eq!(
            msg,
            format!("could not listen on 127.0.0.1:8080 ({os}). {hint}")
        );
        assert_eq!(msg.contains("Windows"), cfg!(windows), "{msg}");
    }

    /// WSAEACCES, what Windows returns for a port in an excluded range.
    #[cfg(windows)]
    #[test]
    fn wsaeacces_10013_is_permission_denied() {
        let e = Error::from_raw_os_error(10013);
        assert_eq!(e.kind(), ErrorKind::PermissionDenied);
        let msg = ServerError::Bind {
            addr: "127.0.0.1:8080".into(),
            source: e,
        }
        .to_string();
        assert!(msg.contains("reserved by Windows"), "{msg}");
    }
}
