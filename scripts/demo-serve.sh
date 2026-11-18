#!/usr/bin/env bash
# Demo: build (if needed) + serve a GGUF on 127.0.0.1:8080.
# Usage: ARTIFICER_MODEL=path/to/your-model.gguf scripts/demo-serve.sh   (or pass the path as the first argument)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-${ARTIFICER_MODEL:-}}"
if [[ -z "$MODEL" ]]; then
  echo "usage: ARTIFICER_MODEL=path/to/your-model.gguf scripts/demo-serve.sh   (or scripts/demo-serve.sh path/to/your-model.gguf)" >&2
  echo "No model given. Use any GGUF of a supported architecture (README: Architecture support); this script never downloads models." >&2
  exit 2
fi
HOST="${ARTIFICER_HOST:-127.0.0.1}"
PORT="${ARTIFICER_PORT:-8080}"
CTX="${ARTIFICER_CTX:-2048}"
BIN="$ROOT/target/release/artificer"

if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  echo "Check the path, or pass another GGUF as the first argument or via ARTIFICER_MODEL." >&2
  exit 1
fi

# The model id is the GGUF file name without `.gguf` (path/to/your-model.gguf -> your-model).
MODEL_ID="$(basename "$MODEL" .gguf)"

# Fail fast if port already in use
if command -v ss >/dev/null 2>&1; then
  if ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .; then
    echo "error: port ${PORT} already in use (HOST=${HOST}). Stop the other process or set ARTIFICER_PORT." >&2
    exit 1
  fi
elif command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:"${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "error: port ${PORT} already in use (HOST=${HOST}). Stop the other process or set ARTIFICER_PORT." >&2
    exit 1
  fi
fi

if [[ ! -x "$BIN" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: binary missing at $BIN and cargo not found in PATH." >&2
    echo "Install Rust (cargo) or place a prebuilt artificer binary there." >&2
    exit 1
  fi
  echo "==> building release artificer-cli..."
  cargo build -p artificer-cli --release
fi

echo "==> serving $MODEL on http://${HOST}:${PORT} (ctx=${CTX}, model-id=${MODEL_ID})"
echo "    smoke:   ./scripts/demo-smoke.sh"
echo "    health:  curl -s http://${HOST}:${PORT}/health"
echo "    mcp:     curl -s http://${HOST}:${PORT}/mcp -H 'Content-Type: application/json' -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}'"
exec "$BIN" serve "$MODEL" --host "$HOST" --port "$PORT" --ctx "$CTX" --kv-f16 --model-id "$MODEL_ID"
