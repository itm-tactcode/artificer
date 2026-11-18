#!/usr/bin/env bash
# Artificer end-to-end demo: inspect -> one-shot run -> bench -> serve + OpenAI API + MCP -> stop.
# Local only. Uses a GGUF already on disk (never downloads). Starts one server on loopback
# and stops it (by its own PID) before exiting.
#
# Usage: scripts/demo.sh MODEL.gguf   (or set ARTIFICER_MODEL; there is no default model)
# Env:   ARTIFICER_BIN   binary to use (default: target/release/artificer, else `artificer` on PATH)
#        ARTIFICER_MODEL model path, used when no argument is given (required: one of the two)
#        DEMO_PORT       loopback port for the serve step (default 18080; must be free)
#        DEMO_BACKEND    cpu (default) or cuda (needs a --features cuda build + NVIDIA driver/CUDA libs)
#        DEMO_SKIP_BENCH=1 to skip the bench step
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-${ARTIFICER_MODEL:-}}"
if [[ -z "$MODEL" ]]; then
  echo "usage: scripts/demo.sh path/to/your-model.gguf   (or ARTIFICER_MODEL=path/to/your-model.gguf scripts/demo.sh)" >&2
  echo "No model given. Use any GGUF of a supported architecture (README: Architecture support); this script never downloads models." >&2
  exit 2
fi
PORT="${DEMO_PORT:-18080}"
BACKEND="${DEMO_BACKEND:-cpu}"
BASE="http://127.0.0.1:${PORT}"

if [[ -n "${ARTIFICER_BIN:-}" ]]; then BIN="$ARTIFICER_BIN"
elif [[ -x "$ROOT/target/release/artificer" ]]; then BIN="$ROOT/target/release/artificer"
elif command -v artificer >/dev/null; then BIN="$(command -v artificer)"
else echo "error: no artificer binary; build with: cargo build --release -p artificer-cli" >&2; exit 1; fi
[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL (this script never downloads models)" >&2; exit 1; }
command -v curl >/dev/null || { echo "error: curl is required for the serve step" >&2; exit 1; }

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "binary: $BIN ($("$BIN" --version))"
step "1/4 inspect: GGUF header + metadata (first lines)"
INSPECT="$("$BIN" inspect "$MODEL" --metadata)"   # capture first: piping into head would SIGPIPE the CLI
printf '%s\n' "$INSPECT" | head -n 25

step "2/4 run: one-shot generation (backend=$BACKEND, greedy)"
"$BIN" run "$MODEL" --backend "$BACKEND" -p "What color is the sky? Answer in one sentence." \
  --max-tokens 48 --ctx 2048 --kv-f16

if [[ "${DEMO_SKIP_BENCH:-0}" != 1 ]]; then
  step "3/4 bench: prefill + decode tok/s and peak RSS"
  "$BIN" bench "$MODEL" --backend "$BACKEND" --max-tokens 64 --ctx 2048 --kv-f16 --runs 2 --warmup 1
fi

step "4/4 serve: OpenAI-compatible API + MCP on $BASE"
if curl -s --max-time 1 "$BASE/health" >/dev/null 2>&1; then
  echo "error: something already answers on $BASE; set DEMO_PORT to a free port" >&2; exit 1
fi
LOG="$(mktemp "${TMPDIR:-/tmp}/artificer-demo.XXXXXX.log")"
"$BIN" serve "$MODEL" --backend "$BACKEND" --host 127.0.0.1 --port "$PORT" --ctx 2048 --kv-f16 \
  --model-id "$(basename "$MODEL" .gguf)" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { if kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
for _ in $(seq 1 120); do
  curl -sf --max-time 1 "$BASE/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "error: server exited; log:" >&2; cat "$LOG" >&2; exit 1; }
  sleep 0.5
done
echo "GET /health -> $(curl -sf "$BASE/health")"
echo "GET /v1/models -> $(curl -sf "$BASE/v1/models")"
echo "POST /v1/chat/completions (non-streaming):"
curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":32,"temperature":0}'
echo; echo "POST /v1/chat/completions (stream=true, SSE; last 3 events):"
curl -sfN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}],"max_tokens":32,"temperature":0,"stream":true}' \
  | grep '^data:' | tail -n 3
echo "POST /mcp tools/list:"
curl -sf "$BASE/mcp" -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
echo
step "stopping server (pid $SERVER_PID); server log: $LOG"
cleanup
trap - EXIT
echo "demo complete"
