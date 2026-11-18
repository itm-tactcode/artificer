#!/usr/bin/env bash
# Assumes artificer serve is already up. Verifies health + stream chat + MCP tools/list.
set -euo pipefail

HOST="${ARTIFICER_HOST:-127.0.0.1}"
PORT="${ARTIFICER_PORT:-8080}"
BASE="http://${HOST}:${PORT}"
FAIL=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; FAIL=1; }

echo "==> smoke against ${BASE}"

# --- 1. Health ---
health_out=""
if health_out="$(curl -sf --max-time 5 "${BASE}/health" 2>/dev/null)" && [[ -n "$health_out" ]]; then
  pass "GET /health  (${health_out})"
else
  fail "GET /health  (is demo-serve.sh running on ${HOST}:${PORT}?)"
fi

# --- 2. Streaming chat completions (a small model on CPU: ~15–20s OK) ---
chat_body='{"messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":32,"temperature":0,"stream":true}'
chat_out=""
chat_rc=0
set +e
chat_start=$(date +%s)
chat_out="$(curl -sfN --max-time 120 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$chat_body" 2>/dev/null)"
chat_rc=$?
chat_end=$(date +%s)
set -e
chat_secs=$((chat_end - chat_start))

if [[ $chat_rc -eq 0 ]] && printf '%s' "$chat_out" | grep -q 'data:' && printf '%s' "$chat_out" | grep -q '\[DONE\]'; then
  pass "POST /v1/chat/completions stream  (~${chat_secs}s wall; 15–20s OK for a small model on CPU)"
else
  fail "POST /v1/chat/completions stream  (rc=${chat_rc}, ~${chat_secs}s; expect SSE data: + [DONE])"
fi

# --- 3. MCP tools/list ---
mcp_body='{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
mcp_out=""
if mcp_out="$(curl -sf --max-time 10 "${BASE}/mcp" \
  -H 'Content-Type: application/json' \
  -d "$mcp_body" 2>/dev/null)" \
  && printf '%s' "$mcp_out" | grep -q '"tools"'; then
  pass "POST /mcp tools/list  (${mcp_out})"
else
  fail "POST /mcp tools/list"
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "==> SMOKE FAILED"
  exit 1
fi
echo "==> SMOKE PASSED"
exit 0
