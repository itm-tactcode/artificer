# Artificer demo path

**Goal:** local OpenAI-compatible server + MCP in under ~15 minutes on Linux.

## Prerequisites

- Linux x86_64, enough free RAM for your model (a ~1B Q4 GGUF needs about 2 GB)
- Rust/`cargo` **or** a prebuilt `target/release/artificer` binary
- A GGUF model you already have, of an architecture listed under **Architecture support** in the [README](README.md)
  (a small ~1B Q4 model keeps this fast). The scripts never download models; GGUF files are available from
  Hugging Face and other sources. Below it is `path/to/your-model.gguf`.

## Timed first-run checklist (~15 min)

| Step | Action | Expect |
|------|--------|--------|
| 1 | From the repo root: `ARTIFICER_MODEL=path/to/your-model.gguf ./scripts/demo-serve.sh` | Server stays up on `127.0.0.1:8080` (first build may take a few minutes if binary missing) |
| 2 | Second terminal: `./scripts/demo-smoke.sh` | Three `PASS:` lines + `SMOKE PASSED` |

**CPU note:** streaming chat with a small model on CPU can take **~15–20s wall** for a short completion. That is normal (prefill-dominated on CPU), not a hang.

`demo-serve.sh` needs the model (`ARTIFICER_MODEL` or the first argument; there is no default) and exits with a usage
message without one. Other env overrides: `ARTIFICER_HOST`, `ARTIFICER_PORT`, `ARTIFICER_CTX`.  
`--model-id` is the GGUF file name without `.gguf` (`path/to/your-model.gguf` → `your-model`).

### Manual install / binary path

```bash
cargo build -p artificer-cli --release
./target/release/artificer serve path/to/your-model.gguf \
  --host 127.0.0.1 --port 8080 --ctx 2048 --kv-f16 --model-id your-model
```

## Smoke script (preferred)

With the server already running:

```bash
cd <repo>   # your Artificer checkout
./scripts/demo-smoke.sh
```

Runs: `GET /health` → streaming `POST /v1/chat/completions` → `POST /mcp` `tools/list`. Exits non-zero on any failure.

### Manual curls (optional)

```bash
curl -s http://127.0.0.1:8080/health

curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":32,"temperature":0,"stream":true}'

curl -s http://127.0.0.1:8080/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

### MCP stdio (optional — agent hosts)

```bash
./target/release/artificer mcp path/to/your-model.gguf --ctx 2048 --kv-f16
# then send newline-delimited JSON-RPC on stdin, e.g. tools/list
```

## Works-with recipe: Cursor MCP (stdio)

1. Build once (if needed): `cargo build -p artificer-cli --release`
2. Merge the `mcpServers.artificer` block from `scripts/mcp.cursor.example.json` into your existing Cursor MCP config (`~/.cursor/mcp.json`; create it only if you don't have one). Don't copy the example file over your config, which would replace your other servers. Then change both `/REPLACE_WITH_HOME` placeholders to your home directory and `path/to/your-model.gguf` to your model
(the `--model-id` value `your-model` is free text). Paths must be **absolute**.

```bash
# Example absolute paths, with <home> = your home directory (what `echo $HOME` prints):
#   command: <home>/Projects/Artificer/target/release/artificer
#   model:   <home>/path/to/your-model.gguf
```

3. Reload MCP in Cursor; tools `chat` / `complete` should appear.

Prefer `./scripts/demo-smoke.sh` (HTTP) for the fastest proof without configuring an IDE.

## Out of scope (this demo)

This walkthrough covers the CPU build with a small model of your own. GPU builds, MoE models, the optional desktop app and hosted inference aren't covered here.
