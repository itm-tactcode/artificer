# Security

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Please don't open a public issue for a security problem.

## Safe use

Artificer is a local inference engine. It is not a hardened multi-user service.

- **The server is local by default.** `artificer serve` binds to `127.0.0.1` unless you pass `--host`. It answers only requests whose `Host` header is `127.0.0.1`, `localhost` or `[::1]` (with the port it listens on, or no port) or the non-loopback `--host` you gave, and whose `Origin` header, if any, is `http://127.0.0.1:<port>`, `http://localhost:<port>` or `http://[::1]:<port>`. Other requests get `403`. This is designed to keep web pages you visit from using the server, including through DNS rebinding. Every `POST` must send `Content-Type: application/json` (otherwise `415`). `--allowed-hosts` / `ARTIFICER_ALLOWED_HOSTS` and `--allowed-origins` / `ARTIFICER_ALLOWED_ORIGINS` add comma-separated values to those lists; `*` turns that check off (for example behind a reverse proxy). Turning it off removes the DNS-rebinding protection, so do this only when something in front of the server checks these headers.
- **Authentication is optional, and there is no TLS.** With `--api-key <KEY>` or `ARTIFICER_API_KEY`, every route except `GET /health` needs `Authorization: Bearer <KEY>` (otherwise `401`). Prefer the environment variable: other local users can see command-line arguments. Without a key, anyone who can reach the port can run generation and read the output, over both `/v1/chat/completions` and `POST /mcp`. `serve` refuses to start on a non-loopback `--host` (such as `0.0.0.0`, `::` or a hostname) without a key unless you pass `--allow-insecure-bind`. The key is sent in plain text: before exposing the server beyond your own machine, put it behind a reverse proxy that adds TLS.
- **Treat untrusted GGUF files with care.** The loader bounds-checks the file and maps it read-only, but a malformed or malicious model file is still untrusted input. Load models from sources you trust.
- **Tools are never executed by the engine.** OpenAI-style `tools` and `tool_calls` only shape the prompt and parse the model's output. Your client runs any tool and sends back the `role: "tool"` result, so validate model-generated tool arguments in your own code as you would any untrusted input.
