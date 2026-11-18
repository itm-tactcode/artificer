# Security

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Please don't open a public issue for a security problem.

## Safe use

Artificer is a local inference engine. It is not a hardened multi-user service.

- **The server is local by default.** `artificer serve` binds to `127.0.0.1` unless you pass `--host`, and it logs a warning when you bind to a non-loopback address.
- **There is no authentication or TLS.** Anyone who can reach the port can run generation and read the output, over both `/v1/chat/completions` and `POST /mcp`. Before exposing the server beyond your own machine, put it behind a reverse proxy that adds authentication and TLS.
- **Treat untrusted GGUF files with care.** The loader bounds-checks the file and maps it read-only, but a malformed or malicious model file is still untrusted input. Load models from sources you trust.
- **Tools are never executed by the engine.** OpenAI-style `tools` and `tool_calls` only shape the prompt and parse the model's output. Your client runs any tool and sends back the `role: "tool"` result, so validate model-generated tool arguments in your own code as you would any untrusted input.
