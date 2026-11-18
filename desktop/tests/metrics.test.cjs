// node --test desktop/tests/  (no dependencies)
const test = require("node:test");
const assert = require("node:assert/strict");
const m = require("../dist/metrics.js");

test("badge: cuda with device", () => {
  const b = m.backendBadge({ requested: "cuda", active: "cuda", device: "NVIDIA GeForce RTX 4050 Laptop GPU", fallback_reason: null });
  assert.equal(b.text, "GPU · CUDA · RTX 4050 Laptop");
  assert.equal(b.tone, "gpu");
  assert.equal(b.title, "CUDA device: NVIDIA GeForce RTX 4050 Laptop GPU");
});

test("badge: plain cpu", () => {
  assert.deepEqual(m.backendBadge({ requested: "cpu", active: "cpu" }).text, "CPU");
  assert.equal(m.backendBadge({ active: "cpu" }).tone, "cpu");
});

test("badge: cuda requested, fell back → amber with reason in title", () => {
  const b = m.backendBadge({ requested: "cuda", active: "cpu", device: null, fallback_reason: "CUDA backend: no device" });
  assert.equal(b.text, "CPU (GPU unavailable)");
  assert.equal(b.tone, "fallback");
  assert.equal(b.title, "CUDA backend: no device");
});

test("badge: missing backend is never a silent CPU", () => {
  for (const x of [null, undefined, {}, { active: "" }]) {
    const b = m.backendBadge(x);
    assert.equal(b.tone, "unknown");
    assert.notEqual(b.text, "CPU");
  }
});

test("formatRate", () => {
  assert.equal(m.formatRate(12.44), "12.4");
  assert.equal(m.formatRate(310.2), "310");
  assert.equal(m.formatRate(99.96), "100.0"); // < 100 branch, toFixed(1) rounds
  assert.equal(m.formatRate(0), "–");
  assert.equal(m.formatRate(null), "–");
  assert.equal(m.formatRate(NaN), "–");
});

test("rolling rate: steady 12.5 tok/s, ignores pre-window history", () => {
  const r = new m.RollingRate(2000);
  assert.equal(r.rate(), null);
  r.push(0);
  assert.equal(r.rate(), null);
  for (let i = 1; i <= 50; i++) r.push(i * 80); // 80 ms per token
  assert.ok(Math.abs(r.rate() - 12.5) < 1e-9);
  assert.equal(r.count, 51);
  // Slow down to 200 ms/token; after > window the old fast tokens are gone.
  let t = 50 * 80;
  for (let i = 0; i < 20; i++) { t += 200; r.push(t); }
  assert.ok(Math.abs(r.rate() - 5) < 1e-9, String(r.rate()));
});

test("live and final readouts", () => {
  assert.equal(m.liveReadout(12.4), "Decoding 12.4 tok/s");
  assert.equal(m.liveReadout(null), "Decoding – tok/s");
  assert.equal(
    m.finalReadout({ prefill_tokens: 31, prefill_ms: 100, decode_tokens: 128, decode_ms: 10322.58 }, { totalTokens: 128 }),
    "Prefill 310 tok/s · Decode 12.4 tok/s · 128 tokens"
  );
  // Server total with a thousands separator.
  assert.equal(
    m.finalReadout({ prefill_tokens: 31, prefill_ms: 100, decode_tokens: 1024, decode_ms: 56320 }, { totalTokens: 1024 }),
    "Prefill 310 tok/s · Decode 18.2 tok/s · 1,024 tokens"
  );
  // No server total → no count (never a client count), timings or not.
  assert.equal(m.finalReadout({ prefill_tokens: 31, prefill_ms: 100, decode_tokens: 128, decode_ms: 10322.58 }, {}),
    "Prefill 310 tok/s · Decode 12.4 tok/s");
  // After a Stop: speed only, or the server total if it sent one.
  assert.equal(m.finalReadout(null, { rate: 9.04, tokens: 40 }), "Decode 9.0 tok/s");
  assert.equal(m.finalReadout(null, { rate: 9.04, totalTokens: 1500 }), "Decode 9.0 tok/s · 1,500 tokens");
  assert.ok(!/client count/.test(m.finalReadout(null, { rate: 9.04, tokens: 40 })));
  assert.equal(m.groupDigits(1234567), "1,234,567");
  // Nothing generated (Stop during prefill): prefill speed only, no "Decode – · 0 tokens".
  assert.equal(m.finalReadout({ prefill_tokens: 31, prefill_ms: 4000, decode_tokens: 0, decode_ms: 0 }, { rate: null, totalTokens: 0 }),
    "Prefill 7.8 tok/s");
  assert.equal(m.finalReadout({ prefill_tokens: 31, prefill_ms: 4000, decode_tokens: 0, decode_ms: 0 }, {}), "Prefill 7.8 tok/s");
  assert.ok(!/Decode|0 tokens/.test(m.finalReadout({ prefill_tokens: 31, prefill_ms: 4000, decode_tokens: 0, decode_ms: 0 }, { totalTokens: 0 })));
  // One token generated still shows decode + total.
  assert.equal(m.finalReadout({ prefill_tokens: 31, prefill_ms: 100, decode_tokens: 1, decode_ms: 50 }, { totalTokens: 1 }),
    "Prefill 310 tok/s · Decode 20.0 tok/s · 1 token");
});

test("badge: short device name keeps unknown vendors intact", () => {
  assert.equal(m.backendBadge({ active: "cuda", device: "NVIDIA RTX A2000" }).text, "GPU · CUDA · RTX A2000");
  assert.equal(m.backendBadge({ active: "cuda", device: "GPU" }).text, "GPU · CUDA · GPU");
  assert.equal(m.backendBadge({ active: "cuda", device: "" }).text, "GPU · CUDA");
});

test("weights path: middle truncation keeps the filename intact", () => {
  const p = "/data/YOU/Projects/Artificer/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf";
  assert.deepEqual(m.pathCandidates(p), [
    p,
    "/data/YOU/Projects/\u2026/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "/data/YOU/\u2026/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "/data/\u2026/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "\u2026/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
  ]);
  assert.equal(m.middleTruncatePath(p, 200), p);
  assert.equal(m.middleTruncatePath(p, 48), "/data/YOU/\u2026/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf");
  assert.equal(m.middleTruncatePath(p, 10), "\u2026/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"); // never cuts the filename
});

test("weights path: short, relative, Windows and non-path values", () => {
  assert.deepEqual(m.pathCandidates("/x.gguf"), ["/x.gguf"]);
  assert.deepEqual(m.pathCandidates("x.gguf"), ["x.gguf"]);
  assert.deepEqual(m.pathCandidates("models/x.gguf"), ["models/x.gguf", "\u2026/x.gguf"]);
  assert.deepEqual(m.pathCandidates("C:\\m\\a\\x.gguf"), ["C:\\m\\a\\x.gguf", "C:\\m\\\u2026\\x.gguf", "\u2026\\x.gguf"]);
  assert.deepEqual(m.pathCandidates("no .gguf file on disk"), ["no .gguf file on disk"]);
  assert.deepEqual(m.pathCandidates(null), [""]);
});
