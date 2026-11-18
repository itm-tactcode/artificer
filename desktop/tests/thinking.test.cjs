// node --test desktop/tests/  (no dependencies)
const test = require("node:test");
const assert = require("node:assert/strict");
const t = require("../dist/thinking.js");

const NM = { implicitOpen: true };
const OTHER = { implicitOpen: false };

test("split: Nemotron implicit opener, closed by </think>", () => {
  const p = t.split("We need a factorial.\nUse a loop.\n</think>\n```python\ndef f(n): ...\n```", NM);
  assert.equal(p.state, "closed");
  assert.equal(p.thinking, "We need a factorial.\nUse a loop.");
  assert.equal(p.answer, "```python\ndef f(n): ...\n```");
});

test("split: explicit <think> opener, any model", () => {
  const p = t.split("<think>\nplan it\n</think>\n\nThe sky is blue.", OTHER);
  assert.deepEqual([p.state, p.thinking, p.answer], ["closed", "plan it", "The sky is blue."]);
  const q = t.split("  <think>plan</think>Answer", NM);
  assert.deepEqual([q.state, q.thinking, q.answer], ["closed", "plan", "Answer"]);
});

test("split: no-think reply has no section", () => {
  for (const opts of [OTHER, undefined]) {
    const p = t.split("Hello there. 1 < 2 and that's fine.", opts);
    assert.equal(p.state, "none");
    assert.equal(p.thinking, "");
    assert.equal(p.answer, "Hello there. 1 < 2 and that's fine.");
  }
  // Empty think block → no section, answer intact.
  const e = t.split("<think>\n\n</think>\n\nJust the answer.", OTHER);
  assert.deepEqual([e.state, e.thinking, e.answer], ["none", "", "Just the answer."]);
  // A Nemotron reply that ended normally without </think> was not thinking.
  const f = t.finalSplit("Plain reply, no tags.", NM, "stop");
  assert.deepEqual([f.state, f.thinking, f.answer], ["none", "", "Plain reply, no tags."]);
});

test("split: still-open think while streaming and at the cap", () => {
  const s = t.split("Let me think about blue light scattering", NM);
  assert.deepEqual([s.state, s.thinking, s.answer], ["open", "Let me think about blue light scattering", ""]);
  const x = t.split("<think>step 1, step 2", OTHER);
  assert.deepEqual([x.state, x.thinking, x.answer], ["open", "step 1, step 2", ""]);
  const cap = t.finalSplit("Rayleigh scattering means shorter wavel", NM, "length");
  assert.deepEqual([cap.state, cap.thinking, cap.answer], ["open", "Rayleigh scattering means shorter wavel", ""]);
  const user = t.finalSplit("<think>half a thought", OTHER, "user");
  assert.equal(user.state, "open");
  assert.equal(user.answer, "");
});

test("split: tags split across deltas never flash as text", () => {
  assert.equal(t.split("thinking...</thi", NM).thinking, "thinking...");
  assert.equal(t.split("thinking...</thi", NM).answer, "");
  assert.equal(t.split("<th", OTHER).answer, "");
  assert.equal(t.finalSplit("<th", OTHER, "length").answer, "<th");
  assert.equal(t.split("<", NM).state, "open");
});

test("formatCount: thousands separators", () => {
  assert.equal(t.formatCount(0), "0");
  assert.equal(t.formatCount(999), "999");
  assert.equal(t.formatCount(1024), "1,024");
  assert.equal(t.formatCount(128), "128");
  assert.equal(t.formatCount(32768), "32,768");
  assert.equal(t.formatCount(1234567), "1,234,567");
  assert.equal(t.formatCount(undefined), "");
});

test("header: streaming counts up, done shows server thinking tokens", () => {
  assert.equal(t.headerText({ streaming: true, elapsedMs: 4300 }), "Thinking\u2026 4s");
  assert.equal(t.headerText({ streaming: true, elapsedMs: 200 }), "Thinking\u2026 0s");
  assert.equal(t.headerText({ elapsedMs: 11600, state: "closed", finishReason: "stop", thinkingTokens: 418 }),
    "Thought for 12s \u00b7 418 tokens");
  assert.equal(t.headerText({ elapsedMs: 65000, state: "closed", thinkingTokens: 1500 }), "Thought for 1m 5s \u00b7 1,500 tokens");
  assert.equal(t.headerText({ elapsedMs: 3000, state: "closed", thinkingTokens: 1 }), "Thought for 3s \u00b7 1 token");
  // No count from the server → no count (the UI never counts tokens itself).
  assert.equal(t.headerText({ elapsedMs: 3000, state: "closed", thinkingTokens: null }), "Thought for 3s");
  assert.equal(t.headerText({ elapsedMs: 9000, state: "open", finishReason: "length", maxTokens: 1024 }),
    "Thinking stopped at the 1,024-token limit");
});

test("notice copy for each finish reason", () => {
  assert.equal(t.noticeText("length", { maxTokens: 1024, answered: true }), "Stopped at the 1,024-token limit.");
  assert.equal(t.noticeText("length", { maxTokens: 1024, answered: false }), "Stopped at the 1,024-token limit before an answer.");
  assert.equal(t.noticeText("length", { maxTokens: null, answered: true }), "Stopped at the token limit.");
  assert.equal(t.noticeText("user", {}), "Stopped by you.");
  assert.equal(t.noticeText("stop", { maxTokens: 1024 }), null);
  assert.equal(t.noticeText("error", { maxTokens: 1024 }), null);
  assert.equal(t.noticeText(null, {}), null);
  for (const r of ["stop", "length", "user", "error", null]) {
    const s = t.noticeText(r, { maxTokens: 128, answered: true });
    if (s) assert.ok(!/continue/i.test(s));
  }
});

test("readDone: top-level done-event outcome fields, read defensively", () => {
  const d = t.readDone({
    finish_reason: "length", max_tokens: 1024, tokens: { thinking: 418, answer: 606, total: 1024 }, thinking_ms: 11600,
    choices: [{ finish_reason: "length" }], usage: { completion_tokens: 1024 }
  }, 128);
  assert.deepEqual(d, { finishReason: "length", maxTokens: 1024, thinkingTokens: 418, answerTokens: 606, totalTokens: 1024, thinkingMs: 11600,
    prefillMs: null, limitHitInThinking: null, errorMessage: null, connectionDropped: false });
  // Today's server: choices[0].finish_reason + usage; limit = what the request sent.
  const legacy = t.readDone({ choices: [{ finish_reason: "stop" }], usage: { completion_tokens: 9 } }, 128);
  assert.deepEqual(legacy, { finishReason: "stop", maxTokens: 128, thinkingTokens: null, answerTokens: null, totalTokens: 9, thinkingMs: null,
    prefillMs: null, limitHitInThinking: null, errorMessage: null, connectionDropped: false });
  const junk = t.readDone({ finish_reason: "banana", max_tokens: "x", tokens: { thinking: -1 }, thinking_ms: "soon" });
  assert.deepEqual(junk, { finishReason: null, maxTokens: null, thinkingTokens: null, answerTokens: null, totalTokens: null, thinkingMs: null,
    prefillMs: null, limitHitInThinking: null, errorMessage: null, connectionDropped: false });
  assert.equal(t.readDone(null).finishReason, null);
  for (const r of ["stop", "length", "user", "error"]) assert.equal(t.readDone({ finish_reason: r }).finishReason, r);
});

test("readDone: tolerated aliases (cancelled, top-level thinking_tokens / completion_tokens)", () => {
  assert.equal(t.readDone({ finish_reason: "cancelled" }).finishReason, "user");
  assert.equal(t.readDone({ choices: [{ finish_reason: "cancelled" }] }).finishReason, "user");
  const a = t.readDone({ finish_reason: "stop", thinking_tokens: 300, completion_tokens: 410 }, 512);
  assert.deepEqual([a.thinkingTokens, a.totalTokens, a.maxTokens], [300, 410, 512]);
  // The primary shape wins over aliases.
  const b = t.readDone({ tokens: { thinking: 418, total: 512 }, thinking_tokens: 1, completion_tokens: 2, usage: { completion_tokens: 3 } });
  assert.deepEqual([b.thinkingTokens, b.totalTokens], [418, 512]);
  assert.equal(t.noticeText(t.readDone({ finish_reason: "cancelled" }).finishReason, {}), "Stopped by you.");
});

test("header uses server thinking_ms when given (prefill excluded)", () => {
  const done = t.readDone({ finish_reason: "stop", thinking_ms: 11600, tokens: { thinking: 418 } });
  const browserMs = 14900; // includes prefill
  assert.equal(t.headerText({ elapsedMs: done.thinkingMs != null ? done.thinkingMs : browserMs, state: "closed", thinkingTokens: done.thinkingTokens }),
    "Thought for 12s \u00b7 418 tokens");
  const old = t.readDone({ finish_reason: "stop" });
  assert.equal(t.headerText({ elapsedMs: old.thinkingMs != null ? old.thinkingMs : browserMs, state: "closed", thinkingTokens: old.thinkingTokens }),
    "Thought for 15s");
});

test("copy excludes thinking", () => {
  const raw = "secret plan\n</think>\nThe answer.";
  assert.equal(t.copyText(raw, NM, "stop"), "The answer.");
  assert.ok(!t.copyText(raw, NM, "stop").includes("secret"));
  assert.equal(t.copyText("<think>plan</think>Hi", OTHER, "stop"), "Hi");
  assert.equal(t.copyText("No thinking here.", OTHER, "stop"), "No thinking here.");
  assert.equal(t.copyText("cut off mid thought", NM, "length"), "");
  const sel = t.joinCopySegments([
    { kind: "chrome", text: "ASSISTANT" },
    { kind: "thinking", text: "\u25b8Thought for 3s\nsecret plan" },
    { kind: "answer", text: "The answer." },
    { kind: "chrome", text: "Stopped at the 128-token limit." }
  ]);
  assert.equal(sel, "The answer.");
});

test("thinking body sticks to the bottom unless scrolled up", () => {
  assert.equal(t.atBottom(760, 240, 1000), true);
  assert.equal(t.atBottom(757, 240, 1000), true);
  assert.equal(t.atBottom(500, 240, 1000), false);
});

test("implicit opener: template_opens_think when reported, else Nemotron name match", () => {
  assert.equal(t.implicitOpenFor("NVIDIA-Nemotron3-Nano-4B-Q4_K_M"), true);
  assert.equal(t.implicitOpenFor("tiny-vicuna-1b.q4_k_m"), false);
  assert.equal(t.implicitOpenFor(""), false);
  assert.equal(t.implicitOpenFor("tiny-vicuna-1b.q4_k_m", true), true);
  assert.equal(t.implicitOpenFor("NVIDIA-Nemotron3-Nano-4B-Q4_K_M", false), false);
  assert.equal(t.implicitOpenFor("NVIDIA-Nemotron3-Nano-4B-Q4_K_M", null), true);
});

// Final chunks copied from a recorded server API capture.
test("recorded final chunk: Nemotron cap mid-think final chunk", () => {
  const d = t.readDone(JSON.parse('{"id":"chatcmpl-1eb98fdc","object":"chat.completion.chunk","created":1792615761,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":28,"completion_tokens":3,"total_tokens":31},"timings":{"prefill_tokens":28,"prefill_ms":2937.546,"decode_tokens":3,"decode_ms":230.658,"prefill_tok_s":9.532,"decode_tok_s":13.006},"finish_reason":"length","max_tokens":3,"tokens":{"thinking":3,"answer":0,"total":3},"thinking_ms":230.658,"prefill_ms":2937.546,"limit_hit_in_thinking":true}'));
  assert.deepEqual([d.finishReason, d.maxTokens, d.thinkingTokens, d.totalTokens, d.thinkingMs, d.prefillMs, d.limitHitInThinking],
    ["length", 3, 3, 3, 230.658, 2937.546, true]);
  assert.equal(t.headerText({ elapsedMs: d.thinkingMs, state: "open", finishReason: d.finishReason, maxTokens: d.maxTokens, limitHitInThinking: d.limitHitInThinking }),
    "Thinking stopped at the 3-token limit");
  assert.equal(t.noticeText(d.finishReason, { maxTokens: d.maxTokens, answered: !t.limitInThinking(d.finishReason, d.limitHitInThinking, "open") }),
    "Stopped at the 3-token limit before an answer.");
});

test("recorded final chunk: error final chunk carries the message", () => {
  const d = t.readDone(JSON.parse('{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":null}],"finish_reason":"error","max_tokens":8,"tokens":{"thinking":0,"answer":0,"total":0},"thinking_ms":0.0,"prefill_ms":0.0,"limit_hit_in_thinking":false,"error":{"message":"prompt too long: 9000 > 4096","type":"server_error"}}'));
  assert.equal(d.finishReason, "error");
  assert.equal(d.errorMessage, "prompt too long: 9000 > 4096");
  assert.equal(t.noticeText("error", {}), null); // errors use the existing error turn, not the notice
});

test("recorded final chunk: tiny-vicuna normal stop (non-stream body shape)", () => {
  const d = t.readDone(JSON.parse('{"id":"chatcmpl-b7d3","object":"chat.completion","created":1792615756,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":" The sky is blue."},"finish_reason":"stop"}],"usage":{"prompt_tokens":55,"completion_tokens":6,"total_tokens":61},"finish_reason":"stop","max_tokens":64,"tokens":{"thinking":0,"answer":6,"total":6},"thinking_ms":0.0,"prefill_ms":3636.995,"limit_hit_in_thinking":false}'));
  assert.deepEqual([d.finishReason, d.thinkingTokens, d.totalTokens, d.limitHitInThinking], ["stop", 0, 6, false]);
  assert.equal(t.noticeText(d.finishReason, { maxTokens: d.maxTokens, answered: true }), null);
});

test("limit in thinking: server flag wins over the parse", () => {
  assert.equal(t.limitInThinking("length", true, "closed"), true);
  assert.equal(t.limitInThinking("length", false, "open"), false);
  assert.equal(t.limitInThinking("length", null, "open"), true);
  assert.equal(t.limitInThinking("length", null, "closed"), false);
  assert.equal(t.limitInThinking("stop", true, "open"), false);
});

test("user stop: server final chunk after cancel", () => {
  const d = t.readDone({ finish_reason: "user", choices: [{ finish_reason: "stop" }], max_tokens: 128, tokens: { thinking: 40, answer: 12, total: 52 }, thinking_ms: 2100 });
  assert.equal(d.finishReason, "user");
  assert.equal(t.noticeText(d.finishReason, {}), "Stopped by you.");
  assert.equal(t.headerText({ elapsedMs: d.thinkingMs, state: "closed", finishReason: "user", thinkingTokens: d.thinkingTokens }), "Thought for 2s \u00b7 40 tokens");
  // Stream cut after the grace period: no counts, still a user stop.
  const cut = t.readDone({ finish_reason: "user", connection_dropped: true });
  assert.deepEqual([cut.finishReason, cut.totalTokens, cut.connectionDropped], ["user", null, true]);
});

test("Stop pending: 'Stopping…' until the server confirms, then 'Stopped by you.'", () => {
  assert.equal(t.noticeText("user", { pending: true }), "Stopping\u2026");
  assert.equal(t.noticeText("user", { pending: true, beforeReply: true }), "Stopping\u2026");
  assert.equal(t.noticeText("user", {}), "Stopped by you.");
  // pending only applies to a user stop
  assert.equal(t.noticeText("length", { pending: true, maxTokens: 1024, answered: true }), "Stopped at the 1,024-token limit.");
});

test("Stop during prefill: 'Stopped by you before the reply started.'", () => {
  assert.equal(t.noticeText("user", { beforeReply: true }), "Stopped by you before the reply started.");
  // Server total decides when sent; else whether any text arrived.
  assert.equal(t.stoppedBeforeReply(0, "anything"), true);
  assert.equal(t.stoppedBeforeReply(12, ""), false);
  assert.equal(t.stoppedBeforeReply(null, ""), true);
  assert.equal(t.stoppedBeforeReply(null, "  \n"), true);
  assert.equal(t.stoppedBeforeReply(null, "Sunlight"), false);
  // Mock prefill-stop final chunk end to end.
  const d = t.readDone({ finish_reason: "user", tokens: { thinking: 0, answer: 0, total: 0 }, prefill_ms: 4000 });
  assert.equal(t.noticeText(d.finishReason, { beforeReply: t.stoppedBeforeReply(d.totalTokens, "") }), "Stopped by you before the reply started.");
});

test("error notice: fixed main line, server message as detail", () => {
  assert.deepEqual(t.errorNotice("prompt too long: 9000 tokens > context 4096"),
    { main: "Couldn\u2019t generate a reply.", detail: "prompt too long: 9000 tokens > context 4096" });
  assert.deepEqual(t.errorNotice("  "), { main: "Couldn\u2019t generate a reply.", detail: null });
  assert.deepEqual(t.errorNotice(undefined), { main: "Couldn\u2019t generate a reply.", detail: null });
  const d = t.readDone({ finish_reason: "error", error: { message: "prompt too long: 9000 tokens > context 4096", type: "server_error" } });
  assert.equal(t.errorNotice(d.errorMessage).detail, "prompt too long: 9000 tokens > context 4096");
  assert.equal(t.noticeText("error", {}), null); // errors go through errorNotice, not noticeText
});

test("copy leaves out the error notice and its detail", () => {
  assert.equal(t.joinCopySegments([
    { kind: "answer", text: "" },
    { kind: "chrome", text: "Couldn\u2019t generate a reply.prompt too long" }
  ]), "");
});
