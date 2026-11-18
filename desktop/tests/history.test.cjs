// node --test desktop/tests/  (no dependencies)
// Chat history sent with each turn and the launch caps (app-state.js).
const test = require("node:test");
const assert = require("node:assert/strict");
const s = require("../dist/app-state.js");
const T = require("../dist/thinking.js");

test("history: turns stay in order, the new user message last", () => {
  let h = s.historyAdd([], "hi");
  h = s.historyFinish(h, "hello");
  h = s.historyAdd(h, "thank you");
  assert.deepEqual(h, [
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
    { role: "user", content: "thank you" }
  ]);
});

test("history: helpers return new lists", () => {
  const a = s.historyAdd([], "hi");
  const b = s.historyFinish(a, "hello");
  assert.equal(a.length, 1);
  assert.equal(b.length, 2);
});

test("history: the assistant turn is the answer only, never the thinking", () => {
  const reply = "reasoning...\n</think>\nanswer";
  const answer = s.replyAnswer(T, reply, true, "stop");
  assert.equal(answer, "answer");
  const h = s.historyFinish(s.historyAdd([], "hi"), answer);
  assert.deepEqual(h[1], { role: "assistant", content: "answer" });
  assert.equal(JSON.stringify(h).indexOf("reasoning"), -1);
  // An explicit <think> block on a model whose template does not open it.
  assert.equal(s.replyAnswer(T, "<think>plan</think>\nok", false, "stop"), "ok");
  // A reply with no thinking is sent back whole.
  assert.equal(s.replyAnswer(T, "plain reply", false, null), "plain reply");
  assert.equal(s.replyAnswer(null, "a</think>b", false, null), "a</think>b");
});

test("history: a failed send drops its user turn", () => {
  const before = s.historyFinish(s.historyAdd([], "hi"), "hello");
  const after = s.historyFinish(s.historyAdd(before, "again"), "");
  assert.deepEqual(after, before);
  assert.deepEqual(s.historyFinish(s.historyAdd([], "x"), null), []);
});

test("history: a Stop keeps a partial answer, else drops the pair", () => {
  const partial = s.replyAnswer(T, "think\n</think>\npart", true, "user");
  assert.deepEqual(s.historyFinish(s.historyAdd([], "q"), partial), [
    { role: "user", content: "q" },
    { role: "assistant", content: "part" }
  ]);
  // Stopped while still thinking: no answer, nothing kept.
  const none = s.replyAnswer(T, "still thinking", true, "user");
  assert.equal(none, "");
  assert.deepEqual(s.historyFinish(s.historyAdd([], "q"), none), []);
  // Whitespace only counts as no answer.
  assert.deepEqual(s.historyFinish(s.historyAdd([], "q"), " \n "), []);
});

test("history: a finish with no pending user turn changes nothing", () => {
  // A new model cleared the list while a reply was still finishing.
  assert.deepEqual(s.historyFinish([], "late answer"), []);
});

test("caps: read from client_caps; missing or malformed is no streaming", () => {
  assert.deepEqual(s.readCaps({ stream: true, loaded_model: true }), { stream: true, loaded_model: true });
  assert.deepEqual(s.readCaps(null), { stream: false, loaded_model: false });
  assert.deepEqual(s.readCaps({ stream: "yes" }), { stream: false, loaded_model: false });
});
