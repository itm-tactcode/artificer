// node --test desktop/tests/  (no dependencies)
// Runtime (src/server.rs) results → window states: ensure errors and pushed events.
const test = require("node:test");
const assert = require("node:assert/strict");
const s = require("../dist/app-state.js");

const PATH = "/home/you/models/my-model-Q4_K_M.gguf";

test("ensure: load-failed prefix is state 5 with the runtime's message", () => {
  const f = s.ensureFailure("load-failed: invalid GGUF magic");
  assert.deepEqual(f, { state: "load-failed", error: "invalid GGUF magic" });
  assert.ok(s.STATES.indexOf("load-failed") >= 0);
});

test("ensure: runtime-failed prefix is state 4 (RETRY shown)", () => {
  const f = s.ensureFailure({ message: "runtime-failed: could not listen on a local port (Address already in use)" });
  assert.equal(f.state, "runtime-failed");
  assert.equal(f.error, "could not listen on a local port (Address already in use)");
  const v = s.viewFor("runtime-failed", { path: PATH, error: f.error });
  assert.equal(v.retry, true);
});

test("ensure: a panic or unexpected exit is state 4", () => {
  const f = s.ensureFailure("runtime-failed: the runtime stopped unexpectedly (boom)");
  assert.equal(f.state, "runtime-failed");
});

test("ensure: the path check's own lines stay path-invalid", () => {
  for (const line of [
    "model path must be absolute",
    "no file at this path (No such file or directory (os error 2))",
    "only an existing .gguf file can be selected"
  ]) {
    assert.deepEqual(s.ensureFailure(line), { state: "path-invalid", error: line });
  }
});

test("ensure: anything else is state 4, never swallowed", () => {
  assert.deepEqual(s.ensureFailure("IPC down"), { state: "runtime-failed", error: "IPC down" });
});

test("events: starting is state 2, ready only remembered, exited/failed are state 4", () => {
  assert.deepEqual(s.runtimeEventState({ kind: "starting" }, PATH), {
    state: "starting",
    ctx: { path: PATH, runtimeUp: false }
  });
  assert.equal(s.runtimeEventState({ kind: "ready", endpoint: "x" }, PATH), null);
  assert.equal(s.runtimeEventState(null, PATH), null);
  for (const kind of ["exited", "failed"]) {
    const r = s.runtimeEventState({ kind: kind, error: "the runtime stopped unexpectedly (boom)" }, PATH);
    assert.equal(r.state, "runtime-failed");
    assert.equal(r.ctx.error, "the runtime stopped unexpectedly (boom)");
  }
});

test("events while the picker is open: CHANGE and LOAD stay disabled", () => {
  for (const ev of [{ kind: "starting" }, { kind: "exited", error: "x" }, { kind: "failed", error: "x" }]) {
    const r = s.runtimeEventState(ev, PATH);
    const controls = s.controlsFor(s.viewFor(r.state, r.ctx), true);
    assert.equal(controls.changeDisabled, true, ev.kind);
    assert.equal(controls.loadDisabled, true, ev.kind);
  }
});
