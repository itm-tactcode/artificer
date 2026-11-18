// node --test desktop/tests/  (no dependencies)
// View states of the first-launch empty state (design addendum empty-state-r1/SPEC.md).
const test = require("node:test");
const assert = require("node:assert/strict");
const s = require("../dist/app-state.js");

const SERVER = { endpoint: "127.0.0.1:8080", reused: true };
const PATH = "C:\\Users\\you\\models\\my-model-Q4_K_M.gguf";

/* Words no state may show: no compute backend, no CLI command, no picker tool, no bare "unreachable". */
const BANNED = [/\bcpu\b/i, /\bcuda\b/i, /\bgpu\b/i, /artificer serve/i, /\bserve\b/i, /zenity/i, /portal/i, /powershell/i, /osascript/i, /unreachable/i, /:\d{2,5}\b/];

function allStrings(v) {
  return [v.model, v.runtime, v.endpoint, v.note, v.weightsValue, v.weightsStatus, v.pathError, v.pathErrorDetail, v.line, v.sub, v.detail, v.placeholder]
    .filter((x) => typeof x === "string");
}

test("state 1 no-model: SPEC strings, CHANGE primary, path row shown, composer off", () => {
  const v = s.viewFor("no-model");
  assert.equal(v.pipTone, "idle");
  assert.equal(v.model, "model: none");
  assert.equal(v.runtime, "runtime: not started");
  assert.equal(v.endpoint, null);
  assert.equal(v.note, null);
  assert.equal(v.weightsValue, "No model selected");
  assert.equal(v.weightsEmpty, true);
  assert.equal(v.backendBadge, false);
  assert.equal(v.changeEmphasis, "primary");
  assert.equal(v.changeDisabled, false);
  assert.equal(v.pathRow, true);
  assert.equal(v.pathInvalid, false);
  assert.equal(v.pathError, null);
  assert.equal(v.line, "Choose a .gguf model file with {CHANGE} to start.");
  assert.equal(v.sub, "Artificer runs models on this computer. It doesn\u2019t download any.");
  assert.equal(v.retry, false);
  assert.equal(v.composerDisabled, true);
  assert.equal(v.sendDisabled, true);
  assert.equal(v.placeholder, "Load a model to start chatting");
});

test("state 1 with a server already up (no model): runtime ready, endpoint shown, still no-model", () => {
  const v = s.viewFor("no-model", { server: SERVER });
  assert.equal(v.runtime, "runtime: ready");
  assert.equal(v.endpoint, "endpoint: 127.0.0.1:8080");
  assert.equal(v.pipTone, "idle");
  assert.equal(v.changeEmphasis, "primary");
  assert.equal(v.composerDisabled, true);
});

test("state 2 starting: busy pip, loading strings, everything disabled", () => {
  const v = s.viewFor("starting", { path: PATH });
  assert.equal(v.pipTone, "busy");
  assert.equal(v.model, "model: loading\u2026");
  assert.equal(v.runtime, "runtime: starting\u2026");
  assert.equal(v.endpoint, null);
  assert.equal(v.weightsValue, PATH);
  assert.equal(v.weightsStatus, "Loading\u2026");
  assert.equal(v.changeDisabled, true);
  assert.equal(v.changeEmphasis, null);
  assert.equal(v.pathRow, false);
  assert.equal(v.line, "Starting runtime\u2026 Loading my-model-Q4_K_M.gguf");
  assert.equal(v.sub, "Large files can take a while.");
  assert.equal(v.composerDisabled, true);
  assert.equal(v.sendDisabled, true);
});

test("state 2 when the runtime is already up drops 'Starting runtime…'", () => {
  const v = s.viewFor("starting", { path: "/home/you/models/a.gguf", runtimeUp: true });
  assert.equal(v.line, "Loading a.gguf");
  assert.equal(v.runtime, "runtime: ready");
});

test("state 3 ready: model id, endpoint from the server, composer on", () => {
  const v = s.viewFor("ready", { model: "my-model-Q4_K_M", path: PATH, server: { endpoint: "127.0.0.1:41234", reused: false } });
  assert.equal(v.pipTone, "ok");
  assert.equal(v.model, "model: my-model-Q4_K_M");
  assert.equal(v.runtime, "runtime: ready");
  assert.equal(v.endpoint, "endpoint: 127.0.0.1:41234");
  assert.equal(v.note, null);
  assert.equal(v.weightsValue, PATH);
  assert.equal(v.weightsEmpty, false);
  assert.equal(v.backendBadge, true);
  assert.equal(v.changeEmphasis, null);
  assert.equal(v.pathRow, false);
  assert.equal(v.line, "Ask anything. Replies are generated on this computer.");
  assert.equal(v.sub, null);
  assert.equal(v.composerDisabled, false);
  assert.equal(v.sendDisabled, false);
  assert.equal(v.placeholder, "Message Artificer");
  assert.equal(v.focusComposer, true);
});

test("state 3 with no endpoint reported keeps the endpoint hidden (no invented port)", () => {
  const v = s.viewFor("ready", { model: "m", server: { endpoint: null, reused: false } });
  assert.equal(v.endpoint, null);
});

test("state 4 runtime-failed: warn tone, verbatim error, RETRY, composer off", () => {
  const err = "failed to bind 127.0.0.1: address in use";
  const v = s.viewFor("runtime-failed", { error: err, path: PATH });
  assert.equal(v.pipTone, "warn");
  assert.equal(v.runtime, "runtime: " + err);
  assert.equal(v.runtimeTone, "warn");
  assert.equal(v.model, "model: none");
  assert.equal(v.endpoint, null);
  assert.equal(v.weightsValue, PATH);
  assert.equal(v.changeEmphasis, null);
  assert.equal(v.changeDisabled, false);
  assert.equal(v.pathRow, true);
  assert.equal(v.line, "The runtime didn\u2019t start.");
  assert.equal(v.detail, err);
  assert.equal(v.retry, true);
  assert.equal(v.composerDisabled, true);
  assert.equal(v.sendDisabled, true);
});

test("state 4 never shows an empty runtime line", () => {
  const v = s.viewFor("runtime-failed", { error: "" });
  assert.notEqual(v.runtime, "runtime: ");
  assert.equal(v.detail, "No error message was reported.");
});

test("state 5 load-failed: inline under the path box, failed path kept, runtime ready", () => {
  const err = "POST /v1/models/load: 400 unsupported tensor type";
  const v = s.viewFor("load-failed", { error: err, path: PATH, server: SERVER });
  assert.equal(v.pipTone, "idle");
  assert.equal(v.model, "model: none");
  assert.equal(v.runtime, "runtime: ready");
  assert.equal(v.endpoint, "endpoint: 127.0.0.1:8080");
  assert.equal(v.weightsValue, "No model selected");
  assert.equal(v.changeEmphasis, "primary");
  assert.equal(v.pathRow, true);
  assert.equal(v.pathInvalid, true);
  assert.equal(v.pathValue, PATH);
  assert.equal(v.pathError, "The runtime couldn\u2019t load this file.");
  assert.equal(v.pathErrorDetail, err);
  assert.equal(v.line, "Choose a .gguf model file with {CHANGE} to start.");
  assert.equal(v.composerDisabled, true);
});

test("state 6 picker-unavailable: inline hint, path box invalid, CHANGE still primary", () => {
  const v = s.viewFor("picker-unavailable");
  assert.equal(v.runtime, "runtime: not started");
  assert.equal(v.pathRow, true);
  assert.equal(v.pathInvalid, true);
  assert.equal(v.pathError, "No file picker is available on this system. Paste the path to a .gguf file instead.");
  assert.equal(v.pathErrorDetail, null);
  assert.equal(v.changeEmphasis, "primary");
  assert.equal(v.composerDisabled, true);
  assert.equal(s.PICKER_UNAVAILABLE, "picker-unavailable");
});

test("state 7 reused: ready plus the note", () => {
  const v = s.viewFor("reused", { model: "their-model-Q8_0", path: "/m/their-model-Q8_0.gguf", server: SERVER });
  assert.equal(v.pipTone, "ok");
  assert.equal(v.model, "model: their-model-Q8_0");
  assert.equal(v.endpoint, "endpoint: 127.0.0.1:8080");
  assert.equal(v.note, "Connected to your running server");
  assert.equal(v.composerDisabled, false);
  assert.equal(v.sendDisabled, false);
});

test("path-invalid: one line per cause, OS error text as the muted detail", () => {
  const v = s.viewFor("path-invalid", { error: "model path must be absolute", path: "model.gguf" });
  assert.equal(v.pathError, "That isn\u2019t a full path. Paste the full path to a .gguf file.");
  assert.equal(v.pathErrorDetail, null);
  const g = s.viewFor("path-invalid", { error: "only an existing .gguf file can be selected", path: "/m/a.bin" });
  assert.equal(g.pathError, "That isn\u2019t a .gguf file.");
  assert.equal(g.pathErrorDetail, null);
  const n = s.viewFor("path-invalid", { error: "no file at this path (No such file or directory (os error 2))", path: "/m/x.gguf" });
  assert.equal(n.pathError, "No file was found at this path.");
  assert.equal(n.pathErrorDetail, "No such file or directory (os error 2)");
  const w = s.pathProblem("no file at this path (The system cannot find the file specified. (os error 2))");
  assert.deepEqual(w, { line: "No file was found at this path.", detail: "The system cannot find the file specified. (os error 2)" });
  assert.deepEqual(s.pathProblem("the picked path is not valid UTF-8"), { line: "the picked path is not valid UTF-8", detail: null });
  // The typographic apostrophe, exactly.
  assert.ok(v.pathError.includes("isn\u2019t") && !v.pathError.includes("isn't"));
  assert.equal(v.pathInvalid, true);
  assert.equal(v.pathValue, "model.gguf");
  assert.equal(v.composerDisabled, true);
});

test("composer and SEND are enabled only in ready and reused", () => {
  for (const st of s.STATES) {
    const v = s.viewFor(st, { model: "m", path: PATH, server: SERVER, error: "e" });
    const on = st === "ready" || st === "reused";
    assert.equal(v.composerDisabled, !on, st);
    assert.equal(v.sendDisabled, !on, st);
    assert.equal(v.placeholder, on ? "Message Artificer" : "Load a model to start chatting", st);
  }
});

test("no state's copy names a backend, a CLI command, a picker tool, a fixed port or 'unreachable'", () => {
  for (const st of s.STATES) {
    // No server and no error text from outside: only the window's own copy.
    const v = s.viewFor(st, { model: "m", path: "/m/a.gguf" });
    for (const text of allStrings(v)) {
      for (const re of BANNED) assert.doesNotMatch(text, re, st + ": " + text);
    }
  }
});

test("helpers: file name, path errors, typed-path check", () => {
  assert.equal(s.fileName(PATH), "my-model-Q4_K_M.gguf");
  assert.equal(s.fileName("/home/you/models/b.gguf"), "b.gguf");
  assert.equal(s.isPathError("model path must be absolute"), true);
  assert.equal(s.isPathError("no file at this path (No such file or directory (os error 2))"), true);
  assert.equal(s.isPathError("only an existing .gguf file can be selected"), true);
  assert.equal(s.isPathError("POST /v1/models/load: 500"), false);
  assert.equal(s.typedPathProblem(""), "model path must be absolute");
  assert.equal(s.typedPathProblem("model.gguf"), "model path must be absolute");
  assert.equal(s.typedPathProblem("/m/model.bin"), "only an existing .gguf file can be selected");
  assert.equal(s.typedPathProblem("/m/model.gguf"), null);
  assert.equal(s.typedPathProblem("  C:\\m\\model.GGUF  "), null);
  assert.equal(s.typedPathProblem("C:/m/model.gguf"), null);
  assert.equal(s.typedPathProblem("\\\\nas\\models\\x.gguf"), null);
  assert.equal(s.errorText({ message: " boom " }), "boom");
});

test("CHANGE result: picker-unavailable renders state 6 pointing at the typed path box", () => {
  const r = s.pickResult(false, "picker-unavailable");
  assert.equal(r.state, "picker-unavailable");
  assert.equal(r.focusPath, true);
  const v = s.viewFor(r.state, r.ctx);
  assert.equal(v.pathRow, true);
  assert.equal(v.pathInvalid, true);
  assert.equal(v.pathError, "No file picker is available on this system. Paste the path to a .gguf file instead.");
  assert.equal(v.changeEmphasis, "primary");
  assert.equal(v.composerDisabled, true);
});

test("CHANGE result: a plain cancel renders nothing; a path loads; other errors are shown", () => {
  assert.equal(s.pickResult(true, null), null);
  assert.equal(s.pickResult(true, undefined), null);
  assert.equal(s.pickResult(true, ""), null);
  assert.deepEqual(s.pickResult(true, "/m/a.gguf"), { load: "/m/a.gguf" });
  const r = s.pickResult(false, "only an existing .gguf file can be selected");
  assert.equal(r.state, "path-invalid");
  assert.equal(s.viewFor(r.state, r.ctx).pathError, "That isn\u2019t a .gguf file.");
});

test("bridge string 'picker-unavailable' (client_bridge::PICKER_UNAVAILABLE) maps to state 6 exactly", () => {
  assert.equal(s.PICKER_UNAVAILABLE, "picker-unavailable");
  assert.equal(s.pickResult(false, "picker-unavailable").state, "picker-unavailable");
  // Tauri may hand the error over as a string or an Error-like object.
  assert.equal(s.pickResult(false, { message: "picker-unavailable" }).state, "picker-unavailable");
  // Near misses are not state 6.
  for (const near of ["Picker-unavailable", "picker unavailable", "picker-unavailable: x"]) {
    assert.equal(s.pickResult(false, near).state, "path-invalid", near);
  }
});

test("picker open: CHANGE and LOAD disabled, re-enabled on pick, cancel or error", () => {
  for (const st of ["no-model", "load-failed", "picker-unavailable", "path-invalid", "runtime-failed", "ready", "reused"]) {
    const v = s.viewFor(st, { model: "m", path: "/m/a.gguf", server: SERVER, error: "e" });
    assert.deepEqual(s.controlsFor(v, true), { changeDisabled: true, loadDisabled: true }, st + " open");
    const closed = s.controlsFor(v, false);
    assert.equal(closed.changeDisabled, false, st + " closed");
    assert.equal(closed.loadDisabled, false, st + " closed");
  }
  // After the picker closes, the state's own disabled flags still apply (loading keeps both off).
  const loading = s.viewFor("starting", { path: "/m/a.gguf" });
  assert.deepEqual(s.controlsFor(loading, false), { changeDisabled: true, loadDisabled: true });
});

test("backend badge: hidden until it has text, and only where the state shows it", () => {
  const ready = s.viewFor("ready", { model: "m", server: SERVER });
  assert.equal(s.badgeVisible(ready, ""), false);
  assert.equal(s.badgeVisible(ready, "  "), false);
  assert.equal(s.badgeVisible(ready, null), false);
  assert.equal(s.badgeVisible(ready, "Backend unknown"), true);
  for (const st of ["no-model", "starting", "runtime-failed", "load-failed", "picker-unavailable", "path-invalid"]) {
    assert.equal(s.badgeVisible(s.viewFor(st, {}), "GPU"), false, st);
  }
});
