// node --test desktop/tests/  (no dependencies)
// The shipped window markup starts empty: no fixture turns, no prefilled composer, no fake health.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const html = fs.readFileSync(path.join(__dirname, "..", "dist", "index.html"), "utf8");

test("index.html has no fixture conversation or placeholder health values", () => {
  for (const bad of [
    "Run a local inference check",
    "the runtime responded",
    "still the one loaded",
    "local placeholder",
    "path/to/your-model.gguf",
    "runtime: ready",
    "Backend unknown",
    "unreachable"
  ]) {
    assert.equal(html.includes(bad), false, bad);
  }
  assert.doesNotMatch(html, /class="turn\b/, "no resting turn articles");
});

test("composer textarea is empty, disabled, with the SPEC placeholder; Send disabled", () => {
  const m = html.match(/<textarea([^>]*)>([\s\S]*?)<\/textarea>/);
  assert.ok(m, "textarea present");
  assert.equal(m[2].trim(), "", "no prefilled text");
  assert.match(m[1], /\bdisabled\b/);
  assert.match(m[1], /placeholder="Load a model to start chatting"/);
  assert.match(html, /<button type="button" class="composer-send" disabled>Send<\/button>/);
});

test("resting markup is state 1 and loads app-state.js before app.js", () => {
  assert.match(html, /<span data-health-model>model: none<\/span>/);
  assert.match(html, /<span data-health-runtime>runtime: not started<\/span>/);
  assert.match(html, /<span data-health-endpoint hidden><\/span>/);
  assert.match(html, /data-weights-path data-empty>No model selected</);
  assert.match(html, /class="weights-change" data-emphasis="primary">Change</);
  assert.match(html, /<label class="weights-pathrow-label" for="weights-path-input">or paste a path<\/label>/);
  assert.match(html, /id="weights-path-input"[^>]*placeholder="Full path to a \.gguf file"/);
  assert.match(html, /class="weights-load">Load</);
  assert.match(html, /Choose a \.gguf model file with <span class="ref-change">Change<\/span> to start\./);
  assert.match(html, /<link rel="stylesheet" href="empty-state\.css">/);
  assert.ok(html.indexOf('src="app-state.js"') < html.indexOf('src="app.js"'));
});

test("no copy tells the user to run a server or names a picker tool", () => {
  for (const re of [/artificer serve/i, /\bserve\b/i, /zenity/i, /\bcuda\b/i, /\bcpu\b/i]) {
    assert.doesNotMatch(html, re);
  }
});
