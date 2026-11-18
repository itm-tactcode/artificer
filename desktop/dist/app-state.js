/* Artificer window view states (first-launch empty state, design addendum empty-state-r1).
   Pure: viewFor(state, ctx) returns every string and on/off flag the window shows for one state;
   app.js applies it to the DOM. Loaded by index.html before app.js (window.ArtificerAppState)
   and by desktop/tests/app-state.test.cjs under `node --test` (module.exports).

   States (SPEC.md numbering):
     "no-model"           1  no model loaded (also the resting markup in index.html)
     "starting"           2  starting the runtime and/or loading a file
     "ready"              3  model loaded, empty transcript
     "runtime-failed"     4  the runtime did not start (error + RETRY)
     "load-failed"        5  the runtime could not load the file (inline under the path box)
     "picker-unavailable" 6  no file picker on this system (inline under the path box)
     "reused"             7  ready, on a server that was already running
     "path-invalid"          not in the mock: the typed or picked path is not an existing
                             absolute .gguf (checked before the runtime sees it); as 6, with one
                             line per cause and the OS error text as the muted detail (like 5).

   ctx: { server: { endpoint, reused } | null, model, path, error, runtimeUp }.
   Copy never names a compute backend, a fixed port, a CLI command or a picker tool. */
(function (root) {
  var NO_MODEL_LINE = "Choose a .gguf model file with {CHANGE} to start.";
  var NO_MODEL_SUB = "Artificer runs models on this computer. It doesn\u2019t download any.";
  var READY_LINE = "Ask anything. Replies are generated on this computer.";
  var COMPOSER_OFF = "Load a model to start chatting";
  var COMPOSER_ON = "Message Artificer";
  var NO_MODEL_SELECTED = "No model selected";
  var PICKER_UNAVAILABLE_TEXT =
    "No file picker is available on this system. Paste the path to a .gguf file instead.";
  var LOAD_FAILED_TEXT = "The runtime couldn\u2019t load this file.";
  var RUNTIME_FAILED_LINE = "The runtime didn\u2019t start.";
  var REUSED_NOTE = "Connected to your running server";
  var PATH_NOT_FULL = "That isn\u2019t a full path. Paste the full path to a .gguf file.";
  var PATH_NOT_GGUF = "That isn\u2019t a .gguf file.";
  var PATH_NOT_FOUND = "No file was found at this path.";
  /* The bridge's own error text for "no picker on this system" (client_bridge::PICKER_UNAVAILABLE). */
  var PICKER_UNAVAILABLE = "picker-unavailable";
  /* Prefixes of the bridge's path checks (client_bridge::checked_weights_path). */
  var PATH_ERROR_PREFIXES = [
    "model path must be absolute",
    "no file at this path",
    "only an existing .gguf file can be selected"
  ];
  var STATES = [
    "no-model", "starting", "ready", "runtime-failed", "load-failed",
    "picker-unavailable", "reused", "path-invalid"
  ];

  function fileName(path) {
    var s = String(path || "");
    var parts = s.split(/[\\/]/);
    return parts[parts.length - 1] || s;
  }

  function errorText(err) {
    var s = typeof err === "string" ? err : err && err.message ? String(err.message) : "";
    s = s.replace(/^\s+|\s+$/g, "");
    return s || "No error message was reported.";
  }

  function isPathError(err) {
    var s = typeof err === "string" ? err : "";
    for (var i = 0; i < PATH_ERROR_PREFIXES.length; i++) {
      if (s.indexOf(PATH_ERROR_PREFIXES[i]) === 0) return true;
    }
    return false;
  }

  /* Quick check before anything is sent: a non-empty absolute path ending in .gguf
     (POSIX `/…`, Windows `C:\…` / `C:/…`, or a UNC `\\server\…`). The bridge checks again. */
  function typedPathProblem(text) {
    var s = String(text || "").replace(/^\s+|\s+$/g, "");
    if (!s) return "model path must be absolute";
    if (!/^(\/|[A-Za-z]:[\\/]|\\\\)/.test(s)) return "model path must be absolute";
    if (!/\.gguf$/i.test(s)) return "only an existing .gguf file can be selected";
    return null;
  }

  /* path-invalid copy: one line per cause, the OS error text (if any) as the muted detail. */
  function pathProblem(err) {
    var text = errorText(err);
    if (text.indexOf(PATH_ERROR_PREFIXES[0]) === 0) return { line: PATH_NOT_FULL, detail: null };
    if (text.indexOf(PATH_ERROR_PREFIXES[2]) === 0) return { line: PATH_NOT_GGUF, detail: null };
    if (text.indexOf(PATH_ERROR_PREFIXES[1]) === 0) {
      var os = text.slice(PATH_ERROR_PREFIXES[1].length).replace(/^\s*\(/, "").replace(/\)\s*$/, "");
      os = os.replace(/^\s+|\s+$/g, "");
      return { line: PATH_NOT_FOUND, detail: os || null };
    }
    return { line: text, detail: null };
  }

  /* CHANGE / LOAD enabled flags: the state's own, and both off while the picker is open. */
  function controlsFor(view, pickerOpen) {
    return {
      changeDisabled: !!(view.changeDisabled || pickerOpen),
      loadDisabled: !!(view.pathRowDisabled || pickerOpen)
    };
  }

  /* The backend badge shows only where the state allows it and once it has text. */
  function badgeVisible(view, text) {
    return !!(view.backendBadge && String(text || "").replace(/^\s+|\s+$/g, ""));
  }

  function serverBits(ctx, v) {
    var server = ctx.server || null;
    if (server) {
      v.runtime = "runtime: ready";
      v.endpoint = server.endpoint ? "endpoint: " + server.endpoint : null;
    }
  }

  /* No model loaded: states 1, 5, 6 and path-invalid share this base. */
  function noModelBase(ctx) {
    var v = base();
    serverBits(ctx, v);
    return v;
  }

  function base() {
    return {
      pipTone: "idle",
      model: "model: none",
      runtime: "runtime: not started",
      runtimeTone: null,
      endpoint: null, // null → the endpoint span is hidden (no server yet)
      note: null, // health note; null → hidden
      weightsValue: NO_MODEL_SELECTED,
      weightsEmpty: true,
      weightsStatus: null, // "Loading…" while loading; null → hidden
      backendBadge: false,
      changeEmphasis: "primary",
      changeDisabled: false,
      pathRow: true,
      pathRowDisabled: false,
      pathInvalid: false,
      pathValue: null, // null → leave the input as the user typed it
      pathError: null,
      pathErrorDetail: null,
      line: NO_MODEL_LINE, // {CHANGE} marks the inline CHANGE reference
      sub: NO_MODEL_SUB,
      detail: null,
      retry: false,
      composerDisabled: true,
      placeholder: COMPOSER_OFF,
      sendDisabled: true,
      focusComposer: false
    };
  }

  function viewFor(state, ctx) {
    ctx = ctx || {};
    var v;
    switch (state) {
      case "starting": {
        v = base();
        var up = !!(ctx.runtimeUp || ctx.server);
        v.pipTone = "busy";
        v.model = "model: loading\u2026";
        v.runtime = up ? "runtime: ready" : "runtime: starting\u2026";
        if (ctx.server && ctx.server.endpoint) v.endpoint = "endpoint: " + ctx.server.endpoint;
        v.weightsValue = ctx.path || "";
        v.weightsEmpty = !ctx.path;
        v.weightsStatus = "Loading\u2026";
        v.changeEmphasis = null;
        v.changeDisabled = true;
        v.pathRow = false;
        v.pathRowDisabled = true;
        var loading = "Loading " + fileName(ctx.path);
        v.line = up ? loading : "Starting runtime\u2026 " + loading;
        v.sub = "Large files can take a while.";
        return v;
      }
      case "ready":
      case "reused": {
        v = base();
        serverBits(ctx, v);
        v.pipTone = "ok";
        v.model = "model: " + (ctx.model || "");
        v.runtime = "runtime: ready";
        v.note = state === "reused" ? REUSED_NOTE : null;
        v.weightsValue = ctx.path || ctx.model || "";
        v.weightsEmpty = false;
        v.backendBadge = true;
        v.changeEmphasis = null;
        v.pathRow = false;
        v.line = READY_LINE;
        v.sub = null;
        v.composerDisabled = false;
        v.placeholder = COMPOSER_ON;
        v.sendDisabled = false;
        v.focusComposer = true;
        return v;
      }
      case "runtime-failed": {
        v = base();
        var err = errorText(ctx.error);
        v.pipTone = "warn";
        v.runtime = "runtime: " + err;
        v.runtimeTone = "warn";
        v.weightsValue = ctx.path || NO_MODEL_SELECTED;
        v.weightsEmpty = !ctx.path;
        v.changeEmphasis = null;
        v.line = RUNTIME_FAILED_LINE;
        v.sub = null;
        v.detail = err;
        v.retry = true;
        return v;
      }
      case "load-failed":
        v = noModelBase(ctx);
        v.pathInvalid = true;
        v.pathValue = ctx.path || "";
        v.pathError = LOAD_FAILED_TEXT;
        v.pathErrorDetail = errorText(ctx.error);
        return v;
      case "picker-unavailable":
        v = noModelBase(ctx);
        v.pathInvalid = true;
        v.pathError = PICKER_UNAVAILABLE_TEXT;
        return v;
      case "path-invalid":
        v = noModelBase(ctx);
        v.pathInvalid = true;
        if (ctx.path != null) v.pathValue = ctx.path;
        var problem = pathProblem(ctx.error);
        v.pathError = problem.line;
        v.pathErrorDetail = problem.detail;
        return v;
      case "no-model":
      default:
        return noModelBase(ctx);
    }
  }

  /* What one CHANGE click ends in, from pick_local_weights' result:
       resolved path → { load: path }   (load it)
       resolved null → null             (a real Cancel: the window changes nothing, shows nothing)
       "picker-unavailable" rejection → state 6 (no file picker; points at the typed path box)
       any other rejection → path-invalid with the bridge's reason (never swallowed). */
  function pickResult(ok, value) {
    if (ok) {
      if (value == null || value === "") return null;
      return { load: String(value) };
    }
    var text = errorText(value);
    if (text === PICKER_UNAVAILABLE) return { state: "picker-unavailable", ctx: {}, focusPath: true };
    return { state: "path-invalid", ctx: { error: text } };
  }

  /* runtime_ensure rejection → the state it means. The bridge prefixes its own errors:
       "load-failed: " (the runtime could not load the file, state 5) and "runtime-failed: " (the
       runtime did not start or stopped, state 4 with RETRY). A path-check line is path-invalid. */
  var LOAD_FAILED = "load-failed: ";
  var RUNTIME_FAILED = "runtime-failed: ";
  function ensureFailure(err) {
    var text = errorText(err);
    if (text.indexOf(LOAD_FAILED) === 0) return { state: "load-failed", error: text.slice(LOAD_FAILED.length) };
    if (text.indexOf(RUNTIME_FAILED) === 0) return { state: "runtime-failed", error: text.slice(RUNTIME_FAILED.length) };
    if (isPathError(text)) return { state: "path-invalid", error: text };
    return { state: "runtime-failed", error: text };
  }

  /* A server event → { state, ctx } to show, or null (only remembered: "ready").
     "starting" → state 2; "exited" / "failed" → state 4 (RETRY). */
  function runtimeEventState(ev, path) {
    if (!ev) return null;
    if (ev.kind === "starting") return { state: "starting", ctx: { path: path, runtimeUp: false } };
    if (ev.kind === "failed" || ev.kind === "exited") return { state: "runtime-failed", ctx: { path: path, error: ev.error } };
    return null;
  }

  /* `client_caps` → { stream, loaded_model }. Read at launch whether or not a server is running
     (a cold launch has none until a model is loaded). Missing or failed → no streaming. */
  function readCaps(reported) {
    var c = reported && typeof reported === "object" ? reported : {};
    return { stream: c.stream === true, loaded_model: c.loaded_model === true };
  }

  /* Chat history sent with each turn, oldest first, the new user message last:
     [{ role: "user" | "assistant", content }]. Assistant turns hold the answer only, never the
     thinking. Each call returns a new list. */
  function historyAdd(turns, text) {
    return (turns || []).concat([{ role: "user", content: String(text) }]);
  }

  /* A reply ended. answer: the reply without its thinking (replyAnswer). Empty (a failed send,
     a Stop before any answer, the limit hit while thinking) drops the user turn too, so the next
     send never carries a question without its answer. */
  function historyFinish(turns, answer) {
    var list = (turns || []).slice();
    if (!list.length || list[list.length - 1].role !== "user") return list;
    var text = String(answer == null ? "" : answer).replace(/^\s+|\s+$/g, "");
    if (text) list.push({ role: "assistant", content: text });
    else list.pop();
    return list;
  }

  /* A reply that failed because the chat no longer fits the model's context: the runtime's
     "context length exceeded: need N, max M" (not the request check's "prompt too long").
     The history is cleared and the error notice says so. */
  var CONTEXT_OVERFLOW = "context length exceeded";
  var CONTEXT_OVERFLOW_NOTICE =
    "This conversation got too long for the model, so it was cleared. Send your message again to start fresh.";
  function isContextOverflow(text) {
    return typeof text === "string" && text.indexOf(CONTEXT_OVERFLOW) !== -1;
  }

  /* The answer part of a finished reply (thinking.js finalSplit), the whole text without T. */
  function replyAnswer(T, text, implicitOpen, finishReason) {
    if (!T) return String(text == null ? "" : text);
    return T.finalSplit(text, { implicitOpen: !!implicitOpen }, finishReason).answer;
  }

  var api = {
    STATES: STATES,
    readCaps: readCaps,
    historyAdd: historyAdd,
    historyFinish: historyFinish,
    replyAnswer: replyAnswer,
    isContextOverflow: isContextOverflow,
    CONTEXT_OVERFLOW_NOTICE: CONTEXT_OVERFLOW_NOTICE,
    ensureFailure: ensureFailure,
    runtimeEventState: runtimeEventState,
    pickResult: pickResult,
    pathProblem: pathProblem,
    controlsFor: controlsFor,
    badgeVisible: badgeVisible,
    PICKER_UNAVAILABLE: PICKER_UNAVAILABLE,
    viewFor: viewFor,
    fileName: fileName,
    errorText: errorText,
    isPathError: isPathError,
    typedPathProblem: typedPathProblem
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ArtificerAppState = api;
})(typeof window !== "undefined" ? window : this);
