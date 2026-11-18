/* Artificer window. Styles are in desktop/dist/chrome.css and empty-state.css.
   Resting markup matches index.html (state 1, no model); app-state.js holds each state's strings.
   The still-arriving assistant bubble (caret plus "Still arriving" label) is created only when
   chat-stream delivers a chunk.
   No typewriter. No second HTTP client. */
(function () {
  var healthModel = document.querySelector("[data-health-model]");
  var healthRuntime = document.querySelector("[data-health-runtime]");
  var weightsPath = document.querySelector("[data-weights-path]");
  var transcript = document.querySelector(".transcript");
  var composer = document.querySelector(".composer");
  var draft = composer ? composer.querySelector("textarea") : null;
  var send = composer ? composer.querySelector(".composer-send") : null;
  var change = document.querySelector(".weights-change");
  var S = window.ArtificerAppState;
  if (!healthModel || !healthRuntime || !weightsPath || !transcript || !draft || !send || !change || !S) {
    return;
  }
  var healthPip = document.querySelector(".health-pip");
  var healthEndpoint = document.querySelector("[data-health-endpoint]");
  var healthNote = document.querySelector("[data-health-note]");
  var weightsStatus = document.querySelector("[data-weights-status]");
  var pathRow = document.querySelector("[data-weights-pathrow]");
  var pathInput = document.getElementById("weights-path-input");
  var pathLoad = document.querySelector(".weights-load");
  var emptyState = document.querySelector("[data-empty-state]");
  var emptyLine = document.querySelector("[data-empty-line]");
  var emptySub = document.querySelector("[data-empty-sub]");
  var emptyDetail = document.querySelector("[data-empty-detail]");
  var retry = document.querySelector("[data-empty-retry]");

  var tauri = window.__TAURI__;
  var invoke = tauri && tauri.core && tauri.core.invoke;
  var caps = { stream: false, loaded_model: false };
  var weightFiles = [];
  var launchModel = "";
  var templateOpensThink = null; // /v1/models `template_opens_think`, when the server reports it
  var weightsNote = document.querySelector("[data-weights-note]");
  var backendBadgeEl = document.querySelector("[data-backend-badge]");
  var M = window.ArtificerMetrics;
  var T = window.ArtificerThinking;
  var THINK_TICK_MS = 1000;
  var messageSeq = 0;
  /* Open/closed Thinking state per message id. Collapsed unless the user opened it. */
  var thinkingOpen = {};
  var RATE_TICK_MS = 500;
  var busy = false;
  var activeStream = null;
  var appState = "no-model";
  var appCtx = {};
  var chatEnabled = false; // composer + Send: only once a model is loaded
  var server = null; // { endpoint, reused } once a server answers /health
  var pendingPath = ""; // last picked or typed path, for RETRY
  var pickerOpen = false; // CHANGE and LOAD stay disabled while the native dialog is open
  var lastView = null; // the view setAppState applied last
  var history = []; // turns sent with each message (app-state.js historyAdd/historyFinish)

  /* Weights-row badge from /health `backend`. Missing → "Backend unknown", never CPU. */
  function showBackend(backend) {
    if (!backendBadgeEl || !M) return;
    var b = M.backendBadge(backend);
    backendBadgeEl.textContent = b.text;
    backendBadgeEl.setAttribute("data-tone", b.tone);
    backendBadgeEl.title = b.title;
    backendBadgeEl.hidden = !(lastView && S.badgeVisible(lastView, b.text));
    fitWeightsPath(); // badge width changed → room for the path changed
  }

  /* Weights path: full absolute path in data-full-path + title; shown middle-truncated
     (filename intact) to the longest form that fits the row. */
  function getWeightsPath() {
    return weightsPath.getAttribute("data-full-path") || weightsPath.textContent;
  }

  function fitWeightsPath() {
    var full = getWeightsPath();
    var forms = M && M.pathCandidates ? M.pathCandidates(full) : [full];
    for (var i = 0; i < forms.length; i++) {
      weightsPath.textContent = forms[i];
      if (weightsPath.scrollWidth <= weightsPath.clientWidth) return;
    }
  }

  function setWeightsPath(text) {
    weightsPath.setAttribute("data-full-path", text);
    weightsPath.title = text;
    fitWeightsPath();
  }
  window.addEventListener("resize", fitWeightsPath);

  function hasTurns() {
    return !!transcript.querySelector(".turn");
  }

  /* Empty-state line: "{CHANGE}" becomes the inline CHANGE reference (text only, not a button). */
  function setEmptyLine(text) {
    while (emptyLine.firstChild) emptyLine.removeChild(emptyLine.firstChild);
    var parts = String(text || "").split("{CHANGE}");
    for (var i = 0; i < parts.length; i++) {
      if (i > 0) {
        var ref = document.createElement("span");
        ref.className = "ref-change";
        ref.textContent = "Change";
        emptyLine.appendChild(ref);
      }
      if (parts[i]) emptyLine.appendChild(document.createTextNode(parts[i]));
    }
  }

  function showText(el, text) {
    if (!el) return;
    el.hidden = text == null || text === "";
    el.textContent = el.hidden ? "" : text;
  }

  /* One view state from app-state.js onto the DOM (states 1-7 of the empty-state-r1 addendum). */
  function setAppState(state, ctx) {
    appState = state;
    appCtx = ctx || {};
    var v = S.viewFor(state, { server: server, model: appCtx.model, path: appCtx.path, error: appCtx.error, runtimeUp: appCtx.runtimeUp });
    if (healthPip) healthPip.setAttribute("data-tone", v.pipTone);
    healthModel.textContent = v.model;
    healthRuntime.textContent = v.runtime;
    if (v.runtimeTone) healthRuntime.setAttribute("data-tone", v.runtimeTone);
    else healthRuntime.removeAttribute("data-tone");
    showText(healthEndpoint, v.endpoint);
    showText(healthNote, v.note);
    if (v.weightsEmpty) weightsPath.setAttribute("data-empty", "");
    else weightsPath.removeAttribute("data-empty");
    setWeightsPath(v.weightsValue);
    showText(weightsStatus, v.weightsStatus);
    lastView = v;
    if (backendBadgeEl) backendBadgeEl.hidden = !S.badgeVisible(v, backendBadgeEl.textContent);
    if (v.changeEmphasis) change.setAttribute("data-emphasis", v.changeEmphasis);
    else change.removeAttribute("data-emphasis");
    var controls = S.controlsFor(v, pickerOpen);
    change.disabled = controls.changeDisabled;
    if (pathRow) pathRow.hidden = !v.pathRow;
    if (pathInput) {
      pathInput.disabled = v.pathRowDisabled;
      if (v.pathValue != null) pathInput.value = v.pathValue;
      if (v.pathInvalid) {
        pathInput.setAttribute("aria-invalid", "true");
        pathInput.setAttribute("aria-describedby", "weights-error");
      } else {
        pathInput.removeAttribute("aria-invalid");
        pathInput.removeAttribute("aria-describedby");
      }
    }
    if (pathLoad) pathLoad.disabled = controls.loadDisabled;
    if (weightsNote) {
      weightsNote.hidden = !v.pathError;
      weightsNote.textContent = v.pathError || "";
      if (v.pathErrorDetail) {
        var detail = document.createElement("span");
        detail.className = "weights-error-detail";
        detail.textContent = v.pathErrorDetail;
        weightsNote.appendChild(detail);
      }
    }
    // The empty-state block shows only while the transcript has no turns.
    var empty = !hasTurns();
    if (emptyState) emptyState.hidden = !empty;
    if (empty) transcript.setAttribute("data-empty", "");
    else transcript.removeAttribute("data-empty");
    if (emptyLine) setEmptyLine(v.line);
    showText(emptySub, v.sub);
    showText(emptyDetail, v.detail);
    if (retry) retry.hidden = !v.retry;
    chatEnabled = !v.composerDisabled;
    draft.disabled = v.composerDisabled;
    if (v.composerDisabled || !hasTurns()) draft.placeholder = v.placeholder;
    if (!busy) send.disabled = v.sendDisabled;
    if (v.focusComposer && draft.focus) draft.focus();
  }

  function leaveEmptyState() {
    if (emptyState) emptyState.hidden = true;
    transcript.removeAttribute("data-empty");
  }

  function refreshBackend() {
    return invoke("check_health")
      .then(function (view) {
        showBackend(view && view.backend);
        return view;
      })
      .catch(function () {
        showBackend(null);
        return null;
      });
  }

  function setModel(id) {
    if (!id) return;
    launchModel = id;
    if (appState === "ready" || appState === "reused") healthModel.textContent = "model: " + id;
  }

  function scrollTurn(node) {
    if (node && node.scrollIntoView) node.scrollIntoView(false);
  }

  function appendUser(text) {
    var article = document.createElement("article");
    article.className = "turn turn-user";
    var role = document.createElement("span");
    role.className = "turn-role";
    role.textContent = "User";
    var p = document.createElement("p");
    p.textContent = text;
    article.appendChild(role);
    article.appendChild(p);
    leaveEmptyState();
    transcript.appendChild(article);
    scrollTurn(article);
  }

  function appendFinished(text) {
    var article = document.createElement("article");
    article.className = "turn turn-assistant";
    var role = document.createElement("span");
    role.className = "turn-role";
    role.textContent = "Assistant";
    var p = document.createElement("p");
    p.textContent = text;
    article.appendChild(role);
    article.appendChild(p);
    transcript.appendChild(article);
    scrollTurn(article);
    return article;
  }

  /* Streaming reply: assistant bubble, caret, "Still arriving" label. */
  function beginStream() {
    var article = document.createElement("article");
    article.className = "turn turn-assistant turn-streaming";
    article.setAttribute("aria-busy", "true");
    var role = document.createElement("span");
    role.className = "turn-role";
    role.textContent = "Assistant";
    var p = document.createElement("p");
    var answerNode = document.createTextNode("");
    p.appendChild(answerNode);
    var caret = document.createElement("span");
    caret.className = "caret";
    caret.setAttribute("aria-hidden", "true");
    p.appendChild(caret);
    var id = "m" + ++messageSeq;
    article.setAttribute("data-message-id", id);
    var thinking = buildThinking(id);
    var note = document.createElement("span");
    note.className = "stream-note stream-rate";
    note.setAttribute("data-stream-rate", "");
    note.setAttribute("aria-live", "off");
    note.textContent = "Still arriving";
    article.appendChild(role);
    article.appendChild(thinking.section);
    article.appendChild(p);
    article.appendChild(note);
    transcript.appendChild(article);
    scrollTurn(article);
    draft.value = "";
    draft.placeholder = "Follow-up";
    var view = {
      id: id, article: article, p: p, answerNode: answerNode, caret: caret, note: note, text: "",
      meter: null, tick: null, thinking: thinking, parse: null, implicitOpen: T ? T.implicitOpenFor(launchModel, templateOpensThink) : false,
      startMs: nowMs(), thinkEndMs: null, thinkTick: null, stopped: false, done: false
    };
    article.__view = view;
    if (T) {
      view.thinkTick = setInterval(function () { renderThinkingHeader(view); }, THINK_TICK_MS);
    }
    if (M) {
      view.meter = new M.RollingRate(2000);
      // Re-render the rolling average ~every 500 ms (not per delta) so the number is readable.
      view.tick = setInterval(function () {
        if (view.meter.count > 0) note.textContent = M.liveReadout(view.meter.rate());
      }, RATE_TICK_MS);
    }
    return view;
  }

  function nowMs() {
    return window.performance && performance.now ? performance.now() : Date.now();
  }

  function appendDelta(view, chunk) {
    if (!view || !chunk || view.frozen || view.done) return;
    view.text += chunk;
    if (view.meter) view.meter.push(nowMs()); // arrival times for tok/s only; never a token count
    renderReply(view, false, null);
    scrollTurn(view.article);
  }

  /* Thinking section: a real <button> header (Enter/Space are native), body below it.
     Hidden until the reply has thinking text. Collapsed by default. */
  function buildThinking(id) {
    var section = document.createElement("div");
    section.className = "thinking";
    section.hidden = true;
    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "thinking-toggle";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-controls", id + "-thinking");
    var chevron = document.createElement("span");
    chevron.className = "thinking-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "\u25b8";
    var label = document.createElement("span");
    label.className = "thinking-label";
    toggle.appendChild(chevron);
    toggle.appendChild(label);
    var body = document.createElement("div");
    body.className = "thinking-body";
    body.id = id + "-thinking";
    body.setAttribute("role", "region");
    body.setAttribute("aria-label", "Thinking");
    body.hidden = true;
    var t = { section: section, toggle: toggle, chevron: chevron, label: label, body: body, stick: true };
    body.addEventListener("scroll", function () {
      t.stick = T.atBottom(body.scrollTop, body.clientHeight, body.scrollHeight);
    });
    toggle.addEventListener("click", function () {
      setThinkingOpen(id, t, toggle.getAttribute("aria-expanded") !== "true");
    });
    section.appendChild(toggle);
    section.appendChild(body);
    setThinkingOpen(id, t, !!thinkingOpen[id]);
    return t;
  }

  function setThinkingOpen(id, t, open) {
    thinkingOpen[id] = open;
    t.toggle.setAttribute("aria-expanded", open ? "true" : "false");
    t.chevron.textContent = open ? "\u25be" : "\u25b8";
    t.body.hidden = !open;
    if (open && t.stick) t.body.scrollTop = t.body.scrollHeight;
  }

  function renderThinkingHeader(view) {
    if (!T || !view.parse || !view.parse.thinking) return;
    var end = view.thinkEndMs != null ? view.thinkEndMs : nowMs();
    var serverMs = view.finish ? view.finish.thinkingMs : null; // server-measured, prefill excluded
    view.thinking.label.textContent = T.headerText({
      streaming: !view.done && view.parse.state === "open",
      elapsedMs: serverMs != null ? serverMs : end - view.startMs,
      state: view.parse.state,
      finishReason: view.finish ? view.finish.finishReason : null,
      thinkingTokens: view.finish ? view.finish.thinkingTokens : null,
      maxTokens: view.finish ? view.finish.maxTokens : null,
      limitHitInThinking: view.finish ? view.finish.limitHitInThinking : null
    });
  }

  /* Re-split the whole reply and paint thinking + answer. final: stream has ended. */
  function renderReply(view, final, finishReason) {
    if (!T) {
      view.answerNode.data = view.text;
      return;
    }
    var opts = { implicitOpen: view.implicitOpen };
    var parse = final ? T.finalSplit(view.text, opts, finishReason) : T.split(view.text, opts);
    if (view.thinkEndMs == null && (parse.state === "closed" || final)) view.thinkEndMs = nowMs();
    view.parse = parse;
    var t = view.thinking;
    if (parse.thinking) {
      t.section.hidden = false;
      if (t.body.textContent !== parse.thinking) t.body.textContent = parse.thinking;
      if (t.stick && !t.body.hidden) t.body.scrollTop = t.body.scrollHeight;
    } else {
      t.section.hidden = true;
    }
    view.answerNode.data = parse.answer;
    renderThinkingHeader(view);
  }

  /* The one notice row under the reply (limit / Stop / error). detail: optional second
     line in muted mono (the server's error message). */
  function showNotice(view, text, reason, detail) {
    var old = view.article.querySelectorAll(".stream-notice");
    for (var i = 0; i < old.length; i++) old[i].remove();
    if (!text) return;
    var n = document.createElement("p");
    n.className = "stream-notice";
    n.setAttribute("data-reason", reason);
    n.setAttribute("role", reason === "error" ? "alert" : "status");
    n.appendChild(document.createTextNode(text));
    if (detail) {
      var d = document.createElement("span");
      d.className = "stream-notice-detail";
      d.textContent = detail;
      n.appendChild(d);
    }
    view.p.parentNode.insertBefore(n, view.p.nextSibling);
  }

  function showError(view, message) {
    var e = T ? T.errorNotice(message) : { main: "Couldn\u2019t generate a reply.", detail: message };
    if (S.isContextOverflow(message)) e.main = S.CONTEXT_OVERFLOW_NOTICE; // same notice, detail unchanged
    if (view.note.parentNode) view.note.remove(); // no readout on an error
    showNotice(view, e.main, "error", e.detail);
    scrollTurn(view.article);
  }

  function stopLive(view) {
    if (view.tick) clearInterval(view.tick);
    view.tick = null;
    if (view.thinkTick) clearInterval(view.thinkTick);
    view.thinkTick = null;
    view.article.classList.remove("turn-streaming");
    view.article.removeAttribute("aria-busy");
    if (view.caret.parentNode) view.caret.remove();
  }

  /* Stop pressed: freeze what the user sees now (caret gone, "Stopped by you."), ignore
     later deltas, and wait for the server's final chunk (finish_reason "user" + its
     counts) to fill in the header count and readout. Prefill can't be interrupted, so a
     Stop during prefill shows "Stopping…" until prefill ends (or the bridge's grace
     period cuts the stream). */
  function freezeForStop(view) {
    if (!view || view.frozen || view.done) return;
    view.frozen = true;
    view.stopped = true;
    stopLive(view);
    if (view.thinkEndMs == null) view.thinkEndMs = nowMs();
    view.finish = T ? T.readDone({ finish_reason: "user" }) : { finishReason: "user" };
    renderReply(view, true, "user");
    // "Stopping…" until the server confirms (final chunk) or the bridge's grace cut.
    if (T) showNotice(view, T.noticeText("user", { pending: true }), "user");
    if (view.note.parentNode) view.note.remove(); // no readout while pending
    scrollTurn(view.article);
  }

  function finishStream(view, content, timings, done) {
    if (!view || view.done) return;
    view.done = true;
    stopLive(view);
    var finish = T ? T.readDone(done) : { finishReason: null, totalTokens: null };
    if (view.stopped && !finish.finishReason) finish.finishReason = "user";
    if (finish.finishReason !== "error" && M && view.meter && (timings || view.meter.count > 0 || finish.totalTokens != null)) {
      // Freeze: server prefill/decode split when present, else the client rate.
      // Token count: server total only; none after a Stop unless the server sent one.
      view.note.textContent = M.finalReadout(timings, { rate: view.meter.rate(), totalTokens: finish.totalTokens });
      if (!view.note.parentNode) view.article.appendChild(view.note);
    } else if (view.note.parentNode) {
      view.note.remove();
    }
    // After a Stop that the server confirmed, keep the text the user saw (the server
    // stops within one token). Any other outcome: the server's text is the truth.
    var keepFrozen = view.stopped && finish.finishReason === "user";
    if (!keepFrozen && typeof content === "string" && content.length && content !== view.text) {
      view.text = content;
    }
    view.finish = finish;
    renderReply(view, true, finish.finishReason);
    if (finish.finishReason === "error") {
      showError(view, finish.errorMessage);
      return;
    }
    if (T) {
      var inThinking = T.limitInThinking(finish.finishReason, finish.limitHitInThinking, view.parse && view.parse.state);
      showNotice(view, T.noticeText(finish.finishReason, {
        maxTokens: finish.maxTokens,
        answered: !inThinking && !!(view.parse && view.parse.answer.replace(/\s+/g, "")),
        beforeReply: finish.finishReason === "user" && T.stoppedBeforeReply(finish.totalTokens, view.text)
      }), finish.finishReason);
    }
    scrollTurn(view.article);
  }

  /* Copying a reply copies the answer only. A selection that touches a Thinking
     section has those parts dropped; a selection wholly inside one body is left alone. */
  transcript.addEventListener("copy", function (ev) {
    if (!T || !window.getSelection || !ev.clipboardData) return;
    var sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return;
    var range = sel.getRangeAt(0);
    var anc = range.commonAncestorContainer;
    var el = anc.nodeType === 1 ? anc : anc.parentNode;
    if (el && el.closest && el.closest(".thinking-body")) return;
    var frag = range.cloneContents();
    if (!frag.querySelector(".thinking, .thinking-body, .thinking-toggle, .stream-notice, .stream-note, .turn-role") && !(el && el.closest && el.closest(".thinking"))) return;
    var segments = [];
    var kids = frag.childNodes;
    for (var i = 0; i < kids.length; i++) collectSegments(kids[i], segments);
    ev.clipboardData.setData("text/plain", T.joinCopySegments(segments));
    ev.preventDefault();
  });

  function collectSegments(node, out) {
    if (node.nodeType === 3) {
      if (node.data) out.push({ kind: "other", text: node.data });
      return;
    }
    if (node.nodeType !== 1) return;
    if (node.classList.contains("thinking") || node.classList.contains("thinking-body") || node.classList.contains("thinking-toggle")) {
      out.push({ kind: "thinking", text: node.textContent });
      return;
    }
    if (node.classList.contains("stream-notice") || node.classList.contains("stream-note") || node.classList.contains("caret") || node.classList.contains("turn-role")) {
      out.push({ kind: "chrome", text: node.textContent });
      return;
    }
    if (node.tagName === "P") {
      out.push({ kind: "answer", text: node.textContent });
      return;
    }
    for (var c = node.firstChild; c; c = c.nextSibling) collectSegments(c, out);
  }

  /* `.gguf` files in the weights folder: only used to show the full path of a model a running
     server already has loaded. Never shown as if it were loaded. */
  function showWeights(files) {
    weightFiles = files || [];
  }

  function fileForModel(id) {
    if (!id) return "";
    return weightFiles.filter(function (file) {
      return file.indexOf(id) !== -1;
    })[0] || "";
  }

  setAppState("no-model");
  if (!invoke) {
    setAppState("runtime-failed", { error: "The window cannot reach the app (no Tauri IPC)." });
    return;
  }

  /* ---- Runtime hook (Eng1: in-process server) -------------------------------------------------
     The only two places the window learns about a server:
       runtime.probe()   at launch: is a server already answering? Never starts one.
       runtime.ensure()  before a load (CHANGE, LOAD, RETRY): resolve once a server answers,
                         starting the built-in one if there is none; reject with its error text.
       runtime.onEvent({ kind: "starting" | "ready" | "failed" | "exited", endpoint, reused, error })
                         push updates from the server (e.g. a Tauri event) into the view.
     The bridge (src/server.rs) reuses an Artificer server already on 127.0.0.1:8080, else runs the
     server in the app on a free local port with the picked file; its events arrive on
     "artificer://runtime". ensure(path) resolves { endpoint, reused, loaded, model }: loaded is
     true when the built-in server was just started with that file (no second load). */
  var runtime = {
    probe: function () {
      return invoke("runtime_probe").then(function (found) {
        if (!found) throw new Error("no runtime is running");
        return refreshBackend().then(function () {
          return { endpoint: found.endpoint, reused: !!found.reused };
        });
      });
    },
    ensure: function (path) {
      return invoke("runtime_ensure", { path: path });
    },
    onEvent: function (ev) {
      if (!ev) return;
      if (ev.kind === "ready") {
        server = { endpoint: ev.endpoint || null, reused: !!ev.reused };
        return;
      }
      var shown = S.runtimeEventState(ev, pendingPath);
      if (!shown) return;
      if (shown.state === "runtime-failed") {
        server = null;
        chatEnabled = false;
      }
      setAppState(shown.state, shown.ctx);
    }
  };
  window.ArtificerRuntime = runtime;
  if (tauri.event && tauri.event.listen) {
    tauri.event.listen("artificer://runtime", function (e) {
      runtime.onEvent(e && e.payload);
    });
  }
  /* ---- end runtime hook ------------------------------------------------------------------------ */

  /* Client caps come from this build, not from a server: read them even on a cold launch (no
     server until a model is loaded), or every send takes the non-streaming path. */
  var capsReady = invoke("client_caps").then(
    function (reported) {
      caps = S.readCaps(reported);
    },
    function () {}
  );

  /* Launch: show state 1, then look for a server that is already running (state 7 when it has a
     model). No server is not an error: one is needed only once a model is picked. */
  var launchReady = runtime
    .probe()
    .then(function (found) {
      server = found;
      return capsReady.then(function () {
        if (!caps.loaded_model) return "";
        return invoke("loaded_model");
      });
    })
    .then(function (id) {
      if (!id) {
        setAppState("no-model");
        return "";
      }
      launchModel = id;
      return Promise.all([refreshModelInfo(), invoke("local_weight_files").then(showWeights).catch(function () {})])
        .then(function () {
          setAppState(server && server.reused ? "reused" : "ready", { model: id, path: fileForModel(id) || id });
          return id;
        });
    })
    .catch(function () {
      server = null;
      setAppState("no-model");
      return "";
    });

  Promise.all([launchReady, invoke("smoke_prompt").catch(function () { return ""; })])
    .then(function (result) {
      var modelId = result[0];
      var smoke = (result[1] || "").replace(/^\s+|\s+$/g, "");
      if (!modelId || !smoke || !chatEnabled) return;
      if (!caps.stream || !tauri.core.Channel) return;
      // After the resting frame has painted, then stream. Unset env never reaches here.
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          startStream(smoke);
        });
      });
    });

  invoke("local_weight_files").then(showWeights).catch(function () {});

  /* /v1/models `template_opens_think` (optional). Missing → name match in thinking.js. */
  function refreshModelInfo() {
    return invoke("loaded_model_info")
      .then(function (info) {
        templateOpensThink = info && typeof info.template_opens_think === "boolean" ? info.template_opens_think : null;
      })
      .catch(function () {
        templateOpensThink = null;
      });
  }

  /* Chat turns on only once the caps and the template flag are in, so the first reply streams
     and its thinking folds. A state change meanwhile (a server event) wins. */
  function showLoadedId(id, path) {
    if (!id) return;
    launchModel = id;
    refreshBackend();
    return Promise.all([capsReady, refreshModelInfo()]).then(function () {
      if (appState !== "starting") return;
      setAppState(server && server.reused ? "reused" : "ready", { model: id, path: path });
    });
  }

  /* One load: runtime first (state 2 → 4 on failure), then the file (→ 3/7, or 5 on failure). */
  function loadChosen(path) {
    path = String(path || "").replace(/^\s+|\s+$/g, "");
    var problem = S.typedPathProblem(path);
    if (problem) {
      setAppState("path-invalid", { path: path, error: problem });
      return;
    }
    pendingPath = path;
    chatEnabled = false;
    history = []; // a new model starts a new conversation
    setAppState("starting", { path: path, runtimeUp: !!server });
    runtime
      .ensure(path)
      .then(
        function (found) {
          server = { endpoint: found.endpoint, reused: !!found.reused };
          if (found.loaded && found.model) {
            showLoadedId(found.model, path); // the built-in server started with this file
            return;
          }
          setAppState("starting", { path: path, runtimeUp: true });
          return invoke("load_local_weights", { path: path }).then(
            function (id) {
              showLoadedId(id, path);
            },
            function (err) {
              var text = S.errorText(err);
              setAppState(S.isPathError(text) ? "path-invalid" : "load-failed", { path: path, error: text });
            }
          );
        },
        function (err) {
          var failed = S.ensureFailure(err);
          if (failed.state === "runtime-failed") server = null;
          setAppState(failed.state, { path: path, error: failed.error });
        }
      );
  }

  /* Picker open/closed: CHANGE and LOAD off while it is open (same disabled style as loading). */
  function setPickerOpen(open) {
    pickerOpen = open;
    if (!lastView) return;
    var controls = S.controlsFor(lastView, open);
    change.disabled = controls.changeDisabled;
    if (pathLoad) pathLoad.disabled = controls.loadDisabled;
  }

  function applyPick(result) {
    setPickerOpen(false); // pick, cancel or error: re-enable before anything else
    if (!result) return; // a real Cancel
    if (result.load) {
      loadChosen(result.load);
      return;
    }
    setAppState(result.state, result.ctx);
    if (result.focusPath && pathInput && pathInput.focus) pathInput.focus();
  }

  /* CHANGE: native picker. Cancel keeps the current state; errors are shown, never swallowed. */
  change.addEventListener("click", function () {
    if (change.disabled || pickerOpen) return;
    setPickerOpen(true);
    invoke("pick_local_weights").then(
      function (picked) {
        applyPick(S.pickResult(true, picked));
      },
      function (err) {
        applyPick(S.pickResult(false, err));
      }
    );
  });

  if (pathLoad && pathInput) {
    pathLoad.addEventListener("click", function () {
      if (pickerOpen) return;
      loadChosen(pathInput.value);
    });
    pathInput.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && !pickerOpen) {
        ev.preventDefault();
        loadChosen(pathInput.value);
      }
    });
  }

  /* RETRY (state 4): check for / start the runtime again, then reload the last file, if any. */
  if (retry) {
    retry.addEventListener("click", function () {
      if (pendingPath) {
        loadChosen(pendingPath);
        return;
      }
      runtime
        .probe()
        .then(function (found) {
          server = found;
          setAppState("no-model");
        })
        .catch(function (err) {
          setAppState("runtime-failed", { error: S.errorText(err) });
        });
    });
  }

  function startStream(text) {
    if (busy || !text || !chatEnabled) return;
    if (!caps.stream || !tauri.core.Channel) return;
    busy = true;
    appendUser(text);
    history = S.historyAdd(history, text);
    activeStream = beginStream();
    var view = activeStream;
    setSendLabel("Stop");
    var channel = new tauri.core.Channel();
    channel.onmessage = function (chunk) {
      if (typeof chunk === "string") appendDelta(view, chunk);
    };
    // Let the caret paint, then read real deltas. Leave the streaming chrome
    // up briefly after the last chunk so a frame can show it.
    setTimeout(function () {
      invoke("chat_stream", { messages: history, onDelta: channel })
        .then(function (turn) {
          setTimeout(function () {
            finishStream(view, turn && turn.content, turn && turn.timings, turn);
            finishHistory(view);
            if (turn && turn.model) setModel(turn.model);
            endBusy();
          }, view.stopped ? 0 : 900);
        })
        .catch(function (err) {
          if (view.stopped) {
            // The stream was cut after our Stop: finish with what we have (no server count).
            finishStream(view, null, null, { finish_reason: "user" });
            finishHistory(view);
            endBusy();
            return;
          }
          // The request itself failed: same error notice, the bridge's message as detail.
          finishStream(view, null, null, { finish_reason: "error", error: { message: errText(err) } });
          finishHistory(view);
          endBusy();
        });
    }, 80);
  }

  /* The finished reply's answer joins the history; an error drops the user turn, and a context
     overflow clears the whole history. */
  function finishHistory(view) {
    var reason = view.finish ? view.finish.finishReason : null;
    if (reason === "error" && S.isContextOverflow(view.finish.errorMessage)) {
      history = [];
      return;
    }
    history = S.historyFinish(history, reason === "error" ? "" : S.replyAnswer(T, view.text, view.implicitOpen, reason));
  }

  var sendLabel = send.textContent;
  function setSendLabel(text) {
    send.textContent = text;
  }

  function endBusy() {
    busy = false;
    activeStream = null;
    send.disabled = !chatEnabled;
    setSendLabel(sendLabel);
  }

  function errText(err) {
    var s = typeof err === "string" ? err : err && err.message ? err.message : "";
    return s || "The runtime did not respond.";
  }

  /* Stop while streaming: freeze the reply now, then the bridge posts
     /v1/chat/cancel and keeps reading until the server's final chunk (or cuts the
     stream after its grace period). chat_stream's result finishes the turn. */
  function stopStream() {
    var view = activeStream;
    if (!view || view.done || view.frozen) return;
    freezeForStop(view);
    setSendLabel("Stopping\u2026");
    send.disabled = true; // disabled style until the server confirms
    invoke("stop_stream").catch(function () {});
  }

  send.addEventListener("click", function () {
    if (busy && activeStream && !activeStream.done) {
      stopStream(); // a second click while "Stopping…" is ignored
      return;
    }
    if (busy || !chatEnabled) return;
    var text = draft.value.replace(/^\s+|\s+$/g, "");
    if (!text) return;
    if (caps.stream && tauri.core.Channel) {
      startStream(text);
      return;
    }
    busy = true;
    appendUser(text);
    history = S.historyAdd(history, text);
    draft.value = "";
    draft.placeholder = "Follow-up";
    invoke("chat_complete", { messages: history })
      .then(function (turn) {
        var content = (turn && turn.content) || "";
        appendFinished(content);
        history = S.historyFinish(history, S.replyAnswer(T, content, T ? T.implicitOpenFor(launchModel, templateOpensThink) : false, null));
        if (turn && turn.model) setModel(turn.model);
        busy = false;
      })
      .catch(function (err) {
        // The bridge's error string; a context overflow clears the history.
        var overflow = S.isContextOverflow(errText(err));
        appendFinished(overflow ? S.CONTEXT_OVERFLOW_NOTICE : "The runtime did not respond.");
        history = overflow ? [] : S.historyFinish(history, "");
        busy = false;
      });
  });
})();
