// node --test desktop/tests/  (no dependencies)
// The whole window (dist/*.js) in a minimal fake DOM with a fake Tauri bridge: a cold launch
// (no server yet) must still read client_caps, so SEND streams and Nemotron's `</think>` folds.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const DIST = path.join(__dirname, "..", "dist");
const MODEL = "NVIDIA-Nemotron3-Nano-4B-Q4_K_M";
const WEIGHTS = "/models/" + MODEL + ".gguf";
const REPLY = "reasoning...\n</think>\nanswer";

/* ---- minimal DOM: only what app.js touches ---- */
class Node {
  constructor() { this.childNodes = []; this.parentNode = null; }
  get firstChild() { return this.childNodes[0] || null; }
  get nextSibling() {
    const k = this.parentNode ? this.parentNode.childNodes : [];
    return k[k.indexOf(this) + 1] || null;
  }
  appendChild(c) { if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.childNodes.push(c); return c; }
  insertBefore(c, ref) {
    if (!ref) return this.appendChild(c);
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this;
    this.childNodes.splice(this.childNodes.indexOf(ref), 0, c);
    return c;
  }
  removeChild(c) { this.childNodes.splice(this.childNodes.indexOf(c), 1); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
}
class Text extends Node {
  constructor(data) { super(); this.nodeType = 3; this.data = String(data); }
  get textContent() { return this.data; }
}
class Element extends Node {
  constructor(tag, attrs) {
    super();
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.attrs = {};
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.placeholder = "";
    this.title = "";
    this.scrollTop = 0; this.scrollWidth = 0; this.clientWidth = 0; this.clientHeight = 0; this.scrollHeight = 0;
    for (const k in attrs || {}) this.setAttribute(k, attrs[k]);
  }
  get className() { return this.attrs.class || ""; }
  set className(v) { this.attrs.class = String(v); }
  get id() { return this.attrs.id || ""; }
  set id(v) { this.attrs.id = String(v); }
  get classList() {
    const el = this;
    const list = () => el.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => list().includes(c),
      add: (c) => { if (!list().includes(c)) el.className = list().concat(c).join(" "); },
      remove: (c) => { el.className = list().filter((x) => x !== c).join(" "); }
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  removeAttribute(k) { delete this.attrs[k]; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { this.childNodes.forEach((c) => { c.parentNode = null; }); this.childNodes = []; if (v !== "") this.appendChild(new Text(v)); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  click() { (this.listeners.click || []).forEach((fn) => fn({ preventDefault() {} })); }
  focus() {}
  scrollIntoView() {}
  matches(sel) {
    return sel.split(",").some((one) => {
      const s = one.trim();
      if (s[0] === ".") return s.slice(1).split(".").every((c) => this.classList.contains(c));
      if (s[0] === "#") return this.id === s.slice(1);
      if (s[0] === "[") return this.hasAttribute(s.slice(1, -1));
      return this.tagName === s.toUpperCase();
    });
  }
  closest(sel) { for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (n.matches(sel)) return n; return null; }
  querySelectorAll(sel) {
    const out = [];
    const walk = (n) => n.childNodes.forEach((c) => { if (c.nodeType === 1) { if (c.matches(sel)) out.push(c); walk(c); } });
    walk(this);
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

/* index.html's window, reduced to the elements app.js looks up. */
function buildDocument() {
  const h = (tag, attrs, kids) => { const e = new Element(tag, attrs); (kids || []).forEach((k) => e.appendChild(k)); return e; };
  const body = h("body", {}, [
    h("span", { class: "health-pip" }), h("span", { "data-health-model": "" }), h("span", { "data-health-runtime": "" }),
    h("span", { "data-health-endpoint": "" }), h("span", { "data-health-note": "" }),
    h("span", { class: "weights-path", "data-weights-path": "" }), h("span", { "data-weights-status": "" }),
    h("span", { "data-backend-badge": "" }), h("button", { class: "weights-change" }),
    h("div", { "data-weights-pathrow": "" }, [h("input", { id: "weights-path-input" }), h("button", { class: "weights-load" }), h("p", { "data-weights-note": "" })]),
    h("div", { class: "transcript" }, [
      h("div", { "data-empty-state": "" }, [h("p", { "data-empty-line": "" }), h("p", { "data-empty-sub": "" }), h("p", { "data-empty-detail": "" }), h("button", { "data-empty-retry": "" })])
    ]),
    h("div", { class: "composer" }, [h("textarea"), h("button", { class: "composer-send" })])
  ]);
  return {
    body: body,
    querySelector: (s) => body.querySelector(s),
    querySelectorAll: (s) => body.querySelectorAll(s),
    getElementById: (id) => body.querySelector("#" + id),
    createElement: (tag) => new Element(tag),
    createTextNode: (t) => new Text(t)
  };
}

/* Fake Tauri: a cold start (runtime_probe → null), a client that streams, a built-in server
   that loads the picked file. Every invoke is logged in order. */
function boot(appSource) {
  const calls = [];
  const sends = [];
  const queued = []; // chat_stream turns to return instead of REPLY, oldest first
  class Channel { constructor() { this.onmessage = null; } toJSON() { return "channel"; } }
  // Args as the bridge gets them: serialized (also leaves the vm realm's arrays behind).
  const wire = (args) => JSON.parse(JSON.stringify(args));
  const replies = {
    runtime_probe: () => null,
    client_caps: () => ({ stream: true, loaded_model: true }),
    loaded_model: () => MODEL,
    loaded_model_info: () => ({ id: MODEL, template_opens_think: true }),
    check_health: () => ({ status: "ok", backend: { active: "cpu" } }),
    local_weight_files: () => [],
    smoke_prompt: () => "",
    runtime_ensure: () => ({ endpoint: "127.0.0.1:54289", reused: false, loaded: true, model: MODEL }),
    stop_stream: () => ({ requested: false, server_ack: false, dropped: false }),
    chat_complete: (args) => { sends.push({ cmd: "chat_complete", args: wire(args) }); return { content: REPLY, model: MODEL }; },
    chat_stream: (args) => {
      sends.push({ cmd: "chat_stream", args: wire(args) });
      if (queued.length) return queued.shift();
      ["reasoning...", "\n</th", "ink>\n", "answer"].forEach((d) => args.onDelta.onmessage(d));
      return { content: REPLY, model: MODEL, finish_reason: "stop" };
    }
  };
  const win = {
    console, setTimeout, clearTimeout, setInterval, clearInterval, Date, Math, JSON, Promise, Error, String, RegExp, Object, Array,
    performance: { now: () => Date.now() },
    requestAnimationFrame: (fn) => setTimeout(fn, 0),
    addEventListener() {},
    __TAURI__: {
      core: {
        Channel,
        invoke: (cmd, args) => {
          calls.push(cmd);
          if (!replies[cmd]) return Promise.reject(new Error("unknown command " + cmd));
          try { return Promise.resolve(replies[cmd](args || {})); } catch (e) { return Promise.reject(e); }
        }
      },
      event: { listen: () => Promise.resolve(() => {}) }
    }
  };
  win.window = win;
  win.document = buildDocument();
  vm.createContext(win);
  for (const f of ["metrics.js", "thinking.js", "app-state.js"]) vm.runInContext(fs.readFileSync(path.join(DIST, f), "utf8"), win, { filename: f });
  vm.runInContext(appSource, win, { filename: "app.js" });
  const doc = win.document;
  return {
    calls, sends, doc, queued,
    load(p) { doc.getElementById("weights-path-input").value = p; doc.querySelector(".weights-load").click(); },
    send(text) { doc.querySelector("textarea").value = text; doc.querySelector(".composer-send").click(); },
    lastAssistant() { const a = doc.querySelectorAll(".turn-assistant"); return a[a.length - 1] || null; }
  };
}

const settle = (ms) => new Promise((r) => setTimeout(r, ms || 20));
/* Long enough for startStream's 80 ms start and finishStream's 900 ms hold. */
const REPLY_MS = 1200;

module.exports = { boot, settle, REPLY_MS, MODEL, WEIGHTS, REPLY };

if (require.main === module) {
  const appSource = fs.readFileSync(path.join(DIST, "app.js"), "utf8");

  test("cold launch with no server still reads client_caps", async () => {
    const w = boot(appSource);
    await settle();
    assert.ok(w.calls.includes("runtime_probe"));
    assert.ok(w.calls.includes("client_caps"), "client_caps: " + w.calls.join(","));
  });

  test("load then SEND streams, and Nemotron's </think> folds into Thinking", async () => {
    const w = boot(appSource);
    await settle();
    w.load(WEIGHTS);
    await settle();
    assert.equal(w.doc.querySelector("textarea").disabled, false);
    w.send("hi");
    await settle(REPLY_MS);
    assert.deepEqual(w.sends.map((s) => s.cmd), ["chat_stream"]);
    const a = w.lastAssistant();
    const thinking = a.querySelector(".thinking");
    assert.equal(thinking.hidden, false);
    assert.equal(a.querySelector(".thinking-body").hidden, true); // collapsed
    assert.equal(a.querySelector(".thinking-body").textContent, "reasoning...");
    assert.equal(a.querySelector("p").textContent, "answer");
    assert.ok(a.textContent.indexOf("</think>") === -1);
  });

  test("a follow-up sends the earlier turns, answer only", async () => {
    const w = boot(appSource);
    await settle();
    w.load(WEIGHTS);
    await settle();
    w.send("hi");
    await settle(REPLY_MS);
    w.send("thank you");
    await settle(REPLY_MS);
    assert.equal(w.sends.length, 2);
    assert.deepEqual(w.sends[0].args.messages, [{ role: "user", content: "hi" }]);
    assert.deepEqual(w.sends[1].args.messages, [
      { role: "user", content: "hi" },
      { role: "assistant", content: "answer" },
      { role: "user", content: "thank you" }
    ]);
    assert.equal("prompt" in w.sends[1].args, false);
  });

  test("loading another model starts a new conversation", async () => {
    const w = boot(appSource);
    await settle();
    w.load(WEIGHTS);
    await settle();
    w.send("hi");
    await settle(REPLY_MS);
    w.load("/models/other-Q4_K_M.gguf");
    await settle();
    w.send("who are you");
    await settle(REPLY_MS);
    assert.deepEqual(w.sends[1].args.messages, [{ role: "user", content: "who are you" }]);
  });

  test("a context overflow clears the history and says so; another error does not", async () => {
    const w = boot(appSource);
    await settle();
    w.load(WEIGHTS);
    await settle();
    w.send("hi");
    await settle(REPLY_MS);
    // The bridge's ChatTurn for a stream that ended with the server's error chunk.
    const failed = (message) => ({ content: "", model: MODEL, finish_reason: "error", error: { message } });
    const notice = () => w.lastAssistant().querySelector(".stream-notice");
    w.queued.push(failed("prompt too long: 9000 > 4096"));
    w.send("long one");
    await settle(REPLY_MS);
    assert.equal(notice().childNodes[0].textContent, "Couldn’t generate a reply.");
    const overflow = "context length exceeded: need 9000, max 8192";
    w.queued.push(failed(overflow));
    w.send("again");
    await settle(REPLY_MS);
    // The other error dropped only its own turn.
    assert.deepEqual(w.sends[2].args.messages, [
      { role: "user", content: "hi" },
      { role: "assistant", content: "answer" },
      { role: "user", content: "again" }
    ]);
    assert.equal(notice().getAttribute("data-reason"), "error");
    assert.equal(notice().childNodes[0].textContent,
      "This conversation got too long for the model, so it was cleared. Send your message again to start fresh.");
    assert.equal(notice().querySelector(".stream-notice-detail").textContent, overflow);
    assert.equal(w.doc.querySelector("textarea").value, "");
    w.send("fresh");
    await settle(REPLY_MS);
    assert.deepEqual(w.sends[3].args.messages, [{ role: "user", content: "fresh" }]);
  });

  test("chat stays off until the model's template flag is read", async () => {
    const w = boot(appSource);
    await settle();
    w.load(WEIGHTS);
    // Same tick as the load: the server answered, model info not read yet.
    assert.equal(w.doc.querySelector("textarea").disabled, true);
    await settle();
    assert.equal(w.doc.querySelector("textarea").disabled, false);
    const i = w.calls.indexOf("loaded_model_info");
    assert.ok(i > w.calls.indexOf("runtime_ensure") && i !== -1);
  });
}
