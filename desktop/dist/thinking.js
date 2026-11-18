/* Pure helpers for the Thinking section, the stop notice and copy. No DOM, no timers.
   Loaded by index.html before app.js (window.ArtificerThinking) and by
   desktop/tests/thinking.test.cjs under `node --test` (module.exports).

   Thinking = the text before (and including) `</think>`. A leading `<think>` opener is
   optional: Nemotron's chat template already ends the prompt with `<think>\n`, so its
   output starts inside the thinking and only `</think>` appears (implicitOpen). For
   other models thinking is shown only after an explicit `<think>`.
   The UI never counts tokens: every count and the limit come from the done event. */
(function (root) {
  "use strict";

  var OPEN = "<think>";
  var CLOSE = "</think>";
  var REASONS = ["stop", "length", "user", "error"];
  /* Tolerated aliases: "cancelled" (an earlier server field name) reads as a user stop. */
  var REASON_ALIASES = { cancelled: "user" };

  function reasonOf(v) {
    if (typeof v !== "string") return null;
    var r = REASON_ALIASES[v] || v;
    return REASONS.indexOf(r) !== -1 ? r : null;
  }

  /* Length of the longest suffix of s that is a proper prefix of tag ("</thi"),
     so a tag split across deltas never flashes as text. */
  function partialTagSuffix(s, tag) {
    for (var k = Math.min(tag.length - 1, s.length); k > 0; k--) {
      if (s.slice(s.length - k) === tag.slice(0, k)) return k;
    }
    return 0;
  }

  /* split(text, {implicitOpen}) -> {thinking, answer, state}
     state: "none"   no thinking section (thinking === "")
            "open"   still inside the thinking (no </think> yet)
            "closed" </think> seen; answer is what follows it */
  function split(text, opts) {
    var s = String(text == null ? "" : text);
    var implicitOpen = !!(opts && opts.implicitOpen);
    var lead = s.replace(/^\s+/, "");
    var explicit = lead.indexOf(OPEN) === 0;
    var body = explicit ? lead.slice(OPEN.length) : s;
    var end = body.indexOf(CLOSE);
    if (end !== -1) {
      var thinking = body.slice(0, end).replace(/^\s+|\s+$/g, "");
      var answer = body.slice(end + CLOSE.length).replace(/^\s+/, "");
      if (!thinking) return { thinking: "", answer: answer, state: "none", explicit: explicit };
      return { thinking: thinking, answer: answer, state: "closed", explicit: explicit };
    }
    if (explicit || implicitOpen) {
      var cut = partialTagSuffix(body, CLOSE);
      var open = body.slice(0, body.length - cut).replace(/^\s+|\s+$/g, "");
      // While streaming an implicit-open reply, a lone "<" might be the start of "</think>".
      if (!explicit && !open) return { thinking: "", answer: "", state: "open", explicit: false };
      return { thinking: open, answer: "", state: "open", explicit: explicit };
    }
    // A reply that so far is only "<th" may still become "<think>": hold it back.
    if (lead && OPEN.indexOf(lead) === 0) return { thinking: "", answer: "", state: "none", explicit: false, held: true };
    return { thinking: "", answer: s, state: "none", explicit: false };
  }

  /* Final split once the stream has ended. An implicit-open reply that ended normally
     without ever closing its thinking was not thinking: show it as the answer.
     At the limit (or a user stop) an open thinking stays thinking. */
  function finalSplit(text, opts, finishReason) {
    var p = split(text, opts);
    if (p.state === "open" && !p.explicit && (finishReason === "stop" || finishReason == null)) {
      return { thinking: "", answer: String(text == null ? "" : text).replace(/^\s+/, ""), state: "none", explicit: false };
    }
    if (p.state === "open" && !p.thinking) return { thinking: "", answer: "", state: "none", explicit: p.explicit };
    if (p.held) return { thinking: "", answer: String(text), state: "none", explicit: false };
    return p;
  }

  /* Answer only, never the thinking. */
  function copyText(text, opts, finishReason) {
    return finalSplit(text, opts, finishReason).answer;
  }

  /* Join DOM-free segments of a selection, dropping thinking ones.
     segments: [{kind: "thinking"|"chrome"|"answer"|"other", text}]; only answer/other are kept
     (chrome = role label, stop notice, tok/s readout, caret). */
  function joinCopySegments(segments) {
    return (segments || [])
      .filter(function (x) { return x && (x.kind === "answer" || x.kind === "other") && x.text; })
      .map(function (x) { return x.text; })
      .join("\n");
  }

  /* 1024 -> "1,024". Fixed en-US grouping so copy is stable across locales. */
  function formatCount(n) {
    if (typeof n !== "number" || !isFinite(n)) return "";
    var s = String(Math.round(Math.abs(n)));
    var out = "";
    while (s.length > 3) {
      out = "," + s.slice(s.length - 3) + out;
      s = s.slice(0, s.length - 3);
    }
    return (n < 0 ? "-" : "") + s + out;
  }

  /* Seconds for the header: "4s", "1m 5s". */
  function formatSeconds(ms, roundUp) {
    var t = typeof ms === "number" && ms > 0 ? ms / 1000 : 0;
    var sec = roundUp ? Math.max(1, Math.round(t)) : Math.floor(t);
    if (sec < 60) return sec + "s";
    return Math.floor(sec / 60) + "m " + (sec % 60) + "s";
  }

  function count(v) {
    return typeof v === "number" && isFinite(v) && v >= 0 ? v : null;
  }

  function first() {
    for (var i = 0; i < arguments.length; i++) if (arguments[i] != null) return arguments[i];
    return null;
  }

  /* Read the done event defensively. Primary (server outcome fields, top level of the final chunk):
       {finish_reason: 'stop'|'length'|'user'|'error', max_tokens: N,
        tokens: {thinking, answer, total}, thinking_ms, prefill_ms,
        limit_hit_in_thinking, error: {message}}
     Fallbacks: choices[0].finish_reason; the requested max_tokens; usage.completion_tokens
     (plain OpenAI). Tolerated aliases: finish_reason 'cancelled', top-level
     thinking_tokens / completion_tokens. */
  function readDone(done, requestedMax) {
    var d = done && typeof done === "object" ? done : {};
    var c = d.choices && d.choices[0];
    var reason = first(reasonOf(d.finish_reason), c ? reasonOf(c.finish_reason) : null);
    var max = count(d.max_tokens) || count(requestedMax) || null;
    var t = d.tokens && typeof d.tokens === "object" ? d.tokens : {};
    var u = d.usage && typeof d.usage === "object" ? d.usage : {};
    return {
      finishReason: reason,
      maxTokens: max,
      thinkingTokens: first(count(t.thinking), count(d.thinking_tokens)),
      answerTokens: count(t.answer),
      totalTokens: first(count(t.total), count(d.completion_tokens), count(u.completion_tokens)),
      thinkingMs: count(d.thinking_ms),
      prefillMs: count(d.prefill_ms),
      limitHitInThinking: typeof d.limit_hit_in_thinking === "boolean" ? d.limit_hit_in_thinking : null,
      errorMessage: d.error && typeof d.error.message === "string" && d.error.message ? d.error.message : null,
      connectionDropped: d.connection_dropped === true
    };
  }

  function limitPhrase(maxTokens) {
    return maxTokens ? "the " + formatCount(maxTokens) + "-token limit" : "the token limit";
  }

  /* Did the limit land while still thinking? Server's limit_hit_in_thinking when sent,
     else the parse (no </think> seen). */
  function limitInThinking(finishReason, limitHitInThinking, state) {
    if (finishReason !== "length") return false;
    if (typeof limitHitInThinking === "boolean") return limitHitInThinking;
    return state === "open";
  }

  /* Header label. o: {streaming, elapsedMs, state, finishReason, thinkingTokens, maxTokens,
     limitHitInThinking} */
  function headerText(o) {
    var x = o || {};
    if (x.streaming) return "Thinking\u2026 " + formatSeconds(x.elapsedMs, false);
    if (limitInThinking(x.finishReason, x.limitHitInThinking, x.state)) {
      return "Thinking stopped at " + limitPhrase(x.maxTokens);
    }
    var s = "Thought for " + formatSeconds(x.elapsedMs, true);
    if (x.thinkingTokens != null) {
      s += " \u00b7 " + formatCount(x.thinkingTokens) + (x.thinkingTokens === 1 ? " token" : " tokens");
    }
    return s;
  }

  /* One-line notice under the reply, or null. Errors keep the existing error style. */
  /* o: {maxTokens, answered, pending, beforeReply}. pending: Stop pressed, server not
     confirmed yet. beforeReply: the server generated nothing (Stop during prefill). */
  function noticeText(finishReason, o) {
    var x = o || {};
    if (finishReason === "length") {
      return "Stopped at " + limitPhrase(x.maxTokens) + (x.answered ? "." : " before an answer.");
    }
    if (finishReason === "user") {
      if (x.pending) return "Stopping\u2026";
      return x.beforeReply ? "Stopped by you before the reply started." : "Stopped by you.";
    }
    return null;
  }

  /* Did a stopped reply produce nothing? Server total when sent, else what arrived. */
  function stoppedBeforeReply(totalTokens, receivedText) {
    if (typeof totalTokens === "number") return totalTokens === 0;
    return !String(receivedText || "").replace(/\s+/g, "");
  }

  /* Error notice (finish_reason "error" or a failed request): fixed main line, the
     server's message as detail (muted mono), or no detail. */
  function errorNotice(message) {
    var m = typeof message === "string" ? message.replace(/^\s+|\s+$/g, "") : "";
    return { main: "Couldn\u2019t generate a reply.", detail: m || null };
  }

  /* Thinking-body auto-scroll: stay pinned to the bottom unless the user scrolled up. */
  function atBottom(scrollTop, clientHeight, scrollHeight, slack) {
    return scrollTop + clientHeight >= scrollHeight - (slack == null ? 4 : slack);
  }

  /* Does the reply start inside the thinking? /v1/models `template_opens_think` when the
     server reports it; else a name match (Nemotron's template ends the prompt with <think>). */
  function implicitOpenFor(modelId, templateOpensThink) {
    if (typeof templateOpensThink === "boolean") return templateOpensThink;
    return /nemotron/i.test(String(modelId || ""));
  }

  var api = {
    split: split,
    finalSplit: finalSplit,
    copyText: copyText,
    joinCopySegments: joinCopySegments,
    formatCount: formatCount,
    formatSeconds: formatSeconds,
    readDone: readDone,
    headerText: headerText,
    noticeText: noticeText,
    atBottom: atBottom,
    implicitOpenFor: implicitOpenFor,
    limitInThinking: limitInThinking,
    stoppedBeforeReply: stoppedBeforeReply,
    errorNotice: errorNotice
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ArtificerThinking = api;
})(typeof window !== "undefined" ? window : this);
