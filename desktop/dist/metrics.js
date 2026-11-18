/* Pure helpers for the backend badge and the tok/s readout. No DOM, no timers.
   Loaded by index.html before app.js (window.ArtificerMetrics) and by
   desktop/tests/metrics.test.cjs under `node --test` (module.exports). */
(function (root) {
  "use strict";

  /* Badge for the Weights row from the `/health` `backend` object.
     Never reports a silent CPU: a missing object is "Backend unknown". */
  function backendBadge(backend) {
    if (!backend || typeof backend.active !== "string" || !backend.active) {
      return { text: "Backend unknown", tone: "unknown", title: "The server did not report a compute backend." };
    }
    var active = backend.active.toLowerCase();
    if (active === "cuda") {
      var device = (backend.device || "").trim();
      var short = shortDevice(device);
      return {
        text: "GPU · CUDA" + (short ? " · " + short : ""),
        tone: "gpu",
        title: device ? "CUDA device: " + device : "CUDA backend"
      };
    }
    if (active === "cpu") {
      var reason = (backend.fallback_reason || "").trim();
      if (reason) {
        return { text: "CPU (GPU unavailable)", tone: "fallback", title: reason };
      }
      var requested = (backend.requested || "").toLowerCase();
      return { text: "CPU", tone: "cpu", title: requested === "cuda" ? "CUDA requested; running on CPU" : "CPU backend" };
    }
    return { text: backend.active, tone: "unknown", title: "Backend: " + backend.active };
  }

  /* Short device name for the badge (full name stays in the tooltip):
     drop vendor/brand prefixes and a trailing "GPU",
     e.g. "NVIDIA GeForce RTX 4050 Laptop GPU" -> "RTX 4050 Laptop". */
  function shortDevice(name) {
    var s = (name || "").trim();
    s = s.replace(/^NVIDIA\s+/i, "").replace(/^(GeForce|Quadro|Tesla)\s+/i, "").replace(/\s+GPU$/i, "").trim();
    return s || (name || "").trim();
  }

  /* tok/s for display: one decimal under 100, whole number at or above. */
  function formatRate(rate) {
    if (typeof rate !== "number" || !isFinite(rate) || rate <= 0) return "–";
    return rate >= 100 ? String(Math.round(rate)) : rate.toFixed(1);
  }

  function rate(tokens, ms) {
    if (!(tokens > 0) || !(ms > 0)) return null;
    return (tokens * 1000) / ms;
  }

  /* Rolling decode rate from client-side delta arrival times (ms).
     Rate = (events in window - 1) / (newest - oldest in window), so the
     first token after prefill does not count as instantaneous decode. */
  function RollingRate(windowMs) {
    this.windowMs = windowMs || 2000;
    this.times = [];
    this.count = 0;
  }
  RollingRate.prototype.push = function (nowMs) {
    this.times.push(nowMs);
    this.count += 1;
    this.trim(nowMs);
  };
  RollingRate.prototype.trim = function (nowMs) {
    var cutoff = nowMs - this.windowMs;
    while (this.times.length > 2 && this.times[0] < cutoff) this.times.shift();
  };
  RollingRate.prototype.rate = function () {
    var n = this.times.length;
    if (n < 2) return null;
    var span = this.times[n - 1] - this.times[0];
    return span > 0 ? ((n - 1) * 1000) / span : null;
  };

  function liveReadout(r) {
    return "Decoding " + formatRate(r) + " tok/s";
  }

  /* 1024 -> "1,024" (fixed en-US grouping). */
  function groupDigits(n) {
    var s = String(Math.round(n));
    var out = "";
    while (s.length > 3) {
      out = "," + s.slice(s.length - 3) + out;
      s = s.slice(0, s.length - 3);
    }
    return s + out;
  }

  /* Frozen line after the stream ends. Server timings preferred; else the client rate.
     The token count is the server's total only (done-event tokens.total, else
     usage.completion_tokens); without one there is no count. The UI never counts. */
  function finalReadout(timings, client) {
    var c = client || {};
    var total = typeof c.totalTokens === "number" && c.totalTokens >= 0 ? " · " + groupDigits(c.totalTokens) + (c.totalTokens === 1 ? " token" : " tokens") : "";
    if (timings && !(timings.decode_tokens > 0) && !(c.totalTokens > 0) && timings.prefill_tokens > 0) {
      // Nothing generated (e.g. Stop during prefill): prefill speed only.
      return "Prefill " + formatRate(rate(timings.prefill_tokens, timings.prefill_ms)) + " tok/s";
    }
    if (timings && (timings.decode_tokens > 0 || timings.prefill_tokens > 0)) {
      return "Prefill " + formatRate(rate(timings.prefill_tokens, timings.prefill_ms)) +
        " tok/s · Decode " + formatRate(rate(timings.decode_tokens, timings.decode_ms)) +
        " tok/s" + total;
    }
    return "Decode " + formatRate(c.rate) + " tok/s" + total;
  }

  /* Weights path display: shorter forms that keep the filename intact, longest first.
     "/data/YOU/Projects/Artificer/x.gguf" -> [full, "/data/YOU/Projects/…/x.gguf",
     "/data/YOU/…/x.gguf", "/data/…/x.gguf", "…/x.gguf"]. Handles "/" or "\\" separators. */
  function pathCandidates(path) {
    var s = String(path == null ? "" : path);
    var sep = s.indexOf("/") === -1 && s.indexOf("\\") !== -1 ? "\\" : "/";
    var cut = s.lastIndexOf(sep);
    if (cut <= 0) return [s];
    var base = s.slice(cut + 1);
    var parts = s.slice(0, cut).split(sep);
    var out = [s];
    for (var k = parts.length - 2; k >= 1; k--) {
      out.push(parts.slice(0, k + 1).join(sep) + sep + "\u2026" + sep + base);
    }
    out.push("\u2026" + sep + base);
    return out.filter(function (c, i) { return out.indexOf(c) === i; });
  }

  /* Middle-truncate a path to at most maxChars, filename intact (the shortest form,
     "…/<file>", is returned even if it is still longer than maxChars). */
  function middleTruncatePath(path, maxChars) {
    var c = pathCandidates(path);
    for (var i = 0; i < c.length; i++) if (c[i].length <= maxChars) return c[i];
    return c[c.length - 1];
  }

  var api = {
    backendBadge: backendBadge,
    pathCandidates: pathCandidates,
    middleTruncatePath: middleTruncatePath,
    formatRate: formatRate,
    groupDigits: groupDigits,
    rate: rate,
    RollingRate: RollingRate,
    liveReadout: liveReadout,
    finalReadout: finalReadout
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ArtificerMetrics = api;
})(typeof window !== "undefined" ? window : this);
