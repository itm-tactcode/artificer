#!/usr/bin/env python3
"""Generate THIRD-PARTY-NOTICES.md for the shipped binaries, offline, stdlib only (python >= 3.11 for tomllib).

Usage (from the repository root; see scripts/third-party-notices/README.md):
  python3 scripts/third-party-notices/gen.py [--crates scripts/third-party-notices/crates.txt] [--upstream DIR] \
      . THIRD-PARTY-NOTICES.md [summary.json]
--crates: the crates compiled into a shipped binary, "name version scope" per line (written by crates.sh from
  `cargo tree` over the release targets). Workspace crates are ignored; a listed crate missing from Cargo.lock is an error.
--upstream: license files fetched once from the crates' repositories at the version tag (or the published commit),
  listed with URL and sha256 in <DIR>/SOURCES.tsv; used where a crate package ships no license file or no copyright
  line. Default: scripts/third-party-notices/upstream.
License metadata and texts come from the vendored crate sources under $CARGO_HOME/registry/src/*/<name>-<version>/
(default ~/.cargo). No cargo, no network.
"""
import glob, hashlib, json, os, re, sys, tomllib, datetime
from collections import defaultdict

args = sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__))
cli_only_file, upstream_dir = os.path.join(HERE, "crates.txt"), os.path.join(HERE, "upstream")
while args and args[0].startswith("--"):
    flag = args.pop(0)
    if flag in ("--crates", "--cli-only"):
        cli_only_file = args.pop(0)
    elif flag == "--upstream":
        upstream_dir = args.pop(0)
    else:
        raise SystemExit(f"unknown flag {flag}")
if len(args) not in (2, 3):
    raise SystemExit(__doc__)
repo, out_md = args[0], args[1]
json_out = args[2] if len(args) > 2 else None
lock = tomllib.load(open(os.path.join(repo, "Cargo.lock"), "rb"))
pkgs = lock["package"]
by_name = defaultdict(list)
for p in pkgs:
    by_name[p["name"]].append(p)
key = lambda p: (p["name"], p["version"])
ws = {p["name"]: p for p in pkgs if "source" not in p}
ws_root = tomllib.load(open(os.path.join(repo, "Cargo.toml"), "rb"))
members = ws_root["workspace"]["members"]
ws_manifest = {}
for m in members:
    t = tomllib.load(open(os.path.join(repo, m, "Cargo.toml"), "rb"))
    ws_manifest[t["package"]["name"]] = t

def normal_dep_names(t, include_build):
    """Package names from [dependencies] (+ target-specific, + build if asked); no dev-deps."""
    tables = [t.get("dependencies", {})]
    if include_build:
        tables.append(t.get("build-dependencies", {}))
    for tgt in t.get("target", {}).values():
        tables.append(tgt.get("dependencies", {}))
        if include_build:
            tables.append(tgt.get("build-dependencies", {}))
    names = set()
    wsdeps = ws_root.get("workspace", {}).get("dependencies", {})
    for tab in tables:
        for k, v in tab.items():
            name = k
            if isinstance(v, dict):
                if v.get("package"):
                    name = v["package"]
                elif v.get("workspace"):
                    wv = wsdeps.get(k, {})
                    if isinstance(wv, dict) and wv.get("package"):
                        name = wv["package"]
            names.add(name)
    return names

def resolve(dep):
    parts = dep.split()
    cands = by_name.get(parts[0], [])
    if len(parts) > 1:
        cands = [c for c in cands if c["version"] == parts[1]]
    if len(cands) != 1:
        raise SystemExit(f"ambiguous/missing lock dep {dep!r}: {len(cands)}")
    return cands[0]

def walk(root_name):
    seen, stack = set(), [ws[root_name]]
    while stack:
        p = stack.pop()
        if key(p) in seen:
            continue
        seen.add(key(p))
        deps = [resolve(d) for d in p.get("dependencies", [])]
        if "source" not in p:  # workspace member: drop dev-dependencies
            allowed = normal_dep_names(ws_manifest[p["name"]], include_build=True)
            deps = [d for d in deps if d["name"] in allowed]
        stack.extend(deps)
    return {k for k in seen if k[0] not in ws}

cli = walk("artificer-cli")
desk = walk("artificer-desktop") if "artificer-desktop" in ws else set()
all_reg = {key(p) for p in pkgs if "source" in p}
dev_only = all_reg - cli - desk
lock_total = len(all_reg)
walk_cli = len(cli)
if cli_only_file:
    want = set()
    cli_targets = {}
    for ln in open(cli_only_file):
        parts = ln.split()
        if len(parts) < 2 or ln.startswith("#"):
            continue
        n, v = parts[0], parts[1].lstrip("v")
        if n in ws:
            continue
        if (n, v) not in all_reg:
            raise SystemExit(f"--cli-only: {n} {v} is not a crates.io package in Cargo.lock")
        want.add((n, v))
        if len(parts) > 2:
            cli_targets[(n, v)] = parts[2]
    all_reg, cli, desk, dev_only = want, set(want), set(), set()

reg_roots = glob.glob(os.path.join(os.environ.get("CARGO_HOME") or os.path.expanduser("~/.cargo"), "registry", "src", "*", ""))
def crate_dir(n, v):
    for r in reg_roots:
        d = os.path.join(r, f"{n}-{v}")
        if os.path.isdir(d):
            return d
    return None

LIC_RE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|UNLICENSE|COPYRIGHT|AUTHORS)", re.I)
# Attribution files: reproduced per crate, never classified or folded into a shared license text (Lawyer review item 15).
ATTR_RE = re.compile(r"^(COPYRIGHT|AUTHORS|NOTICE)|THIRD[-_]PARTY", re.I)
# License files below a crate's top level that the scan above does not see (Lawyer checklist 2026-11-14, items 2-3):
# rfd's second MIT notice and the Wayland protocols' COPYING. Their copyright lines join the crate's own.
EXTRA_FILES = {
    ("rfd", "0.17.2"): ["src/backend/xdg_desktop_portal/window_identifier/LICENSE"],
    ("wayland-protocols", "0.32.13"): ["protocols/COPYING"],
}
info = {}
for n, v in sorted(all_reg):
    d = crate_dir(n, v)
    rec = {"name": n, "version": v, "dir": d, "license": None, "license_file": None,
           "repository": None, "authors": [], "files": {}}
    if d:
        try:
            t = tomllib.load(open(os.path.join(d, "Cargo.toml"), "rb"))
            pk = t.get("package", {})
            rec["license"] = pk.get("license")
            rec["license_file"] = pk.get("license-file")
            rec["repository"] = pk.get("repository")
            rec["authors"] = [re.sub(r"\s*<[^>]*>", "", a).strip() for a in pk.get("authors", []) or []]
        except Exception as e:
            rec["error"] = str(e)
        names = [f for f in os.listdir(d) if LIC_RE.match(f) and os.path.isfile(os.path.join(d, f))]
        if rec["license_file"]:
            names.append(rec["license_file"])
        for f in sorted(set(names)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                rec["files"][f] = open(fp, encoding="utf-8", errors="replace").read()
        for sub in ("LICENSES", "licenses", "license"):
            sd = os.path.join(d, sub)
            if os.path.isdir(sd):
                for f in sorted(os.listdir(sd)):
                    fp = os.path.join(sd, f)
                    if os.path.isfile(fp):
                        rec["files"][f"{sub}/{f}"] = open(fp, encoding="utf-8", errors="replace").read()
    for f in EXTRA_FILES.get((n, v), []):
        fp = os.path.join(d, f) if d else ""
        if not os.path.isfile(fp):
            raise SystemExit(f"{n} {v}: expected license file {f} is missing from the crate source")
        rec["files"][f] = open(fp, encoding="utf-8", errors="replace").read()
    info[(n, v)] = rec

# Upstream license files (Lawyer checklist 2026-11-13, item 3): fetched once at the version tag (or the published
# commit) and vendored with URL + sha256 in SOURCES.tsv. Merged into the crate's files, tagged with their URL.
upstream_src = defaultdict(list)   # (name, version) -> [(file, url, ref, ref_kind)]
upstream_meta = {}                 # non-crate rows (WebView2 SDK, SPDX texts)
for ln in open(os.path.join(upstream_dir, "SOURCES.tsv")):
    row = ln.rstrip("\n").split("\t")
    if row[0] == "crate" or len(row) < 7:
        continue
    n, v, f, ref, kind, url, digest = row[:7]
    k = (n, v)
    if f == "-":
        upstream_src[k].append((None, url, ref, kind)); continue
    fp = os.path.join(upstream_dir, f"{n}-{v}" if n != "spdx" else "spdx", f)
    data = open(fp, "rb").read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit(f"{fp}: sha256 does not match SOURCES.tsv")
    if k in info:
        info[k]["files"][f"upstream {f}"] = data.decode("utf-8", errors="replace")
        upstream_src[k].append((f, url, ref, kind))
    else:
        upstream_meta[(n, v, f)] = (url, data.decode("utf-8", errors="replace"))
SPDX_TEXT = {f[:-4]: t for (n, v, f), (u, t) in upstream_meta.items() if n == "spdx"}
SPDX_URL = {f[:-4]: u for (n, v, f), (u, t) in upstream_meta.items() if n == "spdx"}
# Crates with no license file in the package or upstream: the standard SPDX text (Lawyer item 3).
for k, rec in info.items():
    if rec["dir"] and not rec["files"] and rec.get("license") in SPDX_TEXT:
        rec["files"][f"SPDX {rec['license']}"] = SPDX_TEXT[rec["license"]]
        rec["spdx_standard"] = True

COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|CDDL|EUPL|OSL|CC-BY-SA)", re.I)
COMMON = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib", "Unicode-3.0",
          "Unicode-DFS-2016", "Apache-2.0 WITH LLVM-exception", "0BSD", "BSL-1.0", "CC0-1.0", "Unlicense",
          "MIT-0"}
def norm_expr(e):
    return re.sub(r"\s+", " ", (e or "").replace("/", " OR ")).strip()
def ids(e):
    return {x for x in re.split(r"\s+(?:OR|AND)\s+|[()]", norm_expr(e)) if x.strip()}

def classify(fname, text):
    t = text.lower()
    # Unicode before MIT: the Unicode V3 permission grant also says "permission is hereby granted, free of charge".
    if "unicode license v3" in t or ("unicode" in t and "data files" in t):
        return "Unicode-3.0"
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "permission is hereby granted, free of charge" in t:
        return "MIT"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if ("neither the name" in t or "endorse or promote" in t) else "BSD-2-Clause"
    if "permission to use, copy, modify, and/or distribute this software" in t or "permission to use, copy, modify, and distribute this software" in t:
        return "ISC"
    if "this software is provided 'as-is'" in t:
        return "Zlib"
    if "boost software license" in t:
        return "BSL-1.0"
    if "this is free and unencumbered software" in t:
        return "Unlicense"
    if "mozilla public license" in t:
        return "MPL-2.0"
    if "creative commons" in t and "cc0" in t:
        return "CC0-1.0"
    if fname.upper().startswith("NOTICE"):
        return "NOTICE"
    return "other"

def copyright_lines(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^(copyright|\(c\)|©)", s, re.I) and "notice" not in s.lower() and "holder" not in s.lower() \
                and "[yyyy]" not in s.lower() and "{yyyy}" not in s.lower() and len(s) < 200 \
                and not s.endswith(":") and "ordered alphabetically" not in s.lower():
            out.append(s)
    return out

# Stated elections for OR-licensed crates (Lawyer review items 5 and 9). Others: see license_used().
ELECT = {
    "brotli-decompressor": ("BSD-3-Clause", "elected from BSD-3-Clause OR MIT; the upstream LICENSE is BSD-3-Clause with the Dropbox copyright"),
    "r-efi": ("MIT", "elected from MIT OR Apache-2.0 OR LGPL-2.1-or-later; the LGPL text is not reproduced"),
    "ryu": ("Apache-2.0", "elected from Apache-2.0 OR BSL-1.0"),
    "aho-corasick": ("MIT", "elected from Unlicense OR MIT"),
    "memchr": ("MIT", "elected from Unlicense OR MIT"),
    "wasi": ("Apache-2.0", "elected from Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT; its LICENSE-MIT has no copyright line"),
}
# Copyright lines not found by the file scan, with where they come from (Lawyer review items 3, 6, 7).
OVERRIDE = {
    ("atomic-waker", "1.1.2"): ["Copyright (c) 2016 Alex Crichton", "Copyright (c) 2017 The Tokio Authors",
                                "(from the crate's LICENSE-THIRD-PARTY: code derived from futures-rs)"],
    ("utf8_iter", "1.0.4"): ["Copyright Mozilla Foundation; portions adapted from the Rust standard library "
                             "(The Rust Project Developers), Apache-2.0 OR MIT."],
    ("valuable", "0.1.1"): ["Copyright (c) 2021 Valuable Contributors",
                            "(from the upstream repository's LICENSE, github.com/tokio-rs/valuable; the crate package ships no license file)"],
}
# Pointer lines appended after a crate's copyright lines.
NOTE_AFTER = {
    ("rfd", "0.17.2"): ["(second MIT notice, Bilal Elmoussaoui 2020: the crate's "
                        "src/backend/xdg_desktop_portal/window_identifier/LICENSE, full text in the Wayland and file "
                        "dialog section)"],
    ("wayland-protocols", "0.32.13"): ["(protocols/COPYING in full, and the headers of the protocol XML files used, "
                                       "in the Wayland and file dialog section)"],
    ("tauri-plugin", "2.7.2"): ["Credit: Tauri Apps Contributors. The crate package ships no license file; "
                                "Apache-2.0 OR MIT per Cargo.toml, texts from the Tauri repository at tag "
                                "tauri-plugin-v2.7.2 (as for zmij, from upstream)"],
}
# MIT-only crates whose line must come from the vendored source; flagged if the source has none (Lawyer item 4).
FROM_SOURCE = {"zmij", "r-efi"}
# Lawyer-approved attribution wording (2026-11-04) for crates whose source has no copyright notice. Used verbatim,
# never rendered as a "Copyright (c)" line, and not flagged.
ATTRIB_TEXT = {("zmij", "1.0.23"): "zmij 1.0.23, by David Tolnay (author per Cargo.toml `authors`; the upstream "
               "LICENSE-MIT carries no copyright notice). Used under the MIT License, full text in the License texts section."}

def split_top(expr, op):
    out, depth, cur = [], 0, ""
    toks = re.split(r"(\(|\)|\s+)", expr)
    i = 0
    parts = [t for t in toks if t and not t.isspace()]
    for t in parts:
        if t == "(": depth += 1
        if t == ")": depth -= 1
        if t == op and depth == 0:
            out.append(cur.strip()); cur = ""; continue
        cur += (" " if cur and t != ")" and not cur.endswith("(") else "") + t
    out.append(cur.strip())
    return [re.sub(r"^\((.*)\)$", r"\1", x).strip() for x in out if x]

def license_used(name, expr, has_mit_line):
    """Return (used, note, flag) for an SPDX expression."""
    if name in ELECT:
        return ELECT[name][0], ELECT[name][1], None
    terms, used, notes, flag = split_top(expr, "AND"), [], [], None
    for term in terms:
        opts = split_top(term, "OR")
        if len(opts) == 1:
            used.append(opts[0]); continue
        if "MIT" in opts and has_mit_line:
            used.append("MIT"); notes.append(f"elected from {term}")
        elif "Apache-2.0" in opts:
            used.append("Apache-2.0"); notes.append(f"elected from {term}" + ("" if "MIT" not in opts else "; no MIT copyright line"))
        elif "MIT" in opts:
            used.append("MIT"); notes.append(f"elected from {term}")
        else:
            used.append(term); flag = f"no election rule for `{term}`"
    if len(terms) > 1:
        notes.append("all AND terms apply")
    return " AND ".join(used), "; ".join(notes), flag

flags = []
review = []   # resolved by a Lawyer rule but worth a look
groups = defaultdict(list)
texts = defaultdict(dict)  # lic -> hash -> (text, [crates])
mit_copy = {}
attr_files = defaultdict(dict)   # crate -> fname -> text
uni_copy = defaultdict(list)     # crate -> Unicode copyright lines
for k in sorted(all_reg):
    r = info[k]
    expr = norm_expr(r["license"]) if r["license"] else (f"(license-file: {r['license_file']})" if r["license_file"] else "NONE")
    r["expr"] = expr
    if r["dir"] is None: expr = "UNVERIFIED (source not fetched locally)"; r["expr"] = expr
    groups[expr].append(k)
    crate = f"{k[0]} {k[1]}"
    if r["dir"] is None:
        flags.append((crate, "UNVERIFIED: source never fetched on this machine, so it was not compiled into any local Linux build "
                             "(platform-specific or desktop-only). License must be checked with `cargo fetch` or by hand"))
    elif not r["license"]:
        flags.append((crate, f"no SPDX `license` field ({expr})"))
    if r["license"]:
        unusual = ids(r["license"]) - COMMON
        if norm_expr(r["license"]) == "MPL-2.0":
            pass  # listed with source links in the MPL-2.0 section (Lawyer item 2)
        elif COPYLEFT.search(r["license"]):
            alt = " OR " in norm_expr(r["license"])
            flags.append((crate, f"copyleft term in `{expr}`" + (" (permissive alternative offered via OR)" if alt else "")))
        elif unusual:
            flags.append((crate, f"unusual license id(s) {sorted(unusual)} in `{expr}`"))
    if r["dir"] and not r["files"]:
        flags.append((crate, "no LICENSE/COPYING file in the crate source"))
    for fname, text in r["files"].items():
        if ATTR_RE.search(os.path.basename(fname).removeprefix("upstream ")):
            attr_files[crate][fname] = text
            continue
        lic = classify(fname, text)
        if lic == "MIT":
            mit_copy.setdefault(crate, []).extend(x for x in copyright_lines(text) if x not in mit_copy.get(crate, []))
            continue
        if lic == "Unicode-3.0":
            uni_copy[crate].extend(copyright_lines(text))
        h = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()[:12]
        texts[lic].setdefault(h, [text, []])[1].append(f"{crate} ({fname})")

attrib = {}
for k in sorted(all_reg):
    r = info[k]; crate = f"{k[0]} {k[1]}"
    if not r["license"]:
        continue
    lines = list(OVERRIDE.get(k, []))
    if k[0] in FROM_SOURCE:
        src = [x for f, t in attr_files.get(crate, {}).items() for x in copyright_lines(t)] + mit_copy.get(crate, [])
        if src:
            lines += [x for x in src if x not in lines] + ["(from the crate's own " +
                      ", ".join(sorted(set(attr_files.get(crate, {})) | ({"LICENSE-MIT"} if mit_copy.get(crate) else set()))) + ")"]
        elif k in ATTRIB_TEXT:
            lines.append(ATTRIB_TEXT[k])
        else:
            fb = ", ".join(r["authors"]) or "unknown"
            lines.append(f"Copyright (c) {fb} (from Cargo.toml `authors`; the vendored source has no copyright line)")
            flags.append((crate, "MIT-only and no copyright line in the vendored source: attribution taken from Cargo.toml `authors`, needs counsel OK (Lawyer Q3)"))
    has_mit = bool(mit_copy.get(crate))
    used, note, flag = license_used(k[0], norm_expr(r["license"]), has_mit or (k[0] in FROM_SOURCE and bool(lines)))
    if flag:
        flags.append((crate, flag))
    if "MIT" in used.split(" AND ") and k[0] not in FROM_SOURCE:
        lines += [x for x in mit_copy.get(crate, []) if not any(x in y for y in lines)]
    for f, t in attr_files.get(crate, {}).items():
        lines += [x for x in copyright_lines(t) if x not in lines and k[0] not in FROM_SOURCE and k not in OVERRIDE]
    # Copyright lines of the elected non-MIT license files (BSD, ISC, Zlib, ...), e.g. the Dropbox line in alloc-stdlib.
    for f, t in r["files"].items():
        c = classify(f, t)
        if c not in ("MIT", "Apache-2.0", "Unicode-3.0", "MPL-2.0", "other", "NOTICE") and c in used.split(" AND "):
            lines += [x for x in copyright_lines(t) if x not in lines]
    if "Unicode-3.0" in used:
        lines += [x for x in uni_copy.get(crate, []) if x not in lines] + ["(Unicode-3.0 text below)"]
    if "MIT" in used.split(" AND ") and k not in ATTRIB_TEXT and not any(x.lower().startswith("copyright") for x in lines):
        if r["authors"]:
            lines.append(f"Copyright (c) {', '.join(r['authors'])} (authors per Cargo.toml; the crate's and the upstream "
                         "license files have no copyright line)")
        else:
            lines.append(f"Copyright (c) The {k[0]} contributors (no `authors` in Cargo.toml and no copyright line in the "
                         f"crate's or the upstream license files; repository {r['repository'] or 'not given'})")
            review.append((crate, "MIT, no copyright line anywhere and no `authors` in Cargo.toml: attributed to "
                                  f"\"The {k[0]} contributors\""))
    if not lines and r["authors"]:
        lines.append(f"Authors (Cargo.toml): {', '.join(r['authors'])}")
    for f, url, ref, kind in upstream_src.get(k, []):
        if f:
            lines.append(f"(upstream {f}: {url})")
        else:
            lines.append(f"(no license file in the package or upstream at {ref}; the standard SPDX {r['license']} text applies)")
    if not r["files"] or all(x.startswith(("upstream ", "SPDX ")) for x in r["files"]):
        note = (note + "; " if note else "") + ("license text: upstream" if any(x.startswith("upstream ") for x in r["files"]) else "license text: standard SPDX")
    lines += [x for x in NOTE_AFTER.get(k, []) if x not in lines]
    attrib[k] = (used, note, lines)

def scope(k):
    if cli_only_file:
        by = defaultdict(list)
        for x in cli_targets.get(k, "cli:?").split(","):
            b, _, t = x.partition(":")
            by[b].append(t)
        return "; ".join(f"{'CLI' if b == 'cli' else 'desktop'}: {', '.join(ts)}" for b, ts in sorted(by.items()))
    s = []
    if k in cli: s.append("cli")
    if k in desk: s.append("desktop")
    if not s: s.append("dev/other")
    return ",".join(s)

# ggml / llama.cpp provenance (Founder: YES, 2026-11-04). Text exact as ruled. Copyright line verified against the local
# llama.cpp checkout at 8642541b (LICENSE line 3). Files (Eng1 audit): adapted: crates/core/src/quant/q4k_repack.rs,
# q50_repack.rs, q50.rs, q6k.rs; spec/layout: q8k.rs, dequant.rs, backend/cuda_kernels.cu; method: forward/moe.rs.
GGML_TEXT = ("Portions of Artificer's CPU quantization code (Q4_K×8 and Q5_0×4 weight repacking and SIMD quantized dot-product "
             "kernels) are adapted from ggml / llama.cpp (https://github.com/ggml-org/llama.cpp), Copyright (c) 2023-2026 The ggml "
             "authors, MIT License.")
GGML_MIT = """MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
GGML_SPEC = ("GGUF format and quant block layouts implemented from the ggml GGUF specification; some tokenizer test fixtures "
             "were generated with llama.cpp.")
L = []
w = L.append
n_cli = sum(1 for k in all_reg if "cli:" in cli_targets.get(k, ""))
n_desk = sum(1 for k in all_reg if "desktop:" in cli_targets.get(k, ""))
mpl = sorted(k for k in all_reg if norm_expr(info[k]["license"]) == "MPL-2.0")
w("# Third-party notices\n")
w("> Covers every binary in the Artificer 0.1.0 release: the `artificer` command-line binary and the desktop app "
  "(`artificer-desktop`, Tauri 2) for Linux x86_64, Windows x86_64 and macOS aarch64. Generated from `Cargo.lock` by "
  "`scripts/third-party-notices/crates.sh` (the crate set: `cargo tree --offline --locked -e normal,build` for "
  "`artificer-cli` and `artificer-desktop` on `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` and "
  "`aarch64-apple-darwin`, the Linux CLI with and without `--features cuda`) and `scripts/third-party-notices/gen.py` "
  "(Python stdlib, no network). License fields come from each crate's vendored `Cargo.toml` and license texts from its "
  "own files in the cargo registry source; where a crate package ships no license file or no copyright line, the "
  "license file from its repository at the version tag (or the published commit) is used, vendored in "
  "`scripts/third-party-notices/upstream/` with its URL and sha256.\n")
w("## Scope and caveats\n")
w("- **Artificer itself** is dual-licensed MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). Its workspace crates are not listed here.")
w(f"- **{len(all_reg)}** crates.io packages: every crate compiled for a shipped binary on any of the three targets "
  f"({n_cli} for the CLI, {n_desk} for the desktop app; many are shared). Normal and build dependencies are included "
  "(build scripts and proc-macros are listed even though they are not linked into the binary); dev-dependencies are "
  "excluded. The Scope column says which binary and target compile each crate. The `cuda` feature adds no crates.")
w(f"- `Cargo.lock` lists {lock_total} crates.io packages for the whole workspace; the {lock_total - len(all_reg)} not listed here "
  "are only used by tests, benchmarks or tools, or only on targets we do not ship (see the coverage check at the end).")
w("- **CUDA:** no NVIDIA binaries or code ship with Artificer: no NVIDIA headers, PTX/cubin, cuBLAS stubs or NVRTC builtins are linked, embedded or bundled. `libcuda`, `libcudart`, `libcublas` and NVRTC are opened at run time with `dlopen` from the user's own CUDA installation, which is under NVIDIA's own license (EULA); NVRTC compiles only Artificer's own kernel source at run time. NVIDIA and CUDA are trademarks of NVIDIA Corporation. Artificer is not affiliated with or endorsed by NVIDIA.")
w("- **ggml / llama.cpp:** some CPU quantization code is adapted from ggml / llama.cpp (MIT); see the section below. Separately, the developer-only `ARTIFICER_GGML_FFI` / `ARTIFICER_CPU_REPACK` path can `dlopen` ggml CPU libraries already on the machine; no ggml library is bundled and normal inference never loads them.")
w("- **Desktop app, system libraries:** on Linux the desktop app links the system GTK 3, WebKitGTK 4.1, libsoup 3, "
  "JavaScriptCore, GLib, cairo, pango, gdk-pixbuf and D-Bus libraries dynamically (shared libraries of the user's "
  "distribution, under their own licenses, mostly LGPL); the download bundles no shared libraries. On macOS it uses "
  "the system WebKit framework. On Windows it uses the Microsoft Edge WebView2 Runtime installed on the system, and "
  "the Microsoft WebView2 loader is statically linked into the `.exe` (see the WebView2 section below).")
if any(k[0] == "wayland-client" for k in all_reg):
    w("- **Desktop app, Wayland (Linux only):** the file dialog (`tauri-plugin-dialog` / `rfd`, XDG desktop portal) "
      "uses the Wayland client crates on Linux. The system `libwayland-client` is a shared library of the user's "
      "distribution, loaded at run time and not bundled, so it needs no notice here. The Wayland crates "
      "(`wayland-backend`, `wayland-client`, `wayland-protocols`, `wayland-scanner`, `wayland-sys` and their "
      "dependencies) are compiled only for the Linux desktop app; the Scope column gives the binary and target of every crate.")
if mpl:
    w(f"- **MPL-2.0:** {len(mpl)} crates of the desktop app are under the Mozilla Public License 2.0; see the MPL-2.0 section for their source.")
w("- The license used for each crate (with the election for OR-licensed crates; `A/B` in a crate's metadata means "
  "`A OR B`) and its copyright lines are listed per crate below; each full license text is reproduced once.\n")

w("## Flagged crates\n")
if flags:
    w("| Crate | Scope | Flag |\n|---|---|---|")
    for c, f in flags:
        n, v = c.split(" ")
        w(f"| `{c}` | {scope((n, v))} | {f} |")
else:
    w("None.")
w("")
if review:
    w("Resolved by rule, listed for review:\n")
    for c, f in review:
        w(f"- `{c}`: {f}")
    w("")

w("## License summary\n")
if cli_only_file:
    w("| License expression | Crates |\n|---|---|")
    for expr, ks in sorted(groups.items(), key=lambda x: -len(x[1])):
        w(f"| `{expr}` | {len(ks)} |")
else:
    w("| License expression | Crates (all) | in CLI | in desktop |\n|---|---|---|---|")
    for expr, ks in sorted(groups.items(), key=lambda x: -len(x[1])):
        w(f"| `{expr}` | {len(ks)} | {sum(k in cli for k in ks)} | {sum(k in desk for k in ks)} |")
w("")

w("## Crates by license\n")
for expr, ks in sorted(groups.items(), key=lambda x: -len(x[1])):
    w(f"### `{expr}` ({len(ks)})\n")
    w("| Crate | Version | Scope | Repository |\n|---|---|---|---|")
    for k in ks:
        repo_url = info[k]["repository"] or ""
        w(f"| `{k[0]}` | {k[1]} | {scope(k)} | {repo_url} |")
    w("")

w("## Code adapted from ggml / llama.cpp\n")
w(GGML_TEXT + "\n")
w("```text\n" + GGML_MIT + "\n```\n")
w(GGML_SPEC + "\n")

if mpl:
    w("## MPL-2.0 components\n")
    w("These crates are used unmodified, as published on crates.io. Their source code (the Covered Software, in "
      "Source Code Form) is available at the links below:\n")
    w("| Crate | Version | Scope | crates.io | Source (docs.rs) | Upstream repository |\n|---|---|---|---|---|---|")
    for k in mpl:
        n, v = k
        ups = upstream_src.get(k, [])
        repo_url = (info[k]["repository"] or "").removesuffix(".git")
        if ups:
            f, url, ref, kind = ups[0]
            up = f"{repo_url}/tree/{ref} ({'tag ' + ref if kind == 'tag' else 'published commit ' + ref[:12]})"
        else:
            up = repo_url
        w(f"| `{n}` | {v} | {scope(k)} | https://crates.io/crates/{n}/{v} | https://docs.rs/crate/{n}/{v}/source/ | {up} |")
    w("")
    w("The MPL-2.0 code stays under MPL-2.0, and our license (MIT OR Apache-2.0) doesn't change its terms. The full "
      "MPL-2.0 text is reproduced once, in the License texts section.\n")

wv2 = {f: (u, t) for (n, v, f), (u, t) in upstream_meta.items() if n == "microsoft.web.webview2"}
if wv2 and any(k[0] == "webview2-com-sys" for k in all_reg):
    ver = next(v for (n, v, f) in upstream_meta if n == "microsoft.web.webview2")
    wk = next(k for k in all_reg if k[0] == "webview2-com-sys")
    w("## Microsoft WebView2 loader (Windows desktop app)\n")
    w(f"The Windows desktop app statically links Microsoft's WebView2 loader: `{wk[0]} {wk[1]}` (pulled in by "
      "`webview2-com`, used by `wry` / `tauri` on Windows) ships `WebView2LoaderStatic.lib` from the "
      f"Microsoft.Web.WebView2 {ver} NuGet package and links it with `kind = \"static\"` on the MSVC target, so "
      "`artificer-desktop.exe` imports no `WebView2Loader.dll`. The WebView2 Runtime itself is not shipped; it is "
      f"part of the user's Windows installation. Source of the texts below: {next(iter(wv2.values()))[0]}\n")
    for f in ("LICENSE.txt", "NOTICE.txt"):
        if f in wv2:
            w(f"### Microsoft.Web.WebView2 {ver}: {f}\n")
            w("```text\n" + wv2[f][1].strip().replace("```", "'''") + "\n```\n")

WL_XML = [  # (crate, version, file in the crate, what uses it)
    ("wayland-protocols", "0.32.13", "protocols/unstable/xdg-foreign/xdg-foreign-unstable-v2.xml",
     "`zxdg_exporter_v2` / `zxdg_exported_v2`, used by `rfd` to give the portal a handle to the app window"),
    ("wayland-client", "0.31.15", "wayland.xml", "the core Wayland protocol (`wl_registry`, `wl_surface`), used by `wayland-client`"),
]
wl = [(n, v, f, why) for n, v, f, why in WL_XML if (n, v) in all_reg]
if wl or ("rfd", "0.17.2") in all_reg:
    w("## Wayland and file dialog (Linux desktop app only)\n")
    vcs = lambda n, v: json.load(open(os.path.join(crate_dir(n, v), ".cargo_vcs_info.json")))["git"]
    if ("wayland-protocols", "0.32.13") in all_reg:
        g = vcs("wayland-protocols", "0.32.13")
        w("`wayland-protocols 0.32.13` ships the Wayland protocol XML files and their `protocols/COPYING`. There is no "
          "version tag for it in github.com/Smithay/wayland-rs; the published commit is "
          f"{g['sha1']} (https://github.com/Smithay/wayland-rs/tree/{g['sha1']}/wayland-protocols/protocols). "
          "`protocols/COPYING` in full:\n")
        t = info[("wayland-protocols", "0.32.13")]["files"]["protocols/COPYING"]
        w("```text\n" + t.strip().replace("```", "'''") + "\n```\n")
    for n, v, f, why in wl:
        t = open(os.path.join(crate_dir(n, v), f), encoding="utf-8").read()
        m = re.search(r"<copyright>(.*?)</copyright>", t, re.S)
        if not m:
            raise SystemExit(f"{n} {v} {f}: no <copyright> header")
        g = vcs(n, v)
        path = (g.get("path_in_vcs") or n) + "/" + f
        w(f"### `{n} {v}`: {f}\n")
        w(f"Defines {why}. Source: https://github.com/Smithay/wayland-rs/blob/{g['sha1']}/{path} (published commit). Header:\n")
        body = "\n".join(l.strip() for l in m.group(1).strip("\n").splitlines())
        w("```text\n" + body.strip() + "\n```\n")
    if ("rfd", "0.17.2") in all_reg:
        f = "src/backend/xdg_desktop_portal/window_identifier/LICENSE"
        w(f"### `rfd 0.17.2`: {f}\n")
        w("A second MIT notice inside `rfd`, for its window-identifier code, alongside the crate's own LICENSE "
          "(Copyright (c) 2022 Bartłomiej Maryńczak):\n")
        w("```text\n" + info[("rfd", "0.17.2")]["files"][f].strip().replace("```", "'''") + "\n```\n")

w("## License used and copyright per crate\n")
w("For OR-licensed crates the license Artificer uses is stated (\"elected\"). Dual MIT/Apache-2.0 crates whose MIT file has no "
  "copyright line are used under Apache-2.0. Lines in parentheses say where an attribution came from.\n")
w("| Crate | SPDX expression | License used | Copyright / attribution |\n|---|---|---|---|")
for k in sorted(all_reg):
    if k not in attrib: continue
    used, note, lines = attrib[k]
    cell = "<br>".join(l.replace("|", "\\|") for l in lines) if lines else "-"
    w(f"| `{k[0]} {k[1]}` | `{norm_expr(info[k]['license'])}` | {used}{' ('+note+')' if note else ''} | {cell} |")
w("")

if attr_files:
    w("## Per-crate COPYRIGHT, AUTHORS, NOTICE and third-party files\n")
    for c in sorted(attr_files):
        for f, t in sorted(attr_files[c].items()):
            w(f"### `{c}`: {f}\n")
            w("```text\n" + t.strip().replace("```", "\'\'\'") + "\n```\n")

w("## License texts\n")
w("### MIT\n")
mit_sample = None
for k in sorted(all_reg):
    for f, t in info[k]["files"].items():
        if classify(f, t) == "MIT" and "permission is hereby granted" in t.lower():
            mit_sample = (k, f, t); break
    if mit_sample: break
if mit_sample:
    body = mit_sample[2]
    i = body.lower().find("permission is hereby granted")
    w("Copyright (c) <year> <copyright holders> (see the per-crate table above)\n")
    w("```text\n" + body[i:].strip() + "\n```\n")
order = ["Apache-2.0", "Unicode-3.0"] + sorted(x for x in texts if x not in ("Apache-2.0", "Unicode-3.0"))
def text_needed(lic, user):
    """Keep a license text only for crates whose license used includes it (elections drop the rest)."""
    if lic == "other": return True
    crate = user.rsplit(" (", 1)[0]; n, v = crate.split(" ", 1)
    a = attrib.get((n, v))
    return a is None or lic in a[0].split(" AND ")
for lic in order:
    if lic not in texts: continue
    variants = {h: [t, [u for u in us if text_needed(lic, u)]] for h, (t, us) in texts[lic].items()}
    variants = {h: v for h, v in variants.items() if v[1]}
    if not variants: continue
    w(f"### {lic}\n" if lic != "other" else "### Other license files (pointers, SPDX metadata, combined COPYRIGHT files)\n")
    if lic == "MPL-2.0":
        users = sorted({u.rsplit(" (", 1)[0] for v in variants.values() for u in v[1]})
        std = SPDX_TEXT.get("MPL-2.0") or max(variants.values(), key=lambda v: len(v[1]))[0]
        w(f"Used by: {', '.join('`'+u+'`' for u in users)}. The crates' own copies are this text"
          " (aside from whitespace and, in some copies, `http://` instead of `https://` in the Exhibit A URL). Standard text from " + SPDX_URL.get("MPL-2.0", "the SPDX license list") + ":\n")
        w("```text\n" + std.strip() + "\n```\n")
        continue
    if lic == "Apache-2.0":
        # one full canonical text; list how many variant files exist
        full = [v for v in variants.values() if "END OF TERMS AND CONDITIONS" in v[0]]
        pick = max(full or variants.values(), key=lambda v: len(v[1]))
        users = sum(len(v[1]) for v in variants.values())
        w(f"{users} crate license files are Apache-2.0 ({len(variants)} textual variants, mostly whitespace / appendix differences). One full text:\n")
        w("```text\n" + pick[0].strip() + "\n```\n")
        continue
    for h, (text, users) in variants.items():
        w(f"Used by: {', '.join('`'+u+'`' for u in users)}\n")
        w("```text\n" + text.strip().replace("```", "'''") + "\n```\n")

# Coverage (Lawyer item 7): every shipped crate must appear in the per-crate table and in the crates-by-license tables.
in_tables = {k for ks in groups.values() for k in ks}
missing = sorted((all_reg - in_tables) | (all_reg - set(attrib)))
w("## Coverage check\n")
w(f"- Shipped crates (from `crates.txt`): {len(all_reg)}; listed above: {len(in_tables & set(attrib))}; "
  f"missing: {len(missing)}{' (' + ', '.join(f'{n} {v}' for n, v in missing) + ')' if missing else ''}.")
w(f"- `Cargo.lock` crates.io packages: {lock_total}; shipped: {len(all_reg)}; not shipped: {lock_total - len(all_reg)}.\n")
if missing:
    raise SystemExit(f"coverage: {len(missing)} shipped crates not covered: {missing}")
text = "\n".join(L).rstrip("\n") + "\n"
open(out_md, "w", newline="\n").write(text)
if json_out:
    lock_all = {key(p) for p in pkgs if "source" in p}
    json.dump({"lock_total": lock_total, "shipped": len(all_reg), "covered": len(in_tables & set(attrib)),
               "missing": [f"{n} {v}" for n, v in missing],
               "not_shipped": sorted(f"{n} {v}" for n, v in lock_all - all_reg),
               "total": len(all_reg), "cli": len(cli), "desktop": len(desk), "desktop_only": len(desk - cli),
               "dev_only": len(dev_only), "groups": {e: len(k) for e, k in groups.items()},
               "groups_cli": {e: sum(k in cli for k in ks) for e, ks in groups.items()},
               "flags": flags, "text_kinds": {k: len(v) for k, v in texts.items()},
               "dev_only_list": sorted(f"{n} {v}" for n, v in dev_only)}, open(json_out, "w"), indent=1)
