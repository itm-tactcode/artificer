#!/usr/bin/env python3
"""Fetch a crate's license files from its GitHub repository at the version tag (or, without a tag, at the commit in
the package's .cargo_vcs_info.json) into upstream/<name>-<version>/ and add them to upstream/SOURCES.tsv.

Usage: python3 scripts/third-party-notices/fetch_upstream.py NAME VERSION [NAME VERSION ...]
Needs `gh` (read-only GitHub API calls) and the crate in the local cargo registry. Run it only when the crate set
changes; gen.py itself never uses the network. Rows for non-crate texts (the WebView2 SDK license, SPDX standard
texts) are added by hand with their source URL and sha256.
Tags tried, in order: <name>-v<version>, <name>-<version>, <name>@<version>, <name>/v<version>, v<version>, <version>.
Files taken: LICENSE*, COPYING*, COPYRIGHT* in the crate's directory in the repository, else in the repository root.
"""
import base64, glob, hashlib, json, os, re, subprocess, sys, tomllib
HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, "upstream")
REG = glob.glob(os.path.join(os.environ.get("CARGO_HOME") or os.path.expanduser("~/.cargo"), "registry", "src", "*", ""))
LIC = re.compile(r"^(LICEN[CS]E|COPYING|COPYRIGHT)([-_.].*)?$", re.I)

def gh(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None

def fetch(n, v):
    d = next((os.path.join(r, f"{n}-{v}") for r in REG if os.path.isdir(os.path.join(r, f"{n}-{v}"))), None)
    if not d:
        raise SystemExit(f"{n} {v}: not in the cargo registry (run `cargo fetch`)")
    pk = tomllib.load(open(os.path.join(d, "Cargo.toml"), "rb"))["package"]
    vcs = {}
    if os.path.exists(os.path.join(d, ".cargo_vcs_info.json")):
        vcs = json.load(open(os.path.join(d, ".cargo_vcs_info.json")))
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)", pk.get("repository") or "")
    if not m:
        raise SystemExit(f"{n} {v}: repository is not on GitHub ({pk.get('repository')}); add it by hand")
    repo = f"{m.group(1)}/{m.group(2)}"
    tag = next((t for t in (f"{n}-v{v}", f"{n}-{v}", f"{n}@{v}", f"{n}/v{v}", f"v{v}", v)
                if (gh(f"repos/{repo}/git/ref/tags/{t}") or {}).get("ref") == f"refs/tags/{t}"), None)
    ref = tag or vcs.get("git", {}).get("sha1")
    kind = "tag" if tag else "commit (crate's .cargo_vcs_info.json; no version tag)"
    rows = []
    for sub in [vcs.get("path_in_vcs", "")] + [""]:
        lst = gh(f"repos/{repo}/contents/{sub}?ref={ref}" if sub else f"repos/{repo}/contents?ref={ref}")
        files = [e for e in (lst or []) if e.get("type") == "file" and LIC.match(e["name"])]
        if files:
            break
    if not files:
        return [[n, v, "-", ref, kind.split(" (")[0], f"https://github.com/{repo}/tree/{ref}", "-"]]
    os.makedirs(os.path.join(UP, f"{n}-{v}"), exist_ok=True)
    for e in files:
        data = base64.b64decode(gh(f"repos/{repo}/contents/{e['path']}?ref={ref}")["content"])
        open(os.path.join(UP, f"{n}-{v}", e["name"]), "wb").write(data)
        rows.append([n, v, e["name"], ref, kind, f"https://github.com/{repo}/blob/{ref}/{e['path']}",
                     hashlib.sha256(data).hexdigest()])
    return rows

args = sys.argv[1:]
if not args or len(args) % 2:
    raise SystemExit(__doc__)
path = os.path.join(UP, "SOURCES.tsv")
lines = open(path).read().splitlines() if os.path.exists(path) else ["crate\tversion\tfile\tref\tref_kind\turl\tsha256"]
for n, v in zip(args[::2], args[1::2]):
    lines = [l for l in lines if not l.startswith(f"{n}\t{v}\t")] + ["\t".join(r) for r in fetch(n, v)]
open(path, "w").write("\n".join(lines) + "\n")
