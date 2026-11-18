# THIRD-PARTY-NOTICES generator

`THIRD-PARTY-NOTICES.md` (repository root) covers every crate compiled into a shipped binary: the `artificer` CLI
and the desktop app (`artificer-desktop`) for Linux x86_64, Windows x86_64 and macOS aarch64. Regenerate it whenever
`Cargo.lock` changes, from the repository root:

```bash
bash scripts/third-party-notices/crates.sh          # cargo tree (offline, metadata only) -> crates.txt
python3 scripts/third-party-notices/gen.py . THIRD-PARTY-NOTICES.md
```

- `crates.sh`: the crate set, `cargo tree --offline --locked -e normal,build` for `artificer-cli` (Linux also with
  `--features cuda`) and `artificer-desktop` on the three target triples. Writes `crates.txt` (name, version and
  which binary/target compiles the crate).
- `gen.py`: Python 3.11+ stdlib, no cargo, no network. Reads `Cargo.lock`, `crates.txt`, each crate's vendored
  `Cargo.toml` and license files in the cargo registry, and `upstream/`. It fails if a shipped crate is not covered,
  and the file ends with the coverage check.
- `upstream/`: license files fetched once from the crates' repositories at the version tag (or the published
  commit when there is no tag), for crates whose package ships no license file or no copyright line. Also the
  Microsoft WebView2 SDK license and notice (its loader is statically linked into the Windows desktop app) and the
  SPDX standard MPL-2.0 text. `SOURCES.tsv` records the URL, ref and sha256 of every file; `gen.py` checks the
  sha256s. `fetch_upstream.py NAME VERSION ...` (needs `gh`) adds or refreshes crate rows.

Rules applied (Lawyer checklist, 2026-11-13): OR expressions (`A/B` is `A OR B`) name the license used; MIT without
a copyright line anywhere gets `Copyright (c) <authors per Cargo.toml>`; MPL-2.0 crates get source links and the
MPL-2.0 text once; each full license text is reproduced once.
