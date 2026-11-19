# Third-party notices

> Covers every binary in the Artificer 0.1.1 release: the `artificer` command-line binary and the desktop app (`artificer-desktop`, Tauri 2) for Linux x86_64, Windows x86_64 and macOS aarch64. Generated from `Cargo.lock` by `scripts/third-party-notices/crates.sh` (the crate set: `cargo tree --offline --locked -e normal,build` for `artificer-cli` and `artificer-desktop` on `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` and `aarch64-apple-darwin`, the Linux CLI with and without `--features cuda`) and `scripts/third-party-notices/gen.py` (Python stdlib, no network). License fields come from each crate's vendored `Cargo.toml` and license texts from its own files in the cargo registry source; where a crate package ships no license file or no copyright line, the license file from its repository at the version tag (or the published commit) is used, vendored in `scripts/third-party-notices/upstream/` with its URL and sha256.

## Scope and caveats

- **Artificer itself** is dual-licensed MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). Its workspace crates are not listed here.
- **361** crates.io packages: every crate compiled for a shipped binary on any of the three targets (88 for the CLI, 341 for the desktop app; many are shared). Normal and build dependencies are included (build scripts and proc-macros are listed even though they are not linked into the binary); dev-dependencies are excluded. The Scope column says which binary and target compile each crate. The `cuda` feature adds no crates.
- `Cargo.lock` lists 455 crates.io packages for the whole workspace; the 94 not listed here are only used by tests, benchmarks or tools, or only on targets we do not ship (see the coverage check at the end).
- **CUDA:** no NVIDIA binaries or code ship with Artificer: no NVIDIA headers, PTX/cubin, cuBLAS stubs or NVRTC builtins are linked, embedded or bundled. `libcuda`, `libcudart`, `libcublas` and NVRTC are opened at run time with `dlopen` from the user's own CUDA installation, which is under NVIDIA's own license (EULA); NVRTC compiles only Artificer's own kernel source at run time. NVIDIA and CUDA are trademarks of NVIDIA Corporation. Artificer is not affiliated with or endorsed by NVIDIA.
- **ggml / llama.cpp:** some CPU quantization code is adapted from ggml / llama.cpp (MIT); see the section below. Separately, the developer-only `ARTIFICER_GGML_FFI` / `ARTIFICER_CPU_REPACK` path can `dlopen` ggml CPU libraries already on the machine; no ggml library is bundled and normal inference never loads them.
- **Desktop app, system libraries:** on Linux the desktop app links the system GTK 3, WebKitGTK 4.1, libsoup 3, JavaScriptCore, GLib, cairo, pango, gdk-pixbuf and D-Bus libraries dynamically (shared libraries of the user's distribution, under their own licenses, mostly LGPL); the download bundles no shared libraries. On macOS it uses the system WebKit framework. On Windows it uses the Microsoft Edge WebView2 Runtime installed on the system, and the Microsoft WebView2 loader is statically linked into the `.exe` (see the WebView2 section below).
- **Desktop app, Wayland (Linux only):** the file dialog (`tauri-plugin-dialog` / `rfd`, XDG desktop portal) uses the Wayland client crates on Linux. The system `libwayland-client` is a shared library of the user's distribution, loaded at run time and not bundled, so it needs no notice here. The Wayland crates (`wayland-backend`, `wayland-client`, `wayland-protocols`, `wayland-scanner`, `wayland-sys` and their dependencies) are compiled only for the Linux desktop app; the Scope column gives the binary and target of every crate.
- **MPL-2.0:** 5 crates of the desktop app are under the Mozilla Public License 2.0; see the MPL-2.0 section for their source.
- The license used for each crate (with the election for OR-licensed crates; `A/B` in a crate's metadata means `A OR B`) and its copyright lines are listed per crate below; each full license text is reproduced once.

## Flagged crates

None.

Resolved by rule, listed for review:

- `objc2-foundation 0.3.2`: MIT, no copyright line anywhere and no `authors` in Cargo.toml: attributed to "The objc2-foundation contributors"

## License summary

| License expression | Crates |
|---|---|
| `MIT OR Apache-2.0` | 168 |
| `MIT` | 103 |
| `Apache-2.0 OR MIT` | 33 |
| `Unicode-3.0` | 18 |
| `Unlicense OR MIT` | 6 |
| `Zlib OR Apache-2.0 OR MIT` | 6 |
| `MPL-2.0` | 5 |
| `Apache-2.0` | 4 |
| `BSD-3-Clause` | 2 |
| `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | 2 |
| `MIT OR Zlib OR Apache-2.0` | 2 |
| `0BSD OR MIT OR Apache-2.0` | 1 |
| `BSD-3-Clause AND MIT` | 1 |
| `BSD-3-Clause OR MIT` | 1 |
| `Apache-2.0 AND MIT` | 1 |
| `CC0-1.0 OR MIT-0 OR Apache-2.0` | 1 |
| `Zlib` | 1 |
| `ISC` | 1 |
| `MIT AND BSD-3-Clause` | 1 |
| `MIT OR Apache-2.0 OR Zlib` | 1 |
| `Apache-2.0 OR BSL-1.0` | 1 |
| `Apache-2.0 WITH LLVM-exception` | 1 |
| `(MIT OR Apache-2.0) AND Unicode-3.0` | 1 |

## Crates by license

### `MIT OR Apache-2.0` (168)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `anstream` | 1.0.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-cli/anstyle.git |
| `anstyle` | 1.0.14 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-cli/anstyle.git |
| `anstyle-parse` | 1.0.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-cli/anstyle.git |
| `anstyle-query` | 1.1.5 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-cli/anstyle.git |
| `anstyle-wincon` | 3.0.11 | CLI: windows-x86_64 | https://github.com/rust-cli/anstyle.git |
| `anyhow` | 1.0.104 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/anyhow |
| `base64` | 0.23.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/marshallpierce/rust-base64 |
| `bitflags` | 1.3.2 | desktop: linux-x86_64 | https://github.com/bitflags/bitflags |
| `bitflags` | 2.13.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/bitflags/bitflags |
| `block-buffer` | 0.10.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/RustCrypto/utils |
| `camino` | 1.2.6 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/camino-rs/camino |
| `cargo-platform` | 0.1.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/cargo |
| `cc` | 1.10.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/cc-rs |
| `cfg-expr` | 0.15.8 | desktop: linux-x86_64 | https://github.com/EmbarkStudios/cfg-expr |
| `cfg-if` | 1.0.5 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/cfg-if |
| `clap` | 4.6.7 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/clap-rs/clap |
| `clap_builder` | 4.6.7 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/clap-rs/clap |
| `clap_derive` | 4.6.7 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/clap-rs/clap |
| `clap_lex` | 1.1.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/clap-rs/clap |
| `colorchoice` | 1.0.5 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-cli/anstyle.git |
| `cookie` | 0.18.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/SergioBenitez/cookie-rs |
| `core-foundation` | 0.10.1 | desktop: macos-aarch64 | https://github.com/servo/core-foundation-rs |
| `core-foundation-sys` | 0.8.7 | desktop: macos-aarch64 | https://github.com/servo/core-foundation-rs |
| `core-graphics` | 0.25.0 | desktop: macos-aarch64 | https://github.com/servo/core-foundation-rs |
| `core-graphics-types` | 0.2.0 | desktop: macos-aarch64 | https://github.com/servo/core-foundation-rs |
| `cpufeatures` | 0.2.17 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/RustCrypto/utils |
| `crc32fast` | 1.5.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/srijs/rust-crc32fast |
| `crossbeam-channel` | 0.5.17 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/crossbeam-rs/crossbeam |
| `crossbeam-utils` | 0.8.23 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/crossbeam-rs/crossbeam |
| `crypto-common` | 0.1.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/RustCrypto/traits |
| `deranged` | 0.5.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/jhpratt/deranged |
| `digest` | 0.10.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/RustCrypto/traits |
| `dirs` | 7.0.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://codeberg.org/dirs/dirs-rs |
| `dirs-sys` | 0.5.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dirs-dev/dirs-sys-rs |
| `displaydoc` | 0.2.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/yaahc/displaydoc |
| `downcast-rs` | 1.2.1 | desktop: linux-x86_64 | https://github.com/marcianx/downcast-rs |
| `dtoa` | 1.0.11 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/dtoa |
| `dyn-clone` | 1.0.20 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/dyn-clone |
| `embed_plist` | 1.2.2 | desktop: macos-aarch64 | https://github.com/nvzqz/embed-plist-rs |
| `erased-serde` | 0.4.11 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/erased-serde |
| `fdeflate` | 0.3.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/image-rs/fdeflate |
| `field-offset` | 0.3.6 | desktop: linux-x86_64 | https://github.com/Diggsey/rust-field-offset |
| `find-msvc-tools` | 0.1.15 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/cc-rs |
| `flate2` | 1.1.10 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/flate2-rs |
| `foreign-types` | 0.5.0 | desktop: macos-aarch64 | https://github.com/sfackler/foreign-types |
| `foreign-types-macros` | 0.2.4 | desktop: macos-aarch64 | https://github.com/sfackler/foreign-types |
| `foreign-types-shared` | 0.3.1 | desktop: macos-aarch64 | https://github.com/sfackler/foreign-types |
| `form_urlencoded` | 1.2.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-url |
| `futures-channel` | 0.3.34 | desktop: linux-x86_64 | https://github.com/rust-lang/futures-rs |
| `futures-core` | 0.3.34 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/futures-rs |
| `futures-executor` | 0.3.34 | desktop: linux-x86_64 | https://github.com/rust-lang/futures-rs |
| `futures-io` | 0.3.34 | desktop: linux-x86_64 | https://github.com/rust-lang/futures-rs |
| `futures-macro` | 0.3.34 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/futures-rs |
| `futures-task` | 0.3.34 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/futures-rs |
| `futures-util` | 0.3.34 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/futures-rs |
| `getrandom` | 0.3.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-random/getrandom |
| `getrandom` | 0.4.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-random/getrandom |
| `glob` | 0.3.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/glob |
| `hashbrown` | 0.12.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/hashbrown |
| `hashbrown` | 0.17.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/hashbrown |
| `heck` | 0.4.1 | desktop: linux-x86_64 | https://github.com/withoutboats/heck |
| `heck` | 0.5.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/withoutboats/heck |
| `html5ever` | 0.39.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/html5ever |
| `http` | 1.5.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hyperium/http |
| `httparse` | 1.10.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/seanmonstar/httparse |
| `httpdate` | 1.0.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/pyfisch/httpdate |
| `ident_case` | 1.0.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/TedDriggs/ident_case |
| `idna` | 1.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-url/ |
| `is_terminal_polyfill` | 1.70.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/polyfill-rs/is_terminal_polyfill |
| `itoa` | 1.0.18 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/itoa |
| `json-patch` | 4.2.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/idubrov/json-patch |
| `jsonptr` | 0.7.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/chanced/jsonptr |
| `keyboard-types` | 0.8.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-windowing/keyboard-types |
| `lazy_static` | 1.5.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang-nursery/lazy-static.rs |
| `libc` | 0.2.191 | CLI: linux-x86_64, macos-aarch64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/libc |
| `lock_api` | 0.4.14 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Amanieu/parking_lot |
| `log` | 0.4.34 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/log |
| `markup5ever` | 0.39.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/html5ever |
| `memmap2` | 0.9.12 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/RazrFalcon/memmap2-rs |
| `mime` | 0.3.17 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hyperium/mime |
| `num-conv` | 0.2.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/jhpratt/num-conv |
| `once_cell` | 1.21.4 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/matklad/once_cell |
| `once_cell_polyfill` | 1.70.2 | CLI: windows-x86_64 | https://github.com/polyfill-rs/once_cell_polyfill |
| `parking_lot` | 0.12.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Amanieu/parking_lot |
| `parking_lot_core` | 0.9.12 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Amanieu/parking_lot |
| `percent-encoding` | 2.3.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-url/ |
| `pkg-config` | 0.3.34 | desktop: linux-x86_64 | https://github.com/rust-lang/pkg-config-rs |
| `png` | 0.18.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/image-rs/image-png |
| `powerfmt` | 0.2.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/jhpratt/powerfmt |
| `proc-macro-crate` | 1.3.1 | desktop: linux-x86_64 | https://github.com/bkchr/proc-macro-crate |
| `proc-macro-crate` | 2.0.2 | desktop: linux-x86_64 | https://github.com/bkchr/proc-macro-crate |
| `proc-macro-error` | 1.0.4 | desktop: linux-x86_64 | https://gitlab.com/CreepySkeleton/proc-macro-error |
| `proc-macro-error-attr` | 1.0.4 | desktop: linux-x86_64 | https://gitlab.com/CreepySkeleton/proc-macro-error |
| `proc-macro2` | 1.0.107 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/proc-macro2 |
| `quote` | 1.0.47 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/quote |
| `regex` | 1.13.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/regex |
| `regex-automata` | 0.4.18 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/regex |
| `regex-syntax` | 0.8.11 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/regex |
| `rustc_version` | 0.4.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/djc/rustc-version-rs |
| `scoped-tls` | 1.0.1 | desktop: linux-x86_64 | https://github.com/alexcrichton/scoped-tls |
| `scopeguard` | 1.2.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/bluss/scopeguard |
| `semver` | 1.0.28 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/semver |
| `serde` | 1.0.229 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/serde-rs/serde |
| `serde-untagged` | 0.1.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/serde-untagged |
| `serde_core` | 1.0.229 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/serde-rs/serde |
| `serde_derive` | 1.0.229 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/serde-rs/serde |
| `serde_derive_internals` | 0.29.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/serde-rs/serde |
| `serde_json` | 1.0.154 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/serde-rs/json |
| `serde_path_to_error` | 0.1.20 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/path-to-error |
| `serde_repr` | 0.1.21 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/serde-repr |
| `serde_spanned` | 0.6.9 | desktop: linux-x86_64 | https://github.com/toml-rs/toml |
| `serde_spanned` | 1.1.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/toml-rs/toml |
| `serde_urlencoded` | 0.7.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/nox/serde_urlencoded |
| `serde_with` | 3.24.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/jonasbb/serde_with/ |
| `serde_with_macros` | 3.24.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/jonasbb/serde_with/ |
| `serialize-to-javascript` | 0.1.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/chippers/serialize-to-javascript |
| `serialize-to-javascript-impl` | 0.1.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/chippers/serialize-to-javascript |
| `servo_arc` | 0.4.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/stylo |
| `sha2` | 0.10.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/RustCrypto/hashes |
| `shlex` | 2.0.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/comex/rust-shlex |
| `siphasher` | 1.0.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/jedisct1/rust-siphash |
| `smallvec` | 1.16.4 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-smallvec |
| `socket2` | 0.6.5 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/socket2 |
| `softbuffer` | 0.4.8 | desktop: windows-x86_64 | https://github.com/rust-windowing/softbuffer |
| `stable_deref_trait` | 1.2.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/storyyeller/stable_deref_trait |
| `string_cache` | 0.9.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/string-cache |
| `string_cache_codegen` | 0.6.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/string-cache |
| `syn` | 1.0.109 | desktop: linux-x86_64 | https://github.com/dtolnay/syn |
| `syn` | 2.0.119 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/syn |
| `syn` | 3.0.9 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/syn |
| `system-deps` | 6.2.2 | desktop: linux-x86_64 | https://github.com/gdesmott/system-deps |
| `tendril` | 0.5.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/html5ever |
| `thiserror` | 1.0.69 | desktop: linux-x86_64 | https://github.com/dtolnay/thiserror |
| `thiserror` | 2.0.21 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/thiserror |
| `thiserror-impl` | 1.0.69 | desktop: linux-x86_64 | https://github.com/dtolnay/thiserror |
| `thiserror-impl` | 2.0.21 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/thiserror |
| `thread_local` | 1.1.10 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Amanieu/thread_local-rs |
| `time` | 0.3.55 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/time-rs/time |
| `time-core` | 0.1.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/time-rs/time |
| `time-macros` | 0.2.32 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/time-rs/time |
| `toml` | 0.8.2 | desktop: linux-x86_64 | https://github.com/toml-rs/toml |
| `toml` | 1.1.8+spec-1.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/toml-rs/toml |
| `toml_datetime` | 0.6.3 | desktop: linux-x86_64 | https://github.com/toml-rs/toml |
| `toml_datetime` | 1.1.2+spec-1.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/toml-rs/toml |
| `toml_edit` | 0.19.15 | desktop: linux-x86_64 | https://github.com/toml-rs/toml |
| `toml_edit` | 0.20.2 | desktop: linux-x86_64 | https://github.com/toml-rs/toml |
| `toml_parser` | 1.1.5+spec-1.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/toml-rs/toml |
| `toml_writer` | 1.1.3+spec-1.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/toml-rs/toml |
| `typeid` | 1.0.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/typeid |
| `typenum` | 1.20.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/paholg/typenum |
| `unicode-segmentation` | 1.13.3 | desktop: windows-x86_64 | https://github.com/unicode-rs/unicode-segmentation |
| `url` | 2.5.8 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-url |
| `version_check` | 0.9.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/SergioBenitez/version_check |
| `web-time` | 1.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/daxpedda/web-time |
| `web_atoms` | 0.2.6 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/html5ever |
| `windows` | 0.62.2 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-collections` | 0.3.2 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-core` | 0.62.2 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-future` | 0.3.2 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-implement` | 0.60.2 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-interface` | 0.59.3 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-link` | 0.2.1 | CLI: windows-x86_64; desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-numerics` | 0.3.1 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-result` | 0.4.1 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-strings` | 0.5.1 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-sys` | 0.61.2 | CLI: windows-x86_64; desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-threading` | 0.2.1 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |
| `windows-version` | 0.1.7 | desktop: windows-x86_64 | https://github.com/microsoft/windows-rs |

### `MIT` (103)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `atk` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `atk-sys` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `axum` | 0.8.9 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/axum |
| `axum-core` | 0.5.6 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/axum |
| `block2` | 0.6.2 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `bytes` | 1.12.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/bytes |
| `cairo-rs` | 0.18.5 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `cairo-sys-rs` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `cargo_metadata` | 0.19.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/oli-obk/cargo_metadata |
| `cfb` | 0.14.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mdsteele/rust-cfb |
| `darling` | 0.24.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/TedDriggs/darling |
| `darling_core` | 0.24.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/TedDriggs/darling |
| `darling_macro` | 0.24.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/TedDriggs/darling |
| `derive_more` | 2.1.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/JelteF/derive_more |
| `derive_more-impl` | 2.1.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/JelteF/derive_more |
| `dlib` | 0.5.3 | desktop: linux-x86_64 | https://github.com/elinorbgr/dlib |
| `dlopen2` | 0.8.2 | desktop: linux-x86_64 | https://github.com/OpenByteDev/dlopen2 |
| `dlopen2_derive` | 0.4.3 | desktop: linux-x86_64 | https://github.com/OpenByteDev/dlopen2 |
| `dom_query` | 0.28.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/niklak/dom_query |
| `embed-resource` | 3.0.12 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/nabijaczleweli/rust-embed-resource |
| `gdk` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `gdk-pixbuf` | 0.18.5 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `gdk-pixbuf-sys` | 0.18.0 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `gdk-sys` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `gdkwayland-sys` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `gdkx11` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `gdkx11-sys` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `generic-array` | 0.14.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/fizyk20/generic-array.git |
| `gio` | 0.18.4 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `gio-sys` | 0.18.1 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `glib` | 0.18.5 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `glib-macros` | 0.18.5 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `glib-sys` | 0.18.1 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `gobject-sys` | 0.18.0 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `gtk` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `gtk-sys` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `gtk3-macros` | 0.18.2 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk3-rs |
| `http-body` | 1.1.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hyperium/http-body |
| `http-body-util` | 0.1.5 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hyperium/http-body |
| `hyper` | 1.12.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hyperium/hyper |
| `hyper-util` | 0.1.21 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hyperium/hyper-util |
| `ico` | 0.6.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mdsteele/rust-ico |
| `infer` | 0.22.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/bojand/infer |
| `javascriptcore-rs` | 1.1.2 | desktop: linux-x86_64 | https://github.com/tauri-apps/javascriptcore-rs |
| `javascriptcore-rs-sys` | 1.1.1 | desktop: linux-x86_64 | https://github.com/tauri-apps/javascriptcore-rs |
| `matchers` | 0.2.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hawkw/matchers |
| `memoffset` | 0.9.1 | desktop: linux-x86_64 | https://github.com/Gilnaa/memoffset |
| `mio` | 1.2.4 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/mio |
| `new_debug_unreachable` | 1.0.6 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mbrubeck/rust-debug-unreachable |
| `nu-ansi-term` | 0.50.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/nushell/nu-ansi-term |
| `objc2` | 0.6.5 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-encode` | 4.1.0 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-foundation` | 0.3.2 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `pango` | 0.18.3 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `pango-sys` | 0.18.0 | desktop: linux-x86_64 | https://github.com/gtk-rs/gtk-rs-core |
| `phf` | 0.13.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-phf/rust-phf |
| `phf_codegen` | 0.13.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-phf/rust-phf |
| `phf_generator` | 0.13.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-phf/rust-phf |
| `phf_macros` | 0.13.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-phf/rust-phf |
| `phf_shared` | 0.13.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-phf/rust-phf |
| `plist` | 1.10.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/ebarnard/rust-plist/ |
| `precomputed-hash` | 0.1.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/emilio/precomputed-hash |
| `quick-xml` | 0.41.0 | desktop: linux-x86_64 | https://github.com/tafia/quick-xml |
| `quick-xml` | 0.42.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tafia/quick-xml |
| `rfd` | 0.17.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/PolyMeilex/rfd |
| `schemars` | 0.8.22 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/GREsau/schemars |
| `schemars_derive` | 0.8.22 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/GREsau/schemars |
| `sharded-slab` | 0.1.7 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hawkw/sharded-slab |
| `simd-adler32` | 0.3.10 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mcountryman/simd-adler32 |
| `slab` | 0.4.12 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/slab |
| `soup3` | 0.5.0 | desktop: linux-x86_64 | https://gitlab.gnome.org/World/Rust/soup3-rs |
| `soup3-sys` | 0.5.0 | desktop: linux-x86_64 | https://gitlab.gnome.org/World/Rust/soup3-rs |
| `strsim` | 0.11.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rapidfuzz/strsim-rs |
| `synstructure` | 0.14.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mystor/synstructure |
| `tauri-winres` | 0.3.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/winres |
| `tokio` | 1.54.1 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tokio |
| `tokio-macros` | 2.7.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tokio |
| `tower` | 0.5.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tower-rs/tower |
| `tower-http` | 0.6.11 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tower-rs/tower-http |
| `tower-layer` | 0.3.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tower-rs/tower |
| `tower-service` | 0.3.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tower-rs/tower |
| `tracing` | 0.1.44 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tracing |
| `tracing-attributes` | 0.1.31 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tracing |
| `tracing-core` | 0.1.36 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tracing |
| `tracing-log` | 0.2.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tracing |
| `tracing-subscriber` | 0.3.23 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tokio-rs/tracing |
| `urlpattern` | 0.6.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/denoland/rust-urlpattern |
| `version-compare` | 0.2.1 | desktop: linux-x86_64 | https://gitlab.com/timvisee/version-compare |
| `wayland-backend` | 0.3.17 | desktop: linux-x86_64 | https://github.com/smithay/wayland-rs |
| `wayland-client` | 0.31.15 | desktop: linux-x86_64 | https://github.com/smithay/wayland-rs |
| `wayland-protocols` | 0.32.13 | desktop: linux-x86_64 | https://github.com/smithay/wayland-rs |
| `wayland-scanner` | 0.31.11 | desktop: linux-x86_64 | https://github.com/smithay/wayland-rs |
| `wayland-sys` | 0.31.11 | desktop: linux-x86_64 | https://github.com/smithay/wayland-rs |
| `webkit2gtk` | 2.0.2 | desktop: linux-x86_64 | https://github.com/tauri-apps/webkit2gtk-rs |
| `webkit2gtk-sys` | 2.0.2 | desktop: linux-x86_64 | https://github.com/tauri-apps/webkit2gtk-rs |
| `webview2-com` | 0.39.1 | desktop: windows-x86_64 | https://github.com/wravery/webview2-rs |
| `webview2-com-macros` | 0.8.1 | desktop: windows-x86_64 | https://github.com/wravery/webview2-rs |
| `webview2-com-sys` | 0.39.1 | desktop: windows-x86_64 | https://github.com/wravery/webview2-rs |
| `winnow` | 0.5.40 | desktop: linux-x86_64 | https://github.com/winnow-rs/winnow |
| `winnow` | 1.0.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/winnow-rs/winnow |
| `x11` | 2.21.0 | desktop: linux-x86_64 | https://github.com/AltF02/x11-rs.git |
| `x11-dl` | 2.21.0 | desktop: linux-x86_64 | https://github.com/AltF02/x11-rs.git |
| `zmij` | 1.0.23 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/zmij |

### `Apache-2.0 OR MIT` (33)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `atomic-waker` | 1.1.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/smol-rs/atomic-waker |
| `autocfg` | 1.5.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/cuviper/autocfg |
| `bit-set` | 0.8.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/contain-rs/bit-set |
| `bit-vec` | 0.8.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/contain-rs/bit-vec |
| `cargo_toml` | 1.0.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://gitlab.com/lib.rs/cargo_toml |
| `ctor` | 1.0.13 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mmastrac/linktime |
| `dbus` | 0.9.12 | desktop: linux-x86_64 | https://github.com/diwic/dbus-rs |
| `equivalent` | 1.0.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/indexmap-rs/equivalent |
| `fastrand` | 2.5.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/smol-rs/fastrand |
| `fnv` | 1.0.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-fnv |
| `idna_adapter` | 1.2.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hsivonen/idna_adapter |
| `indexmap` | 1.9.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/bluss/indexmap |
| `indexmap` | 2.14.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/indexmap-rs/indexmap |
| `libdbus-sys` | 0.2.7 | desktop: linux-x86_64 | https://github.com/diwic/dbus-rs |
| `muda` | 0.20.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/muda |
| `pin-project-lite` | 0.2.17 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/taiki-e/pin-project-lite |
| `pollster` | 0.4.0 | desktop: linux-x86_64 | https://github.com/zesterer/pollster |
| `rustc-hash` | 2.1.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-lang/rustc-hash |
| `tauri` | 2.12.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-build` | 2.7.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-codegen` | 2.7.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-macros` | 2.7.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-plugin` | 2.7.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-plugin-dialog` | 2.8.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/plugins-workspace |
| `tauri-plugin-fs` | 2.7.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/plugins-workspace |
| `tauri-runtime` | 2.12.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-runtime-wry` | 2.12.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `tauri-utils` | 2.10.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri |
| `utf8_iter` | 1.0.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/hsivonen/utf8_iter |
| `utf8parse` | 0.2.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/alacritty/vte |
| `uuid` | 1.29.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/uuid-rs/uuid |
| `window-vibrancy` | 0.8.1 | desktop: macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tauri-plugin-vibrancy |
| `wry` | 0.57.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/wry |

### `Unicode-3.0` (18)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `icu_collections` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `icu_locale_core` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `icu_normalizer` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `icu_normalizer_data` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `icu_properties` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `icu_properties_data` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `icu_provider` | 2.4.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `litemap` | 0.8.3 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `potential_utf` | 0.1.6 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `tinystr` | 0.8.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `writeable` | 0.6.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `yoke` | 0.8.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `yoke-derive` | 0.8.4 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `zerofrom` | 0.1.8 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `zerofrom-derive` | 0.1.8 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `zerotrie` | 0.2.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `zerovec` | 0.11.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |
| `zerovec-derive` | 0.11.7 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/unicode-org/icu4x |

### `Unlicense OR MIT` (6)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `aho-corasick` | 1.1.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/BurntSushi/aho-corasick |
| `byteorder` | 1.5.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/BurntSushi/byteorder |
| `memchr` | 2.8.3 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/BurntSushi/memchr |
| `same-file` | 1.0.6 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/BurntSushi/same-file |
| `walkdir` | 2.5.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/BurntSushi/walkdir |
| `winapi-util` | 0.1.11 | desktop: windows-x86_64 | https://github.com/BurntSushi/winapi-util |

### `Zlib OR Apache-2.0 OR MIT` (6)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `dispatch2` | 0.3.1 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-app-kit` | 0.3.2 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-core-foundation` | 0.3.2 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-exception-helper` | 0.1.1 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-quartz-core` | 0.3.2 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |
| `objc2-web-kit` | 0.3.2 | desktop: macos-aarch64 | https://github.com/madsmtm/objc2 |

### `MPL-2.0` (5)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `cssparser` | 0.37.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-cssparser |
| `cssparser-macros` | 0.7.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/rust-cssparser |
| `dtoa-short` | 0.3.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/upsuper/dtoa-short |
| `option-ext` | 0.2.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/soc/option-ext.git |
| `selectors` | 0.38.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/servo/stylo |

### `Apache-2.0` (4)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `memo-map` | 0.3.4 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mitsuhiko/memo-map |
| `minijinja` | 2.24.0 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/mitsuhiko/minijinja |
| `sync_wrapper` | 1.0.2 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Actyx/sync_wrapper |
| `tao` | 0.37.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/tauri-apps/tao |

### `BSD-3-Clause` (2)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `alloc-no-stdlib` | 3.0.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dropbox/rust-alloc-no-stdlib |
| `alloc-stdlib` | 0.3.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dropbox/rust-alloc-no-stdlib |

### `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` (2)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `linux-raw-sys` | 0.12.1 | desktop: linux-x86_64 | https://github.com/sunfishcode/linux-raw-sys |
| `rustix` | 1.1.5 | desktop: linux-x86_64 | https://github.com/bytecodealliance/rustix |

### `MIT OR Zlib OR Apache-2.0` (2)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `miniz_oxide` | 0.8.9 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide |
| `miniz_oxide` | 0.9.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/Frommi/miniz_oxide/tree/master/miniz_oxide |

### `0BSD OR MIT OR Apache-2.0` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `adler2` | 2.0.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/oyvindln/adler2 |

### `BSD-3-Clause AND MIT` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `brotli` | 9.1.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dropbox/rust-brotli |

### `BSD-3-Clause OR MIT` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `brotli-decompressor` | 6.2.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dropbox/rust-brotli-decompressor |

### `Apache-2.0 AND MIT` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `dpi` | 0.1.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-windowing/winit |

### `CC0-1.0 OR MIT-0 OR Apache-2.0` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `dunce` | 1.0.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://gitlab.com/kornelski/dunce |

### `Zlib` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `foldhash` | 0.2.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/orlp/foldhash |

### `ISC` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `libloading` | 0.7.4 | desktop: linux-x86_64 | https://github.com/nagisa/rust_libloading/ |

### `MIT AND BSD-3-Clause` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `matchit` | 0.8.4 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/ibraheemdev/matchit |

### `MIT OR Apache-2.0 OR Zlib` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `raw-window-handle` | 0.6.2 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/rust-windowing/raw-window-handle |

### `Apache-2.0 OR BSL-1.0` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `ryu` | 1.0.23 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/ryu |

### `Apache-2.0 WITH LLVM-exception` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `target-lexicon` | 0.12.16 | desktop: linux-x86_64 | https://github.com/bytecodealliance/target-lexicon |

### `(MIT OR Apache-2.0) AND Unicode-3.0` (1)

| Crate | Version | Scope | Repository |
|---|---|---|---|
| `unicode-ident` | 1.0.27 | CLI: linux-x86_64, macos-aarch64, windows-x86_64; desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://github.com/dtolnay/unicode-ident |

## Code adapted from ggml / llama.cpp

Portions of Artificer's CPU quantization code (Q4_K×8 and Q5_0×4 weight repacking and SIMD quantized dot-product kernels) are adapted from ggml / llama.cpp (https://github.com/ggml-org/llama.cpp), Copyright (c) 2023-2026 The ggml authors, MIT License.

```text
MIT License

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
SOFTWARE.
```

GGUF format and quant block layouts implemented from the ggml GGUF specification; some tokenizer test fixtures were generated with llama.cpp.

## MPL-2.0 components

These crates are used unmodified, as published on crates.io. Their source code (the Covered Software, in Source Code Form) is available at the links below:

| Crate | Version | Scope | crates.io | Source (docs.rs) | Upstream repository |
|---|---|---|---|---|---|
| `cssparser` | 0.37.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://crates.io/crates/cssparser/0.37.0 | https://docs.rs/crate/cssparser/0.37.0/source/ | https://github.com/servo/rust-cssparser/tree/v0.37.0 (tag v0.37.0) |
| `cssparser-macros` | 0.7.1 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://crates.io/crates/cssparser-macros/0.7.1 | https://docs.rs/crate/cssparser-macros/0.7.1/source/ | https://github.com/servo/rust-cssparser/tree/cssparser-macros-v0.7.1 (tag cssparser-macros-v0.7.1) |
| `dtoa-short` | 0.3.5 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://crates.io/crates/dtoa-short/0.3.5 | https://docs.rs/crate/dtoa-short/0.3.5/source/ | https://github.com/upsuper/dtoa-short/tree/2d905cdb8b2e08163dc0d015f529877fe657b4ef (published commit 2d905cdb8b2e) |
| `option-ext` | 0.2.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://crates.io/crates/option-ext/0.2.0 | https://docs.rs/crate/option-ext/0.2.0/source/ | https://github.com/soc/option-ext/tree/272f22fc9ea1ac6b08f01704af52c4ac338df4e2 (published commit 272f22fc9ea1) |
| `selectors` | 0.38.0 | desktop: linux-x86_64, macos-aarch64, windows-x86_64 | https://crates.io/crates/selectors/0.38.0 | https://docs.rs/crate/selectors/0.38.0/source/ | https://github.com/servo/stylo/tree/572ecba2d1600e7c3d490586692a209faf703baa (published commit 572ecba2d160) |

The MPL-2.0 code stays under MPL-2.0, and our license (MIT OR Apache-2.0) doesn't change its terms. The full MPL-2.0 text is reproduced once, in the License texts section.

## Microsoft WebView2 loader (Windows desktop app)

The Windows desktop app statically links Microsoft's WebView2 loader: `webview2-com-sys 0.39.1` (pulled in by `webview2-com`, used by `wry` / `tauri` on Windows) ships `WebView2LoaderStatic.lib` from the Microsoft.Web.WebView2 1.0.3800.47 NuGet package and links it with `kind = "static"` on the MSVC target, so `artificer-desktop.exe` imports no `WebView2Loader.dll`. The WebView2 Runtime itself is not shipped; it is part of the user's Windows installation. Source of the texts below: https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.3800.47

### Microsoft.Web.WebView2 1.0.3800.47: LICENSE.txt

```text
Copyright (C) Microsoft Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * The name of Microsoft Corporation, or the names of its contributors 
may not be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Microsoft.Web.WebView2 1.0.3800.47: NOTICE.txt

```text
NOTICES AND INFORMATION
Do Not Translate or Localize

This software incorporates material from third parties. Microsoft makes certain
open source code available at https://3rdpartysource.microsoft.com, or you may
send a check or money order for US $5.00, including the product name, the open
source component name, and version number, to:

Source Code Compliance Team
Microsoft Corporation
One Microsoft Way
Redmond, WA 98052
USA

Notwithstanding any other terms, you may reverse engineer this software to the
extent required to debug changes to any libraries licensed under the GNU Lesser
General Public License.

----------------------------------------------------------------

Antlr3.Runtime 3.5.2-rc1 - BSD 3-Clause

[The "BSD license"]
Copyright (c) 2011 The ANTLR Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---------------------------------------------------------

---------------------------------------------------------

StringTemplate4 4.0.9-rc1 - BSD 3-Clause

[The "BSD license"]
Copyright (c) 2011 The ANTLR Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---------------------------------------------------------
```

## Wayland and file dialog (Linux desktop app only)

`wayland-protocols 0.32.13` ships the Wayland protocol XML files and their `protocols/COPYING`. There is no version tag for it in github.com/Smithay/wayland-rs; the published commit is 661cf141f5de4ec83559125b3d8bb1c79f25116b (https://github.com/Smithay/wayland-rs/tree/661cf141f5de4ec83559125b3d8bb1c79f25116b/wayland-protocols/protocols). `protocols/COPYING` in full:

```text
Copyright © 2008-2013 Kristian Høgsberg
Copyright © 2010-2013 Intel Corporation
Copyright © 2013      Rafael Antognolli
Copyright © 2013      Jasper St. Pierre
Copyright © 2014      Jonas Ådahl
Copyright © 2014      Jason Ekstrand
Copyright © 2014-2015 Collabora, Ltd.
Copyright © 2015      Red Hat Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next
paragraph) shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

---

The above is the version of the MIT "Expat" License used by X.org:

    http://cgit.freedesktop.org/xorg/xserver/tree/COPYING
```

### `wayland-protocols 0.32.13`: protocols/unstable/xdg-foreign/xdg-foreign-unstable-v2.xml

Defines `zxdg_exporter_v2` / `zxdg_exported_v2`, used by `rfd` to give the portal a handle to the app window. Source: https://github.com/Smithay/wayland-rs/blob/661cf141f5de4ec83559125b3d8bb1c79f25116b/wayland-protocols/protocols/unstable/xdg-foreign/xdg-foreign-unstable-v2.xml (published commit). Header:

```text
Copyright © 2015-2016 Red Hat Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next
paragraph) shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

### `wayland-client 0.31.15`: wayland.xml

Defines the core Wayland protocol (`wl_registry`, `wl_surface`), used by `wayland-client`. Source: https://github.com/Smithay/wayland-rs/blob/05196740da57e41b60a9e9f8e35079e9bd29b89a/wayland-client/wayland.xml (published commit). Header:

```text
Copyright © 2008-2011 Kristian Høgsberg
Copyright © 2010-2011 Intel Corporation
Copyright © 2012-2013 Collabora, Ltd.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice (including the
next paragraph) shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### `rfd 0.17.2`: src/backend/xdg_desktop_portal/window_identifier/LICENSE

A second MIT notice inside `rfd`, for its window-identifier code, alongside the crate's own LICENSE (Copyright (c) 2022 Bartłomiej Maryńczak):

```text
MIT License

Copyright (c) 2020 Bilal Elmoussaoui

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
SOFTWARE.
```

## License used and copyright per crate

For OR-licensed crates the license Artificer uses is stated ("elected"). Dual MIT/Apache-2.0 crates whose MIT file has no copyright line are used under Apache-2.0. Lines in parentheses say where an attribution came from.

| Crate | SPDX expression | License used | Copyright / attribution |
|---|---|---|---|
| `adler2 2.0.1` | `0BSD OR MIT OR Apache-2.0` | Apache-2.0 (elected from 0BSD OR MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Jonas Schievink, oyvindln |
| `aho-corasick 1.1.5` | `Unlicense OR MIT` | MIT (elected from Unlicense OR MIT) | Copyright (c) 2015 Andrew Gallant |
| `alloc-no-stdlib 3.0.0` | `BSD-3-Clause` | BSD-3-Clause | Copyright (c) 2016 Dropbox, Inc. |
| `alloc-stdlib 0.3.0` | `BSD-3-Clause` | BSD-3-Clause (license text: upstream) | Copyright (c) 2016 Dropbox, Inc.<br>(upstream LICENSE: https://github.com/dropbox/rust-alloc-no-stdlib/blob/0.3.0/LICENSE) |
| `anstream 1.0.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `anstyle 1.0.14` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `anstyle-parse 1.0.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `anstyle-query 1.1.5` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `anstyle-wincon 3.0.11` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `anyhow 1.0.104` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `atk 0.18.2` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `atk-sys 0.18.2` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk3-rs/blob/00133512bfc8bbb8e9be59b20e93406c6a3eeb21/atk/sys/LICENSE) |
| `atomic-waker 1.1.2` | `Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 OR MIT; no MIT copyright line) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors<br>(from the crate's LICENSE-THIRD-PARTY: code derived from futures-rs) |
| `autocfg 1.5.1` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2018 Josh Stone |
| `axum 0.8.9` | `MIT` | MIT | Copyright (c) 2019 axum Contributors |
| `axum-core 0.5.6` | `MIT` | MIT | Copyright (c) 2019–2025 axum Contributors |
| `base64 0.23.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2025 Alice Maz, Marshall Pierce |
| `bit-set 0.8.0` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2023 The Rust Project Developers |
| `bit-vec 0.8.0` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2023 The Rust Project Developers |
| `bitflags 1.3.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `bitflags 2.13.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `block-buffer 0.10.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018-2019 The RustCrypto Project Developers |
| `block2 0.6.2` | `MIT` | MIT (license text: upstream) | Copyright (c) Mads Marquart (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/block2-0.6.2/LICENSE.md) |
| `brotli 9.1.0` | `BSD-3-Clause AND MIT` | BSD-3-Clause AND MIT (all AND terms apply) | Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors.<br>Copyright (c) 2016 Dropbox, Inc. |
| `brotli-decompressor 6.2.0` | `BSD-3-Clause OR MIT` | BSD-3-Clause (elected from BSD-3-Clause OR MIT; the upstream LICENSE is BSD-3-Clause with the Dropbox copyright) | Copyright (c) 2016 Dropbox, Inc.<br>(upstream LICENSE: https://github.com/dropbox/rust-brotli-decompressor/blob/6.2.0/LICENSE) |
| `byteorder 1.5.0` | `Unlicense OR MIT` | MIT (elected from Unlicense OR MIT) | Copyright (c) 2015 Andrew Gallant |
| `bytes 1.12.1` | `MIT` | MIT | Copyright (c) 2018 Carl Lerche |
| `cairo-rs 0.18.5` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `cairo-sys-rs 0.18.2` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk-rs-core/blob/0.18.2/cairo/sys/LICENSE) |
| `camino 1.2.6` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Without Boats, Ashley Williams, Steve Klabnik, Rain |
| `cargo-platform 0.1.9` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | - |
| `cargo_metadata 0.19.2` | `MIT` | MIT | Copyright (c) Oliver Schneider (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE-MIT: https://github.com/oli-obk/cargo_metadata/blob/0.19.2/LICENSE-MIT) |
| `cargo_toml 1.0.1` | `Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 OR MIT; no MIT copyright line) | Authors (Cargo.toml): Kornel |
| `cc 1.10.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `cfb 0.14.0` | `MIT` | MIT | Copyright (c) 2017 Matthew D. Steele |
| `cfg-expr 0.15.8` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2019 Embark Studios |
| `cfg-if 1.0.5` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `clap 4.6.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `clap_builder 4.6.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `clap_derive 4.6.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `clap_lex 1.1.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `colorchoice 1.0.5` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `cookie 0.18.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 Sergio Benitez<br>Copyright (c) 2014 Alex Crichton |
| `core-foundation 0.10.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2012-2013 Mozilla Foundation |
| `core-foundation-sys 0.8.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2012-2013 Mozilla Foundation |
| `core-graphics 0.25.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2012-2013 Mozilla Foundation |
| `core-graphics-types 0.2.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2012-2013 Mozilla Foundation |
| `cpufeatures 0.2.17` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2020-2025 The RustCrypto Project Developers |
| `crc32fast 1.5.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018 Sam Rijs, Alex Crichton and contributors |
| `crossbeam-channel 0.5.17` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2019 The Crossbeam Project Developers<br>COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS<br>copyright protection under copyright law or other applicable laws.<br>Copyright (c) 2009 The Go Authors. All rights reserved.<br>copyright license to reproduce, prepare Derivative Works of,<br>(c) You must retain, in the Source form of any Derivative Works |
| `crossbeam-utils 0.8.23` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2019 The Crossbeam Project Developers |
| `crypto-common 0.1.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2021 RustCrypto Developers |
| `cssparser 0.37.0` | `MPL-2.0` | MPL-2.0 | Authors (Cargo.toml): Simon Sapin<br>(upstream LICENSE: https://github.com/servo/rust-cssparser/blob/v0.37.0/LICENSE) |
| `cssparser-macros 0.7.1` | `MPL-2.0` | MPL-2.0 | Authors (Cargo.toml): Simon Sapin<br>(upstream LICENSE: https://github.com/servo/rust-cssparser/blob/cssparser-macros-v0.7.1/macros/LICENSE) |
| `ctor 1.0.13` | `Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 OR MIT; no MIT copyright line) | Authors (Cargo.toml): Matt Mastracci |
| `darling 0.24.1` | `MIT` | MIT | Copyright (c) 2017 Ted Driggs |
| `darling_core 0.24.1` | `MIT` | MIT | Copyright (c) 2017 Ted Driggs |
| `darling_macro 0.24.1` | `MIT` | MIT | Copyright (c) 2017 Ted Driggs |
| `dbus 0.9.12` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2014-2018 David Henningsson <diwic@ubuntu.com> and other contributors |
| `deranged 0.5.9` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2024 Jacob Pratt et al. |
| `derive_more 2.1.1` | `MIT` | MIT | Copyright (c) 2016 Jelte Fennema |
| `derive_more-impl 2.1.1` | `MIT` | MIT | Copyright (c) 2016 Jelte Fennema |
| `digest 0.10.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 Artyom Pavlov |
| `dirs 7.0.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018-2019 dirs-rs contributors |
| `dirs-sys 0.5.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018-2019 dirs-rs contributors |
| `dispatch2 0.3.1` | `Zlib OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Zlib OR Apache-2.0 OR MIT; no MIT copyright line; license text: upstream) | Authors (Cargo.toml): Mads Marquart, Mary<br>(upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/dispatch2-0.3.1/LICENSE.md) |
| `displaydoc 0.2.7` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Jane Lusby |
| `dlib 0.5.3` | `MIT` | MIT | Copyright (c) 2015 Victor Berger |
| `dlopen2 0.8.2` | `MIT` | MIT (license text: upstream) | Copyright (c) 2017 Szymon Wieloch<br>Copyright (C) 2019 Ahmed Masud <ahmed.masud@saf.ai><br>Copyright (C) 2022 OpenByte <development.openbyte@gmail.com><br>(upstream LICENSE: https://github.com/OpenByteDev/dlopen2/blob/cc80e4a0a90d499b677fdf7743699b4b3a43a989/LICENSE) |
| `dlopen2_derive 0.4.3` | `MIT` | MIT (license text: upstream) | Copyright (c) 2017 Szymon Wieloch<br>Copyright (C) 2019 Ahmed Masud <ahmed.masud@saf.ai><br>Copyright (C) 2022 OpenByte <development.openbyte@gmail.com><br>(upstream LICENSE: https://github.com/OpenByteDev/dlopen2/blob/cc80e4a0a90d499b677fdf7743699b4b3a43a989/LICENSE) |
| `dom_query 0.28.0` | `MIT` | MIT | Copyright (c) 2023 Mykola Humanov |
| `downcast-rs 1.2.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2020 Ashish Myles and contributors |
| `dpi 0.1.2` | `Apache-2.0 AND MIT` | Apache-2.0 AND MIT (all AND terms apply) | Copyright (c) 2018 Jorge Aparicio<br>Copyright © 2005-2020 Rich Felker, et al.<br>Copyright © 1993,2004 Sun Microsystems or<br>Copyright © 2003-2011 David Schultz or<br>Copyright © 2003-2009 Steven G. Kargl or<br>Copyright © 2003-2009 Bruce D. Evans or<br>Copyright © 2008 Stephen L. Moshier or<br>Copyright © 2017-2018 Arm Limited |
| `dtoa 1.0.11` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `dtoa-short 0.3.5` | `MPL-2.0` | MPL-2.0 | Authors (Cargo.toml): Xidorn Quan<br>(upstream LICENSE: https://github.com/upsuper/dtoa-short/blob/2d905cdb8b2e08163dc0d015f529877fe657b4ef/LICENSE) |
| `dunce 1.0.5` | `CC0-1.0 OR MIT-0 OR Apache-2.0` | Apache-2.0 (elected from CC0-1.0 OR MIT-0 OR Apache-2.0) | Authors (Cargo.toml): Kornel |
| `dyn-clone 1.0.20` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `embed-resource 3.0.12` | `MIT` | MIT | Copyright (c) 2017 nabijaczleweli |
| `embed_plist 1.2.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2020 Nikolai Vazquez |
| `equivalent 1.0.3` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2016--2023 |
| `erased-serde 0.4.11` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `fastrand 2.5.0` | `Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 OR MIT; no MIT copyright line) | Authors (Cargo.toml): Stjepan Glavina |
| `fdeflate 0.3.7` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): The image-rs Developers |
| `field-offset 0.3.6` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016-2021 Diggory Blake, and other contributors. |
| `find-msvc-tools 0.1.15` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `flate2 1.1.10` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014-2026 Alex Crichton |
| `fnv 1.0.7` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 Contributors |
| `foldhash 0.2.0` | `Zlib` | Zlib | Copyright (c) 2024 Orson Peters |
| `foreign-types 0.5.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 The foreign-types Developers |
| `foreign-types-macros 0.2.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 The foreign-types Developers |
| `foreign-types-shared 0.3.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 The foreign-types Developers |
| `form_urlencoded 1.2.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2013-2016 The rust-url developers |
| `futures-channel 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `futures-core 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `futures-executor 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `futures-io 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `futures-macro 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `futures-task 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `futures-util 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Alex Crichton<br>Copyright (c) 2017 The Tokio Authors |
| `gdk 0.18.2` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `gdk-pixbuf 0.18.5` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `gdk-pixbuf-sys 0.18.0` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk-rs-core/blob/0.18.0/gdk-pixbuf/sys/LICENSE) |
| `gdk-sys 0.18.2` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk3-rs/blob/00133512bfc8bbb8e9be59b20e93406c6a3eeb21/gdk/sys/LICENSE) |
| `gdkwayland-sys 0.18.2` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk3-rs/blob/00133512bfc8bbb8e9be59b20e93406c6a3eeb21/gdkwayland/sys/LICENSE) |
| `gdkx11 0.18.2` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `gdkx11-sys 0.18.2` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk3-rs/blob/00133512bfc8bbb8e9be59b20e93406c6a3eeb21/gdkx11/sys/LICENSE) |
| `generic-array 0.14.7` | `MIT` | MIT | Copyright (c) 2015 Bartłomiej Kamiński |
| `getrandom 0.3.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018-2025 The rust-random Project Developers<br>Copyright (c) 2014 The Rust Project Developers |
| `getrandom 0.4.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018-2026 The rust-random Project Developers<br>Copyright (c) 2014 The Rust Project Developers |
| `gio 0.18.4` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `gio-sys 0.18.1` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk-rs-core/blob/0.18.1/gio/sys/LICENSE) |
| `glib 0.18.5` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `glib-macros 0.18.5` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `glib-sys 0.18.1` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk-rs-core/blob/0.18.1/glib/sys/LICENSE) |
| `glob 0.3.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `gobject-sys 0.18.0` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk-rs-core/blob/0.18.0/glib/gobject-sys/LICENSE) |
| `gtk 0.18.2` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `gtk-sys 0.18.2` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk3-rs/blob/00133512bfc8bbb8e9be59b20e93406c6a3eeb21/gtk/sys/LICENSE) |
| `gtk3-macros 0.18.2` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `hashbrown 0.12.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Amanieu d'Antras |
| `hashbrown 0.17.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Amanieu d'Antras |
| `heck 0.4.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 The Rust Project Developers |
| `heck 0.5.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 The Rust Project Developers |
| `html5ever 0.39.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The html5ever Project Developers |
| `http 1.5.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 http-rs authors |
| `http-body 1.1.0` | `MIT` | MIT | Copyright (c) 2019-2026 Sean McArthur & Hyper Contributors |
| `http-body-util 0.1.5` | `MIT` | MIT | Copyright (c) 2019-2026 Sean McArthur & Hyper Contributors |
| `httparse 1.10.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015-2025 Sean McArthur |
| `httpdate 1.0.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Pyfisch |
| `hyper 1.12.0` | `MIT` | MIT | Copyright (c) 2014-2026 Sean McArthur |
| `hyper-util 0.1.21` | `MIT` | MIT | Copyright (c) 2023-2025 Sean McArthur |
| `ico 0.6.0` | `MIT` | MIT | Copyright (c) 2018 Matthew D. Steele |
| `icu_collections 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `icu_locale_core 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `icu_normalizer 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `icu_normalizer_data 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `icu_properties 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `icu_properties_data 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `icu_provider 2.4.0` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `ident_case 1.0.1` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Ted Driggs |
| `idna 1.1.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2013-2025 The rust-url developers |
| `idna_adapter 1.2.2` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) The rust-url developers |
| `indexmap 1.9.3` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2016--2017 |
| `indexmap 2.14.2` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2016--2017 |
| `infer 0.22.0` | `MIT` | MIT | Copyright (c) 2019 Bojan |
| `is_terminal_polyfill 1.70.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `itoa 1.0.18` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `javascriptcore-rs 1.1.2` | `MIT` | MIT | Copyright (c) 2013-2021, The Gtk-rs Project Developers.<br>Copyright (c) 2021, Tauri Programme within The Commons Conservancy. |
| `javascriptcore-rs-sys 1.1.1` | `MIT` | MIT | Copyright (c) 2013-2017, The Gtk-rs Project Developers. |
| `json-patch 4.2.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 Ivan Dubrov |
| `jsonptr 0.7.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2022 Chance Dinkins |
| `keyboard-types 0.8.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 Pyfisch |
| `lazy_static 1.5.1` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Marvin Löbel |
| `libc 0.2.191` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) The Rust Project Developers |
| `libdbus-sys 0.2.7` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2014-2018 David Henningsson <diwic@ubuntu.com> and other contributors |
| `libloading 0.7.4` | `ISC` | ISC | Copyright © 2015, Simonas Kazlauskas |
| `linux-raw-sys 0.12.1` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT; no MIT copyright line) | Copyrights in the `linux-raw-sys` project are retained by their contributors. |
| `litemap 0.8.3` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `lock_api 0.4.14` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 The Rust Project Developers |
| `log 0.4.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `markup5ever 0.39.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The html5ever Project Developers |
| `matchers 0.2.0` | `MIT` | MIT | Copyright (c) 2019 Eliza Weisman |
| `matchit 0.8.4` | `MIT AND BSD-3-Clause` | MIT AND BSD-3-Clause (all AND terms apply) | Copyright (c) 2022 Ibraheem Ahmed<br>Copyright (c) 2013, Julien Schmidt |
| `memchr 2.8.3` | `Unlicense OR MIT` | MIT (elected from Unlicense OR MIT) | Copyright (c) 2015 Andrew Gallant |
| `memmap2 0.9.12` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2020 Yevhenii Reizner<br>Copyright (c) 2015 Dan Burkert |
| `memo-map 0.3.4` | `Apache-2.0` | Apache-2.0 | Authors (Cargo.toml): Armin Ronacher |
| `memoffset 0.9.1` | `MIT` | MIT | Copyright (c) 2017 Gilad Naaman |
| `mime 0.3.17` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Sean McArthur |
| `minijinja 2.24.0` | `Apache-2.0` | Apache-2.0 | Authors (Cargo.toml): Armin Ronacher |
| `miniz_oxide 0.8.9` | `MIT OR Zlib OR Apache-2.0` | MIT (elected from MIT OR Zlib OR Apache-2.0) | Copyright 2013-2014 RAD Game Tools and Valve Software<br>Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC<br>Copyright (c) 2017 Frommi<br>Copyright (c) 2017-2024 oyvindln |
| `miniz_oxide 0.9.1` | `MIT OR Zlib OR Apache-2.0` | MIT (elected from MIT OR Zlib OR Apache-2.0) | Copyright 2013-2014 RAD Game Tools and Valve Software<br>Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC<br>Copyright (c) 2017 Frommi<br>Copyright (c) 2017-2024 oyvindln |
| `mio 1.2.4` | `MIT` | MIT | Copyright (c) 2014 Carl Lerche and other MIO contributors |
| `muda 0.20.0` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2022-2022 Tauri Programme within The Commons Conservancy |
| `new_debug_unreachable 1.0.6` | `MIT` | MIT | Copyright (c) 2015 Jonathan Reem |
| `nu-ansi-term 0.50.3` | `MIT` | MIT | Copyright (c) 2014 Benjamin Sago<br>Copyright (c) 2021-2022 The Nushell Project Developers |
| `num-conv 0.2.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Jacob Pratt |
| `objc2 0.6.5` | `MIT` | MIT (license text: upstream) | Copyright (c) Mads Marquart (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/objc2-0.6.5/LICENSE.md) |
| `objc2-app-kit 0.3.2` | `Zlib OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Zlib OR Apache-2.0 OR MIT; no MIT copyright line; license text: upstream) | (upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md) |
| `objc2-core-foundation 0.3.2` | `Zlib OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Zlib OR Apache-2.0 OR MIT; no MIT copyright line; license text: upstream) | (upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md) |
| `objc2-encode 4.1.0` | `MIT` | MIT (license text: upstream) | Copyright (c) Mads Marquart (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/objc2-encode-4.1.0/LICENSE.md) |
| `objc2-exception-helper 0.1.1` | `Zlib OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Zlib OR Apache-2.0 OR MIT; no MIT copyright line; license text: upstream) | Authors (Cargo.toml): Mads Marquart<br>(upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/objc2-exception-helper-0.1.1/LICENSE.md) |
| `objc2-foundation 0.3.2` | `MIT` | MIT (license text: upstream) | Copyright (c) The objc2-foundation contributors (no `authors` in Cargo.toml and no copyright line in the crate's or the upstream license files; repository https://github.com/madsmtm/objc2)<br>(upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md) |
| `objc2-quartz-core 0.3.2` | `Zlib OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Zlib OR Apache-2.0 OR MIT; no MIT copyright line; license text: upstream) | (upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md) |
| `objc2-web-kit 0.3.2` | `Zlib OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Zlib OR Apache-2.0 OR MIT; no MIT copyright line; license text: upstream) | (upstream LICENSE.md: https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md) |
| `once_cell 1.21.4` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Aleksey Kladov |
| `once_cell_polyfill 1.70.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `option-ext 0.2.0` | `MPL-2.0` | MPL-2.0 | Authors (Cargo.toml): Simon Ochsenreither<br>(upstream LICENSE.txt: https://github.com/soc/option-ext/blob/272f22fc9ea1ac6b08f01704af52c4ac338df4e2/LICENSE.txt) |
| `pango 0.18.3` | `MIT` | MIT | Copyrights in the gtk-rs Project project are retained by their contributors. |
| `pango-sys 0.18.0` | `MIT` | MIT | Copyright (c) The gtk-rs Project Developers (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE: https://github.com/gtk-rs/gtk-rs-core/blob/0.18.0/pango/sys/LICENSE) |
| `parking_lot 0.12.5` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 The Rust Project Developers |
| `parking_lot_core 0.9.12` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 The Rust Project Developers |
| `percent-encoding 2.3.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2013-2025 The rust-url developers |
| `phf 0.13.1` | `MIT` | MIT | Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi |
| `phf_codegen 0.13.1` | `MIT` | MIT | Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi |
| `phf_generator 0.13.1` | `MIT` | MIT | Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi |
| `phf_macros 0.13.1` | `MIT` | MIT | Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi |
| `phf_shared 0.13.1` | `MIT` | MIT | Copyright (c) 2014-2022 Steven Fackler, Yuki Okushi |
| `pin-project-lite 0.2.17` | `Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 OR MIT; no MIT copyright line) | - |
| `pkg-config 0.3.34` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `plist 1.10.3` | `MIT` | MIT | Copyright (c) 2015 Edward Barnard |
| `png 0.18.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 nwin |
| `pollster 0.4.0` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2020-2021 Joshua Barretto |
| `potential_utf 0.1.6` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `powerfmt 0.2.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2023 Jacob Pratt et al. |
| `precomputed-hash 0.1.1` | `MIT` | MIT | Copyright (c) 2017 Emilio Cobos Álvarez |
| `proc-macro-crate 1.3.1` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Bastian Köcher |
| `proc-macro-crate 2.0.2` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Bastian Köcher |
| `proc-macro-error 1.0.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2019-2020 CreepySkeleton |
| `proc-macro-error-attr 1.0.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2019-2020 CreepySkeleton |
| `proc-macro2 1.0.107` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay, Alex Crichton |
| `quick-xml 0.41.0` | `MIT` | MIT | Copyright (c) 2016 Johann Tuffe |
| `quick-xml 0.42.0` | `MIT` | MIT | Copyright (c) 2016 Johann Tuffe |
| `quote 1.0.47` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `raw-window-handle 0.6.2` | `MIT OR Apache-2.0 OR Zlib` | MIT (elected from MIT OR Apache-2.0 OR Zlib) | Copyright (c) 2019 Osspial |
| `regex 1.13.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `regex-automata 0.4.18` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `regex-syntax 0.8.11` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The Rust Project Developers |
| `rfd 0.17.2` | `MIT` | MIT | Copyright (c) 2022 Bartłomiej Maryńczak<br>Copyright (c) 2020 Bilal Elmoussaoui<br>(second MIT notice, Bilal Elmoussaoui 2020: the crate's src/backend/xdg_desktop_portal/window_identifier/LICENSE, full text in the Wayland and file dialog section) |
| `rustc-hash 2.1.3` | `Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 OR MIT; no MIT copyright line) | Authors (Cargo.toml): The Rust Project Developers |
| `rustc_version 0.4.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 The Rust Project Developers |
| `rustix 1.1.5` | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | Apache-2.0 (elected from Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT; no MIT copyright line) | Copyrights in the `rustix` project are retained by their contributors. |
| `ryu 1.0.23` | `Apache-2.0 OR BSL-1.0` | Apache-2.0 (elected from Apache-2.0 OR BSL-1.0) | Authors (Cargo.toml): David Tolnay |
| `same-file 1.0.6` | `Unlicense OR MIT` | MIT (elected from Unlicense OR MIT) | Copyright (c) 2017 Andrew Gallant |
| `schemars 0.8.22` | `MIT` | MIT | Copyright (c) 2019 Graham Esau |
| `schemars_derive 0.8.22` | `MIT` | MIT | Copyright (c) 2019 Graham Esau |
| `scoped-tls 1.0.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `scopeguard 1.2.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016-2019 Ulrik Sverdrup "bluss" and scopeguard developers |
| `selectors 0.38.0` | `MPL-2.0` | MPL-2.0 (license text: standard SPDX) | Authors (Cargo.toml): The Servo Project Developers<br>(no license file in the package or upstream at 572ecba2d1600e7c3d490586692a209faf703baa; the standard SPDX MPL-2.0 text applies) |
| `semver 1.0.28` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `serde 1.0.229` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Erick Tryzelaar, David Tolnay |
| `serde-untagged 0.1.9` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `serde_core 1.0.229` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Erick Tryzelaar, David Tolnay |
| `serde_derive 1.0.229` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Erick Tryzelaar, David Tolnay |
| `serde_derive_internals 0.29.1` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Erick Tryzelaar, David Tolnay |
| `serde_json 1.0.154` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | - |
| `serde_path_to_error 0.1.20` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `serde_repr 0.1.21` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `serde_spanned 0.6.9` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `serde_spanned 1.1.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `serde_urlencoded 0.7.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 Anthony Ramine |
| `serde_with 3.24.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 |
| `serde_with_macros 3.24.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 |
| `serialize-to-javascript 0.1.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2021 Chip Reed |
| `serialize-to-javascript-impl 0.1.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2021 Chip Reed |
| `servo_arc 0.4.3` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): The Servo Project Developers |
| `sha2 0.10.9` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2006-2009 Graydon Hoare<br>Copyright (c) 2009-2013 Mozilla Foundation<br>Copyright (c) 2016 Artyom Pavlov |
| `sharded-slab 0.1.7` | `MIT` | MIT | Copyright (c) 2019 Eliza Weisman |
| `shlex 2.0.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 Nicholas Allegra (comex). |
| `simd-adler32 0.3.10` | `MIT` | MIT | Copyright (c) [2021] [Marvin Countryman] |
| `siphasher 1.0.4` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Frank Denis |
| `slab 0.4.12` | `MIT` | MIT | Copyright (c) 2019 Carl Lerche |
| `smallvec 1.16.4` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2018 The Servo Project Developers |
| `socket2 0.6.5` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `softbuffer 0.4.8` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright 2022 Kirill Chibisov |
| `soup3 0.5.0` | `MIT` | MIT | Copyright (c) 2013-2017, The Gtk-rs Project Developers. |
| `soup3-sys 0.5.0` | `MIT` | MIT | Copyright (c) 2013-2017, The Gtk-rs Project Developers. |
| `stable_deref_trait 1.2.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017 Robert Grosse |
| `string_cache 0.9.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2012-2013 Mozilla Foundation |
| `string_cache_codegen 0.6.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2012-2013 Mozilla Foundation |
| `strsim 0.11.1` | `MIT` | MIT | Copyright (c) 2015 Danny Guo<br>Copyright (c) 2016 Titus Wormer <tituswormer@gmail.com><br>Copyright (c) 2018 Akash Kurdekar |
| `syn 1.0.109` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `syn 2.0.119` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `syn 3.0.9` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `sync_wrapper 1.0.2` | `Apache-2.0` | Apache-2.0 | Authors (Cargo.toml): Actyx AG |
| `synstructure 0.14.0` | `MIT` | MIT | Copyright 2016 Nika Layzell |
| `system-deps 6.2.2` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): Guillaume Desmottes, Josh Triplett |
| `tao 0.37.2` | `Apache-2.0` | Apache-2.0 | Authors (Cargo.toml): Tauri Programme within The Commons Conservancy, The winit contributors |
| `target-lexicon 0.12.16` | `Apache-2.0 WITH LLVM-exception` | Apache-2.0 WITH LLVM-exception | Authors (Cargo.toml): Dan Gohman |
| `tauri 2.12.1` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-build 2.7.1` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-codegen 2.7.4` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-macros 2.7.4` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-plugin 2.7.2` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT; license text: upstream) | Copyright (c) 2017 - Present Tauri Apps Contributors<br>(upstream LICENSE-APACHE-2.0: https://github.com/tauri-apps/tauri/blob/tauri-plugin-v2.7.2/LICENSE-APACHE-2.0)<br>(upstream LICENSE-MIT: https://github.com/tauri-apps/tauri/blob/tauri-plugin-v2.7.2/LICENSE-MIT)<br>(upstream LICENSE.spdx: https://github.com/tauri-apps/tauri/blob/tauri-plugin-v2.7.2/LICENSE.spdx)<br>Credit: Tauri Apps Contributors. The crate package ships no license file; Apache-2.0 OR MIT per Cargo.toml, texts from the Tauri repository at tag tauri-plugin-v2.7.2 (as for zmij, from upstream) |
| `tauri-plugin-dialog 2.8.4` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-plugin-fs 2.7.1` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-runtime 2.12.2` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-runtime-wry 2.12.3` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-utils 2.10.2` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2017 - Present Tauri Apps Contributors |
| `tauri-winres 0.3.7` | `MIT` | MIT | Copyright (c) 2023 - Present Tauri Apps Contributors<br>Copyright (c) 2016 Max Resch |
| `tendril 0.5.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 Keegan McAllister |
| `thiserror 1.0.69` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `thiserror 2.0.21` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `thiserror-impl 1.0.69` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `thiserror-impl 2.0.21` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `thread_local 1.1.10` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2016 The Rust Project Developers |
| `time 0.3.55` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Jacob Pratt et al. |
| `time-core 0.1.9` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Jacob Pratt et al. |
| `time-macros 0.2.32` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Jacob Pratt et al. |
| `tinystr 0.8.5` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `tokio 1.54.1` | `MIT` | MIT | Copyright (c) Tokio Contributors |
| `tokio-macros 2.7.2` | `MIT` | MIT | Copyright (c) 2019 Yoshua Wuyts<br>Copyright (c) Tokio Contributors |
| `toml 0.8.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `toml 1.1.8+spec-1.1.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `toml_datetime 0.6.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Alex Crichton |
| `toml_datetime 1.1.2+spec-1.1.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `toml_edit 0.19.15` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `toml_edit 0.20.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `toml_parser 1.1.5+spec-1.1.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `toml_writer 1.1.3+spec-1.1.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Individual contributors |
| `tower 0.5.3` | `MIT` | MIT | Copyright (c) 2019 Tower Contributors |
| `tower-http 0.6.11` | `MIT` | MIT | Copyright (c) 2019-2021 Tower Contributors |
| `tower-layer 0.3.3` | `MIT` | MIT | Copyright (c) 2019 Tower Contributors |
| `tower-service 0.3.3` | `MIT` | MIT | Copyright (c) 2019 Tower Contributors |
| `tracing 0.1.44` | `MIT` | MIT | Copyright (c) 2019 Tokio Contributors |
| `tracing-attributes 0.1.31` | `MIT` | MIT | Copyright (c) 2019 Tokio Contributors |
| `tracing-core 0.1.36` | `MIT` | MIT | Copyright (c) 2019 Tokio Contributors |
| `tracing-log 0.2.0` | `MIT` | MIT | Copyright (c) 2019 Tokio Contributors |
| `tracing-subscriber 0.3.23` | `MIT` | MIT | Copyright (c) 2019 Tokio Contributors |
| `typeid 1.0.3` | `MIT OR Apache-2.0` | Apache-2.0 (elected from MIT OR Apache-2.0; no MIT copyright line) | Authors (Cargo.toml): David Tolnay |
| `typenum 1.20.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 Paho Lurie-Gregg |
| `unicode-ident 1.0.27` | `(MIT OR Apache-2.0) AND Unicode-3.0` | Apache-2.0 AND Unicode-3.0 (elected from MIT OR Apache-2.0; no MIT copyright line; all AND terms apply) | Copyright © 1991-2023 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `unicode-segmentation 1.13.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2015 The Rust Project Developers |
| `url 2.5.8` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2013-2025 The rust-url developers |
| `urlpattern 0.6.0` | `MIT` | MIT | Copyright (c) 2021 the Deno authors |
| `utf8_iter 1.0.4` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright Mozilla Foundation; portions adapted from the Rust standard library (The Rust Project Developers), Apache-2.0 OR MIT. |
| `utf8parse 0.2.2` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2016 Joe Wilm |
| `uuid 1.29.0` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2014 The Rust Project Developers<br>Copyright (c) 2018 Ashley Mannix, Christopher Armstrong, Dylan DPC, Hunar Roop Kahlon |
| `version-compare 0.2.1` | `MIT` | MIT | Copyright (c) 2017 Tim Visée |
| `version_check 0.9.5` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2017-2018 Sergio Benitez |
| `walkdir 2.5.0` | `Unlicense OR MIT` | MIT (elected from Unlicense OR MIT) | Copyright (c) 2015 Andrew Gallant |
| `wayland-backend 0.3.17` | `MIT` | MIT | Copyright (c) 2015 Elinor Berger |
| `wayland-client 0.31.15` | `MIT` | MIT | Copyright (c) 2015 Elinor Berger |
| `wayland-protocols 0.32.13` | `MIT` | MIT | Copyright (c) 2015 Elinor Berger<br>Copyright © 2008-2013 Kristian Høgsberg<br>Copyright © 2010-2013 Intel Corporation<br>Copyright © 2013      Rafael Antognolli<br>Copyright © 2013      Jasper St. Pierre<br>Copyright © 2014      Jonas Ådahl<br>Copyright © 2014      Jason Ekstrand<br>Copyright © 2014-2015 Collabora, Ltd.<br>Copyright © 2015      Red Hat Inc.<br>(protocols/COPYING in full, and the headers of the protocol XML files used, in the Wayland and file dialog section) |
| `wayland-scanner 0.31.11` | `MIT` | MIT | Copyright (c) 2015 Elinor Berger |
| `wayland-sys 0.31.11` | `MIT` | MIT | Copyright (c) 2015 Elinor Berger |
| `web-time 1.1.0` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2023 dAxpeDDa |
| `web_atoms 0.2.6` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) 2014 The html5ever Project Developers |
| `webkit2gtk 2.0.2` | `MIT` | MIT | Copyright (c) 2016 Boucher, Antoni <bouanto@zoho.com><br>Copyright (c) 2017-2021, The Gtk-rs Project Developers.<br>Copyright (c) 2021, Tauri Programme within The Commons Conservancy |
| `webkit2gtk-sys 2.0.2` | `MIT` | MIT | Copyright (c) 2016 Boucher, Antoni <bouanto@zoho.com> |
| `webview2-com 0.39.1` | `MIT` | MIT (license text: upstream) | Copyright (c) 2021 Bill Avery<br>(upstream LICENSE: https://github.com/wravery/webview2-rs/blob/edc2caf886175ccaebe86078c9cfe1ae2a187328/LICENSE) |
| `webview2-com-macros 0.8.1` | `MIT` | MIT (license text: upstream) | Copyright (c) 2021 Bill Avery<br>(upstream LICENSE: https://github.com/wravery/webview2-rs/blob/dffa41a8a46d3f5565eefbff2de57d38d399f158/LICENSE) |
| `webview2-com-sys 0.39.1` | `MIT` | MIT (license text: upstream) | Copyright (c) 2021 Bill Avery<br>(upstream LICENSE: https://github.com/wravery/webview2-rs/blob/edc2caf886175ccaebe86078c9cfe1ae2a187328/LICENSE) |
| `winapi-util 0.1.11` | `Unlicense OR MIT` | MIT (elected from Unlicense OR MIT) | Copyright (c) 2017 Andrew Gallant |
| `window-vibrancy 0.8.1` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2020-2022 Tauri Programme within The Commons Conservancy |
| `windows 0.62.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-collections 0.3.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-core 0.62.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-future 0.3.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-implement 0.60.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-interface 0.59.3` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-link 0.2.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-numerics 0.3.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-result 0.4.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-strings 0.5.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-sys 0.61.2` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-threading 0.2.1` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `windows-version 0.1.7` | `MIT OR Apache-2.0` | MIT (elected from MIT OR Apache-2.0) | Copyright (c) Microsoft Corporation. |
| `winnow 0.5.40` | `MIT` | MIT | Copyrights in winnow are retained by their contributors. No<br>copyright assignment is required to contribute to winnow.<br>(upstream COPYRIGHT: https://github.com/winnow-rs/winnow/blob/v0.5.40/COPYRIGHT)<br>(upstream LICENSE-MIT: https://github.com/winnow-rs/winnow/blob/v0.5.40/LICENSE-MIT) |
| `winnow 1.0.4` | `MIT` | MIT | Copyrights in winnow are retained by their contributors. No<br>copyright assignment is required to contribute to winnow.<br>(upstream COPYRIGHT: https://github.com/winnow-rs/winnow/blob/v1.0.4/COPYRIGHT)<br>(upstream LICENSE-MIT: https://github.com/winnow-rs/winnow/blob/v1.0.4/LICENSE-MIT) |
| `writeable 0.6.4` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `wry 0.57.1` | `Apache-2.0 OR MIT` | MIT (elected from Apache-2.0 OR MIT) | Copyright (c) 2020-2023 Ngo Iok Ui & Tauri Programme within The Commons Conservancy |
| `x11 2.21.0` | `MIT` | MIT | Copyright (c) daggerbot, Erle Pereira, AltF02 (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE-MIT: https://github.com/AltF02/x11-rs/blob/v2.21.0/x11/LICENSE-MIT) |
| `x11-dl 2.21.0` | `MIT` | MIT | Copyright (c) daggerbot, Erle Pereira, AltF02 (authors per Cargo.toml; the crate's and the upstream license files have no copyright line)<br>(upstream LICENSE-MIT: https://github.com/AltF02/x11-rs/blob/v2.21.0/x11-dl/LICENSE-MIT) |
| `yoke 0.8.4` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `yoke-derive 0.8.4` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `zerofrom 0.1.8` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `zerofrom-derive 0.1.8` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `zerotrie 0.2.5` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `zerovec 0.11.9` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `zerovec-derive 0.11.7` | `Unicode-3.0` | Unicode-3.0 | Copyright © 2020-2024 Unicode, Inc.<br>(Unicode-3.0 text below) |
| `zmij 1.0.23` | `MIT` | MIT | zmij 1.0.23, by David Tolnay (author per Cargo.toml `authors`; the upstream LICENSE-MIT carries no copyright notice). Used under the MIT License, full text in the License texts section. |

## Per-crate COPYRIGHT, AUTHORS, NOTICE and third-party files

### `atk 0.18.2`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `atomic-waker 1.1.2`: LICENSE-THIRD-PARTY

```text
===============================================================================

Copyright (c) 2016 Alex Crichton
Copyright (c) 2017 The Tokio Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

===============================================================================

Copyright (c) 2016 Alex Crichton
Copyright (c) 2017 The Tokio Authors

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

### `cairo-rs 0.18.5`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `core-graphics 0.25.0`: COPYRIGHT

```text
Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
option. All files in the project carrying such notice may not be
copied, modified, or distributed except according to those terms.
```

### `crossbeam-channel 0.5.17`: LICENSE-THIRD-PARTY

```text
===============================================================================

matching.go
https://creativecommons.org/licenses/by/3.0/legalcode

Creative Commons Legal Code

Attribution 3.0 Unported

    CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
    LEGAL SERVICES. DISTRIBUTION OF THIS LICENSE DOES NOT CREATE AN
    ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
    INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
    REGARDING THE INFORMATION PROVIDED, AND DISCLAIMS LIABILITY FOR
    DAMAGES RESULTING FROM ITS USE.

License

THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE
TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY
BE CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS
CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
CONDITIONS.

1. Definitions

 a. "Adaptation" means a work based upon the Work, or upon the Work and
    other pre-existing works, such as a translation, adaptation,
    derivative work, arrangement of music or other alterations of a
    literary or artistic work, or phonogram or performance and includes
    cinematographic adaptations or any other form in which the Work may be
    recast, transformed, or adapted including in any form recognizably
    derived from the original, except that a work that constitutes a
    Collection will not be considered an Adaptation for the purpose of
    this License. For the avoidance of doubt, where the Work is a musical
    work, performance or phonogram, the synchronization of the Work in
    timed-relation with a moving image ("synching") will be considered an
    Adaptation for the purpose of this License.
 b. "Collection" means a collection of literary or artistic works, such as
    encyclopedias and anthologies, or performances, phonograms or
    broadcasts, or other works or subject matter other than works listed
    in Section 1(f) below, which, by reason of the selection and
    arrangement of their contents, constitute intellectual creations, in
    which the Work is included in its entirety in unmodified form along
    with one or more other contributions, each constituting separate and
    independent works in themselves, which together are assembled into a
    collective whole. A work that constitutes a Collection will not be
    considered an Adaptation (as defined above) for the purposes of this
    License.
 c. "Distribute" means to make available to the public the original and
    copies of the Work or Adaptation, as appropriate, through sale or
    other transfer of ownership.
 d. "Licensor" means the individual, individuals, entity or entities that
    offer(s) the Work under the terms of this License.
 e. "Original Author" means, in the case of a literary or artistic work,
    the individual, individuals, entity or entities who created the Work
    or if no individual or entity can be identified, the publisher; and in
    addition (i) in the case of a performance the actors, singers,
    musicians, dancers, and other persons who act, sing, deliver, declaim,
    play in, interpret or otherwise perform literary or artistic works or
    expressions of folklore; (ii) in the case of a phonogram the producer
    being the person or legal entity who first fixes the sounds of a
    performance or other sounds; and, (iii) in the case of broadcasts, the
    organization that transmits the broadcast.
 f. "Work" means the literary and/or artistic work offered under the terms
    of this License including without limitation any production in the
    literary, scientific and artistic domain, whatever may be the mode or
    form of its expression including digital form, such as a book,
    pamphlet and other writing; a lecture, address, sermon or other work
    of the same nature; a dramatic or dramatico-musical work; a
    choreographic work or entertainment in dumb show; a musical
    composition with or without words; a cinematographic work to which are
    assimilated works expressed by a process analogous to cinematography;
    a work of drawing, painting, architecture, sculpture, engraving or
    lithography; a photographic work to which are assimilated works
    expressed by a process analogous to photography; a work of applied
    art; an illustration, map, plan, sketch or three-dimensional work
    relative to geography, topography, architecture or science; a
    performance; a broadcast; a phonogram; a compilation of data to the
    extent it is protected as a copyrightable work; or a work performed by
    a variety or circus performer to the extent it is not otherwise
    considered a literary or artistic work.
 g. "You" means an individual or entity exercising rights under this
    License who has not previously violated the terms of this License with
    respect to the Work, or who has received express permission from the
    Licensor to exercise rights under this License despite a previous
    violation.
 h. "Publicly Perform" means to perform public recitations of the Work and
    to communicate to the public those public recitations, by any means or
    process, including by wire or wireless means or public digital
    performances; to make available to the public Works in such a way that
    members of the public may access these Works from a place and at a
    place individually chosen by them; to perform the Work to the public
    by any means or process and the communication to the public of the
    performances of the Work, including by public digital performance; to
    broadcast and rebroadcast the Work by any means including signs,
    sounds or images.
 i. "Reproduce" means to make copies of the Work by any means including
    without limitation by sound or visual recordings and the right of
    fixation and reproducing fixations of the Work, including storage of a
    protected performance or phonogram in digital form or other electronic
    medium.

2. Fair Dealing Rights. Nothing in this License is intended to reduce,
limit, or restrict any uses free from copyright or rights arising from
limitations or exceptions that are provided for in connection with the
copyright protection under copyright law or other applicable laws.

3. License Grant. Subject to the terms and conditions of this License,
Licensor hereby grants You a worldwide, royalty-free, non-exclusive,
perpetual (for the duration of the applicable copyright) license to
exercise the rights in the Work as stated below:

 a. to Reproduce the Work, to incorporate the Work into one or more
    Collections, and to Reproduce the Work as incorporated in the
    Collections;
 b. to create and Reproduce Adaptations provided that any such Adaptation,
    including any translation in any medium, takes reasonable steps to
    clearly label, demarcate or otherwise identify that changes were made
    to the original Work. For example, a translation could be marked "The
    original work was translated from English to Spanish," or a
    modification could indicate "The original work has been modified.";
 c. to Distribute and Publicly Perform the Work including as incorporated
    in Collections; and,
 d. to Distribute and Publicly Perform Adaptations.
 e. For the avoidance of doubt:

     i. Non-waivable Compulsory License Schemes. In those jurisdictions in
        which the right to collect royalties through any statutory or
        compulsory licensing scheme cannot be waived, the Licensor
        reserves the exclusive right to collect such royalties for any
        exercise by You of the rights granted under this License;
    ii. Waivable Compulsory License Schemes. In those jurisdictions in
        which the right to collect royalties through any statutory or
        compulsory licensing scheme can be waived, the Licensor waives the
        exclusive right to collect such royalties for any exercise by You
        of the rights granted under this License; and,
   iii. Voluntary License Schemes. The Licensor waives the right to
        collect royalties, whether individually or, in the event that the
        Licensor is a member of a collecting society that administers
        voluntary licensing schemes, via that society, from any exercise
        by You of the rights granted under this License.

The above rights may be exercised in all media and formats whether now
known or hereafter devised. The above rights include the right to make
such modifications as are technically necessary to exercise the rights in
other media and formats. Subject to Section 8(f), all rights not expressly
granted by Licensor are hereby reserved.

4. Restrictions. The license granted in Section 3 above is expressly made
subject to and limited by the following restrictions:

 a. You may Distribute or Publicly Perform the Work only under the terms
    of this License. You must include a copy of, or the Uniform Resource
    Identifier (URI) for, this License with every copy of the Work You
    Distribute or Publicly Perform. You may not offer or impose any terms
    on the Work that restrict the terms of this License or the ability of
    the recipient of the Work to exercise the rights granted to that
    recipient under the terms of the License. You may not sublicense the
    Work. You must keep intact all notices that refer to this License and
    to the disclaimer of warranties with every copy of the Work You
    Distribute or Publicly Perform. When You Distribute or Publicly
    Perform the Work, You may not impose any effective technological
    measures on the Work that restrict the ability of a recipient of the
    Work from You to exercise the rights granted to that recipient under
    the terms of the License. This Section 4(a) applies to the Work as
    incorporated in a Collection, but this does not require the Collection
    apart from the Work itself to be made subject to the terms of this
    License. If You create a Collection, upon notice from any Licensor You
    must, to the extent practicable, remove from the Collection any credit
    as required by Section 4(b), as requested. If You create an
    Adaptation, upon notice from any Licensor You must, to the extent
    practicable, remove from the Adaptation any credit as required by
    Section 4(b), as requested.
 b. If You Distribute, or Publicly Perform the Work or any Adaptations or
    Collections, You must, unless a request has been made pursuant to
    Section 4(a), keep intact all copyright notices for the Work and
    provide, reasonable to the medium or means You are utilizing: (i) the
    name of the Original Author (or pseudonym, if applicable) if supplied,
    and/or if the Original Author and/or Licensor designate another party
    or parties (e.g., a sponsor institute, publishing entity, journal) for
    attribution ("Attribution Parties") in Licensor's copyright notice,
    terms of service or by other reasonable means, the name of such party
    or parties; (ii) the title of the Work if supplied; (iii) to the
    extent reasonably practicable, the URI, if any, that Licensor
    specifies to be associated with the Work, unless such URI does not
    refer to the copyright notice or licensing information for the Work;
    and (iv) , consistent with Section 3(b), in the case of an Adaptation,
    a credit identifying the use of the Work in the Adaptation (e.g.,
    "French translation of the Work by Original Author," or "Screenplay
    based on original Work by Original Author"). The credit required by
    this Section 4 (b) may be implemented in any reasonable manner;
    provided, however, that in the case of a Adaptation or Collection, at
    a minimum such credit will appear, if a credit for all contributing
    authors of the Adaptation or Collection appears, then as part of these
    credits and in a manner at least as prominent as the credits for the
    other contributing authors. For the avoidance of doubt, You may only
    use the credit required by this Section for the purpose of attribution
    in the manner set out above and, by exercising Your rights under this
    License, You may not implicitly or explicitly assert or imply any
    connection with, sponsorship or endorsement by the Original Author,
    Licensor and/or Attribution Parties, as appropriate, of You or Your
    use of the Work, without the separate, express prior written
    permission of the Original Author, Licensor and/or Attribution
    Parties.
 c. Except as otherwise agreed in writing by the Licensor or as may be
    otherwise permitted by applicable law, if You Reproduce, Distribute or
    Publicly Perform the Work either by itself or as part of any
    Adaptations or Collections, You must not distort, mutilate, modify or
    take other derogatory action in relation to the Work which would be
    prejudicial to the Original Author's honor or reputation. Licensor
    agrees that in those jurisdictions (e.g. Japan), in which any exercise
    of the right granted in Section 3(b) of this License (the right to
    make Adaptations) would be deemed to be a distortion, mutilation,
    modification or other derogatory action prejudicial to the Original
    Author's honor and reputation, the Licensor will waive or not assert,
    as appropriate, this Section, to the fullest extent permitted by the
    applicable national law, to enable You to reasonably exercise Your
    right under Section 3(b) of this License (right to make Adaptations)
    but not otherwise.

5. Representations, Warranties and Disclaimer

UNLESS OTHERWISE MUTUALLY AGREED TO BY THE PARTIES IN WRITING, LICENSOR
OFFERS THE WORK AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY
KIND CONCERNING THE WORK, EXPRESS, IMPLIED, STATUTORY OR OTHERWISE,
INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTIBILITY,
FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF
LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OF ABSENCE OF ERRORS,
WHETHER OR NOT DISCOVERABLE. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION
OF IMPLIED WARRANTIES, SO SUCH EXCLUSION MAY NOT APPLY TO YOU.

6. Limitation on Liability. EXCEPT TO THE EXTENT REQUIRED BY APPLICABLE
LAW, IN NO EVENT WILL LICENSOR BE LIABLE TO YOU ON ANY LEGAL THEORY FOR
ANY SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES
ARISING OUT OF THIS LICENSE OR THE USE OF THE WORK, EVEN IF LICENSOR HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

7. Termination

 a. This License and the rights granted hereunder will terminate
    automatically upon any breach by You of the terms of this License.
    Individuals or entities who have received Adaptations or Collections
    from You under this License, however, will not have their licenses
    terminated provided such individuals or entities remain in full
    compliance with those licenses. Sections 1, 2, 5, 6, 7, and 8 will
    survive any termination of this License.
 b. Subject to the above terms and conditions, the license granted here is
    perpetual (for the duration of the applicable copyright in the Work).
    Notwithstanding the above, Licensor reserves the right to release the
    Work under different license terms or to stop distributing the Work at
    any time; provided, however that any such election will not serve to
    withdraw this License (or any other license that has been, or is
    required to be, granted under the terms of this License), and this
    License will continue in full force and effect unless terminated as
    stated above.

8. Miscellaneous

 a. Each time You Distribute or Publicly Perform the Work or a Collection,
    the Licensor offers to the recipient a license to the Work on the same
    terms and conditions as the license granted to You under this License.
 b. Each time You Distribute or Publicly Perform an Adaptation, Licensor
    offers to the recipient a license to the original Work on the same
    terms and conditions as the license granted to You under this License.
 c. If any provision of this License is invalid or unenforceable under
    applicable law, it shall not affect the validity or enforceability of
    the remainder of the terms of this License, and without further action
    by the parties to this agreement, such provision shall be reformed to
    the minimum extent necessary to make such provision valid and
    enforceable.
 d. No term or provision of this License shall be deemed waived and no
    breach consented to unless such waiver or consent shall be in writing
    and signed by the party to be charged with such waiver or consent.
 e. This License constitutes the entire agreement between the parties with
    respect to the Work licensed here. There are no understandings,
    agreements or representations with respect to the Work not specified
    here. Licensor shall not be bound by any additional provisions that
    may appear in any communication from You. This License may not be
    modified without the mutual written agreement of the Licensor and You.
 f. The rights granted under, and the subject matter referenced, in this
    License were drafted utilizing the terminology of the Berne Convention
    for the Protection of Literary and Artistic Works (as amended on
    September 28, 1979), the Rome Convention of 1961, the WIPO Copyright
    Treaty of 1996, the WIPO Performances and Phonograms Treaty of 1996
    and the Universal Copyright Convention (as revised on July 24, 1971).
    These rights and subject matter take effect in the relevant
    jurisdiction in which the License terms are sought to be enforced
    according to the corresponding provisions of the implementation of
    those treaty provisions in the applicable national law. If the
    standard suite of rights granted under applicable copyright law
    includes additional rights not granted under this License, such
    additional rights are deemed to be included in the License; this
    License is not intended to restrict the license of any rights under
    applicable law.


Creative Commons Notice

    Creative Commons is not a party to this License, and makes no warranty
    whatsoever in connection with the Work. Creative Commons will not be
    liable to You or any party on any legal theory for any damages
    whatsoever, including without limitation any general, special,
    incidental or consequential damages arising in connection to this
    license. Notwithstanding the foregoing two (2) sentences, if Creative
    Commons has expressly identified itself as the Licensor hereunder, it
    shall have all rights and obligations of Licensor.

    Except for the limited purpose of indicating to the public that the
    Work is licensed under the CCPL, Creative Commons does not authorize
    the use by either party of the trademark "Creative Commons" or any
    related trademark or logo of Creative Commons without the prior
    written consent of Creative Commons. Any permitted use will be in
    compliance with Creative Commons' then-current trademark usage
    guidelines, as may be published on its website or otherwise made
    available upon request from time to time. For the avoidance of doubt,
    this trademark restriction does not form part of this License.

    Creative Commons may be contacted at https://creativecommons.org/.

===============================================================================

The Go Programming Language
https://golang.org/LICENSE

Copyright (c) 2009 The Go Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

===============================================================================

The Rust Programming Language
https://github.com/rust-lang/rust/blob/master/LICENSE-MIT

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

===============================================================================

The Rust Programming Language
https://github.com/rust-lang/rust/blob/master/LICENSE-APACHE

                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

   To apply the Apache License to your work, attach the following
   boilerplate notice, with the fields enclosed by brackets "[]"
   replaced with your own identifying information. (Don't include
   the brackets!)  The text should be enclosed in the appropriate
   comment syntax for the file format. We also recommend that a
   file or class name and description of purpose be included on the
   same "printed page" as the copyright notice for easier
   identification within third-party archives.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### `gdk 0.18.2`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `gdk-pixbuf 0.18.5`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `gdkx11 0.18.2`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `gio 0.18.4`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `glib 0.18.5`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `glib-macros 0.18.5`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `gtk 0.18.2`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `gtk3-macros 0.18.2`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `linux-raw-sys 0.12.1`: COPYRIGHT

```text
Short version for non-lawyers:

`linux-raw-sys` is triple-licensed under Apache 2.0 with the LLVM Exception,
Apache 2.0, and MIT terms.


Longer version:

Copyrights in the `linux-raw-sys` project are retained by their contributors.
No copyright assignment is required to contribute to the `linux-raw-sys`
project.

Some files include code derived from Rust's `libstd`; see the comments in
the code for details.

Except as otherwise noted (below and/or in individual files), `linux-raw-sys`
is licensed under:

 - the Apache License, Version 2.0, with the LLVM Exception
   <LICENSE-Apache-2.0_WITH_LLVM-exception> or
   <http://llvm.org/foundation/relicensing/LICENSE.txt>
 - the Apache License, Version 2.0
   <LICENSE-APACHE> or
   <http://www.apache.org/licenses/LICENSE-2.0>,
 - or the MIT license
   <LICENSE-MIT> or
   <http://opensource.org/licenses/MIT>,

at your option.
```

### `pango 0.18.3`: COPYRIGHT

```text
The gtk-rs Project is licensed under the MIT license, see the LICENSE file or
<http://opensource.org/licenses/MIT>.

Copyrights in the gtk-rs Project project are retained by their contributors.
No copyright assignment is required to contribute to the gtk-rs Project
project.

For full authorship information, see the version control history.

This project provides interoperability with various GNOME libraries but
doesn't distribute any parts of them. Distributing compiled libraries and
executables that link to those libraries may be subject to terms of the GNU
LGPL or other licenses. For more information check the license of each GNOME
library.
```

### `rustix 1.1.5`: COPYRIGHT

```text
Short version for non-lawyers:

`rustix` is triple-licensed under Apache 2.0 with the LLVM Exception,
Apache 2.0, and MIT terms.


Longer version:

Copyrights in the `rustix` project are retained by their contributors.
No copyright assignment is required to contribute to the `rustix`
project.

Some files include code derived from Rust's `libstd`; see the comments in
the code for details.

Except as otherwise noted (below and/or in individual files), `rustix`
is licensed under:

 - the Apache License, Version 2.0, with the LLVM Exception
   <LICENSE-Apache-2.0_WITH_LLVM-exception> or
   <http://llvm.org/foundation/relicensing/LICENSE.txt>
 - the Apache License, Version 2.0
   <LICENSE-APACHE> or
   <http://www.apache.org/licenses/LICENSE-2.0>,
 - or the MIT license
   <LICENSE-MIT> or
   <http://opensource.org/licenses/MIT>,

at your option.
```

### `unicode-segmentation 1.13.3`: COPYRIGHT

```text
Licensed under the Apache License, Version 2.0
<LICENSE-APACHE or
http://www.apache.org/licenses/LICENSE-2.0> or the MIT
license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
at your option. All files in the project carrying such
notice may not be copied, modified, or distributed except
according to those terms.
```

### `utf8_iter 1.0.4`: COPYRIGHT

```text
Copyright Mozilla Foundation

Licensed under the Apache License (Version 2.0), or the MIT license,
(the "Licenses") at your option. You may not use this file except in
compliance with one of the Licenses. You may obtain copies of the
Licenses at:

   https://www.apache.org/licenses/LICENSE-2.0
   https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the Licenses is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the Licenses for the specific language governing permissions and
limitations under the Licenses.

--

Test code is dedicated to the Public Domain when so designated (see
the individual files for PD/CC0-dedicated sections).

--

The implementation for Utf8CharIndices was adapted from the
CharIndices implementation of the Rust standard library at revision
ab32548539ec38a939c1b58599249f3b54130026
(https://github.com/rust-lang/rust/blob/ab32548539ec38a939c1b58599249f3b54130026/library/core/src/str/iter.rs).

Excerpt from https://github.com/rust-lang/rust/blob/ab32548539ec38a939c1b58599249f3b54130026/COPYRIGHT ,
which refers to
https://github.com/rust-lang/rust/blob/ab32548539ec38a939c1b58599249f3b54130026/LICENSE-APACHE
and
https://github.com/rust-lang/rust/blob/ab32548539ec38a939c1b58599249f3b54130026/LICENSE-MIT
:

For full authorship information, see the version control history or
https://thanks.rust-lang.org

Except as otherwise noted (below and/or in individual files), Rust is
licensed under the Apache License, Version 2.0 <LICENSE-APACHE> or
<http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
<LICENSE-MIT> or <http://opensource.org/licenses/MIT>, at your option.
```

### `winnow 0.5.40`: upstream COPYRIGHT

```text
Short version for non-lawyers:

Winnow is licensed under MIT terms.

Longer version:

Copyrights in winnow are retained by their contributors. No
copyright assignment is required to contribute to winnow.

Except as otherwise noted (below and/or in individual files), winnow is
licensed under the MIT license
<LICENSE-MIT> or <http://opensource.org/licenses/MIT>, at your option.

Winnow is forked from nom and, similarly, Geal and other nom contributors
retain copyright on their contributions.
```

### `winnow 1.0.4`: upstream COPYRIGHT

```text
Short version for non-lawyers:

Winnow is licensed under MIT terms.

Longer version:

Copyrights in winnow are retained by their contributors. No
copyright assignment is required to contribute to winnow.

Except as otherwise noted (below and/or in individual files), winnow is
licensed under the MIT license
<LICENSE-MIT> or <http://opensource.org/licenses/MIT>, at your option.

Winnow is forked from nom and, similarly, Geal and other nom contributors
retain copyright on their contributions.
```

## License texts

### MIT

Copyright (c) <year> <copyright holders> (see the per-crate table above)

```text
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

### Apache-2.0

54 crate license files are Apache-2.0 (7 textual variants, mostly whitespace / appendix differences). One full text:

```text
Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS
```

### Unicode-3.0

Used by: `icu_collections 2.4.0 (LICENSE)`, `icu_locale_core 2.4.0 (LICENSE)`, `icu_normalizer 2.4.0 (LICENSE)`, `icu_normalizer_data 2.4.0 (LICENSE)`, `icu_properties 2.4.0 (LICENSE)`, `icu_properties_data 2.4.0 (LICENSE)`, `icu_provider 2.4.0 (LICENSE)`, `litemap 0.8.3 (LICENSE)`, `potential_utf 0.1.6 (LICENSE)`, `tinystr 0.8.5 (LICENSE)`, `writeable 0.6.4 (LICENSE)`, `yoke 0.8.4 (LICENSE)`, `yoke-derive 0.8.4 (LICENSE)`, `zerofrom 0.1.8 (LICENSE)`, `zerofrom-derive 0.1.8 (LICENSE)`, `zerotrie 0.2.5 (LICENSE)`, `zerovec 0.11.9 (LICENSE)`, `zerovec-derive 0.11.7 (LICENSE)`

```text
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 2020-2024 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

SPDX-License-Identifier: Unicode-3.0

—

Portions of ICU4X may have been adapted from ICU4C and/or ICU4J.
ICU 1.8.1 to ICU 57.1 © 1995-2016 International Business Machines Corporation and others.
```

Used by: `unicode-ident 1.0.27 (LICENSE-UNICODE)`

```text
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2023 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

### BSD-3-Clause

Used by: `alloc-no-stdlib 3.0.0 (LICENSE)`, `alloc-stdlib 0.3.0 (upstream LICENSE)`, `brotli 9.1.0 (LICENSE.BSD-3-Clause)`, `brotli-decompressor 6.2.0 (LICENSE)`, `brotli-decompressor 6.2.0 (upstream LICENSE)`

```text
Copyright (c) 2016 Dropbox, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Used by: `matchit 0.8.4 (LICENSE.httprouter)`

```text
BSD 3-Clause License

Copyright (c) 2013, Julien Schmidt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### ISC

Used by: `libloading 0.7.4 (LICENSE)`

```text
Copyright © 2015, Simonas Kazlauskas

Permission to use, copy, modify, and/or distribute this software for any purpose with or without
fee is hereby granted, provided that the above copyright notice and this permission notice appear
in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
```

### MPL-2.0

Used by: `cssparser 0.37.0`, `cssparser-macros 0.7.1`, `dtoa-short 0.3.5`, `option-ext 0.2.0`, `selectors 0.38.0`. The crates' own copies are this text (aside from whitespace and, in some copies, `http://` instead of `https://` in the Exhibit A URL). Standard text from https://github.com/spdx/license-list-data/blob/v3.29.0/text/MPL-2.0.txt:

```text
Mozilla Public License Version 2.0
==================================

1. Definitions
--------------

1.1. "Contributor"
    means each individual or legal entity that creates, contributes to
    the creation of, or owns Covered Software.

1.2. "Contributor Version"
    means the combination of the Contributions of others (if any) used
    by a Contributor and that particular Contributor's Contribution.

1.3. "Contribution"
    means Covered Software of a particular Contributor.

1.4. "Covered Software"
    means Source Code Form to which the initial Contributor has attached
    the notice in Exhibit A, the Executable Form of such Source Code
    Form, and Modifications of such Source Code Form, in each case
    including portions thereof.

1.5. "Incompatible With Secondary Licenses"
    means

    (a) that the initial Contributor has attached the notice described
        in Exhibit B to the Covered Software; or

    (b) that the Covered Software was made available under the terms of
        version 1.1 or earlier of the License, but not also under the
        terms of a Secondary License.

1.6. "Executable Form"
    means any form of the work other than Source Code Form.

1.7. "Larger Work"
    means a work that combines Covered Software with other material, in 
    a separate file or files, that is not Covered Software.

1.8. "License"
    means this document.

1.9. "Licensable"
    means having the right to grant, to the maximum extent possible,
    whether at the time of the initial grant or subsequently, any and
    all of the rights conveyed by this License.

1.10. "Modifications"
    means any of the following:

    (a) any file in Source Code Form that results from an addition to,
        deletion from, or modification of the contents of Covered
        Software; or

    (b) any new file in Source Code Form that contains any Covered
        Software.

1.11. "Patent Claims" of a Contributor
    means any patent claim(s), including without limitation, method,
    process, and apparatus claims, in any patent Licensable by such
    Contributor that would be infringed, but for the grant of the
    License, by the making, using, selling, offering for sale, having
    made, import, or transfer of either its Contributions or its
    Contributor Version.

1.12. "Secondary License"
    means either the GNU General Public License, Version 2.0, the GNU
    Lesser General Public License, Version 2.1, the GNU Affero General
    Public License, Version 3.0, or any later versions of those
    licenses.

1.13. "Source Code Form"
    means the form of the work preferred for making modifications.

1.14. "You" (or "Your")
    means an individual or a legal entity exercising rights under this
    License. For legal entities, "You" includes any entity that
    controls, is controlled by, or is under common control with You. For
    purposes of this definition, "control" means (a) the power, direct
    or indirect, to cause the direction or management of such entity,
    whether by contract or otherwise, or (b) ownership of more than
    fifty percent (50%) of the outstanding shares or beneficial
    ownership of such entity.

2. License Grants and Conditions
--------------------------------

2.1. Grants

Each Contributor hereby grants You a world-wide, royalty-free,
non-exclusive license:

(a) under intellectual property rights (other than patent or trademark)
    Licensable by such Contributor to use, reproduce, make available,
    modify, display, perform, distribute, and otherwise exploit its
    Contributions, either on an unmodified basis, with Modifications, or
    as part of a Larger Work; and

(b) under Patent Claims of such Contributor to make, use, sell, offer
    for sale, have made, import, and otherwise transfer either its
    Contributions or its Contributor Version.

2.2. Effective Date

The licenses granted in Section 2.1 with respect to any Contribution
become effective for each Contribution on the date the Contributor first
distributes such Contribution.

2.3. Limitations on Grant Scope

The licenses granted in this Section 2 are the only rights granted under
this License. No additional rights or licenses will be implied from the
distribution or licensing of Covered Software under this License.
Notwithstanding Section 2.1(b) above, no patent license is granted by a
Contributor:

(a) for any code that a Contributor has removed from Covered Software;
    or

(b) for infringements caused by: (i) Your and any other third party's
    modifications of Covered Software, or (ii) the combination of its
    Contributions with other software (except as part of its Contributor
    Version); or

(c) under Patent Claims infringed by Covered Software in the absence of
    its Contributions.

This License does not grant any rights in the trademarks, service marks,
or logos of any Contributor (except as may be necessary to comply with
the notice requirements in Section 3.4).

2.4. Subsequent Licenses

No Contributor makes additional grants as a result of Your choice to
distribute the Covered Software under a subsequent version of this
License (see Section 10.2) or under the terms of a Secondary License (if
permitted under the terms of Section 3.3).

2.5. Representation

Each Contributor represents that the Contributor believes its
Contributions are its original creation(s) or it has sufficient rights
to grant the rights to its Contributions conveyed by this License.

2.6. Fair Use

This License is not intended to limit any rights You have under
applicable copyright doctrines of fair use, fair dealing, or other
equivalents.

2.7. Conditions

Sections 3.1, 3.2, 3.3, and 3.4 are conditions of the licenses granted
in Section 2.1.

3. Responsibilities
-------------------

3.1. Distribution of Source Form

All distribution of Covered Software in Source Code Form, including any
Modifications that You create or to which You contribute, must be under
the terms of this License. You must inform recipients that the Source
Code Form of the Covered Software is governed by the terms of this
License, and how they can obtain a copy of this License. You may not
attempt to alter or restrict the recipients' rights in the Source Code
Form.

3.2. Distribution of Executable Form

If You distribute Covered Software in Executable Form then:

(a) such Covered Software must also be made available in Source Code
    Form, as described in Section 3.1, and You must inform recipients of
    the Executable Form how they can obtain a copy of such Source Code
    Form by reasonable means in a timely manner, at a charge no more
    than the cost of distribution to the recipient; and

(b) You may distribute such Executable Form under the terms of this
    License, or sublicense it under different terms, provided that the
    license for the Executable Form does not attempt to limit or alter
    the recipients' rights in the Source Code Form under this License.

3.3. Distribution of a Larger Work

You may create and distribute a Larger Work under terms of Your choice,
provided that You also comply with the requirements of this License for
the Covered Software. If the Larger Work is a combination of Covered
Software with a work governed by one or more Secondary Licenses, and the
Covered Software is not Incompatible With Secondary Licenses, this
License permits You to additionally distribute such Covered Software
under the terms of such Secondary License(s), so that the recipient of
the Larger Work may, at their option, further distribute the Covered
Software under the terms of either this License or such Secondary
License(s).

3.4. Notices

You may not remove or alter the substance of any license notices
(including copyright notices, patent notices, disclaimers of warranty,
or limitations of liability) contained within the Source Code Form of
the Covered Software, except that You may alter any license notices to
the extent required to remedy known factual inaccuracies.

3.5. Application of Additional Terms

You may choose to offer, and to charge a fee for, warranty, support,
indemnity or liability obligations to one or more recipients of Covered
Software. However, You may do so only on Your own behalf, and not on
behalf of any Contributor. You must make it absolutely clear that any
such warranty, support, indemnity, or liability obligation is offered by
You alone, and You hereby agree to indemnify every Contributor for any
liability incurred by such Contributor as a result of warranty, support,
indemnity or liability terms You offer. You may include additional
disclaimers of warranty and limitations of liability specific to any
jurisdiction.

4. Inability to Comply Due to Statute or Regulation
---------------------------------------------------

If it is impossible for You to comply with any of the terms of this
License with respect to some or all of the Covered Software due to
statute, judicial order, or regulation then You must: (a) comply with
the terms of this License to the maximum extent possible; and (b)
describe the limitations and the code they affect. Such description must
be placed in a text file included with all distributions of the Covered
Software under this License. Except to the extent prohibited by statute
or regulation, such description must be sufficiently detailed for a
recipient of ordinary skill to be able to understand it.

5. Termination
--------------

5.1. The rights granted under this License will terminate automatically
if You fail to comply with any of its terms. However, if You become
compliant, then the rights granted under this License from a particular
Contributor are reinstated (a) provisionally, unless and until such
Contributor explicitly and finally terminates Your grants, and (b) on an
ongoing basis, if such Contributor fails to notify You of the
non-compliance by some reasonable means prior to 60 days after You have
come back into compliance. Moreover, Your grants from a particular
Contributor are reinstated on an ongoing basis if such Contributor
notifies You of the non-compliance by some reasonable means, this is the
first time You have received notice of non-compliance with this License
from such Contributor, and You become compliant prior to 30 days after
Your receipt of the notice.

5.2. If You initiate litigation against any entity by asserting a patent
infringement claim (excluding declaratory judgment actions,
counter-claims, and cross-claims) alleging that a Contributor Version
directly or indirectly infringes any patent, then the rights granted to
You by any and all Contributors for the Covered Software under Section
2.1 of this License shall terminate.

5.3. In the event of termination under Sections 5.1 or 5.2 above, all
end user license agreements (excluding distributors and resellers) which
have been validly granted by You or Your distributors under this License
prior to termination shall survive termination.

************************************************************************
*                                                                      *
*  6. Disclaimer of Warranty                                           *
*  -------------------------                                           *
*                                                                      *
*  Covered Software is provided under this License on an "as is"       *
*  basis, without warranty of any kind, either expressed, implied, or  *
*  statutory, including, without limitation, warranties that the       *
*  Covered Software is free of defects, merchantable, fit for a        *
*  particular purpose or non-infringing. The entire risk as to the     *
*  quality and performance of the Covered Software is with You.        *
*  Should any Covered Software prove defective in any respect, You     *
*  (not any Contributor) assume the cost of any necessary servicing,   *
*  repair, or correction. This disclaimer of warranty constitutes an   *
*  essential part of this License. No use of any Covered Software is   *
*  authorized under this License except under this disclaimer.         *
*                                                                      *
************************************************************************

************************************************************************
*                                                                      *
*  7. Limitation of Liability                                          *
*  --------------------------                                          *
*                                                                      *
*  Under no circumstances and under no legal theory, whether tort      *
*  (including negligence), contract, or otherwise, shall any           *
*  Contributor, or anyone who distributes Covered Software as          *
*  permitted above, be liable to You for any direct, indirect,         *
*  special, incidental, or consequential damages of any character      *
*  including, without limitation, damages for lost profits, loss of    *
*  goodwill, work stoppage, computer failure or malfunction, or any    *
*  and all other commercial damages or losses, even if such party      *
*  shall have been informed of the possibility of such damages. This   *
*  limitation of liability shall not apply to liability for death or   *
*  personal injury resulting from such party's negligence to the       *
*  extent applicable law prohibits such limitation. Some               *
*  jurisdictions do not allow the exclusion or limitation of           *
*  incidental or consequential damages, so this exclusion and          *
*  limitation may not apply to You.                                    *
*                                                                      *
************************************************************************

8. Litigation
-------------

Any litigation relating to this License may be brought only in the
courts of a jurisdiction where the defendant maintains its principal
place of business and such litigation shall be governed by laws of that
jurisdiction, without reference to its conflict-of-law provisions.
Nothing in this Section shall prevent a party's ability to bring
cross-claims or counter-claims.

9. Miscellaneous
----------------

This License represents the complete agreement concerning the subject
matter hereof. If any provision of this License is held to be
unenforceable, such provision shall be reformed only to the extent
necessary to make it enforceable. Any law or regulation which provides
that the language of a contract shall be construed against the drafter
shall not be used to construe this License against a Contributor.

10. Versions of the License
---------------------------

10.1. New Versions

Mozilla Foundation is the license steward. Except as provided in Section
10.3, no one other than the license steward has the right to modify or
publish new versions of this License. Each version will be given a
distinguishing version number.

10.2. Effect of New Versions

You may distribute the Covered Software under the terms of the version
of the License under which You originally received the Covered Software,
or under the terms of any subsequent version published by the license
steward.

10.3. Modified Versions

If you create software not governed by this License, and you want to
create a new license for such software, you may create and use a
modified version of this License if you rename the license and remove
any references to the name of the license steward (except to note that
such modified license differs from this License).

10.4. Distributing Source Code Form that is Incompatible With Secondary
Licenses

If You choose to distribute Source Code Form that is Incompatible With
Secondary Licenses under the terms of this version of the License, the
notice described in Exhibit B of this License must be attached.

Exhibit A - Source Code Form License Notice
-------------------------------------------

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at https://mozilla.org/MPL/2.0/.

If it is not possible or desirable to put the notice in a particular
file, then You may include the notice in a location (such as a LICENSE
file in a relevant directory) where a recipient would be likely to look
for such a notice.

You may add additional accurate notices of copyright ownership.

Exhibit B - "Incompatible With Secondary Licenses" Notice
---------------------------------------------------------

  This Source Code Form is "Incompatible With Secondary Licenses", as
  defined by the Mozilla Public License, v. 2.0.
```

### Zlib

Used by: `foldhash 0.2.0 (LICENSE)`

```text
Copyright (c) 2024 Orson Peters

This software is provided 'as-is', without any express or implied warranty. In
no event will the authors be held liable for any damages arising from the use of
this software.

Permission is granted to anyone to use this software for any purpose, including
commercial applications, and to alter it and redistribute it freely, subject to
the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim
    that you wrote the original software. If you use this software in a product,
    an acknowledgment in the product documentation would be appreciated but is
    not required.

2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
```

### Other license files (pointers, SPDX metadata, combined COPYRIGHT files)

Used by: `aho-corasick 1.1.5 (COPYING)`, `byteorder 1.5.0 (COPYING)`, `memchr 2.8.3 (COPYING)`, `same-file 1.0.6 (COPYING)`, `walkdir 2.5.0 (COPYING)`, `winapi-util 0.1.11 (COPYING)`

```text
This project is dual-licensed under the Unlicense and MIT licenses.

You may use this code under the terms of either license.
```

Used by: `block2 0.6.2 (upstream LICENSE.md)`, `dispatch2 0.3.1 (upstream LICENSE.md)`, `objc2 0.6.5 (upstream LICENSE.md)`, `objc2-app-kit 0.3.2 (upstream LICENSE.md)`, `objc2-core-foundation 0.3.2 (upstream LICENSE.md)`, `objc2-encode 4.1.0 (upstream LICENSE.md)`, `objc2-exception-helper 0.1.1 (upstream LICENSE.md)`, `objc2-foundation 0.3.2 (upstream LICENSE.md)`, `objc2-quartz-core 0.3.2 (upstream LICENSE.md)`, `objc2-web-kit 0.3.2 (upstream LICENSE.md)`

```text
# License

The licensing of these crates is a bit complicated:
- The crates `objc2`, `block2`, `objc2-foundation` and `objc2-encode` are
  [currently][#23] licensed under [the MIT license][MIT].
- All other crates are trio-licensed under the [Zlib], [Apache-2.0] or [MIT]
  license, at your option.

Furthermore, the crates are (usually automatically) derived from Apple SDKs,
and that may have implications for licensing, see below for details.

[#23]: https://github.com/madsmtm/objc2/issues/23
[MIT]: https://opensource.org/license/MIT
[Zlib]: https://zlib.net/zlib_license.html
[Apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0


## Apple SDKs

These crates are derived from Apple SDKs shipped with Xcode. You can obtain a
copy of the Xcode license at:

https://www.apple.com/legal/sla/docs/xcode.pdf

Or by typing `xcodebuild -license` in your terminal.

From reading the license, it is unclear whether distributing derived works
such as these crates are allowed?

But in any case, to practically use these crates, you will have to link, and
that only works when you have the correct Xcode SDK available to provide the
required `.tbd` files, which is why we choose to still use the normal SPDX
identifiers in the crates (Xcode is required to use the crates, and when using
Xcode you have already agreed to the Xcode license).
```

Used by: `muda 0.20.0 (LICENSE.spdx)`

```text
SPDXVersion: SPDX-2.1
DataLicense: CC0-1.0
PackageName: muda
DataFormat: SPDXRef-1
PackageSupplier: Organization: The Tauri Programme in the Commons Conservancy
PackageHomePage: https://tauri.app
PackageLicenseDeclared: Apache-2.0
PackageLicenseDeclared: MIT
PackageCopyrightText: 2020-2022, The Tauri Programme in the Commons Conservancy
PackageSummary: <text>Menu Utilities for Desktop Applications.
                </text>
PackageComment: <text>The package includes the following libraries; see
Relationship information.
                </text>
Created: 2022-12-05T09:00:00Z
PackageDownloadLocation: git://github.com/tauri-apps/muda
PackageDownloadLocation: git+https://github.com/tauri-apps/muda.git
PackageDownloadLocation: git+ssh://github.com/tauri-apps/muda.git
Creator: Person: Daniel Thompson-Yvetot
```

Used by: `tao 0.37.2 (LICENSE.spdx)`

```text
SPDXVersion: SPDX-2.1
DataLicense: CC0-1.0
PackageName: tao
DataFormat: SPDXRef-1
PackageSupplier: Organization: The Tauri Programme in the Commons Conservancy
PackageHomePage: https://tauri.app
PackageLicenseDeclared: Apache-2.0
PackageCopyrightText: 2021-2023, The Tauri Programme in the Commons Conservancy
PackageSummary: <text>Tao is the official, rust-based window manager for Wry.
                </text>
PackageComment: <text>The package includes the following libraries; see
Relationship information.
                </text>
Created: 2020-05-20T09:00:00Z
PackageDownloadLocation: git://github.com/tauri-apps/tao
PackageDownloadLocation: git+https://github.com/tauri-apps/tao.git
PackageDownloadLocation: git+ssh://github.com/tauri-apps/tao.git
Creator: Person: Daniel Thompson-Yvetot
```

Used by: `tauri-plugin 2.7.2 (upstream LICENSE.spdx)`

```text
SPDXVersion: SPDX-2.1
DataLicense: CC0-1.0
PackageName: tauri
DataFormat: SPDXRef-1
PackageSupplier: Organization: The Tauri Programme in the Commons Conservancy
PackageHomePage: https://tauri.app
PackageLicenseDeclared: Apache-2.0
PackageLicenseDeclared: MIT
PackageCopyrightText: 2019-2025, The Tauri Programme in the Commons Conservancy
PackageSummary: <text>Tauri is a rust project that enables developers to make secure
and small desktop applications using a web frontend.
                </text>
PackageComment: <text>The package includes the following libraries; see
Relationship information.
                </text>
Created: 2019-05-20T09:00:00Z
PackageDownloadLocation: git://github.com/tauri-apps/tauri
PackageDownloadLocation: git+https://github.com/tauri-apps/tauri.git
PackageDownloadLocation: git+ssh://github.com/tauri-apps/tauri.git
Creator: Person: Daniel Thompson-Yvetot
```

Used by: `tauri-plugin-dialog 2.8.4 (LICENSE.spdx)`, `tauri-plugin-fs 2.7.1 (LICENSE.spdx)`

```text
SPDXVersion: SPDX-2.1
DataLicense: CC0-1.0
PackageName: tauri
DataFormat: SPDXRef-1
PackageSupplier: Organization: The Tauri Programme in the Commons Conservancy
PackageHomePage: https://tauri.app
PackageLicenseDeclared: Apache-2.0
PackageLicenseDeclared: MIT
PackageCopyrightText: 2019-2022, The Tauri Programme in the Commons Conservancy
PackageSummary: <text>Tauri is a rust project that enables developers to make secure
and small desktop applications using a web frontend.
                </text>
PackageComment: <text>The package includes the following libraries; see
Relationship information.
                </text>
Created: 2019-05-20T09:00:00Z
PackageDownloadLocation: git://github.com/tauri-apps/tauri
PackageDownloadLocation: git+https://github.com/tauri-apps/tauri.git
PackageDownloadLocation: git+ssh://github.com/tauri-apps/tauri.git
Creator: Person: Daniel Thompson-Yvetot
```

Used by: `typenum 1.20.1 (LICENSE)`

```text
MIT OR Apache-2.0
```

Used by: `window-vibrancy 0.8.1 (LICENSE.spdx)`

```text
SPDXVersion: SPDX-2.1
DataLicense: CC0-1.0
PackageName: window-vibrancy
DataFormat: SPDXRef-1
PackageSupplier: Organization: The Tauri Programme in the Commons Conservancy
PackageHomePage: https://tauri.app
PackageLicenseDeclared: Apache-2.0
PackageLicenseDeclared: MIT
PackageCopyrightText: 2020-2022, The Tauri Programme in the Commons Conservancy
PackageSummary: <text>Make your windows vibrant.
                </text>
PackageComment: <text>The package includes the following libraries; see
Relationship information.
                </text>
Created: 2020-05-20T09:00:00Z
PackageDownloadLocation: git://github.com/tauri-apps/window-vibrancy
PackageDownloadLocation: git+https://github.com/tauri-apps/window-vibrancy.git
PackageDownloadLocation: git+ssh://github.com/tauri-apps/window-vibrancy.git
Creator: Person: Daniel Thompson-Yvetot
```

Used by: `wry 0.57.1 (LICENSE.spdx)`

```text
SPDXVersion: SPDX-2.1
DataLicense: CC0-1.0
PackageName: wry
DataFormat: SPDXRef-1
PackageSupplier: Organization: The Tauri Programme in the Commons Conservancy
PackageHomePage: https://tauri.app
PackageLicenseDeclared: Apache-2.0
PackageLicenseDeclared: MIT
PackageCopyrightText: 2020-2023, The Tauri Programme in the Commons Conservancy
PackageSummary: <text>Wry is the official, rust-based webview
windowing service for Tauri.
                </text>
PackageComment: <text>The package includes the following libraries; see
Relationship information.
                </text>
Created: 2020-05-20T09:00:00Z
PackageDownloadLocation: git://github.com/tauri-apps/wry
PackageDownloadLocation: git+https://github.com/tauri-apps/wry.git
PackageDownloadLocation: git+ssh://github.com/tauri-apps/wry.git
Creator: Person: Daniel Thompson-Yvetot
```

## Coverage check

- Shipped crates (from `crates.txt`): 361; listed above: 361; missing: 0.
- `Cargo.lock` crates.io packages: 455; shipped: 361; not shipped: 94.
