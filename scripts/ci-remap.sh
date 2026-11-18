#!/usr/bin/env bash
# CI only: append --remap-path-prefix flags for the release builds to RUSTFLAGS via $GITHUB_ENV, so
# runner paths (cargo registry, rustup home, checkout) do not end up in the binaries:
#   $CARGO_HOME/registry/src  -> /cargo/registry/src
#   $RUSTUP_HOME              -> /rustup
#   workspace                 -> /artificer
# REMAP_WORKSPACE is the workflow's `${{ github.workspace }}` (passed as an env var by the step); when
# empty, $GITHUB_WORKSPACE is used. If $GITHUB_WORKSPACE differs from it, that path is remapped too.
# The Windows jobs use scripts/ci-remap.ps1 (backslash and forward-slash forms).
set -euo pipefail
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
rustup_home="${RUSTUP_HOME:-$HOME/.rustup}"
workspace="${REMAP_WORKSPACE:-}"
[ -n "$workspace" ] || workspace="${GITHUB_WORKSPACE}"
trim() { printf '%s' "${1%/}"; }
cargo_reg="$(trim "$cargo_home")/registry/src"
rustup_home="$(trim "$rustup_home")"
workspace="$(trim "$workspace")"
ws_env="$(trim "${GITHUB_WORKSPACE:-}")"
flags="${RUSTFLAGS:-}"
append() {
  local from="$1" to="$2"
  [ -n "$from" ] || return 0
  flags="$flags --remap-path-prefix=${from}=${to}"
}
append "$cargo_reg" "/cargo/registry/src"
append "$rustup_home" "/rustup"
append "$workspace" "/artificer"
if [ -n "$ws_env" ] && [ "$ws_env" != "$workspace" ]; then
  append "$ws_env" "/artificer"
fi
echo "RUSTFLAGS=${flags}" >> "$GITHUB_ENV"
echo "Using RUSTFLAGS=${flags}"
