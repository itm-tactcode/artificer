#!/usr/bin/env bash
# Local Linux install for Artificer, under $PREFIX (default ~/.local). Publishes nothing.
# From a source checkout (scripts/install.sh): builds the CLI and the desktop app, then installs them.
#   Usage: ./scripts/install.sh [--cuda] [--no-desktop] [--prefix DIR]
# From the extracted desktop download (./install.sh next to bin/): installs the prebuilt desktop app, no build.
#   Usage: ./install.sh [--prefix DIR]
# Both install the hicolor icons and the launcher entry for the desktop app; scripts/uninstall.sh removes it all.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"; CUDA=0; DESKTOP=1
while [ $# -gt 0 ]; do case "$1" in
  --cuda) CUDA=1;; --no-desktop) DESKTOP=0;;
  --prefix) PREFIX="$2"; shift;; -h|--help) sed -n 2,7p "$0"; exit 0;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; shift; done
if [ -f "$HERE/bin/artificer-desktop" ]; then
  [ "$CUDA" = 0 ] || { echo "error: --cuda needs a source checkout; this is the prebuilt desktop download" >&2; exit 2; }
  [ "$DESKTOP" = 1 ] || { echo "nothing to do: this download only contains the desktop app" >&2; exit 0; }
fi
BIN="$PREFIX/bin"; mkdir -p "$BIN"
MANIFEST="$PREFIX/share/artificer/installed-files"; mkdir -p "$(dirname "$MANIFEST")"
touch "$MANIFEST"   # kept across installs (e.g. CLI from source, then the desktop download)
if [ -f "$HERE/bin/artificer-desktop" ]; then
  # Prebuilt desktop download: bin/, share/applications/ and share/icons/hicolor/ next to this script.
  DESKTOP_BIN="$HERE/bin/artificer-desktop"
  ICONS="$HERE/share/icons/hicolor"; ICON_FILE() { echo "$ICONS/${1}x${1}/apps/artificer.png"; }
  DESKTOP_FILE="$HERE/share/applications/io.tactcode.artificer.desktop"
else
  ROOT="$(cd "$HERE/.." && pwd)"
  command -v cargo >/dev/null || { echo "error: cargo not found; install Rust from https://rustup.rs" >&2; exit 1; }
  FEAT=(); [ "$CUDA" = 1 ] && FEAT=(--features cuda)
  echo "==> building artificer CLI (release${CUDA:+, cuda=$CUDA})"
  ( cd "$ROOT" && cargo build --release -p artificer-cli "${FEAT[@]}" )
  install -m755 "$ROOT/target/release/artificer" "$BIN/artificer"
  echo "$BIN/artificer" >> "$MANIFEST"
  if [ "$DESKTOP" = 1 ]; then
    echo "==> building desktop app (release)"
    ( cd "$ROOT" && cargo build --release -p artificer-desktop )
  fi
  DESKTOP_BIN="$ROOT/target/release/artificer-desktop"
  ICON_FILE() { echo "$ROOT/desktop/icons/src/artificer-$1.png"; }
  DESKTOP_FILE="$ROOT/desktop/io.tactcode.artificer.desktop"
fi
if [ "$DESKTOP" = 1 ]; then
  install -m755 "$DESKTOP_BIN" "$BIN/artificer-desktop"
  echo "$BIN/artificer-desktop" >> "$MANIFEST"
  # hicolor icons: the locked app icons (16/32/48 pixel icons, 64-512 silhouette), one per size.
  for n in 16 32 48 64 128 256 512; do
    ICON="$PREFIX/share/icons/hicolor/${n}x${n}/apps/artificer.png"; mkdir -p "$(dirname "$ICON")"
    install -m644 "$(ICON_FILE "$n")" "$ICON"
    echo "$ICON" >> "$MANIFEST"
  done
  APP="$PREFIX/share/applications/io.tactcode.artificer.desktop"; mkdir -p "$(dirname "$APP")"
  sed "s|^Exec=.*|Exec=$BIN/artificer-desktop|" "$DESKTOP_FILE" > "$APP"
  chmod 644 "$APP"
  echo "$APP" >> "$MANIFEST"
fi
awk '!seen[$0]++' "$MANIFEST" > "$MANIFEST.new" && mv "$MANIFEST.new" "$MANIFEST"
echo "==> installed (all files recorded for uninstall.sh):"; cat "$MANIFEST"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "note: add $BIN to PATH";; esac
if [ -x "$BIN/artificer" ] && grep -qx "$BIN/artificer" "$MANIFEST"; then "$BIN/artificer" --version; fi
