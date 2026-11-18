#!/usr/bin/env bash
# Assemble the unsigned Artificer.app for macOS from a release build of the desktop app, ad-hoc sign it
# (`codesign -s -`, no Developer ID), and zip it. Used by the macOS CI job; macOS only.
# Usage: scripts/macos-app.sh <path/to/artificer-desktop> <out-dir>
# Writes <out-dir>/Artificer.app and <out-dir>/artificer-<version>-macos-aarch64-desktop.zip. The zip holds
# Artificer.app (with THIRD-PARTY-NOTICES.md in Contents/Resources, added before signing) and, beside it,
# THIRD-PARTY-NOTICES.md, LICENSE-APACHE and LICENSE-MIT, byte-identical to the repository files.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,5p "$0" >&2; exit 2; }
BIN="$1"; OUT="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/desktop/Cargo.toml" | head -n 1)"
[ -n "$VERSION" ] || { echo "error: no version in desktop/Cargo.toml" >&2; exit 1; }
[ -x "$BIN" ] || { echo "error: $BIN is not an executable" >&2; exit 1; }
APP="$OUT/Artificer.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
install -m 0755 "$BIN" "$APP/Contents/MacOS/artificer-desktop"
install -m 0644 "$ROOT/desktop/icons/icon.icns" "$APP/Contents/Resources/artificer.icns"
install -m 0644 "$ROOT/THIRD-PARTY-NOTICES.md" "$APP/Contents/Resources/THIRD-PARTY-NOTICES.md"
sed "s/@VERSION@/$VERSION/g" "$ROOT/desktop/macos/Info.plist" > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist"
codesign --force -s - "$APP"
codesign --verify --strict --verbose=2 "$APP"
codesign -dv --verbose=4 "$APP" 2>&1
cmp "$ROOT/THIRD-PARTY-NOTICES.md" "$APP/Contents/Resources/THIRD-PARTY-NOTICES.md"
STAGE="$OUT/zip-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/Artificer.app"
install -m 0644 "$ROOT/THIRD-PARTY-NOTICES.md" "$ROOT/LICENSE-APACHE" "$ROOT/LICENSE-MIT" "$STAGE/"
codesign --verify --strict --verbose=2 "$STAGE/Artificer.app"
ZIP="$OUT/artificer-$VERSION-macos-aarch64-desktop.zip"
rm -f "$ZIP"
ditto -c -k "$STAGE" "$ZIP"
rm -rf "$STAGE"
echo "wrote $ZIP"
