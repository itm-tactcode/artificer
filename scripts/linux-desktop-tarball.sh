#!/usr/bin/env bash
# Pack a release build of the Linux desktop app as a plain .tar.gz (no AppImage, no .deb). Used by the Linux
# desktop CI job. Layout, inside artificer-<version>-linux-x86_64-desktop/:
#   install.sh, uninstall.sh          install.sh installs the files below under --prefix (default ~/.local)
#   bin/artificer-desktop
#   share/applications/io.tactcode.artificer.desktop
#   share/icons/hicolor/<n>x<n>/apps/artificer.png   n = 16 32 48 64 128 256 512
#   README.md, LICENSE-APACHE, LICENSE-MIT, THIRD-PARTY-NOTICES.md
# Usage: scripts/linux-desktop-tarball.sh <path/to/artificer-desktop> <out-dir>
# Writes <out-dir>/artificer-<version>-linux-x86_64-desktop.tar.gz. Entries are sorted, owned by 0:0 and dated
# SOURCE_DATE_EPOCH (default: the HEAD commit time), and gzip stores no name or time.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,10p "$0" >&2; exit 2; }
BIN="$1"; OUT="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/desktop/Cargo.toml" | head -n 1)"
[ -n "$VERSION" ] || { echo "error: no version in desktop/Cargo.toml" >&2; exit 1; }
[ -x "$BIN" ] || { echo "error: $BIN is not an executable" >&2; exit 1; }
EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct)}"
NAME="artificer-$VERSION-linux-x86_64-desktop"
STAGE="$OUT/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE/bin" "$STAGE/share/applications"
install -m 0755 "$BIN" "$STAGE/bin/artificer-desktop"
install -m 0755 "$ROOT/scripts/install.sh" "$ROOT/scripts/uninstall.sh" "$STAGE/"
install -m 0644 "$ROOT/desktop/io.tactcode.artificer.desktop" "$STAGE/share/applications/"
for n in 16 32 48 64 128 256 512; do
  mkdir -p "$STAGE/share/icons/hicolor/${n}x${n}/apps"
  install -m 0644 "$ROOT/desktop/icons/src/artificer-$n.png" "$STAGE/share/icons/hicolor/${n}x${n}/apps/artificer.png"
done
install -m 0644 "$ROOT/README.md" "$ROOT/LICENSE-APACHE" "$ROOT/LICENSE-MIT" "$ROOT/THIRD-PARTY-NOTICES.md" "$STAGE/"
TGZ="$OUT/$NAME.tar.gz"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$EPOCH" --format=gnu \
  -C "$OUT" -cf - "$NAME" | gzip -9 -n > "$TGZ"
tar -tzvf "$TGZ"
echo "wrote $TGZ"
