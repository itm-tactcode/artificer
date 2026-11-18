#!/usr/bin/env bash
# Removes exactly the files scripts/install.sh recorded. Models and the source checkout are untouched.
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local}"; [ "${1:-}" = --prefix ] && PREFIX="$2"
MANIFEST="$PREFIX/share/artificer/installed-files"
[ -f "$MANIFEST" ] || { echo "nothing installed under $PREFIX"; exit 0; }
while read -r f; do [ -n "$f" ] && rm -fv "$f"; done < "$MANIFEST"
rm -fv "$MANIFEST"; rmdir "$(dirname "$MANIFEST")" 2>/dev/null || true
