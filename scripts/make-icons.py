#!/usr/bin/env python3
"""Build desktop/icons/icon.ico and icon.icns from the locked Artificer app icons.

Sources are desktop/icons/src/artificer-<n>.png: byte copies of the locked set in logo-final/app-icons/
(16 / 32 / 48 are the locked pixel icons from logo-final/; 64 and up are renders of the 06 silhouette).
Each PNG is stored in the container byte for byte: nothing is decoded, resized or re-encoded, so the
output is deterministic and the small sizes are never derived from a larger one.

  scripts/make-icons.py           write icon.ico and icon.icns
  scripts/make-icons.py --check   verify both files match the sources exactly (exit 1 if not)

Standard library only.
"""
import struct
import sys
from pathlib import Path

ICONS = Path(__file__).resolve().parent.parent / "desktop" / "icons"
SRC = ICONS / "src"
PNG_SIG = b"\x89PNG\r\n\x1a\n"

# Windows ICO frames (PNG-compressed entries).
ICO_SIZES = [16, 32, 48, 256]
# macOS ICNS chunks: (OSType, source size). Same mapping as the iconset in app-icons/README.md:
# icon_16x16, icon_16x16@2x, icon_32x32, icon_32x32@2x, icon_128x128, icon_128x128@2x,
# icon_256x256, icon_256x256@2x, icon_512x512, icon_512x512@2x.
ICNS_CHUNKS = [
    (b"icp4", 16), (b"ic11", 32), (b"icp5", 32), (b"ic12", 64), (b"ic07", 128),
    (b"ic13", 256), (b"ic08", 256), (b"ic14", 512), (b"ic09", 512), (b"ic10", 1024),
]


def png(n):
    path = SRC / f"artificer-{n}.png"
    data = path.read_bytes()
    if data[:8] != PNG_SIG or data[12:16] != b"IHDR":
        sys.exit(f"{path}: not a PNG")
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    if (w, h, depth, ctype) != (n, n, 8, 6):
        sys.exit(f"{path}: expected {n}x{n} 8-bit RGBA, got {w}x{h} depth {depth} type {ctype}")
    return data


def build_ico():
    frames = [png(n) for n in ICO_SIZES]
    out = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    for n, data in zip(ICO_SIZES, frames):
        # width/height 0 means 256; 1 plane, 32 bpp, byte length, offset of the PNG.
        out += struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return out + b"".join(frames)


def build_icns():
    body = b"".join(t + struct.pack(">I", 8 + len(png(n))) + png(n) for t, n in ICNS_CHUNKS)
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def check_ico(data):
    ok = True
    _, kind, count = struct.unpack("<HHH", data[:6])
    if kind != 1 or count != len(ICO_SIZES):
        return False
    for i, n in enumerate(ICO_SIZES):
        w, h, _, _, _, _, size, off = struct.unpack("<BBBBHHII", data[6 + 16 * i:22 + 16 * i])
        same = (w or 256, h or 256) == (n, n) and data[off:off + size] == png(n)
        print(f"icon.ico  {n:4d}x{n:<4d} <- artificer-{n}.png  identical={same}")
        ok &= same
    return ok


def check_icns(data):
    if data[:4] != b"icns" or struct.unpack(">I", data[4:8])[0] != len(data):
        return False
    ok, pos, seen = True, 8, []
    while pos < len(data):
        kind, length = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        n = dict(ICNS_CHUNKS).get(kind)
        same = n is not None and data[pos + 8:pos + length] == png(n)
        print(f"icon.icns {kind.decode('latin-1')} <- artificer-{n}.png  identical={same}")
        ok &= same
        seen.append(kind)
        pos += length
    return ok and seen == [t for t, _ in ICNS_CHUNKS]


def main():
    ico, icns = build_ico(), build_icns()
    if sys.argv[1:] == ["--check"]:
        ok = True
        for name, want, check in (("icon.ico", ico, check_ico), ("icon.icns", icns, check_icns)):
            path = ICONS / name
            data = path.read_bytes() if path.exists() else b""
            ok &= check(data) and data == want
        print("ALL_IDENTICAL" if ok else "MISMATCH")
        return 0 if ok else 1
    if sys.argv[1:]:
        sys.exit(__doc__)
    (ICONS / "icon.ico").write_bytes(ico)
    (ICONS / "icon.icns").write_bytes(icns)
    print(f"wrote {ICONS / 'icon.ico'} ({len(ico)} bytes), {ICONS / 'icon.icns'} ({len(icns)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
