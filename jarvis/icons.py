"""App icons for adding the web interface to a phone's home screen, drawn in code so no image files ship.

Text only, as the owner chose: a gold "J" on a charcoal tile, in the shared brand palette (BRAND-RULES.md in
joshuawayzwright/joshuawayzwright). It is a letter, not a logo: Jarvis has no approved logo, and the brand rules
forbid inventing one. The letter is built from simple shapes (a bar, a stem and a curved hook) in a 0-100 grid,
the same shapes in the SVG and the PNGs. PNG is needed because iOS ignores SVG home-screen icons; the encoder is
the minimal one PNG allows: 8-bit RGBA, one zlib-compressed IDAT chunk.
"""

from __future__ import annotations

import struct
import zlib
from functools import lru_cache

CHARCOAL = (0x0A, 0x0A, 0x0A, 255)
GOLD = (0xD4, 0xAF, 0x37, 255)
CLEAR = (0, 0, 0, 0)

# The "J" in a 0-100 grid: top bar, stem, and a hook (half ring) ending in a short upturn on the left.
# The SVG draws the same outline as one path, so no seams show where the parts meet.
BAR = (38, 22, 72, 32)       # x0, y0, x1, y1
STEM = (56, 22, 67, 62)
HOOK_CENTRE, HOOK_OUTER, HOOK_INNER = (45.5, 62), 21.5, 10.5
TIP = (24, 54, 35, 62)

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<rect width="100" height="100" rx="22" fill="#0A0A0A"/>
<path fill="#D4AF37" d="M38 22 H72 V32 H67 V62 A21.5 21.5 0 0 1 24 62 V54 H35 V62 A10.5 10.5 0 0 0 56 62 V32 H38 Z"/>
</svg>"""


def _inside(box: tuple[float, float, float, float], u: float, v: float) -> bool:
    return box[0] <= u <= box[2] and box[1] <= v <= box[3]


def _pixel(x: float, y: float, size: int, rounded: bool) -> tuple[int, int, int, int]:
    u, v = x / size * 100, y / size * 100  # same 0-100 coordinates as the SVG
    if rounded:
        r = 22
        cx, cy = min(max(u, r), 100 - r), min(max(v, r), 100 - r)
        if (u - cx) ** 2 + (v - cy) ** 2 > r * r:
            return CLEAR
    if _inside(BAR, u, v) or _inside(STEM, u, v) or _inside(TIP, u, v):
        return GOLD
    hx, hy = HOOK_CENTRE
    d = ((u - hx) ** 2 + (v - hy) ** 2) ** 0.5
    if v >= hy and HOOK_INNER <= d <= HOOK_OUTER:
        return GOLD
    return CHARCOAL


@lru_cache(maxsize=4)
def png(size: int, rounded: bool = True) -> bytes:
    """The icon as a PNG of size x size pixels. iOS rounds corners itself, so it gets a square one."""
    rows = bytearray()
    for y in range(size):
        rows.append(0)  # filter type: none
        for x in range(size):
            # Four samples per pixel, averaged, so the curve and corners are smooth rather than jagged.
            samples = [_pixel(x + dx, y + dy, size, rounded) for dx in (0.25, 0.75) for dy in (0.25, 0.75)]
            alpha = sum(c[3] for c in samples)
            if alpha == 0:
                rows.extend(CLEAR)
                continue
            rows.extend(round(sum(c[i] * c[3] for c in samples) / alpha) for i in range(3))
            rows.append(round(alpha / len(samples)))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))
