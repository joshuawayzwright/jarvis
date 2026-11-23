"""The web pages and icons use the owner's brand palette (BRAND-RULES.md in joshuawayzwright/joshuawayzwright)."""

from __future__ import annotations

import re

from jarvis import icons, web

PALETTE = {"#0A0A0A", "#8A8F98", "#D4AF37", "#0087FF"}  # charcoal, weathered steel, burnished gold, electric blue


def test_both_pages_share_the_brand_palette_and_wordmark() -> None:
    for page in (web.PAGE, web.DASHBOARD_PAGE):
        assert web.BRAND_CSS in page and "/*BRAND*/" not in page
        assert 'class="wordmark">JARVIS' in page  # text-only wordmark: no invented logo
        assert '<meta name="theme-color" content="#0A0A0A">' in page
        assert "prefers-color-scheme" not in page  # one dark brand theme, not a second off-brand palette
        old = {"#0b6bcb", "#5aa9ff", "#3d93ee", "#f6f7f9", "#0f1216"}  # the earlier, unbranded colours
        assert not old & {c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}\b", page)}
    for colour in PALETTE:
        assert colour in web.BRAND_CSS
    assert web.manifest("t")["theme_color"] == "#0A0A0A"


def test_icon_is_a_gold_letter_on_charcoal() -> None:
    assert 'fill="#0A0A0A"' in icons.SVG and 'fill="#D4AF37"' in icons.SVG
    data = icons.png(64)
    assert data.startswith(b"\x89PNG")
    assert icons._pixel(5, 32, 64, rounded=False) == icons.CHARCOAL  # background
    assert icons._pixel(39, 30, 64, rounded=False) == icons.GOLD  # the J's stem
