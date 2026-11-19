"""The chat page's light Markdown renderer, run in Node against a tiny fake DOM.

The renderer must format bold, italics, inline code, https links, bullets and headings, keep code
blocks as <pre>, leave half-streamed markers alone, and never turn model text into markup.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from jarvis.web import PAGE

FAKE_DOM = r"""
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
class Text { constructor(t) { this.t = t; } html() { return esc(this.t); } }
class El {
  constructor(tag) { this.tag = tag; this.kids = []; this.attrs = {}; }
  appendChild(n) { this.kids.push(n); return n; }
  set textContent(t) { this.kids = t ? [new Text(t)] : []; }
  set href(v) { this.attrs.href = v; } set target(v) { this.attrs.target = v; } set rel(v) { this.attrs.rel = v; }
  html() {
    const a = Object.entries(this.attrs).map(([k, v]) => ` ${k}="${esc(v)}"`).join("");
    return `<${this.tag}${a}>${this.kids.map(k => k.html()).join("")}</${this.tag}>`;
  }
}
const document = { createElement: t => new El(t), createTextNode: t => new Text(t) };
"""

CASES = [
    ('- **"Status of `owner/repo`?"** go', '• <strong>"Status of <code>owner/repo</code>?"</strong> go'),
    ("*italic* and _this_, not snake_case_name or 2*3*4", "<em>italic</em> and <em>this</em>, not snake_case_name or 2*3*4"),
    ("## Heading", "<strong>Heading</strong>"),
    ("[docs](https://example.com) [bad](javascript:alert(1))",
     '<a href="https://example.com" target="_blank" rel="noopener noreferrer">docs</a> [bad](javascript:alert(1))'),
    ("<img src=x onerror=alert(1)> **<b>x</b>**", "&lt;img src=x onerror=alert(1)&gt; <strong>&lt;b&gt;x&lt;/b&gt;</strong>"),
    ("Streaming **unfinish", "Streaming **unfinish"),
    ("Before\n```bash\necho **hi**\n```\nafter **b**", "Before\n<pre>echo **hi**</pre>after <strong>b</strong>"),
    ("* star\n  - nested", "• star\n  • nested"),
]


def renderer_source() -> str:
    start = PAGE.index("// Light Markdown")
    return PAGE[start:PAGE.index("// Speak sentence", start)]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to run the page's JavaScript")
def test_chat_page_markdown_rendering() -> None:
    script = FAKE_DOM + renderer_source() + r"""
const cases = JSON.parse(process.argv[1]);
console.log(JSON.stringify(cases.map(([text]) => { const d = new El("div"); render(d, text); return d.kids.map(k => k.html()).join(""); })));
"""
    result = subprocess.run(["node", "-e", script, json.dumps(CASES)], capture_output=True, text=True, timeout=30, check=True)
    got = json.loads(result.stdout)
    for (text, want), actual in zip(CASES, got):
        assert actual == want, f"{text!r} rendered as {actual!r}"
