"""Web and phone interface: chat with Jarvis in a browser, with streaming replies, approvals and voice,
plus a dashboard of CI health, alerts and session stats at /dashboard.

Voice uses the browser's own speech recognition and speech synthesis (Web Speech API), so a phone
works as a microphone and speaker with nothing installed. Every request needs the secret token
printed at start-up; the server listens on localhost unless told otherwise.
"""

from __future__ import annotations

import hmac
import json
import queue
import secrets
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import anthropic

from .agent import Jarvis
from . import icons
from .dashboard import Dashboard

APPROVAL_TIMEOUT_S = 300
MAX_ACTIVITY = 40
ACTIVITY_TEXT = 160
MAX_BODY = 64_000


class WebSession:
    """Runs one Jarvis turn at a time and turns its callbacks into a stream of JSON events."""

    def __init__(self, jarvis: Jarvis, auto_approve: bool = False) -> None:
        self.jarvis = jarvis
        self.auto_approve = auto_approve
        self.lock = threading.Lock()
        self.events: queue.Queue[dict[str, Any] | None] | None = None
        self.pending: dict[str, tuple[threading.Event, list[bool]]] = {}
        # What Jarvis has been doing, newest first, for the dashboard's activity feed.
        self.activity: deque[dict[str, Any]] = deque(maxlen=MAX_ACTIVITY)
        jarvis.ctx.approve = self.approve_tool
        jarvis.ctx.confirm_private = self.confirm_private
        jarvis.on_delta = lambda text: self._emit({"type": "delta", "text": text})
        jarvis.on_text = lambda text: self._emit({"type": "block_end"})
        jarvis.on_tool = self._on_tool
        jarvis.on_notice = self._on_notice

    def _record(self, kind: str, text: str) -> None:
        text = " ".join(text.split())
        self.activity.appendleft({"time": time.time(), "kind": kind,
                                  "text": text[:ACTIVITY_TEXT] + ("…" if len(text) > ACTIVITY_TEXT else "")})

    def recent_activity(self) -> list[dict[str, Any]]:
        return list(self.activity)

    def _on_notice(self, text: str) -> None:
        self._record("notice", text)
        self._emit({"type": "notice", "text": text})

    def _on_tool(self, line: str) -> None:
        self._record("tool", line)
        self._emit({"type": "tool", "line": line})

    def _emit(self, event: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.put(event)

    def _ask(self, question: str) -> bool:
        if self.events is None:
            return False
        approval_id = uuid.uuid4().hex
        done, answer = threading.Event(), [False]
        self.pending[approval_id] = (done, answer)
        self._record("approval", question)
        self._emit({"type": "approval", "id": approval_id, "question": question})
        done.wait(APPROVAL_TIMEOUT_S)  # unanswered approvals are declined
        self.pending.pop(approval_id, None)
        self._record("allowed" if answer[0] else "declined", question)
        return answer[0]

    def approve_tool(self, question: str) -> bool:
        return True if self.auto_approve else self._ask(question)

    def confirm_private(self, question: str) -> bool:
        return self._ask(question)  # screenshots always ask, even with auto-approve

    def answer(self, approval_id: str, approved: bool) -> bool:
        entry = self.pending.get(approval_id)
        if entry is None:
            return False
        entry[1][0] = approved
        entry[0].set()
        return True

    def start_turn(self, message: str) -> queue.Queue[dict[str, Any] | None] | None:
        """Start a turn on a worker thread. Returns its event queue, or None if a turn is running."""
        if not self.lock.acquire(blocking=False):
            return None
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.events = events
        self._record("ask", message)

        def work() -> None:
            checkpoint = len(self.jarvis.messages)
            try:
                reply = self.jarvis.ask(message)
                self._record("done", reply or "(no reply text)")
                events.put({"type": "done", "reply": reply})
            except anthropic.APIError as err:
                del self.jarvis.messages[checkpoint:]
                self._record("error", f"Claude API error: {getattr(err, 'message', err)}")
                events.put({"type": "error", "message": f"Claude API error: {getattr(err, 'message', err)}"})
            except Exception as err:  # noqa: BLE001 - report anything to the browser rather than hang
                del self.jarvis.messages[checkpoint:]
                self._record("error", f"{type(err).__name__}: {err}")
                events.put({"type": "error", "message": f"{type(err).__name__}: {err}"})
            finally:
                events.put(None)
                self.events = None
                self.lock.release()

        threading.Thread(target=work, name="jarvis-web-turn", daemon=True).start()
        return events


def make_handler(session: WebSession, token: str, dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Jarvis"

        def log_message(self, *args: object) -> None:  # keep the terminal quiet
            return None

        def _authorised(self) -> bool:
            supplied = self.headers.get("X-Jarvis-Token") or parse_qs(urlparse(self.path).query).get("token", [""])[0]
            return hmac.compare_digest(supplied.encode(), token.encode())

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def _bytes(self, body: bytes, content_type: str, cache: str = "no-store") -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            # Home-screen icons hold no secrets, and phones fetch them without the token.
            if path in ICON_ROUTES:
                kind, payload = ICON_ROUTES[path]
                self._bytes(payload() if callable(payload) else payload, kind, cache="public, max-age=86400")
                return
            pages = {"/": PAGE, "/dashboard": DASHBOARD_PAGE}
            if path not in pages and path not in ("/api/dashboard", "/api/team", "/manifest.webmanifest") and not path.startswith("/api/runs/"):
                self._json(404, {"error": "not found"})
                return
            if not self._authorised():
                self._json(401, {"error": "Open the URL with the token Jarvis printed at start-up."})
                return
            if path == "/api/team":
                team = session.jarvis.team
                self._json(200, team.snapshot() if team else {"enabled": False})
                return
            if path == "/api/dashboard":
                self._json(200, dashboard.snapshot(busy=session.lock.locked(), activity=session.recent_activity(),
                                                   waiting=len(session.pending)))
                return
            if path.startswith("/api/runs/"):
                log = dashboard.read_run_log(path.removeprefix("/api/runs/"))
                if log is None:
                    self._json(404, {"error": "no such run log"})
                else:
                    self._json(200, log)
                return
            if path == "/manifest.webmanifest":
                # The installed app opens the dashboard with the token, like a bookmark of the printed link.
                self._bytes(json.dumps(manifest(token)).encode(), "application/manifest+json")
                return
            body = pages[path].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")  # the token is in the URL
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorised():
                self._json(401, {"error": "unauthorised"})
                return
            try:
                data = self._body()
            except (ValueError, json.JSONDecodeError) as err:
                self._json(400, {"error": str(err)})
                return
            path = urlparse(self.path).path
            if path == "/api/approve":
                ok = session.answer(str(data.get("id", "")), bool(data.get("approved")))
                self._json(200 if ok else 404, {"ok": ok})
            elif path in ("/api/jobs/run", "/api/jobs/briefing", "/api/jobs/stop"):
                if dashboard.jobs is None:
                    self._json(404, {"error": "Jobs aren't available in this session."})
                    return
                if path == "/api/jobs/run":
                    error = dashboard.jobs.start_run(str(data.get("goal", "")))
                elif path == "/api/jobs/stop":
                    error = dashboard.jobs.cancel()
                else:
                    error = dashboard.jobs.start_briefing()
                self._json(409 if error else 202, {"error": error} if error else {"ok": True})
            elif path == "/api/dashboard/refresh":
                ok = dashboard.refresh()
                self._json(200 if ok else 409, {"ok": ok} if ok else {"error": "No repositories are being watched. Start Jarvis with --web --watch."})
            elif path == "/api/reset":
                if session.lock.locked():
                    self._json(409, {"error": "Jarvis is busy."})
                else:
                    session.jarvis.reset()
                    self._json(200, {"ok": True})
            elif path == "/api/chat":
                message = str(data.get("message", "")).strip()
                if not message:
                    self._json(400, {"error": "empty message"})
                    return
                events = session.start_turn(message)
                if events is None:
                    self._json(409, {"error": "Jarvis is still working on the last request."})
                    return
                self._stream(events)
            else:
                self._json(404, {"error": "not found"})

        def _stream(self, events: queue.Queue[dict[str, Any] | None]) -> None:
            """Newline-delimited JSON events until the turn ends; the browser reads them as they arrive."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                event = events.get()
                if event is None:
                    break
                try:
                    self.wfile.write(json.dumps(event).encode() + b"\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    continue  # the browser left; let the turn finish so the conversation stays valid
            self.close_connection = True

    return Handler


ICON_ROUTES: dict[str, tuple[str, Any]] = {
    "/icon.svg": ("image/svg+xml", icons.SVG.encode()),
    "/icon-192.png": ("image/png", lambda: icons.png(192)),
    "/icon-512.png": ("image/png", lambda: icons.png(512)),
    "/apple-touch-icon.png": ("image/png", lambda: icons.png(180, rounded=False)),
}


def manifest(token: str) -> dict[str, Any]:
    return {
        "name": "Jarvis", "short_name": "Jarvis", "display": "standalone",
        "start_url": f"/dashboard?token={token}", "scope": "/",
        "background_color": "#0A0A0A", "theme_color": "#0A0A0A",
        "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
                  {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
    }


def serve(session: WebSession, host: str = "127.0.0.1", port: int = 8765, token: str | None = None,
          dashboard: Dashboard | None = None) -> tuple[ThreadingHTTPServer, str]:
    token = token or secrets.token_urlsafe(24)
    dashboard = dashboard or Dashboard(session.jarvis, auto_approve=session.auto_approve)
    server = ThreadingHTTPServer((host, port), make_handler(session, token, dashboard))
    server.daemon_threads = True
    return server, token


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0A0A0A">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Jarvis">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<title>Jarvis</title>
<style>
/*BRAND*/
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 var(--font); display:flex; flex-direction:column; height:100dvh; }
  header label { font-size:14px; color:var(--muted); display:flex; gap:6px; align-items:center; }
  button { font:inherit; border-radius:10px; border:1px solid var(--line); background:var(--panel); color:var(--text); padding:8px 14px; cursor:pointer; }
  button.primary { background:var(--gold); border-color:var(--gold); color:var(--on-gold); font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:min(760px, 100%); padding:10px 14px; border-radius:14px; background:var(--panel); border:1px solid var(--line);
         white-space:pre-wrap; overflow-wrap:anywhere; }
  .you { align-self:flex-end; background:var(--you); }
  .tool { font-size:13px; color:var(--muted); padding:0 4px; }
  .error { color:var(--warn); }
  .notice { border-color:var(--warn); color:var(--muted); font-size:14px; }
  pre { background:var(--code); padding:10px; border-radius:8px; overflow-x:auto; white-space:pre; margin:6px 0; font-size:14px; }
  .msg code { background:var(--code); padding:1px 5px; border-radius:5px; font-size:.92em; }
  .approval { border-color:var(--warn); }
  .approval div { margin-top:8px; display:flex; gap:8px; }
  form { display:flex; gap:8px; padding:12px 16px calc(12px + env(safe-area-inset-bottom)); border-top:1px solid var(--line); background:var(--panel); }
  textarea { flex:1; resize:none; font:inherit; padding:10px; border-radius:10px; border:1px solid var(--line); background:var(--bg); color:var(--text); min-height:44px; max-height:160px; }
  .navlink { color:var(--accent); text-decoration:none; font-size:15px; }
  #mic.on { background:var(--gold); border-color:var(--gold); color:var(--on-gold); }
  #crew { border-bottom:1px solid var(--line); background:var(--panel); padding:8px 16px 10px; }
  #crew[hidden] { display:none; }
  #crew summary { cursor:pointer; font-size:14px; color:var(--muted); display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  #crew summary b { color:var(--text); }
  .badge { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }
  .badge.passed, .badge.clean { color:var(--ok); border-color:var(--ok); }
  .badge.failed, .badge.unverified { color:var(--warn); border-color:var(--warn); }
  .meter { flex:1; min-width:120px; max-width:240px; height:6px; border-radius:3px; background:var(--code); overflow:hidden; }
  .meter i { display:block; height:100%; background:var(--accent); }
  .meter.hot i { background:var(--warn); }
  .agents { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:8px; margin-top:8px; }
  .agent { border:1px solid var(--line); border-radius:10px; padding:8px 10px; font-size:13px; min-width:0; }
  .agent .top { display:flex; align-items:center; gap:6px; font-weight:600; font-size:14px; }
  .agent .meta, .agent .last { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--muted); flex:none; }
  .s-working .dot { background:var(--accent); animation:pulse 1s ease-in-out infinite; }
  .s-done .dot, .s-chief .dot { background:var(--ok); }
  .s-error .dot { background:var(--warn); }
  .s-error .last { color:var(--warn); }
  @keyframes pulse { 50% { opacity:.3; } }
  @media (prefers-reduced-motion: reduce) { .s-working .dot { animation:none; } }
</style>
</head>
<body>
<header>
  <h1 class="wordmark">JARVIS</h1>
  <label><input type="checkbox" id="speak"> Speak replies</label>
  <a id="dash" class="navlink" href="#">Dashboard</a>
  <button id="reset" type="button">New chat</button>
</header>
<details id="crew" hidden open>
  <summary><b>Crew</b> <span class="badge" id="gate"></span> <span id="busy"></span>
    <span class="meter" id="meter" hidden><i></i></span> <span id="budget"></span></summary>
  <div class="agents" id="agents"></div>
</details>
<main id="log" aria-live="polite"></main>
<form id="form">
  <button id="mic" type="button" aria-label="Talk" title="Talk">🎙</button>
  <textarea id="input" rows="1" placeholder="Ask Jarvis anything…" autofocus></textarea>
  <button class="primary" id="send" type="submit">Send</button>
</form>
<script>
const token = new URLSearchParams(location.search).get("token") || "";
{ const m = document.createElement("link"); m.rel = "manifest"; m.href = "/manifest.webmanifest?token=" + encodeURIComponent(token); document.head.append(m); }
const log = document.getElementById("log"), input = document.getElementById("input");
const send = document.getElementById("send"), mic = document.getElementById("mic"), speakBox = document.getElementById("speak");
document.getElementById("dash").href = "/dashboard?token=" + encodeURIComponent(token);
// The dashboard's "Fix with Jarvis" opens the chat with the request filled in (in the fragment, never sent to
// the server); you read it and press Send.
const asked = new URLSearchParams(location.hash.slice(1)).get("ask");
if (asked) { document.getElementById("input").value = asked; history.replaceState(null, "", location.pathname + location.search); }
const api = (path, body) => fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Jarvis-Token": token}, body: JSON.stringify(body || {})});

function add(cls, text) { const el = document.createElement("div"); el.className = cls; el.textContent = text || ""; log.appendChild(el); log.scrollTop = log.scrollHeight; return el; }
// Light Markdown: **bold**, *italic*, _italic_, `code`, https links, "- " bullets and "#" headings.
// Built from DOM nodes and textContent only (never innerHTML), so model text can't inject markup.
// Unclosed markers (common mid-stream) stay as plain text until the closing marker arrives.
const INLINE = /(`[^`\n]+`)|\*\*([^*\n]+?)\*\*|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(?<![\w*])\*(?![\s*])([^*\n]+?)(?<!\s)\*(?![\w*])|(?<![\w])_(?![\s_])([^_\n]+?)(?<!\s)_(?![\w])/g;
function inline(parent, text) {
  const re = new RegExp(INLINE.source, "g");   // own copy: the bold branch recurses, and a shared lastIndex would loop
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    let node;
    if (m[1]) { node = document.createElement("code"); node.textContent = m[1].slice(1, -1); }
    else if (m[2] !== undefined) { node = document.createElement("strong"); inline(node, m[2]); }
    else if (m[3] !== undefined) { node = document.createElement("a"); node.href = m[4]; node.target = "_blank"; node.rel = "noopener noreferrer"; node.textContent = m[3]; }
    else { node = document.createElement("em"); node.textContent = m[5] !== undefined ? m[5] : m[6]; }
    parent.appendChild(node);
    last = re.lastIndex;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}
function prose(el, text) {
  text.split("\n").forEach((line, i) => {
    if (i) el.appendChild(document.createTextNode("\n"));
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) { const s = document.createElement("strong"); inline(s, heading[1]); el.appendChild(s); return; }
    const bullet = line.match(/^(\s*)[-*+]\s+(.*)$/);
    if (bullet) { el.appendChild(document.createTextNode(bullet[1] + "• ")); inline(el, bullet[2]); return; }
    inline(el, line);
  });
}
function renderMarkdown(el, text) {   // code blocks as <pre>, the rest as light Markdown (never innerHTML)
  el.textContent = "";
  text.split(/```[^\n]*\n?/).forEach((part, i) => {
    if (i % 2) { const pre = document.createElement("pre"); pre.textContent = part.replace(/\n$/, ""); el.appendChild(pre); }
    else if (part) prose(el, part);
  });
}

// Speak sentence by sentence as the reply streams in; code blocks stay on screen.
let spoken = "", fence = false;
function speakNew(full, final) {
  if (!speakBox.checked || !("speechSynthesis" in window)) return;
  let rest = full.slice(spoken.length);
  const re = final ? /[\s\S]+/ : /[\s\S]*?(?:[.!?:](?=\s)|\n)/;
  let m;
  while ((m = rest.match(re)) && m[0]) {
    let chunk = m[0]; spoken += chunk; rest = rest.slice(chunk.length);
    const pieces = chunk.split("```");
    let say = "";
    pieces.forEach((p, i) => { if (i > 0) { fence = !fence; if (fence) say += " The code is on screen. "; } if (!fence) say += p; });
    say = say.replace(/[*_#`>|]/g, " ").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").trim();
    if (say) speechSynthesis.speak(new SpeechSynthesisUtterance(say));
    if (final) break;
  }
}

async function chat(text) {
  add("msg you", text); input.value = ""; send.disabled = true; speechSynthesis && speechSynthesis.cancel();
  spoken = ""; fence = false;
  let bubble = null, full = "";
  try {
    const res = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json", "X-Jarvis-Token": token}, body: JSON.stringify({message: text})});
    if (!res.ok) { add("msg error", (await res.json()).error || res.statusText); return; }
    const reader = res.body.getReader(), decoder = new TextDecoder(); let buffer = "";
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const event = JSON.parse(buffer.slice(0, nl)); buffer = buffer.slice(nl + 1);
        if (event.type === "delta") { if (!bubble) { bubble = add("msg"); full = ""; spoken = ""; fence = false; } full += event.text; renderMarkdown(bubble, full); speakNew(full, false); log.scrollTop = log.scrollHeight; }
        else if (event.type === "block_end") { if (bubble) speakNew(full, true); bubble = null; }
        else if (event.type === "tool") add("tool", "⚙ " + event.line);
        else if (event.type === "approval") askApproval(event);
        else if (event.type === "error") add("msg error", event.message);
        else if (event.type === "notice") add("msg notice", event.text);
      }
    }
  } catch (err) { add("msg error", "Connection lost: " + err); }
  finally { send.disabled = false; input.focus(); }
}

function askApproval(event) {
  const box = add("msg approval", event.question), row = document.createElement("div");
  for (const [label, ok] of [["Allow", true], ["Decline", false]]) {
    const b = document.createElement("button"); b.textContent = label; if (ok) b.className = "primary";
    b.onclick = async () => { row.remove(); box.appendChild(document.createTextNode(ok ? "  ✓ allowed" : "  ✗ declined")); await api("/api/approve", {id: event.id, approved: ok}); };
    row.appendChild(b);
  }
  box.appendChild(row);
}

// Crew panel: shown in team mode, refreshed every 2 seconds.
const crew = document.getElementById("crew"), agentsBox = document.getElementById("agents");
const fmt = n => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(n);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
async function refreshCrew() {
  try {
    const res = await fetch("/api/team", {headers: {"X-Jarvis-Token": token}});
    if (!res.ok) return;
    const team = await res.json();
    if (!team.enabled) { crew.hidden = true; return false; }
    crew.hidden = false;
    const gate = document.getElementById("gate");
    gate.textContent = "Quality gate: " + team.gate; gate.className = "badge " + team.gate;
    const working = team.agents.filter(a => a.status === "working").map(a => a.name);
    document.getElementById("busy").textContent = working.length ? "Working: " + working.join(", ") : "";
    const meter = document.getElementById("meter"), budget = document.getElementById("budget");
    if (team.budget) {
      const share = Math.min(1, team.budget.used / team.budget.limit);
      meter.hidden = false; meter.firstElementChild.style.width = (share * 100) + "%"; meter.classList.toggle("hot", share >= 0.8);
      budget.textContent = fmt(team.budget.used) + " / " + fmt(team.budget.limit) + " tokens";
    } else { meter.hidden = true; budget.textContent = ""; }
    agentsBox.textContent = "";
    for (const a of team.agents) {
      const card = el("div", "agent s-" + a.status);
      card.title = a.title + " · " + a.model + (a.last ? "\n" + a.last : "");
      const top = el("div", "top"); top.append(el("span", "dot"), el("span", "", a.name)); card.appendChild(top);
      const tasks = a.tasks === null ? "chief" : a.tasks + (a.tasks === 1 ? " task" : " tasks");
      card.appendChild(el("div", "meta", tasks + " · " + fmt(a.input + a.output) + " tok"));
      const last = a.status === "chief" ? "Leads the crew" : a.status === "idle" ? a.mode : (a.last || a.status);
      card.appendChild(el("div", "last", last));
      agentsBox.appendChild(card);
    }
    return true;
  } catch (err) { return; }
}
if (innerWidth < 600) crew.open = false;  // phones: the summary line is enough until tapped
refreshCrew().then(on => { if (on !== false) setInterval(refreshCrew, 2000); });

document.getElementById("form").onsubmit = e => { e.preventDefault(); const t = input.value.trim(); if (t) chat(t); };
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); document.getElementById("form").requestSubmit(); } });
document.getElementById("reset").onclick = async () => { const r = await api("/api/reset"); if (r.ok) log.textContent = ""; else add("msg error", (await r.json()).error); };

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!Recognition) { mic.disabled = true; mic.title = "This browser has no speech recognition"; }
else {
  const rec = new Recognition(); rec.interimResults = true; rec.lang = navigator.language || "en-US";
  let listening = false;
  rec.onresult = e => { input.value = Array.from(e.results).map(r => r[0].transcript).join(""); };
  rec.onend = () => { listening = false; mic.classList.remove("on"); const t = input.value.trim(); if (t && !send.disabled) { speakBox.checked = true; chat(t); } };
  mic.onclick = () => { if (listening) { rec.stop(); return; } speechSynthesis && speechSynthesis.cancel(); input.value = ""; listening = true; mic.classList.add("on"); rec.start(); };
}
</script>
</body>
</html>
"""


DASHBOARD_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0A0A0A">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Jarvis">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<title>Jarvis Dashboard</title>
<style>
/*BRAND*/
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 var(--font); }
  header { position:sticky; top:0; z-index:1; }
  a { color:var(--accent); text-decoration:none; }
  button { font:inherit; border-radius:10px; border:1px solid var(--line); background:var(--panel); color:var(--text); padding:6px 12px; cursor:pointer; }
  button.primary { background:var(--gold); border-color:var(--gold); color:var(--on-gold); font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  main { max-width:1100px; margin:0 auto; padding:16px; display:grid; gap:16px; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
  .kpi, section { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
  .kpi .label { color:var(--muted); font-size:13px; }
  .kpi .value { font-size:28px; font-weight:600; font-variant-numeric:tabular-nums; }
  .kpi .sub { color:var(--muted); font-size:12px; }
  .bad { color:var(--bad); } .ok { color:var(--ok); }
  h2 { font-size:15px; margin:0 0 10px; display:flex; align-items:center; gap:8px; }
  h2 .meta { margin-left:auto; color:var(--muted); font-weight:400; font-size:13px; }
  .cols { display:grid; grid-template-columns:2fr 1fr; gap:16px; align-items:start; }
  @media (max-width: 820px) { .cols { grid-template-columns:1fr; } }
  .repos { display:grid; gap:10px; }
  .repo { border:1px solid var(--line); border-radius:12px; padding:10px 12px; }
  .repo.failing { border-color:var(--bad); }
  .repo-head { display:flex; gap:8px; align-items:center; font-weight:600; overflow-wrap:anywhere; }
  .repo-head .count { margin-left:auto; font-weight:400; font-size:13px; color:var(--muted); white-space:nowrap; }
  .wf { display:flex; align-items:center; gap:8px; padding:6px 0; border-top:1px solid var(--line); flex-wrap:wrap; }
  .wf:first-of-type { border-top:0; }
  .dot { width:10px; height:10px; border-radius:50%; flex:none; background:var(--ok); }
  .dot.failing { background:var(--bad); }
  .wf .name { flex:1; min-width:120px; overflow-wrap:anywhere; }
  .chip { background:var(--chip); border-radius:999px; padding:1px 8px; font-size:12px; color:var(--muted); font-family:ui-monospace, monospace; }
  .muted { color:var(--muted); }
  ul.list { list-style:none; margin:0; padding:0; display:grid; gap:8px; }
  ul.list li { font-size:14px; overflow-wrap:anywhere; }
  ul.list .when { color:var(--muted); font-size:12px; display:block; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:6px 12px; margin:0; font-size:14px; }
  dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
  .empty { color:var(--muted); font-size:14px; }
  .errors li { color:var(--warn); }
  .stack { display:grid; gap:16px; }
  .pill { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
  .pill.busy { color:var(--warn); border-color:var(--warn); }
  a.pill.busy { font-weight:600; }
  header h1 { white-space:nowrap; }
  @media (max-width: 480px) { header .wide { display:none; } header { gap:8px; } }
  .hist { display:inline-flex; gap:2px; align-items:flex-end; }
  .hist i { width:6px; height:14px; border-radius:2px; background:var(--ok); opacity:.85; }
  .hist i.failing { background:var(--bad); } .hist i.other { background:var(--line); height:8px; }
  .rate { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; min-width:34px; text-align:right; }
  .tag { font-size:11px; border:1px solid var(--line); border-radius:6px; padding:0 5px; color:var(--muted); margin-left:4px; }
  .act { display:grid; grid-template-columns:auto 1fr; gap:2px 8px; align-items:baseline; }
  .act .kind { font-size:12px; font-family:ui-monospace, monospace; color:var(--muted); min-width:58px; }
  .act .kind.error, .act .kind.declined { color:var(--bad); } .act .kind.approval { color:var(--warn); }
  .act .kind.done, .act .kind.allowed { color:var(--ok); }
  .act .when { grid-column:2; }
  #notify.on { background:var(--accent); border-color:var(--accent); color:var(--on-accent); }
  .run-head { display:flex; gap:8px; align-items:baseline; }
  .run-head .goal { flex:1; overflow-wrap:anywhere; }
  .status { font-size:12px; border-radius:999px; padding:0 8px; border:1px solid var(--line); white-space:nowrap; }
  .status.done { color:var(--ok); border-color:var(--ok); }
  .status.failed, .status.over { color:var(--bad); border-color:var(--bad); }
  .status.running { color:var(--warn); border-color:var(--warn); }
  .logbtn { font-size:12px; padding:2px 8px; margin-top:4px; }
  pre.runlog { background:var(--chip); border-radius:8px; padding:10px; margin:6px 0 0; max-height:420px; overflow:auto;
               font-size:12px; line-height:1.45; white-space:pre-wrap; overflow-wrap:anywhere; }
  button.small { font-size:13px; padding:3px 10px; margin-left:8px; }
  .briefing { white-space:pre-wrap; overflow-wrap:anywhere; font-size:14px; max-height:340px; overflow:auto; }
  .jobnow { display:flex; gap:10px; align-items:center; flex-wrap:wrap; border:1px solid var(--warn); border-radius:10px;
            padding:8px 10px; margin-bottom:10px; font-size:14px; }
  .jobnow[hidden] { display:none; }   /* display:flex above would otherwise override the hidden attribute */
  .jobnow .what { flex:1; min-width:160px; overflow-wrap:anywhere; }
  .runform { display:flex; gap:8px; margin-bottom:12px; align-items:flex-start; }
  .runform textarea { flex:1; font:inherit; font-size:14px; padding:8px; border-radius:10px; border:1px solid var(--line);
                      background:var(--bg); color:var(--text); resize:vertical; min-height:42px; }
  .chart { position:relative; }
  .chart svg { display:block; width:100%; height:auto; overflow:visible; }
  .chart .axis { stroke:var(--line); stroke-width:1; }
  .chart .lbl { fill:var(--muted); font-size:10px; }
  .chart .hit { fill:transparent; cursor:default; }
  .chart .bar { fill:var(--bar); pointer-events:none; }   /* the column behind it takes the hover */
  .chart .hit:hover + .bar, .chart .bar.on { opacity:.75; }
  .tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:6px 8px; font-size:12px; box-shadow:0 4px 14px rgba(0,0,0,.12); white-space:nowrap; transform:translate(-50%, -100%); }
  .spendtable { width:100%; font-size:13px; border-collapse:collapse; margin-top:6px; }
  .spendtable td, .spendtable th { text-align:left; padding:2px 4px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
  .spendtable td:last-child, .spendtable th:last-child { text-align:right; }
  details summary { cursor:pointer; color:var(--muted); font-size:13px; margin-top:6px; }
  .tag.brief { color:var(--accent); border-color:var(--accent); }
  .meter { height:6px; border-radius:999px; background:var(--chip); margin-top:6px; overflow:hidden; }
  .meter i { display:block; height:100%; background:var(--ok); }
  .meter i.warn { background:var(--warn); } .meter i.over { background:var(--bad); }
</style>
</head>
<body>
<header>
  <h1 class="wordmark">JARVIS<span class="wide sub"> · Dashboard</span></h1>
  <a id="state" class="pill">…</a>
  <button id="notify" type="button" title="Notify me when a build breaks" hidden>🔔</button>
  <button id="refresh" type="button">Check now</button>
  <a id="chat" href="#">Chat</a>
</header>
<main>
  <div class="kpis" id="kpis"></div>
  <section id="briefing-card">
    <h2>Briefing <span class="meta" id="briefing-meta"></span><button id="briefing-go" type="button" class="small">Generate</button></h2>
    <div id="briefing" class="briefing"></div>
  </section>
  <div class="cols">
    <div class="stack">
    <section>
      <h2>CI on default branches <span class="meta" id="checked"></span></h2>
      <div id="repos" class="repos"></div>
      <ul id="errors" class="list errors"></ul>
    </section>
    <section><h2>Open pull requests <span class="meta" id="prcount"></span></h2><ul id="pulls" class="list"></ul></section>
    <section><h2>Autonomous runs <span class="meta">latest 10</span></h2>
      <div id="jobnow" class="jobnow" hidden></div>
      <form id="runform" class="runform">
        <textarea id="goal" rows="2" maxlength="4000" placeholder="Goal for a new autonomous run, e.g. Fix the failing Android build in owner/repo"></textarea>
        <button id="rungo" class="primary" type="submit">Start run</button>
      </form>
      <p id="runhint" class="empty" hidden></p>
      <ul id="runs" class="list"></ul></section>
    </div>
    <div class="stack">
      <section id="spend-card"><h2>Spending <span class="meta" id="spend-meta"></span></h2>
        <div class="chart" id="spend"></div>
        <details><summary>Show as table</summary><table class="spendtable" id="spendtable"></table></details>
      </section>
      <section><h2>Activity</h2><ul id="activity" class="list"></ul></section>
      <section><h2>Alerts</h2><ul id="alerts" class="list"></ul></section>
      <section><h2>Session</h2><dl id="session"></dl></section>
      <section><h2>Memory <span class="meta" id="memcount"></span></h2><ul id="memory" class="list"></ul></section>
    </div>
  </div>
</main>
<script>
const token = new URLSearchParams(location.search).get("token") || "";
{ const m = document.createElement("link"); m.rel = "manifest"; m.href = "/manifest.webmanifest?token=" + encodeURIComponent(token); document.head.append(m); }
const headers = {"X-Jarvis-Token": token};
document.getElementById("chat").href = "/?token=" + encodeURIComponent(token);
const $ = id => document.getElementById(id);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function link(url, text) {   // only GitHub-style https links become clickable
  if (typeof url === "string" && url.startsWith("https://")) { const a = el("a", "", text); a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; return a; }
  return el("span", "muted", text);
}
const money = v => v == null ? "–" : v < 0.01 && v > 0 ? "<$0.01" : "$" + v.toFixed(2);
const plural = (n, word) => n + " " + word + (n === 1 ? "" : "s");
const fmt = n => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k" : n.toLocaleString();
function ago(t, now) {
  if (!t) return "never"; const s = Math.max(0, Math.round(now - t));
  return s < 60 ? s + "s ago" : s < 3600 ? Math.floor(s / 60) + " min ago" : Math.floor(s / 3600) + " h ago";
}
function agoIso(iso, now) { const t = Date.parse(iso); return isNaN(t) ? "unknown" : ago(t / 1000, now); }
function kpi(label, value, sub, cls) {
  const box = el("div", "kpi"); box.append(el("div", "label", label), el("div", "value " + (cls || ""), value));
  if (sub) box.append(el("div", "sub", sub)); return box;
}
const short = (text, n) => text.length > n ? text.slice(0, n).trimEnd() + "…" : text;   // long goals: the start, full text on hover
let jobsState = null;   // the latest jobs status, for the Fix buttons
function fixLink(prompt) {
  const b = el("button", "primary", "Fix with Jarvis");
  b.onclick = async () => {
    if (jobsState && jobsState.allow_runs) {   // started with --auto: one tap starts a background fix run
      if (jobsState.current) { alert("A " + jobsState.current.kind + " is already running. Stop it or wait for it to finish."); return; }
      if (!confirm("Start an autonomous fix run? Jarvis will investigate, change code and open a draft pull request for you to review. It counts toward the spending limit.")) return;
      if (await startJob("/api/jobs/run", {goal: prompt})) b.textContent = "Fix started";
      return;
    }
    location.href = "/?token=" + encodeURIComponent(token) + "#ask=" + encodeURIComponent(prompt);   // otherwise: the chat, to review and send
  };
  return b;
}

function render(d) {
  const ci = d.ci, s = d.session, u = s.usage || {};
  const state = $("state");
  if (s.waiting_for_approval) {   // an approval is waiting in the chat: make it obvious and one tap away
    state.textContent = "Needs your approval"; state.className = "pill busy"; state.href = $("chat").href;
  } else { state.textContent = s.busy ? "Working" : "Idle"; state.className = "pill" + (s.busy ? " busy" : ""); state.removeAttribute("href"); }
  const tokens = (u.input_tokens || 0) + (u.output_tokens || 0) + (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0);
  const k = $("kpis"); k.textContent = "";
  if (ci.enabled) {
    k.append(kpi("Failing workflows", String(ci.totals.failing), ci.totals.failing ? "needs attention" : "all green", ci.totals.failing ? "bad" : "ok"),
             kpi("Passing workflows", String(ci.totals.passing), "of " + ci.totals.workflows),
             kpi("Repositories", String(ci.totals.repos), "watched"),
             kpi("Open pull requests", String(ci.pulls.length), plural(ci.pulls.filter(p => p.draft).length, "draft")));
  } else {
    k.append(kpi("CI watch", "Off", "start with --web --watch"));
  }
  const costTile = kpi("Cost this session", money(s.cost_usd),
    s.budget_usd ? "of " + money(s.budget_usd) + " limit · " + fmt(tokens) + " tokens" : fmt(tokens) + " tokens · " + plural(u.requests || 0, "request"));
  if (s.budget_usd && s.cost_usd != null) {
    const share = s.cost_usd / s.budget_usd, bar = el("div", "meter"), fill = el("i", share >= 1 ? "over" : share >= 0.8 ? "warn" : "");
    fill.style.width = Math.min(100, Math.round(share * 100)) + "%"; bar.append(fill); costTile.append(bar);
  }
  k.append(costTile);

  const repos = $("repos"); repos.textContent = "";
  $("refresh").disabled = !ci.enabled || ci.checking;
  if (!ci.enabled) {
    repos.append(el("p", "empty", "Not watching any repositories. Start Jarvis with “jarvis --web --watch” (your account) or “jarvis --web --watch owner owner/repo”."));
    $("checked").textContent = "";
  } else {
    $("checked").textContent = ci.checking ? "checking…" : "checked " + ago(ci.last_checked, d.now) + " · every " + Math.round(ci.interval_s / 60) + " min";
    if (!ci.repos.length) repos.append(el("p", "empty", ci.last_checked ? "No completed workflow runs found on the watched default branches." : "First check in progress…"));
    for (const r of ci.repos) {
      const card = el("div", "repo" + (r.failing ? " failing" : ""));
      const head = el("div", "repo-head"); head.append(link("https://github.com/" + r.repo, r.repo),
        el("span", "count " + (r.failing ? "bad" : "ok"), r.failing ? r.failing + " failing" : "healthy"));
      card.append(head);
      for (const w of r.workflows) {
        const row = el("div", "wf");
        row.append(el("span", "dot " + w.status), el("span", "name", w.workflow));
        const h = w.history || [], graded = h.filter(x => x !== "other");
        if (h.length > 1) {
          const strip = el("span", "hist"); strip.title = "Last " + h.length + " runs, oldest first";
          for (const x of h) strip.append(el("i", x));
          const rate = graded.length ? Math.round(100 * graded.filter(x => x === "passing").length / graded.length) + "%" : "–";
          row.append(strip, el("span", "rate", rate));
        }
        row.append(el("span", "chip", w.commit), link(w.url, w.status === "failing" ? "logs" : "run"));
        if (w.fix_prompt) row.append(fixLink(w.fix_prompt));
        card.append(row);
      }
      repos.append(card);
    }
  }
  const errors = $("errors"); errors.textContent = "";
  for (const e of (ci.errors || [])) errors.append(el("li", "", e));

  const pulls = $("pulls"); pulls.textContent = "";
  $("prcount").textContent = ci.enabled ? String(ci.pulls.length) : "";
  if (!ci.enabled || !ci.pulls.length) pulls.append(el("li", "empty", ci.enabled ? "No open pull requests in the watched repositories." : "Pull requests appear when CI watch is on."));
  else for (const pr of [...ci.pulls].sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)))) {
    const li = el("li"); li.append(el("span", "when", pr.repo + " · " + (pr.author || "unknown") + " · updated " + agoIso(pr.updated_at, d.now)),
      link(pr.url, "#" + pr.number + " " + pr.title));
    if (pr.draft) li.append(el("span", "tag", "draft"));
    pulls.append(li);
  }

  renderJobs(d.jobs, d.now);
  renderSpend(d.spend);

  const runs = $("runs"); runs.textContent = "";
  if (!d.runs.length) runs.append(el("li", "empty", "No autonomous runs yet. Runs started with --auto or --watch --fix appear here."));
  for (const r of d.runs) {
    const li = el("li"), head = el("div", "run-head");
    // A run that stopped writing for 30 minutes most likely died with its process (Ctrl+C, crash, reboot).
    const stalled = r.status === "running" && d.now - (r.updated || r.started || d.now) > 1800;
    const status = stalled ? "stalled" : r.status;
    const goal = el("span", "goal", r.kind === "briefing" ? "Briefing" : short(r.goal || "(no goal)", 160));
    if (r.kind !== "briefing") goal.title = r.goal || "";
    if (r.kind === "briefing") goal.append(el("span", "tag brief", "read-only"));
    head.append(goal, el("span", "status " + (stalled ? "failed" : r.status), status));
    const meta = [ago(r.started, d.now), plural(r.steps || 0, "step")];
    if (r.errors) meta.push(plural(r.errors, "tool error"));
    if (r.cost_usd != null) meta.push(money(r.cost_usd));
    meta.push(r.log);
    li.append(head, el("span", "when", meta.join(" · ")));
    for (const url of r.pull_requests) { li.append(link(url, url.replace("https://github.com/", "")), document.createTextNode(" ")); }
    const btn = el("button", "logbtn", openLogs.has(r.log) ? "Hide log" : "Show log");
    btn.onclick = () => { if (openLogs.has(r.log)) openLogs.delete(r.log); else openLogs.add(r.log); load(); };
    const actions = el("div"); actions.append(btn); li.append(actions);
    if (openLogs.has(r.log)) {
      const pre = el("pre", "runlog", logCache[r.log] || "Loading…"); li.append(pre);
      if (!logCache[r.log] || r.status === "running") fetchLog(r.log, pre, r.status === "running");   // a running log keeps growing
    }
    runs.append(li);
  }

  document.title = (ci.enabled && ci.totals.failing ? "(" + ci.totals.failing + " failing) " : "") + "Jarvis Dashboard";
  notifyNew(ci.alerts || []);

  const act = $("activity"); act.textContent = "";
  if (!s.activity.length) act.append(el("li", "empty", "Nothing yet. Requests, tool calls and approvals from the chat show up here."));
  for (const a of s.activity.slice(0, 12)) {
    const li = el("li", "act"); li.append(el("span", "kind " + a.kind, a.kind), el("span", "", a.text), el("span", "when", ago(a.time, d.now)));
    act.append(li);
  }

  const alerts = $("alerts"); alerts.textContent = "";
  if (!ci.enabled || !ci.alerts.length) alerts.append(el("li", "empty", ci.enabled ? "No alerts yet. Builds that break or recover show up here." : "Alerts appear when CI watch is on."));
  else for (const a of ci.alerts) {
    const li = el("li"); li.append(el("span", "when", ago(a.time, d.now)), el("strong", a.kind === "failing" ? "bad" : "ok", a.kind === "failing" ? "Broke " : "Recovered "),
      document.createTextNode(a.repo + " · " + a.workflow + " "), el("span", "chip", a.commit));
    alerts.append(li);
  }

  const sess = $("session"); sess.textContent = "";
  const rows = [["Model", s.model], ["Effort", s.effort], ["Mode", s.autonomous ? "Autonomous" : s.auto_approve ? "Auto-approve" : "Asks before changes"],
                ["Tools", s.tools + (s.mcp ? " (MCP: " + s.mcp + ")" : "")], ["Up since", ago(s.started_at, d.now)],
                ["Tokens", fmt(u.input_tokens || 0) + " in · " + fmt(u.output_tokens || 0) + " out · " + fmt(u.cache_read_input_tokens || 0) + " cached"],
                ["Est. cost", s.cost_usd == null ? "no price for this model" : money(s.cost_usd) + " at list price"],
                ["Spending limit", s.budget_usd ? money(s.budget_usd) + " (--budget)" : "none"]];
  for (const [a, b] of rows) sess.append(el("dt", "", a), el("dd", "", String(b)));

  const mem = $("memory"); mem.textContent = "";
  $("memcount").textContent = s.memory.enabled ? plural(s.memory.count, "note") : "off";
  if (!s.memory.recent.length) mem.append(el("li", "empty", s.memory.enabled ? "Nothing remembered yet." : "Memory is off (--no-memory)."));
  for (const n of s.memory.recent) mem.append(el("li", "", n));
}

// Browser notifications for builds that break or recover while this page is open (needs HTTPS or localhost).
const notifyBtn = $("notify"); let lastAlert = null;
function wantNotify() { try { return localStorage.getItem("jarvis-notify") === "1"; } catch (e) { return false; } }
function setNotify(on) { try { localStorage.setItem("jarvis-notify", on ? "1" : "0"); } catch (e) {} notifyBtn.classList.toggle("on", on); }
if ("Notification" in window && window.isSecureContext) {
  notifyBtn.hidden = false; setNotify(wantNotify() && Notification.permission === "granted");
  notifyBtn.onclick = async () => {
    if (notifyBtn.classList.contains("on")) { setNotify(false); return; }
    setNotify((await Notification.requestPermission()) === "granted");
  };
}
let lastFinished = null;
function notifyFinished(f) {   // a browser notification when a run or briefing finishes (the first load only sets the baseline)
  const at = f ? f.finished : 0;
  if (lastFinished !== null && at > lastFinished && notifyBtn.classList.contains("on") && Notification.permission === "granted") {
    const title = f.kind === "briefing" ? (f.status === "done" ? "Your briefing is ready" : "Briefing " + f.status)
      : ({done: "Run finished", cancelled: "Run stopped"}[f.status] || "Run " + f.status);
    new Notification(title, {body: f.kind === "briefing" ? "Open the dashboard to read it." : f.goal.slice(0, 120), tag: "jarvis-job"});
  }
  lastFinished = Math.max(lastFinished || 0, at);
}
function notifyNew(alerts) {
  const newest = alerts.length ? alerts[0].time : 0;
  if (lastAlert !== null && notifyBtn.classList.contains("on") && Notification.permission === "granted") {
    for (const a of alerts.filter(a => a.time > lastAlert).reverse()) {
      new Notification(a.kind === "failing" ? "Build broke: " + a.workflow : "Build recovered: " + a.workflow,
                       {body: a.repo + " · " + a.commit, tag: a.repo + "/" + a.workflow});
    }
  }
  lastAlert = Math.max(lastAlert || 0, newest);   // the first load only sets the baseline
}

// Run logs open in place. Which ones are open survives the 15-second refresh; texts are cached.
const openLogs = new Set(), logCache = {};
async function fetchLog(name, pre, follow) {
  try {
    const res = await fetch("/api/runs/" + encodeURIComponent(name), {headers});
    const data = await res.json();
    logCache[name] = res.ok ? (data.truncated ? "… (start of the log cut; showing the end)\n\n" : "") + data.text : (data.error || "Couldn't load the log.");
  } catch (err) { logCache[name] = "Couldn't load the log: " + err; }
  if (pre.isConnected) {   // a running log follows its end unless you scrolled up; a finished one opens at the top
    const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
    pre.textContent = logCache[name]; if (follow && atEnd) pre.scrollTop = pre.scrollHeight;
  }
}

/*MARKDOWN*/
// Briefing card, run form and job state. Jobs run in the background on the server; the page only starts them.
function renderJobs(j, now) {
  const card = $("briefing-card"), form = $("runform"), hint = $("runhint"), banner = $("jobnow");
  jobsState = j;
  if (!j) { card.hidden = true; form.hidden = true; hint.hidden = true; banner.hidden = true; return; }
  banner.hidden = !j.current; banner.textContent = "";
  if (j.current) {
    const what = j.current.kind === "briefing" ? "Making the briefing" : "Running: " + j.current.goal;
    const label = el("span", "what", short(what, 110) + " · started " + ago(j.current.started, now)); label.title = what;
    banner.append(label);
    const stop = el("button", "", j.current.stopping ? "Stopping…" : "Stop");
    stop.disabled = !!j.current.stopping;
    stop.title = "Stops before Jarvis's next step; a command already running finishes first.";
    stop.onclick = async () => { if (confirm("Stop this " + j.current.kind + "? Changes it already made stay as they are.")) await startJob("/api/jobs/stop"); };
    banner.append(stop);
  }
  notifyFinished(j.last_finished);
  const busy = !!j.current, b = j.briefing;
  card.hidden = false;
  const go = $("briefing-go");
  go.disabled = busy; go.textContent = busy && j.current.kind === "briefing" ? "Working…" : (b ? "Refresh" : "Generate");
  const meta = [];
  if (b) { meta.push("made " + ago(b.generated_at, now)); if (b.cost_usd != null) meta.push(money(b.cost_usd)); }
  if (j.daily_at) meta.push("daily at " + j.daily_at);
  $("briefing-meta").textContent = meta.join(" · ");
  const box = $("briefing");
  if (busy && j.current.kind === "briefing") box.textContent = "Jarvis is reading your repositories… (started " + ago(j.current.started, now) + ")";
  else if (b) renderMarkdown(box, b.text);
  else box.textContent = "No briefing yet. Generate one to see what needs your attention across your repositories: failing CI, pull requests waiting for review and new issues. It uses tokens and changes nothing" + (j.daily_at ? "; one is also made each day at " + j.daily_at + "." : ".");
  if (j.last_error) { const e = el("p", "error", j.last_error); box.append(document.createElement("br"), e); }

  form.hidden = !j.allow_runs; hint.hidden = j.allow_runs;
  if (!j.allow_runs) hint.textContent = "To start runs from here, start Jarvis with --web --auto. Runs started with --auto or --watch --fix still appear below.";
  $("rungo").disabled = busy;
  $("rungo").textContent = busy && j.current.kind === "run" ? "Running…" : "Start run";
}
async function startJob(path, body) {
  const res = await fetch(path, {method: "POST", headers: {...headers, "Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  if (!res.ok) { let msg = res.statusText; try { msg = (await res.json()).error || msg; } catch (e) {} alert(msg); return false; }
  setTimeout(load, 400); return true;
}
$("briefing-go").onclick = () => startJob("/api/jobs/briefing");
$("runform").onsubmit = async e => {
  e.preventDefault(); const goal = $("goal").value.trim(); if (!goal) return;
  if (await startJob("/api/jobs/run", {goal})) $("goal").value = "";
};

// Spending per day: one series, so no legend; hover any bar (or read the table) for the exact figures.
let spendKey = "";
function renderSpend(spend) {
  const card = $("spend-card"); if (!spend) { card.hidden = true; return; }
  card.hidden = false;
  const key = JSON.stringify(spend.days);
  if (key === spendKey) return;   // unchanged: keep the chart (and any open tooltip) as it is
  spendKey = key;
  const days = spend.days, total = days.reduce((a, d) => a + d.usd, 0), today = days[days.length - 1];
  $("spend-meta").textContent = "last " + days.length + " days · " + money(total) + " · today " + money(today.usd);
  const W = 320, H = 120, top = 14, bottom = 18, gap = 2, n = days.length, bw = W / n;
  const max = Math.max(...days.map(d => d.usd)), scale = max > 0 ? (H - top - bottom) / max : 0;
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Estimated spending per day for the last " + n + " days; total " + money(total));
  const mk = (tag, attrs) => { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const base = H - bottom;
  svg.append(mk("line", {x1: 0, x2: W, y1: base, y2: base, class: "axis"}));
  if (max > 0) { const t = mk("text", {x: 0, y: top - 4, class: "lbl"}); t.textContent = "peak " + money(max); svg.append(t); }
  const tip = el("div", "tip"), box = $("spend"); tip.hidden = true;
  days.forEach((d, i) => {
    const x = i * bw + gap / 2, w = bw - gap, h = d.usd * scale, r = Math.min(4, w / 2, h);
    const hit = mk("rect", {x: i * bw, y: 0, width: bw, height: H, class: "hit"});   // the whole column is the target
    const bar = h > 0 ? mk("path", {class: "bar", d: "M" + x + "," + base + "V" + (base - h + r) + "Q" + x + "," + (base - h) + " " + (x + r) + "," + (base - h) +
      "H" + (x + w - r) + "Q" + (x + w) + "," + (base - h) + " " + (x + w) + "," + (base - h + r) + "V" + base + "Z"}) : mk("g", {});
    const label = new Date(d.date + "T12:00:00").toLocaleDateString(undefined, {weekday: "short", day: "numeric", month: "short"});
    const parts = Object.entries(d.by_kind).filter(([, v]) => v > 0).map(([k, v]) => k + " " + money(v));
    hit.addEventListener("pointerenter", () => {
      tip.textContent = label + ": " + money(d.usd) + (parts.length > 1 ? " (" + parts.join(", ") + ")" : parts.length ? " (" + parts[0].split(" ")[0] + ")" : "");
      tip.hidden = false; tip.style.top = Math.max(0, (base - h - 6) / H * 100) + "%";
      const room = box.clientWidth, half = tip.offsetWidth / 2, centre = (x + w / 2) / W * room;
      tip.style.left = Math.min(Math.max(centre, half), room - half) + "px";   // stay inside the card
      if (bar.classList) bar.classList.add("on");
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; if (bar.classList) bar.classList.remove("on"); });
    svg.append(hit, bar);
    if (i === 0 || i === n - 1 || i === Math.floor((n - 1) / 2)) {   // a few dates, not every one
      const t = mk("text", {x: i * bw + bw / 2, y: H - 4, "text-anchor": i === 0 ? "start" : i === n - 1 ? "end" : "middle", class: "lbl"});
      if (i === 0) t.setAttribute("x", 0); if (i === n - 1) t.setAttribute("x", W);
      t.textContent = i === n - 1 ? "today" : new Date(d.date + "T12:00:00").toLocaleDateString(undefined, {day: "numeric", month: "short"});
      svg.append(t);
    }
  });
  box.textContent = ""; box.append(svg, tip);
  const table = $("spendtable"); table.textContent = "";
  const head = document.createElement("tr"); for (const h of ["Day", "Chat", "Runs", "Briefings", "Total"]) head.append(el("th", "", h)); table.append(head);
  for (const d of [...days].reverse()) {
    const tr = document.createElement("tr");
    for (const v of [d.date, money(d.by_kind.chat || 0), money(d.by_kind.run || 0), money(d.by_kind.briefing || 0), money(d.usd)]) tr.append(el("td", "", v));
    table.append(tr);
  }
}

let timer = null;
async function load() {
  clearTimeout(timer);
  try {
    const res = await fetch("/api/dashboard", {headers});
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json(); render(data);
    timer = setTimeout(load, data.ci.checking ? 2000 : 15000);
  } catch (err) {
    $("state").textContent = "Offline"; $("state").className = "pill busy";
    timer = setTimeout(load, 15000);
  }
}
$("refresh").onclick = async () => {
  $("refresh").disabled = true;
  await fetch("/api/dashboard/refresh", {method: "POST", headers: {...headers, "Content-Type": "application/json"}, body: "{}"});
  setTimeout(load, 800);
};
document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
load();
</script>
</body>
</html>
"""


# The light Markdown renderer lives once, in the chat page; the dashboard uses the same code for the briefing.
MARKDOWN_JS = PAGE[PAGE.index("// Light Markdown"):PAGE.index("// Speak sentence")]
DASHBOARD_PAGE = DASHBOARD_PAGE.replace("/*MARKDOWN*/", MARKDOWN_JS)


# Brand: Joshua Wright's shared palette (BRAND-RULES.md in joshuawayzwright/joshuawayzwright): charcoal #0A0A0A,
# weathered steel #8A8F98, burnished gold #D4AF37, restrained electric blue #0087FF. One dark theme, because the
# palette's blue and steel are too faint for text on white. Surfaces are tints of charcoal; lines are steel.
# Red and green are functional CI status colours only, always shown with a word ("failing", "healthy"), never alone.
# Text-only wordmark: there is no approved Jarvis logo, and the brand rules forbid inventing one.
BRAND_CSS = """  :root { color-scheme:dark;
          --charcoal:#0A0A0A; --steel:#8A8F98; --gold:#D4AF37; --electric:#0087FF;
          --bg:var(--charcoal); --panel:#141517; --chip:#1C1D20; --code:#1C1D20; --line:rgba(138,143,152,.28);
          --text:#F2F2F0; --muted:var(--steel); --accent:var(--electric); --on-accent:var(--charcoal); --on-gold:var(--charcoal);
          --warn:var(--gold); --ok:#4CC38A; --bad:#FF6B5A; --bar:var(--electric); --you:rgba(0,135,255,.14);
          --font:Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
          --display:Rajdhani, Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:12px 16px; background:var(--charcoal);
           border-bottom:1px solid rgba(212,175,55,.45); }
  .wordmark { font-family:var(--display); font-size:19px; font-weight:700; letter-spacing:.2em; color:var(--gold); margin:0; flex:1; }
  .wordmark .sub { color:var(--steel); letter-spacing:.08em; font-weight:600; }
  a:focus-visible, button:focus-visible, textarea:focus-visible, input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }"""
PAGE = PAGE.replace("/*BRAND*/", BRAND_CSS)
DASHBOARD_PAGE = DASHBOARD_PAGE.replace("/*BRAND*/", BRAND_CSS)
