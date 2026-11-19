"""Spending ledger, background jobs (dashboard runs and briefings, daily briefing) and their web routes."""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import cli, jobs, web
from jarvis.agent import Jarvis, Spend
from jarvis.dashboard import Dashboard
from jarvis.spend import SpendLedger
from jarvis.tools import ToolContext
from test_voice import FakeStream

NOON = time.mktime((2026, 11, 19, 12, 0, 0, 0, 0, -1))  # local noon, 19 Nov 2026
DAY = 86_400


def opus(output_tokens: int) -> dict[str, int]:
    return {"output_tokens": output_tokens}  # Opus 5.5 output: $20 per million tokens


def test_ledger_totals_per_day_and_kind(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    assert [d["usd"] for d in ledger.daily(3, now=NOON)] == [0, 0, 0]  # no file yet
    ledger.record("claude-opus-5-5", opus(50_000), "chat", when=NOON)          # $1.00 today
    ledger.record("claude-opus-5-5", opus(25_000), "run", when=NOON - 60)      # $0.50 today
    ledger.record("claude-opus-5-5", opus(5_000), "briefing", when=NOON - DAY)  # $0.10 yesterday
    ledger.record("claude-opus-5-5", opus(5_000), "chat", when=NOON - 30 * DAY)  # outside the window
    ledger.record("no-price-model", opus(5_000), "chat", when=NOON)            # not recorded
    days = ledger.daily(3, now=NOON)
    assert [d["date"] for d in days] == ["2026-11-17", "2026-11-18", "2026-11-19"]
    assert [d["usd"] for d in days] == [0, pytest.approx(0.1), pytest.approx(1.5)]
    assert days[-1]["by_kind"] == {"chat": pytest.approx(1.0), "run": pytest.approx(0.5)}
    assert len((tmp_path / "spend.jsonl").read_text().splitlines()) == 4


def test_ledger_reads_only_new_lines_and_survives_bad_ones(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    ledger = SpendLedger(path)
    ledger.record("claude-opus-5-5", opus(50_000), "chat", when=NOON)
    assert ledger.daily(1, now=NOON)[0]["usd"] == pytest.approx(1.0)
    with path.open("a") as handle:
        handle.write("not json\n")
        handle.write('{"t": %d, "usd": 2, "kind": "weird"}\n' % NOON)  # unknown kind counts as chat
        handle.write('{"t": %d, "usd": 4' % NOON)  # half-written line: read once complete
    assert ledger.daily(1, now=NOON)[0]["usd"] == pytest.approx(3.0)
    with path.open("a") as handle:
        handle.write(', "kind": "run"}\n')
    assert ledger.daily(1, now=NOON)[0]["by_kind"] == {"chat": pytest.approx(3.0), "run": pytest.approx(4.0)}
    path.write_text("")  # truncated: starts over
    assert ledger.daily(1, now=NOON)[0]["usd"] == 0


def test_every_agent_request_goes_to_the_ledger(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    reply = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")],
                            usage=SimpleNamespace(input_tokens=0, output_tokens=50_000, cache_read_input_tokens=0,
                                                  cache_creation_input_tokens=0))
    messages = SimpleNamespace(stream=lambda **kw: FakeStream(reply))
    jarvis = Jarvis(ctx=ToolContext(workspace=tmp_path, approve=lambda q: False),
                    client=SimpleNamespace(beta=SimpleNamespace(messages=messages)),
                    on_text=lambda t: None, spend=Spend(ledger, kind="run"))
    jarvis.ask("hi")
    entry = json.loads((tmp_path / "spend.jsonl").read_text())
    assert entry["kind"] == "run" and entry["model"] == "claude-opus-5-5" and entry["usd"] == pytest.approx(1.0)


class FakeAgent:
    model = "claude-opus-5-5"

    def __init__(self, reply: str | None, gate: threading.Event | None = None, fail: bool = False) -> None:
        self.reply, self.gate, self.fail = reply, gate, fail
        self.usage = {"output_tokens": 0}
        self.asked: list[str] = []

    def ask(self, goal: str) -> str | None:
        self.asked.append(goal)
        if self.gate:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("network down")
        self.usage["output_tokens"] += 5_000
        return self.reply

    def over_budget(self) -> bool:
        return False


def runner(tmp_path: Path, agents: list, allow_runs: bool = False, **kw) -> tuple[jobs.JobRunner, list]:
    made: list = []

    def make(kind: str, log) -> FakeAgent:
        agent = agents.pop(0)
        made.append((kind, agent))
        return agent

    return jobs.JobRunner(make, tmp_path / "runs", tmp_path / "briefing.json",
                          briefing_prompt=kw.pop("prompt", lambda: "Give me my briefing."), allow_runs=allow_runs, **kw), made


def wait_idle(r: jobs.JobRunner) -> None:
    for _ in range(500):
        if r.current is None:
            return
        time.sleep(0.01)
    raise AssertionError("job didn't finish")


def test_briefing_is_saved_logged_and_survives_a_restart(tmp_path: Path) -> None:
    r, made = runner(tmp_path, [FakeAgent("CI is green. 2 PRs need review.")])
    assert r.start_briefing() is None
    wait_idle(r)
    assert made[0][0] == "briefing" and made[0][1].asked == ["Give me my briefing."]
    assert r.briefing["text"] == "CI is green. 2 PRs need review." and r.briefing["cost_usd"] == pytest.approx(0.1)
    summary = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert summary["kind"] == "briefing" and summary["status"] == "done"
    again, _ = runner(tmp_path, [])
    assert again.status()["briefing"]["text"].startswith("CI is green")


def test_runs_need_permission_and_one_job_runs_at_a_time(tmp_path: Path) -> None:
    blocked, _ = runner(tmp_path, [])
    assert "--web --auto" in blocked.start_run("fix the build")
    gate = threading.Event()
    r, made = runner(tmp_path, [FakeAgent("Opened https://github.com/a/b/pull/3", gate)], allow_runs=True)
    assert r.start_run("   ") == "Describe the goal for the run."
    assert "under" in r.start_run("x" * 5000)
    assert r.start_run("Fix   the\nbuild") is None
    for _ in range(500):
        if r.status()["current"] and r.status()["current"]["log"]:
            break
        time.sleep(0.01)
    status = r.status()["current"]
    assert status["kind"] == "run" and status["goal"] == "Fix the build" and status["log"].endswith(".md")
    assert "already running" in r.start_briefing()
    gate.set()
    wait_idle(r)
    summary = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert summary["status"] == "done" and summary["pull_requests"] == ["https://github.com/a/b/pull/3"]


def test_failures_are_reported_on_the_dashboard(tmp_path: Path) -> None:
    r, _ = runner(tmp_path, [FakeAgent(None, fail=True), FakeAgent("")], allow_runs=True)
    r.start_run("go")
    wait_idle(r)
    assert "RuntimeError: network down" in r.status()["last_error"]
    assert json.loads(next((tmp_path / "runs").glob("*.json")).read_text())["status"] == "failed"
    r.start_briefing()
    wait_idle(r)
    assert "didn't finish" in r.last_error and r.briefing is None  # an empty briefing isn't saved

    def no_token() -> str:
        raise RuntimeError("no GitHub token")

    broken, _ = runner(tmp_path / "b", [], prompt=no_token)
    assert "no GitHub token" in broken.start_briefing()


def test_daily_briefing_is_due_once_a_day_after_its_time(tmp_path: Path) -> None:
    now = [NOON - 4 * 3600]  # 08:00
    r, made = runner(tmp_path, [FakeAgent("Morning."), FakeAgent("Again")], daily_at=(9, 30), clock=lambda: now[0])
    assert not r.due(now[0])
    now[0] = NOON  # 12:00
    assert r.due(now[0])
    r.tick()
    wait_idle(r)
    assert r.briefing["text"] == "Morning." and r.status()["daily_at"] == "09:30"
    assert not r.due(now[0])  # made today
    r.tick()
    assert len(made) == 1
    now[0] = NOON + DAY
    assert r.due(now[0])
    r.briefing = None
    r._daily_tried = time.strftime("%Y-%m-%d", time.localtime(now[0]))
    assert not r.due(now[0])  # tried today already (e.g. it failed): no retry loop
    assert not runner(tmp_path, [])[0].due(NOON)  # no --brief-daily


def test_parse_daily() -> None:
    assert jobs.parse_daily("07:05") == (7, 5)
    for bad in ["7", "24:00", "07:60", "7:5", "noon"]:
        with pytest.raises(ValueError):
            jobs.parse_daily(bad)


def test_cli_checks_brief_daily() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--brief-daily", "08:00", "hi"])  # needs --web
    with pytest.raises(SystemExit):
        cli.main(["--web", "--brief-daily", "8am"])


@pytest.fixture
def served(tmp_path: Path):
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.record("claude-opus-5-5", opus(50_000), "chat")
    gate = threading.Event()
    r, _ = runner(tmp_path, [FakeAgent("done", gate)], allow_runs=True)
    jarvis = Jarvis(ctx=ToolContext(workspace=tmp_path, approve=lambda q: False), client=SimpleNamespace())
    session = web.WebSession(jarvis)
    httpd, token = web.serve(session, "127.0.0.1", 0, dashboard=Dashboard(jarvis, jobs=r, ledger=ledger,
                                                                          runs_dir=tmp_path / "runs"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=httpd.server_address[1], token=token, runner=r, gate=gate)
    gate.set()
    httpd.shutdown()


def call(port: int, method: str, path: str, token: str | None, body: dict | None = None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, json.dumps(body or {}) if method == "POST" else None,
                 {"X-Jarvis-Token": token, "Content-Type": "application/json"} if token else {})
    res = conn.getresponse()
    return res.status, json.loads(res.read() or b"{}")


def test_job_routes_and_snapshot(served) -> None:
    assert call(served.port, "POST", "/api/jobs/run", None, {"goal": "x"})[0] == 401
    status, snap = call(served.port, "GET", "/api/dashboard", served.token)
    assert snap["jobs"]["allow_runs"] is True and snap["jobs"]["current"] is None
    assert len(snap["spend"]["days"]) == 14 and snap["spend"]["days"][-1]["usd"] == pytest.approx(1.0)
    assert call(served.port, "POST", "/api/jobs/run", served.token, {"goal": "Fix CI"}) == (202, {"ok": True})
    status, body = call(served.port, "POST", "/api/jobs/briefing", served.token)
    assert status == 409 and "already running" in body["error"]
    assert call(served.port, "GET", "/api/dashboard", served.token)[1]["jobs"]["current"]["kind"] == "run"
    served.gate.set()
    wait_idle(served.runner)
    runs = call(served.port, "GET", "/api/dashboard", served.token)[1]["runs"]
    assert runs[0]["kind"] == "run" and runs[0]["status"] == "done"


def test_without_jobs_the_routes_are_off(tmp_path: Path) -> None:
    jarvis = Jarvis(ctx=ToolContext(workspace=tmp_path, approve=lambda q: False), client=SimpleNamespace())
    httpd, token = web.serve(web.WebSession(jarvis), "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        assert call(port, "POST", "/api/jobs/briefing", token)[0] == 404
        snap = call(port, "GET", "/api/dashboard", token)[1]
        assert snap["jobs"] is None and snap["spend"] is None
    finally:
        httpd.shutdown()


def test_job_agents_briefing_is_read_only_and_runs_are_autonomous(tmp_path: Path) -> None:
    args = SimpleNamespace(workspace=tmp_path / "repos", no_memory=True, model="claude-opus-5-5", effort="high",
                           team=True, team_models={}, sentinel_mcp="", token_budget=None, max_steps=None, budget=5.0)
    make = cli.make_job_agent(args, None, SpendLedger(tmp_path / "spend.jsonl"))
    log = SimpleNamespace(text=lambda t: None, tool=lambda t: None, result=lambda *a: None)
    brief = make("briefing", log)
    assert brief.ctx.approve("Write file?") is False and brief.ctx.confirm_private("Screenshot?") is False
    assert not brief.autonomous and brief.team is None and brief.spend.kind == "briefing"
    run = make("run", log)
    assert run.ctx.approve("Write file?") is True and run.ctx.confirm_private("Screenshot?") is False
    assert run.autonomous and run.team is not None and run.budget_usd == 5.0 and run.spend.kind == "run"
    assert run.spend is not brief.spend  # each job has its own --budget
