"""Stopping dashboard jobs (including a busy team-mode specialist), finish reports, notifications and run costs."""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from jarvis import cli, jobs, web
from jarvis.agent import Jarvis, Spend
from jarvis.dashboard import Dashboard
from jarvis.runlog import RunLog
from jarvis.tools import ToolContext
from test_team import CrewMessages, chief, delegate, tool_turn
from test_voice import FakeStream


class SlowAgent:
    """Works in steps until told to stop, like Jarvis checking should_stop before each request."""

    model = "claude-opus-5-5"

    def __init__(self) -> None:
        self.usage = {"output_tokens": 0}
        self.should_stop = None
        self.stop_reason = "idle"
        self.started = threading.Event()

    def ask(self, goal: str) -> str:
        self.started.set()
        for _ in range(500):
            if self.should_stop and self.should_stop():
                self.stop_reason = "cancelled"
                return "Looked at the build.\n\n[Stopped: cancelled.]"
            self.usage["output_tokens"] += 100
            time.sleep(0.01)
        return "finished"

    def over_budget(self) -> bool:
        return False


def make_runner(tmp_path: Path, agent: object, **kw) -> tuple[jobs.JobRunner, list[dict]]:
    finished: list[dict] = []
    r = jobs.JobRunner(lambda kind, log: agent, tmp_path / "runs", tmp_path / "briefing.json",
                       briefing_prompt=lambda: "brief", allow_runs=True, on_finish=finished.append, **kw)
    return r, finished


def wait_for(check, what: str) -> None:
    for _ in range(500):
        if check():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def test_stop_cancels_the_running_job_and_reports_it(tmp_path: Path) -> None:
    agent = SlowAgent()
    r, finished = make_runner(tmp_path, agent)
    assert r.cancel() == "Nothing is running."
    assert r.start_run("Fix the Android build") is None
    wait_for(agent.started.is_set, "the run to start")
    assert r.cancel() is None
    assert r.status()["current"]["stopping"] is True
    wait_for(lambda: r.current is None, "the run to stop")
    assert agent.stop_reason == "cancelled"
    done = r.status()["last_finished"]
    assert done["status"] == "cancelled" and done["goal"] == "Fix the Android build" and done["log"].endswith(".md")
    assert finished == [done] and r.status()["last_error"] is None  # stopping on request isn't an error
    summary = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert summary["status"] == "cancelled"
    assert "Stopped from the dashboard" in next((tmp_path / "runs").glob("*.md")).read_text()

    agent2 = SlowAgent()  # the next job starts with a fresh stop signal
    r.make_agent = lambda kind, log: agent2
    agent2.ask = lambda goal: "all done"  # type: ignore[method-assign]
    assert r.start_run("next") is None
    wait_for(lambda: r.current is None, "the second run")
    assert r.status()["last_finished"]["status"] == "done"


def test_a_failing_notification_does_not_break_the_runner(tmp_path: Path) -> None:
    def explode(job: dict) -> None:
        raise RuntimeError("webhook down")

    agent = SlowAgent()
    agent.ask = lambda goal: "done"  # type: ignore[method-assign]
    r = jobs.JobRunner(lambda kind, log: agent, tmp_path / "runs", tmp_path / "b.json", briefing_prompt=lambda: "b",
                       allow_runs=True, on_finish=explode)
    r.start_run("go")
    wait_for(lambda: r.current is None, "the run")
    assert r.status()["last_finished"]["status"] == "done" and r.start_run("again") is None


def test_stop_reaches_a_busy_team_specialist(tmp_path: Path) -> None:
    stop = threading.Event()

    class StopsDuringScout(CrewMessages):
        def stream(self, **kwargs: object) -> FakeStream:
            if self.who(str(kwargs["system"])) == "scout":
                stop.set()  # the user presses Stop while Scout is working
            return super().stream(**kwargs)

    fake = StopsDuringScout({
        "chief": [tool_turn(delegate("d1", "scout", "Research rate limiting"))],
        "scout": [tool_turn(("s1", "list_local_repos", {}))],  # Scout would carry on with more steps
    })
    jarvis, _ = chief(tmp_path, fake)
    jarvis.should_stop = stop.is_set
    reply = jarvis.ask("Add rate limiting")
    assert jarvis.stop_reason == "cancelled" and "[Stopped: cancelled.]" in reply
    assert len(fake.requests("scout")) == 1 and len(fake.requests("chief")) == 1  # nobody made another request


def test_run_cost_counts_every_agent_sharing_the_budget(tmp_path: Path) -> None:
    spend = Spend()
    lead = Jarvis(ctx=ToolContext(workspace=tmp_path, approve=lambda q: False), client=SimpleNamespace(), spend=spend)
    helper = Jarvis(ctx=ToolContext(workspace=tmp_path, approve=lambda q: False), client=SimpleNamespace(), spend=spend)
    lead.usage["output_tokens"] = 50_000  # $1.00 before the run
    log = RunLog(tmp_path / "runs", lead)
    log.goal("team run")
    lead.usage["output_tokens"] += 50_000  # $1.00 by the chief
    helper.usage["output_tokens"] += 100_000  # $2.00 by a specialist
    log.finish(True, "done")
    assert json.loads(log.summary_path.read_text())["cost_usd"] == 3.0


def test_job_messages_for_notify() -> None:
    title, body = cli.job_message({"kind": "run", "status": "done", "goal": "Fix   CI", "cost_usd": 1.234,
                                   "pull_requests": ["https://github.com/a/b/pull/9"]})
    assert title == "Jarvis finished a run"
    assert body == "Goal: Fix CI\nPull requests: https://github.com/a/b/pull/9\nEstimated cost: $1.23"
    assert cli.job_message({"kind": "run", "status": "cancelled", "goal": "x"})[0] == "Jarvis run stopped"
    assert cli.job_message({"kind": "briefing", "status": "done", "reply": "All green."}) == (
        "Jarvis: your briefing is ready", "All green.")
    sent: list[tuple[str, str]] = []
    notifier = SimpleNamespace(send=lambda t, b: sent.append((t, b)))
    cli.job_notifier(notifier)({"kind": "run", "status": "failed", "goal": "y"})  # type: ignore[arg-type]
    assert sent == [("Jarvis run failed", "Goal: y")] and cli.job_notifier(None) is None


def test_stop_route(tmp_path: Path) -> None:
    agent = SlowAgent()
    r, _ = make_runner(tmp_path, agent)
    jarvis = Jarvis(ctx=ToolContext(workspace=tmp_path, approve=lambda q: False), client=SimpleNamespace())
    httpd, token = web.serve(web.WebSession(jarvis), "127.0.0.1", 0, dashboard=Dashboard(jarvis, jobs=r))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def post(path: str, tok: str | None = token) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.request("POST", path, "{}", {"X-Jarvis-Token": tok} if tok else {})
        res = conn.getresponse()
        return res.status, json.loads(res.read())

    try:
        assert post("/api/jobs/stop", None)[0] == 401
        assert post("/api/jobs/stop") == (409, {"error": "Nothing is running."})
        r.start_run("go")
        wait_for(agent.started.is_set, "the run")
        assert post("/api/jobs/stop") == (202, {"ok": True})
        wait_for(lambda: r.current is None, "the stop")
    finally:
        httpd.shutdown()
