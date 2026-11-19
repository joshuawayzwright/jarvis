"""Background jobs started from the web dashboard: autonomous runs and the daily briefing.

Each job gets its own agent, so the chat stays free while it works, and its own run log, so it shows in
the dashboard's run list with a live log. One job runs at a time.

- A **run** works through a goal autonomously (research, build, test, draft PR), like `jarvis --auto`.
  Starting one from the dashboard is allowed only when the web server itself was started with --auto:
  the person who started Jarvis decides whether the dashboard may act without asking.
- A **briefing** is read-only: its agent declines every write, command, commit and push. It answers the
  same question as `jarvis --brief` and is kept in ~/.jarvis/briefing.json, so it survives a restart.
  With --brief-daily HH:MM a new one is made each day at that local time.

Every job counts toward --budget on its own: each gets the full limit, not a share of the chat's.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .runlog import RunLog

MAX_GOAL_CHARS = 4_000
MAX_BRIEFING_CHARS = 20_000

MakeAgent = Callable[[str, RunLog], Any]  # (kind, log) -> a Jarvis wired to that log


def parse_daily(text: str) -> tuple[int, int]:
    """'08:30' -> (8, 30). Raises ValueError for anything else."""
    hours, _, minutes = text.strip().partition(":")
    h, m = int(hours), int(minutes)
    if not (0 <= h < 24 and 0 <= m < 60) or len(minutes) != 2:
        raise ValueError(f"expected HH:MM, got {text!r}")
    return h, m


class JobRunner:
    def __init__(self, make_agent: MakeAgent, runs_dir: Path, state_path: Path,
                 briefing_prompt: Callable[[], str], allow_runs: bool = False,
                 daily_at: tuple[int, int] | None = None, clock: Callable[[], float] = time.time) -> None:
        self.make_agent = make_agent
        self.runs_dir = runs_dir
        self.state_path = state_path
        self.briefing_prompt = briefing_prompt
        self.allow_runs = allow_runs
        self.daily_at = daily_at
        self.clock = clock
        self._lock = threading.Lock()
        self.current: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.briefing: dict[str, Any] | None = self._load_briefing()
        self._daily_tried: str | None = None  # the day an automatic briefing was last attempted
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # Briefing state

    def _load_briefing(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and isinstance(data.get("text"), str) else None

    def _save_briefing(self, briefing: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(briefing, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    # Starting jobs

    def start_run(self, goal: str) -> str | None:
        """Start an autonomous run. Returns an error message, or None when it started."""
        goal = " ".join(goal.split())
        if not self.allow_runs:
            return "Starting runs from the dashboard needs Jarvis started with --web --auto."
        if not goal:
            return "Describe the goal for the run."
        if len(goal) > MAX_GOAL_CHARS:
            return f"Keep the goal under {MAX_GOAL_CHARS:,} characters."
        return self._launch("run", goal)

    def start_briefing(self, trigger: str = "manual") -> str | None:
        """Start a briefing. Returns an error message, or None when it started."""
        try:
            prompt = self.briefing_prompt()
        except Exception as err:  # noqa: BLE001 - e.g. no GitHub token to find the account
            return f"Couldn't prepare the briefing: {err}"
        return self._launch("briefing", prompt, trigger=trigger)

    def _launch(self, kind: str, goal: str, trigger: str = "manual") -> str | None:
        with self._lock:
            if self.current is not None:
                return f"A {self.current['kind']} is already running; it has to finish first."
            self.current = {"kind": kind, "goal": goal, "started": self.clock(), "trigger": trigger, "log": None}
            self.last_error = None
        thread = threading.Thread(target=self._work, args=(kind, goal), name=f"jarvis-{kind}", daemon=True)
        thread.start()
        self._threads.append(thread)
        return None

    def _work(self, kind: str, goal: str) -> None:
        reply: str | None = None
        log: RunLog | None = None
        try:
            log = RunLog(self.runs_dir, kind=kind)
            agent = self.make_agent(kind, log)
            log.jarvis = agent
            log.goal(goal)
            with self._lock:
                if self.current is not None:
                    self.current["log"] = log.path.name
            reply = agent.ask(goal)
            ok = bool(reply) and not agent.over_budget()
            log.finish(ok, reply)
            if kind == "briefing" and ok:
                briefing = {"text": (reply or "")[:MAX_BRIEFING_CHARS], "generated_at": self.clock(),
                            "cost_usd": log.summary.get("cost_usd"), "log": log.path.name}
                self._save_briefing(briefing)
                self.briefing = briefing
            elif not ok:
                self.last_error = f"The {kind} didn't finish. See its log ({log.path.name})."
        except Exception as err:  # noqa: BLE001 - report it on the dashboard rather than lose the thread
            self.last_error = f"The {kind} stopped: {type(err).__name__}: {err}"
            if log is not None:
                log.finish(False, None)
        finally:
            with self._lock:
                self.current = None

    # Daily briefing

    def due(self, now: float) -> bool:
        """True when today's automatic briefing should start: past the time, none made today, none tried today."""
        if self.daily_at is None:
            return False
        local = time.localtime(now)
        today = time.strftime("%Y-%m-%d", local)
        if (local.tm_hour, local.tm_min) < self.daily_at or self._daily_tried == today:
            return False
        made = self.briefing.get("generated_at") if self.briefing else None
        return not (isinstance(made, (int, float)) and time.strftime("%Y-%m-%d", time.localtime(made)) == today)

    def tick(self) -> None:
        now = self.clock()
        if self.current is None and self.due(now):
            self._daily_tried = time.strftime("%Y-%m-%d", time.localtime(now))  # one attempt a day, even if it fails
            self.start_briefing(trigger="daily")

    def start(self, every_s: float = 60) -> None:
        if self.daily_at is None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(every_s)

        threading.Thread(target=loop, name="jarvis-daily-brief", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    # Snapshot

    def status(self) -> dict[str, Any]:
        with self._lock:
            current = dict(self.current) if self.current else None
        return {
            "allow_runs": self.allow_runs,
            "current": current,
            "last_error": self.last_error,
            "briefing": dict(self.briefing) if self.briefing else None,
            "daily_at": f"{self.daily_at[0]:02d}:{self.daily_at[1]:02d}" if self.daily_at else None,
        }
