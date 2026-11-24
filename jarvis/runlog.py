"""Run logs: a Markdown record of each autonomous run plus a JSON summary the web dashboard lists."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import pricing

if TYPE_CHECKING:
    from .agent import Jarvis


PR_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")


class RunLog:
    """Markdown record of an autonomous run: the goal, every tool call and result, and the outcome.

    Next to each `<stamp>.md` is a `<stamp>.json` summary (goal, status, steps, pull requests, tokens and
    estimated cost) that the web dashboard lists. A second goal in the same session starts a new pair of files.
    """

    def __init__(self, directory: Path, jarvis: Jarvis | None = None, kind: str = "run") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.jarvis = jarvis  # set after the agent exists; supplies the model and token usage
        self.kind = kind  # "run" for autonomous work, "briefing" for a dashboard briefing
        self.summary: dict[str, object] = {}
        self._usage_start: dict[str, int] = {}
        self._open()

    def _open(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path, n = self.directory / f"{stamp}.md", 1
        while path.exists():
            n += 1
            path = self.directory / f"{stamp}-{n}.md"
        self.path = path
        self.summary_path = path.with_suffix(".json")
        self.summary = {"goal": None, "status": "running", "started": time.time(), "updated": time.time(),
                        "kind": self.kind, "steps": 0, "errors": 0, "pull_requests": [], "log": path.name}
        self._write(f"# Jarvis run {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def _save(self) -> None:
        self.summary["updated"] = time.time()
        if self.summary.get("goal") is None:
            return  # an interactive session that never set a goal isn't listed as a run
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        tmp.replace(self.summary_path)

    def _note_pull_requests(self, text: str) -> None:
        found = self.summary["pull_requests"]
        assert isinstance(found, list)
        for url in PR_URL.findall(text):
            if url not in found:
                found.append(url)

    def goal(self, text: str) -> None:
        if self.summary.get("goal") is not None:
            self._open()  # e.g. watch --fix: one run per fix
        self.summary["goal"] = text
        if self.jarvis is not None:
            self.summary["model"] = self.jarvis.model
            self._usage_start = dict(self.jarvis.usage)
            self._usd_start = self.jarvis.cost_usd() if hasattr(self.jarvis, "cost_usd") else None
        self._write(f"## Goal\n\n{text}\n\n## Steps\n")
        self._save()

    def text(self, text: str) -> None:
        self._write(f"\n**Jarvis:** {text}\n")
        self._note_pull_requests(text)
        self._save()

    def tool(self, line: str) -> None:
        self._write(f"- `{line}`")
        self.summary["steps"] = int(self.summary["steps"]) + 1  # type: ignore[call-overload]
        self._save()

    def finish(self, ok: bool, reply: str | None = None, status: str | None = None) -> None:
        """Record how the run ended, with the tokens it used and their estimated cost.

        `status` overrides the outcome, e.g. "cancelled" for a run stopped from the dashboard."""
        self.summary["status"] = status or ("done" if ok else "failed")
        if status is None and self.jarvis is not None and self.jarvis.over_budget():
            self.summary["status"] = "over budget"
        if reply:
            self.summary["reply"] = reply[:500]
            self._note_pull_requests(reply)
        if self.jarvis is not None:
            usage = {k: v - self._usage_start.get(k, 0) for k, v in self.jarvis.usage.items()}
            self.summary["usage"] = usage
            self.summary["cost_usd"] = pricing.estimate_cost(self.jarvis.model, usage)
            # In team mode the specialists share the chief's Spend: count their cost too, not just the chief's.
            start = getattr(self, "_usd_start", None)
            total = self.jarvis.cost_usd() if hasattr(self.jarvis, "cost_usd") else None
            if start is not None and total is not None:
                self.summary["cost_usd"] = round(total - start, 6)
        outcome = {"done": "Finished", "cancelled": "Stopped from the dashboard"}.get(str(self.summary["status"]),
                                                                                      "Stopped with an error")
        self._write(f"\n## Outcome\n\n{outcome}.\n")
        self._save()

    def crew(self, status: str) -> None:
        self.section("Crew", status)

    def section(self, title: str, body: str) -> None:
        self._write(f"\n## {title}\n\n```\n{body}\n```")

    def result(self, name: str, content: object, is_error: bool) -> None:
        if is_error:
            self.summary["errors"] = int(self.summary["errors"]) + 1  # type: ignore[call-overload]
        if not isinstance(content, str):
            content = "[image and text content]"
        else:
            self._note_pull_requests(content)
        self._save()
        head = content.strip().splitlines()[:6]
        marker = "ERROR " if is_error else ""
        self._write("  ```\n  " + marker + "\n  ".join(line[:200] for line in head) + "\n  ```")
