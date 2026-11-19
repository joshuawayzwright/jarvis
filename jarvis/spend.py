"""A spending ledger: one line per Claude request, so the dashboard can chart cost over days.

Every agent (chat, autonomous runs, briefings, and in team mode every specialist) appends a JSON line to
~/.jarvis/spend.jsonl after each response: when, which model, what kind of work, and the estimated cost at
list price (see pricing.py). Requests to models without a known price are not recorded.

Lines are short and appended with a single write in append mode, so several Jarvis processes can share the
file. Reading is incremental: the ledger remembers how far it has read and parses only new lines, so a
dashboard refreshing every few seconds stays cheap however long the file grows.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import pricing

KINDS = ("chat", "run", "briefing")  # anything else is counted as "chat"


class SpendLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._offset = 0
        self._by_day: dict[str, dict[str, float]] = {}  # "YYYY-MM-DD" (local) -> kind -> USD

    def record(self, model: str, usage: Mapping[str, int], kind: str, when: float | None = None) -> None:
        """Append one request's estimated cost. Never raises: a full disk mustn't stop the agent."""
        cost = pricing.estimate_cost(model, usage)
        if cost is None:
            return
        line = json.dumps({"t": round(when if when is not None else time.time(), 3), "model": model,
                           "kind": kind if kind in KINDS else "chat", "usd": round(cost, 6)})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def _catch_up(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            self._offset, self._by_day = 0, {}
            return
        if size < self._offset:  # truncated or replaced: start over
            self._offset, self._by_day = 0, {}
        if size == self._offset:
            return
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            data = handle.read(size - self._offset)
        end = data.rfind(b"\n") + 1  # a line still being written is read next time
        self._offset += end
        for raw in data[:end].splitlines():
            try:
                entry = json.loads(raw)
                day = time.strftime("%Y-%m-%d", time.localtime(float(entry["t"])))
                usd = float(entry["usd"])
            except (ValueError, KeyError, TypeError):
                continue  # a corrupt or hand-edited line is skipped, not fatal
            kind = entry.get("kind") if entry.get("kind") in KINDS else "chat"
            totals = self._by_day.setdefault(day, {})
            totals[kind] = totals.get(kind, 0.0) + usd

    def daily(self, days: int = 14, now: float | None = None) -> list[dict[str, Any]]:
        """The last `days` days, oldest first, each with its total and a split by kind. Days with no spend are 0."""
        with self._lock:
            self._catch_up()
            now = time.time() if now is None else now
            today = time.localtime(now)
            out = []
            for back in range(days - 1, -1, -1):
                # Step back by calendar days at local noon, so daylight-saving changes can't skip or repeat a day.
                noon = time.mktime((today.tm_year, today.tm_mon, today.tm_mday - back, 12, 0, 0, 0, 0, -1))
                day = time.strftime("%Y-%m-%d", time.localtime(noon))
                kinds = dict(self._by_day.get(day, {}))
                out.append({"date": day, "usd": round(sum(kinds.values()), 4),
                            "by_kind": {k: round(v, 4) for k, v in kinds.items()}})
            return out
