"""The codespace start script: it's wired into the devcontainer, and its no-start paths explain what to do."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".devcontainer" / "start-jarvis.sh"


def run(env: dict[str, str]) -> str:
    base = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "JARVIS_AUTOSTART", "JARVIS_AUTOSTART_ARGS")}
    result = subprocess.run(["bash", str(SCRIPT)], env={**base, **env}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_devcontainer_runs_the_start_script_on_attach() -> None:
    config = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text())
    assert config["postAttachCommand"] == "bash .devcontainer/start-jarvis.sh"
    assert 8765 in config["forwardPorts"]
    assert "ANTHROPIC_API_KEY" in config["secrets"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_script_can_be_turned_off_and_explains_a_missing_key(tmp_path: Path) -> None:
    off = run({"JARVIS_AUTOSTART": "0", "ANTHROPIC_API_KEY": "x", "JARVIS_HOME": str(tmp_path)})
    assert "auto-start is off" in off and "jarvis --web --watch --auto --budget 5" in off
    no_key = run({"JARVIS_HOME": str(tmp_path)})
    assert "needs your Anthropic API key" in no_key and "github.com/settings/codespaces" in no_key
