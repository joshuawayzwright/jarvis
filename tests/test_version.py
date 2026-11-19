"""The package version is declared twice; keep them in step."""

from __future__ import annotations

import re
from pathlib import Path

import jarvis


def test_package_and_pyproject_versions_match() -> None:
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert declared is not None
    assert jarvis.__version__ == declared.group(1)
