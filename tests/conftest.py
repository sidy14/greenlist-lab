"""Shared pytest fixtures for ai-text-lab tests."""
import os
import pathlib
import subprocess
import sys

import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> pathlib.Path:
    return PROJECT_ROOT


@pytest.fixture
def clean_text() -> str:
    return "This is a simple, clean sentence. Nothing hidden here."


@pytest.fixture
def dirty_text() -> str:
    return (
        "First line is clean.\n"
        "Second\u200b line has a ZWSP.\n"
        "Third h\u0430s a Cyrillic a in mostly latin text here.\n"
        "Fourth\u00a0line uses NBSP.\n"
        "Fifth has tatweel: \u0640\u0640\u0640\u0640.\n"
    )


@pytest.fixture
def run_cli(project_root):
    """Run `python -m src.cli <args>` and return the CompletedProcess."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def _run(*args, input_text: str | None = None):
        cmd = [sys.executable, "-m", "src.cli", *args]
        return subprocess.run(
            cmd,
            cwd=str(project_root),
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    return _run
