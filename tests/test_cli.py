"""End-to-end tests for the CLI, invoked via subprocess."""
import json
import pathlib

import pytest


def test_help_returns_zero(run_cli):
    res = run_cli("--help")
    assert res.returncode == 0
    assert "analyze" in res.stdout
    assert "clean" in res.stdout
    assert "diff" in res.stdout


def test_analyze_clean_text(run_cli, tmp_path: pathlib.Path):
    f = tmp_path / "clean.txt"
    f.write_text("A simple clean sentence.", encoding="utf-8")
    res = run_cli("analyze", str(f))
    assert res.returncode == 0
    assert "ANALYSIS REPORT" in res.stdout
    assert "none detected" in res.stdout


def test_analyze_dirty_text_finds_artifacts(run_cli, tmp_path: pathlib.Path):
    f = tmp_path / "dirty.txt"
    f.write_text("hello\u200bworld", encoding="utf-8")
    res = run_cli("analyze", str(f))
    assert res.returncode == 0
    assert "hidden chars" in res.stdout


def test_analyze_json_mode(run_cli, tmp_path: pathlib.Path):
    f = tmp_path / "x.txt"
    f.write_text("a\u200bb", encoding="utf-8")
    res = run_cli("analyze", str(f), "--json")
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["surface"]["total_findings"] >= 1


def test_clean_subcommand_writes_file(run_cli, tmp_path: pathlib.Path):
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    src.write_text("a\u200bb", encoding="utf-8")
    res = run_cli("clean", str(src), "-o", str(dst))
    assert res.returncode == 0
    assert dst.read_text(encoding="utf-8") == "ab"


def test_diff_subcommand(run_cli, tmp_path: pathlib.Path):
    f = tmp_path / "x.txt"
    f.write_text("p\u0430rt", encoding="utf-8")
    res = run_cli("diff", str(f), "--no-color")
    assert res.returncode == 0
    assert "Line-level" in res.stdout or "Character-level" in res.stdout


def test_stdin_input(run_cli):
    res = run_cli("analyze", "-", input_text="a\u200bb\n")
    assert res.returncode == 0
    assert "hidden chars" in res.stdout
