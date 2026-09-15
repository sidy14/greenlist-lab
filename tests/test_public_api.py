"""Tests for the public ai_text_lab package API."""
import ai_text_lab


def test_public_exports_present():
    required = {"analyze", "clean", "diff",
                "Analyzer", "SurfaceDetector", "Normalizer"}
    assert required.issubset(set(ai_text_lab.__all__))


def test_analyze_returns_report(clean_text):
    report = ai_text_lab.analyze(clean_text)
    assert hasattr(report, "to_human")
    assert report.surface["total_findings"] == 0


def test_clean_returns_string(dirty_text):
    cleaned = ai_text_lab.clean(dirty_text)
    assert isinstance(cleaned, str)
    assert "\u200b" not in cleaned
    assert "\u00a0" not in cleaned


def test_diff_default_mode(dirty_text):
    out = ai_text_lab.diff(dirty_text)
    assert isinstance(out, str)
    assert len(out) > 0
    assert "Line-level" in out


def test_diff_chars_mode(dirty_text):
    out = ai_text_lab.diff(dirty_text, mode="chars")
    assert "U+200B" in out or "->" in out
