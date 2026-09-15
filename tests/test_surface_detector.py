"""Tests for surface_detector.SurfaceDetector."""
import pytest

from src.surface_detector import SurfaceDetector


@pytest.fixture
def detector():
    return SurfaceDetector()


def test_clean_text_has_no_findings(detector, clean_text):
    report = detector.scan(clean_text)
    assert report.findings == []
    assert report.counts == {}


def test_zero_width_space_detected(detector):
    text = "hello\u200bworld"
    report = detector.scan(text)
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.category == "A"
    assert f.code == "ZWSP"
    assert f.position == 5


def test_bidi_control_detected(detector):
    text = "safe\u202eevil"
    report = detector.scan(text)
    codes = [f.code for f in report.findings]
    assert "RLO" in codes
    assert report.counts["A"] == 1


def test_tag_character_detected(detector):
    text = "hidden\U000E0041here"
    report = detector.scan(text)
    codes = [f.code for f in report.findings]
    assert "TAG" in codes


def test_homoglyph_detected_in_latin_text(detector):
    text = "This is p\u0430rtly Cyrillic."
    report = detector.scan(text)
    b_findings = [f for f in report.findings if f.category == "B"]
    assert len(b_findings) == 1
    assert b_findings[0].char == "\u0430"


def test_homoglyph_ignored_in_cyrillic_text(detector):
    # Pure Cyrillic should not be reported as homoglyphs.
    text = "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"
    report = detector.scan(text)
    assert report.counts.get("B", 0) == 0


def test_tatweel_run_detected(detector):
    text = "before \u0640\u0640\u0640 after"
    report = detector.scan(text)
    c_findings = [f for f in report.findings if f.category == "C"]
    assert any(f.code == "TATWEEL" for f in c_findings)


def test_nbsp_detected(detector):
    text = "word\u00a0word"
    report = detector.scan(text)
    codes = [f.code for f in report.findings]
    assert "NBSP" in codes


def test_trailing_whitespace_detected(detector):
    text = "line one   \nline two"
    report = detector.scan(text)
    codes = [f.code for f in report.findings]
    assert "TRAIL_WS" in codes


def test_summary_reports_counts(detector, dirty_text):
    report = detector.scan(dirty_text)
    summary = report.summary()
    assert "findings=" in summary
    assert "length=" in summary
