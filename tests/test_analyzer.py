"""Tests for analyzer.Analyzer."""
import json

from src.analyzer import Analyzer


def test_clean_text_zero_findings(clean_text):
    report = Analyzer(model_id=None).analyze(clean_text)
    assert report.surface["total_findings"] == 0
    assert report.normalization["changed"] is False


def test_dirty_text_has_findings_and_cleans(dirty_text):
    report = Analyzer(model_id=None).analyze(dirty_text)
    assert report.surface["total_findings"] > 0
    assert report.normalization["changed"] is True
    assert report.cleaned_text != dirty_text


def test_statistical_not_attempted_without_model(clean_text):
    report = Analyzer(model_id=None).analyze(clean_text)
    assert report.statistical["status"] == "not_attempted"


def test_json_serialisation_roundtrip(dirty_text):
    report = Analyzer(model_id=None).analyze(dirty_text)
    raw = report.to_json()
    data = json.loads(raw)
    assert data["text_length"] == len(dirty_text)
    assert data["surface"]["total_findings"] > 0


def test_human_report_has_sections(clean_text):
    report = Analyzer(model_id=None).analyze(clean_text)
    text = report.to_human()
    assert "AI-TEXT-LAB ANALYSIS REPORT" in text
    assert "Surface artifacts" in text
    assert "Statistical watermark" in text
    assert "Normalization" in text
