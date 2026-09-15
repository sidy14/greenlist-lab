"""Tests for the 4-layer universal detection system."""
import pytest

from src.vendor_signatures import VendorDetector
from src.arabic_detector import ArabicDetector
from src.universal_analyzer import UniversalAnalyzer


# ----------------------------------------------------------- vendor

def test_o3_nnbsp_detected():
    det = VendorDetector()
    text = "The amount is\u202f100\u202fdollars."
    r = det.scan(text)
    assert "OpenAI-o3" in r.detected_vendors


def test_no_false_positive_on_plain_text():
    det = VendorDetector()
    r = det.scan("This is a normal English sentence.")
    assert len(r.detected_vendors) == 0


def test_llama_template_detected():
    det = VendorDetector()
    text = "<|start_header_id|>user<|end_header_id|>Hello"
    r = det.scan(text)
    assert "Meta-Llama" in r.detected_vendors


# ----------------------------------------------------------- arabic detector

@pytest.fixture
def ar_det():
    return ArabicDetector()


def test_arabic_clean_text(ar_det):
    text = "الشمس تشرق من الشرق والقمر يظهر في الليل."
    r = ar_det.detect(text)
    assert r.is_arabic is True
    assert r.ai_score < 0.5


def test_arabic_ai_phrases(ar_det):
    text = (
        "يجدر بالذكر أن هذا مهم. من الجدير بالذكر أن الفكرة رائعة. "
        "في الختام نؤكد ذلك. تجدر الإشارة إلى أن النتائج ممتازة."
    )
    r = ar_det.detect(text)
    assert r.is_arabic is True
    assert len(r.formal_phrases_found) >= 3
    assert r.ai_score >= 0.6


def test_arabic_not_applicable_for_english(ar_det):
    r = ar_det.detect("This is plain English text.")
    assert r.is_arabic is False


# ----------------------------------------------------------- universal

@pytest.fixture
def analyzer():
    # fast mode: no torch required
    return UniversalAnalyzer(run_stylometric=False)


def test_universal_o3(analyzer):
    text = "The amount is\u202f100\u202fdollars."
    r = analyzer.analyze(text)
    assert r.final_ai_score == 1.0
    assert "AI" in r.final_verdict


def test_universal_arabic_ai(analyzer):
    text = (
        "يجدر بالذكر أن هذا مهم. من الجدير بالذكر أن الفكرة رائعة. "
        "في الختام نؤكد ذلك."
    )
    r = analyzer.analyze(text)
    assert r.final_ai_score >= 0.6
    assert "AI" in r.final_verdict


def test_universal_clean_human(analyzer):
    text = "The cat sat on the mat and looked at the birds outside."
    r = analyzer.analyze(text)
    assert r.final_ai_score <= 0.3
    assert "human" in r.final_verdict


def test_evidence_populated(analyzer):
    text = "The amount is\u202f100\u202fdollars."
    r = analyzer.analyze(text)
    assert len(r.evidence) > 0
    assert any("L1" in e for e in r.evidence)


def test_json_serialisable(analyzer):
    import json
    text = "Some test text."
    r = analyzer.analyze(text)
    data = json.loads(r.to_json())
    assert "final_ai_score" in data
    assert "evidence" in data