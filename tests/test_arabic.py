"""Tests for the Arabic module."""
import pytest

from src.arabic_surface import ArabicSurfaceDetector
from src.arabic_normalizer import ArabicNormalizer


@pytest.fixture
def detector():
    return ArabicSurfaceDetector()


@pytest.fixture
def normalizer():
    return ArabicNormalizer()


CLEAN_AR = "مرحبا بكم في هذا الاختبار"


def test_clean_arabic_no_findings(detector):
    report = detector.scan(CLEAN_AR)
    assert report.is_arabic is True
    assert report.findings == []


def test_latin_text_not_arabic(detector):
    report = detector.scan("this is plain english")
    assert report.is_arabic is False


def test_arabic_indic_digits_detected(detector):
    report = detector.scan("العدد ١٢٣")
    codes = [f.code for f in report.findings]
    assert "ARABIC_INDIC_DIGIT" in codes
    assert report.counts["AR2"] == 3


def test_eastern_arabic_digits_detected(detector):
    report = detector.scan("عدد ۱۲۳ فارسی")
    codes = [f.code for f in report.findings]
    assert "EASTERN_ARABIC_DIGIT" in codes


def test_arabic_punctuation_detected(detector):
    report = detector.scan("كلمة، وأخرى؟")
    codes = [f.code for f in report.findings]
    assert "ARABIC_PUNCT" in codes


def test_alef_variant_detected(detector):
    report = detector.scan("كتاب أحمد")
    codes = [f.code for f in report.findings]
    assert "ALEF_VARIANT" in codes


def test_presentation_form_detected(detector):
    # U+FEBD = ARABIC LETTER HAMZA ISOLATED FORM ... use MEDIAL FORM of MEEM
    text = "اﻟﻌرﺑﻳﺔ"  # contains presentation forms
    report = detector.scan(text)
    codes = [f.code for f in report.findings]
    assert "PRESENTATION_FORM" in codes


def test_digits_normalized(normalizer):
    r = normalizer.process("١٢٣")
    assert r.cleaned == "123"


def test_eastern_digits_normalized(normalizer):
    r = normalizer.process("۱۲۳")
    assert r.cleaned == "123"


def test_arabic_punctuation_normalized(normalizer):
    r = normalizer.process("أ،ب؟ج؛")
    assert r.cleaned == "أ,ب?ج;"


def test_alef_normalization_optional_off(normalizer):
    # default: alef variants preserved
    r = normalizer.process("أحمد")
    assert "أ" in r.cleaned


def test_alef_normalization_on():
    n = ArabicNormalizer(alef=True)
    r = n.process("أحمد")
    assert r.cleaned.startswith("ا")


def test_tashkeel_strip_aggressive():
    n = ArabicNormalizer(tashkeel_strip=True)
    r = n.process("مَرْحَبًا")
    assert "\u064e" not in r.cleaned
    assert "\u064b" not in r.cleaned


def test_idempotence_normalize(detector, normalizer):
    text = "عدد ١٢٣، وكلمة"
    once = normalizer.process(text).cleaned
    twice = normalizer.process(once).cleaned
    assert once == twice


def test_clean_arabic_unchanged_by_default(normalizer):
    r = normalizer.process(CLEAN_AR)
    assert r.changed is False
