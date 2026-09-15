"""Tests for normalizer.Normalizer."""
import pytest

from src.normalizer import Normalizer


@pytest.fixture
def normalizer():
    return Normalizer()


def test_zwsp_removed(normalizer):
    result = normalizer.process("a\u200bb")
    assert result.cleaned == "ab"
    assert result.removed_chars >= 1


def test_homoglyph_replaced(normalizer):
    # Enough Latin letters to clear the 0.85 dominance threshold.
    text = "This p\u0430rt is mostly latin text."
    result = normalizer.process(text)
    assert "\u0430" not in result.cleaned
    assert "part" in result.cleaned


def test_homoglyph_left_when_text_is_cyrillic(normalizer):
    # Pure Cyrillic — must not be touched.
    text = "\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"
    result = normalizer.process(text)
    assert result.cleaned == text


def test_tatweel_run_collapsed(normalizer):
    # After removing the tatweel run, "x  y" collapses to "x y".
    result = normalizer.process("x \u0640\u0640\u0640 y")
    assert result.cleaned == "x y"


def test_nbsp_becomes_space(normalizer):
    result = normalizer.process("a\u00a0b")
    assert result.cleaned == "a b"


def test_smart_quotes_normalised(normalizer):
    result = normalizer.process("\u2018hi\u2019 \u201cthere\u201d")
    assert result.cleaned == "'hi' \"there\""


def test_trailing_whitespace_stripped(normalizer):
    result = normalizer.process("line   \nnext")
    assert result.cleaned == "line\nnext"


def test_idempotence(normalizer, dirty_text):
    once = normalizer.process(dirty_text).cleaned
    twice = normalizer.process(once).cleaned
    assert once == twice


def test_empty_string(normalizer):
    result = normalizer.process("")
    assert result.cleaned == ""
    assert result.changed is False


def test_clean_text_unchanged(normalizer, clean_text):
    result = normalizer.process(clean_text)
    assert result.changed is False


def test_removed_runs_counts_multiple_categories(normalizer, dirty_text):
    result = normalizer.process(dirty_text)
    assert result.removed_runs >= 3
