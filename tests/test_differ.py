"""Tests for differ module."""
from src.differ import diff_chars, diff_lines, render_full


def test_identical_texts_no_line_changes():
    text = "hello\nworld"
    out = diff_lines(text, text, color=False)
    assert "no line-level changes" in out


def test_identical_texts_no_char_changes():
    text = "hello"
    out = diff_chars(text, text, color=False)
    assert "no character-level changes" in out


def test_removed_char_shown_as_codepoint():
    original = "a\u200bb"
    cleaned = "ab"
    out = diff_chars(original, cleaned, color=False)
    assert "U+200B" in out


def test_replace_arrow_present():
    out = diff_chars("p\u0430rt", "part", color=False)
    assert "->" in out


def test_full_renders_both_sections():
    out = render_full("a\u200bb", "ab", color=False)
    assert "Line-level" in out
    assert "Character-level" in out


def test_no_color_mode_has_no_ansi():
    out = render_full("a\u200bb", "ab", color=False)
    assert "\033[" not in out
