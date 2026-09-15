"""
normalizer.py — cleans surface artifacts from text.

Produces a sanitized version of the input, removing every artifact that
surface_detector.py can find. Does NOT attempt to remove statistical
watermarks (Categories E, F), which are irremovable by surface means.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from surface_detector import (
    ZERO_WIDTH, BIDI, HOMOGLYPHS, NON_STANDARD_SPACES,
)


@dataclass
class NormalizeResult:
    original: str
    cleaned: str
    removed_chars: int
    removed_runs: int

    @property
    def changed(self) -> bool:
        return self.original != self.cleaned

    def summary(self) -> str:
        delta = len(self.original) - len(self.cleaned)
        return (f"chars removed={self.removed_chars}  runs collapsed={self.removed_runs}  "
                f"net delta={delta}")


class Normalizer:
    def __init__(
        self,
        *,
        strip_invisible: bool = True,
        strip_tatweel_runs: bool = True,
        dehomoglyph: bool = True,
        normalize_spaces: bool = True,
        collapse_whitespace: bool = True,
        normalize_quotes: bool = True,
        strip_trailing_ws: bool = True,
    ) -> None:
        self.strip_invisible = strip_invisible
        self.strip_tatweel_runs = strip_tatweel_runs
        self.dehomoglyph = dehomoglyph
        self.normalize_spaces = normalize_spaces
        self.collapse_whitespace = collapse_whitespace
        self.normalize_quotes = normalize_quotes
        self.strip_trailing_ws = strip_trailing_ws

    def process(self, text: str) -> NormalizeResult:
        original = text
        removed = 0
        runs = 0

        if self.strip_invisible:
            keep = []
            for ch in text:
                if ch in ZERO_WIDTH or ch in BIDI:
                    removed += 1
                    continue
                if "\U000E0000" <= ch <= "\U000E007F":
                    removed += 1
                    continue
                cp = ord(ch)
                if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
                    removed += 1
                    continue
                keep.append(ch)
            text = "".join(keep)

        if self.strip_tatweel_runs:
            new = re.sub("\u0640{2,}", "", text)
            if new != text:
                removed += len(text) - len(new)
                runs += 1
                text = new

        if self.dehomoglyph:
            letters = [c for c in text if c.isalpha()]
            if letters:
                latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
                if latin / len(letters) >= 0.85:
                    new = "".join(HOMOGLYPHS.get(c, c) for c in text)
                    if new != text:
                        runs += 1
                        text = new

        if self.normalize_spaces:
            new = "".join(" " if c in NON_STANDARD_SPACES else c for c in text)
            if new != text:
                runs += 1
                text = new

        if self.normalize_quotes:
            table = {
                "\u2018":"'", "\u2019":"'", "\u201a":"'", "\u201b":"'",
                "\u201c":'"', "\u201d":'"', "\u201e":'"', "\u201f":'"',
                "\u00ab":'"', "\u00bb":'"',
                "\u2032":"'", "\u2033":'"',
            }
            new = "".join(table.get(c, c) for c in text)
            if new != text:
                runs += 1
                text = new

        if self.strip_trailing_ws:
            new = re.sub(r"[ \t]+\n", "\n", text)
            if new != text:
                runs += 1
                text = new

        if self.collapse_whitespace:
            new = re.sub(r"\n{3,}", "\n\n", text)
            new = re.sub(r"[ \t]{2,}", " ", new)
            if new != text:
                runs += 1
                text = new

        text = unicodedata.normalize("NFC", text)

        return NormalizeResult(
            original=original,
            cleaned=text,
            removed_chars=removed,
            removed_runs=runs,
        )
