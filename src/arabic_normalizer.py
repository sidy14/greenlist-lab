"""
arabic_normalizer.py — cleans Arabic-specific surface artifacts.

Controls (each independently toggleable):
    digits          : Arabic-Indic / Eastern-Arabic digits → ASCII
    punctuation     : Arabic punctuation → ASCII equivalents
    alef            : ا variants → bare alef
    ya              : ى ئ → ي
    tashkeel        : drop isolated diacritics
    presentation    : convert presentation forms to base (via NFKC)
    tashkeel_strip  : drop ALL tashkeel (aggressive)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from arabic_surface import (
    TASHKEEL, ARABIC_INDIC, EASTERN_ARABIC, ARABIC_SEPARATORS,
    ARABIC_PUNCT, ALEF_VARIANTS, YA_VARIANTS,
    _is_arabic_letter, _is_presentation_form,
)


@dataclass
class ArabicNormalizeResult:
    original: str
    cleaned: str
    removed_chars: int
    replaced_chars: int
    runs: int

    @property
    def changed(self) -> bool:
        return self.original != self.cleaned

    def summary(self) -> str:
        delta = len(self.original) - len(self.cleaned)
        return (f"removed={self.removed_chars}  replaced={self.replaced_chars}  "
                f"runs={self.runs}  net_delta={delta}")


class ArabicNormalizer:
    def __init__(
        self,
        *,
        digits: bool = True,
        punctuation: bool = True,
        alef: bool = False,           # off by default (changes meaning)
        ya: bool = False,             # off by default (changes meaning)
        tashkeel_isolated: bool = True,
        presentation: bool = True,
        tashkeel_strip: bool = False, # off by default (aggressive)
    ) -> None:
        self.digits = digits
        self.punctuation = punctuation
        self.alef = alef
        self.ya = ya
        self.tashkeel_isolated = tashkeel_isolated
        self.presentation = presentation
        self.tashkeel_strip = tashkeel_strip

    def process(self, text: str) -> ArabicNormalizeResult:
        original = text
        removed = 0
        replaced = 0
        runs = 0

        # Presentation forms (NFKC does the heavy lifting)
        if self.presentation:
            new = "".join(
                unicodedata.normalize("NFKC", c) if _is_presentation_form(c) else c
                for c in text
            )
            if new != text:
                replaced += sum(1 for a, b in zip(text, new) if a != b)
                runs += 1
                text = new

        # Digits
        if self.digits:
            out = []
            hit = False
            for c in text:
                if c in ARABIC_INDIC:
                    out.append(ARABIC_INDIC[c]); hit = True
                elif c in EASTERN_ARABIC:
                    out.append(EASTERN_ARABIC[c]); hit = True
                elif c in ARABIC_SEPARATORS:
                    out.append("." if c == "\u066b" else ","); hit = True
                else:
                    out.append(c)
            if hit:
                replaced += sum(1 for a, b in zip(text, "".join(out)) if a != b)
                runs += 1
                text = "".join(out)

        # Punctuation
        if self.punctuation:
            out = []
            hit = False
            for c in text:
                if c in ARABIC_PUNCT:
                    out.append(ARABIC_PUNCT[c]); hit = True
                else:
                    out.append(c)
            if hit:
                replaced += sum(1 for a, b in zip(text, "".join(out)) if a != b)
                runs += 1
                text = "".join(out)

        # Alef variants (optional)
        if self.alef:
            out = []
            hit = False
            for c in text:
                if c in ALEF_VARIANTS:
                    out.append(ALEF_VARIANTS[c]); hit = True
                else:
                    out.append(c)
            if hit:
                replaced += sum(1 for a, b in zip(text, "".join(out)) if a != b)
                runs += 1
                text = "".join(out)

        # Ya variants (optional)
        if self.ya:
            out = []
            hit = False
            for c in text:
                if c in YA_VARIANTS:
                    out.append(YA_VARIANTS[c]); hit = True
                else:
                    out.append(c)
            if hit:
                replaced += sum(1 for a, b in zip(text, "".join(out)) if a != b)
                runs += 1
                text = "".join(out)

        # Tashkeel
        if self.tashkeel_strip:
            keep = []
            for c in text:
                if c in TASHKEEL:
                    removed += 1
                else:
                    keep.append(c)
            if len(keep) != len(text):
                runs += 1
                text = "".join(keep)

        elif self.tashkeel_isolated:
            out = []
            for i, c in enumerate(text):
                if c in TASHKEEL:
                    prev = text[i-1] if i > 0 else ""
                    if not _is_arabic_letter(prev):
                        removed += 1
                        continue
                out.append(c)
            if len(out) != len(text):
                runs += 1
                text = "".join(out)

        return ArabicNormalizeResult(
            original=original,
            cleaned=text,
            removed_chars=removed,
            replaced_chars=replaced,
            runs=runs,
        )
