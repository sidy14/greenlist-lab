"""
surface_detector.py — detects visible/invisible surface artifacts.

Covers Categories A, B, C from our watermark taxonomy:
    A. Hidden characters (ZWSP, ZWJ, bidi controls, tag chars)
    B. Homoglyphs (Cyrillic/Greek/Full-width look-alikes)
    C. Structural fingerprints (tatweel padding, trailing whitespace,
       variation selectors, doubled spaces)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Iterator


# ---- character tables ---------------------------------------------------

ZERO_WIDTH = {
    "\u200b": "ZWSP", "\u200c": "ZWNJ", "\u200d": "ZWJ",
    "\u2060": "WJ", "\ufeff": "BOM", "\u00ad": "SHY",
    "\u180e": "MVS",
}

BIDI = {
    "\u200e": "LRM", "\u200f": "RLM",
    "\u202a": "LRE", "\u202b": "RLE", "\u202c": "PDF",
    "\u202d": "LRO", "\u202e": "RLO",
    "\u2066": "LRI", "\u2067": "RLI", "\u2068": "FSI", "\u2069": "PDI",
    "\u061c": "ALM",
}

HOMOGLYPHS = {
    # Cyrillic
    "а":"a","А":"A","е":"e","Е":"E","о":"o","О":"O","р":"p","Р":"P",
    "с":"c","С":"C","у":"y","У":"Y","х":"x","Х":"X","і":"i","І":"I",
    "ј":"j","Ј":"J","ѕ":"s","Ѕ":"S","ԁ":"d","ɡ":"g",
    # Greek
    "ν":"v","ο":"o","Ο":"O","α":"a","Α":"A","ρ":"p","Ρ":"P",
    "τ":"t","Τ":"T","υ":"u","Υ":"Y","ϲ":"c","Ϲ":"C","ϳ":"j",
    # Full-width ASCII
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},
}

NON_STANDARD_SPACES = {
    "\u00a0":"NBSP", "\u1680":"OGHAM", "\u2000":"EN QUAD",
    "\u2001":"EM QUAD", "\u2002":"EN SPACE", "\u2003":"EM SPACE",
    "\u2004":"3-PER-EM", "\u2005":"4-PER-EM", "\u2006":"6-PER-EM",
    "\u2007":"FIGURE", "\u2008":"PUNCT", "\u2009":"THIN",
    "\u200a":"HAIR", "\u202f":"NNBSP", "\u205f":"MMSP",
    "\u3000":"IDEOGRAPHIC",
}


@dataclass
class Finding:
    category: str      # A | B | C
    code: str          # short label
    position: int
    char: str
    description: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SurfaceReport:
    text_length: int
    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, f: Finding) -> None:
        self.findings.append(f)
        self.counts[f.category] = self.counts.get(f.category, 0) + 1

    def has_issues(self) -> bool:
        return len(self.findings) > 0

    def summary(self) -> str:
        lines = [f"length={self.text_length}  findings={len(self.findings)}"]
        for cat in ("A", "B", "C"):
            n = self.counts.get(cat, 0)
            if n:
                lines.append(f"  category {cat}: {n}")
        return "\n".join(lines)


# ---- main detector ------------------------------------------------------

class SurfaceDetector:
    def scan(self, text: str) -> SurfaceReport:
        report = SurfaceReport(text_length=len(text))

        # Category A — hidden chars
        for i, ch in enumerate(text):
            if ch in ZERO_WIDTH:
                report.add(Finding("A", ZERO_WIDTH[ch], i, ch,
                    f"{ZERO_WIDTH[ch]} (U+{ord(ch):04X})"))
            elif ch in BIDI:
                report.add(Finding("A", BIDI[ch], i, ch,
                    f"{BIDI[ch]} (U+{ord(ch):04X})"))
            elif "\U000E0000" <= ch <= "\U000E007F":
                report.add(Finding("A", "TAG", i, ch,
                    f"Unicode tag U+{ord(ch):04X}"))

        # Category B — homoglyphs (only if text is mostly Latin)
        letters = [c for c in text if c.isalpha()]
        if letters:
            latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
            if latin / len(letters) >= 0.85:
                for i, ch in enumerate(text):
                    if ch in HOMOGLYPHS:
                        report.add(Finding("B", "HOMOGLYPH", i, ch,
                            f"'{ch}' (U+{ord(ch):04X}) looks like '{HOMOGLYPHS[ch]}'"))

        # Category C — structural
        for m in re.finditer("\u0640{2,}", text):
            report.add(Finding("C", "TATWEEL", m.start(), m.group(),
                f"tatweel run length {m.end()-m.start()}"))

        for i, ch in enumerate(text):
            cp = ord(ch)
            if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
                prev = text[i-1] if i else ""
                if prev and prev.isascii() and (prev.isalpha() or prev == " "):
                    report.add(Finding("C", "VAR_SEL", i, ch,
                        f"variation selector U+{cp:04X}"))

        for i, ch in enumerate(text):
            if ch in NON_STANDARD_SPACES:
                report.add(Finding("C", NON_STANDARD_SPACES[ch], i, ch,
                    f"{NON_STANDARD_SPACES[ch]} U+{ord(ch):04X}"))

        pos = 0
        for line in text.split("\n"):
            stripped = line.rstrip(" \t")
            if stripped != line:
                report.add(Finding("C", "TRAIL_WS", pos + len(stripped), "",
                    f"{len(line)-len(stripped)} trailing space(s)"))
            pos += len(line) + 1

        return report
