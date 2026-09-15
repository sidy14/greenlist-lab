"""
arabic_surface.py — Arabic-specific surface detector.

Detects eight categories of Arabic text anomalies that are either
(a) legitimate typographic variation worth normalising, or
(b) suspicious padding used by steganographic/watermarking channels.

    AR1  Diacritics (tashkeel)
    AR2  Arabic-Indic and Eastern-Arabic digits
    AR3  Arabic punctuation (، ؛ ؟ ٪ ۔)
    AR4  Alef variants (ا أ إ آ ٱ)
    AR5  Ya/Ta-Marbuta variants (ي ى ئ ة ه)
    AR6  Presentation forms (U+FB50–U+FEFF, legacy)
    AR7  Isolated diacritics (tashkeel without base letter)
    AR8  Unusual diacritic density (>20% of characters)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict


# ---- character tables ---------------------------------------------------

TASHKEEL = {
    "\u064b": "FATHATAN", "\u064c": "DAMMATAN", "\u064d": "KASRATAN",
    "\u064e": "FATHA", "\u064f": "DAMMA", "\u0650": "KASRA",
    "\u0651": "SHADDA", "\u0652": "SUKUN",
    "\u0653": "MADDAH", "\u0654": "HAMZA ABOVE",
    "\u0655": "HAMZA BELOW", "\u0656": "SUBSCRIPT ALEF",
    "\u0657": "INVERTED DAMMA", "\u0658": "MARK NOON GHUNNA",
    "\u065f": "WAVY HAMZA BELOW",
    "\u0670": "SUPERSCRIPT ALEF",
}

ARABIC_INDIC = {chr(0x0660 + i): str(i) for i in range(10)}
EASTERN_ARABIC = {chr(0x06F0 + i): str(i) for i in range(10)}
ARABIC_SEPARATORS = {
    "\u066b": "ARABIC DECIMAL SEP",    # ٫
    "\u066c": "ARABIC THOUSAND SEP",   # ٬
}

ARABIC_PUNCT = {
    "\u060c": ",",   # ،
    "\u061b": ";",   # ؛
    "\u061f": "?",   # ؟
    "\u066a": "%",   # ٪
    "\u066d": "*",   # ٭
    "\u06d4": ".",   # ۔
}

ALEF_VARIANTS = {
    "\u0623": "\u0627",   # أ → ا
    "\u0625": "\u0627",   # إ → ا
    "\u0622": "\u0627",   # آ → ا
    "\u0671": "\u0627",   # ٱ → ا
}

YA_VARIANTS = {
    "\u0649": "\u064a",   # ى → ي
    "\u0626": "\u064a",   # ئ → ي
}

# Arabic block ranges
ARABIC_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)


def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ARABIC_RANGES)


def _is_arabic_letter(ch: str) -> bool:
    if not _is_arabic_char(ch):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("L")   # Letter


def _is_presentation_form(ch: str) -> bool:
    cp = ord(ch)
    return (0xFB50 <= cp <= 0xFDFF) or (0xFE70 <= cp <= 0xFEFF)


# ---- data classes -------------------------------------------------------

@dataclass
class ArabicFinding:
    category: str      # AR1..AR8
    code: str
    position: int
    char: str
    description: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArabicReport:
    text_length: int
    arabic_char_count: int
    arabic_ratio: float
    findings: list[ArabicFinding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    tashkeel_density: float = 0.0

    def add(self, f: ArabicFinding) -> None:
        self.findings.append(f)
        self.counts[f.category] = self.counts.get(f.category, 0) + 1

    @property
    def is_arabic(self) -> bool:
        # Treat as Arabic if at least 15% of characters are Arabic
        return self.arabic_ratio >= 0.15

    def summary(self) -> str:
        lines = [
            f"length={self.text_length}  "
            f"arabic={self.arabic_char_count} ({self.arabic_ratio:.1%})  "
            f"tashkeel_density={self.tashkeel_density:.2%}"
        ]
        for cat in ("AR1","AR2","AR3","AR4","AR5","AR6","AR7","AR8"):
            n = self.counts.get(cat, 0)
            if n:
                lines.append(f"  {cat}: {n}")
        return "\n".join(lines)


# ---- detector -----------------------------------------------------------

class ArabicSurfaceDetector:
    def __init__(self, *, tashkeel_density_threshold: float = 0.20) -> None:
        self.tashkeel_density_threshold = tashkeel_density_threshold

    def scan(self, text: str) -> ArabicReport:
        arabic_chars = sum(1 for c in text if _is_arabic_char(c))
        ratio = arabic_chars / max(len(text), 1)
        report = ArabicReport(
            text_length=len(text),
            arabic_char_count=arabic_chars,
            arabic_ratio=ratio,
        )

        if not report.is_arabic:
            return report

        tashkeel_count = 0

        for i, ch in enumerate(text):
            # AR1 — diacritics
            if ch in TASHKEEL:
                tashkeel_count += 1
                report.add(ArabicFinding(
                    "AR1", TASHKEEL[ch], i, ch,
                    f"{TASHKEEL[ch]} U+{ord(ch):04X}",
                ))
                # AR7 — isolated diacritic (not adjacent to a letter)
                prev = text[i-1] if i > 0 else ""
                if not _is_arabic_letter(prev):
                    report.add(ArabicFinding(
                        "AR7", "ISOLATED_TASHKEEL", i, ch,
                        f"{TASHKEEL[ch]} not preceded by an Arabic letter",
                    ))
            # AR2 — digits
            elif ch in ARABIC_INDIC:
                report.add(ArabicFinding(
                    "AR2", "ARABIC_INDIC_DIGIT", i, ch,
                    f"digit {ch} (U+{ord(ch):04X}) → {ARABIC_INDIC[ch]}",
                ))
            elif ch in EASTERN_ARABIC:
                report.add(ArabicFinding(
                    "AR2", "EASTERN_ARABIC_DIGIT", i, ch,
                    f"digit {ch} (U+{ord(ch):04X}) → {EASTERN_ARABIC[ch]}",
                ))
            elif ch in ARABIC_SEPARATORS:
                report.add(ArabicFinding(
                    "AR2", "ARABIC_SEPARATOR", i, ch,
                    f"{ARABIC_SEPARATORS[ch]} U+{ord(ch):04X}",
                ))
            # AR3 — Arabic punctuation
            elif ch in ARABIC_PUNCT:
                report.add(ArabicFinding(
                    "AR3", "ARABIC_PUNCT", i, ch,
                    f"'{ch}' → '{ARABIC_PUNCT[ch]}'",
                ))
            # AR4 — alef variants
            elif ch in ALEF_VARIANTS:
                report.add(ArabicFinding(
                    "AR4", "ALEF_VARIANT", i, ch,
                    f"'{ch}' (U+{ord(ch):04X})",
                ))
            # AR5 — ya variants
            elif ch in YA_VARIANTS:
                report.add(ArabicFinding(
                    "AR5", "YA_VARIANT", i, ch,
                    f"'{ch}' (U+{ord(ch):04X})",
                ))
            # AR6 — presentation forms
            elif _is_presentation_form(ch):
                report.add(ArabicFinding(
                    "AR6", "PRESENTATION_FORM", i, ch,
                    f"presentation form U+{ord(ch):04X}",
                ))

        report.tashkeel_density = (
            tashkeel_count / max(len(text), 1)
        )

        # AR8 — abnormal tashkeel density
        if report.tashkeel_density > self.tashkeel_density_threshold:
            report.add(ArabicFinding(
                "AR8", "HIGH_TASHKEEL_DENSITY", 0, "",
                f"tashkeel density {report.tashkeel_density:.1%} "
                f"exceeds threshold {self.tashkeel_density_threshold:.0%}",
            ))

        return report
