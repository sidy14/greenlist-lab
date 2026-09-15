"""
arabic_detector.py - Arabic-specific AI text detection (v2).

Uses Arabic-specific markers that commercial detectors miss:
  - Formal connective phrases (يجدر بالذكر، في الختام، ...)
  - Tashkeel density and uniformity
  - Sentence opener diversity
  - Formulaic structure
"""
from __future__ import annotations
import re
import statistics
from dataclasses import dataclass

from arabic_surface import ArabicSurfaceDetector


FORMAL_OPENERS = [
    "\u064a\u062c\u062f\u0631 \u0628\u0627\u0644\u0630\u0643\u0631",
    "\u0645\u0646 \u0627\u0644\u062c\u062f\u064a\u0631 \u0628\u0627\u0644\u0630\u0643\u0631",
    "\u0641\u064a \u0627\u0644\u062e\u062a\u0627\u0645",
    "\u062a\u062c\u062f\u0631 \u0627\u0644\u0625\u0634\u0627\u0631\u0629",
    "\u0645\u0646 \u0627\u0644\u0645\u0647\u0645 \u0645\u0644\u0627\u062d\u0637\u0629",
    "\u0645\u0646 \u0627\u0644\u0645\u0647\u0645 \u0627\u0644\u062a\u0623\u0643\u064a\u062f",
    "\u0628\u0627\u0644\u0646\u0633\u0628\u0629 \u0644\u0647\u0630\u0627",
    "\u0648\u0628\u0627\u0644\u062a\u0627\u0644\u064a",
    "\u0628\u0646\u0627\u0621\u064b \u0639\u0644\u0649 \u0630\u0644\u0643",
    "\u0645\u0645\u0627 \u0644\u0627 \u0634\u0643 \u0641\u064a\u0647",
]


@dataclass
class ArabicAIResult:
    is_arabic: bool
    tashkeel_density: float
    connector_diversity: float
    formal_phrases_found: list[str]
    ai_score: float
    verdict: str


class ArabicDetector:
    def __init__(self) -> None:
        self.arabic_surface = ArabicSurfaceDetector()

    def detect(self, text: str) -> ArabicAIResult:
        ar = self.arabic_surface.scan(text)
        if not ar.is_arabic:
            return ArabicAIResult(False, 0.0, 0.0, [], 0.0, "not-arabic")

        formal_found = [p for p in FORMAL_OPENERS if p in text]

        # Sentence opener diversity
        sentences = [s.strip() for s in re.split(r"[.\u061f!\u060c]+", text)
                     if len(s.strip()) > 5]
        first_words = []
        for s in sentences:
            w = s.split()
            if w:
                first_words.append(w[0])
        diversity = len(set(first_words)) / len(first_words) if first_words else 1.0

        # Cumulative AI score
        score = 0.0

        # Formal phrase count (strongest signal)
        n_formal = len(formal_found)
        if n_formal >= 3:
            score += 0.6
        elif n_formal == 2:
            score += 0.4
        elif n_formal == 1:
            score += 0.2

        # Tashkeel (AI tends to over-diacritize)
        if ar.tashkeel_density > 0.15:
            score += 0.25
        elif ar.tashkeel_density > 0.05:
            score += 0.1

        # Low diversity in sentence openers
        if len(first_words) >= 4 and diversity < 0.5:
            score += 0.15
        elif len(first_words) >= 3 and diversity < 0.7:
            score += 0.1

        # Overly uniform sentence lengths
        if len(sentences) >= 3:
            lens = [len(s.split()) for s in sentences]
            mean_l = statistics.mean(lens)
            std_l = statistics.pstdev(lens) if len(lens) > 1 else 0.0
            burst = std_l / mean_l if mean_l > 0 else 0.0
            if burst < 0.35:
                score += 0.15

        score = min(score, 1.0)

        if score > 0.7:
            verdict = "likely AI-generated"
        elif score < 0.3:
            verdict = "likely human-written"
        else:
            verdict = "uncertain"

        return ArabicAIResult(
            is_arabic=True,
            tashkeel_density=ar.tashkeel_density,
            connector_diversity=diversity,
            formal_phrases_found=formal_found,
            ai_score=score,
            verdict=verdict,
        )