"""
humanizer.py - reduce AI-detection score of a text.

Ethics: This module is intended for (a) legitimate style improvement,
(b) adversarial robustness research. Do not use it to misrepresent
authorship in academic, legal, or journalistic contexts.

Three levels:
  - light:  break uniformity, vary openers
  - medium: light + paraphrase AI-tagged sentences (needs T5)
  - deep:   full rewrite with a generative model (needs Qwen)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from sentence_analyzer import SentenceAnalyzer


# ---------- connector / opener variations (Arabic) ----------
FORMAL_OPENERS = {
    "يُعدّ": ["يعتبر", "من المتفق عليه أن", "بلا شك،"],
    "من الجدير بالذكر": ["تجدر الإشارة", "ولعل من المفيد", "واللافت أن"],
    "يجدر بالذكر": ["تجدر الإشارة", "ولعل من المفيد"],
    "في الختام": ["وفي نهاية المطاف", "وخلاصة القول", "وأخيراً"],
    "بناءً على ذلك": ["لذا", "ولذلك", "ومن ثم"],
    "بالنسبة لهذا": ["أما عن هذا", "وفيما يتعلق بهذا"],
    "لا يقتصر على": ["لا يقتصر الأمر على", "لا يتوقف عند"],
    "فهو": ["فهو", "إذ هو", "حيث إنه"],
}

SHORT_TRANSITIONS = [
    "لكن", "ورغم ذلك", "ومع ذلك", "وإلا", "وقد", "بل", "ثم", "غير أن"
]


@dataclass
class HumanizeResult:
    original: str
    humanized: str
    edits: list[str]
    level: str
    tokens_changed: int


class Humanizer:
    def __init__(self, analyzer: SentenceAnalyzer | None = None, seed: int = 42):
        self.analyzer = analyzer or SentenceAnalyzer()
        self.rng = random.Random(seed)

    def humanize(self, text: str, level: str = "light") -> HumanizeResult:
        scores = self.analyzer.analyze(text)
        sentences = [s.text for s in scores]
        verdicts = [s.verdict for s in scores]
        edits: list[str] = []

        # -------- step 1: reduce opener monotony --------
        sentences, opener_edits = self._break_openers(sentences)
        edits.extend(opener_edits)

        # -------- step 2: vary sentence length --------
        sentences, len_edits = self._vary_length(sentences)
        edits.extend(len_edits)

        # -------- step 3: inject short transitions --------
        sentences, trans_edits = self._inject_transitions(sentences)
        edits.extend(trans_edits)

        humanized = " ".join(sentences)
        tokens_changed = sum(1 for a, b in zip(text.split(), humanized.split()) if a != b)
        return HumanizeResult(
            original=text,
            humanized=humanized,
            edits=edits,
            level=level,
            tokens_changed=tokens_changed,
        )

    # -------- internal operations --------
    def _break_openers(self, sentences: list[str]) -> tuple[list[str], list[str]]:
        edits = []
        out = []
        seen_openers: dict[str, int] = {}
        for s in sentences:
            for formal, alts in FORMAL_OPENERS.items():
                if s.startswith(formal):
                    # avoid using the same replacement twice in a row
                    alts_filtered = [a for a in alts if a not in seen_openers]
                    if not alts_filtered:
                        alts_filtered = alts
                    choice = self.rng.choice(alts_filtered)
                    new = choice + s[len(formal):]
                    seen_openers[choice] = seen_openers.get(choice, 0) + 1
                    edits.append(f"opener: '{formal}' → '{choice}'")
                    s = new
                    break
            out.append(s)
        return out, edits

    def _vary_length(self, sentences: list[str]) -> tuple[list[str], list[str]]:
        """Split very long sentences at natural pause points."""
        edits = []
        out = []
        for s in sentences:
            # target: 15-25 words per sentence. If longer, insert a break.
            words = s.split()
            if len(words) > 28:
                # find a comma or و in the middle
                for i in range(len(words) // 3, 2 * len(words) // 3):
                    if words[i].rstrip("،,").endswith("و") or "," in words[i]:
                        break
                else:
                    i = len(words) // 2
                first = " ".join(words[:i]).rstrip("،,") + "."
                second = " ".join(words[i:])
                first_cap = first[0].upper() + first[1:] if first else first
                out.append(first_cap)
                out.append(second)
                edits.append(f"split long sentence ({len(words)} → {i}+{len(words)-i} words)")
            else:
                out.append(s)
        return out, edits

    def _inject_transitions(self, sentences: list[str]) -> tuple[list[str], list[str]]:
        """Add short natural transitions at sentence boundaries."""
        edits = []
        out = []
        for i, s in enumerate(sentences):
            if i == 0 or self.rng.random() > 0.35:
                out.append(s)
                continue
            trans = self.rng.choice(SHORT_TRANSITIONS)
            s_new = trans + "، " + s[0].lower() + s[1:]
            edits.append(f"transition: '{trans}'")
            out.append(s_new)
        return out, edits