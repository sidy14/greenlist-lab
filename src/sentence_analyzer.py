"""sentence_analyzer.py - split text into sentences and score each one."""
from __future__ import annotations
import re
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from arabic_ml_detector import ArabicMLDetector, MODEL_DIR


# Arabic/English sentence boundaries
_SENT_SPLIT = re.compile(r"(?<=[.!?؟۔])\s+")


@dataclass
class SentenceScore:
    text: str
    score: float          # 0.0 = human, 1.0 = AI
    verdict: str          # "human", "uncertain", "ai"
    length: int
    too_short: bool = False


class SentenceAnalyzer:
    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cpu"):
        self.device = torch.device(device)
        self._tok = None
        self._model = None
        self._model_dir = model_dir
        self._rules = ArabicMLDetector(model_dir=model_dir, device=device)

    def _ensure_model(self):
        if self._model is not None:
            return True
        try:
            self._tok = AutoTokenizer.from_pretrained(self._model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_dir
            ).to(self.device)
            self._model.eval()
            return True
        except Exception:
            return False

    @staticmethod
    def split(text: str) -> list[str]:
        """Split paragraphs then sentences, keeping order."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        out = []
        for p in paragraphs:
            for s in _SENT_SPLIT.split(p):
                s = s.strip()
                if s:
                    out.append(s)
        return out

    def analyze(self, text: str, min_chars: int = 40) -> list[SentenceScore]:
        sentences = self.split(text)
        if not sentences:
            return []

        # Split into "long enough" and "too short"
        long_idx = []
        results: list[SentenceScore | None] = [None] * len(sentences)
        for i, s in enumerate(sentences):
            if len(s) >= min_chars:
                long_idx.append(i)
            else:
                # short: use rule-based detector
                r = self._rules.detect(s)
                results[i] = SentenceScore(
                    text=s, score=r.ai_score, verdict=r.verdict,
                    length=len(s), too_short=True,
                )

        if long_idx and self._ensure_model():
            batch = [sentences[i] for i in long_idx]
            with torch.no_grad():
                enc = self._tok(
                    batch, truncation=True, padding=True,
                    max_length=256, return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            for i, p in zip(long_idx, probs):
                if p > 0.90:
                    v = "ai"
                elif p > 0.70:
                    v = "likely_ai"
                elif p > 0.40:
                    v = "uncertain"
                elif p > 0.20:
                    v = "likely_human"
                else:
                    v = "human"
                results[i] = SentenceScore(
                    text=sentences[i], score=float(p), verdict=v,
                    length=len(sentences[i]),
                )
        elif long_idx:
            # no model: fall back to rules for long ones too
            for i in long_idx:
                r = self._rules.detect(sentences[i])
                results[i] = SentenceScore(
                    text=sentences[i], score=r.ai_score,
                    verdict=r.verdict, length=len(sentences[i]),
                )
        return [r for r in results if r is not None]


def aggregate(scores: list[SentenceScore]) -> dict:
    if not scores:
        return {"n": 0, "mean": 0.0, "ai_ratio": 0.0, "human_ratio": 0.0}
    import statistics
    mean = statistics.mean(s.score for s in scores)
    n = len(scores)
    ai = sum(1 for s in scores if s.verdict == "ai")
    likely_ai = sum(1 for s in scores if s.verdict == "likely_ai")
    hu = sum(1 for s in scores if s.verdict == "human")
    likely_hu = sum(1 for s in scores if s.verdict == "likely_human")
    uncertain = sum(1 for s in scores if s.verdict == "uncertain")
    return {
        "n": n,
        "mean": mean,
        "ai_ratio": ai / n,
        "likely_ai_ratio": likely_ai / n,
        "human_ratio": hu / n,
        "likely_human_ratio": likely_hu / n,
        "uncertain_ratio": uncertain / n,
    }