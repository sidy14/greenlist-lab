"""arabic_ml_detector.py - ML-based Arabic AI-text detector."""
from __future__ import annotations
import pathlib
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabic_detector import ArabicDetector

# Try local model first (fast on your machine).
# Falls back to HuggingFace Hub when the local copy is missing
# (e.g. when running on Streamlit Community Cloud).
HF_REPO = "sidy14/arabic-ai-detector"
LOCAL_DIR = "models/arabic-ai-detector"

import pathlib as _pl
if _pl.Path(LOCAL_DIR).exists():
    MODEL_DIR = LOCAL_DIR
else:
    MODEL_DIR = HF_REPO


@dataclass
class ArabicMLResult:
    is_arabic: bool
    ai_score: float
    verdict: str
    method: str
    rule_score: float


class ArabicMLDetector:
    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cpu"):
        self.model_dir = pathlib.Path(model_dir)
        self.device = torch.device(device)
        self._model = None
        self._tokenizer = None
        self._rules = ArabicDetector()

    def _ensure_loaded(self):
        if self._model is not None:
            return True
        if not self.model_dir.exists():
            return False
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir)
            ).to(self.device)
            self._model.eval()
            return True
        except Exception:
            return False

    def detect(self, text: str) -> ArabicMLResult:
        rule_result = self._rules.detect(text)
        if not rule_result.is_arabic:
            return ArabicMLResult(False, 0.0, "not-arabic", "rules",
                                  rule_result.ai_score)

        if not self._ensure_loaded():
            return ArabicMLResult(True, rule_result.ai_score,
                                  rule_result.verdict, "rules",
                                  rule_result.ai_score)

        with torch.no_grad():
            enc = self._tokenizer(
                text, truncation=True, padding="max_length",
                max_length=256, return_tensors="pt",
            ).to(self.device)
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            ai_prob = float(probs[0, 1].item())

        if ai_prob > 0.7:
            verdict = "likely AI-generated"
        elif ai_prob < 0.3:
            verdict = "likely human-written"
        else:
            verdict = "uncertain"

        return ArabicMLResult(True, ai_prob, verdict, "ml",
                              rule_result.ai_score)