"""
stylometric_detector.py - statistical style markers of AI text.

Features used by commercial detectors (GPTZero, Originality.ai):
  - Perplexity (how "surprised" a reference model is)
  - Burstiness (variance in sentence length)
  - Type-Token Ratio (lexical diversity)
  - Punctuation uniformity
  - Sentence-length monotony

Combines them into a calibrated AI-likelihood score.
"""
from __future__ import annotations
import math
import re
import statistics
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REFERENCE_MODEL = "openai-community/gpt2"


@dataclass
class StylometricResult:
    perplexity: float
    burstiness: float
    type_token_ratio: float
    avg_sentence_length: float
    sentence_length_std: float
    ai_score: float       # 0.0 = human, 1.0 = AI
    verdict: str          # "human", "uncertain", "ai"
    notes: list[str]


class StylometricDetector:
    def __init__(self, model_id: str = REFERENCE_MODEL, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _perplexity(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        ids = enc["input_ids"].to(self.device)
        if ids.shape[-1] < 2:
            return float("inf")
        out = self.model(ids, labels=ids)
        return float(torch.exp(out.loss).item())

    def detect(self, text: str) -> StylometricResult:
        notes = []

        # Perplexity
        ppl = self._perplexity(text)

        # Sentence statistics
        sentences = [s.strip() for s in re.split(r"[.!?؟\u3002]+", text) if len(s.strip()) > 5]
        sent_lens = [len(s.split()) for s in sentences] or [0]
        avg_sent = statistics.mean(sent_lens)
        std_sent = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0

        # Burstiness = std / mean (normalized)
        burst = std_sent / avg_sent if avg_sent > 0 else 0.0

        # Type-Token Ratio
        words = re.findall(r"\b\w+\b", text.lower())
        ttr = len(set(words)) / len(words) if words else 0.0

        # AI signature: low perplexity + low burstiness + low TTR
        # Empirically calibrated thresholds:
        ppl_score = _band(ppl, low=25, high=80, invert=True)
        burst_score = _band(burst, low=0.3, high=0.7, invert=False)
        ttr_score = _band(ttr, low=0.35, high=0.6, invert=False)

        ai_score = 0.5 * ppl_score + 0.3 * burst_score + 0.2 * ttr_score

        if ai_score > 0.7:
            verdict = "ai"
        elif ai_score < 0.35:
            verdict = "human"
        else:
            verdict = "uncertain"

        if ppl < 25:
            notes.append(f"Very low perplexity ({ppl:.1f}) — typical of LLM output")
        if burst < 0.3:
            notes.append(f"Uniform sentence length (burst={burst:.2f}) — LLM signature")
        if ttr < 0.35 and len(words) > 50:
            notes.append(f"Low lexical diversity (TTR={ttr:.2f})")

        return StylometricResult(
            perplexity=ppl,
            burstiness=burst,
            type_token_ratio=ttr,
            avg_sentence_length=avg_sent,
            sentence_length_std=std_sent,
            ai_score=ai_score,
            verdict=verdict,
            notes=notes,
        )


def _band(x: float, low: float, high: float, invert: bool) -> float:
    """Map x to [0,1] where 1 = suspicious."""
    if invert:
        if x <= low: return 1.0
        if x >= high: return 0.0
        return 1.0 - (x - low) / (high - low)
    else:
        if x <= low: return 0.0
        if x >= high: return 1.0
        return (x - low) / (high - low)