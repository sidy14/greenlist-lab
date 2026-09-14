"""
synthid_detector.py — SynthID-Text statistical detector (v2).

Fixes vs v1:
  * null baseline is 0.5 (not the value returned by
    expected_mean_g_value, which targets a different test).
  * standard error is computed empirically from the per-token g values
    instead of assuming a fixed 0.5 constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from transformers import AutoTokenizer
from transformers.generation import SynthIDTextWatermarkLogitsProcessor


DEFAULT_KEYS = list(range(1, 31))
NULL_MEAN_G = 0.5  # theoretical mean of uniform [0,1] g-values


@dataclass
class SynthIDDetectionResult:
    z_score: float
    p_value: float
    prediction: bool
    num_tokens: int
    mean_g: float
    std_g: float
    null_mean_g: float
    std_of_mean: float
    ngram_len: int
    n_keys: int
    model_id: str
    z_threshold: float = 4.0

    def summary(self) -> str:
        return (
            f"z={self.z_score:+7.3f}  p={self.p_value:.2e}  "
            f"pred={self.prediction}  mean_g={self.mean_g:.4f}  "
            f"std_g={self.std_g:.4f}  tokens={self.num_tokens}"
        )


class SynthIDDetector:
    def __init__(
        self,
        model_id: str = "openai-community/gpt2",
        *,
        ngram_len: int = 5,
        keys: Optional[list[int]] = None,
        sampling_table_size: int = 65536,
        sampling_table_seed: int = 0,
        context_history_size: int = 1024,
        z_threshold: float = 4.0,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.ngram_len = ngram_len
        self.keys = list(keys) if keys is not None else list(DEFAULT_KEYS)
        self.z_threshold = z_threshold
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.processor = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=ngram_len,
            keys=self.keys,
            sampling_table_size=sampling_table_size,
            sampling_table_seed=sampling_table_seed,
            context_history_size=context_history_size,
            device=self.device,
        )

    def detect_from_text(
        self, text: str, *, prompt: Optional[str] = None
    ) -> SynthIDDetectionResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        full = (prompt.rstrip() + " " + text.lstrip()) if prompt else text
        enc = self.tokenizer(
            full,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
        )
        return self.detect_from_ids(enc["input_ids"])

    def detect_from_ids(
        self, input_ids: Union[torch.Tensor, Sequence[int]]
    ) -> SynthIDDetectionResult:
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([list(input_ids)], dtype=torch.long)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            g = self.processor.compute_g_values(input_ids)

        # Reduce to per-token values if the processor returned [batch, seq, keys]
        if g.dim() > 2:
            g = g.float().mean(dim=-1)
        g = g.float()

        # Only positions with a full n-gram context are informative
        scored_start = max(self.ngram_len - 1, 0)
        scored = g[0, scored_start:]
        L = int(scored.numel())
        if L < 2:
            raise ValueError("sequence too short to score")

        mean_g = float(scored.mean().item())
        std_g = float(scored.std(unbiased=True).item()) if L > 1 else 0.0
        std_of_mean = std_g / math.sqrt(L)

        # Guard against degenerate std (all-equal g values)
        if std_of_mean <= 0.0:
            std_of_mean = 1e-9

        z = (mean_g - NULL_MEAN_G) / std_of_mean
        p = 0.5 * math.erfc(z / math.sqrt(2.0))

        return SynthIDDetectionResult(
            z_score=z,
            p_value=p,
            prediction=z > self.z_threshold,
            num_tokens=L,
            mean_g=mean_g,
            std_g=std_g,
            null_mean_g=NULL_MEAN_G,
            std_of_mean=std_of_mean,
            ngram_len=self.ngram_len,
            n_keys=len(self.keys),
            model_id=self.model_id,
            z_threshold=self.z_threshold,
        )
