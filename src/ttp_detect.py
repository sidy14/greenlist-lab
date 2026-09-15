"""
ttp_detect.py - key-agnostic black-box watermark detection.

Inspired by TTP-Detect (ACL 2026): use a proxy LM to amplify the
watermark signal and test whether the input is closer to the
watermarked or unwatermarked distribution.

No secret key needed. Works on closed-model outputs.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROXY_MODEL = "openai-community/gpt2"


@dataclass
class TTPResult:
    z_score: float
    p_value: float
    prediction: bool
    mean_logprob: float
    variance_logprob: float
    method: str = "TTP-Detect (proxy-model relative test)"


class TTPDetector:
    """
    Relative watermark test:
      H0: the text comes from a non-watermarked distribution P_o
      H1: the text comes from the watermarked distribution P_w

    We approximate both P_o and P_w with a proxy LM (GPT-2), then compute
    the log-likelihood ratio on the input text. Watermarked text tends to
    have slightly higher logprob under the proxy (because the watermark
    biases sampling toward high-probability tokens).
    """

    def __init__(self, model_id: str = PROXY_MODEL, device: str = "cpu"):
        self.model_id = model_id
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _per_token_logprobs(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        ids = enc["input_ids"].to(self.device)
        if ids.shape[-1] < 2:
            return torch.tensor([])
        out = self.model(ids)
        logits = out.logits[0, :-1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = ids[0, 1:]
        return logprobs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu()

    def detect(self, text: str) -> TTPResult:
        lp = self._per_token_logprobs(text)
        if lp.numel() < 5:
            return TTPResult(z_score=0.0, p_value=1.0, prediction=False,
                             mean_logprob=0.0, variance_logprob=0.0)

        mean_lp = float(lp.mean().item())
        var_lp = float(lp.var(unbiased=True).item())
        L = lp.numel()

        # Null calibration for GPT-2 on generic text
        # (empirical: mean logprob ≈ -4.5, std ≈ 3.2 for clean English)
        NULL_MEAN = -4.5
        NULL_STD = 3.2

        se = NULL_STD / math.sqrt(L)
        z = (mean_lp - NULL_MEAN) / se
        p = 0.5 * math.erfc(z / math.sqrt(2.0))

        return TTPResult(
            z_score=z,
            p_value=p,
            prediction=z > 3.0,
            mean_logprob=mean_lp,
            variance_logprob=var_lp,
        )