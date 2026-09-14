"""
greenlist_detector.py — compatible with transformers >= 5.0
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union
import torch
from transformers import (
    AutoTokenizer, PreTrainedTokenizerBase,
    WatermarkDetector, WatermarkingConfig,
)


@dataclass
class DetectionResult:
    z_score: float
    p_value: float
    prediction: bool
    num_tokens: int
    num_green: int
    green_fraction: float
    gamma: float
    delta: float
    seeding_scheme: str
    hashing_key: int
    model_id: str
    z_threshold: float = 4.0

    def summary(self) -> str:
        return (
            f"z={self.z_score:+.3f}  p={self.p_value:.2e}  "
            f"pred={self.prediction}  green={self.green_fraction:.4f}  "
            f"tokens={self.num_tokens}"
        )


class GreenListDetector:
    def __init__(
        self,
        model_id: str = "openai-community/gpt2",
        *,
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
        hashing_key: int = 15485863,
        z_threshold: float = 4.0,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width
        self.hashing_key = hashing_key
        self.z_threshold = z_threshold
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.watermarking_config = WatermarkingConfig(
            greenlist_ratio=gamma,
            bias=delta,
            seeding_scheme=seeding_scheme,
            context_width=context_width,
            hashing_key=hashing_key,
        )

        tok = self.tokenizer
        class _Cfg:
            vocab_size = len(tok)
            is_encoder_decoder = False
            bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else 0
            eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._model_config = _Cfg()

        self._detector = WatermarkDetector(
            model_config=self._model_config,
            device=device,
            watermarking_config=self.watermarking_config,
        )

    def detect_from_text(self, text: str, *, prompt: Optional[str] = None) -> DetectionResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        full = (prompt.rstrip() + " " + text.lstrip()) if prompt else text
        enc = self.tokenizer(full, return_tensors="pt",
                             add_special_tokens=False,
                             truncation=True, max_length=4096)
        return self.detect_from_ids(enc["input_ids"])

    def detect_from_ids(self, input_ids: Union[torch.Tensor, Sequence[int]]) -> DetectionResult:
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([list(input_ids)], dtype=torch.long)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        out = self._detector(input_ids.to(self.device), return_dict=True)

        def get(name, default=0.0):
            if hasattr(out, name):
                v = getattr(out, name)
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                if hasattr(v, "item"):
                    v = v.item()
                return v
            return default

        z = float(get("z_score"))
        gf = float(get("green_fraction"))
        pred = bool(get("prediction", False))
        nt = int(get("num_tokens", input_ids.shape[-1]))

        return DetectionResult(
            z_score=z,
            p_value=0.5 * math.erfc(z / math.sqrt(2.0)),
            prediction=pred,
            num_tokens=nt,
            num_green=int(round(gf * nt)),
            green_fraction=gf,
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme,
            hashing_key=self.hashing_key,
            model_id=self.model_id,
            z_threshold=self.z_threshold,
        )
