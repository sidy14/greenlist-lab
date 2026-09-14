"""
synthid_generate_watermarked.py — paired generation with SynthID.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
)


DEFAULT_KEYS = list(range(1, 31))


@dataclass
class SynthIDGenerationPair:
    prompt: str
    clean: str
    watermarked: str
    model_id: str
    ngram_len: int
    n_keys: int
    clean_seed: int
    wm_seed: int


def load_model(model_id: str, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return tok, model


def generate_pair_synthid(
    model_id: str,
    prompt: str,
    *,
    tokenizer=None,
    model=None,
    ngram_len: int = 5,
    keys: list[int] = None,
    sampling_table_size: int = 65536,
    max_new_tokens: int = 220,
    clean_seed: int = 0,
    wm_seed: int = 1,
    device: str = "cpu",
) -> SynthIDGenerationPair:
    if tokenizer is None or model is None:
        tokenizer, model = load_model(model_id, device)

    if keys is None:
        keys = list(DEFAULT_KEYS)

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    torch.manual_seed(clean_seed)
    with torch.no_grad():
        out_c = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
        )
    clean_text = tokenizer.decode(out_c[0, input_len:], skip_special_tokens=True)

    wm_cfg = SynthIDTextWatermarkingConfig(
        ngram_len=ngram_len,
        keys=keys,
        sampling_table_size=sampling_table_size,
        sampling_table_seed=0,
        context_history_size=1024,
    )
    torch.manual_seed(wm_seed)
    with torch.no_grad():
        out_w = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            watermarking_config=wm_cfg,
        )
    wm_text = tokenizer.decode(out_w[0, input_len:], skip_special_tokens=True)

    return SynthIDGenerationPair(
        prompt=prompt,
        clean=clean_text,
        watermarked=wm_text,
        model_id=model_id,
        ngram_len=ngram_len,
        n_keys=len(keys),
        clean_seed=clean_seed,
        wm_seed=wm_seed,
    )
