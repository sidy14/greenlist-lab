"""
generate_watermarked.py — paired generation with different seeds.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig,
)


@dataclass
class GenerationPair:
    prompt: str
    clean: str
    watermarked: str
    model_id: str
    gamma: float
    delta: float
    seeding_scheme: str
    hashing_key: int
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


def generate_pair(
    model_id: str,
    prompt: str,
    *,
    tokenizer=None,
    model=None,
    gamma: float = 0.25,
    delta: float = 2.0,
    seeding_scheme: str = "lefthash",
    hashing_key: int = 15485863,
    max_new_tokens: int = 220,
    clean_seed: int = 0,
    wm_seed: int = 1,
    device: str = "cpu",
) -> GenerationPair:
    if tokenizer is None or model is None:
        tokenizer, model = load_model(model_id, device)

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    torch.manual_seed(clean_seed)
    with torch.no_grad():
        out_c = model.generate(
            **inputs, do_sample=True, temperature=0.9, top_p=0.95,
            max_new_tokens=max_new_tokens,
        )
    clean_text = tokenizer.decode(out_c[0, input_len:], skip_special_tokens=True)

    wm_cfg = WatermarkingConfig(
        greenlist_ratio=gamma,
        bias=delta,
        seeding_scheme=seeding_scheme,
        context_width=1,
        hashing_key=hashing_key,
    )
    torch.manual_seed(wm_seed)
    with torch.no_grad():
        out_w = model.generate(
            **inputs, do_sample=True, temperature=0.9, top_p=0.95,
            max_new_tokens=max_new_tokens,
            watermarking_config=wm_cfg,
        )
    wm_text = tokenizer.decode(out_w[0, input_len:], skip_special_tokens=True)

    return GenerationPair(
        prompt=prompt, clean=clean_text, watermarked=wm_text,
        model_id=model_id, gamma=gamma, delta=delta,
        seeding_scheme=seeding_scheme, hashing_key=hashing_key,
        clean_seed=clean_seed, wm_seed=wm_seed,
    )
