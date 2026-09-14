"""
paraphrase_attack.py
====================

Two semantic attacks on the green-list watermark:

1. self_paraphrase   — ask GPT-2 to rewrite its own watermarked output
2. cross_paraphrase  — ask T5-small to rewrite it (cross-model)

Expected: z collapses from ~10 to ~0-2, confirming that the watermark
is vulnerable to semantic rewriting — the only effective attack.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from greenlist_detector import GreenListDetector
from generate_watermarked import generate_pair, load_model


PROMPTS = [
    "Write a neutral paragraph about the physics of ocean tides.",
    "Explain how a bill becomes law in the United States.",
    "Describe the water cycle in plain language.",
    "Explain what a hash function is to a high-school student.",
]

SELF_PROMPT = "Rewrite the following paragraph in your own words:\n\n"


def self_paraphrase(text: str, tok, model, max_new_tokens: int = 200) -> str:
    prompt = SELF_PROMPT + text + "\n\nRewrite:"
    inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def cross_paraphrase(text: str, tok, model, max_new_tokens: int = 200) -> str:
    inputs = tok(
        ["paraphrase: " + text],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def main(
    gen_model_id: str = "openai-community/gpt2",
    para_model_id: str = "Vamsi/T5_Paraphrase_Paws",
    max_new_tokens: int = 220,
    out_csv: str = "data/reports/paraphrase_attack.csv",
) -> None:
    print(f"loading generator: {gen_model_id}", flush=True)
    gen_tok, gen_model = load_model(gen_model_id)

    print(f"loading paraphraser: {para_model_id}", flush=True)
    para_tok = AutoTokenizer.from_pretrained(para_model_id)
    para_model = AutoModelForSeq2SeqLM.from_pretrained(para_model_id)
    para_model.eval()

    det = GreenListDetector(gen_model_id)

    rows = []
    for i, prompt in enumerate(PROMPTS):
        pair = generate_pair(
            gen_model_id, prompt,
            tokenizer=gen_tok, model=gen_model,
            max_new_tokens=max_new_tokens,
            clean_seed=500 + i * 2,
            wm_seed=501 + i * 2,
        )

        wm_self = self_paraphrase(pair.watermarked, gen_tok, gen_model)
        wm_cross = cross_paraphrase(pair.watermarked, para_tok, para_model)

        z_orig = det.detect_from_text(pair.watermarked).z_score
        z_self = det.detect_from_text(wm_self).z_score if wm_self.strip() else 0.0
        z_cross = det.detect_from_text(wm_cross).z_score if wm_cross.strip() else 0.0

        rows.append({
            "prompt_idx": i,
            "z_original": round(z_orig, 3),
            "z_self_paraphrase": round(z_self, 3),
            "z_cross_paraphrase": round(z_cross, 3),
        })
        print(f"[{i+1}/{len(PROMPTS)}] orig={z_orig:+6.2f}  self={z_self:+6.2f}  cross={z_cross:+6.2f}", flush=True)

    path = pathlib.Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("── Paraphrase attack summary ──")
    for key in ("z_original", "z_self_paraphrase", "z_cross_paraphrase"):
        vals = [r[key] for r in rows]
        tpr = sum(v > 4 for v in vals) / len(vals)
        print(f"  {key:<22} mean={statistics.mean(vals):+6.2f}  TPR@z>4={tpr:5.0%}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
