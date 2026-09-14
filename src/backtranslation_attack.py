"""
backtranslation_attack.py
=========================

The classic aggressive attack: translate EN -> FR -> EN.  Two small
Helsinki-NLP models (~300 MB each) that work on CPU without GPU.

This should be the strongest attack in our matrix: it breaks token
sequences entirely, so it should collapse the z-score to ~0.
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


def translate(text, tok, model, max_new_tokens=400):
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=max_new_tokens,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def main(
    gen_model_id: str = "openai-community/gpt2",
    en_fr_model: str = "Helsinki-NLP/opus-mt-en-fr",
    fr_en_model: str = "Helsinki-NLP/opus-mt-fr-en",
    max_new_tokens: int = 220,
    out_csv: str = "data/reports/backtranslation_attack.csv",
) -> None:
    print(f"loading generator: {gen_model_id}", flush=True)
    gen_tok, gen_model = load_model(gen_model_id)

    print(f"loading translator EN->FR: {en_fr_model}", flush=True)
    en_fr_tok = AutoTokenizer.from_pretrained(en_fr_model)
    en_fr = AutoModelForSeq2SeqLM.from_pretrained(en_fr_model)
    en_fr.eval()

    print(f"loading translator FR->EN: {fr_en_model}", flush=True)
    fr_en_tok = AutoTokenizer.from_pretrained(fr_en_model)
    fr_en = AutoModelForSeq2SeqLM.from_pretrained(fr_en_model)
    fr_en.eval()

    det = GreenListDetector(gen_model_id)

    rows = []
    for i, prompt in enumerate(PROMPTS):
        pair = generate_pair(
            gen_model_id, prompt,
            tokenizer=gen_tok, model=gen_model,
            max_new_tokens=max_new_tokens,
            clean_seed=700 + i * 2,
            wm_seed=701 + i * 2,
        )

        # Original (baseline)
        z_orig = det.detect_from_text(pair.watermarked).z_score

        # EN -> FR
        fr_text = translate(pair.watermarked, en_fr_tok, en_fr)
        # FR -> EN
        back_text = translate(fr_text, fr_en_tok, fr_en)

        z_back = det.detect_from_text(back_text).z_score if back_text.strip() else 0.0

        rows.append({
            "prompt_idx": i,
            "z_original": round(z_orig, 3),
            "z_backtranslation": round(z_back, 3),
            "en_len": len(pair.watermarked),
            "fr_len": len(fr_text),
            "back_len": len(back_text),
        })
        print(f"[{i+1}/{len(PROMPTS)}] orig={z_orig:+6.2f}  back={z_back:+6.2f}  "
              f"lens={len(pair.watermarked)}/{len(fr_text)}/{len(back_text)}", flush=True)

    path = pathlib.Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("── Back-translation attack summary ──")
    for key in ("z_original", "z_backtranslation"):
        vals = [r[key] for r in rows]
        tpr = sum(v > 4 for v in vals) / len(vals)
        print(f"  {key:<22} mean={statistics.mean(vals):+6.2f}  TPR@z>4={tpr:5.0%}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
