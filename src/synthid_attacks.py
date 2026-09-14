"""synthid_attacks.py — T5-PAWS + GPT-2 free rewrite + back-translation."""
from __future__ import annotations
import csv, pathlib, statistics, sys
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from synthid_detector import SynthIDDetector
from synthid_generate_watermarked import generate_pair_synthid, load_model


PROMPTS = [
    "Write a neutral paragraph about the physics of ocean tides.",
    "Explain how a bill becomes law in the United States.",
    "Describe the water cycle in plain language.",
    "Explain what a hash function is to a high-school student.",
]

NGRAM_LEN = 5
SELF_PROMPT = "Rewrite the following paragraph in your own words:\n\n"


def t5_paraphrase(text, tok, model, max_new_tokens=200):
    inputs = tok(["paraphrase: " + text], return_tensors="pt",
                 truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, temperature=0.9,
                             top_p=0.95, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


def self_rewrite(text, tok, model, max_new_tokens=200):
    prompt = SELF_PROMPT + text + "\n\nRewrite:"
    inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, temperature=0.9,
                             top_p=0.95, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def translate(text, tok, model, max_new_tokens=400):
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, num_beams=4, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


def main(model_id="openai-community/gpt2", max_new_tokens=220,
         out_csv="data/reports/synthid_attacks.csv"):
    print(f"loading generator: {model_id}", flush=True)
    gen_tok, gen_model = load_model(model_id)

    print("loading T5-PAWS ...", flush=True)
    t5_tok = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    t5_model.eval()

    print("loading MarianMT EN->FR ...", flush=True)
    en_fr_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    en_fr = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    en_fr.eval()

    print("loading MarianMT FR->EN ...", flush=True)
    fr_en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en.eval()

    det = SynthIDDetector(model_id, ngram_len=NGRAM_LEN)

    rows = []
    for i, prompt in enumerate(PROMPTS):
        pair = generate_pair_synthid(
            model_id, prompt, tokenizer=gen_tok, model=gen_model,
            ngram_len=NGRAM_LEN, max_new_tokens=max_new_tokens,
            clean_seed=700 + i * 2, wm_seed=701 + i * 2,
        )

        z_orig = det.detect_from_text(pair.watermarked).z_score

        t5_out = t5_paraphrase(pair.watermarked, t5_tok, t5_model)
        z_t5 = det.detect_from_text(t5_out).z_score if t5_out.strip() else 0.0

        self_out = self_rewrite(pair.watermarked, gen_tok, gen_model)
        z_self = det.detect_from_text(self_out).z_score if self_out.strip() else 0.0

        fr = translate(pair.watermarked, en_fr_tok, en_fr)
        back = translate(fr, fr_en_tok, fr_en)
        z_back = det.detect_from_text(back).z_score if back.strip() else 0.0

        rows.append({
            "prompt_idx": i,
            "z_original": round(z_orig, 3),
            "z_t5_paraphrase": round(z_t5, 3),
            "z_self_rewrite": round(z_self, 3),
            "z_backtranslation": round(z_back, 3),
        })
        print(f"[{i+1}/{len(PROMPTS)}] orig={z_orig:+6.2f}  t5={z_t5:+6.2f}  "
              f"self={z_self:+6.2f}  back={z_back:+6.2f}", flush=True)

    path = pathlib.Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print()
    print("── SynthID attacks summary ──")
    for key in ("z_original", "z_t5_paraphrase", "z_self_rewrite", "z_backtranslation"):
        vals = [r[key] for r in rows]
        tpr = sum(v > 4 for v in vals) / len(vals)
        print(f"  {key:<22} mean={statistics.mean(vals):+6.2f}  TPR@z>4={tpr:5.0%}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
