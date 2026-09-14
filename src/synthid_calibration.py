"""
synthid_calibration.py — 8-prompt calibration for SynthID-Text.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from synthid_detector import SynthIDDetector
from synthid_generate_watermarked import generate_pair_synthid, load_model


PROMPTS = [
    "Write a neutral paragraph about the physics of ocean tides.",
    "Explain how a bill becomes law in the United States.",
    "Describe the water cycle in plain language.",
    "Summarise the main causes of the 2008 financial crisis.",
    "Explain what a hash function is to a high-school student.",
    "Describe the structure of a plant cell.",
    "Summarise the plot of Hamlet in three sentences.",
    "Explain why the sky is blue.",
]

MIN_CHARS = 80
NGRAM_LEN = 5


def generate_with_retry(model_id, prompt, tok, model, *, max_new_tokens,
                        base_seed, max_attempts=4):
    for attempt in range(max_attempts):
        seed = base_seed + attempt * 1000
        pair = generate_pair_synthid(
            model_id, prompt, tokenizer=tok, model=model,
            ngram_len=NGRAM_LEN,
            max_new_tokens=max_new_tokens,
            clean_seed=seed, wm_seed=seed + 1,
        )
        ok_c = len(pair.clean.strip()) >= MIN_CHARS
        ok_w = len(pair.watermarked.strip()) >= MIN_CHARS
        if ok_c and ok_w:
            return pair, attempt
        print(f"    retry {attempt+1}: clean_len={len(pair.clean)} wm_len={len(pair.watermarked)}", flush=True)
    return pair, max_attempts


def main(model_id="openai-community/gpt2", max_new_tokens=220,
         out_csv="data/reports/synthid_calibration.csv"):
    print(f"loading {model_id} ...", flush=True)
    tok, model = load_model(model_id)
    det = SynthIDDetector(model_id, ngram_len=NGRAM_LEN)

    rows = []
    failures = 0
    for i, prompt in enumerate(PROMPTS):
        pair, attempts = generate_with_retry(
            model_id, prompt, tok, model,
            max_new_tokens=max_new_tokens,
            base_seed=100 + i * 2,
        )
        rc = det.detect_from_text(pair.clean)
        rw = det.detect_from_text(pair.watermarked)
        detected = rw.z_score > 4.0
        if not detected:
            failures += 1
        rows.append({
            "prompt": prompt,
            "z_clean": round(rc.z_score, 3),
            "z_wm": round(rw.z_score, 3),
            "mean_g_clean": round(rc.mean_g, 4),
            "mean_g_wm": round(rw.mean_g, 4),
            "tokens_clean": rc.num_tokens,
            "tokens_wm": rw.num_tokens,
            "attempts": attempts + 1,
            "detected": detected,
        })
        flag = "OK " if detected else "MISS"
        print(f"[{i+1}/{len(PROMPTS)}] {flag}  z_clean={rc.z_score:+7.2f}  z_wm={rw.z_score:+7.2f}  attempts={attempts+1}", flush=True)

    path = pathlib.Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    zc = [r["z_clean"] for r in rows]
    zw = [r["z_wm"] for r in rows]
    print()
    print("── SynthID calibration summary ──")
    print(f"n              = {len(rows)}")
    print(f"z_clean: mean={statistics.mean(zc):+7.2f}  std={statistics.pstdev(zc):5.2f}  max={max(zc):+7.2f}")
    print(f"z_wm   : mean={statistics.mean(zw):+7.2f}  std={statistics.pstdev(zw):5.2f}  min={min(zw):+7.2f}")
    print(f"separation     = {statistics.mean(zw) - statistics.mean(zc):.2f}")
    print(f"TPR@z>4        = {sum(z > 4 for z in zw)/len(zw):.0%}   ({len(zw)-failures}/{len(zw)})")
    print(f"FPR@z>4        = {sum(z > 4 for z in zc)/len(zc):.0%}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
