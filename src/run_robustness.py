"""
run_robustness.py — robustness matrix.
"""
from __future__ import annotations
import csv, pathlib, random, re, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from greenlist_detector import GreenListDetector
from generate_watermarked import generate_pair, load_model


PROMPTS = [
    "Write a neutral paragraph about the physics of ocean tides.",
    "Explain how a bill becomes law in the United States.",
    "Describe the water cycle in plain language.",
    "Summarise the main causes of the 2008 financial crisis.",
    "Explain what a hash function is to a high-school student.",
    "Describe the structure of a plant cell.",
]


def whisper_light(t):
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    for a, b in [("\u2018","'"),("\u2019","'"),("\u201c",'"'),("\u201d",'"')]:
        t = t.replace(a, b)
    return t.strip()


def word_drop(t, p, rng):
    return " ".join(w for w in t.split() if rng.random() > p)


def adjacent_shuffle(t, p, rng):
    words = t.split()
    for i in range(len(words) - 1):
        if rng.random() < p:
            words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)


TRANSFORMS = {
    "identity":        lambda t, r: t,
    "whisper_light":   lambda t, r: whisper_light(t),
    "lowercase":       lambda t, r: t.lower(),
    "punct_strip":     lambda t, r: re.sub(r"[^\w\s]", "", t),
    "word_drop_10":    lambda t, r: word_drop(t, 0.10, r),
    "word_shuffle_10": lambda t, r: adjacent_shuffle(t, 0.10, r),
}


def main(model_id="openai-community/gpt2", max_new_tokens=220,
         out_csv="data/reports/robustness.csv"):
    print(f"loading {model_id} ...", flush=True)
    tok, model = load_model(model_id)
    det = GreenListDetector(model_id)
    rng = random.Random(0)

    rows = []
    for i, prompt in enumerate(PROMPTS):
        pair = generate_pair(
            model_id, prompt, tokenizer=tok, model=model,
            max_new_tokens=max_new_tokens,
            clean_seed=200 + i, wm_seed=300 + i,
        )
        for name, fn in TRANSFORMS.items():
            perturbed = fn(pair.watermarked, rng)
            r = det.detect_from_text(perturbed)
            rows.append({
                "prompt_idx": i, "transform": name,
                "z": round(r.z_score, 3),
                "pred": r.prediction,
                "tokens": r.num_tokens,
            })
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:45]}...", flush=True)

    path = pathlib.Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print()
    print("── Robustness summary ──")
    by_t = {}
    for r in rows:
        by_t.setdefault(r["transform"], []).append(r["z"])
    for name, zs in by_t.items():
        m = statistics.mean(zs)
        tpr = sum(z > 4 for z in zs) / len(zs)
        print(f"  {name:<18} mean_z={m:+7.2f}   TPR@z>4={tpr:5.0%}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
