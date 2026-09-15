"""
tinyllama_all.py v2 — chat-template-aware, runs four experiments.
"""
from __future__ import annotations
import csv, pathlib, random, re, statistics, sys
import torch
from transformers import AutoTokenizer
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from greenlist_detector import GreenListDetector
from generate_watermarked import load_model
from synthid_detector import SynthIDDetector
from transformers import WatermarkingConfig, SynthIDTextWatermarkingConfig


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW = 250
MIN_CHARS = 40
NGRAM_LEN = 5


CAL_PROMPTS = [
    "Write a neutral paragraph about the physics of ocean tides.",
    "Explain how a bill becomes law in the United States.",
    "Describe the water cycle in plain language.",
    "Summarise the main causes of the 2008 financial crisis.",
    "Explain what a hash function is to a high-school student.",
    "Describe the structure of a plant cell.",
    "Summarise the plot of Hamlet in three sentences.",
    "Explain why the sky is blue.",
]

ROB_PROMPTS = CAL_PROMPTS[:6]


# ---- prompt formatting --------------------------------------------------

def _format_prompt(tok, prompt):
    """Apply chat template if available; else plain text."""
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return prompt


# ---- transforms ---------------------------------------------------------

def _whisper(t):
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    for a, b in [("\u2018","'"), ("\u2019","'"), ("\u201c",'"'), ("\u201d",'"')]:
        t = t.replace(a, b)
    return t.strip()


def _word_drop(t, p, rng):
    return " ".join(w for w in t.split() if rng.random() > p)


def _shuffle(t, p, rng):
    ws = t.split()
    for i in range(len(ws) - 1):
        if rng.random() < p:
            ws[i], ws[i + 1] = ws[i + 1], ws[i]
    return " ".join(ws)


TRANSFORMS = {
    "identity":        lambda t, r: t,
    "whisper_light":   lambda t, r: _whisper(t),
    "lowercase":       lambda t, r: t.lower(),
    "punct_strip":     lambda t, r: re.sub(r"[^\w\s]", "", t),
    "word_drop_10":    lambda t, r: _word_drop(t, 0.10, r),
    "word_shuffle_10": lambda t, r: _shuffle(t, 0.10, r),
}


# ---- core generation ----------------------------------------------------

def _generate(tok, model, prompt, seed, wm_config=None, max_new=MAX_NEW):
    """Generate a completion from a chat-formatted prompt."""
    formatted = _format_prompt(tok, prompt)
    inputs = tok([formatted], return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]

    torch.manual_seed(seed)
    kwargs = dict(
        **inputs,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=max_new,
        pad_token_id=tok.eos_token_id,
    )
    if wm_config is not None:
        kwargs["watermarking_config"] = wm_config

    with torch.no_grad():
        out = model.generate(**kwargs)
    return tok.decode(out[0, input_len:], skip_special_tokens=True)


def _gen_pair(model_id, prompt, tok, model, seed, kind):
    """kind = 'green' or 'synthid'."""
    clean = _generate(tok, model, prompt, seed)
    if kind == "green":
        wm_cfg = WatermarkingConfig(
            greenlist_ratio=0.25,
            bias=2.0,
            seeding_scheme="lefthash",
            context_width=1,
            hashing_key=15485863,
        )
    else:
        wm_cfg = SynthIDTextWatermarkingConfig(
            ngram_len=NGRAM_LEN,
            keys=list(range(1, 31)),
            sampling_table_size=65536,
            sampling_table_seed=0,
            context_history_size=1024,
        )
    wm = _generate(tok, model, prompt, seed + 1, wm_config=wm_cfg)

    class _P:
        pass
    p = _P()
    p.clean = clean
    p.watermarked = wm
    return p


def _try_generate(model_id, prompt, tok, model, base_seed, kind, max_attempts=3):
    for attempt in range(max_attempts):
        seed = base_seed + attempt * 1000
        pair = _gen_pair(model_id, prompt, tok, model, seed, kind)
        if (len(pair.clean.strip()) >= MIN_CHARS
                and len(pair.watermarked.strip()) >= MIN_CHARS):
            return pair, attempt
        print(f"      retry {attempt + 1} (clean={len(pair.clean)} wm={len(pair.watermarked)})",
              flush=True)
    return pair, max_attempts


def _write_csv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---- experiments --------------------------------------------------------

def run_calibration(tok, model, kind):
    det = GreenListDetector(MODEL_ID) if kind == "green" else \
          SynthIDDetector(MODEL_ID, ngram_len=NGRAM_LEN)
    rows = []
    for i, prompt in enumerate(CAL_PROMPTS):
        pair, att = _try_generate(
            MODEL_ID, prompt, tok, model,
            base_seed=100 + i * 2, kind=kind,
        )
        rc = det.detect_from_text(pair.clean)
        rw = det.detect_from_text(pair.watermarked)
        rows.append({
            "prompt": prompt,
            "z_clean": round(rc.z_score, 3),
            "z_wm": round(rw.z_score, 3),
            "attempts": att + 1,
        })
        print(f"  [{i+1}/{len(CAL_PROMPTS)}] z_clean={rc.z_score:+6.2f}  "
              f"z_wm={rw.z_score:+6.2f}", flush=True)
    _write_csv(f"data/reports/tinyllama_{kind}_calibration.csv", rows)

    zc = [r["z_clean"] for r in rows]
    zw = [r["z_wm"] for r in rows]
    print(f"  sep = {statistics.mean(zw) - statistics.mean(zc):.2f}")
    return rows


def run_robustness(tok, model, kind):
    det = GreenListDetector(MODEL_ID) if kind == "green" else \
          SynthIDDetector(MODEL_ID, ngram_len=NGRAM_LEN)
    rng = random.Random(0)
    rows = []
    for i, prompt in enumerate(ROB_PROMPTS):
        pair, _ = _try_generate(
            MODEL_ID, prompt, tok, model,
            base_seed=200 + i, kind=kind,
        )
        for name, fn in TRANSFORMS.items():
            perturbed = fn(pair.watermarked, rng)
            r = det.detect_from_text(perturbed)
            rows.append({
                "prompt_idx": i, "transform": name,
                "z": round(r.z_score, 3),
                "tokens": r.num_tokens,
            })
        print(f"  [{i+1}/{len(ROB_PROMPTS)}] {prompt[:40]}...", flush=True)
    _write_csv(f"data/reports/tinyllama_{kind}_robustness.csv", rows)

    by_t = {}
    for r in rows:
        by_t.setdefault(r["transform"], []).append(r["z"])
    for name, zs in by_t.items():
        print(f"    {name:<18} {statistics.mean(zs):+7.2f}")
    return rows


def main():
    print(f"Loading {MODEL_ID} ...", flush=True)
    tok, model = load_model(MODEL_ID)
    print("Model ready.\n", flush=True)

    print("[1/4] Green-List calibration ...", flush=True)
    run_calibration(tok, model, "green")
    print()
    print("[2/4] Green-List robustness ...", flush=True)
    run_robustness(tok, model, "green")
    print()
    print("[3/4] SynthID calibration ...", flush=True)
    run_calibration(tok, model, "synthid")
    print()
    print("[4/4] SynthID robustness ...", flush=True)
    run_robustness(tok, model, "synthid")

    print()
    print("Done. Four CSVs in data/reports/.")


if __name__ == "__main__":
    main()
