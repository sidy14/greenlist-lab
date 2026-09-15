"""
tinyllama_synthid_resume.py — memory-safe SynthID run on TinyLlama.
Writes CSV incrementally after every prompt.
"""
from __future__ import annotations
import csv, gc, pathlib, random, re, statistics, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from synthid_detector import SynthIDDetector
from generate_watermarked import load_model
from transformers import SynthIDTextWatermarkingConfig


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW = 180
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


def _format_prompt(tok, prompt):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return prompt


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


def _generate(tok, model, prompt, seed, wm_config=None):
    formatted = _format_prompt(tok, prompt)
    inputs = tok([formatted], return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    torch.manual_seed(seed)
    kwargs = dict(
        **inputs, do_sample=True, temperature=0.9, top_p=0.95,
        max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id,
    )
    if wm_config is not None:
        kwargs["watermarking_config"] = wm_config
    with torch.no_grad():
        out = model.generate(**kwargs)
    text = tok.decode(out[0, input_len:], skip_special_tokens=True)
    del out, inputs
    return text


def _wm_cfg():
    return SynthIDTextWatermarkingConfig(
        ngram_len=NGRAM_LEN,
        keys=list(range(1, 31)),
        sampling_table_size=65536,
        sampling_table_seed=0,
        context_history_size=1024,
    )


def _gen_pair(tok, model, prompt, seed):
    clean = _generate(tok, model, prompt, seed)
    gc.collect()
    wm = _generate(tok, model, prompt, seed + 1, wm_config=_wm_cfg())
    gc.collect()
    class _P: pass
    p = _P(); p.clean = clean; p.watermarked = wm
    return p


def _try_pair(tok, model, prompt, base_seed, max_attempts=3):
    for a in range(max_attempts):
        seed = base_seed + a * 1000
        p = _gen_pair(tok, model, prompt, seed)
        if (len(p.clean.strip()) >= MIN_CHARS
                and len(p.watermarked.strip()) >= MIN_CHARS):
            return p, a
        print(f"      retry {a+1}", flush=True)
        gc.collect()
    return p, max_attempts


def _append_row(path, fieldnames, row):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def calibration(tok, model):
    print("[A] SynthID calibration on TinyLlama ...", flush=True)
    csv_path = "data/reports/tinyllama_synthid_calibration.csv"
    pathlib.Path(csv_path).unlink(missing_ok=True)
    fields = ["prompt", "z_clean", "z_wm", "attempts"]
    det = SynthIDDetector(MODEL_ID, ngram_len=NGRAM_LEN)
    for i, prompt in enumerate(CAL_PROMPTS):
        pair, att = _try_pair(tok, model, prompt, base_seed=100 + i * 2)
        rc = det.detect_from_text(pair.clean)
        rw = det.detect_from_text(pair.watermarked)
        _append_row(csv_path, fields, {
            "prompt": prompt,
            "z_clean": round(rc.z_score, 3),
            "z_wm": round(rw.z_score, 3),
            "attempts": att + 1,
        })
        print(f"  [{i+1}/{len(CAL_PROMPTS)}] z_clean={rc.z_score:+6.2f}  "
              f"z_wm={rw.z_score:+6.2f}  (saved)", flush=True)
        del pair, rc, rw
        gc.collect()
    print(f"  wrote {csv_path}", flush=True)


def robustness(tok, model):
    print("[B] SynthID robustness on TinyLlama ...", flush=True)
    csv_path = "data/reports/tinyllama_synthid_robustness.csv"
    pathlib.Path(csv_path).unlink(missing_ok=True)
    fields = ["prompt_idx", "transform", "z", "tokens"]
    det = SynthIDDetector(MODEL_ID, ngram_len=NGRAM_LEN)
    rng = random.Random(0)
    for i, prompt in enumerate(ROB_PROMPTS):
        pair, _ = _try_pair(tok, model, prompt, base_seed=200 + i)
        for name, fn in TRANSFORMS.items():
            perturbed = fn(pair.watermarked, rng)
            r = det.detect_from_text(perturbed)
            _append_row(csv_path, fields, {
                "prompt_idx": i, "transform": name,
                "z": round(r.z_score, 3),
                "tokens": r.num_tokens,
            })
            del r, perturbed
        print(f"  [{i+1}/{len(ROB_PROMPTS)}] {prompt[:40]}...  (saved)", flush=True)
        del pair
        gc.collect()
    print(f"  wrote {csv_path}", flush=True)


def main():
    print(f"Loading {MODEL_ID} ...", flush=True)
    tok, model = load_model(MODEL_ID)
    model.eval()
    print("Ready.\n", flush=True)
    calibration(tok, model)
    gc.collect()
    print()
    robustness(tok, model)
    print("\nDone.")


if __name__ == "__main__":
    main()
