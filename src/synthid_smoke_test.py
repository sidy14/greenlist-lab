"""
synthid_smoke_test.py — quick sanity check of the SynthID pipeline.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from synthid_detector import SynthIDDetector
from synthid_generate_watermarked import generate_pair_synthid, load_model


PROMPT = "Write a short neutral paragraph about the physics of ocean tides."
MODEL_ID = "openai-community/gpt2"


def main():
    print(f"loading {MODEL_ID} ...", flush=True)
    tok, model = load_model(MODEL_ID)

    print("generating pair ...", flush=True)
    pair = generate_pair_synthid(
        MODEL_ID,
        PROMPT,
        tokenizer=tok,
        model=model,
        ngram_len=5,
        max_new_tokens=220,
        clean_seed=0,
        wm_seed=1,
    )

    det = SynthIDDetector(MODEL_ID, ngram_len=5)

    rc = det.detect_from_text(pair.clean)
    rw = det.detect_from_text(pair.watermarked)

    print()
    print("=" * 60)
    print("SYNTHID SMOKE TEST")
    print("=" * 60)
    print(f"CLEAN       : {rc.summary()}")
    print(f"WATERMARKED : {rw.summary()}")
    print()

    if rw.z_score > 4 and rc.z_score < 4:
        print("RESULT: PASS")
    elif rw.z_score > rc.z_score:
        print("RESULT: PARTIAL")
    else:
        print("RESULT: FAIL")


if __name__ == "__main__":
    main()
