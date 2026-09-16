"""Batch validation on ChatGPT and human Arabic text."""
import sys, pathlib, statistics
sys.path.insert(0, "src")
from arabic_ml_detector import ArabicMLDetector

det = ArabicMLDetector()

for folder in ["validation_chatgpt", "validation_human"]:
    print(f"\n=== {folder} ===")
    scores = []
    for p in sorted(pathlib.Path(f"data/{folder}").glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        if len(text.strip()) < 50:
            print(f"  {p.name}: SKIP (too short)")
            continue
        r = det.detect(text)
        scores.append(r.ai_score)
        print(f"  {p.name}: {r.ai_score:.2%}  ({r.verdict})")
    if scores:
        print(f"  ---")
        print(f"  mean: {statistics.mean(scores):.2%}")
        print(f"  min:  {min(scores):.2%}")
        print(f"  max:  {max(scores):.2%}")