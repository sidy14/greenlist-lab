"""Evaluate the trained Arabic AI-detector on the held-out test set."""
import json
import pathlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

MODEL_DIR = "models/arabic-ai-detector"
TEST_PATH = "data/arabic/test.jsonl"


def main():
    print(f"loading model: {MODEL_DIR}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open(TEST_PATH, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"test set: {len(samples)} samples", flush=True)

    texts = [s["text"] for s in samples]
    labels = np.array([s["label"] for s in samples])

    preds = []
    with torch.no_grad():
        for text in texts:
            enc = tok(text, truncation=True, padding="max_length",
                     max_length=256, return_tensors="pt")
            out = model(**enc)
            pred = int(out.logits.argmax(dim=-1).item())
            preds.append(pred)
    preds = np.array(preds)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    cm = confusion_matrix(labels, preds)

    print()
    print("=" * 60)
    print("ARABIC AI-DETECTOR — TEST SET EVALUATION")
    print("=" * 60)
    print(f"samples     : {len(samples)}")
    print(f"accuracy    : {acc:.4f}")
    print(f"precision   : {prec:.4f}")
    print(f"recall      : {rec:.4f}")
    print(f"F1 (binary) : {f1:.4f}")
    print()
    print("confusion matrix:")
    print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    print()
    print(classification_report(labels, preds,
                                target_names=["human", "ai"],
                                digits=3))

    # Save results
    out = pathlib.Path("data/reports/arabic_detector_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(samples),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()