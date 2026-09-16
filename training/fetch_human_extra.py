"""Additional human text from Arabic Wikipedia (different topics)."""
import json, pathlib
from datasets import load_dataset

OUT = pathlib.Path("data/arabic/human_extra.jsonl")
TARGET = 300


def main():
    ds = load_dataset("wikimedia/wikipedia", "20231101.ar",
                      split="train", streaming=True)
    texts = []
    # تخطَّ أول 350 عيّنة (استخدمناها سابقاً)
    for i, item in enumerate(ds):
        if i < 350:
            continue
        t = item.get("text", "").strip()
        if 400 < len(t) < 3000:
            texts.append(t)
        if len(texts) >= TARGET:
            break
        if i % 100 == 0 and i > 350:
            print(f"  collected {len(texts)} (scanned {i})", flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "label": 0}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(texts)} samples")


if __name__ == "__main__":
    main()