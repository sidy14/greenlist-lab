"""Fetch Arabic human text. Two sources: HF datasets, then Wikipedia."""
import json, time, pathlib, sys

OUT = pathlib.Path("data/arabic/human.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
TARGET = 300


def from_hf_datasets():
    """Preferred: pull from HuggingFace datasets (no API issues)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed - skipping HF source", flush=True)
        return []
    try:
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.ar",
            split="train", streaming=True,
        )
    except Exception as e:
        print(f"HF load failed: {e}", flush=True)
        return []
    texts = []
    for item in ds:
        t = item.get("text", "").strip()
        if 400 < len(t) < 4000:
            texts.append(t)
            if len(texts) >= TARGET:
                break
    return texts


def from_wikipedia_api():
    """Fallback: Wikipedia REST API with proper User-Agent."""
    import requests
    UA = {
        "User-Agent": "ai-text-lab/0.2 (research; contact: sidy@example.com)",
        "Accept": "application/json",
    }
    API = "https://ar.wikipedia.org/w/api.php"
    texts = []
    seen = set()
    for _ in range(20):
        if len(texts) >= TARGET:
            break
        try:
            r = requests.get(API, headers=UA, timeout=30, params={
                "action": "query", "list": "random",
                "rnnamespace": 0, "rnlimit": 20, "format": "json",
            })
            r.raise_for_status()
            ids = [p["id"] for p in r.json()["query"]["random"]
                   if p["id"] not in seen]
            seen.update(ids)
            if not ids:
                continue
            r2 = requests.get(API, headers=UA, timeout=60, params={
                "action": "query",
                "pageids": "|".join(map(str, ids)),
                "prop": "extracts", "explaintext": 1,
                "exlimit": len(ids), "format": "json",
            })
            r2.raise_for_status()
            for _, page in r2.json()["query"]["pages"].items():
                t = page.get("extract", "").strip()
                if 400 < len(t) < 4000:
                    texts.append(t)
        except Exception as e:
            print(f"wiki error: {type(e).__name__}: {e}", flush=True)
        print(f"  total={len(texts)}", flush=True)
        time.sleep(0.5)
    return texts


def main():
    print("Trying HuggingFace datasets first ...", flush=True)
    texts = from_hf_datasets()

    if len(texts) < TARGET:
        print(f"Only {len(texts)} from HF. Falling back to Wikipedia API ...",
              flush=True)
        texts += from_wikipedia_api()

    texts = texts[:TARGET]
    with OUT.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "label": 0},
                               ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(texts)} samples")


if __name__ == "__main__":
    main()