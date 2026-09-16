"""Generate Arabic AI text using AraGPT2-base."""
import json, pathlib, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "aubmindlab/aragpt2-base"
OUT = pathlib.Path("data/arabic/ai.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "\u0627\u0643\u062a\u0628 \u0641\u0642\u0631\u0629 \u0639\u0646",
    "\u0627\u0634\u0631\u062d \u0645\u0641\u0647\u0648\u0645",
    "\u0644\u062e\u0635 \u0627\u0644\u0646\u0635 \u0627\u0644\u062a\u0627\u0644\u064a",
    "\u062a\u062d\u062f\u062b \u0639\u0646 \u0623\u0647\u0645\u064a\u0629",
    "\u0642\u062f\u0645 \u0646\u0635\u064a\u062d\u0629 \u062d\u0648\u0644",
    "\u0645\u0627 \u0647\u064a \u0623\u0633\u0628\u0627\u0628",
    "\u0643\u064a\u0641 \u064a\u0645\u0643\u0646 \u062a\u062d\u0633\u064a\u0646",
    "\u0645\u0627 \u0647\u0648 \u062a\u0639\u0631\u064a\u0641",
    "\u0627\u0634\u0631\u062d \u0644\u064a \u0628\u0628\u0633\u0627\u0637\u0629",
    "\u0642\u0627\u0631\u0646 \u0628\u064a\u0646",
    "\u0623\u0639\u0637\u0646\u064a \u062e\u0644\u0627\u0635\u0629",
    "\u0627\u0643\u062a\u0628 \u0645\u0642\u0627\u0644\u0629 \u0639\u0646",
    "\u062d\u0644\u0644 \u0627\u0644\u0645\u0648\u0636\u0648\u0639",
    "\u0627\u0642\u062a\u0631\u062d \u062d\u0644\u0648\u0644\u0627 \u0644\u0645\u0634\u0643\u0644\u0629",
    "\u0627\u0634\u0631\u062d \u0627\u0644\u0641\u0631\u0642 \u0628\u064a\u0646",
    "\u062a\u062d\u062f\u062b \u0639\u0646 \u062a\u0627\u0631\u064a\u062e",
    "\u0645\u0627 \u0647\u064a \u0641\u0648\u0627\u0626\u062f",
    "\u0643\u064a\u0641 \u062a\u0639\u0645\u0644",
    "\u0623\u0648\u0636\u062d \u0628\u0627\u0644\u0623\u0645\u062b\u0644\u0629",
    "\u0627\u0643\u062a\u0628 \u062e\u0637\u0629",
    "\u0635\u0641 \u0627\u0644\u0639\u0644\u0627\u0642\u0629 \u0628\u064a\u0646",
    "\u0627\u0634\u0631\u062d \u0628\u0627\u0644\u062a\u0641\u0635\u064a\u0644",
    "\u0644\u062e\u0635 \u0623\u0647\u0645 \u0627\u0644\u0646\u0642\u0627\u0637",
    "\u0645\u0627 \u0647\u0648 \u062f\u0648\u0631",
    "\u062a\u062d\u062f\u062b \u0639\u0646 \u062a\u0623\u062b\u064a\u0631",
    "\u0627\u0643\u062a\u0628 \u0645\u0642\u062f\u0645\u0629 \u0639\u0646",
    "\u0634\u0631\u062d \u0628\u0634\u0643\u0644 \u0645\u0628\u0633\u0637",
    "\u0645\u0627 \u0647\u064a \u0623\u0647\u0645 \u0627\u0644\u062e\u0637\u0648\u0627\u062a",
    "\u0627\u0642\u062a\u0631\u062d \u0623\u0641\u0643\u0627\u0631\u0627 \u062d\u0648\u0644",
    "\u0627\u0643\u062a\u0628 \u062a\u0642\u0631\u064a\u0631\u0627 \u0639\u0646",
]

SUFFIXES = [
    " \u0627\u0644\u0645\u0646\u0627\u062e \u0648\u0627\u0644\u0628\u064a\u0626\u0629.",
    " \u0627\u0644\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627 \u0627\u0644\u062d\u062f\u064a\u062b\u0629.",
    " \u0627\u0644\u062a\u0639\u0644\u064a\u0645 \u0648\u0627\u0644\u062a\u0639\u0644\u0645.",
    " \u0627\u0644\u0635\u062d\u0629 \u0648\u0627\u0644\u0637\u0628.",
    " \u0627\u0644\u0627\u0642\u062a\u0635\u0627\u062f \u0648\u0627\u0644\u062a\u062c\u0627\u0631\u0629.",
    " \u0627\u0644\u062a\u0627\u0631\u064a\u062e \u0648\u0627\u0644\u062d\u0636\u0627\u0631\u0629.",
    " \u0627\u0644\u0631\u064a\u0627\u0636\u0629 \u0648\u0627\u0644\u0641\u0646\u0648\u0646.",
    " \u0627\u0644\u0639\u0644\u0648\u0645 \u0648\u0627\u0644\u0627\u0628\u062a\u0643\u0627\u0631.",
    " \u0627\u0644\u0633\u064a\u0627\u0633\u0629 \u0648\u0627\u0644\u0645\u062c\u062a\u0645\u0639.",
    " \u0627\u0644\u0633\u064a\u0627\u062d\u0629 \u0648\u0627\u0644\u0633\u0641\u0631.",
]


def main(target=300):
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.eval()

    samples = []
    per_prompt = max(1, target // len(PROMPTS))
    idx = 0
    for i, base in enumerate(PROMPTS):
        for j in range(per_prompt):
            suffix = SUFFIXES[idx % len(SUFFIXES)]
            idx += 1
            prompt = base + suffix
            torch.manual_seed(i * 100 + j)
            inputs = tok([prompt], return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs, do_sample=True, temperature=0.9,
                    top_p=0.95, max_new_tokens=200,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0], skip_special_tokens=True)
            if 200 < len(text) < 4000:
                samples.append(text.strip())
        print(f"  prompt {i+1}/{len(PROMPTS)}: total={len(samples)}", flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for t in samples:
            f.write(json.dumps({"text": t, "label": 1}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(samples)} samples")


if __name__ == "__main__":
    main()