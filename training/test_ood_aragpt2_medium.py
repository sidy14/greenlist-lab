"""True OOD test: AraGPT2-medium (base seen in training, medium NOT)."""
import sys, torch
sys.path.insert(0, "src")
from transformers import AutoTokenizer, AutoModelForCausalLM
from arabic_ml_detector import ArabicMLDetector

MODEL = "aubmindlab/aragpt2-medium"
print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
).eval()

prompts = [
    "\u0627\u0643\u062a\u0628 \u0641\u0642\u0631\u0629 \u0639\u0646 \u0627\u0644\u062a\u0639\u0644\u064a\u0645.",
    "\u0627\u0634\u0631\u062d \u0645\u0641\u0647\u0648\u0645 \u0627\u0644\u0633\u0644\u0627\u0645.",
    "\u062a\u062d\u062f\u062b \u0639\u0646 \u0627\u0644\u0635\u062d\u0629.",
    "\u0645\u0627 \u0647\u064a \u0623\u0647\u0645\u064a\u0629 \u0627\u0644\u0639\u0644\u0645\u061f",
    "\u0635\u0641 \u064a\u0648\u0645\u0627\u064b \u0641\u064a \u0627\u0644\u0631\u064a\u0641.",
    "\u0627\u0634\u0631\u062d \u0645\u0634\u0643\u0644\u0629 \u0627\u0644\u062a\u0644\u0648\u062b.",
    "\u062a\u062d\u062f\u062b \u0639\u0646 \u0627\u0644\u0641\u0646.",
    "\u0645\u0627 \u0647\u0648 \u062f\u0648\u0631 \u0627\u0644\u0623\u0628\u061f",
    "\u0627\u0643\u062a\u0628 \u0639\u0646 \u0627\u0644\u0631\u064a\u0627\u0636\u0629.",
    "\u0627\u0634\u0631\u062d \u0627\u0644\u0627\u0642\u062a\u0635\u0627\u062f.",
]

detector = ArabicMLDetector()
print("\n--- True OOD: AraGPT2-MEDIUM (not in training) ---\n")

scores = []
for i, p in enumerate(prompts):
    inputs = tok([p], return_tensors="pt")
    torch.manual_seed(200 + i)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, temperature=0.9, top_p=0.95,
            max_new_tokens=180, pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    if len(text.strip()) < 100:
        continue
    r = detector.detect(text)
    scores.append(r.ai_score)
    print(f"[{i+1}/10] score={r.ai_score:.2%}  verdict={r.verdict[:25]}")

import statistics
if scores:
    print(f"\nmean AI score: {statistics.mean(scores):.2%}")
    print(f"detection rate: {sum(1 for s in scores if s > 0.7)}/{len(scores)}")