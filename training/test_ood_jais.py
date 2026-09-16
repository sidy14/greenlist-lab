"""True OOD test #2: Jais-family model (completely different family)."""
import sys, torch
sys.path.insert(0, "src")
from transformers import AutoTokenizer, AutoModelForCausalLM
from arabic_ml_detector import ArabicMLDetector

MODEL = "inceptionai/jais-family-590m-chat"
print(f"loading {MODEL} ...", flush=True)
try:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
except Exception as e:
    print(f"FAILED: {e}")
    print("Falling back to Qwen 1.5B (different size, same family)")
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()

prompts = [
    "\u0627\u0643\u062a\u0628 \u0641\u0642\u0631\u0629 \u0639\u0646 \u0627\u0644\u062a\u0643\u0646\u0648\u0644\u0648\u062c\u064a\u0627.",
    "\u0627\u0634\u0631\u062d \u0623\u0647\u0645\u064a\u0629 \u0627\u0644\u0635\u062d\u0629 \u0627\u0644\u0646\u0641\u0633\u064a\u0629.",
    "\u062a\u062d\u062f\u062b \u0639\u0646 \u0627\u0644\u0645\u0646\u0627\u062e.",
    "\u0645\u0627 \u0647\u064a \u0641\u0648\u0627\u0626\u062f \u0627\u0644\u0642\u0631\u0627\u0621\u0629\u061f",
    "\u0635\u0641 \u0645\u062f\u064a\u0646\u0629 \u0639\u0631\u0628\u064a\u0629.",
]

detector = ArabicMLDetector()
print(f"\n--- True OOD: {MODEL} ---\n")

scores = []
for i, p in enumerate(prompts):
    try:
        msgs = [{"role": "user", "content": p}]
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            fmt = p
        inputs = tok([fmt], return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        torch.manual_seed(300 + i)
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=True, temperature=0.9, top_p=0.95,
                max_new_tokens=180, pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, input_len:], skip_special_tokens=True)
        if len(text.strip()) < 100:
            continue
        r = detector.detect(text)
        scores.append(r.ai_score)
        print(f"[{i+1}/5] score={r.ai_score:.2%}  verdict={r.verdict[:25]}")
    except Exception as e:
        print(f"[{i+1}/5] FAILED: {e}")

import statistics
if scores:
    print(f"\nmean AI score: {statistics.mean(scores):.2%}")
    print(f"detection rate: {sum(1 for s in scores if s > 0.7)}/{len(scores)}")