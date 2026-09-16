"""Out-of-distribution test: does the detector work on OTHER models?"""
import sys, torch
sys.path.insert(0, "src")
from transformers import AutoTokenizer, AutoModelForCausalLM
from arabic_ml_detector import ArabicMLDetector

# نولّد نص عربي من Qwen 2.5 (لم يره النموذج أثناء التدريب)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
).eval()

# 10 prompts عربية مختلفة عن prompts التدريب
prompts = [
    "اكتب عن مستقبل التعليم في العالم العربي.",
    "اشرح مشكلة التغير المناخي بالعربية.",
    "صف أهمية القراءة في حياة الإنسان.",
    "تحدث عن تاريخ الأندلس بإيجاز.",
    "ما هي فوائد الرياضة اليومية؟",
    "اشرح كيفية عمل الإنترنت.",
    "تحدث عن تأثير وسائل التواصل الاجتماعي.",
    "ما هي أفضل طريقة لتعلم لغة جديدة؟",
    "اشرح مفهوم الذكاء الاصطناعي.",
    "اكتب نصيحة لطالب جامعي.",
]

detector = ArabicMLDetector()
print("\n--- OOD Test: Qwen 2.5 (not seen in training) ---\n")

results = []
for i, prompt in enumerate(prompts):
    inputs = tok([prompt], return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    torch.manual_seed(100 + i)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, temperature=0.9, top_p=0.95,
            max_new_tokens=200, pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, input_len:], skip_special_tokens=True)
    if len(text.strip()) < 100:
        continue
    r = detector.detect(text)
    results.append(r.ai_score)
    print(f"[{i+1}/10] score={r.ai_score:.2%}  verdict={r.verdict[:20]}")

import statistics
if results:
    print(f"\nmean AI score: {statistics.mean(results):.2%}")
    print(f"detection rate (score>0.7): "
          f"{sum(1 for s in results if s>0.7)}/{len(results)}")