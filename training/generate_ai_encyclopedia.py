"""Generate ChatGPT-style encyclopedia Arabic for training."""
import json, pathlib, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = pathlib.Path("data/arabic/ai_encyclopedia.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

TOPICS = [
    "الإمبراطورية الرومانية", "الحضارة المصرية القديمة",
    "الثورة الصناعية", "علم الفلك الإسلامي",
    "تاريخ الأندلس", "الخلافة العباسية",
    "الجغرافيا الطبيعية", "علم الأحياء",
    "الفلسفة اليونانية", "الأدب العربي",
    "الحضارة الصينية", "الطب الحديث",
    "الفيزياء النووية", "الرياضيات",
    "الاقتصاد العالمي", "العلوم السياسية",
    "اللغة العربية", "الشعر الجاهلي",
    "علم النفس", "علم الاجتماع",
]

def main(target=200):
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()

    samples = []
    per_topic = max(1, target // len(TOPICS))
    for i, topic in enumerate(TOPICS):
        prompt = f"اكتب فقرة موسوعية تفصيلية (300-500 كلمة) عن {topic} بأسلوب ويكيبيديا."
        msgs = [{"role": "user", "content": prompt}]
        fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok([fmt], return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        for j in range(per_topic):
            torch.manual_seed(i * 200 + j)
            with torch.no_grad():
                out = model.generate(
                    **inputs, do_sample=True, temperature=0.85, top_p=0.92,
                    max_new_tokens=500, pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0, input_len:], skip_special_tokens=True)
            if 400 < len(text) < 4000:
                samples.append(text.strip())
        print(f"  topic {i+1}/{len(TOPICS)}: total={len(samples)}", flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for t in samples:
            f.write(json.dumps({"text": t, "label": 1}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(samples)} samples")


if __name__ == "__main__":
    main()