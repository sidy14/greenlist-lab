"""
train_arabic_detector.py
Fine-tune AraELECTRA-base on the collected human vs AI Arabic corpus.
Runs on CPU in 30-45 minutes.
"""
import json
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


MODEL_ID = "aubmindlab/araelectra-base-discriminator"
MAX_LEN = 256
OUT_DIR = "models/arabic-ai-detector"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ArabicDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    print("loading corpus ...", flush=True)
    human = load_split("data/arabic/human.jsonl")
    human_extra = load_split("data/arabic/human_extra.jsonl")
    ai = load_split("data/arabic/ai.jsonl")
    ai_qwen = load_split("data/arabic/ai_qwen.jsonl")
    all_samples = human + human_extra + ai + ai_qwen
    print(f"  human:       {len(human)}", flush=True)
    print(f"  human_extra: {len(human_extra)}", flush=True)
    print(f"  araGPT2:     {len(ai)}", flush=True)
    print(f"  qwen:        {len(ai_qwen)}", flush=True)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = all_samples[:n_train]
    val = all_samples[n_train:n_train + n_val]
    test = all_samples[n_train + n_val:]

    print(f"total: {n}  train: {len(train)}  val: {len(val)}  test: {len(test)}",
          flush=True)

    # Save the test split for later evaluation
    pathlib.Path("data/arabic/test.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in test),
        encoding="utf-8",
    )

    print(f"loading tokenizer + model: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2,
    )

    train_ds = ArabicDataset(train, tokenizer)
    val_ds = ArabicDataset(val, tokenizer)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=50,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        seed=SEED,
        use_cpu=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("training ...", flush=True)
    trainer.train()

    print(f"saving model to {OUT_DIR}", flush=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("done.")


if __name__ == "__main__":
    main()