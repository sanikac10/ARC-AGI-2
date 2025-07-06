# =============================
# File: src/train_sft.py
# Purpose: Full‑parameter supervised fine‑tune on synthetic + harvested data
# =============================

import argparse, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

DS_CONFIG = "deepspeed/zero3.json"  # ZeRO‑3 offloading config

# -----------------------------
# parse CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--resume", help="checkpoint to resume from", default="PerceptionLM-8B")
parser.add_argument("--output", help="where to save new ckpt", default="ckpts/sft_full")
args = parser.parse_args()

# -----------------------------
# load model/tokenizer
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(args.resume, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(args.resume)

# dataset is single file train.jsonl (already synthetic + harvests)
train_ds = load_dataset("json", data_files="data/train.jsonl")

def tok(b):
    return tokenizer(b["context"], truncation=True)

training_args = TrainingArguments(
    output_dir=args.output,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=3e-5,
    num_train_epochs=3,
    fp16=True,
    deepspeed=DS_CONFIG,
    save_total_limit=2,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds["train"].map(tok))
trainer.train(); trainer.save_model(args.output)
