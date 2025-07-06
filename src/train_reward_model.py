# =============================
# File: src/train_reward_model.py
# Purpose: Train scalar reward model with partial credit
# =============================

import argparse
from datasets import load_dataset
from transformers import TrainingArguments
from trl import AutoModelForSequenceClassificationWithValueHead, RewardTrainer

parser = argparse.ArgumentParser(); parser.add_argument("--output", default="ckpts/reward"); args = parser.parse_args()

model = AutoModelForSequenceClassificationWithValueHead.from_pretrained(
    "PerceptionLM-8B", num_labels=1, torch_dtype="auto")

pairs = load_dataset("json", data_files="data/rm_pairs.jsonl")

train_args = TrainingArguments(
    output_dir=args.output,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    num_train_epochs=1,
    fp16=True,
)

trainer = RewardTrainer(model=model, args=train_args, train_dataset=pairs["train"])
trainer.train(); trainer.save_model(args.output)
