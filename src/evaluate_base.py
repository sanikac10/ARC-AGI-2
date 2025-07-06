# =============================
# File: src/evaluate_base.py
# Purpose: Measure accuracy of a checkpoint on the fixed test pool
# =============================

import argparse, numpy as np, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.exec import build_prompt, extract_grid
from arcagi2 import load_pool

parser = argparse.ArgumentParser(); parser.add_argument("--ckpt"); args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(args.ckpt)

tasks = load_pool("data/test_pool.jsonl")
correct = 0
for t in tasks:
    out = model.generate(**tok(build_prompt(t), return_tensors="pt").to(model.device), max_new_tokens=1024)
    grid_pred = extract_grid(tok.decode(out[0]))
    if grid_pred is not None and np.array_equal(grid_pred, np.array(t["gt_grid"])):
        correct += 1

acc = correct / len(tasks)
print(f"{acc:.3f}")