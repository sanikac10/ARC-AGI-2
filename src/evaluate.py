# =============================
# File: src/evaluate.py
# Purpose: Evaluate + optionally harvest correct predictions into JSONL
# =============================

import argparse, json, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.exec import build_prompt, extract_grid
from arcagi2 import load_pool

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--harvest", help="path to write harvested triples (optional)")
args = parser.parse_args()

tasks = load_pool("data/test_pool.jsonl")
model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(args.ckpt)

harvest = []
hit = 0
for t in tasks:
    txt = tok(build_prompt(t), return_tensors="pt").to(model.device)
    out = model.generate(**txt, max_new_tokens=1024)
    grid_pred = extract_grid(tok.decode(out[0]))
    if grid_pred is not None and np.array_equal(grid_pred, np.array(t["gt_grid"])):
        hit += 1
        if args.harvest:
            harvest.append({
                "id": t["id"],
                "input_grid": t["input_grid"],
                "output_grid": t["gt_grid"],
                "context": tok.decode(out[0]),
                "answer_grid": t["gt_grid"],
            })

print(f"Accuracy: {hit/len(tasks):.3f}")

if args.harvest and harvest:
    Path(args.harvest).write_text("\n".join(json.dumps(x) for x in harvest))
    print(f"Harvested {len(harvest)} correct tasks → {args.harvest}")
