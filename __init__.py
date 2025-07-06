# =============================
# File: arcagi2/__init__.py
# Purpose: Thin loader for ARC‑AGI‑2 dataset & utilities
# =============================

import json, random

# Path to the raw ARC‑AGI‑2 JSON (you need to download separately)
ARC_PATH = "data/arcagi2_full.jsonl"

# -----------------------------
# load_tasks
#   • Returns list of task dicts.  `split='public'` loads all tasks;
#     you can add custom logic (e.g. easy‑only).
# -----------------------------

def load_tasks(split: str = "public"):
    with open(ARC_PATH) as f:
        tasks = [json.loads(l) for l in f]
    return tasks

# -----------------------------
# load_pool
#   • Returns bottom‑50% by id hash (deterministic) as test pool.
# -----------------------------

def load_pool(pool_file: str):
    return [json.loads(l) for l in open(pool_file)]

# Helper to create test_pool.jsonl once
if __name__ == "__main__":
    tasks = load_tasks()
    bottom_half = sorted(tasks, key=lambda t: t["id"])[: len(tasks)//2]
    with open("data/test_pool.jsonl", "w") as w:
        for t in bottom_half:
            w.write(json.dumps({"id": t["id"], "input_grid": t["input_grid"], "gt_grid": t["output_grid"]}) + "\n")
    print("test_pool.jsonl generated (bottom‑50%).")

