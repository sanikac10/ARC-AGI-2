# =============================
# File: src/run_experiment.py
# Purpose: End‑to‑end orchestration of SFT → GRPO → Eval → Harvest loop
# =============================

import subprocess, csv, shutil, os, json, sys

CYCLES = 3  # number of self‑training rounds
CKPT_INIT = "ckpts/sft_full"
RESULTS = "results/accuracy_log.csv"

# ensure dirs
os.makedirs("results", exist_ok=True)
if not os.path.isfile(RESULTS):
    csv.writer(open(RESULTS, "w", newline="")).writerow(["cycle", "phase", "accuracy"])

def log(cycle:int, phase:str, acc:float):
    csv.writer(open(RESULTS, "a", newline="")).writerow([cycle, phase, acc])

# baseline
acc = float(subprocess.check_output([sys.executable, "src/evaluate_base.py", "--ckpt", CKPT_INIT]).decode())
log(0, "baseline", acc); print(f"Cycle0 baseline={acc:.3f}")

ckpt = CKPT_INIT
for c in range(1, CYCLES+1):
    # RLHF stage
    subprocess.run([sys.executable, "src/train_policy_grpo.py", "--cycle", str(c)], check=True)
    rl_ckpt = f"ckpts/rl_cycle{c}"

    acc_rl = float(subprocess.check_output([sys.executable, "src/evaluate_base.py", "--ckpt", rl_ckpt]).decode())
    log(c, "post_grpo", acc_rl); print(f"Cycle{c} post_grpo={acc_rl:.3f}")

    harvest_file = f"data/harvest_cycle{c}.jsonl"
    subprocess.run([sys.executable, "src/evaluate.py", "--ckpt", rl_ckpt, "--harvest", harvest_file], check=True)
    # append harvest to train
    if os.path.isfile(harvest_file):
        shutil.copyfileobj(open(harvest_file, "r"), open("data/train.jsonl", "a"))

    # re‑SFT
    out_ckpt = f"ckpts/sft_cycle{c}"
    subprocess.run([sys.executable, "src/train_sft.py", "--resume", rl_ckpt, "--output", out_ckpt], check=True)
    ckpt = out_ckpt