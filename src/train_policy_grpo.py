# =============================
# File: src/train_policy_grpo.py
# Purpose: RLHF fine‑tune with Generalized Reward‑Policy Optimization (PPO‑style)
# =============================

import argparse, json
from datasets import load_dataset
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from similarity import grid_score
from utils.exec import build_prompt, extract_grid

parser = argparse.ArgumentParser(); parser.add_argument("--cycle", required=True); args = parser.parse_args()
cycle = int(args.cycle)

policy_ckpt = f"ckpts/sft_full" if cycle == 1 else f"ckpts/sft_cycle{cycle-1}"
reward_ckpt = "ckpts/reward"

policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_ckpt, torch_dtype="auto", device_map="auto")
reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_ckpt, torch_dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(policy_ckpt)

# simple loader: sample prompts from train set
prompts = [build_prompt(j) for j in load_dataset("json", data_files="data/train.jsonl")["train"]]
loader = prompts[:128]  # truncate for demo

ppo = PPOTrainer(policy, reward_model, kl_coeff=0.05)
for p in loader:
    tokens = tokenizer(p, return_tensors="pt").to(policy.device)
    response, _, _ = ppo.step(tokens, max_new_tokens=512)
    # (TRL handles reward via reward_model internally)

policy.save_pretrained(f"ckpts/rl_cycle{cycle}")
