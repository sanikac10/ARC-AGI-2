# Annotating-ARC-AGI-2
Just annotation code for arc-agi-2
# File structure:
arc‑agi2‑percept/
├── data/
│   ├── train.jsonl
│   ├── test_pool.jsonl
│   ├── harvest_cycle1.jsonl …
│   └── rm_pairs.jsonl
├── results/accuracy_log.csv
├── ckpts/
│   ├── sft_full/ …
│   ├── rl_cycle1/ …
│   └── sft_cycle1/ …
├── deepspeed/zero3.json
├── src/
│   ├── data_prep.py
│   ├── similarity.py
│   ├── train_sft.py
│   ├── train_reward_model.py
│   ├── train_policy_grpo.py
│   ├── evaluate_base.py
│   ├── evaluate.py
│   └── run_experiment.py
└── README.md