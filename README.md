# ARC‑AGI 2 – Iterative LIMO‑Style Pipeline (Perception‑LM‑8B, Full‑SFT)


## 0. High‑Level Flow

```text
┌─────────────── synthetic (847) ────────────────┐
│                Train Dataset                   │
└────────────────────────▲────────────────────────┘
                         │ (full SFT)  ckpt‑0
                         ▼
                 ┌──────────────────┐  bottom‑50% IDs  ┌─────────────────┐
                 │     Policy π₀    │◀────────────────▶│   Test Pool T   │
                 └──────────────────┘    (ARC‑AGI‑2)   └─────────────────┘
                         │  evaluate_base.py               ▲
                         │  log acc‑0                      │
                         ▼                                 │
                ┌──────────────────┐  correct preds        │
                │   Harvest H₁     │──────────┐            │
                └──────────────────┘          ▼            │
                         │           append to train       │
               (Train + H₁) JSONL ◀──────────┘             │
                         │  SFT resume ckpt‑0              │
                         ▼                                 │
         ┌───────────────────────────┐                     │
         │  Policy π₁  (SFT‑full)    │―――――――――――――――――――――┘
         └───────────────────────────┘
                         │  GRPO with RM
                         ▼
         ┌───────────────────────────┐
         │  Policy π₁′ (GRPO fine)   │
         └───────────────────────────┘
                         │ evaluate.py, log acc‑1
                         ▼
              (repeat harvest → π₂…)
```

*Key points*

1. **Training set** is *only* synthetic + harvested correct answers.
2. **Test pool T** = bottom‑half of ARC‑AGI‑2 tasks, never in training until harvested.
3. At each cycle we measure and **log accuracy before and after GRPO**.

---

## 1. Data Buckets & Conventions

| File                        | Purpose                                |
| --------------------------- | -------------------------------------- |
| `data/train.jsonl`          | Synthetic manual+generated (847)       |
| `data/harvest_cycleN.jsonl` | Correct answers from cycle N           |
| `data/test_pool.jsonl`      | Bottom‑50 % ARC‑AGI‑2 tasks (held‑out) |
| `results/accuracy_log.csv`  | Cycle‑wise accuracy snapshots          |

### Test‑Pool Row

```json
{
  "id": "arc2_326",
  "input_grid": "…",
  "gt_grid": "…"
}
```

---



## 2. Reward Model – unchanged logic

*Partial credit via `similarity.grid_score` still in effect.*


## 3. Folder Structure (final)

```
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
```

---

## 4. Accuracy Tracking Cheat‑Sheet

| Cycle | Phase      | CSV label   |
| ----- | ---------- | ----------- |
| 0     | Pre‑RLHF   | `baseline`  |
| N     | After GRPO | `post_grpo` |

Plot `accuracy_log.csv` over cycles to monitor convergence.

---

*All changes integrated: bottom‑50 % test pool, accuracy snapshots before/after each training round, and consistent harvest‑train loop.*
