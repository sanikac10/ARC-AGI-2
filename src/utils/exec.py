# =============================
# File: src/utils/exec.py
# Purpose: Utility helpers for prompt‑building, code execution, and grid parsing
# =============================

from __future__ import annotations
import ast, re, json, textwrap, subprocess, tempfile, os, numpy as np

# -----------------------------
# Constants
# -----------------------------
SPECIAL_TOKENS = {
    "VISEXP": "<VISEXP>",
    "REASON": "<REASON>",
    "CODE": "<CODE>",
}

# -----------------------------
# build_prompt
#   • Wraps a task dict (input_grid, optional gt_grid) into a text prompt that
#     the Perception‑LM understands.  You can tweak format later if needed.
# -----------------------------

def build_prompt(task: dict) -> str:
    """Convert a task into the textual prompt expected by the model."""
    grid_json = json.dumps(task["input_grid"])  # keep as canonical JSON
    prompt = (
        f"You are an ARC‑AGI expert. Given the INPUT grid below, produce a pixel‑perfect\n"
        f"description of the OUTPUT grid.\n\n"
        f"INPUT_GRID = {grid_json}\n\n"
        f"Respond in the following template:\n"
        f"{SPECIAL_TOKENS['VISEXP']} your visual explanation\n"
        f"{SPECIAL_TOKENS['REASON']} your reasoning\n"
        f"{SPECIAL_TOKENS['CODE']} python‑like DSL code solving the puzzle\n"
        f"ANSWER_GRID = <describe final grid here as JSON list of lists>"
    )
    return prompt

# -----------------------------
# extract_grid
#   • Extract the JSON grid that the model emits after "ANSWER_GRID =".
#   • Returns np.ndarray or None when parsing fails.
# -----------------------------

def extract_grid(text: str):
    match = re.search(r"ANSWER_GRID\s*=\s*(\[[^\]]+\])", text)
    if not match:
        return None
    try:
        grid = json.loads(match.group(1))
        return np.array(grid)
    except json.JSONDecodeError:
        return None

# -----------------------------
# run_code (optional sandbox)
#   • Executes DSL python code with a given input grid to produce a grid.
#   • Uses a tmp file + subprocess for isolation + 2‑second time‑out.
# -----------------------------

def run_code(code_snippet: str, input_grid):
    """Run generated code safely and return resulting grid (or None on error)."""
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.join(td, "solver.py")
        # Write minimal driver
        body = (
            "import json, numpy as np\n"
            "from typing import *\n\n"  # allow DSL typing
            f"I = {json.dumps(input_grid)}\n"
            f"{code_snippet}\n"
            "print(json.dumps(solve(I)))  # assumes generated func named solve"
        )
        open(fname, "w").write(body)
        try:
            out = subprocess.check_output(["python", fname], timeout=2)
            return np.array(json.loads(out))
        except Exception:
            return None
