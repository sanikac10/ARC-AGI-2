# =============================
# File: src/similarity.py
# Purpose: Grid‑level similarity score for partial reward
# =============================

import numpy as np

def grid_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return cell‑wise IoU (intersection‑over‑union) over exact matches."""
    if pred.shape != gt.shape:
        return 0.0
    match = (pred == gt)
    return float(match.sum()) / match.size