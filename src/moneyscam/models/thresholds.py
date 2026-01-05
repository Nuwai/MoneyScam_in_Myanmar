from typing import Dict, List, Tuple
import numpy as np


def apply_unsure_policy(probs: np.ndarray, labels: List[str], min_conf: float, min_margin: float) -> Tuple[str, float, str]:
    """
    Apply uncertainty thresholds: if top confidence is low or margin is small, return 'Potential'.
    Returns tuple of (label, confidence, reason).
    """
    top_idx = int(np.argmax(probs))
    sorted_probs = np.sort(probs)[::-1]
    top_prob = sorted_probs[0]
    second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob

    if top_prob < min_conf:
        return "Potential", float(top_prob), "low_confidence"
    if margin < min_margin:
        return "Potential", float(top_prob), "low_margin"
    return labels[top_idx], float(top_prob), "none"


HIGH_PRECISION = {"min_conf": 0.7, "min_margin": 0.2}
HIGH_RECALL = {"min_conf": 0.5, "min_margin": 0.05}
