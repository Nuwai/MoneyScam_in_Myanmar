import numpy as np
from moneyscam.models.thresholds import apply_unsure_policy


def test_low_confidence_triggers_potential():
    probs = np.array([0.3, 0.4, 0.3])
    label, conf, reason = apply_unsure_policy(probs, ["a", "b", "c"], min_conf=0.6, min_margin=0.1)
    assert label == "Potential"
    assert reason == "low_confidence"


def test_low_margin_triggers_potential():
    probs = np.array([0.51, 0.49, 0.0])
    label, _, reason = apply_unsure_policy(probs, ["a", "b", "c"], min_conf=0.5, min_margin=0.05)
    assert label == "Potential"
    assert reason == "low_margin"


def test_confident_prediction_passes():
    probs = np.array([0.9, 0.05, 0.05])
    label, _, reason = apply_unsure_policy(probs, ["a", "b", "c"], min_conf=0.5, min_margin=0.05)
    assert label == "a"
    assert reason == "none"
