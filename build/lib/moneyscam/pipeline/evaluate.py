import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def save_confusion_matrix(y_true, y_pred, labels, out_path: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.colorbar(im, ax=ax)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return cm


def save_metrics(metrics: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_classification_report(y_true, y_pred, out_path: str) -> str:
    report = classification_report(y_true, y_pred)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def collect_per_class_metrics(y_true, y_pred, labels) -> Dict[str, Dict[str, float]]:
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return {lbl: report_dict.get(lbl, {}) for lbl in labels}
