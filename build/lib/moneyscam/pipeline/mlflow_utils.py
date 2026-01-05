import hashlib
import json
import os
from typing import Dict, Optional

import mlflow


def start_mlflow_run(experiment_name: str, run_name: str):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict):
    mlflow.log_params(params)


def log_metrics(metrics: Dict):
    mlflow.log_metrics(metrics)


def log_artifacts(artifact_dir: str):
    mlflow.log_artifacts(artifact_dir)


def log_model_artifact(file_path: str, artifact_path: str = "model"):
    mlflow.log_artifact(file_path, artifact_path=artifact_path)


def tag_best_run():
    mlflow.set_tag("best_run", "true")


def dataset_hash(path: str) -> str:
    """Compute a simple hash for dataset lineage logging."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
