import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Small helper to read environment variables with an optional default."""
    return os.getenv(name, default)


PROJECT_MARKERS = ("configs", "data", "models")


def _has_project_markers(base: Path) -> bool:
    """Return True if the path looks like the project root (contains known marker dirs)."""
    return any((base / marker).exists() for marker in PROJECT_MARKERS)


def _resolve_project_root() -> str:
    """
    Resolve project root:
    1) MONEYSCAM_PROJECT_ROOT env var if set
    2) current working directory (most reliable when installed as package)
    3) walk up from this file to find a directory with known markers
    4) fallback to current working directory
    """
    env_root = _env("MONEYSCAM_PROJECT_ROOT")
    if env_root:
        return str(Path(env_root).resolve())

    cwd_root = Path.cwd()
    if _has_project_markers(cwd_root):
        return str(cwd_root.resolve())

    for parent in Path(__file__).resolve().parents:
        if _has_project_markers(parent):
            return str(parent)

    return str(cwd_root.resolve())


@dataclass
class Paths:
    project_root: str = _resolve_project_root()
    #print(f"Resolved project root at: {project_root}")
    data_path: str = os.path.join(project_root, "data", "processed", "preprocess_mmdt_tokenizer.xlsx")
    # Embedding model paths (override via env FASTTEXT_MODEL_PATH / WORD2VEC_MODEL_PATH)
    fasttext_path: str = _env("FASTTEXT_MODEL_PATH", os.path.join(project_root, "models", "fasttext_mmdt.model"))
    #print(f"Using FastText model path at: {fasttext_path}")
    word2vec_path: str = _env("WORD2VEC_MODEL_PATH", os.path.join(project_root, "models", "word2vec_mmdt.model"))
    models_dir: str = os.path.join(project_root, "models")
    reports_dir: str = os.path.join(project_root, "reports")


@dataclass
class TrainingConfig:
    vectorizer_type: str = "fasttext"
    cv_folds: int = 5
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    max_features: int = 1000
    use_hyperparameter_search: bool = True


@dataclass
class MLflowConfig:
    tracking_uri: str = _env("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name: str = _env("MLFLOW_EXPERIMENT_NAME", "money-scam-mm")
    register_model_name: Optional[str] = _env("MLFLOW_REGISTER_MODEL_NAME", None)


@dataclass
class InferenceConfig:
    threshold_min_conf: float = 0.6
    threshold_min_margin: float = 0.15
    model_path: str = os.path.join(Paths().models_dir, "best_model_scam_detector.skops")


paths = Paths()
training_config = TrainingConfig()
mlflow_config = MLflowConfig()
inference_config = InferenceConfig()
