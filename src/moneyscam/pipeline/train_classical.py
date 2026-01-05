import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from skops.io import dump as skops_dump
from sklearn.feature_extraction.text import TfidfVectorizer

from moneyscam.logging_config import setup_logging
from moneyscam.models.classical import get_classification_pipeline
from moneyscam.pipeline import evaluate
from moneyscam.pipeline import mlflow_utils
from moneyscam.settings import paths, training_config, mlflow_config


logger = setup_logging()


@dataclass
class TrainSettings:
    vectorizer_type: str = training_config.vectorizer_type
    cv_folds: int = training_config.cv_folds
    test_size: float = training_config.test_size
    val_size: float = training_config.val_size
    random_state: int = training_config.random_state
    models: Optional[List[str]] = None
    use_hyperparameter_search: bool = training_config.use_hyperparameter_search
    run_name: Optional[str] = None
    data_path: str = paths.data_path


def load_config(config_path: str) -> TrainSettings:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return TrainSettings(**cfg.get("training", {}))


def build_idf_mapping(text_series: pd.Series) -> Dict[str, float]:
    tfidf_for_idf = TfidfVectorizer(tokenizer=str.split, lowercase=False)
    tfidf_for_idf.fit(text_series)
    vocab = tfidf_for_idf.vocabulary_
    idf_vals = tfidf_for_idf.idf_
    return {tok: idf_vals[idx] for tok, idx in vocab.items()}


def hyperparameter_tuning(pipeline, model_name: str, X_val, y_val):
    param_grids = {
        "Logistic Regression": {
            'classifier__C': np.logspace(-3, 2, 6)  # [0.001, 0.01, 0.1, 1, 10, 100]
        },
        "Decision Tree": {
            'classifier__max_depth': [5, 10, 20, 30, None],
            'classifier__min_samples_split': [2, 3, 5, 8, 10]
        },
        "Random Forest": {
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__max_depth': [10, 20, 30]
        },
        "XGBoost": {
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 10]
        },
        "SVM": {
        'classifier__C': [0.001, 0.01, 0.1,0.5, 1, 10],  # Finer tuning, avoid overfitting
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1,0.5, 1],  
        'classifier__degree': [2, 3, 4] 
        },
        "Naive Bayes": {}  # No hyperparameter tuning needed
        }
    if model_name not in param_grids:
        return pipeline
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grids[model_name],
        scoring="f1_weighted",
        cv=3,
        n_iter=5,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_val, y_val)
    logger.info("Best hyperparameters for %s: %s", model_name, search.best_params_)
    return search.best_estimator_


def evaluate_fold(pipeline, X_val_fold, y_val_fold) -> Tuple[float, float, float]:
    y_pred = pipeline.predict(X_val_fold)
    y_prob = pipeline.predict_proba(X_val_fold)
    fold_f1 = f1_score(y_val_fold, y_pred, average="weighted")
    fold_auc = roc_auc_score(y_val_fold, y_prob, multi_class="ovr")
    fold_logloss = log_loss(y_val_fold, y_prob)
    return fold_f1, fold_auc, fold_logloss


def train_and_evaluate(df: pd.DataFrame, cfg: TrainSettings):
    X = df[["processed_text", "emoji_count", "hashtag_count", "punctuation_counts"]]
    y = df["label"]

    class_counts = {label: int((y == label).sum()) for label in np.unique(y)}
    logger.info("Class distribution: %s", class_counts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    X_train_pool, X_val, y_train_pool, y_val = train_test_split(
        X_train, y_train, test_size=cfg.val_size, random_state=cfg.random_state, stratify=y_train
    )

    idf_mapping = None
    if cfg.vectorizer_type in {"word2vec", "fasttext"}:
        idf_mapping = build_idf_mapping(X_train_pool["processed_text"])
        logger.info("Built IDF mapping for %s tokens", len(idf_mapping))

    if cfg.models:
        models = cfg.models
    else:
        models = ["SVM"] #["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM", "Naive Bayes"]
        if cfg.vectorizer_type in {"word2vec", "fasttext"}:
            models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    best_model = None
    best_model_name = ""
    best_score = -1.0
    model_scores: Dict[str, float] = {}

    for model_type in models:
        run_name = cfg.run_name or f"{cfg.vectorizer_type}_{model_type}_v1"
        with mlflow_utils.start_mlflow_run(mlflow_config.experiment_name, run_name):
            mlflow_utils.log_params(
                {
                    "vectorizer_type": cfg.vectorizer_type,
                    "model_type": model_type,
                    "cv_folds": cfg.cv_folds,
                    "random_state": cfg.random_state,
                    "data_hash": mlflow_utils.dataset_hash(cfg.data_path) if os.path.exists(cfg.data_path) else "unknown",
                }
            )

            fold_f1_scores = []
            fold_auc_scores = []
            fold_logloss_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_pool, y_train_pool)):
                X_train_fold, X_val_fold = X_train_pool.iloc[train_idx], X_train_pool.iloc[val_idx]
                y_train_fold, y_val_fold = y_train_pool.iloc[train_idx], y_train_pool.iloc[val_idx]
                pipeline = get_classification_pipeline(model_type, vectorizer_type=cfg.vectorizer_type, idf_mapping=idf_mapping)
                pipeline.fit(X_train_fold, y_train_fold)
                fold_f1, fold_auc, fold_logloss = evaluate_fold(pipeline, X_val_fold, y_val_fold)
                fold_f1_scores.append(fold_f1)
                fold_auc_scores.append(fold_auc)
                fold_logloss_scores.append(fold_logloss)
                logger.info("Fold %s %s | F1 %.4f | AUC %.4f | LogLoss %.4f", fold + 1, model_type, fold_f1, fold_auc, fold_logloss)

            avg_f1 = float(np.mean(fold_f1_scores))
            avg_auc = float(np.mean(fold_auc_scores))
            avg_logloss = float(np.mean(fold_logloss_scores))
            model_scores[model_type] = avg_f1
            mlflow_utils.log_metrics({"weighted_f1_cv_mean": avg_f1, "roc_auc_cv_mean": avg_auc, "log_loss_cv_mean": avg_logloss})

            if avg_f1 > best_score:
                best_score = avg_f1
                best_model_name = model_type
                best_model = get_classification_pipeline(model_type, vectorizer_type=cfg.vectorizer_type, idf_mapping=idf_mapping)

    if best_model is None:
        raise RuntimeError("No model was trained.")

    if cfg.vectorizer_type == "tfidf" and cfg.use_hyperparameter_search:
        best_model = hyperparameter_tuning(best_model, best_model_name, X_val, y_val)

    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)

    metrics = {
        "accuracy_test": accuracy_score(y_test, y_pred_test),
        "weighted_f1_test": f1_score(y_test, y_pred_test, average="weighted"),
        "macro_f1_test": f1_score(y_test, y_pred_test, average="macro"),
        "precision_test": precision_score(y_test, y_pred_test, average="weighted"),
        "recall_test": recall_score(y_test, y_pred_test, average="weighted"),
        "roc_auc_test": roc_auc_score(y_test, y_prob_test, multi_class="ovr"),
        "log_loss_test": log_loss(y_test, y_prob_test),
    }
    per_class = evaluate.collect_per_class_metrics(y_test, y_pred_test, labels=np.unique(y).tolist())
    metrics.update({f"{lbl}_precision": v.get("precision", 0) for lbl, v in per_class.items()})
    metrics.update({f"{lbl}_recall": v.get("recall", 0) for lbl, v in per_class.items()})
    metrics.update({f"{lbl}_f1": v.get("f1-score", 0) for lbl, v in per_class.items()})

    artifacts_dir = os.path.join(paths.reports_dir, best_model_name)
    os.makedirs(artifacts_dir, exist_ok=True)
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    report_path = os.path.join(artifacts_dir, "classification_report.txt")
    evaluate.save_confusion_matrix(y_test, y_pred_test, labels=np.unique(y).tolist(), out_path=cm_path)
    evaluate.save_metrics(metrics, metrics_path)
    evaluate.save_classification_report(y_test, y_pred_test, report_path)

    model_dir = os.path.join(paths.models_dir, best_model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"{best_model_name}_{cfg.vectorizer_type}.skops"
    model_path = os.path.join(model_dir, model_name)
    skops_dump(best_model, model_path)

    with mlflow_utils.start_mlflow_run(mlflow_config.experiment_name, f"{cfg.vectorizer_type}_{best_model_name}_best") as run:
        mlflow_utils.log_params(
            {
                "vectorizer_type": cfg.vectorizer_type,
                "model_type": best_model_name,
                "cv_folds": cfg.cv_folds,
                "random_state": cfg.random_state,
                "data_hash": mlflow_utils.dataset_hash(cfg.data_path) if os.path.exists(cfg.data_path) else "unknown",
            }
        )
        mlflow_utils.log_metrics(metrics)
        mlflow_utils.log_artifacts(artifacts_dir)
        mlflow_utils.log_model_artifact(model_path)
        mlflow_utils.tag_best_run()

    logger.info("Best model %s with CV F1 %.4f | Test weighted F1 %.4f", best_model_name, best_score, metrics["weighted_f1_test"])
    return best_model, metrics, model_path


def main(config_path: Optional[str] = None):
    cfg = load_config(config_path) if config_path else TrainSettings()
    df = pd.read_excel(cfg.data_path)
    train_and_evaluate(df, cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train classical models with MLflow logging.")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML.")
    args = parser.parse_args()
    main(args.config)
