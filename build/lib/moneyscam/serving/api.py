from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import preprocess_text
from moneyscam.models.thresholds import apply_unsure_policy, HIGH_PRECISION
from moneyscam.serving.model_loader import ModelLoader
from moneyscam.settings import inference_config


app = FastAPI(title="Money Scam Detector API")
loader = ModelLoader()
model = loader.load()


class PredictRequest(BaseModel):
    text: str
    min_conf: Optional[float] = None
    min_margin: Optional[float] = None


class BatchPredictRequest(BaseModel):
    texts: List[str]
    min_conf: Optional[float] = None
    min_margin: Optional[float] = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probs: dict
    policy_reason: str


def _to_features(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        processed, emoji_count, hashtag_count, punctuation_counts = preprocess_text.preprocess_text(t)
        punct_val = punctuation_counts if isinstance(punctuation_counts, (int, float)) else sum(punctuation_counts.values())
        rows.append(
            {
                "processed_text": processed,
                "emoji_count": emoji_count,
                "hashtag_count": hashtag_count,
                "punctuation_counts": punct_val,
            }
        )
    return pd.DataFrame(rows)


def _predict(texts: List[str], min_conf: Optional[float], min_margin: Optional[float]) -> List[PredictResponse]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df_features = _to_features(texts)
    probs = model.predict_proba(df_features)
    labels = list(model.classes_)
    responses = []
    for row_probs in probs:
        label, conf, reason = apply_unsure_policy(
            row_probs,
            labels=labels,
            min_conf=min_conf or inference_config.threshold_min_conf,
            min_margin=min_margin or inference_config.threshold_min_margin,
        )
        responses.append(
            PredictResponse(
                label=label,
                confidence=conf,
                probs={lbl: float(p) for lbl, p in zip(labels, row_probs)},
                policy_reason=reason,
            )
        )
    return responses


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "0.1.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _predict([req.text], req.min_conf, req.min_margin)[0]


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(req: BatchPredictRequest):
    if len(req.texts) == 0:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    return _predict(req.texts, req.min_conf, req.min_margin)


@app.post("/reload_model")
def reload_model():
    global model
    model = loader.reload()
    return {"status": "reloaded"}
