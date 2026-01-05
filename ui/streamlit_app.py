import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import preprocess_text  # noqa: E402
from moneyscam.models.thresholds import apply_unsure_policy
from moneyscam.serving.model_loader import ModelLoader
from moneyscam.settings import inference_config


API_URL_ENV = os.getenv("API_URL")  # If set, can use remote API; otherwise run local inference

LABEL_MAP = {
    "0": "Non-scam",
    "1": "Scam",
    "2": "Potential scam",
    "Potential": "Potential scam",
}

SUGGESTIONS = {
    "0": [
        "No scam indicators detected. Stay alert and verify important details.",
        "Looks safe. Avoid sharing sensitive info unless you trust the source.",
    ],
    "1": [
        "High-risk scam. Do not click links or share personal/financial info.",
        "Scam detected. Block the sender and report if possible.",
    ],
    "2": [
        "This looks suspicious. Verify with official sources before acting.",
        "Potential scam. Be cautious with links, attachments, or urgent requests.",
    ],
    "Potential": [
        "This looks suspicious. Verify with official sources before acting.",
        "Potential scam. Be cautious with links, attachments, or urgent requests.",
    ],
}

POLICY_REASON_TEXT = {
    "none": "Model is confident and margin is sufficient.",
    "low_confidence": "Top class confidence is below the minimum threshold.",
    "low_margin": "Top class is too close to the next-best class (low margin).",
}


@st.cache_resource
def load_local_model():
    loader = ModelLoader()
    return loader.load()


def format_label(raw_label: str) -> str:
    return LABEL_MAP.get(str(raw_label), str(raw_label))


def suggestion_for(raw_label: str) -> str:
    opts = SUGGESTIONS.get(str(raw_label)) or SUGGESTIONS.get("2")
    return random.choice(opts)


def policy_text(reason: str) -> str:
    return POLICY_REASON_TEXT.get(reason, f"Model policy: {reason}")


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


def predict_local(text: str, min_conf: float, min_margin: float) -> Dict:
    model = load_local_model()
    df = _to_features([text])
    probs = model.predict_proba(df)[0]
    labels = [str(lbl) for lbl in model.classes_]
    label, conf, reason = apply_unsure_policy(
        probs,
        labels=labels,
        min_conf=min_conf or inference_config.threshold_min_conf,
        min_margin=min_margin or inference_config.threshold_min_margin,
    )
    return {
        "label": str(label),
        "confidence": float(conf),
        "policy_reason": reason,
        "probs": {lbl: float(p) for lbl, p in zip(labels, probs)},
    }


def predict_remote(text: str, min_conf: float, min_margin: float, api_url: str) -> Dict:
    resp = requests.post(f"{api_url}/predict", json={"text": text, "min_conf": min_conf, "min_margin": min_margin}, timeout=30)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Money Scam Detector", page_icon="🔍", layout="wide")
st.title("Money Scam Detector")
st.caption("FastAPI backend + Streamlit frontend | Uses mmdt-tokenizer preprocessing")

with st.sidebar:
    st.header("Settings")
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.01)
    min_margin = st.slider("Min margin", 0.0, 1.0, 0.15, 0.01)
    mode = st.radio("Inference mode", ["Local model", "Remote API"], index=0 if not API_URL_ENV else 1)
    api_url = st.text_input("API URL (for Remote API)", API_URL_ENV or "http://localhost:8000")
    if mode == "Remote API" and not api_url:
        st.warning("Provide API URL for remote mode.")

text_input = st.text_area("Paste a message to classify", height=160)

def run_predict(text: str) -> Dict:
    if mode == "Remote API":
        return predict_remote(text, min_conf, min_margin, api_url)
    return predict_local(text, min_conf, min_margin)


if st.button("Classify") and text_input.strip():
    with st.spinner("Scoring..."):
        try:
            result = run_predict(text_input)
            display_label = format_label(result["label"])
            friendly_probs = {format_label(k): v for k, v in result["probs"].items()}
            st.subheader(f"{display_label} (conf {result['confidence']:.2f})")
            st.write(policy_text(result["policy_reason"]))
            st.write(suggestion_for(result["label"]))
            st.bar_chart(friendly_probs)
            st.json(result)
        except Exception as e:
            st.error(f"Request failed: {e}")

st.divider()
st.subheader("Batch scoring (CSV upload)")
uploaded = st.file_uploader("Upload CSV (include 'text' column or any columns)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    if "text" in df.columns:
        texts = df["text"].fillna("").astype(str).tolist()
    else:
        st.warning("No 'text' column found. Concatenating all columns per row for scoring.")
        texts = (
            df.fillna("")
            .astype(str)
            .apply(lambda row: " ".join([v for v in row.values if v]).strip(), axis=1)
            .tolist()
        )
    rows = []
    with st.spinner("Scoring batch..."):
        for t in texts:
            try:
                r = run_predict(t)
                friendly_probs = {format_label(k): v for k, v in r.get("probs", {}).items()}
                rows.append(
                    {
                        "input": t[:80] + ("..." if len(t) > 80 else ""),
                        "label": format_label(r.get("label")),
                        "confidence": r.get("confidence"),
                        "policy_reason": policy_text(r.get("policy_reason", "")),
                        "suggestion": suggestion_for(r.get("label")),
                        "probs": json.dumps(friendly_probs),
                    }
                )
            except Exception as e:
                rows.append({"input": t, "label": "error", "confidence": None, "policy_reason": str(e), "suggestion": ""})
    st.write(pd.DataFrame(rows))
