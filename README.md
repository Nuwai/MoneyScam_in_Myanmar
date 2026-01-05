# **Fraud Detection Tool: Safeguard Against Scams**

End-to-end ML project for detecting Burmese-language scam content. Includes data prep, model training and selection, experiment tracking with MLflow, serving via FastAPI, and a Streamlit UI. Demonstrates MLOps skills: reproducible pipelines, experiment management, model artifact handling, containerization, and deployment-ready APIs.

---

## Highlights
- Three-class classifier: Non-scam (0), Scam (1), Potential scam (2)
- Custom preprocessing for Burmese/English, placeholder normalization, emoji/hashtag/punctuation features
- Multiple vectorizers (TF-IDF, Word2Vec, FastText) and classical models; best performer: TF-IDF + SVM
- Stratified CV, class-imbalance-aware training, hyperparameter tuning
- MLflow for runs, metrics, and artifacts (.skops models)
- Serving via FastAPI (uvicorn); Streamlit UI can use local model or remote API
- Docker Compose option for API + MLflow; ready to deploy to a PaaS/VM

## Stack
- Python, scikit-learn, pandas, numpy
- Gensim (Word2Vec/FastText), skops (model persistence)
- MLflow (tracking and artifacts)
- FastAPI + Uvicorn (backend)
- Streamlit (frontend)
- Docker/Docker Compose (orchestration)

## Project layout (key paths)
- `src/moneyscam/` – core package (settings, models, training pipeline, serving)
- `configs/train.yaml` – training config
- `models/` – saved embeddings and classifiers (.skops)
- `mlruns/` – local MLflow tracking store (default)
- `ui/streamlit_app.py` – Streamlit app (local or remote inference)
- `docker-compose.yml` – API + MLflow services

## Training
Run cross-validated training with defaults or override via config:
```bash
make train
# or
uv run python -m moneyscam.pipeline.train_classical --config configs/train.yaml
```
Outputs: metrics, confusion matrix, per-class scores in `reports/`, best model saved to `models/<ModelName>/best_model_*.skops`, and logged to MLflow artifacts.

## Experiments (MLflow)
- Tracking URI (local default): `file:./mlruns`
- Experiment name: `money-scam-mm`
- View UI: `uv run mlflow ui --backend-store-uri file:./mlruns --port 5000`
- Compare runs and download artifacts from the MLflow UI; models logged under `model/`.

## Serving (FastAPI)
Run the API locally:
```bash
uv run uvicorn moneyscam.serving.api:app --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET /health`
- `GET /version`
- `POST /predict` (`{"text": "...", "min_conf": 0.6, "min_margin": 0.15}`)
- `POST /batch_predict`
- `POST /reload_model`

## Frontend (Streamlit)
The Streamlit app can run standalone (loads local model) or call a remote API.
```bash
uv run streamlit run ui/streamlit_app.py
```
- Sidebar: select inference mode (Local model or Remote API via `API_URL`)
- Shows mapped labels, confidence, policy reason, suggestions, and probabilities
- Batch CSV scoring; concatenates all columns if no `text` column exists

## Docker / Compose
Bring up API + MLflow together:
```bash
docker-compose up --build
```
Ports:
- API: 8000
- MLflow UI: 5000
Volumes mount `models/`, `data/`, `configs/`, and `mlruns/`.

## Model artifacts
- Stored in `models/` (e.g., `models/SVM/best_model_tfidf.skops`)
- Logged to MLflow under artifact path `model/`
- Load with skops:
```python
import skops.io as sio
model = sio.load("models/SVM/best_model_tfidf.skops", trusted=None)
```
For MLflow registry, log with `mlflow.sklearn.log_model(..., registered_model_name=...)`.

## Data and preprocessing
- Placeholder normalization: `[URL]`, `[PHONE]`, `[AMOUNT]`, `[EMAIL]`, etc.
- Emoji/hashtag/punctuation counts as numeric features
- Burmese + English tokenization, rare English normalization
- Stratified splits; class weights and CV to handle imbalance

## Results (best model: TF-IDF + SVM)
- Accuracy: 0.8897
- F1 (weighted): 0.8935
- Precision: 0.9019
- Recall: 0.8897
- ROC-AUC: 0.9713
- Potential scam class remains the hardest (precision ~0.64); future work: more data, better class balance, richer embeddings.

## Next steps
- Host FastAPI behind HTTPS and point Streamlit’s `API_URL` to it
- Add registry-based model promotion (MLflow model registry)
- Experiment with Burmese transformers for embeddings
- Ship a Telegram/web bot for public use; build awareness dashboards
- Improve Potential-scam precision: more data, targeted augmentation, class-aware loss
- Add CI/CD (tests + lint + model check), and infra-as-code for reproducible deploys
- Add user feedback loop (flag/label corrections) to drive periodic retraining

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
```
Optional dev tools: `make install`, `make lint`, `make test`.

## Contact
Nu Wai Thet — nuwaithet@gmail.com — https://www.linkedin.com/in/nuwai-thet-sophia/
