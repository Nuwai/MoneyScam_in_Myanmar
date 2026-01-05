# **Fraud Detection Tool: Safeguard Against Scams**

## **Overview**
Financial scams targeting Myanmar’s digital users are increasingly sophisticated. This project builds a tool to detect and alert when a message, advertisement, or post may be fraudulent, especially in the Burmese language and context. It combines real-world data collection, low-resource NLP, feature engineering, classical ML, experiment tracking, and deployment.

## Key Highlights
- Burmese-language scam detection (low-resource NLP)
- Custom preprocessing: placeholder normalization (`[URL]`, `[PHONE]`, `[AMOUNT]`, etc.), emoji/hashtag/punctuation counts, mixed Burmese/English tokens
- Class-imbalance aware (Non-Scam vs Scam vs Potential Scam)
- Multiple vectorizers (TF-IDF, Word2Vec, FastText) and models (SVM, MultinomialNB, RandomForest, XGBoost)
- Stratified CV, hyperparameter tuning; best: TF-IDF + SVM
- MLflow for runs/metrics/artifacts; models saved as `.skops`
- Serving via FastAPI; Streamlit UI can run standalone (local model) or call the API
- Docker Compose option for API + MLflow

## **Features**
- **User-Friendly Interface**: Streamlit UI for checking messages, posts, or ads.
- **Machine Learning Powered**: Trained model predicts scams vs non-scams vs potential scams.
- **Real-Time Feedback**: Instant responses with confidence and policy reason.
- **Awareness**: Suggestions to educate users on scam patterns.

## Folder Structure
```
burmese_money_scam_classification/
├── configs/                # Training configs (e.g., train.yaml)
├── data/                   # Data assets
├── models/                 # Saved embeddings and classifiers (.skops)
├── mlruns/                 # MLflow tracking store (local)
├── notebooks/              # Analysis and MLflow helper notebooks
├── reports/                # Evaluation outputs (metrics, confusion matrix)
├── result_log_files/       # Legacy training logs
├── src/                    # Core package (moneyscam)
│   └── moneyscam/
│       ├── models/         # Pipelines, embeddings, thresholds
│       ├── pipeline/       # Training/evaluation scripts
│       ├── serving/        # FastAPI app + model loader
│       └── settings.py     # Paths/config
├── ui/                     # Streamlit app (local or remote API)
├── docker-compose.yml      # API + MLflow stack
├── Dockerfile.api
├── Makefile                # install/train/api/streamlit/lint/test
├── preprocess_text.py      # Preprocessing utility
└── README.md
```

## **Use Cases**
- Youth awareness for suspicious job offers, investment schemes, phishing
- General public protection on social media, email, messaging
- Advertiser/platform monitoring for legitimacy checks

## **Dataset Information**
- Sources: public social posts (Facebook/Telegram), victim reports, common scam groups, personal messages
- Characteristics: legitimate ads from banks/official sources; diverse scam categories; synthetic short SMS augmentation
- Labeling: Burmese native speakers; three classes — Fraudulent (Red), Non-Fraudulent (Green), Potential Scam (Yellow)

## Methods & Approach
**Data Preprocessing**
- Replace placeholders: `[URL]`, `[PHONE]`, `[AMOUNT]`, `[TID]`, `[HASHTAG]`, `[DATE]`, `[EMAIL]`, `[TELEGRAM]`
- Remove/optionally count emojis; count hashtags/punctuation/numbers
- Lowercase, stopword removal (Burmese + English), rare English → `[ENG]`
- Separate tokenization (Burmese + English), then combine features

**Feature Engineering & Modeling**
- TF-IDF, FastText, Word2Vec + token/structure features
- Imbalance handling: stratified splits, class weights, cross-validation
- Models tried: Logistic Regression, SVM, MultinomialNB, RandomForest, XGBoost

## Model Training Strategy
1. Stratified K-Fold CV (preserves imbalance)
2. Train/Validation/Test splits (80/20; train further split for CV)
3. Model selection via CV metrics (Accuracy, F1, Precision, Recall, ROC-AUC, Log Loss)
4. Hyperparameter tuning (where applicable)
5. Final train on full pool; test on hold-out; artifacts + metrics saved

## Feature Extraction Methods
| **Method**   | **Representation Type** | **Learns from Context?** | **Handles Misspellings?** | **Captures Subwords?** | **Best For**                         |
| :----------- | :---------------------- | :----------------------: | :-----------------------: | :--------------------: | :----------------------------------- |
| **TF-IDF**   | Word frequency-based    |             No            |             No             |            No           | Simple keyword-based classification  |
| **Word2Vec** | Full-word vectors       |             Yes            |             No             |            No          | Understanding semantic relationships |
| **FastText** | Subword-based vectors   |             Yes            |             Yes            |            Yes           | Handling typos & unseen words        |

Key Takeaways
- TF-IDF: Simple/effective for bag-of-words; limited semantics
- Word2Vec: Captures context; weaker on misspellings
- FastText: Robust to noisy Burmese text via subwords/typos

## Experimental Findings
- TF-IDF with word tokenization performed best overall.
- SVM consistently outperformed Naive Bayes, Random Forest, XGBoost.
- Word2Vec and FastText were limited by data size.

## 🏆 Final Model — SVM (Test Set Evaluation)
| **Metric** | **Score** |
| :--------- | :-------: |
| Accuracy   |   0.8897  |
| F1-Score   |   0.8935  |
| Precision  |   0.9019  |
| Recall     |   0.8897  |
| ROC-AUC    |   0.9713  |

Class-wise Performance
| **Class**          | **Precision** | **Recall** | **F1-Score** |
| :----------------- | :-----------: | :--------: | :----------: |
| **Potential Scam** |   **0.64**    |    0.84    |     0.72     |
| **Non-Scam**       |      0.95     |    0.91    |     0.93     |
| **Scam**           |      0.93     |    0.87    |     0.90     |

**Insight:** Overall performance is strong (AUC ≈ 0.97). Potential Scam precision (0.64) is hardest due to overlap with legitimate and confirmed scam text. Next steps: better class balance, richer embeddings, domain-specific cues.

## Deployment Considerations
- Models saved as `.skops`; embeddings in `models/`
- FastAPI backend (uvicorn) for `/predict`; MLflow for artifacts/runs
- Streamlit UI: local inference or remote API via `API_URL`
- Docker Compose: run API + MLflow together; ready for PaaS/VM deploy

## Why This Matters
- Scams target Burmese youth with code-mixed content; few tailored tools exist.
- Filling a gap in local digital safety; modular design is transferable to other languages/platforms.

## Next Steps & Opportunities
- More classes (Investment, Job, Phishing, etc.)
- Real-time bot/web service for public use
- Explore Burmese transformers for embeddings
- More data for minority classes; better balance
- Awareness dashboard/visualization

## **Members**
1. Nu Wai Thet (MMDT 2024.001) — Project Leader
2. Myint Myat Aung Zaw (MMDT 2024.044) — Data annotation
3. Kaung Myat Kyaw (MMDT 2024.073) — Data annotation
4. Dr Myo Thida — Advisor

Project Presentation slide: Burmese money scam_summary.pptx

## **Getting Started**
- Python 3.8+
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `FastAPI`/`uvicorn`, `nltk`, `joblib`, `mlflow`, `streamlit`

Install:
```bash
pip install -r requirements.txt
```

## **License**
MIT

## **Contact**
Email: nuwaithet@gmail.com  
LinkedIn: https://www.linkedin.com/in/nuwai-thet-sophia/
