# **Fraud Detection Tool: Safeguard Against Scams**

## **Overview**
Financial scams targeting Myanmar’s digital users are increasingly sophisticated. This project aims to build a robust tool to **detect and alert** users when a message, advertisement, or post may be fraudulent — especially in the Burmese language and context.  
By combining real-world data collection, low-resource NLP, feature engineering, classical ML and model deployment, the goal is to raise public awareness and reduce scam impact in Myanmar.

This project aims to develop a simple and effective interface that allows users to determine whether a message, post, or advertisement is fraudulent using a machine learning model. The goal is to help the general public, particularly youth, avoid falling into scams, enabling a safer and more productive life.
---
## Key Highlights  
- Focused on Burmese-language scam detection (a low-resource NLP setting)  
- Custom preprocessing pipeline: placeholder normalization (`[URL]`, `[PHONE]`, `[AMOUNT]`, etc.), emoji & hashtag counts, mixed Burmese/English tokens  
- Class-imbalance aware classification (Non-Scam vs Scam vs Potential Scam)  
- Experimentation with multiple algorithms (SVM, MultinomialNB, RandomForest, XGBoost) and TF-IDF + bigrams  
- Modular code: separation of preprocessing, model pipeline, metrics and evaluation  
- Ready for extension: incorporable into a web API / user-interface layer for deployment


## **Features**
- **User-Friendly Interface**: Easy-to-use platform for checking messages, posts, or ads for fraudulence.
- **Machine Learning Powered**: Utilizes a trained model to analyze text and predict potential scams.
- **Real-Time Feedback**: Provides instant results to users after submission.
- **Awareness Campaign**: Educates users on common scam patterns and prevention tips.

---
## Folder Structure  
├── datasets/ # Raw & processed data files
├── burmese_money_scam_classification/ # Main codebase for classification
│ ├── preprocess_text.py # Text cleaning & tokenization module
│ ├── transformer.py # Custom transformers & pipeline definitions
│ ├── metrics.py # Evaluation metrics functions (confusion matrix, ROC-AUC, log-loss)
│ ├── train.py # Model training script (handles splitting, CV, model saving)
│ ├── evaluate.py # Script to evaluate best model on hold-out set
│ └── README.md # Project-specific README
├── requirements.txt # Python dependencies
├── runtime.txt # Runtime environment specification
└── README.md # This top-level project README

## **Use Cases**
- **Youth Awareness**: Helping young individuals identify fraudulent job offers, investment schemes, or phishing messages.
- **General Public Protection**: Assisting everyone in avoiding scams on social media, email, and other platforms.
- **Advertiser and Platform Monitoring**: Helping businesses verify advertisements for legitimacy.

## Methods & Approach
**Data Preprocessing**

  -Replace placeholders: [URL], [PHONE], [AMOUNT], [TID], [HASHTAG], [DATE], [EMAIL], [TELEGRAM]
  -Remove emojis (and optionally count them as features)
  -Count hashtags, punctuation, and standalone numbers (with domain-specific logic)
  -Lowercase normalization, stopword removal (Burmese + English), rare English word replacement with [ENG]
  -Separate tokenization: Burmese tokenization + English tokenization, then combine features

**Feature Engineering & Modeling**

Applied TF-IDF with bigrams alongside standard token features

Considered class imbalance: e.g., stratified train/test/validation splits, weighted/focal loss, sampling techniques

**Models experimented with:**

  -Logistic Regression
  -SVM
  -MultinomialNB
  -RandomForest
  -XGBoost

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, Log Loss

## Deployment Considerations

-Model saved via joblib/pickle for easy loading
-Simple interface (Flask, Streamlit or FastAPI) can be wrapped for real-time classification
-Designed with extendability in mind for additional languages, messaging platforms, or variant scam types
---
## Why This Matters

-Scams in Myanmar increasingly target youth on social platforms and use Burmese/English code-mixing.
-By building tools specifically tailored to Burmese content and vernacular usage, this project fills a critical gap in local digital safety infrastructure.
-The modular design makes it transferable to other contexts or languages with minimal adjustments

## Next Steps & Opportunities

-Expand model to cover more classes: e.g., “Investment Scam”, “Job Scam”, “Phishing”, etc.
-Deploy a real-time Telegram bot or web service for public use.
-Incorporate deep-learning embeddings (e.g., Burmese language models) for richer representation.
-Collect more data (especially for minority classes) to improve performance and generalization.
-Build a public awareness dashboard or visualisation site to promote findings.

## **Members**
1. Nu Wai Thet (MMDT 2024.001)(Project Leader)
2. Myint Myat Aung Zaw (MMDT 2024.044)
3. Kaung Myat Kyaw (MMDT 2024.073)
4. Dr Myo Thida (Advisor)
   
## **Getting Started**

### **Prerequisites**
- **Python 3.8+**
- Required Libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `Flask` (or other framework for the interface)
  - `nltk` (for text preprocessing)
  - `joblib` (for model saving and loading)

Install the dependencies with:
```bash
pip install -r requirements.txt
---
## **Contact**

For any questions, collaboration inquiries or feedback, feel free to reach out to me on GitHub or email: nuwaithet@gmail.com
Follow me on LinkedIn:https://www.linkedin.com/in/nuwai-thet-sophia/
