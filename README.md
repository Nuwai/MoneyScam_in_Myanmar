# **Fraud Detection Tool: Safeguard Against Scams**

## **Overview**
Financial scams targeting Myanmar’s digital users are increasingly sophisticated. This project aims to build a robust tool to **detect and alert** users when a message, advertisement, or post may be fraudulent, especially in the Burmese language and context.  
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
```
├── burmese_money_scam_classification/        # Main project package
│   ├── __pycache__/                          # Cached Python bytecode
│   ├── data/                                 # Processed and intermediate data files
│   ├── models/                               # Saved model files (Word2Vec, FastText, etc.)
│   ├── result_log_files/                     # Training logs and evaluation results
│   ├── Burmese Money Scam Classification report summary.docx  # Project summary report
│   ├── Burmese money scam_summary.pptx       # Presentation summarizing findings and workflow
│   ├── moneyscam_literature_reivew.pptx      # Literature review presentation
│   ├── fasttext.model                        # Trained FastText embedding model
│   ├── money_scam_EDA.ipynb                  # Exploratory Data Analysis notebook
│   ├── preprocess_text.py                    # Text preprocessing and tokenization module
│   ├── streamlit_deployment.py               # Streamlit app for model deployment
│   ├── transformer_pipeline.py               # Custom transformers and ML pipeline
│   ├── word2vec.model                        # Trained Word2Vec embedding model
│   └── word2vwc_fasttext_vectorizer.ipynb    # Embedding training & comparison notebook
│
├── datasets/                                 # Raw and external dataset files
│
└── README.md                                 # Top-level project documentation
```

## **Use Cases**
- **Youth Awareness**: Helping young individuals identify fraudulent job offers, investment schemes, or phishing messages.
- **General Public Protection**: Assisting everyone in avoiding scams on social media, email, and other platforms.
- **Advertiser and Platform Monitoring**: Helping businesses verify advertisements for legitimacy.

---

## **Dataset Information**
**Data Sources**

Data was collected from:
- Public social media posts (Facebook, Telegram)
- Victim-shared experiences
- Common scam groups
- Personal messages sent to mobile phones

**Data Characteristics**

- Due to the scarcity of scam-related Burmese text data, we ensured:
- Collection of legitimate advertisements from banks and official sources
- Inclusion of diverse scam categories (e.g., gambling, investment fraud, fake job postings)
- Generation of synthetic short SMS scam messages to augment the dataset

**Dataset Distribution**

<img width="589" height="455" alt="Image" src="https://github.com/user-attachments/assets/39f94ea2-993c-4ac3-88ed-8e2767d50e80" />

**Labeling:**

Manual labeling by Burmese native speakers
Three-class labeling system:
- Fraudulent (Red Flag): Confirmed scams
- Non-Fraudulent (Green Flag): Legitimate messages
- Potential Scam (Yellow Flag): Suspicious, needs verification

## Methods & Approach
**Data Preprocessing**

    -Replace placeholders: [URL], [PHONE], [AMOUNT], [TID], [HASHTAG], [DATE], [EMAIL], [TELEGRAM]
    -Remove emojis (and optionally count them as features)
    -Count hashtags, punctuation, and standalone numbers (with domain-specific logic)
    -Lowercase normalization, stopword removal (Burmese + English), rare English word replacement with [ENG]
    -Separate tokenization: Burmese tokenization + English tokenization, then combine features

**Feature Engineering & Modeling**

    -Applied TF-IDF, FastText, Word2Vec with bigrams alongside standard token features
    -Considered class imbalance: e.g., stratified train/test/validation splits, weighted/focal loss, sampling techniques

**Models experimented with:**

    -Logistic Regression
    -SVM
    -MultinomialNB
    -RandomForest
    -XGBoost

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, Log Loss

## Model Training Strategy
1. Stratified K-Fold Cross-Validation

    - Maintains the original class distribution in each fold
    - Helps handle imbalanced data
    - Ensures fair and consistent model evaluation

2. Train / Validation / Test Split

    Dataset split:
    - Train: 80%
    - Test: 20%
    
    The Train set was further split into:
    - Train Pool: 80%
    - Validation: 20%

3. Cross-Validation for Model Selection

    - Applied Stratified K-Fold on the Train Pool
    - Models evaluated using:
        - Accuracy, F1-Score, Precision, Recall, ROC-AUC, and Log Loss
    - Best model selected based on F1-Score (balance between precision & recall)

4. Hyperparameter Tuning

    - Tuned the selected model on the Validation Set
    - Objective: Optimize performance while avoiding data leakage

5. Final Training & Testing

    - Trained the best model on the full Training Pool
    - Evaluated final performance on the unseen Test Set
    - Produced the final performance report below

## Feature Extraction Methods
Use three feature extraction methods:
| **Method**   | **Representation Type** | **Learns from Context?** | **Handles Misspellings?** | **Captures Subwords?** | **Best For**                         |
| :----------- | :---------------------- | :----------------------: | :-----------------------: | :--------------------: | :----------------------------------- |
| **TF-IDF**   | Word frequency-based    |             No            |             No             |            No           | Simple keyword-based classification  |
| **Word2Vec** | Full-word vectors       |             Yes            |             No             |            No          | Understanding semantic relationships |
| **FastText** | Subword-based vectors   |             Yes            |             Yes            |            Yes           | Handling typos & unseen words        |

Key Takeaways
- TF-IDF: Simple and effective for bag-of-words models, but does not capture semantic meaning.
- Word2Vec: Captures contextual relationships but struggles with misspellings.
- FastText: Most robust for noisy Burmese text, thanks to its subword and typo handling.
  
## Experimental Findings

- TF-IDF with word tokenization achieved the best overall performance.
- SVM consistently outperformed other models (Naïve Bayes, Random Forest, XGBoost).
- Word2Vec and FastText performed less effectively, likely due to limited training data.

## 🏆 Final Model — SVM (Test Set Evaluation)
| **Metric** | **Score** |
| :--------- | :-------: |
| Accuracy   |   0.8897  |
| F1-Score   |   0.8935  |
| Precision  |   0.9019  |
| Recall     |   0.8897  |
| ROC-AUC    |   0.9713  |

🔹 Class-wise Performance
| **Class**          | **Precision** | **Recall** | **F1-Score** |
| :----------------- | :-----------: | :--------: | :----------: |
| **Potential Scam** |  🟥 **0.64**  |    0.84    |     0.72     |
| **Non-Scam**       |      0.95     |    0.91    |     0.93     |
| **Scam**           |      0.93     |    0.87    |     0.90     |

**Insight:**
While overall performance is strong (AUC ≈ 0.97), the model struggles with Potential Scam precision (0.64), indicating frequent misclassification with Non-Scam or Scam classes — which makes sense since Potential Scam messages often share overlapping vocabulary and structure with both legitimate and confirmed scam texts.
Future work could focus on better class balancing, richer contextual embeddings, or incorporating domain-specific linguistic cues to improve minority class detection.

## Deployment Considerations

    -Model saved via joblib/pickle for easy loading
    -Simple interface (Flask, Streamlit or FastAPI) can be wrapped for real-time classification
    -Designed with extendability in mind for additional languages, messaging platforms, or variant scam types
  
---
## Why This Matters

- Scams in Myanmar increasingly target youth on social platforms and use Burmese/English code-mixing.
- By building tools specifically tailored to Burmese content and vernacular usage, this project fills a critical gap in local digital safety infrastructure.
- The modular design makes it transferable to other contexts or languages with minimal adjustments

## Next Steps & Opportunities

- Expand model to cover more classes: e.g., “Investment Scam”, “Job Scam”, “Phishing”, etc.
- Deploy a real-time Telegram bot or web service for public use.
- Incorporate deep-learning embeddings (e.g., Burmese language models) for richer representation.
- Collect more data (especially for minority classes) to improve performance and generalization.
- Build a public awareness dashboard or visualisation site to promote findings.

## **Members**
1. Nu Wai Thet (MMDT 2024.001)(Project Leader)
2. Myint Myat Aung Zaw (MMDT 2024.044)(Contributor for data annotation)
3. Kaung Myat Kyaw (MMDT 2024.073)(Contributor for data annotation)
4. Dr Myo Thida (Advisor)

**Project Presentation slide** : [Burmese money scam_summary.pptx](https://github.com/user-attachments/files/23211313/Burmese.money.scam_summary.pptx)
   
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
```
---
## **License**

This project is open source under the MIT License

## **Contact**

For any questions, collaboration inquiries or feedback, feel free to reach out to me on GitHub or email: nuwaithet@gmail.com
Follow me on LinkedIn: https://www.linkedin.com/in/nuwai-thet-sophia/
