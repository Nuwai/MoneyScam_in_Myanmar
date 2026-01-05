# from typing import Dict, Optional

# from gensim.models import FastText, Word2Vec
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.base import BaseEstimator
# from xgboost import XGBClassifier

# from .embedding_vectorizer import EmbeddingVectorizer
import sys
print(sys.path)

# from moneyscam.settings import paths
# print("Using embedding vectorizer:")
# print(paths.fasttext_path)

# NUMERIC_FEATURES = ["emoji_count", "hashtag_count", "punctuation_counts"]


# def _load_embedding(vectorizer_type: str):
#     print("Loading embedding model...")
#     print(paths.fasttext_path)
#     if vectorizer_type == "word2vec":
#         return Word2Vec.load(paths.word2vec_path)
#     if vectorizer_type == "fasttext":
#         return FastText.load(paths.fasttext_path)
    
#     raise ValueError(f"Unsupported embedding type: {vectorizer_type}")


# def get_classification_pipeline(model_name: str, vectorizer_type: str = "tfidf", idf_mapping: Optional[Dict[str, float]] = None) -> BaseEstimator:
#     models = {
#         "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=200),
#         "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
#         "Random Forest": RandomForestClassifier(class_weight="balanced"),
#         "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False),
#         "SVM": SVC(probability=True, class_weight="balanced"),
#         "Naive Bayes": MultinomialNB(fit_prior=True),
#     }
#     if model_name not in models:
#         raise ValueError(f"Invalid model_name: {model_name}")
#     classifier = models[model_name]

#     if vectorizer_type == "tfidf":
#         text_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
#     elif vectorizer_type in {"word2vec", "fasttext"}:
#         print("Using embedding vectorizer:")
#         print(paths.fasttext_path)
#         emb_model = _load_embedding(vectorizer_type)
#         text_vectorizer = EmbeddingVectorizer(
#             emb_model,
#             use_tfidf_weights=bool(idf_mapping),
#             idf_mapping=idf_mapping,
#         )
#     else:
#         raise ValueError("Invalid vectorizer_type. Choose 'tfidf', 'word2vec', or 'fasttext'.")

#     numeric_transformer = Pipeline(
#         steps=[
#             ("scaler", MinMaxScaler()),
#         ]
#     )

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("text", text_vectorizer, "processed_text"),
#             ("num", numeric_transformer, NUMERIC_FEATURES),
#         ]
#     )

#     pipeline = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("classifier", classifier),
#         ]
#     )
#     return pipeline
