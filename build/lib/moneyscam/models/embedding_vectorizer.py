import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts text into sentence embeddings using a gensim model (Word2Vec/FastText).
    Supports optional tf-idf weighting and configurable fallback behavior.
    """

    def __init__(self, model, tokenize: bool = True, use_tfidf_weights: bool = False, idf_mapping=None, fallback: str = "mean"):
        self.model = model
        self.vector_size = model.vector_size
        self.tokenize = tokenize
        self.use_tfidf_weights = use_tfidf_weights
        self.idf_mapping = idf_mapping or {}
        self.fallback = fallback
        if fallback == "mean":
            self.default_vec = np.mean(self.model.wv.vectors, axis=0)
        else:
            self.default_vec = np.zeros(self.vector_size)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for text in X:
            words = text.split() if self.tokenize and isinstance(text, str) else text
            words_in_vocab = [w for w in words if w in self.model.wv]
            if len(words_in_vocab) == 0:
                transformed_X.append(self.default_vec.copy())
                continue

            word_vectors = [self.model.wv[w] for w in words_in_vocab]
            if self.use_tfidf_weights and self.idf_mapping:
                weights = [self.idf_mapping.get(w, 1.0) for w in words_in_vocab]
                sent_vec = np.average(word_vectors, axis=0, weights=weights)
            else:
                sent_vec = np.mean(word_vectors, axis=0)

            transformed_X.append(sent_vec)
        return np.array(transformed_X)
