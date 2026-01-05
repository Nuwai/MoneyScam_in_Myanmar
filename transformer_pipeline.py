import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import preprocess_text
import os
import sys
import logging
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from gensim.models import Word2Vec, FastText
from skops.io import dump as skops_dump


# Custom Transformer for Word2Vec & FastText
class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenize=True, use_tfidf_weights=False, idf_mapping=None, fallback="mean"):
        """
        model: Word2Vec or FastText model
        tokenize: If True, will split the text before embedding
        use_tfidf_weights: If True, weight word vectors by provided idf_mapping
        idf_mapping: dict mapping token -> idf weight (only used when use_tfidf_weights=True)
        fallback: 'mean' uses global mean vector when no words found; 'zero' uses zeros
        """
        self.model = model
        self.vector_size = model.vector_size
        self.tokenize = tokenize  # Allow tokenization inside transform
        self.use_tfidf_weights = use_tfidf_weights
        self.idf_mapping = idf_mapping or {}
        self.fallback = fallback
        # Precompute a fallback vector (mean of all embeddings) for all-OOV cases
        if fallback == "mean":
            self.default_vec = np.mean(self.model.wv.vectors, axis=0)
        else:
            self.default_vec = np.zeros(self.vector_size)

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        """
        Convert list of texts (raw or tokenized) into sentence embeddings.
        If a word is not in the vocabulary, it is ignored.
        If no words are found, return a fallback vector (mean or zeros).
        FastText will rarely hit the fallback because it builds subword embeddings.
        """
        transformed_X = []
        
        for text in X:
            # Tokenize if input is raw text (string), otherwise assume it's already tokenized
            words = text.split() if self.tokenize and isinstance(text, str) else text
            words_in_vocab = [w for w in words if w in self.model.wv]
            
            # If no valid words, use fallback vector
            if len(words_in_vocab) == 0:
                transformed_X.append(self.default_vec.copy())
                continue

            # Get word vectors (and optional tf-idf weights)
            word_vectors = [self.model.wv[w] for w in words_in_vocab]
            if self.use_tfidf_weights and self.idf_mapping:
                weights = [self.idf_mapping.get(w, 1.0) for w in words_in_vocab]
                sent_vec = np.average(word_vectors, axis=0, weights=weights)
            else:
                sent_vec = np.mean(word_vectors, axis=0)

            transformed_X.append(sent_vec)
        
        return np.array(transformed_X)
    

# Model Pipeline
def get_classification_pipeline(model, vectorizer_type="tfidf", idf_mapping=None):
    """
    ColumnTransformer:
    ColumnTransformer.fit_transform(X, y) calls .fit(X, y) followed by .transform(X).
    When we call .fit_transform(df), it automatically triggers .transform(X) inside EmbeddingVectorizer.
    This is part of scikit-learn's API design, ensuring transformers behave consistently inside pipelines.
    ColumnTransformer automatically calls .transform(X) when fit_transform(X, y) or transform(X) is called.
    Pipeline calls preprocessor.transform(X) before passing data to the classifier.
    Scikit-learn follows a structured API: Transformers inside ColumnTransformer always execute .transform(X), ensuring smooth feature extraction.
    """

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "XGBoost": XGBClassifier(),
        "SVM": SVC(probability=True, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(fit_prior=True)
    }
    
    classifier = models[model]

    if vectorizer_type == "tfidf":
        text_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
    elif vectorizer_type == "word2vec":
        word2vec_model = Word2Vec.load("word2vec.model")
        text_vectorizer = EmbeddingVectorizer(
            word2vec_model,
            use_tfidf_weights=bool(idf_mapping),
            idf_mapping=idf_mapping,
        )
    elif vectorizer_type == "fasttext":
        fasttext_model = FastText.load("fasttext.model")
        text_vectorizer = EmbeddingVectorizer(
            fasttext_model,
            use_tfidf_weights=bool(idf_mapping),
            idf_mapping=idf_mapping,
        )
    else:
        raise ValueError("Invalid vectorizer_type. Choose 'tfidf', 'word2vec', or 'fasttext'.")

    numeric_features = ['emoji_count', 'hashtag_count', 'punctuation_counts']
    numeric_transformer = Pipeline([
        #('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())  # Use MinMaxScaler instead of StandardScaler not to have negative values
    ])
    
    preprocessor = ColumnTransformer([
        ('text', text_vectorizer, 'processed_text'),
        ('num', numeric_transformer, numeric_features)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def hyperparameter_tuning(best_model, best_model_name, X_val,y_val):

    # Hyperparameter tuning on the best model using X_val, y_val, Validation Set
    # pipeline defines the classifier step as 'classifier', all hyperparameters must be prefixed with 'classifier__

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
        'classifier__max_depth': [10, 20, 30, None]
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

    if best_model_name in param_grids and param_grids[best_model_name]:
        print(f"\n{'='*40}\nHyperparameter Tuning for {best_model_name}\n{'='*40}")
        param_grid = param_grids[best_model_name]

        randomized_search = RandomizedSearchCV(
            best_model, param_grid, scoring='f1_weighted', cv=5, 
            n_iter=10, n_jobs=-1, random_state=42
        )
        randomized_search.fit(X_val, y_val)

        best_model = randomized_search.best_estimator_
        print(f"Best Hyperparameters for {best_model_name}: {randomized_search.best_params_}")

    return best_model


def train_and_evaluate(df,vectorizer_type = "tfidf"):

    """
    Total datasize for Non Scam text: 10905
    Total datasize for Scam text: 5961
    Total datasize for Potential Scam text: 2481

    Stratified K-Fold:

    Ensures class distribution
    Best for imbalanced datasets
    Every fold maintains original class distribution

    Train, Validation, and Test Split:

    Split the data into a train (80%) and test (20%) set first.
    Further split the train (80%) into a train pool (80%) and validation (20%) set.

    Cross-validation:

    Stratified K-Fold cross-validation on the train pool for model selection.
    For each fold, model performance is evaluated using metrics like Accuracy, F1-Score, ROC-AUC, and Log Loss.
    Get the best model based on f1 score which is the harmonice balance between precision and recall

    Perform hyperparamater tuning for best model on validation set to avoid data leakage
    Train Best Model on Full Training Data:

    After evaluating and selecting the best model based on cross-validation, train it on the full training pool.
    Final Test Evaluation using test set:

    After training the best model, evaluate it on the test set, which was not used during training or validation, to get a final performance report.

    """
    # Set up logging
    log_filename = f"result_log_files/money_scam_training_results_{vectorizer_type}.log"
    log_filepath = os.path.join(os.getcwd(), log_filename)
    # Ensure the directory exists before saving the log file
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Redirect stdout to log file and console
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")  # Append mode

        def write(self, message):
            self.terminal.write(message)  # Print to console
            self.log.write(message)       # Save to file

        def flush(self):  # Needed for compatibility with sys.stdout
            self.terminal.flush()
            self.log.flush()

        def close(self):  # Properly close log file
            self.log.close()

    sys.stdout = Logger(log_filepath)  # Redirect all print statements

    logging.info("Training Started...")  # Backup log in case of crash

    X = df[['processed_text', 'emoji_count', 'hashtag_count', 'punctuation_counts']]
    y = df['label']

    class_counts = {label: (y == label).sum() for label in np.unique(y)}
    print(f"Original Class Distribution: {class_counts}")

    # Split into Train and Test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Full Training Set Size: {X_train.shape[0]}")
    print(f"Test Set Size: {X_test.shape[0]}")

    # Further split the train data into Train and Validation sets (80% train, 20% validation)
    X_train_pool, X_val, y_train_pool, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training Pool Size: {X_train_pool.shape[0]}")
    #print(f"Training Pool Size Y: {y_train_pool.shape[0]}")
    print(f"Validation Set Size: {X_val.shape[0]}")
    #print(f"Validation Set Size yval: {y_val.shape[0]}")

    # Build an IDF mapping (token -> idf weight) for embedding weighting, if needed
    idf_mapping = None
    if vectorizer_type in ["word2vec", "fasttext"]:
        tfidf_for_idf = TfidfVectorizer(tokenizer=str.split, lowercase=False)
        tfidf_for_idf.fit(X_train_pool["processed_text"])
        vocab = tfidf_for_idf.vocabulary_
        idf_vals = tfidf_for_idf.idf_
        idf_mapping = {tok: idf_vals[idx] for tok, idx in vocab.items()}
        print(f"Built IDF mapping for {len(idf_mapping)} tokens.")

    print(f"Selected Vectorizer Type is {vectorizer_type}.")
    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = ["SVM", "Decision Tree", "Random Forest", "XGBoost", "Naive Bayes"]
    if vectorizer_type == 'fasttext' or vectorizer_type =='word2vec':
        models = ["Logistic Regression",  "Decision Tree", "Random Forest",  "SVM"] # remove xgboost and naive bayes due to negative values
    
    best_model = None
    best_score = 0
    best_model_name = ""
    model_scores = {}
    # Evaluate models using Stratified K-Fold Cross-Validation
   
    for model_type in models:
        print(f"\n{'='*40}\nTraining Model: {model_type}\n{'='*40}")
        logging.info(f"Training Model: {model_type}")
        fold_f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_pool, y_train_pool)):
            # print(f"Fold {fold+1} Class Distribution:")
            # print(y_train_pool.iloc[train_idx].value_counts())
            # print(y_train_pool.iloc[val_idx].value_counts())
            X_train_fold, X_val_fold = X_train_pool.iloc[train_idx], X_train_pool.iloc[val_idx]
            y_train_fold, y_val_fold = y_train_pool.iloc[train_idx], y_train_pool.iloc[val_idx]  
          
            pipeline = get_classification_pipeline(
                model_type,
                vectorizer_type=vectorizer_type,
                idf_mapping=idf_mapping,
            )  
            pipeline.fit(X_train_fold,y_train_fold)
            
            y_pred = pipeline.predict(X_val_fold)
            y_prob = pipeline.predict_proba(X_val_fold)
            
            # Calculate metrics for each fold
            fold_f1 = f1_score(y_val_fold, y_pred, average='weighted')
            # fold_precision = precision_score(y_val_fold, y_pred, average='weighted')
            # fold_recall = recall_score(y_val_fold, y_pred, average='weighted')
            fold_f1_scores.append(fold_f1)
            
            print(f"\n{'-'*40}\nEvaluating Fold {fold+1} ({model_type})\n{'-'*40}")
            print(f"Accuracy: {accuracy_score(y_val_fold, y_pred):.4f}")
            print("Confusion Matrix:\n", confusion_matrix(y_val_fold, y_pred))
            print("Classification Report:\n", classification_report(y_val_fold, y_pred))
            print(f"F1-Score: {fold_f1:.4f}")
            print(f"ROC-AUC: {roc_auc_score(y_val_fold, y_prob, multi_class='ovr'):.4f}")
            print(f"Log Loss: {log_loss(y_val_fold, y_prob):.4f}")
            print(f"{'-'*40}\n")
        
        # Calculate the average F1 score for this model
        avg_fold_score = np.mean(fold_f1_scores)
        model_scores[model_type] = avg_fold_score  # Save result
        print(f"\nAverage F1-Score for {model_type}: {avg_fold_score:.4f}")
        
        # Update best model based on average fold score
        if avg_fold_score > best_score:
            best_score = avg_fold_score
            best_model_name = model_type
            best_model = pipeline

    # Prepare data for table
    table_data = [[model, f"{score:.4f}"] for model, score in model_scores.items()]
    # Print summary in table format
    print(f"\n{'='*70}\nModel Performance Summary\n{'='*70}")
    print(tabulate(table_data, headers=["Model", "Average F1-Score"], tablefmt="grid"))

    print(f"\n{'='*70}\nBest Model: {best_model_name} (F1-Score: {best_score:.4f})\n{'='*70}")
    
    if vectorizer_type == 'tfidf':
        best_model = hyperparameter_tuning(best_model, best_model_name, X_val,y_val)
    
    # Step 2: Train the best model on the full training set (including train_pool + validation set)
    best_model.fit(X_train, y_train)

    # Step 3: Final Test Evaluation
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)
    
    print(f"\n{'='*40}\nFinal Evaluation on Test Set\n{'='*40}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    print("Classification Report:\n", classification_report(y_test, y_pred_test))
    print(f"F1-Score: {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_test, multi_class='ovr'):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_prob_test):.4f}")
    print(f"{'-'*40}\n")
    print("\nTraining Complete. Logs saved in:", log_filepath)

    # Save the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # Ensure the "models" folder exists
    # joblib.dump(best_model, os.path.join(model_dir, "scam_detector.pkl"))
    # use skops to prevent python version mismatches in streamlit
    skops_dump(best_model, "models/best_model_scam_detector.skops")

    # Restore original stdout after training
    sys.stdout = sys.__stdout__

# Load Data and Run Training
if __name__ == "__main__":

    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "data/preprocess_mmdt_tokenizer.xlsx")
    # Load the dataset
    df = pd.read_excel(data_path)
    # # Xgboost and Naive Bayes does not work with -1
    # df['label'] = df['label'].replace({-1: 1, 1: 2})
    
    train_and_evaluate(df, vectorizer_type = "fasttext")
