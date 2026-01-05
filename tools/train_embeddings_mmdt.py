"""
Train Word2Vec and FastText embeddings on the preprocessed MMDT-tokenized text.

Input: data/preprocess_mmdt_tokenizer.xlsx (expects a 'processed_text' column).
Output:
    - models/word2vec_mmdt.model
    - models/fasttext_mmdt.model
"""

from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec, FastText


def load_processed_texts(excel_path: Path) -> list[list[str]]:
    """Load tokenized sentences from the preprocessed Excel file."""
    df = pd.read_excel(excel_path)
    if "processed_text" not in df.columns:
        raise ValueError("Expected 'processed_text' column in the preprocessed file.")
    texts = df["processed_text"].fillna("").astype(str).tolist()
    # The text is already tokenized with mmdt; split on whitespace to feed gensim
    return [t.split() for t in texts if t.strip()]


def train_word2vec(sentences: list[list[str]], model_path: Path, vector_size: int = 300):
    """Train and save a Word2Vec model."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=1,  # skip-gram
        epochs=10,
    )
    model.save(str(model_path))
    return model_path


def train_fasttext(sentences: list[list[str]], model_path: Path, vector_size: int = 300):
    """Train and save a FastText model."""
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=1,  # skip-gram
        epochs=10,
    )
    model.save(str(model_path))
    return model_path


def main():
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "preprocess_mmdt_tokenizer.xlsx"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_processed_texts(data_path)

    w2v_path = models_dir / "word2vec_mmdt.model"
    ft_path = models_dir / "fasttext_mmdt.model"

    train_word2vec(sentences, w2v_path)
    print(f"Saved Word2Vec model to: {w2v_path}")

    train_fasttext(sentences, ft_path)
    print(f"Saved FastText model to: {ft_path}")


if __name__ == "__main__":
    main()
