import pandas as pd
from typing import Tuple

import preprocess_text  # re-use existing rich preprocessing built for mmdt-tokenizer


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply project preprocessing to a dataframe and expand derived columns.

    Returns a dataframe with: processed_text, emoji_count, hashtag_count, punctuation_counts.
    """
    processed = df[text_col].apply(lambda x: pd.Series(preprocess_text.preprocess_text(x)))
    processed.columns = ["processed_text", "emoji_count", "hashtag_count", "punctuation_counts"]
    df_out = pd.concat([df.drop(columns=[text_col]), processed], axis=1)
    return df_out


def load_processed(path: str) -> pd.DataFrame:
    """Load an already processed dataset (Excel)."""
    return pd.read_excel(path)


def preprocess_text_row(text: str) -> Tuple[str, int, int, dict]:
    """Expose single-text preprocessing for inference."""
    return preprocess_text.preprocess_text(text)
