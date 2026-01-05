import pandas as pd
from moneyscam.settings import paths
from moneyscam.features.preprocess import load_processed


def load_training_data(path: str = None) -> pd.DataFrame:
    """Load processed training data (expects Excel produced by preprocessing)."""
    target = path or paths.data_path
    return load_processed(target)
