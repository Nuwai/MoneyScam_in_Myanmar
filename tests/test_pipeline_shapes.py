import pandas as pd
from moneyscam.models.classical import get_classification_pipeline


def test_pipeline_trains_on_small_dataset():
    data = pd.DataFrame(
        {
            "processed_text": ["hello world", "spam offer", "nice day", "buy now"],
            "emoji_count": [0, 1, 0, 0],
            "hashtag_count": [0, 0, 0, 0],
            "punctuation_counts": [0, 1, 0, 1],
            "label": ["non", "scam", "non", "scam"],
        }
    )
    pipeline = get_classification_pipeline("Logistic Regression", vectorizer_type="tfidf")
    pipeline.fit(data[["processed_text", "emoji_count", "hashtag_count", "punctuation_counts"]], data["label"])
    preds = pipeline.predict(data[["processed_text", "emoji_count", "hashtag_count", "punctuation_counts"]])
    assert len(preds) == len(data)
