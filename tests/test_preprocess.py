import preprocess_text


def test_preprocess_returns_expected_tuple():
    text = "Test message"
    processed_text, emoji_count, hashtag_count, punctuation_counts = preprocess_text.preprocess_text(text)
    assert isinstance(processed_text, str)
    assert isinstance(emoji_count, int)
    assert isinstance(hashtag_count, int)
    assert isinstance(punctuation_counts, (int, dict))
