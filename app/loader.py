"""Nạp các mô hình một lần khi app khởi động."""
import pandas as pd

from src.baseline.segmenter import (
    BigramSegmenter,
    MaximumMatchingSegmenter,
    RegexSegmenter,
    WhitespaceSegmenter,
    build_bigram_model,
    build_word_dictionary,
)
from src.lstm.predict import WordSegmenterPredictor


def load_baselines(train_path):
    """Từ điển và Bigram chỉ xây từ tập train."""
    train_df = pd.read_csv(train_path)
    segments = train_df["segmented"].astype(str).tolist()
    dictionary = build_word_dictionary(segments)
    bigram_model = build_bigram_model(segments)

    return {
        "Whitespace": WhitespaceSegmenter(),
        "Regex": RegexSegmenter(),
        "Maximum Matching": MaximumMatchingSegmenter(dictionary),
        "Bigram (Hybrid)": BigramSegmenter(dictionary, bigram_model),
    }


def load_lstm(config):
    """Trả về None nếu chưa train LSTM."""
    if not (config.best_model_path.is_file() and config.vocab_path.is_file()):
        return None

    predictor = WordSegmenterPredictor(
        config.best_model_path, config.vocab_path, config.device
    )
    predictor.predict_sentence("khởi động")
    return predictor
