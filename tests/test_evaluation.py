import pandas as pd

from src.evaluation.error_analysis import build_error_rows
from src.evaluation.metrics import oov_recall, word_prf
from scripts.evaluate import calculate_metrics


def test_word_metrics_do_not_match_words_across_sentences():
    gold_sentences = [["a"], ["b"]]
    predicted_sentences = [["b"], ["a"]]

    assert word_prf(gold_sentences, predicted_sentences) == (0.0, 0.0, 0.0)
    assert oov_recall(gold_sentences, predicted_sentences, training_words=[]) == 0.0


def test_oov_recall_counts_matching_occurrences_within_each_sentence():
    gold_sentences = [["new", "new", "known"], ["other"]]
    predicted_sentences = [["new"], ["other"]]

    assert oov_recall(
        gold_sentences,
        predicted_sentences,
        training_words=["known"],
    ) == 2 / 3


def test_calculate_metrics_compares_words_within_each_sentence():
    test_df = pd.DataFrame(
        {
            "segmented": ["a", "b"],
            "labels": ["B", "B"],
        }
    )
    predictions = pd.DataFrame(
        {
            "predicted_segmented": ["b", "a"],
            "predicted_labels": ["B", "B"],
        }
    )

    metrics = calculate_metrics(test_df, predictions, training_segments=[])

    assert metrics["word_precision"] == 0.0
    assert metrics["word_recall"] == 0.0
    assert metrics["word_f1"] == 0.0
    assert metrics["oov_recall"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert set(metrics) == {
        "accuracy",
        "word_precision",
        "word_recall",
        "word_f1",
        "oov_recall",
    }


def test_error_analysis_lists_missed_and_extra_words_without_guessing_causes():
    test_df = pd.DataFrame(
        {
            "id": [17],
            "text": ["a b"],
            "segmented": ["a_b"],
            "labels": ["B I"],
        }
    )
    predictions = pd.DataFrame(
        {
            "predicted_segmented": ["a b"],
            "predicted_labels": ["B B"],
        }
    )

    rows = build_error_rows("lstm", test_df, predictions)

    assert rows == [
        {
            "model": "lstm",
            "id": 17,
            "text": "a b",
            "gold_segmented": "a_b",
            "predicted_segmented": "a b",
            "incorrect_gold_words": "a_b",
            "incorrect_predicted_words": "a;b",
        }
    ]
