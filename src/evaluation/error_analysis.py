"""Sentence-level segmentation differences between gold and predictions."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def build_error_rows(
    model_name: str,
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[dict[str, object]]:
    if len(test_df) != len(predictions):
        raise ValueError("Gold/prediction sentence count mismatch")

    rows = []
    for row_index, (_, gold_row) in enumerate(test_df.iterrows()):
        gold_segmented = str(gold_row["segmented"])
        predicted_segmented = str(predictions.iloc[row_index]["predicted_segmented"])
        if gold_segmented == predicted_segmented:
            continue

        gold_words = gold_segmented.split()
        predicted_words = predicted_segmented.split()
        missed_words = list((Counter(gold_words) - Counter(predicted_words)).elements())
        extra_words = list((Counter(predicted_words) - Counter(gold_words)).elements())
        rows.append(
            {
                "model": model_name,
                "id": gold_row.get("id", row_index + 1),
                "text": str(gold_row["text"]),
                "gold_segmented": gold_segmented,
                "predicted_segmented": predicted_segmented,
                "incorrect_gold_words": ";".join(missed_words),
                "incorrect_predicted_words": ";".join(extra_words),
            }
        )
    return rows
