"""Evaluate baseline segmenters and an available LSTM checkpoint on the test set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.baseline.segmenter import (
    BaseSegmenter,
    BigramSegmenter,
    MaximumMatchingSegmenter,
    RegexSegmenter,
    WhitespaceSegmenter,
    build_bigram_model,
    build_word_dictionary,
)
from src.evaluation.error_analysis import build_error_rows
from src.evaluation.metrics import (
    oov_recall,
    tag_accuracy,
    word_prf,
)
from src.lstm.config import LSTMConfig
from src.lstm.predict import WordSegmenterPredictor


def build_baseline_models(training_segments: list[str]) -> dict[str, BaseSegmenter]:
    dictionary = build_word_dictionary(training_segments)
    bigram_model = build_bigram_model(training_segments)
    return {
        "whitespace": WhitespaceSegmenter(),
        "regex": RegexSegmenter(),
        "maximum_matching": MaximumMatchingSegmenter(dictionary),
        "bigram": BigramSegmenter(dictionary, bigram_model),
    }


def calculate_metrics(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    training_segments: list[str],
) -> dict[str, float]:
    gold_tags = [tag for tags in test_df["labels"] for tag in str(tags).split()]
    predicted_tags = [
        tag for tags in predictions["predicted_labels"] for tag in str(tags).split()
    ]
    if len(gold_tags) != len(predicted_tags):
        raise ValueError("Gold and predicted BIO tag counts do not match")

    gold_word_sentences = [
        str(sentence).split() for sentence in test_df["segmented"]
    ]
    predicted_word_sentences = [
        str(sentence).split() for sentence in predictions["predicted_segmented"]
    ]
    precision, recall, word_f1 = word_prf(
        gold_word_sentences,
        predicted_word_sentences,
    )
    training_words = {
        word for sentence in training_segments for word in str(sentence).split()
    }

    return {
        "accuracy": tag_accuracy(gold_tags, predicted_tags),
        "word_precision": precision,
        "word_recall": recall,
        "word_f1": word_f1,
        "oov_recall": oov_recall(
            gold_word_sentences,
            predicted_word_sentences,
            training_words,
        ),
    }


def predict_dataset(
    model: BaseSegmenter | WordSegmenterPredictor,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    predicted_segments = []
    predicted_labels = []
    started = time.perf_counter()
    for text in test_df["text"]:
        segmented, tags = model.predict_sentence(str(text))
        predicted_segments.append(segmented)
        predicted_labels.append(" ".join(tags))
    elapsed = time.perf_counter() - started

    predictions = pd.DataFrame(
        {
            "predicted_segmented": predicted_segments,
            "predicted_labels": predicted_labels,
        },
        index=test_df.index,
    )
    for row_index, (gold, predicted) in enumerate(
        zip(test_df["labels"], predictions["predicted_labels"]), start=1
    ):
        if len(str(gold).split()) != len(str(predicted).split()):
            raise ValueError(f"Sentence {row_index}: gold/predicted token counts differ")
    return predictions, elapsed


def save_comparison_plot(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    metric_names = ["word_f1", "accuracy", "oov_recall"]
    labels = ["Word F1", "Accuracy", "OOV Recall"]
    figure, axis = plt.subplots(figsize=(10, 5))
    model_names = list(metrics)
    width = 0.8 / len(model_names)
    positions = list(range(len(metric_names)))

    for model_index, model_name in enumerate(model_names):
        values = [metrics[model_name][metric] for metric in metric_names]
        offsets = [position - 0.4 + width / 2 + model_index * width for position in positions]
        axis.bar(offsets, values, width=width, label=model_name)

    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Baseline and LSTM evaluation")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    config = LSTMConfig(
        train_path=REPOSITORY_ROOT / "data/processed/train.csv",
        dev_path=REPOSITORY_ROOT / "data/processed/dev.csv",
        test_path=REPOSITORY_ROOT / "data/processed/test.csv",
        best_model_path=REPOSITORY_ROOT / "checkpoints/lstm/best_lstm_model.pt",
        vocab_path=REPOSITORY_ROOT / "checkpoints/lstm/vocab.json",
    )
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    output_dir = REPOSITORY_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    training_segments = train_df["segmented"].astype(str).tolist()
    models: dict[str, BaseSegmenter | WordSegmenterPredictor] = dict(
        build_baseline_models(training_segments)
    )
    if config.best_model_path.is_file() and config.vocab_path.is_file():
        models["lstm"] = WordSegmenterPredictor(
            config.best_model_path,
            config.vocab_path,
            config.device,
        )
    else:
        print(
            "[WARNING] LSTM checkpoint/vocabulary not found; "
            "only baseline results will be produced."
        )

    all_metrics: dict[str, dict[str, float]] = {}
    prediction_tables = []
    error_rows = []
    for model_name, model in models.items():
        predictions, elapsed = predict_dataset(model, test_df)
        model_metrics = calculate_metrics(test_df, predictions, training_segments)
        model_metrics["inference_time_seconds"] = elapsed
        all_metrics[model_name] = model_metrics

        model_predictions = test_df[["id", "text", "segmented", "labels"]].copy()
        model_predictions["model"] = model_name
        model_predictions["predicted_segmented"] = predictions["predicted_segmented"]
        model_predictions["predicted_labels"] = predictions["predicted_labels"]
        prediction_tables.append(model_predictions)
        error_rows.extend(
            build_error_rows(
                model_name,
                test_df,
                predictions,
            )
        )
        print(f"{model_name}: {json.dumps(model_metrics, ensure_ascii=False)}")

    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame.from_dict(all_metrics, orient="index").rename_axis("model").to_csv(
        output_dir / "metrics.csv",
        encoding="utf-8-sig",
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        error_rows,
        columns=[
            "model",
            "id",
            "text",
            "gold_segmented",
            "predicted_segmented",
            "incorrect_gold_words",
            "incorrect_predicted_words",
        ],
    ).to_csv(
        output_dir / "error_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_comparison_plot(all_metrics, output_dir / "figures")
    print(f"Saved metrics, predictions, error analysis, and plot under {output_dir}.")


if __name__ == "__main__":
    main()
