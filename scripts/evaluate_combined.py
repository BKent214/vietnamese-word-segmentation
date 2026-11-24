"""Combine the dictionary-based Bigram segmenter with BiLSTM predictions.

Rule (fixed in advance, no tuning on test):
  1. Start from the Bigram segmentation (very accurate on words seen in train).
  2. Where Bigram leaves a run of single syllables (dictionary could not
     group them), accept a multi-syllable BiLSTM word lying entirely inside
     that run, but only if the word is NOT in the train dictionary
     (i.e. a genuine OOV candidate).

Usage:
  python scripts/evaluate_combined.py [lstm_predictions.csv] [split]
  default: outputs/lstm/test_predictions.csv, split=test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.baseline.segmenter import (
    BigramSegmenter,
    build_bigram_model,
    build_word_dictionary,
)
from src.evaluation.metrics import oov_recall, word_prf


def spans(words: list[str]) -> list[tuple[int, int, str]]:
    out, pos = [], 0
    for w in words:
        n = len(w.split("_"))
        out.append((pos, pos + n, w))
        pos += n
    return out


def combine(bigram_words: list[str], lstm_words: list[str], dictionary: set[str]) -> list[str]:
    b_spans = spans(bigram_words)
    single = set()  # syllable positions where Bigram output a single-syllable word
    for s, e, _ in b_spans:
        if e - s == 1:
            single.add(s)
    accepted = {}  # start -> (end, word)
    for s, e, w in spans(lstm_words):
        if e - s > 1 and w not in dictionary and all(p in single for p in range(s, e)):
            accepted[s] = (e, w)
    result, i = [], 0
    starts = {s: (e, w) for s, e, w in b_spans}
    while i in starts:
        if i in accepted:
            e, w = accepted[i]
        else:
            e, w = starts[i]
        result.append(w)
        i = e
    return result


def score(gold, pred, train_words):
    p, r, f = word_prf(gold, pred)
    return {"word_precision": p, "word_recall": r, "word_f1": f,
            "oov_recall": oov_recall(gold, pred, train_words)}


def main() -> None:
    pred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/lstm/test_predictions.csv"
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    lstm = pd.read_csv(pred_path)

    train_segments = train["segmented"].astype(str).tolist()
    dictionary = build_word_dictionary(train_segments)
    train_words = {w for s in train_segments for w in s.split()}
    bigram = BigramSegmenter(dictionary, build_bigram_model(train_segments))

    gold = [s.split() for s in lstm["segmented"].astype(str)]
    lstm_pred = [s.split() for s in lstm["pred_segmented"].astype(str)]
    bigram_pred = [bigram.predict_sentence(t)[0].split() for t in lstm["text"].astype(str)]
    combined = [combine(b, l, dictionary) for b, l in zip(bigram_pred, lstm_pred)]

    results = {
        "bigram": score(gold, bigram_pred, train_words),
        "lstm": score(gold, lstm_pred, train_words),
        "bigram+lstm": score(gold, combined, train_words),
    }
    for name, m in results.items():
        print(f"{name:12s} " + "  ".join(f"{k}={v*100:.2f}" for k, v in m.items()))
    out = ROOT / "results/combined_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
