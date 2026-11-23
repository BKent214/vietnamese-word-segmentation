from collections import Counter
from typing import Iterable, Sequence


def word_prf(
    gold_sentences: Sequence[Sequence[str]],
    pred_sentences: Sequence[Sequence[str]],
) -> tuple[float, float, float]:
    """Micro-averaged exact word-match scores, matching words within each sentence."""
    if len(gold_sentences) != len(pred_sentences):
        raise ValueError("Gold/prediction sentence count mismatch")

    true_positive = 0
    gold_total = 0
    pred_total = 0
    for gold_words, pred_words in zip(gold_sentences, pred_sentences):
        gold = Counter(gold_words)
        pred = Counter(pred_words)
        true_positive += sum((gold & pred).values())
        gold_total += sum(gold.values())
        pred_total += sum(pred.values())

    precision = true_positive / pred_total if pred_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def tag_accuracy(gold_tags: Sequence[str], pred_tags: Sequence[str]) -> float:
    if len(gold_tags) != len(pred_tags):
        raise ValueError("Gold/prediction tag length mismatch")
    if not gold_tags:
        return 0.0
    return sum(g == p for g, p in zip(gold_tags, pred_tags)) / len(gold_tags)


def oov_recall(
    gold_sentences: Sequence[Sequence[str]],
    pred_sentences: Sequence[Sequence[str]],
    training_words: Iterable[str],
) -> float:
    """Occurrence-level OOV recall, matching words only within the same sentence."""
    if len(gold_sentences) != len(pred_sentences):
        raise ValueError("Gold/prediction sentence count mismatch")

    training = set(training_words)
    matched_oov = 0
    total_oov = 0
    for gold_words, pred_words in zip(gold_sentences, pred_sentences):
        gold_oov = Counter(word for word in gold_words if word not in training)
        pred = Counter(pred_words)
        matched_oov += sum((gold_oov & pred).values())
        total_oov += sum(gold_oov.values())
    return matched_oov / total_oov if total_oov else 0.0