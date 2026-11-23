"""Tests for preprocessing and word/B-I annotation helpers."""

from src.data.annotation import (
    from_segmented_line,
    labels_to_words,
    to_segmented_line,
    words_to_labels,
)
from src.data.preprocessing import normalize_text


WORDS = ["Hiện nay", "xã", "có", "68", "tổ", "nhân dân", "."]


def test_words_to_labels():
    syllables, labels = words_to_labels(WORDS)
    assert syllables == ["Hiện", "nay", "xã", "có", "68", "tổ", "nhân", "dân", "."]
    assert labels == ["B", "I", "B", "B", "B", "B", "B", "I", "B"]


def test_labels_to_words_round_trip():
    syllables, labels = words_to_labels(WORDS)
    assert labels_to_words(syllables, labels) == WORDS


def test_labels_to_words_treats_initial_inside_as_first_word():
    assert labels_to_words(["học", "sinh"], ["I", "I"]) == ["học sinh"]


def test_segmented_line_round_trip():
    line = to_segmented_line(WORDS)
    assert line == "Hiện_nay xã có 68 tổ nhân_dân ."
    assert from_segmented_line(line) == WORDS


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  học   sinh ") == "học sinh"


def test_normalize_text_uses_unicode_nfc():
    assert normalize_text("e\u0301") == "é"
