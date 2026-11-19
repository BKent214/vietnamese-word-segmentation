"""
Kiểm thử phần tiền xử lý. Chạy: pytest tests/test_preprocessing.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import normalize_text
from src.data.annotation import (
    words_to_labels, labels_to_words, to_segmented_line, from_segmented_line,
)

WORDS = ["Hiện nay", "xã", "có", "68", "tổ", "nhân dân", "."]


def test_words_to_labels():
    syllables, labels = words_to_labels(WORDS)
    assert syllables == ["Hiện", "nay", "xã", "có", "68", "tổ", "nhân", "dân", "."]
    assert labels == ["B", "I", "B", "B", "B", "B", "B", "I", "B"]


def test_labels_to_words_round_trip():
    syllables, labels = words_to_labels(WORDS)
    assert labels_to_words(syllables, labels) == WORDS


def test_labels_to_words_starts_with_I():
    assert labels_to_words(["học", "sinh"], ["I", "I"]) == ["học sinh"]


def test_segmented_line():
    line = to_segmented_line(WORDS)
    assert line == "Hiện_nay xã có 68 tổ nhân_dân ."
    assert from_segmented_line(line) == WORDS


def test_normalize_text():
    assert normalize_text("  học   sinh ") == "học sinh"
