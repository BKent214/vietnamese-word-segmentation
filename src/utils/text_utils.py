"""Tiền xử lý văn bản người dùng nhập, đưa về cùng định dạng với data/processed."""
import re

from src.data.preprocessing import normalize_text

PUNCTUATION = re.compile(r"([.,;:!?()\[\]{}\"“”‘’…%])")


def split_punctuation(text):
    """'gia đình.' -> 'gia đình .' (dấu câu thành token riêng)."""
    return PUNCTUATION.sub(r" \1 ", text)


def prepare_input(text):
    text = normalize_text(str(text or ""))
    return normalize_text(split_punctuation(text))
