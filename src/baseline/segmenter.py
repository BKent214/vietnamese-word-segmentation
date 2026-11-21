
"""Baseline methods for Vietnamese syllable-based word segmentation.

All predictors accept whitespace-separated syllables and return:
    (segmented_text, BIO_tags)
where B starts a word and I continues the previous word.
The dictionary and n-gram statistics must be built from TRAIN ONLY.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import re


def words_to_bio(words: Sequence[str]) -> Tuple[str, List[str]]:
    syllables, tags = [], []
    for word in words:
        parts = [p for p in word.split("_") if p]
        if not parts:
            continue
        syllables.extend(parts)
        tags.extend(["B"] + ["I"] * (len(parts) - 1))
    return " ".join(words), tags


def bio_to_segmented(syllables: Sequence[str], tags: Sequence[str]) -> str:
    if len(syllables) != len(tags):
        raise ValueError("Số âm tiết và số nhãn BIO không khớp.")
    words: List[List[str]] = []
    for syl, tag in zip(syllables, tags):
        if tag == "B" or not words:
            words.append([syl])
        elif tag == "I":
            words[-1].append(syl)
        else:
            raise ValueError(f"Nhãn không hợp lệ: {tag}")
    return " ".join("_".join(w) for w in words)


class BaseSegmenter:
    def predict_sentence(self, text: str) -> Tuple[str, List[str]]:
        raise NotImplementedError

    def predict(self, text: str) -> str:
        return self.predict_sentence(text)[0]


class WhitespaceSegmenter(BaseSegmenter):
    """Mỗi âm tiết cách nhau bằng space được xem là một từ."""
    def predict_sentence(self, text: str) -> Tuple[str, List[str]]:
        syllables = str(text).strip().split()
        return " ".join(syllables), ["B"] * len(syllables)


class RegexSegmenter(BaseSegmenter):
    """Rule-based baseline. Chỉ gộp một số mẫu hình thức rõ ràng.

    Đây là baseline heuristic, không thay thế từ điển/ngữ cảnh ngôn ngữ.
    Các rule có thể mở rộng theo phân tích lỗi của nhóm.
    """
    NUMBER  = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
    INTEGER = re.compile(r"^[+-]?\d+$")

    def predict_sentence(self, text: str) -> Tuple[str, List[str]]:
        tokens = str(text).strip().split()
        out: List[str] = []
        i = 0
        while i < len(tokens):
            # Ngày/tháng/năm dạng 12 / 05 / 2024
            if (i + 4 < len(tokens)    and self.INTEGER.match(tokens[i])
                and tokens[i+1] == "/" and self.INTEGER.match(tokens[i+2])
                and tokens[i+3] == "/" and self.INTEGER.match(tokens[i+4])):
                out.append("_".join(tokens[i:i+5])); i += 5; continue

            # Số thập phân bị tách: 3 . 14 hoặc 3 , 14
            if (i + 2 < len(tokens)           and self.INTEGER.match(tokens[i])
                and tokens[i+1] in {".", ","} and self.INTEGER.match(tokens[i+2])):
                out.append("_".join(tokens[i:i+3])); i += 3; continue

            # Số + phần trăm
            if i + 1 < len(tokens) and self.NUMBER.match(tokens[i]) and tokens[i+1] == "%":
                out.append(tokens[i] + "_%"); i += 2; continue

            out.append(tokens[i]); i += 1

        syllables = tokens
        tags = []
        # Chuyển các nhóm output trở lại nhãn cho từng âm tiết.
        for word in out:
            parts = word.split("_")
            tags.extend(["B"] + ["I"] * (len(parts)-1))
        if len(tags) != len(syllables):
            # Không để heuristic làm thay đổi/mất token; fallback an toàn.
            return " ".join(syllables), ["B"] * len(syllables)
        return bio_to_segmented(syllables, tags), tags


def build_word_dictionary(segmented_texts: Iterable[str]) -> Set[str]:
    """Tạo từ điển từ cột segmented của TRAIN."""
    dictionary: Set[str] = set()
    for sentence in segmented_texts:
        dictionary.update(w for w in str(sentence).split() if w)
    return dictionary


class MaximumMatchingSegmenter(BaseSegmenter):
    """Forward maximum matching: chọn từ dài nhất tại vị trí hiện tại."""
    def __init__(self, dictionary: Iterable[str], max_word_syllables: int | None = None):
        self.dictionary = set(dictionary)
        self.max_len = max_word_syllables or max(
            (len(w.split("_")) for w in self.dictionary), default=1
        )

    def segment_words(self, text: str) -> List[str]:
        syllables = str(text).strip().split()
        result, i = [], 0
        while i < len(syllables):
            matched = None
            upper = min(self.max_len, len(syllables) - i)
            for width in range(upper, 0, -1):
                candidate = "_".join(syllables[i:i+width])
                if candidate in self.dictionary:
                    matched = candidate
                    break
            if matched is None:
                matched = syllables[i]  # OOV fallback: một âm tiết thành một từ
            result.append(matched)
            i += len(matched.split("_"))
        return result

    def predict_sentence(self, text: str) -> Tuple[str, List[str]]:
        syllables = str(text).strip().split()
        words = self.segment_words(text)
        tags = []
        for word in words:
            parts = word.split("_")
            tags.extend(["B"] + ["I"] * (len(parts)-1))
        return bio_to_segmented(syllables, tags), tags


@dataclass
class BigramLanguageModel:
    unigram_counts: Counter
    bigram_counts: Counter
    vocab_size: int
    alpha: float = 0.1

    @classmethod
    def fit(cls, segmented_texts: Iterable[str], alpha: float = 0.1):
        unigrams, bigrams = Counter(), Counter()
        for sentence in segmented_texts:
            words = ["<s>"] + str(sentence).split() + ["</s>"]
            unigrams.update(words[:-1])
            bigrams.update(zip(words[:-1], words[1:]))
        vocab_size = max(1, len(unigrams) + 1)
        return cls(unigrams, bigrams, vocab_size, alpha)

    def log_prob(self, previous: str, current: str) -> float:
        numerator = self.bigram_counts[(previous, current)] + self.alpha
        denominator = self.unigram_counts[previous] + self.alpha * self.vocab_size
        return log(numerator / denominator)


class BigramSegmenter(BaseSegmenter):
    """Viterbi segmentation over dictionary candidates using a bigram LM.

    Score = sum log P(word_i | word_{i-1}); OOV syllables are singleton candidates.
    This is a transparent educational implementation, not a byte-for-byte
    reproduction of the 2008 paper's full system.
    """
    def __init__(self, dictionary: Iterable[str], lm: BigramLanguageModel):
        self.dictionary = set(dictionary)
        self.lm = lm
        self.max_len = max((len(w.split("_")) for w in self.dictionary), default=1)

    def _candidates(self, syllables: Sequence[str], start: int) -> List[Tuple[int, str]]:
        candidates = []
        for end in range(start + 1, min(len(syllables), start + self.max_len) + 1):
            word = "_".join(syllables[start:end])
            if word in self.dictionary:
                candidates.append((end, word))
        if not candidates:
            candidates.append((start + 1, syllables[start]))
        return candidates

    def segment_words(self, text: str) -> List[str]:
        syllables = str(text).strip().split()
        n = len(syllables)
        if n == 0:
            return []
        # dp[position][last_word] = (score, previous_word, previous_position)
        dp: List[Dict[str, Tuple[float, str | None, int]]] = [dict() for _ in range(n + 1)]
        dp[0]["<s>"] = (0.0, None, 0)

        for pos in range(n):
            if not dp[pos]:
                continue
            for prev, (prev_score, _, _) in list(dp[pos].items()):
                for end, word in self._candidates(syllables, pos):
                    score = prev_score + self.lm.log_prob(prev, word)
                    old = dp[end].get(word)
                    if old is None or score > old[0]:
                        dp[end][word] = (score, prev, pos)

        # Include sentence-end probability.
        last_word = max(
            dp[n],
            key=lambda w: dp[n][w][0] + self.lm.log_prob(w, "</s>")
        )
        words = []
        pos, word = n, last_word
        while word != "<s>":
            words.append(word)
            _, prev, prev_pos = dp[pos][word]
            word, pos = prev, prev_pos
        return list(reversed(words))

    def predict_sentence(self, text: str) -> Tuple[str, List[str]]:
        syllables = str(text).strip().split()
        words = self.segment_words(text)
        tags = []
        for word in words:
            parts = word.split("_")
            tags.extend(["B"] + ["I"] * (len(parts)-1))
        return bio_to_segmented(syllables, tags), tags


def build_bigram_model(segmented_texts: Iterable[str], alpha: float = 0.1) -> BigramLanguageModel:
    return BigramLanguageModel.fit(segmented_texts, alpha=alpha)
