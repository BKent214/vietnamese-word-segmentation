
from typing import List, Sequence

def bio_to_words(syllables: Sequence[str], tags: Sequence[str]) -> List[str]:
    if len(syllables) != len(tags):
        raise ValueError("Syllable/tag length mismatch")
    words = []
    for syl, tag in zip(syllables, tags):
        if tag == "B" or not words:
            words.append([syl])
        elif tag == "I":
            words[-1].append(syl)
        else:
            raise ValueError(f"Invalid tag: {tag}")
    return ["_".join(w) for w in words]

def word_prf(gold_words: Sequence[str], pred_words: Sequence[str]):
    """Exact word match; precision/recall/F1 over word tokens."""
    from collections import Counter
    g, p      = Counter(gold_words), Counter(pred_words)
    tp        = sum((g & p).values())
    precision = tp / sum(p.values()) if p else 0.0
    recall    = tp / sum(g.values()) if g else 0.0
    f1        = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1

def boundary_prf(gold_tags: Sequence[str], pred_tags: Sequence[str]):
    """B/I boundary metric; B is treated as positive class."""
    tp = sum(g == "B" and p == "B" for g, p in zip(gold_tags, pred_tags))
    fp = sum(g != "B" and p == "B" for g, p in zip(gold_tags, pred_tags))
    fn = sum(g == "B" and p != "B" for g, p in zip(gold_tags, pred_tags))
    
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall    = tp/(tp+fn) if tp+fn else 0.0
    f1        = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1
