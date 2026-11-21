
# Baseline models

Input contract: whitespace-separated Vietnamese syllables. Every method returns
`(pred_segmented, pred_labels)`, where `pred_segmented` joins syllables within a
word using `_` and `pred_labels` is a list of `B`/`I` labels aligned to input
syllables.

## Methods
- `WhitespaceSegmenter`: every whitespace-delimited syllable is one word.
- `RegexSegmenter`: small set of deterministic formatting rules (dates, decimal
  numbers, percentages, brackets/punctuation). It is a heuristic baseline.
- `MaximumMatchingSegmenter`: forward maximum matching with a training-derived
  word dictionary; unknown syllables fall back to singleton words.
- `BigramSegmenter`: dictionary candidate generation plus Viterbi decoding with
  an add-alpha-smoothed bigram language model.

## Example
```python
import pandas as pd
from src.baseline import (
    WhitespaceSegmenter, RegexSegmenter, MaximumMatchingSegmenter,
    BigramSegmenter, build_word_dictionary, build_bigram_model
)

train = pd.read_csv("data/processed/train.csv")
dictionary = build_word_dictionary(train["segmented"])
lm = build_bigram_model(train["segmented"], alpha=0.1)

models = {
    "whitespace": WhitespaceSegmenter(),
    "regex": RegexSegmenter(),
    "maximum_matching": MaximumMatchingSegmenter(dictionary),
    "bigram": BigramSegmenter(dictionary, lm),
}
text = "Hiện nay xã có 68 tổ nhân dân ."
for name, model in models.items():
    segmented, labels = model.predict_sentence(text)
    print(name, segmented, " ".join(labels))
```

**Important:** fit the dictionary and n-gram model on the training portion of each
fold only. Do not build them from dev/test. This implementation is an educational
baseline and should not be described as a complete reproduction of the 2008 paper.
