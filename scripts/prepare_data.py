"""Convert the raw UD Vietnamese-VTB splits to the project's processed CSV format.

Run from the repository root:
    python scripts/prepare_data.py
"""

import csv
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.annotation import labels_to_words, to_segmented_line, words_to_labels
from src.data.preprocessing import read_conllu

RAW_DIR = REPOSITORY_ROOT / "data" / "raw" / "ud_vtb"
OUT_DIR = REPOSITORY_ROOT / "data" / "processed"
SPLITS = ("train", "dev", "test")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        source_path = RAW_DIR / f"vi_vtb-ud-{split}.conllu"
        sentences = read_conllu(source_path)
        errors = 0
        output_path = OUT_DIR / f"{split}.csv"

        with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["id", "text", "segmented", "labels"])

            for sentence_id, words in enumerate(sentences, start=1):
                syllables, labels = words_to_labels(words)
                if labels_to_words(syllables, labels) != words:
                    errors += 1

                writer.writerow(
                    [
                        sentence_id,
                        " ".join(syllables),
                        to_segmented_line(words),
                        " ".join(labels),
                    ]
                )

        print(
            f"{split:5s}: {len(sentences)} câu -> {output_path}, "
            f"lỗi chuyển đổi: {errors}"
        )


if __name__ == "__main__":
    main()
