#!/usr/bin/env python3
"""
Tải và kiểm định tính toàn vẹn của bộ dữ liệu UD Vietnamese-VTB.
Thực hiện: Hải (TV03 - Nhóm thực nghiệm)
"""

import os
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/master/"
DATASET_FILES = [
    "vi_vtb-ud-train.conllu",
    "vi_vtb-ud-dev.conllu",
    "vi_vtb-ud-test.conllu",
]

TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "ud_vtb")


def download_dataset():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Target directory: {TARGET_DIR}")

    for filename in DATASET_FILES:
        filepath = os.path.join(TARGET_DIR, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"✓ {filename} already exists ({os.path.getsize(filepath):,} bytes)")
            continue

        url = BASE_URL + filename
        print(f"Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
            f.write(resp.read())
        print(f"✓ Downloaded {filename} ({os.path.getsize(filepath):,} bytes)")


def verify_statistics():
    print("\n--- Thống kê dữ liệu UD Vietnamese-VTB ---")
    total_sentences = 0
    total_words = 0
    total_tokens = 0
    total_syllables = 0

    for filename in DATASET_FILES:
        filepath = os.path.join(TARGET_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Error: {filepath} does not exist.")
            continue

        sentences = 0
        words = 0
        tokens = 0
        syllables = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# sent_id"):
                    sentences += 1
                elif line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) >= 10:
                        words += 1
                        form = parts[1]
                        upos = parts[3]
                        subparts = form.split()
                        tokens += len(subparts)
                        if upos not in ["PUNCT", "SYM", "NUM"]:
                            syllables += len(subparts)

        total_sentences += sentences
        total_words += words
        total_tokens += tokens
        total_syllables += syllables

        print(
            f"{filename:25s} | Sentences: {sentences:5,d} | Words: {words:6,d} | Tokens: {tokens:6,d} | Est. Syllables: {syllables:6,d}"
        )

    print("-" * 75)
    print(
        f"{'TỔNG CỘNG':25s} | Sentences: {total_sentences:5,d} | Words: {total_words:6,d} | Tokens: {total_tokens:6,d} | Est. Syllables: {total_syllables:6,d}"
    )


if __name__ == "__main__":
    download_dataset()
    verify_statistics()
