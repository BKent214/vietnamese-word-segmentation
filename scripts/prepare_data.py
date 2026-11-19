"""
Tạo dữ liệu đã xử lý từ data/raw/ud_vtb/ và lưu vào data/processed/.
Kết quả: train.csv, dev.csv, test.csv. Mỗi dòng là một câu, gồm 4 cột:
    id         : số thứ tự câu
    text       : câu chưa tách từ      Hiện nay xã có 68 tổ nhân dân .
    segmented  : câu đã tách từ        Hiện_nay xã có 68 tổ nhân_dân .
    labels     : nhãn B/I từng âm tiết B I B B B B B I B
Cột text và labels có cùng số phần tử khi tách theo khoảng trắng.
"""
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import read_conllu
from src.data.annotation import words_to_labels, labels_to_words, to_segmented_line
RAW_DIR = "data/raw/ud_vtb"
OUT_DIR = "data/processed"
SPLITS = ["train", "dev", "test"]
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in SPLITS:
        sentences = read_conllu(f"{RAW_DIR}/vi_vtb-ud-{split}.conllu")
        errors = 0

        # utf-8-sig để Excel hiển thị đúng tiếng Việt có dấu
        with open(f"{OUT_DIR}/{split}.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "segmented", "labels"])

            for i, words in enumerate(sentences, start=1):
                syllables, labels = words_to_labels(words)

                # Kiểm tra hai chiều: ghép nhãn lại phải ra đúng câu gốc
                if labels_to_words(syllables, labels) != words:
                    errors += 1
                writer.writerow([i, " ".join(syllables), to_segmented_line(words), " ".join(labels)])
        print(f"{split:5s}: {len(sentences)} câu -> {OUT_DIR}/{split}.csv, lỗi chuyển đổi: {errors}")

if __name__ == "__main__":
    main()
