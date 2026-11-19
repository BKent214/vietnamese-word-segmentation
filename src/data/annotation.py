"""
Chuyển đổi giữa danh sách từ, nhãn B/I và dòng văn bản đã tách từ.
Thực hiện: Phát (TV04)

    B: âm tiết bắt đầu một từ
    I: âm tiết nằm bên trong từ đang xét
"""


def words_to_labels(words):
    """
    ["Hiện nay", "xã"]  ->  (["Hiện", "nay", "xã"], ["B", "I", "B"])
    """
    syllables = []
    labels = []

    for word in words:
        parts = word.split()
        syllables.extend(parts)
        labels.extend(["B"] + ["I"] * (len(parts) - 1))

    return syllables, labels


def labels_to_words(syllables, labels):
    """
    Ghép ngược nhãn B/I thành từ (dùng sau khi LSTM dự đoán).
    (["Hiện", "nay", "xã"], ["B", "I", "B"])  ->  ["Hiện nay", "xã"]

    Nếu câu bắt đầu bằng I (mô hình đoán sai) thì coi như B.
    """
    words = []

    for syllable, label in zip(syllables, labels):
        if label == "B" or not words:
            words.append(syllable)
        else:
            words[-1] += " " + syllable

    return words


def to_segmented_line(words):
    """
    ["Hiện nay", "xã"]  ->  "Hiện_nay xã"
    """
    return " ".join(word.replace(" ", "_") for word in words)


def from_segmented_line(line):
    """
    "Hiện_nay xã"  ->  ["Hiện nay", "xã"]
    """
    return [word.replace("_", " ") for word in line.split()]
