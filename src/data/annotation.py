"""
Chuyển đổi giữa danh sách từ, nhãn B/I và dòng văn bản đã tách từ.
    B: âm tiết bắt đầu một từ
    I: âm tiết nằm bên trong từ đang xét
"""
def words_to_labels(words):
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
    """
    words = []
    for syllable, label in zip(syllables, labels):
        if label == "B" or not words:
            words.append(syllable)
        else:
            words[-1] += " " + syllable
    return words
def to_segmented_line(words):
    return " ".join(word.replace(" ", "_") for word in words)
def from_segmented_line(line):
    return [word.replace("_", " ") for word in line.split()]
