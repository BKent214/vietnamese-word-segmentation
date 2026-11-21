"Đọc dữ liệu CoNLL-U và chuẩn hóa văn bản."
import unicodedata

def read_conllu(path):
    """
    Đọc file .conllu, trả về danh sách câu.
    Mỗi câu là danh sách từ (lấy từ cột FORM).
    """
    sentences = []
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):      # dòng chú thích: sent_id, text
                continue
            if line == "":                # dòng trống = hết một câu
                if words:
                    sentences.append(words)
                    words = []
                continue
            columns = line.split("\t")
            words.append(columns[1])      # cột 2 là FORM
    if words:                             # câu cuối nếu file không kết thúc bằng dòng trống
        sentences.append(words)
    return sentences
def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())
