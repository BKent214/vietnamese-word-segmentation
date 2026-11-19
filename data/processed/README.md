# Dữ liệu đã xử lý

Tạo bằng lệnh (chạy từ thư mục gốc repo):

```
python scripts/prepare_data.py
```

Gồm 3 file `train.csv`, `dev.csv`, `test.csv`. Mỗi dòng là một câu:

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `id` | Số thứ tự câu | `2` |
| `text` | Câu chưa tách từ (các âm tiết cách nhau bởi dấu cách) | `Hiện nay xã có 68 tổ nhân dân .` |
| `segmented` | Câu đã tách từ, âm tiết trong một từ nối bằng `_` | `Hiện_nay xã có 68 tổ nhân_dân .` |
| `labels` | Nhãn B/I của từng âm tiết | `B I B B B B B I B` |

`text.split()` và `labels.split()` luôn có cùng độ dài.

## Cách đọc

```python
import pandas as pd

train = pd.read_csv("data/processed/train.csv")

# LSTM (Khang): chuỗi âm tiết và nhãn
syllables = train["text"][0].split()
labels = train["labels"][0].split()

# Hybrid (Thành): danh sách từ để xây từ điển và Bigram
from src.data.annotation import from_segmented_line
words = from_segmented_line(train["segmented"][0])
```

## Quy ước

- Từ điển, Bigram và vocabulary của LSTM chỉ xây từ `train.csv`. `dev.csv` để chỉnh tham số, `test.csv` chỉ dùng để đánh giá cuối.
- Đầu vào khi dự đoán là cột `text` của `test.csv`. File dự đoán nộp cho đánh giá cần có cột `id` và `segmented` cùng định dạng, cùng thứ tự câu.
- Mở bằng Excel vẫn hiển thị đúng tiếng Việt (file lưu dạng UTF-8 có BOM).
