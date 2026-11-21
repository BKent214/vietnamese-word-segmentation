# Mô hình BiLSTM cho Bài toán Tách từ Tiếng Việt (Word Segmentation)

Module này cài đặt mô hình mạng nơ-ron hồi quy hai chiều (**BiLSTM**) để giải quyết bài toán tách từ tiếng Việt dưới dạng **Gán nhãn chuỗi (Sequence Labeling)**.

---

## 1. Cấu trúc Thư mục Module

```text
src/lstm/
├── config.py      # Quản lý tập trung toàn bộ đường dẫn và siêu tham số (hyperparameters)
├── dataset.py     # Xây dựng Vocabulary, mã hóa ID, Padding động và tạo Boolean Mask
├── model.py       # Định nghĩa kiến trúc mạng: Embedding -> BiLSTM -> Linear Classifier
├── train.py       # Vòng lặp huấn luyện, tính Validation F1, lưu Checkpoint và vẽ biểu đồ Loss
├── predict.py     # Nạp mô hình đã huấn luyện, dự đoán nhãn BIO và tái tạo từ ghép
└── README.md      # Tài liệu hướng dẫn kỹ thuật và sử dụng module
```

---

## 2. Phát biểu Bài toán (Task Formulation)

Đầu vào là một câu tiếng Việt đã được tách sơ bộ thành các âm tiết cách nhau bởi khoảng trắng. Mô hình dự đoán một nhãn theo chuẩn **BI** (biến thể rút gọn của BIO) cho từng âm tiết:
- **`B` (Begin):** Âm tiết bắt đầu một từ mới.
- **`I` (Inside):** Âm tiết tiếp nối âm tiết đứng trước để tạo thành từ ghép.

**Ví dụ minh họa:**
```text
text:      Hiện nay xã có 68 tổ nhân dân .
labels:    B    I   B  B  B  B  B    I   B
segmented: Hiện_nay xã có 68 tổ nhân_dân .
```

**Quy tắc ràng buộc & Tái tạo từ:**
- Độ dài chuỗi âm tiết đầu vào và chuỗi nhãn dự đoán phải bằng nhau tuyệt đối.
- Khi tái tạo từ, các âm tiết liền kề mang nhãn `I` sẽ được nối với âm tiết trước đó bằng dấu gạch dưới (`_`). Nếu một chuỗi bắt đầu bằng nhãn `I` (do mô hình dự đoán nhiễu), quy tắc tái tạo sẽ tự động coi âm tiết đó là điểm bắt đầu của một từ mới.

---

## 3. Yêu cầu Định dạng Dữ liệu Đầu vào (Data Schema)

Các file `train.csv`, `dev.csv`, `test.csv` trong thư mục `data/processed/` bắt buộc phải có các cột sau:

| Tên cột | Mô tả | Ví dụ |
| :--- | :--- | :--- |
| `text` | Câu gốc chưa tách từ (các âm tiết cách nhau bởi 1 dấu cách) | `Hiện nay xã có 68 tổ nhân dân .` |
| `labels` | Chuỗi nhãn `B` / `I` tương ứng với từng âm tiết | `B I B B B B B I B` |
| `segmented` | *(Tùy chọn lúc train, bắt buộc lúc test)* Câu chuẩn đã nối `_` | `Hiện_nay xã có 68 tổ nhân_dân .` |

---

## 4. Kiến trúc Mô hình (Architecture)

1. **Vocabulary (`dataset.py`):** Ánh xạ mỗi âm tiết (đã chuyển về chữ thường) sang một số nguyên ID. Quy ước `<PAD> = 0`, `<UNK> = 1`. Từ điển **chỉ được xây dựng từ tập huấn luyện (`train.csv`)** nhằm chống rò rỉ dữ liệu (data leakage). Các âm tiết hiếm (tần suất nhỏ hơn `min_freq`) hoặc chưa từng gặp sẽ được ánh xạ về `<UNK>`.
2. **Embedding Layer:** Chuyển đổi từng ID âm tiết rời rạc thành vector biểu diễn liên tục (dense vector) kích thước `embedding_dim` (mặc định `128`).
3. **BiLSTM Layer:** Đọc chuỗi vector theo cả hai chiều (trái sang phải và phải sang trái) để nắm bắt ngữ cảnh xung quanh và sinh ra vector đặc trưng ngữ cảnh tại mỗi vị trí âm tiết.
4. **Linear Classifier:** Tầng tuyến tính chiếu từng vector ngữ cảnh về không gian 3 lớp (`<PAD>`, `B`, `I`).
5. **Argmax Decoding:** Chọn nhãn có xác suất cao nhất tại từng vị trí âm tiết thực. Trong quá trình dự đoán (inference), điểm số (logit) của lớp `<PAD>` bị gán về `-1e9` để đảm bảo mô hình không bao giờ dự đoán ra nhãn đệm.

> **Lưu ý thiết kế:** Đây là mô hình phân loại ở mức token độc lập (token-level classification), không sử dụng tầng CRF (Conditional Random Fields). Mô hình không áp đặt ràng buộc chuyển trạng thái Markov giữa các nhãn liên tiếp ngoài quy tắc hậu xử lý lúc tái tạo từ.

---

## 5. Xử lý Padding, Masking và Hàm mất mát

- **Dynamic Padding:** Các câu trong cùng một batch được đệm động (`pad_sequence`) theo độ dài của câu dài nhất trong batch đó thay vì đệm cố định toàn bộ dataset, giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán.
- **Sequence Packing:** Sử dụng `pack_padded_sequence` trước khi đưa vào BiLSTM để mạng bỏ qua hoàn toàn các bước thời gian chứa `<PAD>`, tránh làm nhiễu trạng thái ẩn (`Hidden State` và `Cell State`).
- **Masked Loss & Metrics:** Hàm mất mát `CrossEntropyLoss(ignore_index=0)` bỏ qua hoàn toàn các vị trí nhãn `<PAD>`. Một mặt nạ Boolean (`mask = input_ids != 0`) cũng được dùng để loại bỏ phần đệm khi tính độ đo đánh giá.

---

## 6. Quy trình Huấn luyện (Training Workflow)

1. Đọc file `train.csv` và xây dựng từ điển `Vocabulary` chỉ dựa trên cột `text`.
2. Khởi tạo `Dataset` và `DataLoader` cho hai tập `train` và `dev` dùng chung bộ từ điển vừa tạo.
3. Huấn luyện mô hình bằng thuật toán tối ưu **AdamW** và hàm mất mát **Cross-Entropy**.
4. Áp dụng kỹ thuật cắt ngưỡng gradient (**Gradient Clipping**, mặc định `5.0`) để chống bùng nổ gradient (exploding gradients).
5. Đánh giá trên tập `dev` sau mỗi epoch bằng độ đo **Token-level Macro-F1** (tính trung bình F1 trên 2 nhãn `B` và `I`) và lưu lại checkpoint có điểm `dev_f1` cao nhất.
6. Xuất các tệp kết quả đầu ra.

**Lệnh chạy huấn luyện (chạy từ thư mục gốc của dự án):**
```bash
python -m src.lstm.train
```
*(Có thể chỉnh sửa đường dẫn và các siêu tham số như `hidden_dim`, `dropout`, `learning_rate`, `epochs` trong file `src/lstm/config.py` trước khi chạy).*

**Các tệp đầu ra (Artifacts) sau khi huấn luyện:**
- `checkpoints/lstm/best_lstm_model.pt`: Chứa trọng số mô hình (`model_state_dict`), siêu tham số kiến trúc (`model_hparams`) và điểm `best_dev_f1`.
- `checkpoints/lstm/vocab.json`: Từ điển ánh xạ âm tiết sang ID dùng cho quá trình suy luận.
- `outputs/lstm/loss_f1_curve.png`: Biểu đồ trực quan hóa Loss và F1-score qua từng epoch trên tập Train và Dev.

---

## 7. Hướng dẫn Dự đoán (Prediction / Inference)

**Dự đoán trên từng câu đơn lẻ (dùng để tích hợp vào Web Demo):**
```python
from pathlib import Path
from src.lstm.predict import WordSegmenterPredictor

predictor = WordSegmenterPredictor(
    checkpoint_path=Path("checkpoints/lstm/best_lstm_model.pt"),
    vocab_path=Path("checkpoints/lstm/vocab.json")
)

segmented, tags = predictor.predict_sentence("Hiện nay xã có 68 tổ nhân dân .")
print("Câu đã tách từ:", segmented)      # Hiện_nay xã có 68 tổ nhân_dân .
print("Chuỗi nhãn BIO:", " ".join(tags)) # B I B B B B B I B
```

**Dự đoán hàng loạt trên file CSV (xuất thêm 2 cột `pred_segmented` và `pred_labels`):**
```python
predictor.predict_csv(
    input_csv=Path("data/processed/test.csv"),
    output_csv=Path("outputs/lstm/test_predictions.csv")
)
```

---

## 8. Lưu ý khi Đánh giá và So sánh Mô hình (Evaluation Protocol)

- **Phân biệt Validation F1 và Final Word F1:** Trong quá trình huấn luyện, script `train.py` báo cáo chỉ số **Token-level Macro-F1** (độ chính xác gán nhãn `B`/`I` trên từng âm tiết) để chọn checkpoint nhanh. Đây **không phải** là điểm F1 tách từ cuối cùng.
- **Tiêu chuẩn đánh giá cuối cùng (Tổ 4 thực hiện):** Cần so khớp chính xác từng từ hoàn chỉnh (**Exact Match Word-level**) giữa câu dự đoán (`pred_segmented`) và câu chuẩn (`segmented`). Toàn bộ các phương pháp (Baseline, Hybrid, BiLSTM) phải được đánh giá trên cùng một tập `test.csv` độc lập với cùng bộ độ đo: **Precision, Recall, F1-score, OOV Recall** và **Thời gian suy luận (Inference Time)**.
- **Định nghĩa từ ngoài từ điển (OOV - Out-of-Vocabulary):** Một từ trong tập kiểm thử (gold word) được coi là OOV nếu **toàn bộ từ ghép đó** không xuất hiện dưới dạng một từ hoàn chỉnh trong tập huấn luyện (`train.csv`). Chỉ số **OOV Recall** là tỷ lệ các từ OOV chuẩn được mô hình phân tách chính xác tuyệt đối. Định nghĩa này cần được trình bày rõ trong báo cáo và slide.

---

## 9. Hạn chế của Mô hình Hiện tại & Hướng Phát triển

- **Chưa có bộ mã hóa mức ký tự (Character/Sub-syllable Encoder):** Các âm tiết chưa từng xuất hiện trong tập train đều bị gộp chung về một token `<UNK>`, khiến mô hình mất đi thông tin hình thái học (ví dụ: chữ viết hoa của tên riêng, chữ số, hoặc dấu câu).
- **Thiếu tầng CRF:** Việc dự đoán nhãn độc lập bằng Softmax tại từng vị trí đôi khi có thể sinh ra chuỗi nhãn phi logic (ví dụ nhãn `I` đứng ngay đầu câu). Bổ sung tầng **BiLSTM-CRF** với giải thuật giải mã Viterbi sẽ khắc phục triệt để nhược điểm này.
- **Không hỗ trợ xử lý trực tuyến (Streaming):** Vì BiLSTM sử dụng ngữ cảnh cả bên trái và bên phải của câu, mô hình cần nhận trọn vẹn cả câu trước khi đưa ra dự đoán.
- **Tính nhất quán dữ liệu:** Quy ước gán nhãn `B`/`I` và cách chuẩn hóa dấu câu/khoảng trắng phải được giữ đồng nhất tuyệt đối xuyên suốt từ khâu tiền xử lý, huấn luyện, dự đoán cho đến đánh giá.