# Hướng dẫn demo tách từ tiếng Việt

Phụ trách: TV10 Phúc. Demo dùng giao diện Gradio trong `app/gradio_app.py`.

## Phân chia thời lượng seminar

| Mốc giờ | Người trình bày | Nội dung |
|---|---|---|
| 00:00-07:00 | Dương | Slide 1-13: tổng quan và Hybrid |
| 07:00-20:00 | Phúc | Slide 14-32: RNN/LSTM, triển khai, dữ liệu, kết quả |
| 20:00-30:00 | Phúc | Slide 33 và demo trực tiếp |
| 30:00-45:00 | Phúc trả lời chính | Vấn đáp |

Dương bàn giao sau slide 13: “Phần tổng quan và Hybrid của em kết thúc tại đây. Sau đây em mời Phúc trình bày từ nguyên lý RNN/LSTM đến triển khai và kết quả thực nghiệm.”

Phúc nhận phần ở slide 14: “Em cảm ơn Dương. Tiếp theo em trình bày nguyên lý RNN/LSTM, cách nhóm triển khai, kết quả thực nghiệm và phần demo.”

## Chuẩn bị trên Windows

Mở PowerShell tại thư mục gốc repository, nơi có `requirements.txt`, `app/` và `src/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install gradio
```

Gradio cần được cài trong cùng môi trường Python chạy ứng dụng. Các lần mở sau không cần tạo lại môi trường nếu đã cài đủ thư viện.

### Tài nguyên cần có

```text
data/processed/train.csv
checkpoints/lstm/best_lstm_model.pt
checkpoints/lstm/vocab.json
```

Dùng checkpoint và vocabulary tương ứng với lần thực nghiệm được trình bày. Checkpoint `.pt` có thể không được theo dõi trong Git; nếu thiếu, nhận đúng cặp file từ người huấn luyện. Không huấn luyện lại trong thời gian demo.

### Khởi chạy

```powershell
.\.venv\Scripts\python.exe -m app.gradio_app
```

Mở địa chỉ local do terminal in ra và giữ terminal mở trong lúc trình chiếu. Cổng có thể thay đổi nếu cổng mặc định đang được dùng.

Trước buổi nói, chạy thử một câu và kiểm tra:

- Tab **So sánh các phương pháp** có Whitespace, Regex, Maximum Matching, Bigram (Hybrid) và LSTM.
- Dòng LSTM không báo thiếu checkpoint.
- Tab **Chi tiết BiLSTM** hiện ID, kích thước các tầng và chuỗi nhãn B/I.
- Có ảnh hoặc video lưu kết quả để dự phòng.

## Kịch bản demo 10 phút

Mốc dưới đây tính riêng từ lúc bắt đầu demo, tương ứng phút 20 của seminar. Phúc vừa nói vừa thao tác.

| Thời gian demo | Thao tác | Điểm giải thích |
|---|---|---|
| 00:00-00:45 | Mở Gradio, chỉ ô nhập và nút Phân tích | Các mô hình nhận cùng một câu; tô màu không tự có nghĩa dự đoán đúng |
| 00:45-02:00 | Nhập `Sinh viên đang học lập trình .` | So ranh giới ở sinh viên và lập trình; đọc đúng đầu ra đang hiện |
| 02:00-03:30 | Dán nguyên câu test id 166 hoặc 268 đã chuẩn bị | So vùng nhà hàng xóm giữa Maximum Matching, Bigram và gold |
| 03:30-05:00 | Dùng nguyên câu test id 29 | Quan sát cửu_vạn; minh họa từ nhiều âm tiết OOV |
| 05:00-07:30 | Mở Chi tiết BiLSTM | Âm tiết -> ID -> Embedding -> BiLSTM -> nhãn B/I -> ghép từ |
| 07:30-08:30 | Mở một lỗi thật đã kiểm tra trước, kèm gold | Ví dụ tách rời giật_mình hoặc nối nhầm; mô hình vẫn có lỗi |
| 08:30-09:30 | Nhận một câu ngắn từ khán giả hoặc dùng câu dự phòng | Nhận xét ranh giới; không tính F1 cho câu chưa có gold |
| 09:30-10:00 | Chốt demo và mời câu hỏi | Kết luận dựa trên cả test; chuyển sang 15 phút vấn đáp |

Câu dự phòng: `Tôi thích học khoa học dữ liệu .`

### Chuẩn bị câu có gold

Mở `data/processed/test.csv`, lọc cột `id` theo các giá trị 2, 29, 166, 268 và lưu nguyên các cột `text`, `segmented`. Đây là mã câu, không phải số dòng hiển thị trong bảng tính.

- Id 29 được tài liệu phân tích dùng để minh họa cửu_vạn. Slide 33 có đoạn “Theo chân cửu vạn , chúng tôi đã chứng kiến”.
- Id 166 và 268 được dùng minh họa nhập nhằng nhà hàng xóm.
- Id 2 được tài liệu nêu ở lỗi giật_mình của BiLSTM.

Kiểm tra lại đầu ra trên checkpoint đang dùng. Ưu tiên nguyên câu có ngữ cảnh đầy đủ. Chỉ khẳng định mô hình sửa được một lỗi khi đầu ra và gold thực tế thể hiện điều đó. Nếu kết quả khác slide, đọc đúng màn hình; không thay đổi dữ liệu hoặc từ điển để ép một kết quả đẹp.

### Lời dẫn mẫu

Mở demo: “Em sẽ cho các mô hình xử lý cùng một câu, sau đó so những tiếng được ghép thành từ. Tiếp theo, em mở phần chi tiết để minh họa cách BiLSTM dự đoán nhãn B và I.”

Giải thích B/I: “B bắt đầu một từ mới, I nối vào từ đang hình thành. Chẳng hạn, khi cửu là B và vạn là I, hai tiếng được ghép thành cửu_vạn.” Chỉ dùng ví dụ này nếu nhãn thực tế đang hiển thị đúng như vậy.

Kết demo: “Demo giúp thấy cách mô hình xác định ranh giới. Kết luận tổng thể vẫn dựa trên 800 câu test. Nhóm xin nhận câu hỏi của thầy cô và các bạn.”

## Những điểm cần nói chính xác

- OOV trong đánh giá là cả từ không có trong tập từ của train. UNK trong giao diện là âm tiết chưa có hoặc quá hiếm trong vocabulary. Hai khái niệm này khác nhau.
- Ô xanh trong kết quả so sánh đánh dấu từ nhiều âm tiết. Trong tab chi tiết, màu nhãn B/I có ý nghĩa riêng.
- Phần trăm nhãn là điểm xác suất của mô hình, không phải cam kết tỷ lệ dự đoán đúng.
- Tab chi tiết trình bày các bước suy luận, kích thước và đầu ra nhãn; không phải hình trực quan từng cổng Forget/Input/Output.
- Bảng thời gian của ứng dụng là trung bình 10 lượt trên CPU cho một câu. Bảng trong slide đo tổng một vòng dự đoán từng câu trên 800 câu; không so trực tiếp hai con số.
- Mã demo giới hạn tối đa 120 đơn vị đầu vào. Nên dùng câu ngắn khi tương tác.
- Ba hướng tiếp cận gồm năm mô hình cụ thể. Phân biệt cách gọi này khi trình bày.

## Xử lý trục trặc

| Tình huống | Cách xử lý |
|---|---|
| Thiếu thư viện | Cài bằng Python trong `.venv`, rồi mở lại ứng dụng |
| Thiếu checkpoint hoặc vocab | Nhận đúng cặp file từ người huấn luyện; không trình bày baseline thành BiLSTM |
| Chạy quá 10 giây chưa có kết quả | Chuyển sang ảnh/video dự phòng, nói rõ đó là kết quả đã lưu |
| Đầu ra khác slide | Đọc đúng đầu ra hiện tại; đối chiếu câu đầy đủ, checkpoint và tiền xử lý sau buổi nói |
| Sắp hết giờ | Bỏ câu mới từ khán giả; giữ ví dụ OOV, B/I và kết luận; vào vấn đáp đúng phút 30 |

Lời dẫn khi lỗi: “Ứng dụng đang gặp trục trặc khi chạy trực tiếp. Em chuyển sang kết quả đã lưu của cùng ví dụ để tiếp tục giải thích.”

## Checklist trước seminar

- [ ] Đã kiểm tra checkpoint, vocabulary và dữ liệu trên máy trình chiếu.
- [ ] Đã chạy thử các câu sẽ dùng và lưu gold, đầu ra thực tế.
- [ ] Đã mở sẵn slide 33, ứng dụng local và kết quả dự phòng.
- [ ] Đã tập một lần chuyển sang ảnh khi demo lỗi.
- [ ] Dương kết thúc slide 13 khoảng phút 7; Phúc kết thúc slide 32 ở phút 20.
- [ ] Demo kết thúc ở phút 30, giữ 15 phút cho vấn đáp.
