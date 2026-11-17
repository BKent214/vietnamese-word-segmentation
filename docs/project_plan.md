# Kế hoạch và Phân công Dự án: Vietnamese Word Segmentation

**Đề tài**: LSTM và Ứng dụng trong bài toán Word Segmentation  
**Môn học**: Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)  

---

## 1. Bảng Phân công Nhiệm vụ Thành viên

| Thành viên | Phân loại | Công việc chi tiết | Sản phẩm bàn giao (Deliverables) | Hạn chót (DL) |
|---|---|---|---|---|
| **Tâm** (TV01) | Lý thuyết | **Tổng quan bài toán và mục tiêu nghiên cứu**: Tìm hiểu khái niệm, vai trò và đặc điểm của Word Segmentation tiếng Việt; phân tích những khó khăn như nhập nhằng ranh giới từ, từ ghép và từ ngoài từ điển. Xác định mục tiêu tổng thể của seminar: đánh giá hai hướng tiếp cận Hybrid Approach và LSTM trong giải quyết bài toán. | Nội dung tổng quan bài toán; đặt vấn đề nghiên cứu; mục tiêu, câu hỏi nghiên cứu và tiêu chí đánh giá hai phương pháp. 3–5 ví dụ minh họa vấn đề WS trong TV. 1 sơ đồ mô tả bài toán. | Hết ngày 17/11 |
| **Dương** (TV02) | Lý thuyết | **Phương pháp truyền thống kết hợp**: Tìm hiểu nguyên lý Maximum Matching và Bigram N-gram; phân tích cách kết hợp hai phương pháp để tạo phương án triển khai giản lược (phân biệt với kiến trúc gốc) để giải quyết nhập nhằng trong Word Segmentation. Tham khảo nghiên cứu *A Hybrid Approach to Word Segmentation of Vietnamese Texts (2008)*. | Nội dung lý thuyết Hybrid Approach; sơ đồ quy trình hoạt động; ví dụ minh họa; ưu điểm và hạn chế của phương pháp. | Hết ngày 17/11 |
| **Hoàng** (TV08) | Lý thuyết | **Kiến trúc LSTM và khả năng ứng dụng**: Tìm hiểu cơ sở RNN, hạn chế của RNN truyền thống (Vanishing Gradient); trình bày kiến trúc LSTM, Cell State, Hidden State và ba cổng Forget, Input, Output. Phân tích cách LSTM khai thác thông tin tuần tự và ứng dụng vào bài toán Word Segmentation dưới dạng Sequence Labeling. | Nội dung kiến trúc LSTM; sơ đồ minh họa; giải thích cơ chế hoạt động; sơ đồ ứng dụng LSTM vào Word Segmentation. | Hết ngày 17/11 |
| **Hải** (TV03) | Thực nghiệm | **Thu thập dữ liệu**: Tìm kiếm, lựa chọn và thu thập bộ dữ liệu tiếng Việt phục vụ bài toán Word Segmentation. Tìm hiểu nguồn gốc, cấu trúc, quy mô và quy ước phân tách từ của dữ liệu. | Bộ dữ liệu gốc (`data/raw/ud_vtb/`); tài liệu mô tả nguồn dữ liệu, cấu trúc và đặc điểm dữ liệu (`docs/methodology/dataset.md`, `data/raw/README.md`); script tải tự động (`scripts/download_data.py`). | Hết ngày 17/11 |
| **Phát** (TV04) | Thực nghiệm | **Khám phá và tiền xử lý**: Kiểm tra, làm sạch và chuẩn hóa dữ liệu. Phân chia Train/Validation/Test; chuẩn bị dữ liệu đầu vào cho hai hướng tiếp cận. | Notebook tiền xử lý; bộ dữ liệu sau xử lý và chia tập (`data/processed/`). | Hết ngày 19/11 |
| **Thành** (TV05) | Thực nghiệm | **Triển khai phương pháp Hybrid**: Nghiên cứu và triển khai phương pháp kết hợp Maximum Matching và Bigram N-gram. Xây dựng từ điển, tạo các phương án phân tách ứng viên bằng Maximum Matching và sử dụng mô hình Bigram để lựa chọn phương án phù hợp. | Code triển khai Hybrid Approach (`src/hybrid/`); từ điển; mô hình Bigram; kết quả dự đoán trên tập kiểm thử (bao gồm cả trường hợp nhập nhằng và OOV). | Hết ngày 21/11 |
| **Khang** (TV06) | Thực nghiệm | **Triển khai phương pháp LSTM**: Nghiên cứu, xây dựng và huấn luyện mô hình LSTM cho bài toán Word Segmentation. Chuẩn bị biểu diễn chuỗi âm tiết, gán nhãn B/I; theo dõi quá trình huấn luyện và thực hiện dự đoán. | Notebook huấn luyện; mô hình LSTM đã huấn luyện (`src/lstm/`); biểu đồ Loss; kết quả dự đoán trên tập kiểm thử. | Hết ngày 21/11 |
| **Phương** (TV07) | Thực nghiệm | **Đánh giá và so sánh**: Tổng hợp kết quả từ hai hướng tiếp cận (Thành và Khang). Thực hiện đánh giá và so sánh bằng các chỉ số Precision, Recall, F1-score, OOV Recall và thời gian xử lý. Phân tích lỗi phân tách, đặc biệt ở các trường hợp từ ghép, nhập nhằng và từ ngoài từ điển. | Bảng kết quả so sánh; biểu đồ đánh giá; ví dụ dự đoán đúng/sai; Tóm tắt ngắn phân tích ưu điểm và hạn chế của hai phương pháp (chỉ ra điểm mạnh, yếu của LSTM so với Hybrid làm nền). | Hết ngày 23/11 |
| **My** (TV09) | Slide & Thuyết trình | **Slide và Tích hợp nội dung**: Tổng hợp nội dung từ nhóm lý thuyết và thực nghiệm. Xây dựng bố cục, thiết kế slide, trình bày các sơ đồ phương pháp, quy trình thực nghiệm, bảng số liệu và biểu đồ so sánh. | File slide thuyết trình hoàn chỉnh (PowerPoint/Canva). | Hết ngày 25/11 |
| **Phúc** (TV10) | Slide & Thuyết trình | **Kịch bản và Điều phối thuyết trình**: Xây dựng kịch bản thuyết trình; sắp xếp trình tự nội dung; phân chia phần trình bày cho các thành viên. Phối hợp chuẩn bị demo, câu hỏi phản biện và tổ chức luyện tập, kiểm tra tổng thể seminar. | Kịch bản thuyết trình chi tiết (`docs/presentation/speaker_notes.md`); kịch bản demo (`docs/presentation/demo_guide.md`); điều phối buổi báo cáo. | 27/11 Thuyết trình |

---

## 2. Kế hoạch Tiến độ Tổng thể (Timeline)

```text
13–14/11: Khởi tạo dự án & cấu trúc mã nguồn Git
15–17/11: Nghiên cứu lý thuyết & Thu thập dữ liệu (Tâm, Dương, Hoàng, Hải)
18–19/11: Tiền xử lý dữ liệu & Gán nhãn BIO (Phát)
20–21/11: Triển khai mô hình Hybrid & Huấn luyện LSTM (Thành, Khang)
22–23/11: Đánh giá thực nghiệm & Phân tích lỗi (Phương)
24–25/11: Hoàn thiện Slide, Demo Streamlit & Thử nghiệm thuyết trình (My, Phúc)
26–27/11: Báo cáo Seminar
```
