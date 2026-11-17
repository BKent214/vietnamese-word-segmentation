# Báo cáo Thu thập và Mô tả Dữ liệu: UD Vietnamese-VTB

**Học phần**: Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)  
**Chủ đề**: Vietnamese Word Segmentation – From Baseline to LSTM  
**Thành viên phụ trách**: Hải (TV03 – Nhóm thực nghiệm – Thu thập dữ liệu)  
**Ngày hoàn thành**: 17/11/2026  

---

## 1. Giới thiệu và Lựa chọn Bộ dữ liệu

Trong bài toán Phân tách từ tiếng Việt (Vietnamese Word Segmentation), chất lượng và quy chuẩn gán nhãn của tập dữ liệu ngữ liệu đóng vai trò quyết định đến độ chính xác của các mô hình học máy và thống kê. Khác với tiếng Anh (ranh giới từ phần lớn trùng khớp với khoảng trắng), tiếng Việt là ngôn ngữ đơn lập, khoảng trắng chủ yếu phân cách âm tiết (syllable), và một từ vựng có thể gồm một, hai hoặc nhiều âm tiết liên kết về ngữ nghĩa và ngữ pháp.

Sau khi khảo sát các nguồn ngữ liệu tiếng Việt công khai (như VLSP 2013, VLSP 2016, Universal Dependencies), nhóm nghiên cứu đã lựa chọn bộ dữ liệu:

**UD Vietnamese-VTB (Universal Dependencies Vietnamese Treebank)**.

### Lý do lựa chọn:
1. **Nguồn gốc chuẩn mực**: Ngữ liệu có nguồn gốc từ Vietnamese Treebank (VTB) thuộc dự án VLSP (Vietnamese Language and Speech Processing), bao gồm các văn bản báo chí chính thống với văn phong đa dạng và cấu trúc ngữ pháp chuẩn mực.
2. **Quy chuẩn quốc tế**: Dữ liệu đã được chuẩn hóa theo chuẩn quốc tế Universal Dependencies (CoNLL-U format), đảm bảo tính nhất quán cao trong phân định ranh giới từ và từ loại.
3. **Phân chia sẵn chuẩn tắc (Pre-split)**: Dữ liệu đã được phân tách sẵn thành ba tập: `train`, `dev` (validation) và `test`, giúp đảm bảo tính khách quan trong đánh giá và so sánh thực nghiệm giữa phương pháp Hybrid và LSTM.
4. **Khả năng tiếp cận và tái lập**: Kho dữ liệu được phát hành mở trên GitHub, thuận tiện cho việc tải trực tiếp, kiểm thử và nhân rộng nghiên cứu mà không gặp rào cản truy cập.

- **Kho lưu trữ chính thức**: [https://github.com/UniversalDependencies/UD_Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB)
- **Tài liệu tham khảo UD**: [https://universaldependencies.org/treebanks/vi_vtb/](https://universaldependencies.org/treebanks/vi_vtb/)
- **Tài liệu nguồn gốc VTB**: Nguyen, T. M. H., Nguyen, P. T., & Tran, H. Q. (2009). *Building a Large Syntactically-Annotated Corpus for Vietnamese*. Proceedings of the Third Linguistic Annotation Workshop (LAW III), ACL-IJCNLP 2009, pp. 147–154. [https://aclanthology.org/W09-3035/](https://aclanthology.org/W09-3035/)

---

## 2. Quy mô và Thống kê Dữ liệu

Toàn bộ các tệp gốc được thu thập và lưu trữ tại `data/raw/ud_vtb/`. Kết quả thống kê chi tiết từ ba tệp dữ liệu gốc như sau:

| Tập dữ liệu | Tệp dữ liệu | Dung lượng | Số câu (`sent_id`) | Số token từ (Cột `FORM`) | Số đơn vị đầu vào | Số âm tiết ước tính |
|---|---|---|---|---|---|---|
| **Train** | `vi_vtb-ud-train.conllu` | 947 KB | 1.400 | 20.215 | 24.973 | 21.345 |
| **Dev** (Validation) | `vi_vtb-ud-dev.conllu` | 1.23 MB | 1.123 | 26.162 | 31.927 | 27.390 |
| **Test** | `vi_vtb-ud-test.conllu` | 539 KB | 800 | 11.692 | 13.857 | 11.839 |
| **Tổng cộng** | - | **2.72 MB** | **3.323** | **58.069** | **70.757** | **60.574** |

### Phương pháp tính toán và giải thích chỉ số:
- **Số câu**: Tổng số định danh `# sent_id` trong tập dữ liệu.
- **Số token từ (`words`)**: Đếm tổng số đơn vị từ vựng theo quy ước của cột `FORM`. Mỗi đơn vị này có thể là từ đơn (`nhà`, `đi`), từ ghép/từ phái sinh (`nhân dân`, `phụ trách`, `khoa học`), dấu câu (`.`, `,`) hoặc số (`68`, `40`).
- **Số đơn vị đầu vào (`tokens`)**: Được tính bằng cách lấy từng chuỗi trong cột `FORM` và phân tách theo khoảng trắng. Đây là số lượng đơn vị ký hiệu tối thiểu mà hệ thống nhận vào trước khi thực hiện ghép từ.
- **Số âm tiết ước tính (`est_syllables`)**: Được tính bằng cách tách cột `FORM` theo khoảng trắng nhưng đã loại trừ các token thuộc nhóm dấu câu (`PUNCT`), ký hiệu đặc biệt (`SYM`) và số (`NUM`).
  - *Lưu ý*: Vì dữ liệu báo chí vẫn chứa tên riêng viết hoa, từ ngoại lai mượn tiếng Pháp/Anh và các danh xưng viết tắt, con số **60.574** là số liệu định lượng tham khảo khoa học, phản ánh sát thực tế ngôn ngữ học của ngữ liệu.

---

## 3. Cấu trúc và Định dạng CoNLL-U

Dữ liệu được lưu trữ theo định dạng bảng chuẩn CoNLL-U (10 trường ngăn cách bởi dấu tab):

```text
# sent_id = train-s1
# text = Hiện nay xã có 68 tổ nhân dân , mỗi tổ phụ trách 40 gia đình .
1	Hiện nay	_	NOUN	N	_	3	obl:tmod	_	_
2	xã	_	NOUN	N	_	3	nsubj	_	_
3	có	_	VERB	V	_	0	root	_	_
4	68	_	NUM	M	_	5	nummod	_	_
5	tổ	_	NOUN	N	_	3	obj	_	_
6	nhân dân	_	NOUN	N	_	5	compound	_	_
7	,	_	PUNCT	`	_	9	punct	_	_
8	mỗi	_	DET	L	_	9	det	_	_
9	tổ	_	NOUN	N	_	10	nsubj	_	_
10	phụ trách	_	VERB	V	_	3	parataxis	_	_
11	40	_	NUM	M	_	12	nummod	_	_
12	gia đình	_	NOUN	N	_	10	obj	_	_
13	.	_	PUNCT	.	_	3	punct	_	_
```

### Phân tích đặc trưng tách từ:
- Cột **`FORM` (Cột 2)** chứa từ vựng hoàn chỉnh đã được xác định ranh giới.
- Đối với từ ghép nhiều âm tiết như `Hiện nay`, `nhân dân`, `phụ trách`, `gia đình`, các âm tiết thành phần được liên kết trong cùng một trường `FORM` và ngăn cách bằng một dấu cách đơn.
- Khi chuyển đổi sang biểu diễn chuẩn nối từ bằng dấu gạch dưới (`_`), câu văn bản trở thành:
  > `Hiện_nay xã có 68 tổ nhân_dân , mỗi tổ phụ_trách 40 gia_đình .`
- Điều này chứng minh rằng việc phân tách đơn thuần bằng dấu cách (Whitespace baseline) sẽ phân rã sai các từ đa âm tiết, khẳng định tính cần thiết của mô hình Hybrid và LSTM.

---

## 4. Quy ước Phân tách từ và Hướng dẫn Khai thác Dữ liệu

### 4.1. Quy ước gán nhãn BIO cho Mô hình LSTM
Mô hình LSTM giải quyết bài toán Word Segmentation dưới dạng bài toán gán nhãn chuỗi (Sequence Labeling). Từ cột `FORM`, pipeline tiền xử lý sẽ chuyển đổi câu thành chuỗi âm tiết và nhãn tương ứng theo sơ đồ **B-I (Beginning - Inside)**:
- **`B` (Beginning)**: Âm tiết bắt đầu một từ mới (hoặc từ đơn 1 âm tiết).
- **`I` (Inside)**: Âm tiết tiếp theo nằm trong cùng một từ ghép với âm tiết trước đó.

**Ví dụ chuyển đổi**:
```text
Âm tiết: Hiện     nay     xã     có     68     tổ     nhân     dân     ,     mỗi     tổ     phụ      trách    40     gia      đình     .
Nhãn:    B        I       B      B      B      B      B        I       B     B       B      B        I        B      B        I        B
```

### 4.2. Xây dựng Tài nguyên cho Mô hình Hybrid
- **Từ điển (Lexicon/Dictionary)**: Trích xuất toàn bộ các từ đa âm tiết và đơn âm tiết từ cột `FORM` trong tập `train` để tạo tập từ vựng chuẩn. Tuyệt đối không dùng tập `test` để xây dựng từ điển nhằm tránh rò rỉ dữ liệu (data leakage).
- **Mô hình Bigram**: Đếm tần suất xuất hiện đồng thời của các cặp từ liền kề $(w_i, w_{i+1})$ trên tập `train` để tính toán ma trận xác suất chuyển tiếp giải quyết nhập nhằng (ambiguity resolution).

---

## 5. Dữ liệu Bàn giao và Quản lý Mã nguồn

1. **Vị trí lưu trữ dữ liệu thô**: Toàn bộ ba tệp dữ liệu gốc được lưu trữ tại `data/raw/ud_vtb/`:
   - `data/raw/ud_vtb/vi_vtb-ud-train.conllu`
   - `data/raw/ud_vtb/vi_vtb-ud-dev.conllu`
   - `data/raw/ud_vtb/vi_vtb-ud-test.conllu`
2. **Nguyên tắc quản lý**: Dữ liệu gốc được lưu trữ cố định và không can thiệp sửa đổi trực tiếp.
3. **Mã nguồn tự động hóa**: Nhóm cung cấp script `scripts/download_data.py` để tự động hóa việc tải và kiểm định tính toàn vẹn của dữ liệu trong trường hợp thiết lập môi trường mới.

---

## 6. Tài liệu Tham khảo

1. Le, H. P., Nguyen, T. M. H., Roussanaly, A., & Ho, T. V. (2008). *A Hybrid Approach to Word Segmentation of Vietnamese Texts*. In Proceedings of LATA 2008, LNCS vol. 5196, pp. 240–249.
2. Nguyen, T. M. H., Nguyen, P. T., & Tran, H. Q. (2009). *Building a Large Syntactically-Annotated Corpus for Vietnamese*. In Proceedings of LAW III, ACL-IJCNLP 2009, pp. 147–154.
3. Universal Dependencies Consortium. (2023). *Universal Dependencies Vietnamese Treebank (UD_Vietnamese-VTB)*. [https://github.com/UniversalDependencies/UD_Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB).
