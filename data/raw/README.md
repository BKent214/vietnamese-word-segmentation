# Dữ liệu Thô (Raw Data) – UD Vietnamese-VTB

Thư mục này chứa bộ dữ liệu gốc tiếng Việt phục vụ bài toán **Vietnamese Word Segmentation** cho dự án seminar.

Dữ liệu được thu thập và quản lý bởi: **Hải (TV03 - Nhóm thực nghiệm)**.

---

## 1. Nguồn dữ liệu

- **Bộ dữ liệu**: Universal Dependencies Vietnamese Treebank (**UD Vietnamese-VTB**)
- **Kho lưu trữ chính thức**: [UD_Vietnamese-VTB (GitHub)](https://github.com/UniversalDependencies/UD_Vietnamese-VTB)
- **Tài liệu tham khảo UD**: [Universal Dependencies vi_vtb Documentation](https://universaldependencies.org/treebanks/vi_vtb/)
- **Bài báo gốc về dữ liệu**: Nguyen et al. (2009), *Building a Large Syntactically-Annotated Corpus for Vietnamese*, [ACL Anthology W09-3035](https://aclanthology.org/W09-3035/)
- **Nguồn gốc ngữ liệu**: Bắt nguồn từ Vietnamese Treebank (VTB) của dự án VLSP, chủ yếu là văn bản báo chí tin tức.
- **Lý do lựa chọn**:
  - Đã được gán nhãn phân tách từ chuẩn xác.
  - Phân chia sẵn thành các tập train, dev (validation), test chuẩn mực.
  - Mã nguồn mở, truy cập trực tiếp và minh bạch cho quá trình tái lập thực nghiệm.

---

## 2. Danh mục tệp dữ liệu bàn giao

Toàn bộ dữ liệu gốc được lưu giữ nguyên bản tại thư mục `data/raw/ud_vtb/`:

| Tệp dữ liệu | Dung lượng | Mô tả |
|---|---|---|
| `data/raw/ud_vtb/vi_vtb-ud-train.conllu` | ~947 KB | Tập huấn luyện (Training set) |
| `data/raw/ud_vtb/vi_vtb-ud-dev.conllu` | ~1.23 MB | Tập phát triển / kiểm định (Validation / Dev set) |
| `data/raw/ud_vtb/vi_vtb-ud-test.conllu` | ~539 KB | Tập đánh giá độc lập (Test set) |

> **Nguyên tắc**: Dữ liệu gốc trong thư mục này được giữ nguyên trạng, không chỉnh sửa trực tiếp. Các bước làm sạch, chuẩn hóa, trích xuất và gán nhãn BIO được thực hiện bởi pipeline tiền xử lý và lưu vào `data/processed/`.

---

## 3. Thống kê quy mô dữ liệu

Số liệu được thống kê trực tiếp từ ba tệp dữ liệu gốc:

| Tập dữ liệu | Số câu (`sent_id`) | Số token từ (Cột `FORM`) | Số đơn vị đầu vào (Tách khoảng trắng) | Số âm tiết ước tính (Bỏ PUNCT, SYM, NUM) |
|---|---|---|---|---|
| **Train** | 1.400 | 20.215 | 24.973 | 21.345 |
| **Dev** | 1.123 | 26.162 | 31.927 | 27.390 |
| **Test** | 800 | 11.692 | 13.857 | 11.839 |
| **Tổng cộng** | **3.323** | **58.069** | **70.757** | **60.574** |

*Ghi chú về cách đếm:*
- **Số câu**: Đếm số dòng bắt đầu bằng `# sent_id`.
- **Số từ**: Đếm tổng số dòng token dữ liệu trong tệp (mỗi dòng tương ứng một từ đơn hoặc từ ghép trong cột `FORM`).
- **Số đơn vị đầu vào**: Lấy từng từ trong cột `FORM` rồi phân tách theo khoảng trắng (bao gồm cả từ ngữ, dấu câu, ký hiệu và số).
- **Số âm tiết ước tính**: Phân tách cột `FORM` theo khoảng trắng sau khi loại bỏ các token mang nhãn `PUNCT` (dấu câu), `SYM` (ký hiệu) và `NUM` (số).

---

## 4. Cấu trúc định dạng dữ liệu (CoNLL-U)

Dữ liệu được lưu trữ theo định dạng chuẩn CoNLL-U. Mỗi câu gồm:
1. Dòng chú thích mã câu: `# sent_id = ...`
2. Dòng văn bản gốc: `# text = ...`
3. Các dòng thông tin từ vựng (mỗi dòng gồm 10 cột ngăn cách bởi dấu tab `\t`):
   - Cột 1: `ID` (vị trí trong câu)
   - Cột 2: `FORM` (từ/hình thái đã được xác định - **cột quan trọng nhất cho Word Segmentation**)
   - Cột 3: `LEMMA`
   - Cột 4: `UPOS` (Universal POS tag)
   - Cột 5: `XPOS` (Language-specific POS tag)
   - Cột 6: `FEATS`
   - Cột 7: `HEAD`
   - Cột 8: `DEPREL`
   - Cột 9: `DEPS`
   - Cột 10: `MISC`

### Ví dụ minh họa

Trích một phần câu trong tập `train` (`vi_vtb-ud-train.conllu`):

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

- Các từ xác định theo cột `FORM`: `Hiện nay` | `xã` | `có` | `68` | `tổ` | `nhân dân` | `,` | `mỗi` | `tổ` | `phụ trách` | `40` | `gia đình` | `.`
- Nếu quy ước nối các âm tiết trong một từ bằng dấu gạch dưới (`_`), câu sẽ trở thành:
  > `Hiện_nay xã có 68 tổ nhân_dân , mỗi tổ phụ_trách 40 gia_đình .`
- Nhận xét: Trong tiếng Việt, từ đa âm tiết chiếm tỷ lệ lớn (`Hiện nay`, `nhân dân`, `phụ trách`, `gia đình`). Cột `FORM` là căn cứ tham chiếu (ground truth) xác định ranh giới từ.

---

## 5. Hướng dẫn sử dụng cho các nhóm tiếp theo

### 5.1. Dành cho nhóm Tiền xử lý (Phát - TV04)
- Đọc các tệp `.conllu` từ `data/raw/ud_vtb/`.
- Trích xuất chuỗi âm tiết và chuỗi nhãn BIO tương ứng:
  - Âm tiết đầu tiên của từ gắn nhãn `B` (Begin).
  - Các âm tiết tiếp theo trong cùng từ gắn nhãn `I` (Inside).
  - Ví dụ: `Hiện/B nay/I xã/B có/B 68/B tổ/B nhân/B dân/I ...`
- Tạo các tệp processed lưu vào `data/processed/`.

### 5.2. Dành cho nhóm Hybrid Approach (Thành - TV05)
- Thu thập từ vựng từ cột `FORM` tập `train` để xây dựng bộ từ điển (Dictionary).
- Xây dựng bảng xác suất Bigram ngôn ngữ dựa trên các chuỗi từ trong tập `train`.

### 5.3. Dành cho nhóm LSTM (Khang - TV06)
- Sử dụng tập dữ liệu đã gán nhãn BIO từ `data/processed/` để huấn luyện mô hình sequence labeling.
- Đánh giá trên tập `dev` và `test` độc lập.
