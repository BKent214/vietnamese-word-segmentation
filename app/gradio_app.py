
import time
from html import escape

import gradio as gr

from app.loader import load_baselines, load_lstm
from src.lstm.config import LSTMConfig
from src.lstm.explain import trace_forward
from src.utils.text_utils import prepare_input

CONFIG = LSTMConfig(device="cpu")
BASELINES = load_baselines(CONFIG.train_path)
LSTM = load_lstm(CONFIG)

LSTM_NAME = "LSTM"
MODEL_NAMES = [*BASELINES, LSTM_NAME]
ALL_MODELS = {**BASELINES, LSTM_NAME: LSTM} if LSTM else dict(BASELINES)

MAX_SYLLABLES = 120
TIMING_RUNS = 10
LOW_CONFIDENCE = 0.7
NO_CHECKPOINT = "Chưa có best_lstm_model.pt. Hãy chạy: python -m src.lstm.train"
COLOR_MAP = {"từ ghép": "#90caf9"}

EXAMPLES = [
    ["Hiện nay xã có 68 tổ nhân dân , mỗi tổ phụ trách 40 gia đình ."],
    ["Sinh viên học xử lý ngôn ngữ tự nhiên ."],
    ["Tôi thích học khoa học dữ liệu và trí tuệ nhân tạo ."],
]

STYLE = """<style>
.ws-row {display: flex; flex-wrap: wrap; gap: 6px;}
.ws-chip {min-width: 48px; padding: 6px 8px; border-radius: 8px; text-align: center;
          color: #111; border: 2px solid transparent;}
.ws-id {background: #eceff1;}
.ws-unk {background: #ffe0b2;}
.ws-b {background: #c8e6c9;}
.ws-i {background: #bbdefb;}
.ws-low {border: 2px dashed #ef6c00;}
.ws-syl {font-weight: 600;}
.ws-small {font-size: 12px; color: #455a64;}
.ws-bar {height: 4px; margin-top: 4px; border-radius: 2px; background: rgba(0, 0, 0, 0.15);}
.ws-bar span {display: block; height: 100%; border-radius: 2px; background: #37474f;}
</style>"""


def clean_input(text):
    syllables = prepare_input(text).split()[:MAX_SYLLABLES]
    return " ".join(syllables)


def to_highlight(segmented):
    """'Hiện_nay xã' -> [('Hiện nay ', 'từ ghép'), ('xã ', None)]"""
    items = []
    for word in segmented.split():
        parts = word.split("_")
        label = "từ ghép" if len(parts) > 1 else None
        items.append((" ".join(parts) + " ", label))
    return items


def measure(model, text):
    segmented, _ = model.predict_sentence(text)

    start = time.perf_counter()
    for _ in range(TIMING_RUNS):
        model.predict_sentence(text)
    average_ms = (time.perf_counter() - start) / TIMING_RUNS * 1000

    return segmented, round(average_ms, 3)


def describe_lstm():
    if LSTM is None:
        return NO_CHECKPOINT

    model = LSTM.model
    directions = 2 if model.bidirectional else 1
    params = sum(p.numel() for p in model.parameters())
    return (
        f"**Kiến trúc đang chạy:** Embedding ({len(LSTM.vocab)} âm tiết, {model.embedding.embedding_dim} chiều)"
        f" → {'Bi' if model.bidirectional else ''}LSTM {model.lstm.num_layers} lớp"
        f" (hidden {model.lstm.hidden_size} × {directions} chiều)"
        f" → Linear → Softmax (B, I). Tổng cộng {params:,} tham số."
    )


def render_timing(rows):
    lines = ["| Mô hình | Số từ | Thời gian TB (ms) |", "|---|---|---|"]
    for name, word_count, average_ms in rows:
        lines.append(f"| {name} | {word_count} | {average_ms} |")
    return "\n".join(lines)


def render_ids(syllables, result):
    chips = []
    for syllable, token_id in zip(syllables, result["ids"]):
        css = "ws-unk" if token_id == result["unk_id"] else "ws-id"
        chips.append(
            f'<div class="ws-chip {css}">'
            f'<div class="ws-syl">{escape(syllable)}</div>'
            f'<div class="ws-small">ID {token_id}</div></div>'
        )
    return STYLE + '<div class="ws-row">' + "".join(chips) + "</div>"


def render_shapes(syllables, result):
    length = len(syllables)
    embed_dim = result["embedded"].shape[1]
    hidden_dim = result["hidden"].shape[1]
    context = "từ cả hai phía của câu" if LSTM.model.bidirectional else "từ các âm tiết đứng trước"
    return "\n".join(
        [
            "| Tầng | Kích thước | Ý nghĩa |",
            "|---|---|---|",
            f"| Chuỗi ID | ({length},) | mỗi âm tiết là một số nguyên |",
            f"| Sau Embedding | ({length}, {embed_dim}) | mỗi âm tiết thành vector {embed_dim} chiều |",
            f"| Sau LSTM | ({length}, {hidden_dim}) | mỗi vector đã chứa ngữ cảnh {context} |",
            f"| Sau Linear và Softmax | ({length}, 2) | P(B) và P(I) của từng âm tiết |",
        ]
    )


def render_probabilities(syllables, result):
    chips = []
    for syllable, tag, p_begin, p_inside in zip(
        syllables, result["tags"], result["p_begin"], result["p_inside"]
    ):
        confidence = float(p_begin if tag == "B" else p_inside)
        percent = round(confidence * 100)
        css = "ws-b" if tag == "B" else "ws-i"
        if confidence < LOW_CONFIDENCE:
            css += " ws-low"
        chips.append(
            f'<div class="ws-chip {css}">'
            f'<div class="ws-syl">{escape(syllable)}</div>'
            f'<div class="ws-small">{tag} · {percent}%</div>'
            f'<div class="ws-bar"><span style="width:{percent}%"></span></div></div>'
        )
    return STYLE + '<div class="ws-row">' + "".join(chips) + "</div>"


def compare(text):
    clean = clean_input(text)
    if not clean:
        return ("", *[[("", None)]] * len(MODEL_NAMES), "")

    views, rows = [], []
    for name in MODEL_NAMES:
        model = ALL_MODELS.get(name)
        if model is None:
            views.append([(NO_CHECKPOINT, None)])
            continue
        segmented, average_ms = measure(model, clean)
        views.append(to_highlight(segmented))
        rows.append((name, len(segmented.split()), average_ms))

    return (clean, *views, render_timing(rows))


def explain(text):
    empty = ("", "", "", "", [("", None)], "")
    if LSTM is None:
        return (NO_CHECKPOINT, *empty[1:])

    clean = clean_input(text)
    if not clean:
        return empty

    syllables = clean.split()
    result = trace_forward(LSTM, syllables)

    return (
        "",
        render_ids(syllables, result),
        render_shapes(syllables, result),
        render_probabilities(syllables, result),
        to_highlight(result["segmented"]),
        " ".join(result["tags"]),
    )


def build_compare_tab():
    views = [
        gr.HighlightedText(
            label=name, combine_adjacent=False, show_legend=False, color_map=COLOR_MAP
        )
        for name in MODEL_NAMES
    ]
    gr.Markdown("### Thời gian xử lý (trung bình 10 lần chạy, CPU)")
    timing = gr.Markdown()
    return views, timing


def build_explain_tab():
    gr.Markdown(describe_lstm())
    notice = gr.Markdown()

    gr.Markdown("### Bước 1. Âm tiết thành ID\nÔ màu cam là `<UNK>`: âm tiết chưa gặp hoặc hiếm trong tập train.")
    step1 = gr.HTML()

    gr.Markdown("### Bước 2 và 3. Embedding và LSTM\nDữ liệu thay đổi kích thước qua từng tầng.")
    shapes = gr.Markdown()

    gr.Markdown(
        "### Bước 4. Dense và Softmax\n"
        "Xanh lá: nhãn B (bắt đầu từ mới). Xanh dương: nhãn I (nối vào từ trước). "
        "Thanh nhỏ và phần trăm cho biết độ chắc chắn. Viền cam nét đứt: model còn do dự (dưới 70%)."
    )
    step4 = gr.HTML()

    gr.Markdown("### Bước 5. Ghép nhãn thành từ")
    step5 = gr.HighlightedText(
        label="Kết quả tách từ", combine_adjacent=False, show_legend=False, color_map=COLOR_MAP
    )
    tags = gr.Textbox(label="Chuỗi nhãn B/I", interactive=False)

    return [notice, step1, shapes, step4, step5, tags]


def build_interface():
    with gr.Blocks(title="Vietnamese Word Segmentation") as demo:
        gr.Markdown("# Tách từ tiếng Việt: Baseline, Hybrid và LSTM")

        text_input = gr.Textbox(label="Văn bản đầu vào", lines=3)
        run_button = gr.Button("Phân tích", variant="primary")
        gr.Examples(examples=EXAMPLES, inputs=text_input)
        clean_output = gr.Textbox(label="Sau tiền xử lý", interactive=False)

        with gr.Tabs():
            with gr.Tab("So sánh các phương pháp"):
                compare_views, timing_output = build_compare_tab()
            with gr.Tab("Chi tiết BiLSTM"):
                explain_outputs = build_explain_tab()

        compare_outputs = [clean_output, *compare_views, timing_output]
        for trigger in (run_button.click, text_input.submit):
            trigger(compare, inputs=text_input, outputs=compare_outputs)
            trigger(explain, inputs=text_input, outputs=explain_outputs)

    return demo


if __name__ == "__main__":
    build_interface().launch(share=True)
