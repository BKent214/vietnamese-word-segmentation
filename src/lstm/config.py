from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class LSTMConfig:
    # Đường dẫn dữ liệu
    train_path: Path = Path("data/processed/train.csv")
    dev_path:   Path = Path("data/processed/dev.csv")
    test_path:  Path = Path("data/processed/test.csv")

    # Đường dẫn lưu trữ đầu ra
    checkpoint_dir:  Path = Path("checkpoints/lstm")
    output_dir:      Path = Path("outputs/lstm")
    best_model_path: Path = Path("checkpoints/lstm/best_lstm_model.pt")
    vocab_path:      Path = Path("checkpoints/lstm/vocab.json")
    loss_plot_path:  Path = Path("outputs/lstm/loss_f1_curve.png")

    # Token đặc biệt & Nhãn BIO
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    pad_idx:   int = 0
    unk_idx:   int = 1

    # Siêu tham số từ điển & kiến trúc mạng
    min_freq:       int = 2         # Âm tiết xuất hiện < 2 lần thành <UNK> để mô hình học cách đoán OOV
    embedding_dim:  int = 128
    hidden_dim:     int = 256
    num_layers:     int = 2
    bidirectional: bool = True      # True = BiLSTM (nhìn cả ngữ cảnh trái & phải), False = LSTM 1 chiều
    dropout:      float = 0.3

    # Siêu tham số huấn luyện
    seed:          int   = 42
    batch_size:    int   = 64
    epochs:        int   = 20
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-5
    grad_clip:     float = 5.0

    # Thiết bị tính toán
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir    .mkdir(parents=True, exist_ok=True)