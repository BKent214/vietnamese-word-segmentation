import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.lstm.config import LSTMConfig
from src.lstm.dataset import TAG2IDX, Vocabulary, WordSegDataset, collate_fn
from src.lstm.model import LSTMWordSegmenter
from src.lstm.metrics import boundary_prf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# def compute_token_f1(preds: List[int], targets: List[int]) -> float:
#     """Tính Macro-F1 trên 2 nhãn B và I (bỏ qua <PAD>)."""
#     f1_scores = []
#     for tag_id in [TAG2IDX["B"], TAG2IDX["I"]]:
#         tp = sum(1 for p, t in zip(preds, targets) if p == tag_id and t == tag_id)
#         fp = sum(1 for p, t in zip(preds, targets) if p == tag_id and t != tag_id)
#         fn = sum(1 for p, t in zip(preds, targets) if p != tag_id and t == tag_id)
#         precision = tp / (tp + fp + 1e-9)
#         recall    = tp / (tp + fn + 1e-9)
#         f1 = 2 * precision * recall / (precision + recall + 1e-9)
#         f1_scores.append(f1)
#     return float(np.mean(f1_scores))


def run_epoch(
    model:      nn.Module,
    dataloader: DataLoader,
    criterion:  nn.Module,
    device:     str,
    optimizer:  torch.optim.Optimizer = None,
    grad_clip:  float = 5.0,
) -> Tuple[float, float]:
    
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss             = 0.0
    total_tokens           = 0
    all_preds:   List[int] = []
    all_targets: List[int] = []

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"]   .to(device)
            lengths   = batch["lengths"]
            mask      = batch["mask"]     .to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(input_ids, lengths)  # [B, L, num_tags]
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            n_tokens = int(mask.sum().item())
            total_loss += loss.item()
            total_tokens += n_tokens

            # Thu thập dự đoán tại các vị trí không phải <PAD>
            # preds = torch.argmax(logits, dim=-1)
            preds = logits.argmax(-1)     
            all_preds  .extend( preds[mask].cpu().tolist())
            all_targets.extend(labels[mask].cpu().tolist())
                 
    # avg_loss = total_loss / len(dataloader)
    # macro_f1 = compute_token_f1(all_preds, all_targets)
    # return avg_loss, macro_f1
    gold_tags = ["B" if x == TAG2IDX["B"] else "I" for x in all_targets]
    pred_tags = ["B" if x == TAG2IDX["B"] else "I" for x in all_preds]
    _, _, boundary_f1 = boundary_prf(gold_tags, pred_tags)
    return total_loss / max(total_tokens, 1), boundary_f1


def plot_training_history(history: Dict[str, List[float]], save_path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["dev_loss"],   label="Dev Loss",   marker="s")
    ax1.set_title("Cross-Entropy Loss qua các Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(epochs, history["train_f1"], label="Train Token F1", marker="o")
    ax2.plot(epochs, history["dev_f1"],   label="Dev Token F1",   marker="s", color="green")
    ax2.set_title("Boundary F1 (B label) qua các Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1-Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_pipeline(config=None) -> Dict[str, List[float]]:
    config = config or LSTMConfig()
    set_seed(config.seed)
    config.ensure_dirs()

    # 1. Xây dựng Vocabulary CHỈ từ tập train
    train_df = pd.read_csv(config.train_path)
    vocab = Vocabulary(
        pad_token = config.pad_token,
        unk_token = config.unk_token,
        min_freq  = config.min_freq,
    )
    vocab.build_from_texts(train_df["text"].tolist())
    vocab.save(config.vocab_path)
    print(f"[INFO] Kích thước Vocabulary (từ tập Train): {len(vocab):,} âm tiết")

    # 2. Khởi tạo Dataset & DataLoader
    train_dataset = WordSegDataset(config.train_path, vocab)
    dev_dataset   = WordSegDataset(config.dev_path,   vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,  collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,   batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 3. Khởi tạo Model, Loss (ignore_index=0 để bỏ qua <PAD>) & Optimizer
    model = LSTMWordSegmenter(
        vocab_size    = len(vocab),
        embedding_dim = config.embedding_dim,
        hidden_dim    = config.hidden_dim,
        num_tags      = len(TAG2IDX),
        num_layers    = config.num_layers,
        bidirectional = config.bidirectional,
        dropout       = config.dropout,
        pad_idx       = config.pad_idx,
    ).to(config.device)

    criterion = nn.CrossEntropyLoss(ignore_index=TAG2IDX["<PAD>"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = {"train_loss": [], "dev_loss": [], "train_f1": [], "dev_f1": []}
    best_dev_f1 = 0.0

    # 4. Vòng lặp huấn luyện
    for epoch in range(1, config.epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, config.device, 
                                         optimizer, config.grad_clip)
        dev_loss, dev_f1     = run_epoch(model, dev_loader,   criterion, config.device)

        history["train_loss"].append(train_loss)
        history["dev_loss"]  .append(dev_loss)
        history["train_f1"]  .append(train_f1)
        history["dev_f1"]    .append(dev_f1)

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - B-F1: {train_f1:.4f} | "
            f"Dev Loss: {dev_loss:.4f} - B-F1: {dev_f1:.4f}"
        )

        # Lưu checkpoint tốt nhất theo Dev F1
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_dev_f1":      best_dev_f1,
                    "model_hparams": {
                        "vocab_size"   : len(vocab),
                        "embedding_dim": config.embedding_dim,
                        "hidden_dim"   : config.hidden_dim,
                        "num_tags"     : len(TAG2IDX),
                        "num_layers"   : config.num_layers,
                        "bidirectional": config.bidirectional,
                        "dropout"      : config.dropout,
                        "pad_idx"      : config.pad_idx,
                    },
                },
                config.best_model_path,
            )

    plot_training_history(history, config.loss_plot_path)
    print(f"[DONE] Đã lưu model tốt nhất (Dev F1 = {best_dev_f1:.4f}) tại {config.best_model_path}")
    return history


if __name__ == "__main__":
    cfg = LSTMConfig()
    train_pipeline(cfg)