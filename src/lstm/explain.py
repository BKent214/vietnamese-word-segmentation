"""Lấy kết quả trung gian của từng tầng LSTM, phục vụ demo giải thích mô hình."""
import torch

from src.lstm.dataset import IDX2TAG


@torch.no_grad()
def trace_forward(predictor, syllables):
    model = predictor.model
    vocab = predictor.vocab

    ids = vocab.encode(syllables)
    input_ids = torch.tensor([ids], dtype=torch.long, device=predictor.device)

    embedded = model.embedding(input_ids)
    hidden, _ = model.lstm(embedded)
    logits = model.classifier(hidden)
    logits[:, :, 0] = -1e9

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    tags = [IDX2TAG[int(i)] for i in probs.argmax(axis=-1)]

    return {
        "ids": ids,
        "unk_id": vocab.stoi[vocab.unk_token],
        "embedded": embedded[0].cpu().numpy(),
        "hidden": hidden[0].cpu().numpy(),
        "p_begin": probs[:, 1],
        "p_inside": probs[:, 2],
        "tags": tags,
        "segmented": predictor.reconstruct_words(syllables, tags),
    }
