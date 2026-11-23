import math

import torch
import torch.nn as nn

from src.lstm.dataset import Vocabulary
from src.lstm.model import LSTMWordSegmenter
from src.lstm.predict import WordSegmenterPredictor
from src.lstm.train import run_epoch


def test_vocabulary_round_trip_preserves_encoding(tmp_path):
    vocabulary = Vocabulary(min_freq=1)
    vocabulary.build_from_texts(["Xin chào"])
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary.save(vocabulary_path)

    loaded = Vocabulary.load(vocabulary_path)

    assert loaded.encode(["xin", "chào", "không_có"]) == vocabulary.encode(
        ["xin", "chào", "không_có"]
    )


def test_predictor_loads_checkpoint_and_returns_one_tag_per_syllable(tmp_path):
    torch.manual_seed(0)
    vocabulary = Vocabulary(min_freq=1)
    vocabulary.build_from_texts(["xin chào"])
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary.save(vocabulary_path)

    model_hparams = {
        "vocab_size": len(vocabulary),
        "embedding_dim": 4,
        "hidden_dim": 3,
        "num_tags": 3,
        "num_layers": 1,
        "bidirectional": True,
        "dropout": 0.0,
        "pad_idx": 0,
    }
    model = LSTMWordSegmenter(**model_hparams)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_hparams": model_hparams,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    predictor = WordSegmenterPredictor(
        checkpoint_path,
        vocabulary_path,
        device="cpu",
    )
    segmented, tags = predictor.predict_sentence("xin chào")

    assert len(tags) == 2
    assert set(tags) <= {"B", "I"}
    assert segmented.replace("_", " ").split() == ["xin", "chào"]


def test_run_epoch_weights_batch_losses_by_token_count():
    class FixedLogitsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_index = 0

        def forward(self, input_ids, lengths):
            self.batch_index += 1
            if self.batch_index == 1:
                return torch.tensor([[[0.0, 3.0, 0.0]]])
            return torch.zeros((1, 3, 3))

    batches = [
        {
            "input_ids": torch.tensor([[1]]),
            "labels": torch.tensor([[1]]),
            "lengths": torch.tensor([1]),
            "mask": torch.tensor([[True]]),
        },
        {
            "input_ids": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 2]]),
            "lengths": torch.tensor([3]),
            "mask": torch.tensor([[True, True, True]]),
        },
    ]
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    first_loss = criterion(torch.tensor([[0.0, 3.0, 0.0]]), torch.tensor([1]))
    second_loss = criterion(torch.zeros((3, 3)), torch.tensor([1, 2, 2]))
    expected_loss = (first_loss.item() + 3 * second_loss.item()) / 4

    average_loss, _ = run_epoch(
        FixedLogitsModel(),
        batches,
        criterion,
        device="cpu",
    )

    assert math.isclose(average_loss, expected_loss, rel_tol=1e-6)