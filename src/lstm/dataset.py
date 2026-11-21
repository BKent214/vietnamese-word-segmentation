from   collections        import Counter
import json
from   pathlib            import Path
from   typing             import Dict, List, Tuple

import pandas as pd
import torch
from   torch.nn.utils.rnn import pad_sequence
from   torch.utils.data   import Dataset


TAG2IDX: Dict[str, int] = {"<PAD>": 0, "B": 1, "I": 2}
IDX2TAG: Dict[int, str] = {v: k for k, v in TAG2IDX.items()}


class Vocabulary:
    """Quản lý ánh xạ Âm tiết <-> ID."""

    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>", min_freq: int = 2):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq  = min_freq

        self.stoi: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.itos: Dict[int, str] = {0: pad_token, 1: unk_token}

    def build_from_texts(self, texts: List[str]) -> None:
        counter = Counter()
        for sentence in texts:
            syllables = str(sentence).strip().lower().split()
            counter.update(syllables)

        for syl, freq in counter.items():
            if freq >= self.min_freq and syl not in self.stoi:
                idx = len(self.stoi)
                self.stoi[syl] = idx
                self.itos[idx] = syl

    def encode(self, syllables: List[str]) -> List[int]:
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(s.lower(), unk_id) for s in syllables]

    def __len__(self) -> int:
        return len(self.stoi)

    def save(self, filepath: Path) -> None:
        payload = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "min_freq": self.min_freq,
            "stoi": self.stoi,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "Vocabulary":
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = cls(
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
            min_freq=payload["min_freq"],
        )
        vocab.stoi = payload["stoi"]
        vocab.itos = {int(v): k for k, v in vocab.stoi.items()}
        return vocab


class WordSegDataset(Dataset):
    """Dataset PyTorch đọc file CSV (id, text, segmented, labels)."""

    def __init__(self, csv_path: Path, vocab: Vocabulary):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for row_idx, row in self.df.iterrows():
            # Theo hướng dẫn của ...\data\processed\README.md
            syllables = str(row["text"])  .strip().split()
            labels    = str(row["labels"]).strip().split()

            if not syllables:
                continue
            if len(syllables) != len(labels):
                raise ValueError(f"{csv_path}, row {row_idx}: len(text) != len(labels)")
            invalid = set(labels) - {"B", "I"}
            if invalid:
                raise ValueError(f"{csv_path}, row {row_idx}: invalid BIO tags {invalid}")
            if labels[0] != "B":
                raise ValueError(f"{csv_path}, row {row_idx}: câu phải bắt đầu bằng B")

            input_ids = torch.tensor(self.vocab.encode(syllables),     dtype=torch.long)
            label_ids = torch.tensor([TAG2IDX[tag] for tag in labels], dtype=torch.long)
            self.samples.append((input_ids, label_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Thực hiện Padding động theo câu dài nhất trong batch và tạo Mask/Lengths.
    """
    input_seqs, label_seqs = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in input_seqs], dtype=torch.long)

    padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(label_seqs, batch_first=True, padding_value=TAG2IDX["<PAD>"])
    mask = padded_inputs != 0  # Boolean mask: True tại vị trí âm tiết thật, False tại <PAD>

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "lengths": lengths,
        "mask": mask,
    }