from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

from src.lstm.config import LSTMConfig
from src.lstm.dataset import IDX2TAG, Vocabulary
from src.lstm.model import LSTMWordSegmenter


class WordSegmenterPredictor:
    """Lớp thực hiện inference và tái tạo câu đã tách từ từ chuỗi nhãn B/I."""

    def __init__(self, checkpoint_path: Path, vocab_path: Path, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocabulary.load(vocab_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        hparams = checkpoint["model_hparams"]

        self.model = LSTMWordSegmenter(**hparams).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @staticmethod
    def reconstruct_words(syllables: List[str], tags: List[str]) -> str:
        """
        Tái tạo câu tách từ từ danh sách âm tiết và nhãn B/I.
        Ví dụ: ['Hiện', 'nay', 'xã'] + ['B', 'I', 'B'] -> 'Hiện_nay xã'
        """
        if not syllables:
            return ""

        words: List[List[str]] = []
        for syl, tag in zip(syllables, tags):
            if   tag == "B" or len(words) == 0:
                words.append([syl])
            elif tag == "I":
                words[-1].append(syl)

        return " ".join("_".join(w) for w in words)

    @torch.no_grad()
    def predict_sentence(self, text: str) -> Tuple[str, List[str]]:
        syllables = str(text).strip().split()
        if not syllables:
            return "", []

        input_ids = torch.tensor([self.vocab.encode(syllables)], dtype=torch.long).to(self.device)
        lengths   = torch.tensor([len(syllables)],               dtype=torch.long)

        logits = self.model(input_ids, lengths)
        # Chặn mô hình dự đoán ra nhãn <PAD> (index 0) lúc inference
        logits[:, :, 0] = -1e9
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

        pred_tags = [IDX2TAG[idx] for idx in pred_ids]
        segmented_text = self.reconstruct_words(syllables, pred_tags)
        return segmented_text, pred_tags

    def predict_csv(self, input_csv: Path, output_csv: Path) -> pd.DataFrame:
        df = pd.read_csv(input_csv)
        pred_segmented_list = []
        pred_labels_list    = []

        for text in df["text"]:
            seg_text, tags = self.predict_sentence(text)
            pred_segmented_list.append(seg_text)
            pred_labels_list.append(" ".join(tags))

        df["pred_segmented"] = pred_segmented_list
        df["pred_labels"]    = pred_labels_list
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return df


if __name__ == "__main__":
    cfg = LSTMConfig()
    predictor = WordSegmenterPredictor(cfg.best_model_path, cfg.vocab_path, cfg.device)

    sample = "Hiện nay xã có 68 tổ nhân dân ."
    seg, tags = predictor.predict_sentence(sample)
    print(f"Input     : {sample}")
    print(f"Pred Tags : {' '.join(tags)}")
    print(f"Segmented : {seg}")