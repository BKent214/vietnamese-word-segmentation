import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMWordSegmenter(nn.Module):
    """
    Mô hình Sequence Labeling cho bài toán Tách từ Tiếng Việt:
    Input IDs -> Embedding Layer -> (Bi)LSTM Layer -> Dropout -> Linear BIO Classifier
    """

    def __init__(
        self,
        vocab_size:     int,
        embedding_dim:  int,
        hidden_dim:     int,
        num_tags:       int = 3,
        num_layers:     int = 2,
        bidirectional: bool = True,
        dropout:      float = 0.3,
        pad_idx:        int = 0,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 1. Embedding Layer: Ánh xạ ID âm tiết sang vector dày đặc
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.emb_dropout = nn.Dropout(dropout)

        # 2. LSTM Layer: Trích xuất đặc trưng ngữ cảnh chuỗi
        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)

        # 3. BIO Classification Layer: Chiếu về không gian nhãn (<PAD>, B, I)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_tags)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor [batch_size, max_seq_len]
            lengths: Tensor [batch_size] chứa độ dài thực của từng câu
        Returns:
            logits: Tensor [batch_size, max_seq_len, num_tags]
        """
        max_len = input_ids.size(1)
        embedded = self.emb_dropout(self.embedding(input_ids))

        # Đóng gói chuỗi để LSTM bỏ qua các bước thời gian chứa <PAD> (Masking)
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_embedded)

        # Giải nén về lại tensor kích thước chuẩn [batch_size, max_seq_len, hidden * directions]
        lstm_out, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_len
        )

        return self.classifier(self.out_dropout(lstm_out))