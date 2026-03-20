# model.py - pos_scale mlp_sequence_delta checkpoint에 맞춘 모델 정의
import torch.nn as nn


class SequenceDeltaMLP(nn.Module):
    """joint(63d) + delta(63d) 시퀀스를 flatten해서 분류하는 MLP.

    입력 shape: (B, T, 126)
    출력 shape: (B, num_classes)
    """

    def __init__(self, seq_len: int, input_dim: int, num_classes: int):
        super().__init__()
        flat_dim = seq_len * input_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_seq):
        x = x_seq.reshape(x_seq.size(0), self.seq_len * self.input_dim)
        return self.net(x)
