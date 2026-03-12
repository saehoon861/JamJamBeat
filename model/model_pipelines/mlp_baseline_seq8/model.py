# mlp_baseline_seq8/model.py - joint(63d) sliding-window(T=8) flatten MLP (mlp_baseline 아키텍처 그대로)
import torch.nn as nn


class MLPBaselineSeq(nn.Module):
    """T=8 시퀀스를 flatten해서 읽는 sequence baseline.

    입력 shape: (B, 8, 63)
    출력 shape: (B, num_classes)
    """

    def __init__(self, seq_len: int, input_dim: int, num_classes: int):
        super().__init__()
        flat_dim = seq_len * input_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq):
        # 시간축 구조를 별도 연산 없이 펼쳐서 큰 frame vector처럼 취급한다.
        x = x_seq.reshape(x_seq.size(0), self.seq_len * self.input_dim)
        return self.net(x)
