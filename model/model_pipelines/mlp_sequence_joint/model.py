# mlp_sequence_joint/model.py - joint(63d) sequence input flattened into a simple MLP classifier
import torch.nn as nn


class SequenceJointMLP(nn.Module):
    """joint-only 시퀀스를 flatten해서 읽는 MLP.

    입력 shape: (B, T, 63)
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
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_seq):
        # 마지막 프레임만 떼지 않고 전체 시퀀스 위치별 가중치를 선형층이 학습하게 한다.
        x = x_seq.reshape(x_seq.size(0), self.seq_len * self.input_dim)
        return self.net(x)
