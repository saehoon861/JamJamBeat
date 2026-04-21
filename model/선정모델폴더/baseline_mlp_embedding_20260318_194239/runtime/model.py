# model.py - baseline mlp_embedding checkpoint에 맞춘 최소 모델 정의
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """Raw 63d joint 입력을 임베딩한 뒤 분류하는 MLP."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.embed(x))
