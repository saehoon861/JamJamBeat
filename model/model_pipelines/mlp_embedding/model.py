# mlp_embedding/model.py - raw joint(63d) 입력, learnable projection embedding MLP
import torch.nn as nn


class MLPEmbedding(nn.Module):
    """raw joint feature를 latent embedding으로 투영한 뒤 분류하는 MLP.

    입력 shape: (B, D)
    출력 shape: (B, num_classes)
    """

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
        # baseline 대비 차이는 첫 단계에서 learnable projection을 거친다는 점이다.
        return self.head(self.embed(x))
