# mlp_baseline_full/model.py - joint+bone+angle(156d) 입력 MLP baseline (bone/angle 피처 가치 검증)
import torch.nn as nn


class MLPBaseline(nn.Module):
    """full feature(156d) 실험용 baseline MLP.

    입력 shape: (B, 156)
    출력 shape: (B, num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # 아키텍처는 baseline과 동일하고 입력 차원만 달라진다.
        return self.net(x)
