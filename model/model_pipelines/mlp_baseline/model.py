# mlp_baseline/model.py - joint(63d) 단일 입력 MLP baseline 분류기
import torch.nn as nn


class MLPBaseline(nn.Module):
    """frame 입력용 가장 단순한 MLP baseline.

    입력 shape: (B, D)
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
        # 단일 프레임 feature를 바로 class logits으로 변환한다.
        return self.net(x)
