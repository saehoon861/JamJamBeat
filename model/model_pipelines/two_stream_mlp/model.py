# two_stream_mlp/model.py - joint stream + bone+angle stream late fusion (2s-AGCN 구조 기반)
import torch
import torch.nn as nn


class StreamMLP(nn.Module):
    """한 스트림(joint 또는 bone+angle)만 인코딩하는 공통 MLP 블록."""

    def __init__(self, input_dim: int, hidden: tuple = (128, 128), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            # 각 스트림은 동일한 구조로 feature를 embedding space로 보낸다.
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, x):
        # 입력 shape: (B, D_stream) -> 출력 shape: (B, hidden[-1])
        return self.net(x)


class TwoStreamMLP(nn.Module):
    """joint stream + bone/angle stream을 late fusion 하는 분류기.

    입력 shape:
    - x_joint: (B, 63)
    - x_bone: (B, 93)
    출력 shape: (B, num_classes)
    """

    def __init__(self, joint_dim: int, bone_dim: int, num_classes: int):
        super().__init__()
        self.joint_stream = StreamMLP(joint_dim)
        self.bone_stream  = StreamMLP(bone_dim)

        # 두 스트림 embedding을 concat한 뒤 최종 분류 head로 보낸다.
        fusion_in = self.joint_stream.out_dim + self.bone_stream.out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_joint, x_bone):
        j = self.joint_stream(x_joint)
        b = self.bone_stream(x_bone)
        # late fusion: joint 표현과 bone/angle 표현을 마지막에 결합한다.
        return self.head(torch.cat([j, b], dim=-1))
