# mobilenetv3_small/model.py - Depthwise Separable Conv 기반 경량 MobileNet-like 분류기
# 입력: 랜드마크 스켈레톤을 렌더링한 단채널 이미지 (1 x H x W)
import torch.nn as nn


class DWConvBlock(nn.Module):
    """depthwise + pointwise conv 조합으로 채널 비용을 줄이는 블록."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MobileNetLike(nn.Module):
    """1채널 스켈레톤 이미지를 읽는 경량 CNN 분류기."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # stem이 Conv2d(1, ...) 인 이유는 입력이 grayscale skeleton image이기 때문이다.
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DWConvBlock(16, 24, stride=1),
            DWConvBlock(24, 40, stride=2),
            DWConvBlock(40, 64, stride=2),
            DWConvBlock(64, 96, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        # backbone feature map을 global pooling 후 선형 분류기로 보낸다.
        return self.head(self.features(x))
