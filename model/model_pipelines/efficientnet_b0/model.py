# efficientnet_b0/model.py - MBConv(SiLU) 기반 EfficientNet-like 분류기
# 입력: 랜드마크 스켈레톤을 렌더링한 단채널 이미지 (1 x H x W)
import torch.nn as nn


class MBConvBlock(nn.Module):
    """expand-depthwise-project 흐름의 EfficientNet 스타일 MBConv 블록."""

    def __init__(self, in_ch: int, out_ch: int, expand: int = 4, stride: int = 1):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.block(x)
        # stride=1 이고 채널 수가 같을 때만 residual을 유지한다.
        return x + y if self.use_res else y


class EfficientNetLike(nn.Module):
    """1채널 skeleton image 입력용 EfficientNet 계열 소형 분류기."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            # RGB crop이 아니라 skeleton grayscale image를 받으므로 입력 채널은 1이다.
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            MBConvBlock(32, 32, expand=1, stride=1),
            MBConvBlock(32, 48, expand=4, stride=2),
            MBConvBlock(48, 64, expand=4, stride=2),
            MBConvBlock(64, 96, expand=4, stride=2),
            MBConvBlock(96, 128, expand=4, stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # MBConv backbone의 출력만 pooled logits으로 연결한다.
        return self.head(self.blocks(self.stem(x)))
