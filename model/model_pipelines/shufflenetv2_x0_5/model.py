# shufflenetv2_x0_5/model.py - Channel Shuffle 기반 극경량 분류기
# 입력: 랜드마크 스켈레톤을 렌더링한 단채널 이미지 (1 x H x W)
import torch
import torch.nn as nn


class ShuffleBlock(nn.Module):
    """channel shuffle로 정보 교환 비용을 낮추는 ShuffleNet 스타일 블록."""

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 2 == 0
        mid = channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
        # 그룹 분할된 채널을 섞어 pointwise / depthwise 경로 간 정보를 교환한다.
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        x = self._channel_shuffle(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.cat([x1, self.conv(x2)], dim=1)


class ShuffleNetLike(nn.Module):
    """1채널 스켈레톤 입력용 ShuffleNet 계열 경량 분류기."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            # 스켈레톤 렌더링 결과가 grayscale이므로 입력 채널 수는 1이다.
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage = nn.Sequential(
            ShuffleBlock(24),
            ShuffleBlock(24),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            ShuffleBlock(48),
            ShuffleBlock(48),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, num_classes))

    def forward(self, x):
        # stem -> shuffled stages -> global pooling -> classifier 순서로 흘린다.
        return self.head(self.stage(self.stem(x)))
