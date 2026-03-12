# cnn1d_tcn/model.py - Temporal Convolutional Network (1D dilated conv) 제스처 분류기
import torch.nn as nn


class TCN1DClassifier(nn.Module):
    """dilated Conv1d로 시간축 패턴을 읽는 TCN 계열 분류기.

    입력 shape: (B, T, D)
    출력 shape: (B, num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq):
        # Conv1d는 channel-first를 기대하므로 feature 차원을 channel 축으로 옮긴다.
        h = self.net(x_seq.transpose(1, 2))
        # 마지막 time step의 표현을 시퀀스 대표값으로 사용한다.
        return self.head(h[..., -1])
