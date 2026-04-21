# mediapipe_hand_landmarker/model.py - MediaPipe Hand Landmarker 프론트엔드 + MLP 분류기
# MediaPipe는 랜드마크 추출 전처리 단계에서 이미 실행됨.
# 여기서는 추출된 joint(63d) 좌표를 받는 MLP baseline을 분류기로 사용.
import torch.nn as nn


class MLPBaseline(nn.Module):
    """뷰어/런타임에서 MediaPipe 프론트엔드 뒤에 붙는 frame 분류기.

    입력 shape: (B, 63)
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
        # 실제 분류기는 mlp_baseline과 동일한 frame MLP다.
        return self.net(x)
