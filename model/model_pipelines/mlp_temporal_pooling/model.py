# mlp_temporal_pooling/model.py - frame-wise MLP embedding followed by mean/max/std temporal pooling
import torch
import torch.nn as nn


class TemporalPoolingMLP(nn.Module):
    """프레임별 embedding 후 temporal summary를 합쳐 분류하는 MLP.

    입력 shape: (B, T, D)
    출력 shape: (B, num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.frame_embed = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_seq):
        # 각 프레임을 같은 MLP로 임베딩한 뒤, 마지막/최대/std 통계를 함께 사용한다.
        h = self.frame_embed(x_seq)
        pooled = torch.cat(
            [
                h[:, -1, :],  # 마지막 프레임 특징 (시퀀스 라벨링 기준)
                h.max(dim=1).values,
                h.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.head(pooled)
