# transformer_embedding/model.py - Transformer Encoder + frame embedding (시계열 제스처 분류)
import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """frame embedding + positional encoding + encoder stack 기반 sequence 모델.

    입력 shape: (B, T, D)
    출력 shape: (B, num_classes)
    """

    def __init__(self, seq_len: int, input_dim: int, num_classes: int, d_model: int = 128):
        super().__init__()
        self.frame_embed = nn.Linear(input_dim, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_seq):
        # 각 frame feature를 d_model로 올리고 위치 임베딩을 더해 순서 정보를 보존한다.
        h = self.frame_embed(x_seq) + self.pos_embed[:, : x_seq.size(1)]
        h = self.encoder(h)
        # 시퀀스 라벨링 기준에 맞춰 마지막 토큰 표현으로 예측한다.
        return self.head(h[:, -1, :])
