import torch
import torch.nn as nn


class LandmarkSpatialTransformer(nn.Module):
    """
    단일 프레임 손 랜드마크 분류 모델
    입력:  (B, 21, 3)
    출력:  (B, num_classes)
    """

    def __init__(
        self,
        num_landmarks: int = 21,
        coord_dim: int = 3,
        num_classes: int = 7,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.2,
        use_cls_token: bool = False,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # 1) 좌표 -> 임베딩
        self.coord_embed = nn.Sequential(
            nn.Linear(coord_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 2) 랜드마크 ID 임베딩
        self.landmark_embed = nn.Embedding(num_landmarks, d_model)

        # 3) CLS 토큰 (선택)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias =True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 5) 정규화
        self.norm = nn.LayerNorm(d_model)

        # 6) 분류기
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        x: (B, 21, 3)
        """
        # [Fix] Viewer supplies (B, 63), so we auto-reshape if needed
        if x.dim() == 2:
            x = x.view(x.shape[0], self.num_landmarks, self.coord_dim)

        B, N, C = x.shape
        assert N == self.num_landmarks, f"Expected {self.num_landmarks} landmarks, got {N}"
        assert C == self.coord_dim, f"Expected coord dim {self.coord_dim}, got {C}"

        # 좌표 임베딩
        x = self.coord_embed(x)   # (B, 21, d_model)

        # 랜드마크 ID 임베딩 추가
        landmark_ids = torch.arange(self.num_landmarks, device=x.device)  # (21,)
        landmark_ids = landmark_ids.unsqueeze(0).expand(B, -1)            # (B, 21)
        landmark_tokens = self.landmark_embed(landmark_ids)               # (B, 21, d_model)

        x = x + landmark_tokens

        # CLS 토큰 추가 옵션
        if self.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_token, x], dim=1)          # (B, 22, d_model)

        # Transformer
        x = self.transformer(x)   # (B, 21 or 22, d_model)
        x = self.norm(x)

        # Pooling
        if self.use_cls_token:
            feat = x[:, 0]        # CLS token 사용
        else:
            feat = x.mean(dim=1)  # mean pooling 사용

        # 분류
        logits = self.classifier(feat)  # (B, num_classes)
        return logits