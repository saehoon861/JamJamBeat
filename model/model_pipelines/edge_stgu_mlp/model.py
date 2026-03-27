# edge_stgu_mlp/model.py - Edge-wise Spatial Topology Gating Unit MLP for raw 21x3 hand landmarks
from __future__ import annotations

import torch
import torch.nn as nn

from _shared import HAND_CONNECTIONS


def _build_directed_edge_index(num_landmarks: int = 21) -> tuple[torch.Tensor, torch.Tensor]:
    edges: list[tuple[int, int]] = []
    for src, dst in HAND_CONNECTIONS:
        edges.append((src, dst))
        edges.append((dst, src))
    for joint_idx in range(num_landmarks):
        edges.append((joint_idx, joint_idx))

    src_index = torch.tensor([src for src, _ in edges], dtype=torch.long)
    dst_index = torch.tensor([dst for _, dst in edges], dtype=torch.long)
    return src_index, dst_index


class EdgeSTGUBlock(nn.Module):
    """Edge-wise gated message passing block.

    Input shape:  (B, 21, d_model)
    Output shape: (B, 21, d_model)
    """

    def __init__(self, d_model: int, gate_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
    ) -> torch.Tensor:
        residual = h
        x = self.norm(h)

        src_feat = x[:, src_index, :]
        dst_feat = x[:, dst_index, :]
        values = self.value_proj(src_feat)
        gates = torch.sigmoid(self.gate_mlp(torch.cat([src_feat, dst_feat], dim=-1)))
        messages = gates * values

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(1, dst_index, messages)
        return residual + aggregated


class EdgeSTGUMLP(nn.Module):
    """Topology-aware MLP for single-frame hand landmark classification.

    Accepted input shapes:
    - (B, 63)
    - (B, 21, 3)
    """

    def __init__(
        self,
        num_landmarks: int = 21,
        coord_dim: int = 3,
        num_classes: int = 7,
        d_model: int = 192,
        num_layers: int = 3,
        gate_hidden: int = 96,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.input_proj = nn.Linear(coord_dim, d_model)
        self.joint_embed = nn.Embedding(num_landmarks, d_model)

        src_index, dst_index = _build_directed_edge_index(num_landmarks=num_landmarks)
        self.register_buffer("src_index", src_index, persistent=False)
        self.register_buffer("dst_index", dst_index, persistent=False)

        self.blocks = nn.ModuleList(
            EdgeSTGUBlock(d_model=d_model, gate_hidden=gate_hidden)
            for _ in range(num_layers)
        )
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    @property
    def num_edges(self) -> int:
        return int(self.src_index.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            expected_dim = self.num_landmarks * self.coord_dim
            if x.shape[1] != expected_dim:
                raise ValueError(f"Expected flattened input dim {expected_dim}, got {x.shape[1]}")
            x = x.reshape(x.shape[0], self.num_landmarks, self.coord_dim)
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (B, 63) or (B, 21, 3), got {tuple(x.shape)}")

        if x.shape[1] != self.num_landmarks or x.shape[2] != self.coord_dim:
            raise ValueError(
                f"Expected landmark tensor shape (*, {self.num_landmarks}, {self.coord_dim}), "
                f"got {tuple(x.shape)}"
            )

        h = self.input_proj(x)
        joint_ids = torch.arange(self.num_landmarks, device=x.device)
        h = h + self.joint_embed(joint_ids).unsqueeze(0)

        for block in self.blocks:
            h = block(h, self.src_index, self.dst_index)

        pooled = self.head_norm(h).mean(dim=1)
        return self.classifier(pooled)
