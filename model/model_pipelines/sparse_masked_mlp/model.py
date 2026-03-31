# sparse_masked_mlp/model.py - Sparse masked MLP for raw 21x3 hand landmarks
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from _shared import HAND_CONNECTIONS


def build_adj_matrix(
    num_nodes: int,
    undirected_edges: list[tuple[int, int]],
    self_loops: bool = True,
) -> torch.Tensor:
    """Build a symmetric adjacency matrix from an undirected edge list."""
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for src, dst in undirected_edges:
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    if self_loops:
        adjacency.fill_diagonal_(1.0)
    return adjacency


def expand_block_mask(A: torch.Tensor, out_block: int, in_block: int) -> torch.Tensor:
    """Expand an NxN adjacency matrix into a block mask for masked linear weights."""
    num_nodes = A.size(0)
    mask = A[:, :, None, None].repeat(1, 1, out_block, in_block)
    return mask.permute(0, 2, 1, 3).contiguous().view(num_nodes * out_block, num_nodes * in_block)


class AnatomicalMaskedLinear(nn.Module):
    """Block-masked linear layer constrained by hand-joint adjacency."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        out_features: int,
        adjacency: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(num_nodes * out_features, num_nodes * in_features))
        self.bias = nn.Parameter(torch.zeros(num_nodes * out_features)) if bias else None

        self.register_buffer(
            "mask",
            expand_block_mask(adjacency, out_block=out_features, in_block=in_features),
            persistent=False,
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, N, Fin), got {tuple(x.shape)}")
        if x.shape[1] != self.num_nodes or x.shape[2] != self.in_features:
            raise ValueError(
                f"Expected shape (*, {self.num_nodes}, {self.in_features}), got {tuple(x.shape)}"
            )

        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, self.num_nodes * self.in_features)
        effective_weight = self.weight * self.mask
        y = F.linear(x_flat, effective_weight, self.bias)
        return y.reshape(batch_size, self.num_nodes, self.out_features)


class MaskedStage(nn.Module):
    """Masked linear stage with post-activation normalization and optional residual."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        out_features: int,
        adjacency: torch.Tensor,
        dropout: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and in_features == out_features
        self.masked_linear = AnatomicalMaskedLinear(
            num_nodes=num_nodes,
            in_features=in_features,
            out_features=out_features,
            adjacency=adjacency,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.masked_linear(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x


class SparseMaskedMLP(nn.Module):
    """Adjacency-masked MLP baseline for single-frame hand landmark classification.

    Accepted input shapes:
    - (B, 63)
    - (B, 21, 3)
    """

    def __init__(
        self,
        num_landmarks: int = 21,
        coord_dim: int = 3,
        num_classes: int = 7,
        hidden_dim: int = 64,
        readout_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        adjacency = build_adj_matrix(num_landmarks, HAND_CONNECTIONS, self_loops=True)
        self.register_buffer("adjacency", adjacency, persistent=False)

        self.stage1 = MaskedStage(
            num_nodes=num_landmarks,
            in_features=coord_dim,
            out_features=hidden_dim,
            adjacency=adjacency,
            dropout=dropout,
            use_residual=False,
        )
        self.stage2 = MaskedStage(
            num_nodes=num_landmarks,
            in_features=hidden_dim,
            out_features=hidden_dim,
            adjacency=adjacency,
            dropout=dropout,
            use_residual=True,
        )
        self.stage3 = MaskedStage(
            num_nodes=num_landmarks,
            in_features=hidden_dim,
            out_features=hidden_dim,
            adjacency=adjacency,
            dropout=dropout,
            use_residual=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, readout_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, num_classes),
        )

    @property
    def input_dim(self) -> int:
        return self.num_landmarks * self.coord_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if x.shape[1] != self.input_dim:
                raise ValueError(f"Expected flattened input dim {self.input_dim}, got {x.shape[1]}")
            x = x.reshape(x.shape[0], self.num_landmarks, self.coord_dim)
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (B, 63) or (B, 21, 3), got {tuple(x.shape)}")

        if x.shape[1] != self.num_landmarks or x.shape[2] != self.coord_dim:
            raise ValueError(
                f"Expected landmark tensor shape (*, {self.num_landmarks}, {self.coord_dim}), "
                f"got {tuple(x.shape)}"
            )

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.mean(dim=1)
        return self.head(x)
