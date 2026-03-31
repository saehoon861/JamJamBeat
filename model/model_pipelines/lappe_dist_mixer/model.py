# lappe_dist_mixer/model.py - LapPE + SPD distance-bucket DistMixer for raw 21x3 hand landmarks
from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

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


def laplacian_pe(A: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Return the smallest non-trivial normalized Laplacian eigenvectors."""
    if A.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got {tuple(A.shape)}")

    num_nodes = A.shape[0]
    if k <= 0:
        raise ValueError("k must be positive for Laplacian positional encoding.")
    if k >= num_nodes:
        raise ValueError(f"k must be smaller than num_nodes={num_nodes}, got {k}")

    deg = A.sum(dim=1)
    d_inv_sqrt = torch.diag(torch.clamp(deg, min=1e-6).pow(-0.5))
    laplacian = torch.eye(num_nodes, device=A.device, dtype=A.dtype) - d_inv_sqrt @ A @ d_inv_sqrt

    evals, evecs = torch.linalg.eigh(laplacian)
    del evals
    idx = torch.arange(1, 1 + k, device=A.device)
    return evecs[:, idx]


def spd_matrix_from_edges(num_nodes: int, undirected_edges: list[tuple[int, int]]) -> torch.Tensor:
    """Compute shortest-path distance matrix using repeated BFS."""
    adj = [set() for _ in range(num_nodes)]
    for src, dst in undirected_edges:
        adj[src].add(dst)
        adj[dst].add(src)

    inf = 10**9
    spd = torch.full((num_nodes, num_nodes), inf, dtype=torch.long)
    for start in range(num_nodes):
        spd[start, start] = 0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adj[node]:
                if spd[start, nxt] > spd[start, node] + 1:
                    spd[start, nxt] = spd[start, node] + 1
                    queue.append(nxt)
    return spd


def random_sign_flip(pe: torch.Tensor) -> torch.Tensor:
    """Apply independent random +/- sign flips to each PE channel."""
    if pe.dim() != 2:
        raise ValueError(f"Expected PE shape (N, k), got {tuple(pe.shape)}")
    signs = torch.where(
        torch.rand(pe.shape[1], device=pe.device) < 0.5,
        -torch.ones(pe.shape[1], device=pe.device, dtype=pe.dtype),
        torch.ones(pe.shape[1], device=pe.device, dtype=pe.dtype),
    )
    return pe * signs.view(1, -1)


class DistTokenMix(nn.Module):
    """Distance-bucket token mixing using SPD-derived masks."""

    def __init__(self, spd: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        if spd.dim() != 2 or spd.shape[0] != spd.shape[1]:
            raise ValueError(f"Expected square SPD matrix, got {tuple(spd.shape)}")
        if torch.any(spd < 0):
            raise ValueError("SPD matrix must contain non-negative distances.")

        self.register_buffer("spd", spd.to(dtype=torch.long), persistent=False)
        self.num_buckets = int(spd.max().item()) + 1
        bucket_masks = torch.stack([(spd == d).to(dtype=torch.float32) for d in range(self.num_buckets)], dim=0)
        self.register_buffer("bucket_masks", bucket_masks, persistent=False)
        self.alpha = nn.Parameter(torch.zeros(self.num_buckets, hidden_dim))
        nn.init.normal_(self.alpha, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected input shape (B, N, D), got {tuple(h.shape)}")

        out = torch.zeros_like(h)
        for dist in range(self.num_buckets):
            bucket = self.bucket_masks[dist].to(dtype=h.dtype)
            mixed = torch.einsum("ij,bjd->bid", bucket, h)
            out = out + mixed * self.alpha[dist].view(1, 1, -1)
        return out


class DistMixerBlock(nn.Module):
    """DistMixer block with SPD token mixing and per-node channel mixing."""

    def __init__(self, spd: torch.Tensor, hidden_dim: int, channel_mlp_hidden: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mix = DistTokenMix(spd=spd, hidden_dim=hidden_dim)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_hidden),
            nn.GELU(),
            nn.Linear(channel_mlp_hidden, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.token_mix(self.token_norm(h))
        h = h + self.channel_mlp(self.channel_norm(h))
        return h


class LapPEDistMixer(nn.Module):
    """LapPE + SPD distance-bucket DistMixer for single-frame hand landmarks.

    Accepted input shapes:
    - (B, 63)
    - (B, 21, 3)
    """

    def __init__(
        self,
        num_landmarks: int = 21,
        coord_dim: int = 3,
        num_classes: int = 7,
        lappe_dim: int = 8,
        hidden_dim: int = 64,
        channel_mlp_hidden: int = 128,
        num_layers: int = 3,
        lappe_sign_flip: bool = True,
    ) -> None:
        super().__init__()
        if num_landmarks != 21:
            raise ValueError("LapPEDistMixer is defined for exactly 21 MediaPipe hand landmarks.")
        if coord_dim != 3:
            raise ValueError("LapPEDistMixer expects 3D landmarks with coord_dim=3.")
        if lappe_dim <= 0:
            raise ValueError("lappe_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.num_classes = num_classes
        self.lappe_dim = lappe_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lappe_sign_flip = lappe_sign_flip

        adjacency = build_adj_matrix(num_landmarks, HAND_CONNECTIONS, self_loops=True)
        spd = spd_matrix_from_edges(num_landmarks, HAND_CONNECTIONS)
        lappe = laplacian_pe(adjacency, k=lappe_dim)

        self.register_buffer("adjacency", adjacency, persistent=False)
        self.register_buffer("spd", spd, persistent=False)
        self.register_buffer("lappe", lappe, persistent=False)

        self.input_proj = nn.Linear(coord_dim + lappe_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            DistMixerBlock(spd=spd, hidden_dim=hidden_dim, channel_mlp_hidden=channel_mlp_hidden)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @property
    def input_dim(self) -> int:
        return self.num_landmarks * self.coord_dim

    def _prepare_lappe(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lappe = self.lappe.to(device=device, dtype=dtype)
        if self.training and self.lappe_sign_flip:
            lappe = random_sign_flip(lappe)
        return lappe.unsqueeze(0).expand(batch_size, -1, -1)

    def alpha_l2_penalty(self) -> torch.Tensor:
        penalties = [block.token_mix.alpha.pow(2).mean() for block in self.blocks]
        return torch.stack(penalties).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if x.shape[1] != self.input_dim:
                raise ValueError(f"Expected flattened input dim {self.input_dim}, got {x.shape[1]}")
            x = x.reshape(x.shape[0], self.num_landmarks, self.coord_dim)
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (B, 63) or (B, 21, 3), got {tuple(x.shape)}")

        if x.shape[1] != self.num_landmarks or x.shape[2] != self.coord_dim:
            raise ValueError(
                f"Expected landmark tensor shape (*, {self.num_landmarks}, {self.coord_dim}), got {tuple(x.shape)}"
            )

        lappe = self._prepare_lappe(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        h = self.input_proj(torch.cat([x, lappe], dim=-1))
        for block in self.blocks:
            h = block(h)
        pooled = self.final_norm(h).mean(dim=1)
        return self.classifier(pooled)
