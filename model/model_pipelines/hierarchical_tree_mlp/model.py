# hierarchical_tree_mlp/model.py - Parent-to-child tree-routing MLP for raw 21x3 hand landmarks
from __future__ import annotations

import torch
import torch.nn as nn

ROOT_NODE = 0
TIP_NODES = (4, 8, 12, 16, 20)
PALM_NODES = frozenset((1, 5, 9, 13, 17))
PARENTS = {
    1: 0,
    5: 1,
    9: 5,
    13: 9,
    17: 13,
    2: 1,
    3: 2,
    4: 3,
    6: 5,
    7: 6,
    8: 7,
    10: 9,
    11: 10,
    12: 11,
    14: 13,
    15: 14,
    16: 15,
    18: 17,
    19: 18,
    20: 19,
}
ORDER = (0, 1, 5, 9, 13, 17, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)


def _validate_tree_spec(num_landmarks: int) -> None:
    if num_landmarks != 21:
        raise ValueError("HierarchicalTreeMLP is defined for exactly 21 MediaPipe hand landmarks.")

    if set(ORDER) != set(range(num_landmarks)):
        raise ValueError("ORDER must contain each landmark index exactly once.")

    if len(PARENTS) != num_landmarks - 1:
        raise ValueError("PARENTS must cover every non-root node.")

    positions = {node: idx for idx, node in enumerate(ORDER)}
    for node, parent in PARENTS.items():
        if node == ROOT_NODE:
            raise ValueError("Root node must not appear in PARENTS.")
        if positions[parent] >= positions[node]:
            raise ValueError("ORDER must process each parent before its child.")


class HierarchicalTreeMLP(nn.Module):
    """Parent-to-child tree-routing MLP for single-frame hand landmark classification.

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
        root_hidden_dim: int = 32,
        edge_hidden_dim: int = 32,
        readout_hidden_dim: int = 128,
        use_lappe: bool = False,
        lappe_dim: int = 0,
    ) -> None:
        super().__init__()
        _validate_tree_spec(num_landmarks)

        if coord_dim != 3:
            raise ValueError("HierarchicalTreeMLP expects 3D landmarks with coord_dim=3.")
        if use_lappe and lappe_dim <= 0:
            raise ValueError("lappe_dim must be positive when use_lappe=True.")
        if not use_lappe and lappe_dim != 0:
            raise ValueError("Set lappe_dim=0 when use_lappe=False.")

        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.parents = dict(PARENTS)
        self.order = tuple(ORDER)
        self.palm_nodes = frozenset(PALM_NODES)
        self.tip_nodes = tuple(TIP_NODES)

        root_in = coord_dim + (lappe_dim if use_lappe else 0)
        edge_in = hidden_dim + (coord_dim * 2) + (lappe_dim if use_lappe else 0)

        self.root_mlp = nn.Sequential(
            nn.Linear(root_in, root_hidden_dim),
            nn.GELU(),
            nn.Linear(root_hidden_dim, hidden_dim),
        )
        self.edge_mlp_palm = nn.Sequential(
            nn.Linear(edge_in, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, hidden_dim),
        )
        self.edge_mlp_finger = nn.Sequential(
            nn.Linear(edge_in, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear((1 + len(self.tip_nodes)) * hidden_dim, readout_hidden_dim),
            nn.GELU(),
            nn.Linear(readout_hidden_dim, num_classes),
        )

    @property
    def input_dim(self) -> int:
        return self.num_landmarks * self.coord_dim

    def _prepare_lappe(self, lappe: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.use_lappe:
            return None
        if lappe is None:
            raise ValueError("lappe must be provided when use_lappe=True.")
        if lappe.dim() != 2:
            raise ValueError(f"Expected lappe shape (21, {self.lappe_dim}), got {tuple(lappe.shape)}")
        if lappe.shape != (self.num_landmarks, self.lappe_dim):
            raise ValueError(f"Expected lappe shape ({self.num_landmarks}, {self.lappe_dim}), got {tuple(lappe.shape)}")

        lappe = lappe.to(device=device, dtype=dtype)
        return lappe.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x: torch.Tensor, lappe: torch.Tensor | None = None) -> torch.Tensor:
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

        batch_size = x.shape[0]
        lappe_batch = self._prepare_lappe(lappe, batch_size=batch_size, device=x.device, dtype=x.dtype)

        root_feat = x[:, ROOT_NODE, :]
        if lappe_batch is not None:
            root_feat = torch.cat([root_feat, lappe_batch[:, ROOT_NODE, :]], dim=-1)

        node_features: dict[int, torch.Tensor] = {ROOT_NODE: self.root_mlp(root_feat)}

        for node in self.order[1:]:
            parent = self.parents[node]
            parent_feat = node_features[parent]
            node_coord = x[:, node, :]
            rel_coord = x[:, node, :] - x[:, parent, :]
            feat = torch.cat([parent_feat, node_coord, rel_coord], dim=-1)
            if lappe_batch is not None:
                feat = torch.cat([feat, lappe_batch[:, node, :]], dim=-1)

            mlp = self.edge_mlp_palm if node in self.palm_nodes else self.edge_mlp_finger
            node_features[node] = mlp(feat)

        readout = torch.cat(
            [node_features[ROOT_NODE]] + [node_features[tip] for tip in self.tip_nodes],
            dim=-1,
        )
        return self.head(readout)
