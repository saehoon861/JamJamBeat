# import torch.nn as nn

# import torch.nn as nn

# class GestureMLP(nn.Module):

#     def __init__(self, input_dim, num_classes):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 50),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.ReLU(),
#             nn.Linear(50, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)


import torch.nn as nn


class GestureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 7,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)