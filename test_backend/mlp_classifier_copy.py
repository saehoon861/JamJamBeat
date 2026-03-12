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

        # hidden_dims 순서대로 Linear-ReLU 블록을 쌓고 마지막에 분류 레이어를 붙인다.
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
        # 추론 서버에서는 이 호출 결과를 그대로 class logits으로 사용한다.
        return self.net(x)
