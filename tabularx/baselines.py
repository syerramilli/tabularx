from __future__ import annotations

from typing import List

from torch import Tensor, nn

from .config import MLPConfig, ResNetConfig
from .utils import activation_from_name


class MLP(nn.Module):
    """
    Multilayer perceptron baseline from Gorishniy et al. (2021), Eq. (1).

    Consists of ``num_layers`` blocks of ``Dropout(Activation(Linear(. )))`` followed
    by a linear prediction head.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        blocks: List[nn.Module] = []
        in_features = config.input_dim
        for _ in range(config.num_layers):
            activation = activation_from_name(config.activation)
            linear = nn.Linear(in_features, config.hidden_dim)
            block = nn.Sequential(
                linear,
                activation,
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            )
            blocks.append(block)
            in_features = config.hidden_dim

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(in_features, config.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.blocks(x)
        return self.head(h)


class ResNetBlock(nn.Module):
    """Residual block mirroring Equation (2) in the FT-Transformer paper."""

    def __init__(self, hidden_dim: int, dropout: float, activation_name: str) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_from_name(activation_name)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout1(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        return x + h


class TabularResNet(nn.Module):
    """
    ResNet-like baseline from Gorishniy et al. (2021), Eq. (2).

    Structure: input linear projection -> residual blocks -> BatchNorm + ReLU -> head.
    """

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config

        self.input_layer = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResNetBlock(config.hidden_dim, config.dropout, config.activation) for _ in range(config.num_blocks)]
        )
        self.prediction_bn = nn.BatchNorm1d(config.hidden_dim)
        self.prediction_activation = activation_from_name(config.activation)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.prediction_bn(h)
        h = self.prediction_activation(h)
        return self.head(h)
