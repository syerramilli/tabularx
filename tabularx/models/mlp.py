from __future__ import annotations

from typing import List

from torch import Tensor, nn

from ..config import MLPConfig
from ..utils import activation_from_name


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
