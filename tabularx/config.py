from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TabMConfig:
    """
    Hyper-parameters that fully describe a TabM-style multilayer perceptron.

    Attributes:
        input_dim: Number of numerical features fed into the backbone.
        output_dim: Size of the regression/logit head.
        hidden_dim: Width of every hidden block.
        num_layers: Number of hidden blocks (referred to as ``N`` in the paper).
        dropout: Dropout probability applied after every block.
        activation: Non-linearity used inside the hidden blocks.
        ensemble_size: Number of implicit submodels ``k``.
        share_batch: Whether the ``forward`` helper should replicate a single
            mini-batch across all submodels (the default TabM♠ strategy).
        dtype: Optional floating point precision hint. The code does not enforce
            the dtype but exposes it so callers can keep track of the intended
            precision.
    """

    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    activation: Literal["relu"] = "relu"
    ensemble_size: int = 32
    share_batch: bool = True
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be > 0")


@dataclass
class MLPConfig:
    """Configuration for the baseline MLP from Gorishniy et al. (2021)."""

    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    activation: Literal["relu"] = "relu"

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")
        if self.hidden_dim <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_dim/num_layers must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


@dataclass
class ResNetConfig:
    """Configuration for the tabular ResNet baseline (Equation 2 in the paper)."""

    input_dim: int
    output_dim: int
    hidden_dim: int = 1024
    num_blocks: int = 4
    dropout: float = 0.2
    activation: Literal["relu"] = "relu"

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")
        if self.hidden_dim <= 0 or self.num_blocks <= 0:
            raise ValueError("hidden_dim/num_blocks must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
