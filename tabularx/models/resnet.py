from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from torch import Tensor, nn

from ..embeddings import PiecewiseLinearEncoding, PiecewiseLinearEmbedding
from ..utils import activation_from_name


@dataclass
class ResNetConfig:
    """
    Configuration for the tabular ResNet baseline (Equation 2 in the paper).

    Attributes:
        input_dim: Number of numerical features.
        output_dim: Size of the output layer.
        hidden_dim: Width of hidden layers and residual blocks.
        num_blocks: Number of residual blocks.
        dropout: Dropout probability applied in residual blocks.
        activation: Non-linearity used in residual blocks.
        use_ple: Whether to use piecewise linear encoding (no learnable params).
        use_embeddings: Whether to use piecewise linear embeddings for input features.
        use_embedding_activation: Whether to apply ReLU in the embedding layer
            (only used when use_embeddings=True).
        d_embedding: Embedding dimension per feature (only used when use_embeddings=True).
    """

    input_dim: int
    output_dim: int
    hidden_dim: int = 1024
    num_blocks: int = 4
    dropout: float = 0.2
    activation: Literal["relu"] = "relu"
    use_ple: bool = False
    use_embeddings: bool = False
    use_embedding_activation: bool = False
    d_embedding: Optional[int] = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")
        if self.hidden_dim <= 0 or self.num_blocks <= 0:
            raise ValueError("hidden_dim/num_blocks must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.use_ple and self.use_embeddings:
            raise ValueError("use_ple and use_embeddings cannot both be True")
        if self.use_embeddings:
            if self.d_embedding is None or self.d_embedding <= 0:
                raise ValueError("d_embedding must be > 0 when use_embeddings=True")


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

    Optionally supports:
    - Piecewise linear encoding (use_ple): Fixed transformation, no learnable params.
    - Piecewise linear embeddings (use_embeddings): Learnable per-feature projections.
    """

    def __init__(
        self,
        config: ResNetConfig,
        bins: Optional[List[Tensor]] = None,
    ) -> None:
        """
        Args:
            config: ResNetConfig instance.
            bins: List of bin edge tensors, one per feature. Required when
                use_ple=True or use_embeddings=True.
        """
        super().__init__()
        self.config = config

        self.encoding: Optional[PiecewiseLinearEncoding] = None
        self.embedding: Optional[PiecewiseLinearEmbedding] = None

        if config.use_ple:
            if bins is None:
                raise ValueError("bins must be provided when use_ple=True")
            self.encoding = PiecewiseLinearEncoding(bins)
            # Output dim after flatten: sum of (n_edges - 1) per feature
            input_features = sum(len(b) - 1 for b in bins)

        elif config.use_embeddings:
            if bins is None:
                raise ValueError("bins must be provided when use_embeddings=True")
            assert config.d_embedding is not None
            self.embedding = PiecewiseLinearEmbedding(
                bins=bins,
                d_embedding=config.d_embedding,
                activation=config.use_embedding_activation,
            )
            input_features = config.input_dim * config.d_embedding

        else:
            input_features = config.input_dim

        self.input_layer = nn.Linear(input_features, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResNetBlock(config.hidden_dim, config.dropout, config.activation) for _ in range(config.num_blocks)]
        )
        self.prediction_bn = nn.BatchNorm1d(config.hidden_dim)
        self.prediction_activation = activation_from_name(config.activation)
        self.head = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ResNet.

        Args:
            x: Input tensor of shape (batch, n_features).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        if self.encoding is not None:
            # x: (batch, n_features) -> (batch, total_bins) via flatten=True
            x = self.encoding(x, flatten=True)

        elif self.embedding is not None:
            # x: (batch, n_features) -> (batch, n_features, d_embedding)
            x = self.embedding(x)
            # Flatten: (batch, n_features, d_embedding) -> (batch, n_features * d_embedding)
            x = x.reshape(x.shape[0], -1)

        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.prediction_bn(h)
        h = self.prediction_activation(h)
        return self.head(h)
