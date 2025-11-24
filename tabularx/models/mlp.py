from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from torch import Tensor, nn

from ..embeddings import PiecewiseLinearEmbedding, PiecewiseLinearEmbeddingConfig
from ..utils import activation_from_name


@dataclass
class MLPConfig:
    """
    Configuration for the baseline MLP from Gorishniy et al. (2021).

    Attributes:
        input_dim: Number of numerical features (only used when use_embeddings=False).
        output_dim: Size of the output layer.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers.
        dropout: Dropout probability applied after each layer.
        activation: Non-linearity used in hidden layers.
        use_embeddings: Whether to use piecewise linear embeddings for input features.
        d_embedding: Embedding dimension per feature (only used when use_embeddings=True).
        n_bins: Target number of bins for piecewise linear embeddings
            (used when computing quantile bins, or for uniform bins if
             precomputed bins are not provided).
        use_precomputed_bins: If True, the MLP __init__ expects an explicit
            `bins` argument and will pass it to PiecewiseLinearEmbedding.
    """

    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    activation: Literal["relu"] = "relu"
    use_embeddings: bool = False
    d_embedding: Optional[int] = None
    n_bins: int = 48
    use_precomputed_bins: bool = False  # <- new

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")
        if self.hidden_dim <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_dim/num_layers must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.use_embeddings:
            if self.d_embedding is None or self.d_embedding <= 0:
                raise ValueError("d_embedding must be > 0 when use_embeddings=True")
            if self.n_bins <= 0:
                raise ValueError("n_bins must be > 0")


class MLP(nn.Module):
    """
    Consists of ``num_layers`` blocks of ``Dropout(Activation(Linear(. )))`` followed
    by a linear prediction head.

    Optionally supports piecewise linear embeddings for numerical features.
    When embeddings are enabled, the input is first transformed from
    (batch, n_features) -> (batch, n_features, d_embedding) -> (batch, n_features * d_embedding)
    before being passed through the MLP blocks.
    """

    def __init__(
        self,
        config: MLPConfig,
        bins: Optional[List[Tensor]] = None,
    ) -> None:
        """
        Args:
            config: MLPConfig instance.
            bins: Optional list of bin edge tensors, one per feature,
                to be passed into PiecewiseLinearEmbedding. Only used when
                config.use_embeddings and config.use_precomputed_bins are True.
        """
        super().__init__()
        self.config = config

        # Optional embedding layer
        self.embedding: Optional[PiecewiseLinearEmbedding] = None
        if config.use_embeddings:
            assert config.d_embedding is not None, "d_embedding must be set when use_embeddings=True"

            embed_config = PiecewiseLinearEmbeddingConfig(
                n_features=config.input_dim,
                d_embedding=config.d_embedding,
                n_bins=config.n_bins,
            )

            if config.use_precomputed_bins:
                if bins is None:
                    raise ValueError(
                        "bins must be provided when use_embeddings=True and "
                        "use_precomputed_bins=True"
                    )
                self.embedding = PiecewiseLinearEmbedding(embed_config, bins=bins)
            else:
                # fallback: internally constructed (uniform) bins
                self.embedding = PiecewiseLinearEmbedding(embed_config)

            in_features = config.input_dim * config.d_embedding
        else:
            in_features = config.input_dim

        # MLP blocks
        blocks: List[nn.Module] = []
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
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch, n_features).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        if self.embedding is not None:
            # x: (batch, n_features) -> (batch, n_features, d_embedding)
            x = self.embedding(x)
            # Flatten: (batch, n_features, d_embedding) -> (batch, n_features * d_embedding)
            x = x.reshape(x.shape[0], -1)

        h = self.blocks(x)
        return self.head(h)