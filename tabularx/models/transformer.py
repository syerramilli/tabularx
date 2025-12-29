from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
from torch import Tensor, nn

from ..embeddings import PiecewiseLinearEmbedding


@dataclass
class FTTransformerConfig:
    """
    Configuration for the Feature Tokenizer Transformer (FT-Transformer).

    Based on: Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (2021)

    Attributes:
        n_features: Number of numerical input features.
        output_dim: Size of the output layer.
        d_model: Transformer model dimension (also used as embedding dimension).
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_ffn: Dimension of the feedforward network (default: 4 * d_model).
        dropout: Dropout probability.
        activation: Activation function in feedforward layers.
        use_embedding_activation: Whether to apply ReLU in the embedding layer.
    """

    n_features: int
    output_dim: int
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 3
    d_ffn: int | None = None
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "relu"
    use_embedding_activation: bool = False

    def __post_init__(self) -> None:
        if self.n_features <= 0 or self.output_dim <= 0:
            raise ValueError("n_features and output_dim must be > 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.d_ffn is None:
            self.d_ffn = 4 * self.d_model
        elif self.d_ffn <= 0:
            raise ValueError("d_ffn must be > 0")


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data.

    Structure:
    1. Piecewise linear embedding: (batch, n_features) -> (batch, n_features, d_model)
    2. Prepend learnable [CLS] token: (batch, n_features + 1, d_model)
    3. Transformer encoder layers
    4. Extract [CLS] token output -> classification head

    The piecewise linear embedding is required for this model as raw numerical
    features need to be tokenized before being processed by the transformer.
    """

    def __init__(
        self,
        config: FTTransformerConfig,
        bins: List[Tensor],
    ) -> None:
        """
        Args:
            config: FTTransformerConfig instance.
            bins: List of bin edge tensors, one per feature. Required for
                piecewise linear embeddings.
        """
        super().__init__()
        self.config = config

        if bins is None:
            raise ValueError("bins must be provided for FTTransformer")
        if len(bins) != config.n_features:
            raise ValueError(
                f"len(bins) ({len(bins)}) must equal n_features ({config.n_features})"
            )

        # Piecewise linear embedding: d_embedding = d_model
        self.embedding = PiecewiseLinearEmbedding(
            bins=bins,
            d_embedding=config.d_model,
            activation=config.use_embedding_activation,
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # Final layer norm and classification head
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the FT-Transformer.

        Args:
            x: Input tensor of shape (batch, n_features).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        batch_size = x.shape[0]

        # Embed features: (batch, n_features) -> (batch, n_features, d_model)
        x = self.embedding(x)

        # Prepend [CLS] token: (batch, n_features + 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoder
        x = self.transformer(x)

        # Extract [CLS] token output (first position)
        cls_output = x[:, 0]

        # Final norm and classification head
        cls_output = self.final_norm(cls_output)
        return self.head(cls_output)
