from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Literal, Optional, Sequence, Tuple

from torch import Tensor, nn

from .adapters import AdapterConfig, EnsembleBlock, EnsembleLinear
from .batching import prepare_ensemble_inputs
from ..utils import activation_from_name, normalize_member_indices


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


class TabMBase(nn.Module):
    """
    Common building blocks shared between TabM and TabM-mini.

    The backbone follows the definition from the paper: ``N`` blocks of
    ``Dropout(ReLU(Linear(.)))`` followed by a linear head. Each block keeps
    track of the ensemble axis and optionally injects BatchEnsemble adapters.
    """

    def __init__(self, config: TabMConfig) -> None:
        super().__init__()
        self.config = config
        self._ensemble_size = config.ensemble_size

        self.blocks = nn.ModuleList()
        self._ensemble_layers: List[EnsembleLinear] = []

        in_features = config.input_dim
        for layer_idx in range(config.num_layers):
            adapter = self._adapter_for_layer(layer_idx)
            linear = EnsembleLinear(
                in_features,
                config.hidden_dim,
                config.ensemble_size,
                adapter=adapter,
                bias=True,
            )
            activation = activation_from_name(config.activation)
            block = EnsembleBlock(linear, dropout=config.dropout, activation=activation)
            self.blocks.append(block)
            self._ensemble_layers.append(linear)
            in_features = config.hidden_dim

        self.head = EnsembleLinear(
            in_features,
            config.output_dim,
            config.ensemble_size,
            adapter=self._head_adapter(),
            bias=True,
        )
        self._ensemble_layers.append(self.head)

    # --------------------------------------------------------------------- API
    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    def forward(
        self,
        x: Tensor,
        *,
        shared_batch: Optional[bool] = None,
        return_submodel_outputs: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Computes the mean prediction across all implicit submodels.

        Args:
            x: Batch of tabular features shaped ``(batch, features)`` or
                ``(batch, ensemble, features)``.
            shared_batch: Overrides ``config.share_batch``. When ``True`` the
                same mini-batch is broadcast to every submodel (TabM♠). When
                ``False`` the caller is responsible for supplying a distinct
                mini-batch per submodel.
            return_submodel_outputs: If ``True`` also returns the per-member
                predictions with shape ``(batch, ensemble, output_dim)``.
        """

        members = self.forward_members(x, shared_batch=shared_batch)
        mean_prediction = members.mean(dim=1)
        if return_submodel_outputs:
            return mean_prediction, members
        return mean_prediction

    def forward_members(self, x: Tensor, *, shared_batch: Optional[bool] = None) -> Tensor:
        """
        Returns the raw predictions of every implicit submodel.
        """

        shared = self.config.share_batch if shared_batch is None else shared_batch
        h = prepare_ensemble_inputs(x, self.ensemble_size, shared_batch=shared)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def prune_submodels(self, indices: Sequence[int]) -> None:
        """
        Permanently discards all ensemble members not listed in ``indices``.

        This implements the greedy pruning idea from Section 5.2 of the paper
        and updates every adapter matrix in-place. The operation is idempotent
        and can be called multiple times.
        """

        normalized = normalize_member_indices(indices, self.ensemble_size)
        if len(normalized) == self.ensemble_size:
            return

        for layer in self._ensemble_layers:
            layer.select_members(normalized)
        self._ensemble_size = len(normalized)
        self.config = replace(self.config, ensemble_size=self._ensemble_size)

    # ----------------------------------------------------------------- factories
    def _adapter_for_layer(self, layer_idx: int) -> AdapterConfig:
        return AdapterConfig.none()

    def _head_adapter(self) -> AdapterConfig:
        return AdapterConfig.none()


class TabMMini(TabMBase):
    """Minimal TabM obtained by keeping only the first ``R`` adapter."""

    def _adapter_for_layer(self, layer_idx: int) -> AdapterConfig:
        if layer_idx == 0:
            return AdapterConfig(use_r=True, use_s=False, use_b=False, r_init="sign")
        return AdapterConfig.none()


class TabM(TabMBase):
    """Full TabM model with adapters in every hidden block."""

    def _adapter_for_layer(self, layer_idx: int) -> AdapterConfig:
        if layer_idx == 0:
            return AdapterConfig(use_r=True, use_s=True, use_b=True, r_init="sign", s_init="sign", b_init="zeros")
        return AdapterConfig(use_r=True, use_s=True, use_b=True, r_init="ones", s_init="ones", b_init="zeros")
