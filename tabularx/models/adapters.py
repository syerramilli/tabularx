from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


AdapterInit = Literal["sign", "ones", "normal", "zeros"]


@dataclass(frozen=True)
class AdapterConfig:
    """Specifies which BatchEnsemble adapters are active inside a layer."""

    use_r: bool = False
    use_s: bool = False
    use_b: bool = False
    r_init: AdapterInit = "sign"
    s_init: AdapterInit = "sign"
    b_init: AdapterInit = "zeros"

    @classmethod
    def none(cls) -> "AdapterConfig":
        return cls(False, False, False, "ones", "ones", "zeros")


def _init_parameter(param: nn.Parameter, strategy: AdapterInit) -> None:
    if strategy == "sign":
        # Random ±1 following the BatchEnsemble recipe.
        values = torch.randint_like(param.data, low=0, high=2, dtype=torch.int64)
        param.data.copy_(values.float().mul_(2).sub_(1))
    elif strategy == "ones":
        param.data.fill_(1.0)
    elif strategy == "normal":
        nn.init.normal_(param)
    elif strategy == "zeros":
        param.data.zero_()
    else:
        raise ValueError(f"Unknown init strategy: {strategy}")


class EnsembleLinear(nn.Module):
    """
    Shared linear projection that optionally injects BatchEnsemble adapters.

    The input is expected to have the shape ``(batch, ensemble, features)`` and the
    output preserves the ``ensemble`` axis. When no adapters are enabled the layer
    behaves like a regular ``nn.Linear`` applied to every submodel independently.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        *,
        adapter: AdapterConfig | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._ensemble_size = ensemble_size
        self.adapter_cfg = adapter or AdapterConfig.none()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.shared_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("shared_bias", None)

        if self.adapter_cfg.use_r:
            self.adapter_r = nn.Parameter(torch.empty(ensemble_size, in_features))
            _init_parameter(self.adapter_r, self.adapter_cfg.r_init)
        else:
            self.register_parameter("adapter_r", None)

        if self.adapter_cfg.use_s:
            self.adapter_s = nn.Parameter(torch.empty(ensemble_size, out_features))
            _init_parameter(self.adapter_s, self.adapter_cfg.s_init)
        else:
            self.register_parameter("adapter_s", None)

        if self.adapter_cfg.use_b:
            self.adapter_b = nn.Parameter(torch.empty(ensemble_size, out_features))
            _init_parameter(self.adapter_b, self.adapter_cfg.b_init)
        else:
            self.register_parameter("adapter_b", None)

    @property
    def ensemble_size(self) -> int:
        return self._ensemble_size

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input with 3 dims (batch, ensemble, features), got {tuple(x.shape)}")
        batch, ensemble, in_features = x.shape
        if ensemble != self._ensemble_size:
            raise ValueError(
                f"Ensemble axis mismatch: got {ensemble}, expected {self._ensemble_size}. "
                "The input probably was not broadcast correctly.",
            )
        if in_features != self.in_features:
            raise ValueError(f"Feature mismatch: got {in_features}, expected {self.in_features}")

        if self.adapter_r is not None:
            adapter_r = self.adapter_r.to(dtype=x.dtype)
            x = x * adapter_r.view(1, ensemble, in_features)

        flat = x.reshape(batch * ensemble, in_features)
        projected = F.linear(flat, self.weight, self.shared_bias)
        projected = projected.reshape(batch, ensemble, self.out_features)

        if self.adapter_s is not None:
            adapter_s = self.adapter_s.to(dtype=projected.dtype)
            projected = projected * adapter_s.view(1, ensemble, self.out_features)
        if self.adapter_b is not None:
            adapter_b = self.adapter_b.to(dtype=projected.dtype)
            projected = projected + adapter_b.view(1, ensemble, self.out_features)
        return projected

    def select_members(self, indices: Sequence[int]) -> None:
        """
        Permanently keep only a subset of ensemble members.

        The helper mirrors the greedy post-hoc pruning procedure described in the
        paper and updates the adapter matrices in-place.
        """

        if len(indices) == self._ensemble_size:
            return
        if len(indices) == 0:
            raise ValueError("Cannot prune all members – at least one member must remain.")
        device = self.weight.device
        idx = torch.as_tensor(indices, device=device, dtype=torch.int64)
        if idx.dim() != 1:
            raise ValueError("indices must be a 1-D tensor or list")
        if torch.any(idx < 0) or torch.any(idx >= self._ensemble_size):
            raise ValueError(f"indices must lie in [0, {self._ensemble_size})")

        if self.adapter_r is not None:
            new_r = self.adapter_r.index_select(0, idx)
            self.adapter_r = nn.Parameter(new_r)
        if self.adapter_s is not None:
            new_s = self.adapter_s.index_select(0, idx)
            self.adapter_s = nn.Parameter(new_s)
        if self.adapter_b is not None:
            new_b = self.adapter_b.index_select(0, idx)
            self.adapter_b = nn.Parameter(new_b)

        self._ensemble_size = len(indices)


class EnsembleBlock(nn.Module):
    """A single Dropout(ReLU(Linear(.))) block that preserves the ensemble axis."""

    def __init__(
        self,
        linear: EnsembleLinear,
        *,
        dropout: float,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.activation = activation or nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
