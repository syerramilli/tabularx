from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


def activation_from_name(name: str) -> nn.Module:
    """Map a config string to an activation module."""

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def normalize_member_indices(indices: Sequence[int], upper_bound: int) -> List[int]:
    """
    Sanitizes a user-supplied list of submodel indices.

    The helper ensures that:
    * every index lies within ``[0, upper_bound)``,
    * duplicates are removed, preserving the original order,
    * at least one member remains.
    """

    if len(indices) == 0:
        raise ValueError("At least one submodel index must be provided.")
    tensor = torch.as_tensor(indices, dtype=torch.int64)
    if torch.any(tensor < 0) or torch.any(tensor >= upper_bound):
        raise ValueError(f"indices must lie in [0, {upper_bound}). Got {indices!r}")

    # Preserve order while removing duplicates.
    seen = set()
    normalized: List[int] = []
    for idx in tensor.tolist():
        if idx not in seen:
            seen.add(idx)
            normalized.append(idx)
    return normalized
