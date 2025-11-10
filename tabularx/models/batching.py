from __future__ import annotations

import torch
from torch import Tensor


def prepare_ensemble_inputs(
    x: Tensor,
    ensemble_size: int,
    *,
    shared_batch: bool = True,
) -> Tensor:
    """
    Reshape input tensors to ``(batch, ensemble, features)``.

    Args:
        x: Tensor of shape ``(batch, features)``, ``(batch, ensemble, features)`` or
            ``(batch * ensemble, features)`` when ``shared_batch=False``.
        ensemble_size: Number of parallel members ``k``.
        shared_batch: If ``True`` and ``x`` is 2-D, the function replicates the
            batch across the ensemble axis (TabM♠). Otherwise, the leading
            dimension must already contain ``batch * ensemble`` objects.
    """

    if x.dim() == 2:
        batch, features = x.shape
        if shared_batch:
            expanded = x.unsqueeze(1).expand(batch, ensemble_size, features)
            return expanded.contiguous()
        if batch % ensemble_size != 0:
            raise ValueError(
                "When shared_batch=False and a 2-D tensor is provided, the batch dimension "
                f"must be divisible by ensemble_size={ensemble_size}. Got {batch}.",
            )
        new_batch = batch // ensemble_size
        return x.reshape(new_batch, ensemble_size, features)

    if x.dim() == 3:
        batch, ensemble, features = x.shape
        if ensemble != ensemble_size:
            raise ValueError(
                f"The ensemble axis of the input ({ensemble}) does not match "
                f"the configured ensemble_size ({ensemble_size}).",
            )
        return x

    raise ValueError("Input must have 2 or 3 dimensions.")


def collapse_ensemble(x: Tensor) -> Tensor:
    """
    Flattens ``(batch, ensemble, features)`` tensors back to ``(batch * ensemble, features)``.
    """

    if x.dim() != 3:
        raise ValueError("collapse_ensemble expects a 3-D tensor.")
    b, k, f = x.shape
    return x.reshape(b * k, f)
