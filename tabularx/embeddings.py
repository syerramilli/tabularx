import torch
from torch import Tensor, nn
from typing import List, Sequence, Optional
from dataclasses import dataclass 

def compute_quantile_bins(
    X: Tensor,
    n_bins: int = 48,
    max_rows: Optional[int] = None,
) -> List[Tensor]:
    """
    Compute per-feature bin edges using empirical quantiles

    Args:
        X: Tensor of shape (n_samples, n_features).
        n_bins: Desired number of bins per feature. The *maximum* number of
            edges is n_bins + 1, but after `.unique()` some features may end
            up with fewer edges (ragged bins).
        max_rows: If not None and X has more than max_rows rows, a random
            subset of max_rows rows is used to approximate the quantiles.

    Returns:
        bins: list of length n_features; each element is a 1D tensor of
              sorted bin edges for that feature (length >= 2).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {tuple(X.shape)}")
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("X must have at least 2 rows.")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    # Optional subsampling for huge datasets
    if max_rows is not None and n_samples > max_rows:
        idx = torch.randperm(n_samples, device=X.device)[:max_rows]
        Xq = X[idx]
    else:
        Xq = X

    # Quantile levels: 0, 1/n_bins, ..., 1
    q_levels = torch.linspace(
        0.0, 1.0, n_bins + 1, device=Xq.device, dtype=Xq.dtype
    )

    # (n_bins+1, n_features)
    q = torch.quantile(Xq, q_levels, dim=0)

    # One 1D tensor per feature, with duplicates removed (and sorted)
    bins: List[Tensor] = [q[:, j].unique() for j in range(n_features)]

    for j, b in enumerate(bins):
        if b.numel() < 2:
            raise ValueError(
                f"Feature {j} ended up with <2 distinct edges after quantiles; "
                "check that it has at least 2 distinct values."
            )

    return bins

class PiecewiseLinearEncoding(nn.Module):
    """
    Piecewise linear encoding for numerical features 

    For each feature j, You provide bin edges: b_0 < b_1 < ... < b_{T_j}. There will be T_j bins. 

    Input shape:  (*, n_features)
    Output shape: (*, n_features, max_n_bins)
    """

    def __init__(
        self, bins: List[Tensor]
    ) -> None:
        
        assert(len(bins) > 0)
        assert(len(bins[0]) > 0)
        super().__init__()
        
        n_features = len(bins)
        max_n_bins = max([len(l) - 1 for l in bins])
        # handle different number of features
        self.register_buffer(
            'mask',
            None if all(len(l) == max_n_bins for l in bins)
            else torch.row_stack([
                torch.cat([
                    # number of bins for this feature - 1
                    torch.ones(len(l) - 1, dtype=torch.bool),
                    # zero padded components
                    torch.zeros(max_n_bins - (len(l) - 1), dtype=torch.bool)
                ]) for l in bins
            ])
        )

        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        for i, bin_edges in enumerate(bins):
            n_bins = len(bin_edges) - 1
            # i is the feature index
            bin_width = bin_edges.diff()

            self.weight[i, :n_bins] = 1. / bin_width
            self.bias[i, :n_bins] = -bin_edges[:-1] / bin_width
        
    
    @property
    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]
    
    def forward(self, x: torch.Tensor, flatten: bool = False) -> torch.Tensor:
        x = torch.addcmul(self.bias, self.weight, x[..., None]).clamp(0., 1.)
        if flatten:
            x = x.flatten(-2) if self.mask is None else x[:, self.mask]
        return x


class PiecewiseLinearEmbedding(nn.Module):
    """
    Piecewise linear embedding for numerical features.

    Applies a PiecewiseLinearEncoding followed by a per-feature linear
    transformation and optional ReLU activation.

    Input shape:  (*, n_features)
    Output shape: (*, n_features, d_embedding)
    """

    def __init__(
        self,
        bins: List[Tensor],
        d_embedding: int,
        activation: bool = True,
    ) -> None:
        super().__init__()

        self.encoding = PiecewiseLinearEncoding(bins)
        self.d_embedding = d_embedding
        self.activation = activation

        n_features = len(bins)
        max_n_bins = self.encoding.get_max_n_bins

        # Linear transformation: (n_features, max_n_bins) -> (n_features, d_embedding)
        self.linear = nn.Parameter(torch.empty(n_features, max_n_bins, d_embedding))
        self.bias = nn.Parameter(torch.zeros(n_features, d_embedding))
        nn.init.xavier_uniform_(self.linear)

        if activation:
            self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, n_features)
        # encoded: (*, n_features, max_n_bins)
        encoded = self.encoding(x)

        # Apply per-feature linear: (*, n_features, max_n_bins) @ (n_features, max_n_bins, d_embedding)
        # -> (*, n_features, d_embedding)
        out = torch.einsum('...fm,fmd->...fd', encoded, self.linear) + self.bias

        if self.activation:
            out = self.relu(out)

        return out