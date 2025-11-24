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
    Compute per-feature bin edges using empirical quantiles, similar to the
    quantile-based branch of `compute_bins` in rtdl-num-embeddings.

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

@dataclass
class PiecewiseLinearEmbeddingConfig:
    """
    Configuration for piecewise linear embeddings for numerical features.

    Attributes:
        n_features: Number of numerical input features.
        d_embedding: Dimensionality of the output embedding for each feature.
        n_bins: Target number of bins when computing bins from data (e.g. via
            `compute_quantile_bins`). Not strictly required if you pass
            explicit bins to the embedding.
    """

    n_features: int
    d_embedding: int
    n_bins: int = 48

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError("n_features must be > 0")
        if self.d_embedding <= 0:
            raise ValueError("d_embedding must be > 0")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be > 0")

class PiecewiseLinearEmbedding(nn.Module):
    """
    Piecewise linear embedding for numerical features with
    **precomputed, possibly ragged bins**.

    For each feature j:
      - You provide bin edges: e_0 < e_1 < ... < e_{M_j}
      - There are M_j - 1 bins / intervals
      - We learn an embedding vector for each edge, and for x in [e_k, e_{k+1}]
        we linearly interpolate between the embeddings at e_k and e_{k+1}.

    Ragged bins are handled by padding edges to a common max length internally.

    Input shape:  (*, n_features)
    Output shape: (*, n_features, d_embedding)
    """

    def __init__(
        self,
        config: PiecewiseLinearEmbeddingConfig,
        bins: Optional[Sequence[Tensor]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.d_embedding = config.d_embedding

        # ----- Prepare bins -----
        if bins is None:
            # Fallback: evenly spaced bins in [-1, 1] per feature.
            # This keeps your current MLP working even if you don't pass real bins.
            edges = torch.linspace(-1.0, 1.0, config.n_bins + 1)
            bins = [edges.clone() for _ in range(self.n_features)]
        else:
            if len(bins) != self.n_features:
                raise ValueError(
                    f"len(bins) ({len(bins)}) must equal n_features ({self.n_features})"
                )

        cleaned_bins: List[Tensor] = []
        for j, b in enumerate(bins):
            if b.ndim != 1:
                raise ValueError(f"bins[{j}] must be 1D, got shape {tuple(b.shape)}")
            b = b.detach()
            if not torch.is_floating_point(b):
                b = b.to(torch.get_default_dtype())
            # Sort & deduplicate to be safe
            b_sorted, _ = torch.sort(b)
            b_unique = b_sorted.unique()
            if b_unique.numel() < 2:
                raise ValueError(
                    f"bins[{j}] must have at least 2 distinct edges, "
                    f"got {b_unique.numel()}"
                )
            cleaned_bins.append(b_unique)

        self._register_bins(cleaned_bins)
        self._create_parameters()

    # ---------- internal helpers for ragged bins ----------

    def _register_bins(self, bins: Sequence[Tensor]) -> None:
        """
        Store ragged bin edges in padded tensors as buffers.

        After this:
            self.bin_edges: (n_features, max_n_edges)
            self.bin_edge_lengths: (n_features,) number of valid edges per feature
        """
        n_features = self.n_features
        edge_lengths = torch.tensor(
            [int(b.numel()) for b in bins], dtype=torch.long
        )
        max_n_edges = int(edge_lengths.max().item())

        # Pad edges along last dimension
        bin_edges = torch.empty(
            n_features, max_n_edges, dtype=bins[0].dtype
        )
        for j, b in enumerate(bins):
            L = b.numel()
            bin_edges[j, :L] = b
            if L < max_n_edges:
                # Pad tail with last edge value (won't be used for interpolation)
                bin_edges[j, L:] = b[-1]

        self.max_n_edges = max_n_edges
        self.max_n_bins = max_n_edges - 1  # maximum number of intervals

        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_edge_lengths", edge_lengths)

    def _create_parameters(self) -> None:
        """
        Create learnable boundary embeddings for each feature and edge.
        Only the first bin_edge_lengths[j] entries are "real" for feature j.
        """
        # (n_features, max_n_edges, d_embedding)
        self.boundary_embeddings = nn.Parameter(
            torch.empty(
                self.n_features,
                self.max_n_edges,
                self.d_embedding,
            )
        )
        nn.init.xavier_uniform_(self.boundary_embeddings)

    # ---------- forward ----------

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed numerical features using piecewise linear interpolation.

        Args:
            x: Tensor of shape (*, n_features).

        Returns:
            Tensor of shape (*, n_features, d_embedding).
        """
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected last dim of x to be {self.n_features}, "
                f"got {x.shape[-1]}"
            )

        orig_shape = x.shape
        batch_dims = orig_shape[:-1]

        # Flatten batch dims: (*, n_features) -> (B, n_features)
        x_flat = x.reshape(-1, self.n_features)
        B = x_flat.shape[0]

        device = x_flat.device
        dtype = x_flat.dtype

        # Move buffers/params to correct device/dtype
        bin_edges = self.bin_edges.to(device=device, dtype=dtype)
        bin_edge_lengths = self.bin_edge_lengths
        boundary_embeddings = self.boundary_embeddings.to(
            device=device, dtype=dtype
        )

        # Output: (B, n_features, d_embedding)
        out = x_flat.new_empty((B, self.n_features, self.d_embedding))

        # Process each feature separately (ragged edges)
        for j in range(self.n_features):
            n_edges_j = int(bin_edge_lengths[j].item())
            n_bins_j = n_edges_j - 1

            edges_j = bin_edges[j, :n_edges_j]         # (n_edges_j,)
            emb_j = boundary_embeddings[j, :n_edges_j] # (n_edges_j, d_embedding)

            x_j = x_flat[:, j]                         # (B,)

            # Interior edges for bucketization
            if n_bins_j > 1:
                inner = edges_j[1:-1]                  # (n_edges_j - 2,)
                # bin_idx in [0, n_bins_j - 1]
                bin_idx = torch.bucketize(x_j, inner)
            else:
                # Only one bin: everything goes to bin 0
                bin_idx = torch.zeros_like(x_j, dtype=torch.long)

            # Left/right edges for this bin
            left_edge = edges_j[bin_idx]               # (B,)
            right_edge = edges_j[bin_idx + 1]          # (B,)

            # t in [0, 1] for interpolation between edges
            denom = right_edge - left_edge
            # Avoid division by zero
            denom = torch.where(
                denom.abs() < 1e-8,
                torch.ones_like(denom),
                denom,
            )
            t = (x_j - left_edge) / denom
            t = t.clamp(0.0, 1.0)                      # (B,)

            # Edge embeddings
            left_emb = emb_j[bin_idx]                  # (B, d_embedding)
            right_emb = emb_j[bin_idx + 1]             # (B, d_embedding)

            t_expanded = t.unsqueeze(1)                # (B, 1)
            out[:, j, :] = (1.0 - t_expanded) * left_emb + t_expanded * right_emb

        # Reshape back to original batch dims
        out = out.reshape(*batch_dims, self.n_features, self.d_embedding)
        return out
