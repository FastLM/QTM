from typing import Tuple
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbel_noise) / max(temperature, 1e-6)
    return F.softmax(y, dim=-1)


def compute_token_masses(selection_probs: torch.Tensor, saliency: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return selection_probs * saliency + (1.0 - selection_probs) * eps


def cluster_tokens(x: torch.Tensor, k: int, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted k-means clustering assignments for tokens.

    Args:
        x: (batch, seq_len, dim)
        k: number of clusters
        weights: (batch, seq_len)
    Returns:
        assignments: (batch, seq_len) cluster indices in [0, k-1]
    """
    b, t, d = x.shape
    assignments = []
    x_np = x.detach().cpu().numpy()
    w_np = weights.detach().cpu().numpy()
    for bi in range(b):
        # KMeans in sklearn does not support sample_weight before v1.3 for kmeans, but we assume >=1.3
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(x_np[bi], sample_weight=w_np[bi])
        assignments.append(torch.from_numpy(km.labels_))
    return torch.stack(assignments, dim=0).to(x.device)


def merge_by_assignments(x: torch.Tensor, masses: torch.Tensor, assignments: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute mass-weighted merged tokens.

    Args:
        x: (batch, seq_len, dim)
        masses: (batch, seq_len)
        assignments: (batch, seq_len)
        k: number of clusters
    Returns:
        merged: (batch, k, dim)
    """
    b, t, d = x.shape
    merged = torch.zeros((b, k, d), dtype=x.dtype, device=x.device)
    for bi in range(b):
        for ci in range(k):
            mask = assignments[bi] == ci
            if mask.any():
                w = masses[bi][mask]
                v = x[bi][mask]
                denom = w.sum().clamp_min(1e-8)
                merged[bi, ci] = (w.unsqueeze(-1) * v).sum(dim=0) / denom
            else:
                # If empty cluster, fallback to zeros (or random token)
                merged[bi, ci] = 0.0
    return merged


def differentiable_token_merge(
    x: torch.Tensor,
    saliency: torch.Tensor,
    k_max: int,
    temperature: float = 0.5,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 2: Differentiable token merging using Gumbel-Softmax for soft selection
    and weighted k-means clustering for structure-aware grouping.

    Args:
        x: (batch, seq_len, dim)
        saliency: (batch, seq_len) normalized saliency
        k_max: number of output tokens
        temperature: Gumbel-Softmax temperature
        eps: small constant in mass

    Returns:
        merged_tokens: (batch, k_max, dim)
        assignments: (batch, seq_len)
    """
    b, t, d = x.shape
    # logits proportional to saliency
    logits = saliency
    sel_probs = gumbel_softmax_sample(logits, temperature=temperature)
    masses = compute_token_masses(sel_probs, saliency, eps=eps)
    assignments = cluster_tokens(x, k=k_max, weights=masses)
    merged = merge_by_assignments(x, masses, assignments, k=k_max)
    return merged, assignments


