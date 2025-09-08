from typing import Tuple
import torch
from sklearn.cluster import KMeans


def random_subsample(x: torch.Tensor, k: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """Randomly pick k tokens from sequence.

    Args:
        x: (batch, seq_len, dim)
        k: number to keep
    Returns:
        (batch, k, dim)
    """
    b, t, d = x.shape
    device = x.device
    if gen is None:
        gen = torch.Generator(device=device)
    idx = torch.stack([torch.randperm(t, generator=gen, device=device)[:k] for _ in range(b)], dim=0)
    out = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, d))
    return out


def topk_norm(x: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k tokens by L2 norm.

    Args:
        x: (batch, seq_len, dim)
    Returns:
        (batch, k, dim)
    """
    norms = x.norm(dim=-1)
    idx = norms.topk(k, dim=1).indices
    d = x.size(-1)
    out = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, d))
    return out


def uniform_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    """Uniformly segment the sequence into k chunks and average.

    Args:
        x: (batch, seq_len, dim)
    Returns:
        (batch, k, dim)
    """
    b, t, d = x.shape
    device = x.device
    # Compute boundaries
    edges = torch.linspace(0, t, steps=k + 1, device=device, dtype=torch.int64)
    pooled = []
    for i in range(k):
        s, e = edges[i].item(), edges[i + 1].item()
        if e <= s:
            e = min(s + 1, t)
        pooled.append(x[:, s:e, :].mean(dim=1, keepdim=True))
    return torch.cat(pooled, dim=1)


def kmeans_unweighted(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run unweighted k-means per batch on CPU (sklearn) to get merged tokens.

    Args:
        x: (batch, seq_len, dim)
        k: clusters
    Returns:
        merged: (batch, k, dim)
        labels: (batch, seq_len)
    """
    b, t, d = x.shape
    x_np = x.detach().cpu().numpy()
    merged = torch.zeros((b, k, d), dtype=x.dtype)
    labels_all = []
    for bi in range(b):
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(x_np[bi])
        labels = torch.from_numpy(km.labels_)
        centroids = torch.from_numpy(km.cluster_centers_).to(x.dtype)
        labels_all.append(labels)
        merged[bi] = centroids
    return merged.to(x.device), torch.stack(labels_all, dim=0).to(x.device)


