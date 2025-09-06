import os
from typing import Tuple
import torch

_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(this_dir), "cuda")
    sources = [
        os.path.join(src_dir, "kmeans_cuda.cpp"),
        os.path.join(src_dir, "kmeans_cuda_kernel.cu"),
    ]
    _EXT = load(
        name="quickmergepp_kmeans",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXT


def weighted_kmeans_assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (n, d) float32 CUDA
        centroids: (k, d) float32 CUDA
    Returns:
        labels: (n,) int64 CUDA
    """
    ext = _load_extension()
    return ext.assign(x, centroids)


def weighted_kmeans_update(x: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """
    Args:
        x: (n, d) float32 CUDA
        weights: (n,) float32 CUDA
        labels: (n,) int64 CUDA
        k: clusters
    Returns:
        new_centroids: (k, d) float32 CUDA
    """
    ext = _load_extension()
    return ext.update(x, weights, labels, k)


