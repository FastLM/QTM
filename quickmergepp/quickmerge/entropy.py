import torch
import torch.nn.functional as F
from typing import Tuple


def compute_attention_matrices(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention matrices per layer given hidden states.

    Args:
        hidden_states: Tensor of shape (num_layers, batch, seq_len, hidden_dim)

    Returns:
        attn: Tensor of shape (num_layers, batch, seq_len, seq_len)
    """
    num_layers, batch_size, seq_len, hidden_dim = hidden_states.shape
    scale = hidden_dim ** 0.5
    # (L, B, T, D) @ (L, B, D, T) -> (L, B, T, T)
    attn_scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2)) / scale
    attn = F.softmax(attn_scores, dim=-1)
    return attn


def attention_entropy(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute per-token attention entropy for each layer.

    Args:
        attn: (num_layers, batch, seq_len, seq_len)
        eps: numerical stability

    Returns:
        entropies: (num_layers, batch, seq_len)
    """
    attn_clamped = attn.clamp_min(eps)
    ent = -(attn_clamped * (attn_clamped.log())).sum(dim=-1)
    return ent


def normalize_per_layer(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize along the last dimension per layer and batch: z-score then min-max to [0,1].

    Args:
        x: (num_layers, batch, seq_len)
    Returns:
        normalized: (num_layers, batch, seq_len)
    """
    # Z-score per (layer,batch)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    z = (x - mean) / std
    # Min-max to [0,1]
    min_vals = z.amin(dim=-1, keepdim=True)
    max_vals = z.amax(dim=-1, keepdim=True)
    denom = (max_vals - min_vals).clamp_min(eps)
    norm = (z - min_vals) / denom
    return norm


def multi_scale_entropy_saliency(
    hidden_states: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute multi-layer attention entropies and averaged normalized saliency scores.

    Args:
        hidden_states: (num_layers, batch, seq_len, hidden_dim)
        eps: stability

    Returns:
        saliency: (batch, seq_len) averaged normalized entropy across layers
        entropies: (num_layers, batch, seq_len)
    """
    attn = compute_attention_matrices(hidden_states)
    ent = attention_entropy(attn, eps=eps)
    ent_norm = normalize_per_layer(ent)
    saliency = ent_norm.mean(dim=0)  # average over layers -> (B, T)
    return saliency, ent


def normalized_entropy_drop(
    incoming_entropy: torch.Tensor,
    outgoing_entropy: torch.Tensor,
    delta: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Normalized Entropy Drop (NED) per token.

    Args:
        incoming_entropy: (num_layers, batch, seq_len) or (batch, seq_len)
        outgoing_entropy: same shape as incoming_entropy
        delta: small constant to avoid division by zero

    Returns:
        ned: same shape as inputs
    """
    return (incoming_entropy - outgoing_entropy) / (incoming_entropy + delta)


