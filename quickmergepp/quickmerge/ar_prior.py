from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal mask
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        y, _ = self.attn(x, x, x, attn_mask=mask)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ARPrior(nn.Module):
    def __init__(self, dim: int = 512, depth: int = 6, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.pos = PositionalEncoding(dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        # Predict next token embedding
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        h = self.pos(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        y = self.head(h)
        return y


def bidirectional_ar_losses(prior: ARPrior, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute forward and backward AR MSE losses over compressed tokens.

    Args:
        prior: ARPrior model
        x: (batch, k, dim) compressed tokens
    Returns:
        l_forward, l_backward, l_total
    """
    # Forward: predict x[:, t+1] from prefix up to t
    y = prior(x)  # (B, K, D) predictions per position
    target_fwd = x[:, 1:, :]
    pred_fwd = y[:, :-1, :]
    l_forward = F.mse_loss(pred_fwd, target_fwd)

    # Backward: reverse sequence and predict previous
    xr = torch.flip(x, dims=[1])
    yr = prior(xr)
    target_bwd = xr[:, 1:, :]
    pred_bwd = yr[:, :-1, :]
    l_backward = F.mse_loss(pred_bwd, target_bwd)

    return l_forward, l_backward, l_forward + l_backward


