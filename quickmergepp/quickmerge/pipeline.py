from typing import Dict, Tuple
import torch

from .entropy import multi_scale_entropy_saliency
from .merge import differentiable_token_merge
from .ar_prior import ARPrior


class QuickMergePP:
    def __init__(self, dim: int, k_max: int, temperature: float = 0.5):
        self.dim = dim
        self.k_max = k_max
        self.temperature = temperature
        self.prior = ARPrior(dim=dim)

    @torch.no_grad()
    def compress(self, hidden_states_per_layer: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run stages 1-2 to produce compressed tokens.

        Args:
            hidden_states_per_layer: (num_layers, batch, seq_len, dim)
        Returns:
            merged: (batch, k_max, dim)
            info: dict with saliency and assignments
        """
        saliency, _ = multi_scale_entropy_saliency(hidden_states_per_layer)
        x = hidden_states_per_layer[-1]  # use last layer embeddings as inputs (B, T, D)
        merged, assignments = differentiable_token_merge(
            x=x, saliency=saliency, k_max=self.k_max, temperature=self.temperature
        )
        return merged, {"saliency": saliency, "assignments": assignments}

    def ar_predict(self, merged: torch.Tensor) -> torch.Tensor:
        """Run forward AR prior to predict next embeddings for each position."""
        return self.prior(merged)


