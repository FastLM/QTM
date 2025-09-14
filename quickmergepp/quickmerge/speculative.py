from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pipeline import QuickMergePP
from .ar_prior import ARPrior


class DraftModel(nn.Module):
    """Lightweight draft model for speculative decoding."""
    
    def __init__(self, vocab_size: int, dim: int, depth: int = 2, heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(1024, dim)  # max_len=1024
        self.transformer = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim*2,
                dropout=0.0, batch_first=True
            ) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, past_key_values=None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        
        for layer in self.transformer:
            x = layer(x, x)  # self-attention only for simplicity
            
        x = self.ln(x)
        logits = self.head(x)
        return logits


class SpeculativeDecoder:
    """Speculative decoding with QuickMerge++ token compression."""
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: DraftModel,
        quickmerge: QuickMergePP,
        max_draft_len: int = 4,
        temperature: float = 1.0
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.quickmerge = quickmerge
        self.max_draft_len = max_draft_len
        self.temperature = temperature
        
    def _sample_from_logits(self, logits: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """Sample tokens from logits."""
        if temperature is None:
            temperature = self.temperature
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1))
        else:
            return logits.argmax(dim=-1)
    
    def _compute_acceptance_prob(self, draft_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        """Compute acceptance probability for each draft token."""
        # Simplified acceptance based on logit agreement
        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Acceptance probability = min(1, target_prob / draft_prob)
        acceptance = torch.min(
            torch.ones_like(draft_probs),
            target_probs / (draft_probs + 1e-8)
        )
        
        # Sample acceptance for each token
        uniform = torch.rand_like(acceptance)
        accepted = uniform < acceptance
        
        return accepted
    
    @torch.no_grad()
    def generate_draft(
        self, 
        input_ids: torch.Tensor, 
        hidden_states: torch.Tensor,
        compressed_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens using compressed representation."""
        batch_size, seq_len = input_ids.shape
        
        # Use compressed tokens as context for draft generation
        draft_input = compressed_tokens  # (batch, k, dim)
        
        # Generate draft tokens
        draft_tokens = []
        current_input = input_ids
        
        for _ in range(self.max_draft_len):
            # Get draft model predictions
            draft_logits = self.draft_model(current_input)
            next_token = self._sample_from_logits(draft_logits[:, -1:, :])
            
            draft_tokens.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=1)
        
        draft_tokens = torch.cat(draft_tokens, dim=1)  # (batch, max_draft_len)
        return draft_tokens, current_input
    
    @torch.no_grad()
    def speculative_decode_step(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Single speculative decoding step.
        
        Returns:
            accepted_tokens: accepted tokens from this step
            num_accepted: number of accepted tokens
            speedup_ratio: estimated speedup
        """
        # Compress input using QuickMerge++
        compressed_tokens, _ = self.quickmerge.compress(hidden_states)
        
        # Generate draft tokens
        draft_tokens, extended_input = self.generate_draft(input_ids, hidden_states, compressed_tokens)
        
        # Get target model predictions for all draft positions
        target_logits = self.target_model(extended_input)
        draft_logits = self.draft_model(extended_input)
        
        # Compute acceptance probabilities
        acceptance_mask = self._compute_acceptance_prob(
            draft_logits[:, -self.max_draft_len:, :],
            target_logits[:, -self.max_draft_len:, :]
        )
        
        # Find first rejection
        num_accepted = 0
        for i in range(self.max_draft_len):
            if acceptance_mask[:, i].all():  # All batches accept this token
                num_accepted += 1
            else:
                break
        
        # Return accepted tokens
        accepted_tokens = draft_tokens[:, :num_accepted] if num_accepted > 0 else torch.empty(0, 0, dtype=torch.long, device=input_ids.device)
        
        # Estimate speedup (simplified)
        speedup_ratio = (1 + num_accepted) / (1 + self.max_draft_len) if num_accepted < self.max_draft_len else self.max_draft_len
        
        return accepted_tokens, num_accepted, speedup_ratio
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        max_new_tokens: int = 50,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate sequence using speculative decoding.
        
        Returns:
            generated_ids: final generated sequence
            stats: generation statistics
        """
        current_ids = input_ids.clone()
        stats = {
            'total_tokens': 0,
            'accepted_tokens': 0,
            'draft_tokens': 0,
            'speedup_ratios': []
        }
        
        for step in range(max_new_tokens):
            # Single speculative step
            accepted_tokens, num_accepted, speedup = self.speculative_decode_step(
                current_ids, hidden_states
            )
            
            # Update statistics
            stats['total_tokens'] += self.max_draft_len
            stats['accepted_tokens'] += num_accepted
            stats['draft_tokens'] += self.max_draft_len
            stats['speedup_ratios'].append(speedup)
            
            if verbose:
                print(f"Step {step}: accepted {num_accepted}/{self.max_draft_len} tokens, speedup: {speedup:.2f}")
            
            # Add accepted tokens to sequence
            if num_accepted > 0:
                current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
            else:
                # No tokens accepted, sample one from target model
                target_logits = self.target_model(current_ids)
                next_token = self._sample_from_logits(target_logits[:, -1:, :])
                current_ids = torch.cat([current_ids, next_token], dim=1)
                stats['total_tokens'] += 1
                stats['accepted_tokens'] += 1
                stats['draft_tokens'] += 1
            
            # Update hidden states for next iteration (simplified)
            if hasattr(self.target_model, 'get_last_hidden_states'):
                hidden_states = self.target_model.get_last_hidden_states()
        
        # Compute final statistics
        stats['acceptance_rate'] = stats['accepted_tokens'] / stats['total_tokens'] if stats['total_tokens'] > 0 else 0
        stats['avg_speedup'] = sum(stats['speedup_ratios']) / len(stats['speedup_ratios']) if stats['speedup_ratios'] else 1.0
        
        return current_ids, stats


def create_speculative_decoder(
    target_model: nn.Module,
    vocab_size: int,
    dim: int,
    quickmerge_dim: int,
    k_max: int,
    max_draft_len: int = 4,
    temperature: float = 1.0
) -> SpeculativeDecoder:
    """Factory function to create a speculative decoder."""
    draft_model = DraftModel(vocab_size=vocab_size, dim=dim)
    quickmerge = QuickMergePP(dim=quickmerge_dim, k_max=k_max)
    
    return SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        quickmerge=quickmerge,
        max_draft_len=max_draft_len,
        temperature=temperature
    )
