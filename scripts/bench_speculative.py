import time
import torch
import torch.nn as nn
from quickmergepp.quickmerge.speculative import create_speculative_decoder, SpeculativeDecoder


class MockTargetModel(nn.Module):
    """Mock target model for benchmarking."""
    
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim*4,
            dropout=0.0, batch_first=True
        )
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.transformer(x, x)
        return self.head(x)
    
    def get_last_hidden_states(self) -> torch.Tensor:
        # Mock hidden states for QuickMerge++
        batch_size = 2
        seq_len = 128
        dim = 64
        num_layers = 4
        return torch.randn(num_layers, batch_size, seq_len, dim, device=next(self.parameters()).device)


@torch.no_grad()
def benchmark_speculative_decoding():
    """Benchmark speculative decoding vs standard generation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # Model parameters
    vocab_size = 1000
    dim = 128
    seq_len = 64
    max_new_tokens = 20
    
    # Create models
    target_model = MockTargetModel(vocab_size, dim).to(device)
    
    # Create speculative decoder
    spec_decoder = create_speculative_decoder(
        target_model=target_model,
        vocab_size=vocab_size,
        dim=dim,
        quickmerge_dim=64,
        k_max=32,
        max_draft_len=4,
        temperature=0.8
    )
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (2, seq_len), device=device)
    hidden_states = target_model.get_last_hidden_states()
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Benchmark speculative decoding
    print("\n=== Speculative Decoding ===")
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    generated_ids, stats = spec_decoder.generate(
        input_ids=input_ids,
        hidden_states=hidden_states,
        max_new_tokens=max_new_tokens,
        verbose=True
    )
    
    torch.cuda.synchronize() if device == 'cuda' else None
    spec_time = time.time() - start_time
    
    print(f"Speculative generation time: {spec_time:.3f}s")
    print(f"Generated sequence length: {generated_ids.shape[1]}")
    print(f"Stats: {stats}")
    
    # Benchmark standard generation (baseline)
    print("\n=== Standard Generation (Baseline) ===")
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    current_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = target_model(current_ids)
        next_token = torch.multinomial(
            torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1
        )
        current_ids = torch.cat([current_ids, next_token], dim=1)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    standard_time = time.time() - start_time
    
    print(f"Standard generation time: {standard_time:.3f}s")
    print(f"Generated sequence length: {current_ids.shape[1]}")
    
    # Compute speedup
    speedup = standard_time / spec_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Average speedup ratio: {stats['avg_speedup']:.2f}")


if __name__ == "__main__":
    benchmark_speculative_decoding()
