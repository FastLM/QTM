QuickMerge++: Fast Token Merging with Autoregressive Prior

Install
```bash
pip install -r requirements.txt
pip install -e .
```

Usage
```python
import torch
from quickmergepp import QuickMergePP

num_layers, batch, seq_len, dim = 4, 2, 128, 64
h = torch.randn(num_layers, batch, seq_len, dim)

qm = QuickMergePP(dim=dim, k_max=54, temperature=0.5)
merged, info = qm.compress(h)
preds = qm.ar_predict(merged)

print(merged.shape, preds.shape)
```

Demo
```bash
python scripts/demo.py
```

Benchmarks
```bash
# Method-level throughput/latency on synthetic inputs
python scripts/bench_methods.py

# CUDA k-means micro-benchmark (requires GPU)
python scripts/bench_kmeans.py

# Speculative decoding vs standard generation
python scripts/bench_speculative.py
```

Speculative Decoding
```python
import torch
from quickmergepp import create_speculative_decoder

# Create speculative decoder with QuickMerge++ compression
spec_decoder = create_speculative_decoder(
    target_model=your_model,
    vocab_size=1000,
    dim=128,
    quickmerge_dim=64,
    k_max=32,
    max_draft_len=4,
    temperature=0.8
)

# Generate with speculative decoding
generated_ids, stats = spec_decoder.generate(
    input_ids=input_tokens,
    hidden_states=encoder_hidden_states,
    max_new_tokens=50
)

print(f"Speedup: {stats['avg_speedup']:.2f}x")
print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
```

Multimodal Applications
```python
from quickmergepp import create_multimodal_pipeline, create_diffusion_pipeline, create_llm_pipeline

# Vision-Language
multimodal = create_multimodal_pipeline(vision_dim=768, text_dim=512, k_max=64)
fused_output, info = multimodal(vision_tokens, text_tokens)

# Diffusion Models
diffusion = create_diffusion_pipeline(unet_dim=768, text_dim=512, k_max=32)
compressed_text, _ = diffusion.compress_text_embeddings(text_embeddings)

# LLM Optimizations
llm = create_llm_pipeline(model_dim=4096, k_max=128)
compressed_k, compressed_v = llm.compress_kv_cache(key_cache, value_cache)
```

Examples
```bash
# Run comprehensive multimodal examples
python scripts/examples_multimodal.py
```

Modules
- Saliency: multi-scale attention entropy
- Merge: Gumbel-Softmax selection + weighted k-means
- AR prior: small causal transformer with bi-directional MSE losses for training
