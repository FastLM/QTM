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
```

Modules
- Saliency: multi-scale attention entropy
- Merge: Gumbel-Softmax selection + weighted k-means
- AR prior: small causal transformer with bi-directional MSE losses for training
