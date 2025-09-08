import time
import json
import torch
from quickmergepp.quickmerge.pipeline import QuickMergePP
from quickmergepp.quickmerge.baselines import random_subsample, topk_norm, uniform_pool, kmeans_unweighted


@torch.no_grad()
def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print({"device": device})
    num_layers, batch, seq_len, dim = 4, 4, 512, 128
    k = 128
    h = torch.randn(num_layers, batch, seq_len, dim, device=device)

    # Prepare inputs
    x_last = h[-1]

    results = {}

    # QuickMerge++
    qm = QuickMergePP(dim=dim, k_max=k, temperature=0.5)
    t0 = time.time()
    merged_qm, info = qm.compress(h)
    if device == 'cuda':
        torch.cuda.synchronize()
    qm_ms = (time.time() - t0) * 1000
    results['quickmergepp_ms'] = qm_ms

    # Random
    t0 = time.time()
    out_rand = random_subsample(x_last, k)
    if device == 'cuda': torch.cuda.synchronize()
    results['random_ms'] = (time.time() - t0) * 1000

    # Top-k norm
    t0 = time.time()
    out_topk = topk_norm(x_last, k)
    if device == 'cuda': torch.cuda.synchronize()
    results['topk_norm_ms'] = (time.time() - t0) * 1000

    # Uniform pool
    t0 = time.time()
    out_uni = uniform_pool(x_last, k)
    if device == 'cuda': torch.cuda.synchronize()
    results['uniform_pool_ms'] = (time.time() - t0) * 1000

    # KMeans unweighted (CPU baseline)
    x_cpu = x_last.detach().cpu()
    t0 = time.time()
    out_km, _ = kmeans_unweighted(x_cpu, k)
    results['kmeans_cpu_ms'] = (time.time() - t0) * 1000

    # Report shapes and simple fidelity metric: retained norm ratio vs average token
    results['shapes'] = {
        'quickmergepp': tuple(merged_qm.shape),
        'random': tuple(out_rand.shape),
        'topk': tuple(out_topk.shape),
        'uniform': tuple(out_uni.shape),
        'kmeans_cpu': tuple(out_km.shape),
    }

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()


