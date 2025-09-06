import time
import torch
from quickmergepp.quickmerge.kmeans_cuda import _load_extension


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print({"device": device})
    n, d, k = 8192, 64, 64
    x = torch.randn(n, d, device=device)
    w = torch.rand(n, device=device)
    if device == 'cuda':
        ext = _load_extension()
        idx = torch.randperm(n, device=device)[:k]
        c = x.index_select(0, idx).contiguous()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            labels = ext.assign(x, c)
            c = ext.update(x, w, labels, k)
        torch.cuda.synchronize()
        print({"cuda_ms": (time.time() - t0) * 1000})
    else:
        print("CUDA not available; skipping")


if __name__ == '__main__':
    main()


