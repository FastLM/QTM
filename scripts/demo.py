import torch
from quickmergepp.quickmerge.pipeline import QuickMergePP


def main():
    torch.manual_seed(0)
    num_layers = 4
    batch = 2
    seq_len = 128
    dim = 64
    # Dummy hidden states per layer
    h = torch.randn(num_layers, batch, seq_len, dim)

    qm = QuickMergePP(dim=dim, k_max=54, temperature=0.5)
    merged, info = qm.compress(h)
    preds = qm.ar_predict(merged)

    print({
        "merged_shape": tuple(merged.shape),
        "preds_shape": tuple(preds.shape),
        "saliency_shape": tuple(info["saliency"].shape),
    })


if __name__ == "__main__":
    main()


