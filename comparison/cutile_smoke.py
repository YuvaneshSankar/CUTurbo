"""Stage 2 — smoke-test his cuTile compress kernel on our GPU.

Invokes TurboQuantEngine.launch_compress_keys() directly on a small synthetic
input, prints output shapes + first few indices. If this prints output, stage 3
(the 4-way benchmark) is viable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HIS = Path(__file__).resolve().parent / "turboquant_cutile"
sys.path.insert(0, str(_HIS))

from turboquant_cutile import TurboQuantEngine


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"device: {torch.cuda.get_device_name(0)}")

    engine = TurboQuantEngine(head_dim=128, total_bits=3, seed=42, device="cuda")
    print(f"engine: total_bits={engine.total_bits}, mse_bits={engine.mse_bits}")

    N = 128
    torch.manual_seed(0)
    K = torch.randn(N, 128, device="cuda", dtype=torch.float16)

    result = engine.launch_compress_keys(K)
    print("compress outputs:")
    for k, v in result.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    print("first 4 rows of indices (cols 0..7):")
    print(result["indices"][:4, :8].cpu().numpy())
    print("first 4 QJL sign rows (cols 0..7):")
    print(result["qjl_signs"][:4, :8].cpu().numpy())
    print("first 4 norms:", result["vec_norms"][:4].cpu().numpy())

    # Sanity: reconstruct x_hat and compute D_mse
    centroids = engine.key_codebook.centroids.to("cuda")
    y_hat = centroids[result["indices"].long()]
    k_mse = (y_hat.float() @ engine.Pi.float()) * result["vec_norms"].float().unsqueeze(-1)
    # Full prod estimator: add residual correction
    qjl_term = result["qjl_signs"].float() @ engine.S.float()
    x_hat = k_mse + engine.correction_scale * result["residual_norms"].float().unsqueeze(-1) * qjl_term

    err_sq = (K.float() - x_hat).pow(2).sum(dim=1).mean().item()
    d_mse = err_sq / 128
    print(f"reconstruction D_mse: {d_mse:.4f} "
          f"(paper band [0.0625, 0.170] for b_mse=2)")


if __name__ == "__main__":
    main()
