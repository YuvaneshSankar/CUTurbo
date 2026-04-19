"""
Lloyd-Max quantization codebooks for TurboQuant.

After the random rotation, each coord of Πx concentrates to N(0, 1/d) (paper
Lemma 1). The optimal scalar quantizer for Gaussian inputs has known centroid
sets (Max 1960, Paez-Glisson 1972). We store unit-variance Lloyd-Max centroids
and scale by 1/√d at use time.

The b=1 and b=2 rows match the paper's reported values in Section 3.1:
  b=1: ±√(2/π) ≈ ±0.7979
  b=2: ±0.4528, ±1.510
"""

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

# Unit-variance Gaussian Lloyd-Max centroids. Sorted ascending.
_LLOYD_MAX_GAUSSIAN = {
    1: [-0.7978845608, 0.7978845608],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    4: [
        -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9423,  1.2562,  1.6180,  2.0690,  2.7326,
    ],
}


def paper_codebook(b: int, d: int) -> np.ndarray:
    """Lloyd-Max centroids scaled for N(0, 1/d) distribution. Returns shape (2^b,)."""
    if b not in _LLOYD_MAX_GAUSSIAN:
        raise ValueError(f"bit-width {b} not supported; must be one of {list(_LLOYD_MAX_GAUSSIAN)}")
    centroids = np.asarray(_LLOYD_MAX_GAUSSIAN[b], dtype=np.float32)
    return centroids / math.sqrt(d)


def lloyd_refine(centroids: np.ndarray, n_samples: int = 2_000_000,
                 n_iter: int = 40, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """Refine Gaussian Lloyd-Max centroids via empirical Lloyd's algorithm.

    Used as a sanity check — should converge back to the hardcoded table.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, sigma, size=n_samples).astype(np.float32)
    c = np.sort(np.asarray(centroids, dtype=np.float32).copy())
    for _ in range(n_iter):
        # assign to nearest centroid
        idx = np.argmin(np.abs(samples[:, None] - c[None, :]), axis=1)
        new_c = np.zeros_like(c)
        for k in range(len(c)):
            m = idx == k
            if m.any():
                new_c[k] = samples[m].mean()
            else:
                new_c[k] = c[k]
        c = np.sort(new_c)
    return c


@lru_cache(maxsize=32)
def _codebook_torch(b: int, d: int, device_str: str) -> torch.Tensor:
    cb = paper_codebook(b, d)
    return torch.from_numpy(cb).to(torch.device(device_str))


def build_codebook(b: int, d: int, device) -> torch.Tensor:
    """Return a cached fp32 codebook tensor on `device` of shape (2^b,)."""
    return _codebook_torch(int(b), int(d), str(device))


if __name__ == "__main__":
    # Sanity check: print paper Table in Section 3.1
    for b in (1, 2, 3, 4):
        cb = paper_codebook(b, d=128)
        print(f"b={b}  centroids (× √d):", [f"{c*math.sqrt(128):+.4f}" for c in cb])
