"""
Pure-PyTorch reference implementations of TurboQuant_mse and TurboQuant_prod.

Used (a) as ground truth for correctness-checking the CUDA kernels, and
(b) as a pedagogical companion that mirrors Algorithms 1 and 2 in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from .codebook import build_codebook


def hadamard_matrix(d: int, device, dtype=torch.float32) -> torch.Tensor:
    """Natural-order (Sylvester) Walsh-Hadamard matrix of size d×d."""
    assert d & (d - 1) == 0 and d >= 1, "d must be a power of 2"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def random_rotation(d: int, device, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (signs, hadamard_matrix) parametrising Π = (1/√d) H diag(signs)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int32) * 2 - 1).to(
        device=device, dtype=torch.float32
    )
    H = hadamard_matrix(d, device)
    return signs, H


def rotate_forward(x: torch.Tensor, signs: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Compute Π·x where Π = (1/√d) H diag(signs). x is (N, d)."""
    d = x.shape[1]
    scale = 1.0 / math.sqrt(d)
    return (x * signs) @ H.T * scale   # Hᵀ = H since H is symmetric


def rotate_inverse(y: torch.Tensor, signs: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Compute Πᵀ·y = diag(signs) · (1/√d) · H · y."""
    d = y.shape[1]
    scale = 1.0 / math.sqrt(d)
    return (y @ H.T) * scale * signs


# ---------------------------------------------------------------------------
# TurboQuant_mse (reference)
# ---------------------------------------------------------------------------

@dataclass
class MSECode:
    indices: torch.Tensor  # (N, d) int64
    # Caller also needs (signs, H, codebook); we pass these around separately
    # so Compare / Benchmark harness can share them with the CUDA path.


def quantize_mse_ref(x: torch.Tensor, b: int, signs: torch.Tensor,
                     H: torch.Tensor, codebook: torch.Tensor) -> MSECode:
    """Reference TurboQuant_mse. Returns per-coord indices (not bit-packed)."""
    y = rotate_forward(x, signs, H)        # (N, d), near-Gaussian coords
    # Nearest centroid per coordinate. (N, d, K) distance tensor — fine for d=128, N≤65k
    # Compute without a giant (N, d, K) to save memory: loop over K.
    N, d = y.shape
    K = codebook.numel()
    best = torch.full((N, d), float("inf"), device=y.device)
    idx = torch.zeros((N, d), dtype=torch.int64, device=y.device)
    for k in range(K):
        dist = (y - codebook[k]).abs()
        mask = dist < best
        best = torch.where(mask, dist, best)
        idx = torch.where(mask, torch.full_like(idx, k), idx)
    return MSECode(indices=idx)


def dequantize_mse_ref(code: MSECode, signs: torch.Tensor, H: torch.Tensor,
                       codebook: torch.Tensor) -> torch.Tensor:
    """Reference dequantization: centroid lookup, then Πᵀ."""
    y_hat = codebook[code.indices]         # (N, d)
    x_hat = rotate_inverse(y_hat, signs, H)
    return x_hat


# ---------------------------------------------------------------------------
# TurboQuant_prod (reference)
# ---------------------------------------------------------------------------

@dataclass
class ProdCode:
    mse_indices: torch.Tensor   # (N, d) int64    — from the b-1 bit MSE quantizer
    qjl_signs: torch.Tensor     # (N, d) ±1 float — sign(S · r)
    r_norm: torch.Tensor        # (N,)   float    — ‖r‖₂


def random_qjl_matrix(d: int, device, seed: int = 1) -> torch.Tensor:
    """Dense i.i.d. N(0,1) matrix used for the 1-bit QJL projection."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(d, d, generator=g).to(device=device, dtype=torch.float32)


def quantize_prod_ref(x: torch.Tensor, b: int, signs: torch.Tensor, H: torch.Tensor,
                      codebook_bm1: torch.Tensor, S: torch.Tensor) -> ProdCode:
    """TurboQuant_prod at total bit-width b. Uses b-1 bits for MSE stage."""
    mse_code = quantize_mse_ref(x, b - 1, signs, H, codebook_bm1)
    x_mse = dequantize_mse_ref(mse_code, signs, H, codebook_bm1)
    r = x - x_mse
    r_norm = r.norm(dim=1)
    # QJL: sign(S · r) — (d,d) @ (N,d)ᵀ gives (d,N); transpose back.
    projection = r @ S.T
    qjl = torch.where(projection >= 0, 1.0, -1.0).to(dtype=torch.float32)
    return ProdCode(mse_indices=mse_code.indices, qjl_signs=qjl, r_norm=r_norm)


def dequantize_prod_ref(code: ProdCode, b: int, signs: torch.Tensor, H: torch.Tensor,
                        codebook_bm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Reconstruct xhat = xhat_mse + (√(π/2)/d) · ‖r‖ · Sᵀ · qjl."""
    d = H.shape[0]
    x_mse = dequantize_mse_ref(MSECode(indices=code.mse_indices), signs, H, codebook_bm1)
    # S: (d,d), qjl: (N,d). Want (N,d): qjl @ S  (since Sᵀ · qjl_row = S.T @ q, i.e. q @ S)
    qjl_term = code.qjl_signs @ S
    scale = math.sqrt(math.pi / 2.0) / d
    x_qjl = scale * code.r_norm.unsqueeze(1) * qjl_term
    return x_mse + x_qjl


# ---------------------------------------------------------------------------
# Convenience wrappers (used by tests + benchmark for quick sanity checks)
# ---------------------------------------------------------------------------

def mse_end_to_end_ref(x: torch.Tensor, b: int, seed: int = 0) -> torch.Tensor:
    d = x.shape[1]
    signs, H = random_rotation(d, x.device, seed=seed)
    codebook = build_codebook(b, d, x.device)
    code = quantize_mse_ref(x, b, signs, H, codebook)
    return dequantize_mse_ref(code, signs, H, codebook)


def prod_end_to_end_ref(x: torch.Tensor, b: int, seed: int = 0) -> torch.Tensor:
    d = x.shape[1]
    signs, H = random_rotation(d, x.device, seed=seed)
    codebook_bm1 = build_codebook(b - 1, d, x.device)
    S = random_qjl_matrix(d, x.device, seed=seed + 97)
    code = quantize_prod_ref(x, b, signs, H, codebook_bm1, S)
    return dequantize_prod_ref(code, b, signs, H, codebook_bm1, S)
