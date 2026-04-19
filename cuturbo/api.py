"""
High-level CUDA-backed TurboQuant API.

  TurboQuantMSE(d, b)   -> .quantize(x), .dequantize(code)
  TurboQuantProd(d, b)  -> .quantize(x), .dequantize(code)

Internally calls the JIT-compiled kernels in csrc/turboquant_kernels.cu.

We re-use the same random Π parameters (signs vector) and QJL projection S
across quantize/dequantize calls, since they are the quantizer's parameters
not per-sample state — matches "Global Parameters" in Algorithms 1 and 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .codebook import build_codebook
from .ext import get_ext


def _make_signs(d: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=g, dtype=torch.int32) * 2 - 1
    return bits.to(device=device, dtype=torch.float32)


def _make_qjl_matrix(d: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(d, d, generator=g).to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# MSE variant
# ---------------------------------------------------------------------------

@dataclass
class MSECode:
    packed: torch.Tensor   # (N, words_per_row) int32 — bit-packed indices


class TurboQuantMSE:
    def __init__(self, d: int, b: int, device, seed: int = 0):
        assert b in (1, 2, 4), "packed kernels require b ∈ {1, 2, 4}"
        assert d & (d - 1) == 0 and d >= 2, "d must be a power of 2"
        self.d = d
        self.b = b
        self.device = torch.device(device)
        self.ext = get_ext()
        self.signs = _make_signs(d, self.device, seed)
        self.codebook = build_codebook(b, d, self.device)

    def quantize(self, x: torch.Tensor) -> MSECode:
        assert x.dim() == 2 and x.shape[1] == self.d
        x = x.to(device=self.device, dtype=torch.float32).contiguous()
        y = self.ext.fwht_forward(x, self.signs)
        packed = self.ext.quantize_pack(y, self.codebook, int(self.b))
        return MSECode(packed=packed)

    def dequantize(self, code: MSECode) -> torch.Tensor:
        y_hat = self.ext.unpack_dequantize(
            code.packed, self.codebook, int(self.b), int(self.d)
        )
        return self.ext.fwht_inverse(y_hat, self.signs)

    def payload_bytes(self, n_vectors: int) -> int:
        """Total storage per-vector·per-coord after packing (ignoring seed etc.)."""
        idx_per_word = 32 // self.b
        words_per_row = (self.d + idx_per_word - 1) // idx_per_word
        return n_vectors * words_per_row * 4


# ---------------------------------------------------------------------------
# Inner-product variant (two-stage: MSE at b-1 bits + 1-bit QJL residual)
# ---------------------------------------------------------------------------

@dataclass
class ProdCode:
    mse_packed: torch.Tensor    # (N, words_per_row) int32 — MSE stage indices
    qjl_packed: torch.Tensor    # (N, d/32)          int32 — packed QJL sign bits
    r_norm:     torch.Tensor    # (N,)               float32


class TurboQuantProd:
    def __init__(self, d: int, b: int, device, seed: int = 0):
        assert b >= 2, "prod variant needs at least 2 total bits (b-1 for MSE stage, 1 for QJL)"
        assert (b - 1) in (1, 2, 4), "b-1 must be in {1, 2, 4} for packed MSE kernel"
        self.d = d
        self.b = b
        self.device = torch.device(device)
        self.ext = get_ext()
        self.mse = TurboQuantMSE(d, b - 1, device, seed=seed)
        self.S = _make_qjl_matrix(d, self.device, seed=seed + 97)

    def quantize(self, x: torch.Tensor) -> ProdCode:
        x = x.to(device=self.device, dtype=torch.float32).contiguous()
        mse_code = self.mse.quantize(x)
        x_mse = self.mse.dequantize(mse_code)
        r = x - x_mse
        r_norm = r.norm(dim=1)
        # S · r: use cuBLAS via torch (S is (d,d), r is (N,d)); produces (N,d)
        projection = r @ self.S.T
        qjl_packed = self.ext.pack_signs(projection.contiguous())
        return ProdCode(mse_packed=mse_code.packed, qjl_packed=qjl_packed, r_norm=r_norm)

    def dequantize(self, code: ProdCode) -> torch.Tensor:
        x_mse = self.mse.dequantize(MSECode(packed=code.mse_packed))
        qjl_signs = self.ext.unpack_signs(code.qjl_packed, int(self.d))  # (N, d) ±1
        qjl_term = qjl_signs @ self.S
        scale = math.sqrt(math.pi / 2.0) / self.d
        return x_mse + scale * code.r_norm.unsqueeze(1) * qjl_term

    def payload_bytes(self, n_vectors: int) -> int:
        mse_bytes = self.mse.payload_bytes(n_vectors)
        qjl_bytes = n_vectors * ((self.d + 31) // 32) * 4
        norm_bytes = n_vectors * 4  # fp32 norm
        return mse_bytes + qjl_bytes + norm_bytes


# ---------------------------------------------------------------------------
# Functional shortcuts
# ---------------------------------------------------------------------------

def quantize_mse(x: torch.Tensor, b: int, seed: int = 0) -> tuple:
    tq = TurboQuantMSE(x.shape[1], b, x.device, seed=seed)
    return tq, tq.quantize(x)


def dequantize_mse(tq: TurboQuantMSE, code: MSECode) -> torch.Tensor:
    return tq.dequantize(code)


def quantize_prod(x: torch.Tensor, b: int, seed: int = 0) -> tuple:
    tq = TurboQuantProd(x.shape[1], b, x.device, seed=seed)
    return tq, tq.quantize(x)


def dequantize_prod(tq: TurboQuantProd, code: ProdCode) -> torch.Tensor:
    return tq.dequantize(code)
