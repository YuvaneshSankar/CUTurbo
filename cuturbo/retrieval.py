"""
Quantized nearest-neighbour indexes built on the TurboQuant kernels.

All indexes implement the same interface:

    idx = FooIndex(device=...)
    idx.build(base)                 # base: (N, d) fp32 cpu/gpu, unit-norm
    top_val, top_idx = idx.search(queries, k=10)   # batched IP top-k

Retrieval uses the standard "quantized-index + chunked re-score" pattern:

  for query_batch in queries:                # (Q, d)
      for doc_chunk in base_chunks:          # (C, d)
          docs = dequant(doc_chunk)
          scores = query_batch @ docs.T      # (Q, C)
          merge_topk(global_top, scores)

For TurboQuant_prod the per-pair score adds the residual-correction term
    scale * ||r_i|| * <qjl_signs_i, S @ y>    with  scale = sqrt(pi/2) / d
which we compute chunk-wise without materialising the full (N, d) reconstruction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from .api import TurboQuantMSE, TurboQuantProd, MSECode, ProdCode
from .ext import get_ext


DEFAULT_DOC_CHUNK = 100_000
DEFAULT_QUERY_BATCH = 100


# ---------------------------------------------------------------------------
# Top-k merge helper
# ---------------------------------------------------------------------------

def _merge_topk(
    top_val: torch.Tensor, top_idx: torch.Tensor,
    new_val: torch.Tensor, new_idx: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two (Q, k) candidate lists, keeping the k highest-scoring."""
    cat_val = torch.cat([top_val, new_val], dim=1)
    cat_idx = torch.cat([top_idx, new_idx], dim=1)
    sel_val, sel_pos = torch.topk(cat_val, k=k, dim=1)
    sel_idx = torch.gather(cat_idx, 1, sel_pos)
    return sel_val, sel_idx


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QuantizedIndex:
    name: str = "abstract"

    def __init__(self, device, doc_chunk: int = DEFAULT_DOC_CHUNK, query_batch: int = DEFAULT_QUERY_BATCH):
        self.device = torch.device(device)
        self.doc_chunk = doc_chunk
        self.query_batch = query_batch
        self.N = 0
        self.d = 0

    def build(self, base: torch.Tensor) -> None:
        raise NotImplementedError

    def index_bytes(self) -> int:
        raise NotImplementedError

    # Per-chunk dequant used by the default search() below.
    # Returns (chunk_size, d) fp32 tensor on self.device.
    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        raise NotImplementedError

    # Some indexes (prod) need access to queries when scoring a chunk.
    # Default: pure dequant-then-GEMM (mse/fp/naive).
    def _score_chunk(self, queries: torch.Tensor, start: int, end: int) -> torch.Tensor:
        docs = self._dequant_chunk(start, end)
        return queries @ docs.T

    def search(self, queries: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        assert queries.dim() == 2 and queries.shape[1] == self.d
        queries = queries.to(self.device, dtype=torch.float32, non_blocking=True)
        Q = queries.shape[0]
        top_val = torch.full((Q, k), -float("inf"), device=self.device)
        top_idx = torch.zeros((Q, k), device=self.device, dtype=torch.long)

        for q_start in range(0, Q, self.query_batch):
            q_end = min(q_start + self.query_batch, Q)
            q_batch = queries[q_start:q_end]
            for doc_start in range(0, self.N, self.doc_chunk):
                doc_end = min(doc_start + self.doc_chunk, self.N)
                scores = self._score_chunk(q_batch, doc_start, doc_end)  # (qb, C)
                chunk_k = min(k, doc_end - doc_start)
                val, pos = torch.topk(scores, k=chunk_k, dim=1)
                idx = pos + doc_start
                if chunk_k < k:
                    pad = k - chunk_k
                    val = torch.cat([val, torch.full((q_end - q_start, pad),
                                                    -float("inf"), device=self.device)], dim=1)
                    idx = torch.cat([idx, torch.zeros((q_end - q_start, pad),
                                                     device=self.device, dtype=torch.long)], dim=1)
                top_val[q_start:q_end], top_idx[q_start:q_end] = _merge_topk(
                    top_val[q_start:q_end], top_idx[q_start:q_end], val, idx, k
                )
        return top_val, top_idx


# ---------------------------------------------------------------------------
# fp32 brute force — the quality ceiling
# ---------------------------------------------------------------------------

class FP32Index(QuantizedIndex):
    name = "fp32"

    def build(self, base: torch.Tensor) -> None:
        self.N, self.d = base.shape
        if base.device == self.device and base.dtype == torch.float32:
            # share storage — caller already has an fp32 GPU tensor
            self.base = base.contiguous()
            return
        # stream the copy in chunks — avoids an fp32 pinned-host spike at N=1M
        out = torch.empty((self.N, self.d), device=self.device, dtype=torch.float32)
        for i in range(0, self.N, self.doc_chunk):
            j = min(i + self.doc_chunk, self.N)
            out[i:j] = base[i:j].to(self.device, dtype=torch.float32, non_blocking=True)
        self.base = out.contiguous()

    def index_bytes(self) -> int:
        return self.N * self.d * 4

    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        return self.base[start:end]


# ---------------------------------------------------------------------------
# fp16 — memory-only (no quantization)
# ---------------------------------------------------------------------------

class FP16Index(QuantizedIndex):
    name = "fp16"

    def build(self, base: torch.Tensor) -> None:
        self.N, self.d = base.shape
        self.base = base.to(self.device, dtype=torch.float16, non_blocking=True).contiguous()

    def index_bytes(self) -> int:
        return self.N * self.d * 2

    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        return self.base[start:end].to(torch.float32)


# ---------------------------------------------------------------------------
# Naive uniform scalar quantization — apples-to-apples competing quantizer
# ---------------------------------------------------------------------------

class NaiveScalarIndex(QuantizedIndex):
    name = "naive-scalar"

    def __init__(self, device, b: int, **kwargs):
        super().__init__(device, **kwargs)
        assert b in (2, 4), "naive scalar: b in {2, 4} for clean byte layouts"
        self.b = b
        self.K = 1 << b

    def build(self, base: torch.Tensor) -> None:
        self.N, self.d = base.shape
        self.name = f"naive-scalar-b{self.b}"

        sigma = base.float().std(dim=0).clamp_min(1e-8).to(self.device)
        self.range = 3.0 * sigma
        self.step = 2.0 * self.range / self.K
        self.centroids = (
            torch.arange(self.K, device=self.device, dtype=torch.float32).unsqueeze(0)
            * self.step.unsqueeze(1) - self.range.unsqueeze(1) + self.step.unsqueeze(1) / 2
        )  # (d, K)

        packed_chunks = []
        for i in range(0, self.N, self.doc_chunk):
            j = min(i + self.doc_chunk, self.N)
            chunk = base[i:j].to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            q = ((chunk + self.range) / self.step).clamp_(0, self.K - 1).round_().to(torch.uint8)
            if self.b == 2:
                q = q.view(q.shape[0], self.d // 4, 4)
                packed_chunks.append(
                    (q[:, :, 0]
                     | (q[:, :, 1] << 2)
                     | (q[:, :, 2] << 4)
                     | (q[:, :, 3] << 6)).contiguous()
                )
            else:  # b == 4
                q = q.view(q.shape[0], self.d // 2, 2)
                packed_chunks.append((q[:, :, 0] | (q[:, :, 1] << 4)).contiguous())
        self.packed = torch.cat(packed_chunks, dim=0).contiguous()

    def index_bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size())

    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        if self.b == 2:
            # unpack 4 indices per byte
            p = self.packed[start:end]                      # (C, d/4) uint8
            idx = torch.stack(
                [(p >> (2 * i)) & 0x3 for i in range(4)], dim=-1,
            ).view(end - start, self.d)
        else:
            p = self.packed[start:end]                      # (C, d/2)
            idx = torch.stack([p & 0xF, (p >> 4) & 0xF], dim=-1).view(end - start, self.d)
        # Gather per-coord centroids
        # centroids: (d, K), idx: (C, d) long
        gather_idx = idx.long().T.contiguous()              # (d, C)
        out = torch.gather(self.centroids, 1, gather_idx).T  # (C, d)
        return out


# ---------------------------------------------------------------------------
# TurboQuant MSE — Algorithm 1 of the paper
# ---------------------------------------------------------------------------

class TurboQuantMSEIndex(QuantizedIndex):
    def __init__(self, device, b: int, seed: int = 0, **kwargs):
        super().__init__(device, **kwargs)
        self.b = b
        self.seed = seed
        self.name = f"turboquant-mse-b{b}"

    def build(self, base: torch.Tensor) -> None:
        self.N, self.d = base.shape
        self.quantizer = TurboQuantMSE(self.d, self.b, self.device, seed=self.seed)
        packed_chunks = []
        for i in range(0, self.N, self.doc_chunk):
            j = min(i + self.doc_chunk, self.N)
            chunk = base[i:j].to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            packed_chunks.append(self.quantizer.quantize(chunk).packed)
        self.packed = torch.cat(packed_chunks, dim=0).contiguous()

    def index_bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size())

    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        chunk = MSECode(packed=self.packed[start:end].contiguous())
        return self.quantizer.dequantize(chunk)


# ---------------------------------------------------------------------------
# TurboQuant Prod — Algorithm 2 (unbiased inner product)
# ---------------------------------------------------------------------------

class TurboQuantProdIndex(QuantizedIndex):
    def __init__(self, device, b: int, seed: int = 0, **kwargs):
        super().__init__(device, **kwargs)
        self.b = b
        self.seed = seed
        self.name = f"turboquant-prod-b{b}"

    def build(self, base: torch.Tensor) -> None:
        self.N, self.d = base.shape
        self.quantizer = TurboQuantProd(self.d, self.b, self.device, seed=self.seed)
        mse_chunks, qjl_chunks, norm_chunks = [], [], []
        for i in range(0, self.N, self.doc_chunk):
            j = min(i + self.doc_chunk, self.N)
            chunk = base[i:j].to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
            code = self.quantizer.quantize(chunk)
            mse_chunks.append(code.mse_packed)
            qjl_chunks.append(code.qjl_packed)
            norm_chunks.append(code.r_norm)
        self.mse_packed = torch.cat(mse_chunks, dim=0).contiguous()
        self.qjl_packed = torch.cat(qjl_chunks, dim=0).contiguous()
        self.r_norm = torch.cat(norm_chunks, dim=0).contiguous()
        self._scale = math.sqrt(math.pi / 2.0) / self.d
        self._ext = get_ext()

    def index_bytes(self) -> int:
        return (
            int(self.mse_packed.numel() * self.mse_packed.element_size())
            + int(self.qjl_packed.numel() * self.qjl_packed.element_size())
            + int(self.r_norm.numel() * self.r_norm.element_size())
        )

    def _score_chunk(self, queries: torch.Tensor, start: int, end: int) -> torch.Tensor:
        # MSE part: <x_mse_i, y> — reuse the mse dequant + GEMM path
        mse_code_chunk = MSECode(packed=self.mse_packed[start:end].contiguous())
        x_mse_chunk = self.quantizer.mse.dequantize(mse_code_chunk)     # (C, d)
        mse_scores = queries @ x_mse_chunk.T                            # (Q, C)

        # Residual correction: scale * ||r_i|| * <qjl_signs_i, S @ y>
        #   Sy = queries @ S.T   (Q, d)
        #   <qjl_signs_i, Sy_j> = Sy @ qjl_signs.T   (Q, C)
        qjl_signs_chunk = self._ext.unpack_signs(
            self.qjl_packed[start:end].contiguous(), int(self.d)
        )                                                               # (C, d) ±1 fp32
        Sy = queries @ self.quantizer.S.T                               # (Q, d)
        qjl_ip = Sy @ qjl_signs_chunk.T                                 # (Q, C)
        correction = self._scale * qjl_ip * self.r_norm[start:end].unsqueeze(0)

        return mse_scores + correction
