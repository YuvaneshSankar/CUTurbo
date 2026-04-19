"""
End-to-end retrieval benchmark on SIFT-1M.

Tests TurboQuant on its natural use case: store a compressed corpus of 1 M
vectors, retrieve top-10 nearest neighbours by inner product, score against
fp32 brute-force ground truth. Reports Recall@10, index size, QPS, and build
time for every method.

Run: python3 benchmark/sift_retrieval.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.datasets import load_sift1m                          # noqa: E402
from benchmark.harness import detect_env, dump_json                 # noqa: E402
from benchmark import plots                                         # noqa: E402

from cuturbo.retrieval import (                                     # noqa: E402
    FP32Index, FP16Index, NaiveScalarIndex,
    TurboQuantMSEIndex, TurboQuantProdIndex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalise_inplace(x: torch.Tensor) -> None:
    x /= x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_search(index, queries: torch.Tensor, k: int,
                 warmup_batches: int, timed_batches: int,
                 batch_size: int) -> dict:
    """Run queries; return per-batch timings + aggregate QPS."""
    Q = queries.shape[0]
    # Warm-up
    for wb in range(warmup_batches):
        s = (wb * batch_size) % Q
        index.search(queries[s:s + batch_size], k=k)
    _cuda_sync()

    # Timed
    start_events, stop_events = [], []
    for _ in range(timed_batches):
        start_events.append(torch.cuda.Event(enable_timing=True))
        stop_events.append(torch.cuda.Event(enable_timing=True))

    for i in range(timed_batches):
        s = (i * batch_size) % Q
        start_events[i].record()
        index.search(queries[s:s + batch_size], k=k)
        stop_events[i].record()
    _cuda_sync()

    per_batch_ms = np.array([start_events[i].elapsed_time(stop_events[i])
                              for i in range(timed_batches)], dtype=np.float64)
    per_query_us = per_batch_ms * 1000.0 / batch_size
    total_s = per_batch_ms.sum() / 1000.0
    qps = (timed_batches * batch_size) / total_s if total_s > 0 else 0.0
    return {
        "per_batch_ms_median": float(np.median(per_batch_ms)),
        "per_batch_ms_p05":    float(np.percentile(per_batch_ms, 5)),
        "per_batch_ms_p95":    float(np.percentile(per_batch_ms, 95)),
        "per_query_us_median": float(np.median(per_query_us)),
        "qps":                 float(qps),
        "batch_size":          int(batch_size),
        "timed_batches":       int(timed_batches),
    }


def _recall_at_k(pred: torch.Tensor, gt: torch.Tensor, k: int) -> float:
    """Fraction of the true top-k that the prediction returns, averaged over queries."""
    pred = pred[:, :k]
    gt_k = gt[:, :k]
    matches = (pred.unsqueeze(-1) == gt_k.unsqueeze(1)).any(dim=-1)  # (Q, k)
    return matches.float().mean().item()


def _full_search(index, queries: torch.Tensor, k: int, batch_size: int) -> torch.Tensor:
    """Run the full query set and return (Q, k) integer top-idx."""
    _, idx = index.search(queries, k=k)
    return idx


def _fp32_rerank(pred_top: torch.Tensor, queries: torch.Tensor,
                 base_gpu: torch.Tensor, batch_size: int = 100) -> torch.Tensor:
    """Two-stage retrieval: re-score the top-k candidate pool with fp32 inner products.

    pred_top: (Q, k) long, candidate doc indices from a quantized index.
    Returns: (Q, k) long, same candidates re-ordered by fp32 IP.
    """
    Q, k = pred_top.shape
    out = torch.empty_like(pred_top)
    for s in range(0, Q, batch_size):
        e = min(s + batch_size, Q)
        q = queries[s:e]                                    # (qb, d)
        cand = pred_top[s:e]                                # (qb, k)
        cand_vecs = base_gpu[cand.flatten()].view(e - s, k, -1)  # (qb, k, d)
        scores = (q.unsqueeze(1) * cand_vecs).sum(dim=-1)   # (qb, k)
        order = scores.argsort(dim=1, descending=True)      # (qb, k)
        out[s:e] = torch.gather(cand, 1, order)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SIFT-1M end-to-end retrieval benchmark")
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/sift1m"))
    ap.add_argument("--out-dir",   type=Path, default=Path("results/sift"))
    ap.add_argument("--k", type=int, default=100,
                    help="Candidate pool size; Recall@{1, 10, 100} computed from it")
    ap.add_argument("--query-batch", type=int, default=100)
    ap.add_argument("--doc-chunk",   type=int, default=100_000)
    ap.add_argument("--warmup-batches", type=int, default=3)
    ap.add_argument("--timed-batches",  type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    env = detect_env()
    dump_json(args.out_dir / "env.json", env)
    print(f"=== SIFT-1M retrieval on {env.gpu_name} ({env.gpu_vram_mib} MiB) ===")

    # -----------------------------------------------------------------
    # Phase 1 — load + L2-normalise
    # -----------------------------------------------------------------
    print("\n[1/4] loading SIFT-1M ...")
    base, query, _sift_gt = load_sift1m(args.cache_dir)
    base = base.float().contiguous()
    query = query.float().contiguous()
    _l2_normalise_inplace(base)
    _l2_normalise_inplace(query)
    N, d = base.shape
    Q = query.shape[0]
    assert N == 1_000_000 and d == 128 and Q == 10_000
    print(f"  base={tuple(base.shape)}  query={tuple(query.shape)}")

    # -----------------------------------------------------------------
    # Phase 2 — ground truth via fp32 brute force on normalised corpus
    # -----------------------------------------------------------------
    print("\n[2/4] computing ground-truth top-10 via fp32 brute force ...")
    gt_cache = args.cache_dir / "gt_ip_top100.pt"
    if gt_cache.exists():
        print(f"  using cached ground truth at {gt_cache}")
        gt_top100 = torch.load(gt_cache)
    else:
        fp32_idx = FP32Index(device, doc_chunk=args.doc_chunk, query_batch=args.query_batch)
        fp32_idx.build(base)
        queries_gpu = query.to(device)
        _, gt_top100 = fp32_idx.search(queries_gpu, k=100)
        gt_top100 = gt_top100.cpu()
        torch.save(gt_top100, gt_cache)
        del fp32_idx, queries_gpu
        gc.collect(); torch.cuda.empty_cache()
    assert gt_top100.shape == (Q, 100)

    # -----------------------------------------------------------------
    # Phase 3 — build + evaluate each method
    # -----------------------------------------------------------------
    methods = [
        ("fp32",          lambda: FP32Index(device, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("fp16",          lambda: FP16Index(device, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("naive-b2",      lambda: NaiveScalarIndex(device, b=2, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("naive-b4",      lambda: NaiveScalarIndex(device, b=4, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("mse-b2",        lambda: TurboQuantMSEIndex(device, b=2, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("mse-b4",        lambda: TurboQuantMSEIndex(device, b=4, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("prod-b2",       lambda: TurboQuantProdIndex(device, b=2, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("prod-b3",       lambda: TurboQuantProdIndex(device, b=3, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
        ("prod-b5",       lambda: TurboQuantProdIndex(device, b=5, doc_chunk=args.doc_chunk, query_batch=args.query_batch)),
    ]

    queries_gpu = query.to(device)
    gt_top100_gpu = gt_top100.to(device)
    base_gpu_fp32 = base.to(device)  # kept resident for the fp32 rerank pass

    results = []
    for name, ctor in methods:
        print(f"\n[3/4] ===== {name} =====")
        gc.collect(); torch.cuda.empty_cache()

        t0 = time.perf_counter()
        idx = ctor()
        idx.build(base_gpu_fp32)          # GPU input; FP32Index shares storage
        _cuda_sync()
        build_s = time.perf_counter() - t0

        bytes_ = idx.index_bytes()
        print(f"  build time   {build_s:6.2f} s")
        print(f"  index size   {bytes_ / 2**20:7.2f} MiB  ({bytes_ / N:.2f} B/vec)")

        # Recall across several k on the full query set
        t0 = time.perf_counter()
        pred_top = _full_search(idx, queries_gpu, k=args.k, batch_size=args.query_batch)
        _cuda_sync()
        full_scan_s = time.perf_counter() - t0
        r_at_1   = _recall_at_k(pred_top, gt_top100_gpu, 1)
        r_at_10  = _recall_at_k(pred_top, gt_top100_gpu, 10)
        r_at_100 = _recall_at_k(pred_top, gt_top100_gpu, args.k)
        # Two-stage: pull the top-100 candidate pool, re-rank with fp32 IP,
        # then measure Recall@10 of the reranked list. This is the production
        # deployment pattern for quantized ANN.
        rerank_top = _fp32_rerank(pred_top, queries_gpu, base_gpu_fp32,
                                  batch_size=args.query_batch)
        r_rerank_at_10 = _recall_at_k(rerank_top, gt_top100_gpu, 10)
        print(f"  Recall@1        {r_at_1:.4f}")
        print(f"  Recall@10       {r_at_10:.4f}")
        print(f"  Recall@100      {r_at_100:.4f}")
        print(f"  Recall@10 (2-stage fp32 rerank)  {r_rerank_at_10:.4f}")
        print(f"  full-scan       {full_scan_s:6.2f} s  ({Q / full_scan_s:.0f} QPS wall-clock)")

        # Precise QPS via CUDA events on a repeated batch
        timing = _time_search(idx, queries_gpu, args.k,
                              warmup_batches=args.warmup_batches,
                              timed_batches=args.timed_batches,
                              batch_size=args.query_batch)
        print(f"  QPS (events) {timing['qps']:.0f}  "
              f"(per-query {timing['per_query_us_median']:.2f} µs)")

        results.append({
            "method": name,
            "recall_at_1":   r_at_1,
            "recall_at_10":  r_at_10,
            "recall_at_100": r_at_100,
            "recall_at_10_rerank": r_rerank_at_10,
            "index_bytes":   bytes_,
            "index_mb":      bytes_ / 2**20,
            "bytes_per_vec": bytes_ / N,
            "build_s":       build_s,
            "full_scan_s":   full_scan_s,
            "qps_wall":      Q / full_scan_s,
            **timing,
        })

        del idx
        gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Phase 4 — write artefacts
    # -----------------------------------------------------------------
    print("\n[4/4] writing results ...")
    dump_json(args.out_dir / "raw" / "retrieval.json", results)

    with open(args.out_dir / "summary.csv", "w") as f:
        f.write("method,recall_at_1,recall_at_10,recall_at_100,recall_at_10_rerank,"
                "index_mb,bytes_per_vec,build_s,qps,per_query_us\n")
        for r in results:
            f.write(f"{r['method']},{r['recall_at_1']:.4f},{r['recall_at_10']:.4f},"
                    f"{r['recall_at_100']:.4f},{r['recall_at_10_rerank']:.4f},"
                    f"{r['index_mb']:.2f},"
                    f"{r['bytes_per_vec']:.1f},{r['build_s']:.2f},{r['qps']:.0f},"
                    f"{r['per_query_us_median']:.2f}\n")

    # Plots
    plots.plot_recall_vs_size       (results, str(args.out_dir / "fig_sift_recall_vs_size.png"))
    plots.plot_recall_vs_qps        (results, str(args.out_dir / "fig_sift_recall_vs_qps.png"))
    plots.plot_recall_rerank_vs_size(results, str(args.out_dir / "fig_sift_rerank_recall.png"))

    prod_vs_mse_rows = []
    for r in results:
        if r["method"].startswith("mse-b"):
            prod_vs_mse_rows.append({"variant": "mse",  "b": int(r["method"].split("b")[-1]),
                                     "recall_at_10": r["recall_at_10"]})
        elif r["method"].startswith("prod-b"):
            prod_vs_mse_rows.append({"variant": "prod", "b": int(r["method"].split("b")[-1]),
                                     "recall_at_10": r["recall_at_10"]})
    plots.plot_prod_vs_mse_recall(prod_vs_mse_rows,
                                   str(args.out_dir / "fig_sift_prod_vs_mse.png"))

    print("\nDONE.")
    print(f"  summary → {args.out_dir / 'summary.csv'}")
    print(f"  plots   → {args.out_dir}/fig_sift_*.png")


if __name__ == "__main__":
    main()
