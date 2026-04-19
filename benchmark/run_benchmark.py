"""
CUTurbo — rigorous end-to-end benchmark harness.

Phases:
  1. Environment dump (hardware / driver / versions) → results/env.json
  2. Correctness sanity (CUDA vs pure-PyTorch reference) across all d values
  3. Accuracy sweep with seed variance (bit-widths b ∈ {1,2,4} for mse,
     {2,3,5} for prod; ≥10 seeds per point to produce error bars)
  4. Bias-vs-IP (paper Fig 2 reproduction)
  5. Latency scaling at fixed d=128, sweeping N
  6. Latency scaling at fixed N, sweeping d
  7. Memory bandwidth utilization bar chart
  8. Compression: real packed bytes vs fp16 baseline
  9. Quality-vs-speed Pareto

Timing: 15 warmup iters + 100 measured iters per configuration, per method.
Results: results/*.png + results/raw/*.json + results/summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import pathlib
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cuturbo.api import TurboQuantMSE, TurboQuantProd
from cuturbo import reference as ref
from cuturbo.codebook import paper_codebook

from benchmark import plots, harness
from benchmark.harness import (
    TimingStats, time_cuda, detect_env,
    measure_peak_vram_bytes, to_gbps,
    FP16Baseline, UniformScalarBaseline, ip_errors, dump_json,
)


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

def make_unit_sphere(N: int, d: int, device, seed: int = 42) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(N, d, generator=g)
    x = x / x.norm(dim=1, keepdim=True)
    return x.to(device=device, dtype=torch.float32)


def make_paired_query(x: torch.Tensor, target_ip: float, seed: int) -> torch.Tensor:
    """y s.t. E[⟨x_i, y_i⟩] ≈ target_ip, y on the sphere, built via one-sample Gram–Schmidt."""
    N, d = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(N, d, generator=g).to(device=x.device, dtype=torch.float32)
    z = z - (z * x).sum(dim=1, keepdim=True) * x     # project onto x⊥
    z = z / z.norm(dim=1, keepdim=True)
    y = target_ip * x + math.sqrt(max(0.0, 1.0 - target_ip * target_ip)) * z
    return y / y.norm(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# One-shot latency measurement for a quant + dequant pair
# ---------------------------------------------------------------------------

def measure_method_latency(name: str, d: int, N: int, x: torch.Tensor,
                           quantize_fn, dequantize_code, bytes_in: int, bytes_out: int,
                           warmup: int, iters: int) -> Dict:
    """Returns latency/throughput stats for quantize and dequantize."""
    q_stats = time_cuda(quantize_fn, warmup=warmup, iters=iters)
    qs = q_stats.summary()

    # Build a valid 'code' once so dequantize has something to decode
    code = quantize_fn()
    def dq(): dequantize_code(code)
    dq_stats = time_cuda(dq, warmup=warmup, iters=iters)
    dqs = dq_stats.summary()

    return {
        "method": name, "d": d, "N": N,
        # quantize
        "quantize_median_us":  qs["median_us"],
        "quantize_mean_us":    qs["mean_us"],
        "quantize_std_us":     qs["std_us"],
        "quantize_p05_us":     qs["p05_us"],
        "quantize_p95_us":     qs["p95_us"],
        "quantize_min_us":     qs["min_us"],
        "quantize_per_iter":   q_stats.per_iter_us,
        "quantize_throughput_gbps": to_gbps(bytes_in, qs["median_us"]),
        # dequantize
        "dequantize_median_us": dqs["median_us"],
        "dequantize_mean_us":   dqs["mean_us"],
        "dequantize_std_us":    dqs["std_us"],
        "dequantize_p05_us":    dqs["p05_us"],
        "dequantize_p95_us":    dqs["p95_us"],
        "dequantize_min_us":    dqs["min_us"],
        "dequantize_per_iter":  dq_stats.per_iter_us,
        "dequantize_throughput_gbps": to_gbps(bytes_out, dqs["median_us"]),
    }


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def phase_correctness(device, ds: List[int]) -> Dict:
    print("\n[phase 1] correctness: CUDA vs pure-PyTorch reference")
    results = {}
    for d in ds:
        torch.manual_seed(0)
        x = make_unit_sphere(128, d, device, seed=0)  # small N — ref is slow
        H = ref.hadamard_matrix(d, device)
        results[d] = {}
        for b in (1, 2, 4):
            tq = TurboQuantMSE(d, b, device, seed=11)
            c = tq.quantize(x); x_hat_cuda = tq.dequantize(c)
            rc = ref.quantize_mse_ref(x, b, tq.signs, H, tq.codebook)
            x_hat_ref = ref.dequantize_mse_ref(rc, tq.signs, H, tq.codebook)
            err = float((x_hat_cuda - x_hat_ref).abs().max().item())
            ok = err < 5e-4
            results[d][b] = {"max_abs_err": err, "pass": ok}
            print(f"  d={d:4d}  b={b}   max |Δ| = {err:.2e}   [{'OK' if ok else 'FAIL'}]")
    return results


# ---------------------------------------------------------------------------
# Accuracy sweep with seed variance
# ---------------------------------------------------------------------------

def phase_accuracy(device, d: int, N: int, mse_bits: List[int], prod_bits: List[int],
                   n_seeds: int) -> Dict:
    print(f"\n[phase 3] accuracy sweep  (d={d}, N={N}, seeds={n_seeds})")

    results = {
        "d": d, "N": N, "mse_bits": mse_bits, "prod_bits": prod_bits,
        "mse": {}, "prod": {},
        "ip_samples_mse": {}, "ip_samples_prod": {},  # one seed each, for histogram plot
    }

    # Single fixed input vector set; randomness is over the quantizer seed.
    x = make_unit_sphere(N, d, device, seed=42)
    y_query = make_unit_sphere(N, d, device, seed=43)

    for b in mse_bits:
        dmses = []
        ip_errs_samples = []
        for s in range(n_seeds):
            tq = TurboQuantMSE(d, b, device, seed=1000 + s)
            c = tq.quantize(x); x_hat = tq.dequantize(c)
            dmses.append(float((x - x_hat).pow(2).sum(dim=1).mean().item()))
            errs = ip_errors(x, x_hat, y_query).detach().cpu().numpy()
            ip_errs_samples.append(errs)
            del tq, c, x_hat
        arr = np.asarray(dmses)
        results["mse"][b] = {
            "n_seeds": n_seeds,
            "d_mse_mean":   float(arr.mean()),
            "d_mse_std":    float(arr.std()),
            "d_mse_min":    float(arr.min()),
            "d_mse_max":    float(arr.max()),
            "d_mse_median": float(np.median(arr)),
            "all":          arr.tolist(),
            "upper_bound":  math.sqrt(3) * math.pi / 2.0 * 4.0 ** -b,
            "lower_bound":  4.0 ** -b,
        }
        results["ip_samples_mse"][b] = ip_errs_samples[0]    # one seed for fig1
        print(f"  TurboQuant_mse  b={b}:  D_mse = {arr.mean():.5f} ± {arr.std():.5f}   "
              f"[min {arr.min():.5f}, max {arr.max():.5f}]   "
              f"paper ≤ {results['mse'][b]['upper_bound']:.5f}")

    for b in prod_bits:
        if (b - 1) not in (1, 2, 4):
            continue
        dprods = []
        ip_errs_samples = []
        for s in range(n_seeds):
            tq = TurboQuantProd(d, b, device, seed=2000 + s)
            c = tq.quantize(x); x_hat = tq.dequantize(c)
            errs = ip_errors(x, x_hat, y_query).detach().cpu().numpy()
            dprods.append(float(np.mean(errs ** 2)))
            ip_errs_samples.append(errs)
            del tq, c, x_hat
        arr = np.asarray(dprods)
        results["prod"][b] = {
            "n_seeds": n_seeds,
            "d_prod_mean":   float(arr.mean()),
            "d_prod_std":    float(arr.std()),
            "d_prod_min":    float(arr.min()),
            "d_prod_max":    float(arr.max()),
            "d_prod_median": float(np.median(arr)),
            "all":           arr.tolist(),
            "upper_bound":   math.sqrt(3) * math.pi * math.pi / d * 4.0 ** -b,
            "lower_bound":   (1.0 / d) * 4.0 ** -b,
        }
        results["ip_samples_prod"][b] = ip_errs_samples[0]
        print(f"  TurboQuant_prod b={b}:  D_prod = {arr.mean():.5f} ± {arr.std():.5f}   "
              f"[min {arr.min():.5f}, max {arr.max():.5f}]   "
              f"paper ≤ {results['prod'][b]['upper_bound']:.5f}")

    return results


# ---------------------------------------------------------------------------
# Bias-vs-IP (Fig 2)
# ---------------------------------------------------------------------------

def phase_bias_vs_ip(device, d: int, N: int, b: int, target_ips: List[float]) -> Dict:
    print(f"\n[phase 4] bias vs ⟨x,y⟩ at b={b}")
    x = make_unit_sphere(N, d, device, seed=42)
    tq_mse  = TurboQuantMSE(d, b, device, seed=11)
    tq_prod = TurboQuantProd(d, b, device, seed=13)
    x_hat_mse  = tq_mse.dequantize(tq_mse.quantize(x))
    x_hat_prod = tq_prod.dequantize(tq_prod.quantize(x))

    out = {"b": b, "ips": target_ips,
           "samples_mse": {}, "samples_prod": {},
           "bias_mse": {}, "bias_prod": {}}
    for ip in target_ips:
        y = make_paired_query(x, ip, seed=int(ip * 1000) + 777)
        em  = ip_errors(x, x_hat_mse,  y).detach().cpu().numpy()
        ep  = ip_errors(x, x_hat_prod, y).detach().cpu().numpy()
        out["samples_mse"][ip]  = em
        out["samples_prod"][ip] = ep
        out["bias_mse"][ip]  = float(em.mean())
        out["bias_prod"][ip] = float(ep.mean())
        print(f"  ⟨x,y⟩={ip:.2f}   mse bias={em.mean():+.5f}   prod bias={ep.mean():+.5f}")

    del tq_mse, tq_prod, x_hat_mse, x_hat_prod
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Latency: one configuration (four methods)
# ---------------------------------------------------------------------------

def bench_one_config(device, d: int, N: int, b: int, warmup: int, iters: int,
                     include_prod: bool) -> List[Dict]:
    """Run all methods at (d, N, b) and return per-method latency rows."""
    x = make_unit_sphere(N, d, device, seed=42)
    bytes_in = N * d * 4     # fp32 input

    rows: List[Dict] = []

    # ---- fp16 cast baseline ----
    fp16 = FP16Baseline(d)
    q_fn = lambda: fp16.quantize(x)
    code = q_fn()
    dq_fn = lambda c=code: fp16.dequantize(c)
    rows.append(measure_method_latency(
        "fp16 cast", d, N, x,
        q_fn, lambda c: fp16.dequantize(c),
        bytes_in=bytes_in, bytes_out=N * d * 2, warmup=warmup, iters=iters))
    rows[-1]["payload_bytes_per_vec"] = fp16.payload_bytes(1)
    del code, fp16

    # ---- Naive scalar ----
    ns = UniformScalarBaseline(d, b, device)
    q_fn = lambda: ns.quantize(x)
    code = q_fn()
    rows.append(measure_method_latency(
        f"naive scalar b={b}", d, N, x,
        q_fn, lambda c: ns.dequantize(c),
        bytes_in=bytes_in, bytes_out=N * d, warmup=warmup, iters=iters))
    rows[-1]["payload_bytes_per_vec"] = ns.payload_bytes(1)
    del code, ns

    # ---- TurboQuant_mse ----
    tq = TurboQuantMSE(d, b, device, seed=101)
    q_fn = lambda: tq.quantize(x)
    code = q_fn()
    rows.append(measure_method_latency(
        f"TurboQuant_mse b={b}", d, N, x,
        q_fn, lambda c: tq.dequantize(c),
        bytes_in=bytes_in,
        bytes_out=(N * ((d + (32 // b - 1)) // (32 // b)) * 4),
        warmup=warmup, iters=iters))
    rows[-1]["payload_bytes_per_vec"] = tq.payload_bytes(1)
    del code, tq

    # ---- TurboQuant_prod ----
    if include_prod and (b - 1) in (1, 2, 4):
        tqp = TurboQuantProd(d, b, device, seed=103)
        q_fn = lambda: tqp.quantize(x)
        code = q_fn()
        rows.append(measure_method_latency(
            f"TurboQuant_prod b={b}", d, N, x,
            q_fn, lambda c: tqp.dequantize(c),
            bytes_in=bytes_in,
            bytes_out=tqp.payload_bytes(N),
            warmup=warmup, iters=iters))
        rows[-1]["payload_bytes_per_vec"] = tqp.payload_bytes(1)
        del code, tqp

    del x
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = pathlib.Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False   # keep accuracy metrics clean

    # ---------- Phase 0: environment dump ----------
    env = detect_env(theoretical_bw_gbps=args.peak_bw_gbps)
    print("\n== CUTurbo rigorous benchmark ==")
    for k, v in dataclasses.asdict(env).items():
        print(f"  {k:22s} {v}")
    dump_json(out_dir / "env.json", env)

    warmup  = args.warmup
    iters   = args.iters
    print(f"\nTiming : {warmup} warmup + {iters} measured iters per config "
          f"(each measurement uses a fresh cudaEvent pair)")

    # ---------- Phase 1: correctness ----------
    corr = phase_correctness(device, ds=sorted(set(args.d_for_correct)))
    dump_json(out_dir / "raw/correctness.json", corr)

    # ---------- Phase 3: accuracy sweep with seed variance ----------
    acc = phase_accuracy(
        device, d=args.d_accuracy, N=args.N_accuracy,
        mse_bits=[1, 2, 4], prod_bits=[2, 3, 5],
        n_seeds=args.n_seeds)
    # Strip large ip_samples arrays before JSON dump
    acc_for_json = {k: (v if k not in ("ip_samples_mse", "ip_samples_prod")
                        else {int(kk): "<array omitted>" for kk in v})
                    for k, v in acc.items()}
    dump_json(out_dir / "raw/accuracy.json", acc_for_json)

    # ---------- Phase 4: bias vs IP ----------
    bias = phase_bias_vs_ip(device, d=args.d_accuracy, N=args.N_accuracy,
                             b=2, target_ips=[0.01, 0.1, 0.3, 0.5])
    bias_for_json = {
        "b": bias["b"],
        "bias_mse":  {str(k): v for k, v in bias["bias_mse"].items()},
        "bias_prod": {str(k): v for k, v in bias["bias_prod"].items()},
    }
    dump_json(out_dir / "raw/bias_vs_ip.json", bias_for_json)

    # ---------- Phase 5 & 6: latency scaling ----------
    print(f"\n[phase 5] latency vs N  (d={args.d_latency}, b={args.b_latency})")
    rows_vs_N: List[Dict] = []
    # prod variant peaks at ~20 * N * d bytes due to cuBLAS S·r workspace.
    # 1.8 GB cap keeps us safe on a 4 GB / 3.4 GB-free Laptop GPU.
    PROD_BYTE_BUDGET = int(1.8e9)
    for N in args.N_sweep:
        include_prod = 20 * N * args.d_latency * 4 <= PROD_BYTE_BUDGET * 10
        # Heuristic above stays generous for typical d=128; override explicitly:
        include_prod = (N * args.d_latency * 4) <= 512_000_000    # ≤ 512 MB input
        print(f"\n  -- config  d={args.d_latency}  N={N}  b={args.b_latency} "
              f"(prod={'on' if include_prod else 'off — would OOM'}) --")
        rows = bench_one_config(device, d=args.d_latency, N=N, b=args.b_latency,
                                warmup=warmup, iters=iters, include_prod=include_prod)
        for r in rows:
            print(f"    {r['method']:24s}  q={r['quantize_median_us']:8.2f} µs   "
                  f"dq={r['dequantize_median_us']:8.2f} µs   "
                  f"q_BW={r['quantize_throughput_gbps']:6.1f} GB/s")
        rows_vs_N.extend(rows)

    print(f"\n[phase 6] latency vs d  (N={args.N_for_d_sweep}, b={args.b_latency})")
    rows_vs_d: List[Dict] = []
    for d in args.d_sweep:
        include_prod = d <= 512   # cap prod sweep to protect VRAM
        print(f"\n  -- config  d={d}  N={args.N_for_d_sweep}  b={args.b_latency} "
              f"(prod={'on' if include_prod else 'off'}) --")
        rows = bench_one_config(device, d=d, N=args.N_for_d_sweep, b=args.b_latency,
                                warmup=warmup, iters=iters, include_prod=include_prod)
        for r in rows:
            print(f"    {r['method']:24s}  q={r['quantize_median_us']:8.2f} µs   "
                  f"dq={r['dequantize_median_us']:8.2f} µs   "
                  f"q_BW={r['quantize_throughput_gbps']:6.1f} GB/s")
        rows_vs_d.extend(rows)

    # For JSON: strip per-iter arrays (they're large). Keep them for plots below.
    _strip = lambda rows: [{k: v for k, v in r.items()
                            if not k.endswith("per_iter")}
                           for r in rows]
    dump_json(out_dir / "raw/latency_vs_N.json", _strip(rows_vs_N))
    dump_json(out_dir / "raw/latency_vs_d.json", _strip(rows_vs_d))

    # ---------- Phase 7: bandwidth utilization (picked rep. config) ----------
    # Use the d=128 / N=largest config for the BW bar chart.
    rep_N = max(args.N_sweep)
    rep_rows = [r for r in rows_vs_N if r["N"] == rep_N]
    bw_rows = []
    for r in rep_rows:
        # effective BW = bytes_in + bytes_out, measured on the quantize kernel
        bytes_in = r["N"] * r["d"] * 4
        bytes_out = r.get("payload_bytes_per_vec", 0) * r["N"]
        bw_rows.append({
            "method": r["method"],
            "effective_bw_gbps": to_gbps(bytes_in + bytes_out, r["quantize_median_us"]),
        })

    # ---------- Phase 8: compression (actual packed bytes) ----------
    comp = []
    for b in (1, 2, 4):
        # Assume d=128 and compute bytes per vector
        idx_per_word = 32 // b
        words = (args.d_accuracy + idx_per_word - 1) // idx_per_word
        actual_bytes = words * 4
        fp16_bytes = args.d_accuracy * 2
        comp.append({"label": f"mse b={b}", "ratio": fp16_bytes / actual_bytes})

    # ---------- Phase 9: Pareto (quality vs speed) ----------
    pareto_rows = []
    for r in [r for r in rows_vs_N if r["N"] == rep_N]:
        name = r["method"]
        # MSE for non-Turbo methods: measure on-demand.
        if "TurboQuant_mse" in name:
            mse_val = acc["mse"][args.b_latency]["d_mse_mean"]
        elif "TurboQuant_prod" in name:
            mse_val = acc["prod"][args.b_latency]["d_prod_mean"]
        elif "fp16" in name:
            mse_val = 1e-7   # tiny but nonzero; fp16 round-trip error
        elif "naive" in name:
            # quick on-the-fly measurement
            x_tmp = make_unit_sphere(min(10_000, rep_N), args.d_accuracy, device, seed=7)
            ns = UniformScalarBaseline(args.d_accuracy, args.b_latency, device)
            xh = ns.dequantize(ns.quantize(x_tmp))
            mse_val = float((x_tmp - xh).pow(2).sum(dim=1).mean().item())
            del x_tmp, ns
        else:
            mse_val = float("nan")
        pareto_rows.append({"method": name,
                            "quantize_gbps": r["quantize_throughput_gbps"],
                            "mse": mse_val})

    # ---------- Plots ----------
    print("\n[plots] writing figures ...")
    fig_dir = out_dir
    plots.plot_ip_error_histograms(acc["ip_samples_mse"], acc["ip_samples_prod"],
                                   str(fig_dir / "fig1_ip_error_hist.png"))
    plots.plot_bias_vs_ip(bias["samples_mse"], bias["samples_prod"],
                          str(fig_dir / "fig2_bias_vs_ip.png"))

    mse_bits_s  = sorted(acc["mse"])
    prod_bits_s = sorted(acc["prod"])
    plots.plot_distortion_vs_bits(
        mse_bits=mse_bits_s,
        mse_vals=[acc["mse"][b]["d_mse_mean"]  for b in mse_bits_s],
        mse_err=[acc["mse"][b]["d_mse_std"]    for b in mse_bits_s],
        prod_bits=prod_bits_s,
        prod_vals=[acc["prod"][b]["d_prod_mean"] for b in prod_bits_s],
        prod_err=[acc["prod"][b]["d_prod_std"]   for b in prod_bits_s],
        d=args.d_accuracy,
        path=str(fig_dir / "fig3_distortion_vs_bits.png"))

    # Scaling plots: group rows by method
    def by_method(rows):
        out: Dict[str, List[Dict]] = {}
        for r in rows:
            out.setdefault(r["method"], []).append(r)
        return out

    plots.plot_scaling_vs_N(by_method(rows_vs_N), d=args.d_latency,
                            path=str(fig_dir / "fig4_scaling_vs_N.png"))
    plots.plot_scaling_vs_d(by_method(rows_vs_d), N=args.N_for_d_sweep,
                            path=str(fig_dir / "fig5_scaling_vs_d.png"))

    # Box plot: use the representative N config
    samples = {r["method"]: r["quantize_per_iter"] for r in rep_rows}
    plots.plot_latency_boxplot(samples, str(fig_dir / "fig6_latency_boxplot.png"))

    plots.plot_bandwidth_utilization(bw_rows, env.theoretical_bw_gbps,
                                     str(fig_dir / "fig7_bandwidth_util.png"))
    plots.plot_compression(comp, str(fig_dir / "fig8_compression.png"))
    plots.plot_pareto(pareto_rows, str(fig_dir / "fig9_pareto.png"))

    # Seed stability box plot (from accuracy phase)
    stab = {}
    for b, info in acc["mse"].items():
        stab.setdefault(b, {})["mse"] = info["all"]
    for b, info in acc["prod"].items():
        stab.setdefault(b, {})["prod"] = info["all"]
    plots.plot_seed_stability(stab, str(fig_dir / "fig10_seed_stability.png"))

    # ---------- CSV summary ----------
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "key", "value"])
        for b, info in acc["mse"].items():
            w.writerow(["accuracy_mse", f"b={b}_mean",  f"{info['d_mse_mean']:.6f}"])
            w.writerow(["accuracy_mse", f"b={b}_std",   f"{info['d_mse_std']:.6f}"])
            w.writerow(["accuracy_mse", f"b={b}_upper", f"{info['upper_bound']:.6f}"])
            w.writerow(["accuracy_mse", f"b={b}_lower", f"{info['lower_bound']:.6f}"])
        for b, info in acc["prod"].items():
            w.writerow(["accuracy_prod", f"b={b}_mean",  f"{info['d_prod_mean']:.6f}"])
            w.writerow(["accuracy_prod", f"b={b}_std",   f"{info['d_prod_std']:.6f}"])
            w.writerow(["accuracy_prod", f"b={b}_upper", f"{info['upper_bound']:.6f}"])
            w.writerow(["accuracy_prod", f"b={b}_lower", f"{info['lower_bound']:.6f}"])
        for ip, v in bias["bias_mse"].items():
            w.writerow(["bias", f"mse_ip={ip}",  f"{v:+.6f}"])
        for ip, v in bias["bias_prod"].items():
            w.writerow(["bias", f"prod_ip={ip}", f"{v:+.6f}"])
        for r in rows_vs_N + rows_vs_d:
            w.writerow([f"latency/{r['method']}/d={r['d']}/N={r['N']}",
                        "quantize_median_us",   f"{r['quantize_median_us']:.4f}"])
            w.writerow([f"latency/{r['method']}/d={r['d']}/N={r['N']}",
                        "dequantize_median_us", f"{r['dequantize_median_us']:.4f}"])
            w.writerow([f"latency/{r['method']}/d={r['d']}/N={r['N']}",
                        "quantize_throughput_gbps", f"{r['quantize_throughput_gbps']:.2f}"])
    print(f"[csv]  wrote {csv_path}")
    print("\nDone.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=15,
                   help="iters to warm up caches & pipeline before measurement")
    p.add_argument("--iters",  type=int, default=100,
                   help="measured iters per configuration (using cudaEvent timing)")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="number of quantizer seeds used to build error bars")
    # Accuracy / Fig 1/2/3 sweep
    p.add_argument("--d-accuracy", type=int, default=128)
    p.add_argument("--N-accuracy", type=int, default=65_536)
    # Scaling sweep
    p.add_argument("--d-latency", type=int, default=128)
    p.add_argument("--b-latency", type=int, default=2)
    p.add_argument("--N-sweep", type=int, nargs="+",
                   default=[65_536, 262_144, 1_048_576])
    p.add_argument("--N-for-d-sweep", type=int, default=131_072)
    p.add_argument("--d-sweep", type=int, nargs="+",
                   default=[64, 128, 256, 512])
    # Correctness
    p.add_argument("--d-for-correct", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024])
    # Reporting
    p.add_argument("--peak-bw-gbps", type=float, default=192.0,
                   help="theoretical peak memory bandwidth, used for % util (RTX 3050 ≈ 192)")
    p.add_argument("--out", type=str,
                   default=str(pathlib.Path(__file__).resolve().parent.parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
