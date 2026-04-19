"""Matplotlib helpers for CUTurbo. All plots are Agg-backend, 130-150 DPI PNGs."""
from __future__ import annotations

import math
import pathlib
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Consistent palette across every figure
PAL = {
    "fp16":   "#95A5A6",
    "naive":  "#C0392B",
    "mse":    "#2E86AB",
    "prod":   "#6D3BB9",
    "upper":  "#C0392B",
    "lower":  "#27AE60",
    "accent": "#E07B39",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "figure.constrained_layout.use": True,
})


# ---------------------------------------------------------------------------
# Figure 1 — error distribution histograms across bit-widths
# ---------------------------------------------------------------------------

def plot_ip_error_histograms(samples_mse: Dict[int, np.ndarray],
                             samples_prod: Dict[int, np.ndarray],
                             path: str,
                             title: str = "Inner-product error distribution (paper Fig 1 reproduction)"):
    bits = sorted(set(samples_mse) | set(samples_prod))
    fig, axes = plt.subplots(2, len(bits), figsize=(3.4 * len(bits), 5.6), sharey="row")
    if len(bits) == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, b in enumerate(bits):
        for i, (label, samples, color) in enumerate([
            ("TurboQuant_prod", samples_prod, PAL["prod"]),
            ("TurboQuant_mse",  samples_mse,  PAL["accent"]),
        ]):
            ax = axes[i, j]
            s = samples.get(b)
            if s is None or s.size == 0:
                ax.axis("off")
                continue
            ax.hist(s, bins=60, alpha=0.85, color=color, edgecolor="black", linewidth=0.2)
            mean = float(np.mean(s))
            ax.axvline(0.0,  linestyle=":",  color="gray",  linewidth=0.8)
            ax.axvline(mean, linestyle="--", color="black", linewidth=0.9,
                       label=f"mean={mean:+.4f}")
            ax.set_title(f"{label}  b={b}")
            ax.set_xlabel("Inner-product error")
            if j == 0:
                ax.set_ylabel("Count")
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=12)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — mse bias grows with ⟨x,y⟩, prod stays centered
# ---------------------------------------------------------------------------

def plot_bias_vs_ip(samples_mse: Dict[float, np.ndarray],
                    samples_prod: Dict[float, np.ndarray],
                    path: str,
                    title: str = "Bias vs ⟨x,y⟩ (paper Fig 2 reproduction)"):
    ips = sorted(set(samples_mse) | set(samples_prod))
    fig, axes = plt.subplots(2, len(ips), figsize=(3.4 * len(ips), 5.6), sharey="row")
    if len(ips) == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, ip in enumerate(ips):
        for i, (label, samples, color) in enumerate([
            ("TurboQuant_prod", samples_prod, PAL["prod"]),
            ("TurboQuant_mse",  samples_mse,  PAL["accent"]),
        ]):
            ax = axes[i, j]
            s = samples[ip]
            ax.hist(s, bins=60, alpha=0.85, color=color, edgecolor="black", linewidth=0.2)
            mean = float(np.mean(s))
            ax.axvline(0.0,  linestyle=":",  color="gray",  linewidth=0.8)
            ax.axvline(mean, linestyle="--", color="black", linewidth=0.9,
                       label=f"mean={mean:+.4f}")
            ax.set_title(f"{label}   ⟨x,y⟩≈{ip}")
            ax.set_xlabel("IP error")
            if j == 0:
                ax.set_ylabel("Count")
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=12)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — distortion vs bits, with Shannon bounds and seed variance
# ---------------------------------------------------------------------------

def plot_distortion_vs_bits(
    mse_bits: Sequence[int],       mse_vals: Sequence[float],  mse_err: Sequence[float],
    prod_bits: Sequence[int],      prod_vals: Sequence[float], prod_err: Sequence[float],
    d: int, path: str):
    """Reproduces paper Figure 3, plus error bars from the seed-variance sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6))

    pb = np.asarray(prod_bits); mb = np.asarray(mse_bits)

    # (a) Inner-product error
    if pb.size:
        lower_ip = (1.0 / d) * 4.0 ** -pb
        upper_ip = math.sqrt(3) * math.pi * math.pi / d * 4.0 ** -pb
        ax1.errorbar(pb, prod_vals, yerr=prod_err, fmt="o-",
                     color=PAL["prod"], lw=2, ms=7, capsize=3,
                     label="TurboQuant_prod (mean ± std over seeds)")
        ax1.plot(pb, upper_ip, "--", color=PAL["upper"], lw=1.4,
                 label="Upper bound: √3·π²/d · 4⁻ᵇ (paper)")
        ax1.plot(pb, lower_ip, "--", color=PAL["lower"], lw=1.4,
                 label="Lower bound: 4⁻ᵇ/d (Shannon)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Bit-width  b")
    ax1.set_ylabel(r"Inner-product error  $D_{\mathrm{prod}}$")
    ax1.set_title("(a) Inner-product error vs bits")
    if (pb.size or mb.size):
        ax1.set_xticks(sorted(set(list(pb) + list(mb))))
    ax1.legend(fontsize=9)

    # (b) Reconstruction MSE
    if mb.size:
        lower_mse = 4.0 ** -mb
        upper_mse = math.sqrt(3) * math.pi / 2.0 * 4.0 ** -mb
        ax2.errorbar(mb, mse_vals, yerr=mse_err, fmt="s-",
                     color=PAL["mse"], lw=2, ms=7, capsize=3,
                     label="TurboQuant_mse (mean ± std over seeds)")
        ax2.plot(mb, upper_mse, "--", color=PAL["upper"], lw=1.4,
                 label="Upper bound: √3·π/2 · 4⁻ᵇ (paper)")
        ax2.plot(mb, lower_mse, "--", color=PAL["lower"], lw=1.4,
                 label="Lower bound: 4⁻ᵇ (Shannon)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Bit-width  b")
    ax2.set_ylabel(r"Reconstruction MSE  $D_{\mathrm{mse}}$")
    ax2.set_title("(b) Reconstruction MSE vs bits")
    if mb.size:
        ax2.set_xticks(list(mb))
    ax2.legend(fontsize=9)

    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scaling — latency vs N (log-log)  and  latency vs d
# ---------------------------------------------------------------------------

def plot_scaling_vs_N(results_by_method: Dict[str, List[dict]],
                      d: int, path: str):
    """One line per method, with p5-p95 shaded band."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    for method, rows in results_by_method.items():
        rows = sorted(rows, key=lambda r: r["N"])
        Ns = [r["N"] for r in rows]
        med_q  = [r["quantize_median_us"] for r in rows]
        p5_q   = [r["quantize_p05_us"]   for r in rows]
        p95_q  = [r["quantize_p95_us"]   for r in rows]
        gbps_q = [r["quantize_throughput_gbps"] for r in rows]

        ax1.plot(Ns, med_q, "o-", lw=2, ms=6, label=method, color=_method_color(method))
        ax1.fill_between(Ns, p5_q, p95_q, alpha=0.15, color=_method_color(method))

        ax2.plot(Ns, gbps_q, "o-", lw=2, ms=6, label=method, color=_method_color(method))

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("N (number of vectors)")
    ax1.set_ylabel("Quantize latency (µs, log)  — shaded band = p5–p95")
    ax1.set_title(f"(a) Latency scaling  (d={d})")
    ax1.legend(fontsize=9)

    ax2.set_xscale("log")
    ax2.set_xlabel("N (number of vectors)")
    ax2.set_ylabel("Throughput (GB/s fp32 input)")
    ax2.set_title(f"(b) Throughput scaling  (d={d})")
    ax2.legend(fontsize=9)

    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_vs_d(results_by_method: Dict[str, List[dict]],
                      N: int, path: str):
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for method, rows in results_by_method.items():
        rows = sorted(rows, key=lambda r: r["d"])
        ds = [r["d"] for r in rows]
        usv = [r["quantize_median_us"] / r["N"] for r in rows]   # µs / vector
        ax.plot(ds, usv, "o-", lw=2, ms=6, label=method, color=_method_color(method))

    # Reference O(d log d) curve (normalized to TurboQuant_mse@d=128 if available)
    mse_rows = results_by_method.get("TurboQuant_mse b=2")
    if mse_rows:
        ref = sorted(mse_rows, key=lambda r: r["d"])
        d0 = ref[0]["d"]; t0 = ref[0]["quantize_median_us"] / ref[0]["N"]
        xs = np.array([r["d"] for r in ref])
        ys = t0 * (xs / d0) * np.log2(xs) / math.log2(d0)
        ax.plot(xs, ys, ":", color="gray", lw=1.3, label=r"reference $O(d \log d)$")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("d (vector dimension)")
    ax.set_ylabel("Quantize latency (µs / vector)")
    ax.set_title(f"Latency vs dimension  (N={N})")
    ax.set_xticks([d for d in sorted({r["d"] for m in results_by_method.values() for r in m})])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Box plot of per-iteration latencies — shows run-to-run variance
# ---------------------------------------------------------------------------

def plot_latency_boxplot(samples_by_method: Dict[str, List[float]],
                         path: str,
                         title: str = "Per-iteration latency distribution (100 iters, 15 warmup)"):
    names = list(samples_by_method)
    data = [samples_by_method[n] for n in names]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                    whis=(5, 95), widths=0.55,
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(_method_color(name))
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(names) + 1), names, rotation=15, ha="right")
    ax.set_ylabel("Latency (µs)  — whiskers are 5/95th percentile")
    ax.set_title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bandwidth utilization bar chart
# ---------------------------------------------------------------------------

def plot_bandwidth_utilization(rows: List[dict], theoretical_bw_gbps: float, path: str):
    names  = [r["method"] for r in rows]
    gbps   = [r["effective_bw_gbps"] for r in rows]
    util   = [g / theoretical_bw_gbps * 100 for g in gbps]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    colors = [_method_color(n) for n in names]
    bars = ax.bar(names, util, color=colors, alpha=0.85)
    for b, g, u in zip(bars, gbps, util):
        ax.annotate(f"{g:.1f} GB/s\n({u:.1f}%)",
                    xy=(b.get_x() + b.get_width() / 2, u),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(100, linestyle="--", color="black", lw=0.8)
    ax.set_ylabel(f"% of theoretical peak ({theoretical_bw_gbps:.0f} GB/s GDDR6)")
    ax.set_title("Effective memory bandwidth utilization (RTX 3050 Laptop)")
    ax.set_ylim(0, max(max(util) * 1.25, 100))
    plt.xticks(rotation=15, ha="right")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Compression — bar chart (real packed bytes vs fp16 baseline)
# ---------------------------------------------------------------------------

def plot_compression(bits_and_actual: List[dict], path: str):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = [r["label"] for r in bits_and_actual]
    ratios = [r["ratio"] for r in bits_and_actual]
    colors = [_ratio_color(i) for i in range(len(labels))]
    bars = ax.bar(labels, ratios, color=colors, alpha=0.85)
    for b, r in zip(bars, ratios):
        ax.annotate(f"{r:.1f}×",
                    xy=(b.get_x() + b.get_width()/2, r),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(1.0, linestyle="--", color="gray", lw=1, label="fp16 baseline")
    ax.set_ylabel("Compression ratio vs fp16 (actual packed bytes)")
    ax.set_title("Memory compression — real packed sizes")
    ax.legend(fontsize=8)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto frontier: accuracy vs speed
# ---------------------------------------------------------------------------

def plot_pareto(rows: List[dict], path: str):
    """Each row: {method, quantize_gbps, mse_distortion, dequantize_gbps}."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for r in rows:
        ax.scatter(r["quantize_gbps"], r["mse"],
                   s=140, color=_method_color(r["method"]),
                   edgecolor="black", linewidth=0.5, zorder=3, label=r["method"])
        ax.annotate(r["method"],
                    xy=(r["quantize_gbps"], r["mse"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Quantize throughput (GB/s, higher is better →)")
    ax.set_ylabel("Reconstruction MSE (lower is better ↓, log)")
    ax.set_title("Quality-vs-speed Pareto frontier")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Seed-stability: D_mse / D_prod across quantizer seeds
# ---------------------------------------------------------------------------

def plot_seed_stability(by_bit: Dict[int, Dict[str, List[float]]], path: str):
    """by_bit[b]['mse'] / ['prod'] = list of distortion values across seeds."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    positions = []
    labels = []
    data = []
    colors = []
    pos = 1
    for b, d in sorted(by_bit.items()):
        for variant in ("mse", "prod"):
            vals = d.get(variant)
            if not vals:
                continue
            positions.append(pos)
            labels.append(f"b={b}\n{variant}")
            data.append(vals)
            colors.append(PAL["mse"] if variant == "mse" else PAL["prod"])
            pos += 1
        pos += 1    # gap between bit groups

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.2),
                    whis=(5, 95), showfliers=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.65)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Distortion (log)")
    ax.set_title("Seed stability of TurboQuant randomized quantizers")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Kernel fusion (fused vs unfused MSE quantize/dequantize)
# ---------------------------------------------------------------------------

def plot_fusion_speedup(rows: List[Dict], path: str,
                        title: str = "Kernel fusion: unfused vs fused TurboQuant_mse latency"):
    """rows: each dict has {config, direction, unfused_us, fused_us, speedup}."""
    # Stable ordering of configs as they appear in the input
    seen = []
    for r in rows:
        if r["config"] not in seen:
            seen.append(r["config"])
    # Two directions: quantize and dequantize. Build one subplot per direction.
    directions = ["quantize", "dequantize"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

    for ax, direction in zip(axes, directions):
        keep = [r for r in rows if r["direction"] == direction]
        configs = [c for c in seen if any(r["config"] == c for r in keep)]
        x = np.arange(len(configs))
        width = 0.38
        unfused_vals = [next(r["unfused_us"] for r in keep if r["config"] == c) for c in configs]
        fused_vals   = [next(r["fused_us"]   for r in keep if r["config"] == c) for c in configs]
        speedups     = [next(r["speedup"]    for r in keep if r["config"] == c) for c in configs]

        ax.bar(x - width / 2, unfused_vals, width, color=PAL["naive"],
               edgecolor="black", linewidth=0.5, label="unfused (two kernels)")
        ax.bar(x + width / 2, fused_vals, width, color=PAL["mse"],
               edgecolor="black", linewidth=0.5, label="fused (one kernel)")

        for xi, uv, fv, sp in zip(x, unfused_vals, fused_vals, speedups):
            y = max(uv, fv) * 1.04
            ax.text(xi, y, f"{sp:.2f}×", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Median latency (µs)")
        ax.set_title(f"{direction}")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(title, fontsize=12)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Retrieval figures (SIFT-1M)
# ---------------------------------------------------------------------------

def plot_recall_vs_size(rows: List[Dict], path: str,
                        title: str = "SIFT-1M: Recall@10 vs index size (Pareto)"):
    """rows: each dict has keys {method, recall_at_10, index_mb}."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in rows:
        ax.scatter(r["index_mb"], r["recall_at_10"],
                   s=72, color=_method_color(r["method"]),
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(r["method"],
                    xy=(r["index_mb"], r["recall_at_10"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Index size (MB)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_recall_vs_qps(rows: List[Dict], path: str,
                       title: str = "SIFT-1M: Recall@10 vs query throughput"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in rows:
        ax.scatter(r["qps"], r["recall_at_10"],
                   s=72, color=_method_color(r["method"]),
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(r["method"],
                    xy=(r["qps"], r["recall_at_10"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Queries per second (higher is better)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_recall_rerank_vs_size(rows: List[Dict], path: str,
                                title: str = "SIFT-1M: two-stage Recall@10 (fp32 rerank) vs index size"):
    """Same Pareto as plot_recall_vs_size but using the production-deployment metric:
    quantized index produces top-100 candidates → fp32 re-ranks them → Recall@10."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in rows:
        y = r.get("recall_at_10_rerank", r.get("recall_at_10"))
        ax.scatter(r["index_mb"], y,
                   s=72, color=_method_color(r["method"]),
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(r["method"],
                    xy=(r["index_mb"], y),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Index size (MB)")
    ax.set_ylabel("Recall@10 (2-stage: quantized shortlist + fp32 rerank)")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_prod_vs_mse_recall(rows: List[Dict], path: str,
                            title: str = "SIFT-1M: TurboQuant_mse vs TurboQuant_prod at matched bits"):
    """rows: each dict has {variant ∈ {mse, prod}, b, recall_at_10}. The prod variant at bit b
    uses b-1 bits for the MSE stage plus a 1-bit QJL residual — this is the paper's fair
    comparison: same total bit budget, unbiased vs biased."""
    mse_rows = sorted([r for r in rows if r["variant"] == "mse"], key=lambda r: r["b"])
    prod_rows = sorted([r for r in rows if r["variant"] == "prod"], key=lambda r: r["b"])
    bits = sorted({r["b"] for r in rows})

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.38
    x = np.arange(len(bits))

    def _bar(variant_rows, offset, color, label):
        values = []
        for b in bits:
            match = [r for r in variant_rows if r["b"] == b]
            values.append(match[0]["recall_at_10"] if match else np.nan)
        ax.bar(x + offset, values, width, color=color, edgecolor="black",
               linewidth=0.5, label=label)
        for xi, v in zip(x + offset, values):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    _bar(mse_rows, -width / 2, PAL["mse"], "TurboQuant_mse (biased)")
    _bar(prod_rows, +width / 2, PAL["prod"], "TurboQuant_prod (unbiased)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"b={b}" for b in bits])
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _method_color(name: str) -> str:
    n = name.lower()
    if "fp16" in n: return PAL["fp16"]
    if "naive" in n: return PAL["naive"]
    if "prod" in n: return PAL["prod"]
    if "mse" in n:  return PAL["mse"]
    return PAL["accent"]


def _ratio_color(i: int) -> str:
    ramp = [PAL["prod"], PAL["mse"], PAL["accent"], PAL["lower"], PAL["naive"]]
    return ramp[i % len(ramp)]
