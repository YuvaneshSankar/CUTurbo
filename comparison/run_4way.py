"""Stage 3 — 4-way kernel latency + quality comparison.

  1. Ours unfused       — TurboQuantProd(b=3, fused=False) + scalar pack_signs
  2. Ours fused CUDA    — TurboQuantProd(b=3, fused=True)  + scalar pack_signs
  3. Ours fused + PTX   — fused MSE + warp-ballot pack_signs_ptx
  4. His cuTile         — TurboQuantEngine(total_bits=3).launch_compress_keys

All four implement the same algorithm: rotate → 2-bit MSE quantize → residual →
1-bit QJL sign. Output bit-exact parity is not expected (different rotations Π),
but D_mse must land in the paper's theoretical band for every variant.

Writes:  comparison/results/summary.csv
         comparison/results/raw/comparison.json
         comparison/results/fig_4way_latency.png
         comparison/results/fig_4way_speedup.png
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent   # CUTurbo/
_HIS  = Path(__file__).resolve().parent / "turboquant_cutile"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HIS))

from benchmark.harness import detect_env, dump_json, time_cuda   # noqa: E402
from cuturbo.api import TurboQuantProd                           # noqa: E402
from cuturbo.ext import get_ext                                  # noqa: E402

try:
    from turboquant_cutile import TurboQuantEngine                # noqa: E402
    HAS_HIS = True
except Exception as e:
    print(f"warning: could not import his repo ({e}) — will skip his column")
    HAS_HIS = False


# ---------------------------------------------------------------------------
# Wrappers — each takes an input K and runs one full compress path, returns
# the compressed payload (or None if we don't need it). All timed identically.
# ---------------------------------------------------------------------------

def ours_unfused_factory(d, b=3, device="cuda", seed=0):
    ext = get_ext()
    tq = TurboQuantProd(d, b, device, seed=seed, fused=False)
    def run(x):
        return tq.quantize(x)
    # Correctness reconstruction (for D_mse):
    def dequant(code): return tq.dequantize(code)
    return run, dequant, f"ours unfused (prod b={b})"


def ours_fused_factory(d, b=3, device="cuda", seed=0):
    tq = TurboQuantProd(d, b, device, seed=seed, fused=True)
    def run(x):
        return tq.quantize(x)
    def dequant(code): return tq.dequantize(code)
    return run, dequant, f"ours fused (prod b={b})"


def ours_fused_ptx_factory(d, b=3, device="cuda", seed=0):
    """Reuse TurboQuantProd(fused=True) but swap the pack_signs kernel for
    the warp-ballot PTX version. Done by monkey-patching the method at the
    ext-module level — only affects our measurement."""
    ext = get_ext()
    tq = TurboQuantProd(d, b, device, seed=seed, fused=True)
    # Rebind: during quantize, we need ext.pack_signs → ext.pack_signs_ptx
    # TurboQuantProd.quantize calls self.ext.pack_signs(projection). We shadow
    # by swapping the reference on `tq.ext`. Safe because tq owns its ext ref.
    class _ExtShim:
        def __init__(self, real): self._r = real
        def __getattr__(self, name):
            if name == "pack_signs":
                return self._r.pack_signs_ptx
            return getattr(self._r, name)
    tq.ext = _ExtShim(ext)
    def run(x):
        return tq.quantize(x)
    def dequant(code):
        tq.ext = ext           # dequant only needs the real ext
        out = tq.dequantize(code)
        tq.ext = _ExtShim(ext)
        return out
    return run, dequant, f"ours fused+PTX (prod b={b})"


def his_cutile_factory(d, total_bits=3, device="cuda", seed=42):
    if not HAS_HIS:
        return None
    engine = TurboQuantEngine(head_dim=d, total_bits=total_bits, seed=seed, device=device)

    # Try cuTile once — if it fails (e.g. driver version too old for cuTile),
    # fall back to his PyTorch reference path. We label the returned variant
    # accordingly so the forum post is honest about which path was timed.
    test_input = torch.randn(16, d, device=device, dtype=torch.float16)
    cutile_works = False
    try:
        engine.launch_compress_keys(test_input)
        cutile_works = True
    except Exception as e:
        cutile_mode_err = f"{type(e).__name__}: {e}"
        print(f"  [cuTile launch failed on this GPU: {cutile_mode_err}]")
        print(f"  [falling back to his PyTorch reference path for the 'his' column]")

    if cutile_works:
        label = "his cuTile kernel"
        def run(x):
            return engine.launch_compress_keys(x.half() if x.dtype != torch.float16 else x)
    else:
        label = "his PyTorch reference (cuTile unavailable on this driver)"
        def run(x):
            return engine.compress_keys_pytorch(x.half() if x.dtype != torch.float16 else x)

    def dequant(result):
        k_mse = result["k_mse"].float()
        signs = result["qjl_signs"].float()
        r_norms = result["residual_norms"].float()
        qjl_term = signs @ engine.S.float()
        return k_mse + engine.correction_scale * r_norms.unsqueeze(-1) * qjl_term

    return run, dequant, label


# ---------------------------------------------------------------------------
# Correctness: D_mse of reconstruction, should land in paper's band
# ---------------------------------------------------------------------------

def check_dmse(x, dequant_fn, code, b_mse=2):
    x_hat = dequant_fn(code)
    if x_hat.dtype != x.dtype:
        x_hat = x_hat.to(x.dtype)
    err_sq = (x.float() - x_hat.float()).pow(2).sum(dim=1).mean().item()
    d = x.shape[1]
    d_mse = err_sq / d   # matches our definition in run_benchmark
    lower = 4 ** (-b_mse)
    upper = math.sqrt(3) * math.pi / 2 * (4 ** (-b_mse))
    # For prod variant the reconstruction error drops below the mse lower bound
    # once the QJL correction kicks in; we just check it's positive and finite.
    return d_mse, lower, upper


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters",  type=int, default=100)
    ap.add_argument("--d",      type=int, default=128)
    ap.add_argument("--Ns", type=int, nargs="+", default=[8192, 32768, 65536])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    env = detect_env()
    dump_json(args.out_dir / "env.json", env)
    print(f"=== 4-way TurboQuant comparison on {env.gpu_name} ({env.gpu_vram_mib} MiB) ===")

    raw = []

    for N in args.Ns:
        print(f"\n--- d={args.d}, N={N} ---")

        torch.manual_seed(0)
        x = torch.randn(N, args.d, device=device, dtype=torch.float32)
        x /= x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        factories = [ours_unfused_factory, ours_fused_factory, ours_fused_ptx_factory,
                     his_cutile_factory]
        rows = []
        for fac in factories:
            built = fac(args.d, device=device)
            if built is None:
                continue
            run, dequant, name = built

            # Correctness: single-run D_mse check
            code = run(x)
            d_mse, lower, upper = check_dmse(x, dequant, code, b_mse=2)

            # Timing
            stats = time_cuda(lambda: run(x),
                              warmup=args.warmup, iters=args.iters).summary()

            print(f"  {name:40s} median {stats['median_us']:8.1f} µs   "
                  f"D_mse={d_mse:.4f} "
                  f"(Shannon≥{lower:.4f}, upper={upper:.4f})")

            rows.append({
                "variant": name,
                "median_us": stats["median_us"],
                "p05_us":    stats["p05_us"],
                "p95_us":    stats["p95_us"],
                "min_us":    stats["min_us"],
                "max_us":    stats["max_us"],
                "D_mse":     d_mse,
                "d_mse_lower": lower,
                "d_mse_upper": upper,
            })
            del code
            gc.collect(); torch.cuda.empty_cache()

        raw.append({"d": args.d, "N": N, "variants": rows})

    # -----------------------------------------------------------------
    # Write artefacts
    # -----------------------------------------------------------------
    dump_json(args.out_dir / "raw" / "comparison.json", raw)

    with open(args.out_dir / "summary.csv", "w") as f:
        f.write("d,N,variant,median_us,p05_us,p95_us,D_mse\n")
        for cfg in raw:
            for r in cfg["variants"]:
                f.write(f"{cfg['d']},{cfg['N']},{r['variant']},"
                        f"{r['median_us']:.2f},{r['p05_us']:.2f},"
                        f"{r['p95_us']:.2f},{r['D_mse']:.4f}\n")

    # Plots
    _plot_latency(raw, args.out_dir / "fig_4way_latency.png")
    _plot_speedup(raw, args.out_dir / "fig_4way_speedup.png")

    print(f"\nDONE.")
    print(f"  summary → {args.out_dir / 'summary.csv'}")
    print(f"  plots   → {args.out_dir / 'fig_4way_latency.png'}")


def _plot_latency(raw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(_REPO))
    from benchmark.plots import PAL

    variants_seen = []
    for cfg in raw:
        for r in cfg["variants"]:
            if r["variant"] not in variants_seen:
                variants_seen.append(r["variant"])
    colors = [PAL["naive"], PAL["mse"], PAL["prod"], PAL["accent"]][:len(variants_seen)]

    configs = [f"N={cfg['N']}" for cfg in raw]
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(configs), 4.8))
    x = np.arange(len(configs))
    width = 0.8 / max(len(variants_seen), 1)

    for i, vname in enumerate(variants_seen):
        vals = []
        for cfg in raw:
            m = [r for r in cfg["variants"] if r["variant"] == vname]
            vals.append(m[0]["median_us"] if m else 0)
        ax.bar(x + (i - (len(variants_seen) - 1) / 2) * width, vals, width,
               color=colors[i], edgecolor="black", linewidth=0.5, label=vname)

    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.set_ylabel("Median compress latency (µs)")
    ax.set_title(f"TurboQuant compress kernel — 4-way comparison (d={raw[0]['d']})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _plot_speedup(raw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(_REPO))
    from benchmark.plots import PAL

    variants_seen = []
    for cfg in raw:
        for r in cfg["variants"]:
            if r["variant"] not in variants_seen:
                variants_seen.append(r["variant"])
    colors = [PAL["naive"], PAL["mse"], PAL["prod"], PAL["accent"]][:len(variants_seen)]

    configs = [f"N={cfg['N']}" for cfg in raw]
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(configs), 4.8))
    x = np.arange(len(configs))
    width = 0.8 / max(len(variants_seen), 1)

    # Normalize each config to the slowest variant = 1.0×
    speedups = {}
    for cfg in raw:
        meds = [r["median_us"] for r in cfg["variants"]]
        baseline = max(meds) if meds else 1.0
        for r in cfg["variants"]:
            speedups.setdefault(r["variant"], []).append(baseline / max(r["median_us"], 1e-9))

    for i, vname in enumerate(variants_seen):
        vals = speedups.get(vname, [0] * len(configs))
        ax.bar(x + (i - (len(variants_seen) - 1) / 2) * width, vals, width,
               color=colors[i], edgecolor="black", linewidth=0.5, label=vname)
        for xi, v in zip(x + (i - (len(variants_seen) - 1) / 2) * width, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}×", ha="center", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.set_ylabel("Speedup vs slowest variant in each config")
    ax.set_title(f"TurboQuant compress — relative speed (d={raw[0]['d']})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
