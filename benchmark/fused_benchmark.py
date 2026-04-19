"""
Fused-vs-unfused MSE kernel benchmark.

Compares TurboQuant_mse with fused=False (two kernel launches:
  fwht_forward → quantize_pack) against fused=True (one kernel).
Same for dequantize. Reports median µs and speedup per config, writes
results/fusion/ artefacts.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.harness import detect_env, dump_json, time_cuda       # noqa: E402
from benchmark import plots                                          # noqa: E402
from cuturbo.api import TurboQuantMSE                                # noqa: E402


def _hbm_bytes_saved_quantize(N: int, d: int) -> int:
    """Unfused writes + reads N*d*4 bytes of intermediate y; fused skips that."""
    return 2 * N * d * 4


def _hbm_bytes_saved_dequantize(N: int, d: int) -> int:
    return 2 * N * d * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results/fusion"))
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters",  type=int, default=100)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    env = detect_env()
    dump_json(args.out_dir / "env.json", env)
    print(f"=== Kernel fusion benchmark on {env.gpu_name} ({env.gpu_vram_mib} MiB) ===")

    # Sweep. Keep total bytes under 1 GB to stay well within 4 GB VRAM.
    configs = [
        (128, 65_536),
        (128, 262_144),
        (256, 65_536),
        (256, 262_144),
        (512, 65_536),
    ]
    bits = [1, 2, 4]

    raw = []
    flat_rows = []

    for d, N in configs:
        x = torch.randn(N, d, device=device, dtype=torch.float32)
        x /= x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        for b in bits:
            name = f"d={d}, N={N}, b={b}"
            print(f"\n--- {name} ---")

            unfused = TurboQuantMSE(d, b, device, seed=0, fused=False)
            fused   = TurboQuantMSE(d, b, device, seed=0, fused=True)

            # Correctness guard (cheap — same seed → same params)
            assert torch.equal(unfused.signs, fused.signs)
            c_unf = unfused.quantize(x)
            c_fus = fused.quantize(x)
            assert torch.equal(c_unf.packed, c_fus.packed), f"{name}: packed mismatch"
            xh_unf = unfused.dequantize(c_unf)
            xh_fus = fused.dequantize(c_fus)
            max_err = (xh_unf - xh_fus).abs().max().item()
            assert max_err < 1e-5, f"{name}: dequant mismatch {max_err}"

            # Timing — quantize
            q_unf = time_cuda(lambda: unfused.quantize(x),
                              warmup=args.warmup, iters=args.iters).summary()
            q_fus = time_cuda(lambda: fused.quantize(x),
                              warmup=args.warmup, iters=args.iters).summary()
            # Timing — dequantize
            d_unf = time_cuda(lambda: unfused.dequantize(c_unf),
                              warmup=args.warmup, iters=args.iters).summary()
            d_fus = time_cuda(lambda: fused.dequantize(c_fus),
                              warmup=args.warmup, iters=args.iters).summary()

            q_speedup = q_unf["median_us"] / max(q_fus["median_us"], 1e-9)
            d_speedup = d_unf["median_us"] / max(d_fus["median_us"], 1e-9)

            bytes_saved_q = _hbm_bytes_saved_quantize(N, d)
            bytes_saved_d = _hbm_bytes_saved_dequantize(N, d)

            print(f"  quantize   unfused {q_unf['median_us']:8.1f} µs   "
                  f"fused {q_fus['median_us']:8.1f} µs   "
                  f"speedup {q_speedup:4.2f}×   "
                  f"HBM saved {bytes_saved_q / 2**20:6.1f} MiB")
            print(f"  dequantize unfused {d_unf['median_us']:8.1f} µs   "
                  f"fused {d_fus['median_us']:8.1f} µs   "
                  f"speedup {d_speedup:4.2f}×   "
                  f"HBM saved {bytes_saved_d / 2**20:6.1f} MiB")

            raw.append({
                "d": d, "N": N, "b": b,
                "quantize": {"unfused": q_unf, "fused": q_fus,
                             "speedup": q_speedup,
                             "hbm_bytes_saved": bytes_saved_q},
                "dequantize": {"unfused": d_unf, "fused": d_fus,
                               "speedup": d_speedup,
                               "hbm_bytes_saved": bytes_saved_d},
            })

            flat_rows.append({
                "config": name, "direction": "quantize",
                "unfused_us": q_unf["median_us"], "fused_us": q_fus["median_us"],
                "speedup": q_speedup,
                "hbm_saved_mib": bytes_saved_q / 2**20,
            })
            flat_rows.append({
                "config": name, "direction": "dequantize",
                "unfused_us": d_unf["median_us"], "fused_us": d_fus["median_us"],
                "speedup": d_speedup,
                "hbm_saved_mib": bytes_saved_d / 2**20,
            })

            del unfused, fused, c_unf, c_fus, xh_unf, xh_fus
            gc.collect(); torch.cuda.empty_cache()

    # Write artefacts
    dump_json(args.out_dir / "raw" / "fusion.json", raw)

    with open(args.out_dir / "summary.csv", "w") as f:
        f.write("d,N,b,direction,unfused_us,fused_us,speedup,hbm_saved_mib\n")
        for r in raw:
            for dir_ in ("quantize", "dequantize"):
                s = r[dir_]
                f.write(f"{r['d']},{r['N']},{r['b']},{dir_},"
                        f"{s['unfused']['median_us']:.2f},"
                        f"{s['fused']['median_us']:.2f},"
                        f"{s['speedup']:.3f},"
                        f"{s['hbm_bytes_saved'] / 2**20:.1f}\n")

    plots.plot_fusion_speedup(flat_rows, str(args.out_dir / "fig_fusion_speedup.png"))

    print("\nDONE.")
    print(f"  summary → {args.out_dir / 'summary.csv'}")
    print(f"  plot    → {args.out_dir / 'fig_fusion_speedup.png'}")


if __name__ == "__main__":
    main()
