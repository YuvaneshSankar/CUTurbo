"""
TurboQuant kernel benchmark: unfused vs fused vs fused+PTX.

Three-column comparison for the MSE pipeline:
  unfused  — two kernels (fwht_forward + quantize_pack), y via HBM round-trip
  fused    — one kernel, y stays in shared memory
  fused_ptx — one kernel, packing loop uses inline PTX `bfi.b32`

Plus a standalone comparison for `pack_signs`:
  scalar  — current kernel, 32 scalar OR-shifts per word
  ptx     — warp-ballot kernel using inline PTX `vote.sync.ballot.b32`

Reports median µs and speedup per config. Writes results/fusion/ artefacts.
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
from cuturbo.ext import get_ext                                      # noqa: E402
from cuturbo.codebook import build_codebook                          # noqa: E402


def _hbm_bytes_saved(N: int, d: int) -> int:
    """Unfused writes + reads N*d*4 bytes of intermediate; fused skips it."""
    return 2 * N * d * 4


def _make_signs(d, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=g, dtype=torch.int32) * 2 - 1
    return bits.to(device=device, dtype=torch.float32)


def bench_fused_mse(args, configs, bits, device, env):
    """Phase 1: unfused / fused / fused_ptx comparison for MSE quant+dequant."""
    ext = get_ext()
    raw = []
    flat_rows = []

    for d, N in configs:
        x = torch.randn(N, d, device=device, dtype=torch.float32)
        x /= x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        signs = _make_signs(d, device, seed=0)

        for b in bits:
            name = f"d={d}, N={N}, b={b}"
            print(f"\n--- MSE {name} ---")

            unfused = TurboQuantMSE(d, b, device, seed=0, fused=False)
            fused   = TurboQuantMSE(d, b, device, seed=0, fused=True)
            # For the PTX path we keep the same signs/codebook but call the
            # ext binding directly so this benchmark stays faithful to
            # the kernel delta (no extra Python overhead).
            cb = fused.codebook

            # Correctness guards
            c_unf = unfused.quantize(x)
            c_fus = fused.quantize(x)
            c_ptx = ext.fused_quantize_ptx(x.contiguous(), signs, cb, int(b))
            assert torch.equal(c_unf.packed, c_fus.packed), f"{name}: fused mismatch"
            assert torch.equal(c_fus.packed, c_ptx),         f"{name}: fused_ptx mismatch"

            # Timing — quantize (three variants)
            t_unf = time_cuda(lambda: unfused.quantize(x),
                              warmup=args.warmup, iters=args.iters).summary()
            t_fus = time_cuda(lambda: fused.quantize(x),
                              warmup=args.warmup, iters=args.iters).summary()
            t_ptx = time_cuda(lambda: ext.fused_quantize_ptx(x, signs, cb, int(b)),
                              warmup=args.warmup, iters=args.iters).summary()

            # Timing — dequantize (two variants; no PTX variant for dequant)
            dt_unf = time_cuda(lambda: unfused.dequantize(c_unf),
                               warmup=args.warmup, iters=args.iters).summary()
            dt_fus = time_cuda(lambda: fused.dequantize(c_fus),
                               warmup=args.warmup, iters=args.iters).summary()

            su_fus_vs_unf = t_unf["median_us"] / max(t_fus["median_us"], 1e-9)
            su_ptx_vs_unf = t_unf["median_us"] / max(t_ptx["median_us"], 1e-9)
            su_ptx_vs_fus = t_fus["median_us"] / max(t_ptx["median_us"], 1e-9)
            su_deq        = dt_unf["median_us"] / max(dt_fus["median_us"], 1e-9)

            bytes_saved = _hbm_bytes_saved(N, d)

            print(f"  quantize   unfused   {t_unf['median_us']:8.1f} µs")
            print(f"             fused     {t_fus['median_us']:8.1f} µs   "
                  f"{su_fus_vs_unf:.2f}× vs unfused")
            print(f"             fused+ptx {t_ptx['median_us']:8.1f} µs   "
                  f"{su_ptx_vs_unf:.2f}× vs unfused   "
                  f"{su_ptx_vs_fus:.2f}× vs fused")
            print(f"  dequantize unfused   {dt_unf['median_us']:8.1f} µs")
            print(f"             fused     {dt_fus['median_us']:8.1f} µs   "
                  f"{su_deq:.2f}× vs unfused")
            print(f"  HBM saved by fusion: {bytes_saved / 2**20:.1f} MiB")

            raw.append({
                "d": d, "N": N, "b": b,
                "quantize": {
                    "unfused": t_unf, "fused": t_fus, "fused_ptx": t_ptx,
                    "speedup_fused_vs_unfused":  su_fus_vs_unf,
                    "speedup_ptx_vs_unfused":    su_ptx_vs_unf,
                    "speedup_ptx_vs_fused":      su_ptx_vs_fus,
                    "hbm_bytes_saved_vs_unfused": bytes_saved,
                },
                "dequantize": {
                    "unfused": dt_unf, "fused": dt_fus,
                    "speedup_fused_vs_unfused": su_deq,
                    "hbm_bytes_saved_vs_unfused": bytes_saved,
                },
            })
            flat_rows.append({
                "config": name, "direction": "quantize",
                "unfused_us":   t_unf["median_us"],
                "fused_us":     t_fus["median_us"],
                "fused_ptx_us": t_ptx["median_us"],
                "speedup_fused":  su_fus_vs_unf,
                "speedup_ptx":    su_ptx_vs_unf,
            })
            flat_rows.append({
                "config": name, "direction": "dequantize",
                "unfused_us":   dt_unf["median_us"],
                "fused_us":     dt_fus["median_us"],
                "fused_ptx_us": None,    # no PTX dequant variant
                "speedup_fused": su_deq,
                "speedup_ptx":   None,
            })

            del unfused, fused, c_unf, c_fus, c_ptx
            gc.collect(); torch.cuda.empty_cache()

    return raw, flat_rows


def bench_pack_signs(args, configs, device):
    """Phase 2: pack_signs (scalar) vs pack_signs_ptx (warp-ballot)."""
    ext = get_ext()
    rows = []

    for d, N in configs:
        x = torch.randn(N, d, device=device, dtype=torch.float32)
        # Verify bit-exact
        pk_scalar = ext.pack_signs(x)
        pk_ptx    = ext.pack_signs_ptx(x)
        assert torch.equal(pk_scalar, pk_ptx), f"pack_signs_ptx mismatch at d={d}, N={N}"

        t_scalar = time_cuda(lambda: ext.pack_signs(x),
                             warmup=args.warmup, iters=args.iters).summary()
        t_ptx    = time_cuda(lambda: ext.pack_signs_ptx(x),
                             warmup=args.warmup, iters=args.iters).summary()
        speedup = t_scalar["median_us"] / max(t_ptx["median_us"], 1e-9)

        print(f"  pack_signs  d={d:4d}, N={N:>7d}   "
              f"scalar {t_scalar['median_us']:8.1f} µs   "
              f"ptx {t_ptx['median_us']:8.1f} µs   "
              f"{speedup:4.2f}×")

        rows.append({
            "d": d, "N": N,
            "scalar": t_scalar, "ptx": t_ptx,
            "speedup_ptx_vs_scalar": speedup,
        })
    return rows


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
    print(f"=== Kernel benchmark on {env.gpu_name} ({env.gpu_vram_mib} MiB) ===")

    mse_configs = [
        (128, 65_536),
        (128, 262_144),
        (256, 65_536),
        (256, 262_144),
        (512, 65_536),
    ]
    bits = [1, 2, 4]

    pack_configs = [
        (128, 65_536),
        (128, 262_144),
        (256, 65_536),
        (256, 262_144),
        (512, 65_536),
        (512, 262_144),
    ]

    print("\n>>> Phase 1: MSE quant/dequant (unfused / fused / fused+PTX)")
    raw_mse, flat_mse = bench_fused_mse(args, mse_configs, bits, device, env)

    print("\n>>> Phase 2: pack_signs (scalar vs warp-ballot PTX)")
    raw_pack = bench_pack_signs(args, pack_configs, device)

    # -----------------------------------------------------------------
    # Write artefacts
    # -----------------------------------------------------------------
    dump_json(args.out_dir / "raw" / "fusion.json",
              {"mse": raw_mse, "pack_signs": raw_pack})

    with open(args.out_dir / "summary.csv", "w") as f:
        # MSE section
        f.write("# MSE: unfused / fused / fused_ptx\n")
        f.write("d,N,b,direction,unfused_us,fused_us,fused_ptx_us,"
                "speedup_fused_vs_unfused,speedup_ptx_vs_unfused,speedup_ptx_vs_fused,"
                "hbm_saved_mib\n")
        for r in raw_mse:
            q = r["quantize"]
            f.write(f"{r['d']},{r['N']},{r['b']},quantize,"
                    f"{q['unfused']['median_us']:.2f},"
                    f"{q['fused']['median_us']:.2f},"
                    f"{q['fused_ptx']['median_us']:.2f},"
                    f"{q['speedup_fused_vs_unfused']:.3f},"
                    f"{q['speedup_ptx_vs_unfused']:.3f},"
                    f"{q['speedup_ptx_vs_fused']:.3f},"
                    f"{q['hbm_bytes_saved_vs_unfused'] / 2**20:.1f}\n")
            d_ = r["dequantize"]
            f.write(f"{r['d']},{r['N']},{r['b']},dequantize,"
                    f"{d_['unfused']['median_us']:.2f},"
                    f"{d_['fused']['median_us']:.2f},,"
                    f"{d_['speedup_fused_vs_unfused']:.3f},,,"
                    f"{d_['hbm_bytes_saved_vs_unfused'] / 2**20:.1f}\n")
        # pack_signs section
        f.write("\n# pack_signs: scalar / warp-ballot PTX\n")
        f.write("d,N,scalar_us,ptx_us,speedup_ptx_vs_scalar\n")
        for r in raw_pack:
            f.write(f"{r['d']},{r['N']},"
                    f"{r['scalar']['median_us']:.2f},"
                    f"{r['ptx']['median_us']:.2f},"
                    f"{r['speedup_ptx_vs_scalar']:.3f}\n")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    plots.plot_fusion_speedup_3col(flat_mse,
                                    str(args.out_dir / "fig_fusion_speedup.png"))
    plots.plot_pack_signs_speedup(raw_pack,
                                   str(args.out_dir / "fig_pack_signs_ptx.png"))

    print("\nDONE.")
    print(f"  summary → {args.out_dir / 'summary.csv'}")
    print(f"  plots   → {args.out_dir}/fig_fusion_speedup.png, fig_pack_signs_ptx.png")


if __name__ == "__main__":
    main()
