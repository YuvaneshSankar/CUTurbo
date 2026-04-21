"""Stage 1 — feasibility probe. Can cuTile run on this GPU at all?

Tries increasingly ambitious things and reports what works. Exit 0 means
the 4-way benchmark is viable; exit 1 means we fall back to design-space
comparison only.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def section(title: str):
    print(f"\n=== {title} ===")


def main() -> int:
    section("GPU / driver")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"], text=True).strip()
        print(f"  {out}")
    except Exception as e:
        print(f"  nvidia-smi failed: {e}")
        return 1

    section("cuda.tile import")
    try:
        import cuda.tile as ct
        print(f"  cuda.tile version: {ct.__version__}")
    except ImportError as e:
        print(f"  ImportError: {e}")
        return 1

    section("torch + CUDA")
    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  torch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            print(f"  device: {props.name} (sm_{props.major}{props.minor})")
    except ImportError as e:
        print(f"  ImportError: {e}")
        print("  (install torch with: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128)")
        return 1

    section("Trivial cuTile kernel JIT on this GPU")
    try:
        @ct.kernel
        def copy_kernel(src, dst, n: int):
            bid = ct.bid(0)
            BLOCK = 128
            start = bid * BLOCK
            tile = ct.load(src, index=(start,), shape=(BLOCK,),
                           padding_mode=ct.PaddingMode.ZERO)
            ct.store(dst, index=(start,), tile=tile)

        import torch
        N = 1024
        src = torch.arange(N, device="cuda", dtype=torch.float32)
        dst = torch.empty_like(src)
        grid = ((N + 127) // 128, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, copy_kernel, (src, dst, N))
        torch.cuda.synchronize()
        ok = torch.allclose(src, dst)
        print(f"  trivial copy kernel: {'OK ✓' if ok else 'FAIL (output mismatch)'}")
        if not ok:
            return 1
    except Exception as e:
        print(f"  cuTile JIT/launch FAILED: {type(e).__name__}: {e}")
        return 1

    section("His repo smoke import")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "turboquant_cutile"))
        from turboquant_cutile import TurboQuantEngine
        engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cuda")
        print(f"  TurboQuantEngine constructed: mse_bits={engine.mse_bits}")

        import torch
        K = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        result = engine.launch_compress_keys(K)
        print("  launch_compress_keys outputs:")
        for k, v in result.items():
            if hasattr(v, "shape"):
                print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v).__name__}")
        print("  his kernel ran on our GPU ✓")
    except Exception as e:
        print(f"  his kernel FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    print("\nSTAGE 1 OK — 4-way benchmark viable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
