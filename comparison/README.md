# comparison/ — cuTile feasibility probe on Ampere

This folder holds a one-shot feasibility check for running [DevTechJr/turboquant_cutile](https://github.com/DevTechJr/turboquant_cutile) on our RTX 3050 Laptop (sm_86, driver 570.211.01 / CUDA 12.8).

**Result:** cuTile refuses to JIT on this GPU because it requires NVIDIA driver ≥ 13.0; ours is 12.8.

```
$ python check_cutile_env.py

=== GPU / driver ===
  NVIDIA GeForce RTX 3050 Laptop GPU, 570.211.01, 8.6
=== cuda.tile import ===
  cuda.tile version: 1.3.0
=== Trivial cuTile kernel JIT on this GPU ===
  cuTile JIT/launch FAILED: RuntimeError: Minimum driver version required is 13.0, got 12.8
```

`cuda-tile 1.3.0` installs and imports on CUDA 12.8, but any `ct.launch(...)` fails at runtime because the library links against a CUDA-13-era runtime. This is a **driver-version gate**, not a compute-capability gate — Ampere hardware could run cuTile if the driver line supported it.

No head-to-head benchmark is possible on this machine. On a Blackwell-class GPU with driver ≥ 13.0, `python run_4way.py` will pick up the real cuTile kernel automatically.

## Files

- `setup.sh` — idempotent clone + venv + install
- `check_cutile_env.py` — the feasibility probe above
- `cutile_smoke.py` — tries to invoke his `turboquant_compress_2bit` on synthetic input (requires cuTile to JIT)
- `run_4way.py` — benchmark driver (skips the cuTile column when it can't JIT)
- `turboquant_cutile/` — his repo, cloned on first run (gitignored)

## References

- His repo: https://github.com/DevTechJr/turboquant_cutile
- cuTile docs: `pip show cuda-tile` / NVIDIA developer docs
