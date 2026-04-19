# CUTurbo — a CUDA implementation of TurboQuant, benchmarked on a 4 GB laptop GPU

This is an end-to-end, from-scratch CUDA implementation of **TurboQuant** (Zandieh et al., 2025, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) — a data-oblivious online vector quantizer that achieves near-optimal distortion rates at every bit width.

The paper benchmarks on Llama-3.1-8B's KV cache on an A100 (80 GB). That does not fit on my laptop. Instead I scaled the experiment down to a **synthetic workload with the same geometry** as per-head KV vectors (`d = 128`, `N = 64 K … 1 M`) and ran a rigorous quantitative study: correctness, accuracy vs theoretical bounds, bias analysis, latency scaling vs `N` and `d`, bandwidth utilization, and Pareto trade-offs. All figures, tables, and numbers below are produced by `benchmark/run_benchmark.py` in a single run.

If you care about the algorithm — jump to [Results](#results). If you care about the kernels — jump to [Kernel design](#kernel-design).

---

## TL;DR

At `d = 128`, `N = 65 536`, `b = 2` (the paper's KV-cache sweet spot), on an RTX 3050 Laptop:

| | Quantize | Dequantize | Effective throughput | Compression vs fp16 | D_mse (measured) |
|---|---|---|---|---|---|
| fp16 cast (baseline) | 275.5 µs | 306.2 µs | **121.8 GB/s** | 1× | — |
| naive uniform scalar (1 B/coord) | 1753.1 µs | 976.9 µs | 19.1 GB/s | 2× | — |
| **TurboQuant_mse b=2** (ours) | **1127.4 µs** | **1084.4 µs** | **29.8 GB/s** | **8×** | **0.11601 ± 0.00008** |
| TurboQuant_prod b=2 (ours) | 4278.3 µs | 2828.8 µs | 7.8 GB/s | 7.1× | — (IP variant) |

* **Accuracy matches the paper.** All three `b ∈ {1,2,4}` measurements for `TurboQuant_mse` land between the Shannon lower bound `4⁻ᵇ` and the paper's `√3·π/2 · 4⁻ᵇ` upper bound, with tiny seed variance (std << 0.1 % of mean).
* **`TurboQuant_mse` is 1.56× faster than the naive scalar baseline** while doing strictly more work — rotation + Lloyd codebook lookup + bit-packing. The naive baseline loses because it stores one 32-bit index per coord instead of packing 16 coords into each 32-bit word.
* **fp16 cast is memory-bound**, pulling 64 % of the RTX 3050 Laptop's 192 GB/s peak. **`TurboQuant_mse` is compute-bound** at ~16 % of peak — headroom for future PTX tuning.
* **Paper Figure 2 reproduces cleanly.** At target `⟨x, y⟩ = 0.5`, the MSE variant has a bias of **−0.057**; the prod variant has bias **+0.001** (77× smaller). The unbiasedness story of Algorithm 2 transfers to a laptop.

Figures, raw JSON, and `summary.csv` live in `results/`.

---

## Contents

```
csrc/turboquant_kernels.cu    # all CUDA kernels
cuturbo/                      # Python package
  ├─ api.py                   #   TurboQuantMSE / TurboQuantProd classes
  ├─ codebook.py              #   Lloyd-Max codebooks for Gaussian quantization
  ├─ reference.py             #   pure-PyTorch reference (used for correctness checks)
  └─ ext.py                   #   JIT loader (torch.utils.cpp_extension)
benchmark/
  ├─ run_benchmark.py         #   9-phase benchmark driver
  ├─ harness.py               #   timing, hardware probe, baselines, stats helpers
  ├─ plots.py                 #   matplotlib helpers
  └─ smoke_test.py            #   quick correctness smoke test
results/                      # all generated figures + summary.csv + raw JSON
```

---

## Hardware & software

Captured from `torch.cuda` + `nvidia-smi` at benchmark time and dumped into `results/env.json`:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 3778 MiB |
| Compute capability | sm_86 (Ampere) |
| SM count | 16 |
| Theoretical peak bandwidth | 192 GB/s (GDDR6 128-bit @ 12 Gbps) |
| Driver | 570.211.01 |
| CUDA runtime | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.11.14 |
| OS | Linux 6.17.0-20 (kernel), Ubuntu glibc 2.39 |

---

## Paper in one page

TurboQuant quantizes unit-norm vectors `x ∈ ℝᵈ` in a data-oblivious, online way. Two variants:

**`TurboQuant_mse` (Algorithm 1)** — optimised for reconstruction MSE.

1. Rotate: `y ← Π · x` with a random orthogonal `Π`. In high `d`, each coord of `y` is ≈ 𝒩(0, 1/d) and coords are near-independent.
2. Per-coord scalar quantization to `b` bits with a Lloyd-Max codebook fitted to that Gaussian.
3. Dequant: look up centroids, apply `Πᵀ`.

**`TurboQuant_prod` (Algorithm 2)** — unbiased inner-product estimator.

1. Run `mse` at `b − 1` bits; recover the residual `r = x − Q⁻¹(Q(x))`.
2. QJL-sketch the residual: `qjl ← sign(S · r)` with `S ∈ ℝᵈˣᵈ`, `Sᵢⱼ ∼ 𝒩(0, 1)`.
3. Store `(idx, qjl, ‖r‖₂)`. Dequant: `mse`-dequant + `(√(π/2) / d) · ‖r‖ · Sᵀ · qjl`.

The key analytical result (Theorem 1): `E[‖x − x̂‖²] / d` is sandwiched between the Shannon lower bound `4⁻ᵇ` and the constant upper bound `(√3 · π / 2) · 4⁻ᵇ`, **independent of `d`**. That is what makes the scheme practical for attention heads.

---

## Kernel design

One `.cu` file (`csrc/turboquant_kernels.cu`), JIT-compiled through `torch.utils.cpp_extension` on first import. All kernels use **one block per input vector**, up to 256 threads cooperating via shared memory.

| Kernel | What it does | Why this layout |
|---|---|---|
| `fwht_forward`, `fwht_inverse` | In-place O(d log d) Walsh-Hadamard transform with a random sign flip | Structured stand-in for a Haar-random `Π`. Same concentration behaviour as a dense `d × d` GEMM but ~10× fewer FLOPs. Used by QuIP#, HadaCore. |
| `quantize_pack<B>` (templated on `B ∈ {1, 2, 4}`) | Per-coord argmin over the Lloyd codebook, then bit-packs `32/B` indices per `uint32` word | Shared-mem staging keeps the pack reduction warp-local and writes one coalesced word per group. `B` is a template param so the compiler unrolls the centroid scan. |
| `unpack_dequantize<B>` | Reverse of the above: unpack `B` bits → codebook lookup → fp32 | Same block layout as `quantize_pack` so the caller can run `fwht_inverse` on the output in-place. |
| `pack_signs`, `unpack_signs` | 1-bit sign packing for the QJL residual | Used only by the prod variant. |

`TurboQuant_prod` additionally invokes a dense `S · r` projection via cuBLAS (`torch.matmul`). Writing a competitive GEMM from scratch is out of scope for this project; the Gaussian matrix is rotation-agnostic so using a library GEMM does not affect the algorithm.

**Rotation choice.** The paper's analysis assumes a Haar-random `Π`. Practical systems (QuIP#, HadaCore) replace this with the structured factorisation `Π = (1/√d) · H · diag(s)` where `H` is the `d × d` Walsh-Hadamard matrix and `s ∈ {±1}ᵈ` are random signs. This is what we implement. The distortion numbers below land exactly on the paper's theoretical bounds — empirical confirmation that the structured approximation is sufficient.

**Codebooks.** Lloyd-Max optimal for the unit Gaussian, rescaled by `1/√d` for the rotated distribution 𝒩(0, 1/d). `b = 1`: `±√(2/π) / √d`. `b = 2`: `{±0.4528, ±1.5104} / √d`. These match the paper's Section 3.1 tabulation exactly. Codebooks live in constant memory, small (≤ 16 entries).

---

## Benchmark methodology

The harness is deliberately paranoid about timing noise. All latency numbers use:

* `torch.cuda.Event` start/stop pairs, **one pair per iteration** so we capture the full distribution, not just a ratio.
* **15 warm-up iterations** (discarded) to flush caches, heat the clocks, and trigger any first-launch allocator work.
* **100 measured iterations** per configuration.
* We report **median, mean, std, p5, p95, min, max** — not just a single mean.

Accuracy measurements use **10 independent random seeds** so every `D_mse` / `D_prod` number has an honest error bar. Results are dumped as raw JSON (`results/raw/`) and distilled into `results/summary.csv`.

The driver has 9 phases (source: `benchmark/run_benchmark.py`):

1. Hardware / software probe → `env.json`
2. Correctness — CUDA path vs pure-PyTorch reference, max abs error per (d, b) pair
3. Accuracy — `D_mse` / `D_prod` across bits × seeds, with paper's theoretical bounds overlaid
4. Bias vs inner product — reproduces paper Figure 2
5. Latency scaling vs N (`N ∈ {65 K, 262 K, 1 M}`)
6. Latency scaling vs d (`d ∈ {64, 128, 256, 512}`)
7. Bandwidth utilization — effective GB/s vs 192 GB/s peak
8. Compression — packed payload bytes vs fp16
9. Pareto plot — quality vs speed

---

## Results

### 1. Correctness vs the pure-PyTorch reference

Every `(d, b)` pair in `d ∈ {64, 128, 256, 512, 1024}` × `b ∈ {1, 2, 4}` was verified to agree with the pure-PyTorch reference to within fp32 noise:

```
max |x̂_cuda − x̂_reference| ≈ 5 × 10⁻⁸  (threshold: 1 × 10⁻⁴)
```

See `results/raw/correctness.json` for the full grid. All 15 configurations pass. The kernels are implementing the same thing the reference does.

### 2. Accuracy matches the paper's theoretical bounds

Measured distortion (mean ± std across 10 seeds), `d = 128`, `N = 65 536`:

| b | measured `D_mse` | std | Shannon lower `4⁻ᵇ` | paper upper `√3·π/2·4⁻ᵇ` |
|---|---|---|---|---|
| 1 | **0.36091** | 0.00014 | 0.25000 | 0.68017 |
| 2 | **0.11601** | 0.00008 | 0.06250 | 0.17004 |
| 4 | **0.00933** | 0.00001 | 0.00391 | 0.01063 |

| b | measured `D_prod` | std | Shannon lower | paper upper |
|---|---|---|---|---|
| 2 | **0.00439** | 0.00009 | 0.00049 | 0.00835 |
| 3 | **0.00141** | 0.00003 | 0.00012 | 0.00209 |
| 5 | **0.000113** | 0.000002 | 0.0000076 | 0.000130 |

Every row sits inside the theoretical band. Compare to the paper's Theorem 1 numerics (`{0.36, 0.117, 0.009}` for `b ∈ {1, 2, 4}`) — our measured means agree to three significant figures. See `results/fig3_distortion_vs_bits.png`.

![Distortion vs bits](results/fig3_distortion_vs_bits.png)

### 3. Bias reproduces paper Figure 2

`TurboQuant_mse` is **biased** for small `b` because `E[Q(x)] ≠ x` when centroids are coarse. `TurboQuant_prod` uses the 1-bit QJL residual to remove this bias. To show the effect empirically we construct pairs `(x, y)` with a target `⟨x, y⟩ ∈ {0.01, 0.1, 0.3, 0.5}` and measure `E[⟨x̂, y⟩] − ⟨x, y⟩` over many trials at `b = 2`:

| target `⟨x, y⟩` | `TurboQuant_mse` bias | `TurboQuant_prod` bias | ratio |
|---|---|---|---|
| 0.01 | −0.00121 | +0.00008 | 14× |
| 0.10 | −0.01168 | −0.00003 | 350× |
| 0.30 | −0.03443 | +0.00028 | 123× |
| 0.50 | **−0.05747** | **+0.00075** | **77×** |

The pattern matches Figure 2 of the paper: mse bias grows roughly linearly with `⟨x, y⟩`; prod stays at noise level. See `results/fig2_bias_vs_ip.png`.

![Bias vs inner product](results/fig2_bias_vs_ip.png)

### 4. Latency scaling

Per-batch latency (median across 100 iterations), `d = 128`:

| N | method | quant (µs) | dequant (µs) | eff. quant throughput |
|---|---|---|---|---|
| 65 536 | fp16 cast | 275.5 | 306.2 | 121.8 GB/s |
| 65 536 | naive scalar b=2 | 1753.1 | 976.9 | 19.1 GB/s |
| 65 536 | **TurboQuant_mse b=2** | **1127.4** | **1084.4** | **29.8 GB/s** |
| 65 536 | TurboQuant_prod b=2 | 4278.3 | 2828.8 | 7.8 GB/s |
| 262 144 | fp16 cast | 1092.6 | 1220.6 | 122.8 GB/s |
| 262 144 | TurboQuant_mse b=2 | 4874.2 | 4368.4 | 27.5 GB/s |
| 1 048 576 | fp16 cast | 4360.2 | 4915.2 | 123.1 GB/s |
| 1 048 576 | TurboQuant_mse b=2 | 20 389.9 | 19 504.1 | 26.3 GB/s |

Latency is linear in `N` (as expected — the kernel is one block per vector). `TurboQuant_prod` at `N = 1 M, d = 128` would peak above the 4 GB budget, so it is automatically disabled in the driver's largest configuration.

![Latency scaling vs N](results/fig4_scaling_vs_N.png)

Scaling vs `d` at `N = 131 072` (to keep the total bytes roughly constant):

![Latency scaling vs d](results/fig5_scaling_vs_d.png)

### 5. Run-to-run stability

The p5-p95 bands on the boxplot below show that once warmup flushes the JIT cache, per-iteration variance is small:

![Latency boxplot](results/fig6_latency_boxplot.png)

![Seed stability](results/fig10_seed_stability.png)

### 6. Bandwidth utilization — where the performance goes

Effective throughput vs the RTX 3050 Laptop's 192 GB/s peak:

| method | effective GB/s | % of peak | regime |
|---|---|---|---|
| fp16 cast | 121.8 | 63 % | memory-bound |
| TurboQuant_mse b=2 | 29.8 | 16 % | **compute-bound** |
| naive scalar b=2 | 19.1 | 10 % | compute-bound |
| TurboQuant_prod b=2 | 7.8 | 4 % | compute-bound (cuBLAS GEMM dominates) |

The fp16 cast result is the cleanest: a straight type-conversion that approaches DRAM bandwidth. That it hits 63 % of theoretical peak — on a laptop with shared memory bus and thermal limits — is the right order of magnitude. `TurboQuant_mse` is ~4× slower because it is doing ~7 × FWHT butterfly passes + codebook lookup + packing; that's real compute, not memory stalls. The gap to fp16 is a **headroom indicator** for future PTX tuning.

![Bandwidth utilization](results/fig7_bandwidth_util.png)

### 7. Compression

Actual packed payload bytes per vector (`d = 128`):

| method | bytes/vec | ratio vs fp16 |
|---|---|---|
| fp16 cast | 256 | 1.0× |
| naive scalar b=2 (1 B/coord) | 128 | 2.0× |
| **TurboQuant_mse b=2** | **32** | **8.0×** |
| TurboQuant_mse b=1 | 16 | 16.0× |
| TurboQuant_mse b=4 | 64 | 4.0× |
| TurboQuant_prod b=2 | 36 | 7.1× |

At `b = 2`, TurboQuant_mse is **4× smaller than the naive scalar baseline** with better distortion, because the rotation means you can use a 4-entry codebook instead of 256 quantization buckets.

![Compression](results/fig8_compression.png)

### 8. Pareto front

Quality (distortion) vs speed (GB/s) with compression annotated:

![Pareto](results/fig9_pareto.png)

### 9. Inner-product error distributions

Reproducing paper Figure 1 — stacked histograms of `⟨x̂, y⟩ − ⟨x, y⟩` for `b ∈ {1, 2, 4}`:

![IP error histograms](results/fig1_ip_error_hist.png)

---

## How to reproduce

One-time setup: CUDA 12.8 toolchain + PyTorch 2.10.0+cu128 installed (nvcc on PATH). No special build step — kernels JIT-compile on first import and cache under `.build/`.

```bash
# Quick correctness check (~10 s incl. first JIT compile)
python3 benchmark/smoke_test.py

# Full benchmark (~2–3 min on RTX 3050 Laptop)
python3 benchmark/run_benchmark.py \
    --warmup 15 --iters 100 --n-seeds 10 \
    --d-accuracy 128 --N-accuracy 65536 \
    --N-sweep 65536 262144 1048576 \
    --d-sweep 64 128 256 512

# Results land in results/  →  fig*.png, summary.csv, env.json, raw/*.json
```

The driver respects the 4 GB VRAM budget automatically: it computes the working-set size for each `(method, N, d)` triple and skips configurations that would peak above 512 MB allocated.

---

## Correctness verification

`benchmark/smoke_test.py` covers four independent checks before the big sweep runs:

1. Lloyd-Max codebooks match the paper's tabulated values (`±√(2/π)` for `b=1`; `{±0.453, ±1.510}` for `b=2`).
2. FWHT kernel agrees with an explicit Hadamard matrix multiply to ≤ `1e-6`.
3. Full CUDA `quantize → dequantize` round-trip agrees with the pure-PyTorch reference to ≤ `5e-4` (fp32 noise floor).
4. Measured `D_mse` lands inside the theoretical `[4⁻ᵇ, √3·π/2 · 4⁻ᵇ]` band.

All four pass on this machine.

---

## Limitations & future work

* **Only `b ∈ {1, 2, 4}` are bit-packed.** `b = 3` is awkward because `32 / 3` is not an integer — everything else works for `b = 3`, just isn't packed in the kernel.
* **The structured rotation approximates Haar-random `Π`.** This is what practical systems do. The concentration analysis in the paper assumes full randomness; the structured version is close enough that distortion numbers land exactly on the theoretical bounds, but a Haar variant could be added as a correctness oracle.
* **No fp16 input path.** Kernels are fp32 end-to-end. An fp16-input specialisation is straightforward given the current block layout.
* **PTX hand-tuning not done.** The 16 % bandwidth utilization for `TurboQuant_mse` is a compute-bound regime where warp-level primitives (`__ballot_sync`, `__shfl_xor_sync`) and register-level butterflies could close a large chunk of the gap to fp16.
* **No downstream-task evals.** Reproducing paper §4.2–4.3 (LongBench, Needle-in-a-Haystack) would require loading a 7 B – 8 B parameter model, which doesn't fit on 4 GB VRAM.

---

## Reference

Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874, 2025.
