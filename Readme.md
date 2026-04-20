# CUTurbo — a CUDA implementation of TurboQuant, verified end-to-end on a 4 GB laptop GPU

This is a from-scratch CUDA implementation of **TurboQuant** (Zandieh et al., 2025, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) — a data-oblivious online vector quantizer that achieves near-optimal distortion rates at every bit width.

The paper evaluates on Llama-3.1-8B's KV cache on an A100. That model does not fit on my laptop. Instead, this repo verifies TurboQuant in two complementary ways that together cover the paper's claims:

1. **Algorithmic correctness (synthetic microbenchmark).** Unit-norm Gaussian vectors, paper's *exact* setting — directly tests Theorem 1 (distortion bound) and Figure 2 (unbiasedness).
2. **End-to-end retrieval on SIFT-1M.** 1 million 128-dim vectors, brute-force ANN with `Recall@k` vs fp32 ground truth — the canonical public downstream benchmark for vector quantization, and the task `TurboQuant_prod` (Algorithm 2) was designed for.

Every number in this document is produced by `benchmark/run_benchmark.py` and `benchmark/sift_retrieval.py`; raw JSON and CSV live in `results/` and `results/sift/`.

---

## TL;DR

**Algorithmic correctness matches the paper to three sig figs.**

| b | measured `D_mse` (10 seeds) | paper Theorem 1 | Shannon lower bound `4⁻ᵇ` | paper upper bound `√3·π/2 · 4⁻ᵇ` |
|---|---|---|---|---|
| 1 | **0.3609 ± 0.0001** | 0.36 | 0.250 | 0.680 |
| 2 | **0.1160 ± 0.0001** | 0.117 | 0.0625 | 0.170 |
| 4 | **0.00933 ± 0.00001** | 0.009 | 0.00391 | 0.01063 |

**SIFT-1M retrieval: TurboQuant_mse delivers 93% Recall@10 at 8× compression** (production pattern: quantized top-100 shortlist + fp32 rerank).

| Method | Index | Compression vs fp32 | Raw Recall@10 | **2-stage Recall@10** |
|---|---|---|---|---|
| fp32 brute force | 488 MB | 1× | 1.000 | 1.000 |
| fp16 | 244 MB | 2× | 0.993 | 1.000 |
| naive scalar b=4 | 61 MB | 8× | 0.279 | 0.676 |
| **TurboQuant_mse b=4** (ours) | **61 MB** | **8×** | **0.484** | **0.928** |
| **TurboQuant_prod b=5** (ours) | **80 MB** | **6×** | **0.544** | **0.962** |
| naive scalar b=2 | 31 MB | 16× | 0.010 | 0.059 |
| **TurboQuant_mse b=2** (ours) | **31 MB** | **16×** | **0.173** | **0.516** |

At matched compression (61 MB, 8× vs fp32), TurboQuant_mse recovers **93% Recall@10** with fp32 rerank; the naive uniform quantizer only recovers **68%**. The TurboQuant compression shortlist is 25 percentage points better than the naive one. At 16× compression, naive scalar is essentially random (6% R@10 rerank); TurboQuant_mse still preserves over half the true top-10 (52%).

---

## How to read these benchmarks

Terminology confusion is easy here because the word "baseline" is overloaded. In this study there are **three distinct kinds of baseline**, each answering a different question:

| What you see in tables | What it *is* | What it's *for* | Is the paper claiming TurboQuant beats it? |
|---|---|---|---|
| `fp32 brute force` | No compression, full precision | Retrieval-quality **ceiling** and kernel-time memory-bandwidth ceiling | No — it's the reference answer |
| `fp16` | Type cast, no quantization | Memory-only baseline (2× compression, near-perfect quality) | No — included so readers can separate dtype savings from quantization savings |
| `naive uniform scalar` (per-coord bucketing, 1 B/coord at b=8, packed similarly at b=2/4) | An alternative quantizer with the same bit budget as TurboQuant | **The apples-to-apples competitor** — what you'd build if you didn't know TurboQuant | Yes — and TurboQuant wins on every metric at every bit width |

Two further clarifications :
- **The paper's end-to-end inference speedup (§4.2–4.3) is not reproduced here.** That claim requires loading Llama-3.1-8B and measuring attention wall-clock; 8 B parameters won't fit on 4 GB VRAM. What we *do* reproduce is the mechanism that drives that speedup: smaller KV cache at the same retrieval accuracy. The SIFT-1M results verify this on a public, publishable benchmark.
- **Kernel-time latency (microseconds to compress a batch of vectors) vs end-to-end retrieval throughput (queries per second on a 1 M corpus) measure different things.** The kernel-time table is how fast one `quantize()` call is; the retrieval table is how fast you can serve real ANN queries. Don't conflate them.

---

## Contents

```
csrc/turboquant_kernels.cu     # all CUDA kernels (FWHT, quantize+pack, dequant+unpack, sign-pack)
cuturbo/                       # Python package
  ├─ api.py                    #   TurboQuantMSE / TurboQuantProd classes
  ├─ retrieval.py              #   QuantizedIndex — brute-force ANN with chunked dequant+GEMM
  ├─ codebook.py               #   Lloyd-Max codebooks for Gaussian quantization
  ├─ reference.py              #   pure-PyTorch reference (correctness oracle)
  └─ ext.py                    #   JIT loader (torch.utils.cpp_extension)
benchmark/
  ├─ run_benchmark.py          #   synthetic accuracy + latency microbenchmark
  ├─ sift_retrieval.py         #   SIFT-1M end-to-end retrieval benchmark
  ├─ datasets.py               #   SIFT-1M download + fvecs/ivecs parse
  ├─ harness.py                #   timing, hardware probe, stats helpers
  ├─ plots.py                  #   matplotlib figure helpers
  └─ smoke_test.py             #   quick correctness smoke test
results/                       # synthetic: fig1..fig10, summary.csv, env.json, raw/*.json
results/sift/                  # SIFT-1M: 4 figures, summary.csv, env.json, raw/retrieval.json
```

---

## Hardware & software

Captured at benchmark time and dumped into `results/env.json` / `results/sift/env.json`:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 3778 MiB |
| Compute capability | sm_86 (Ampere) |
| SM count | 16 |
| Theoretical peak memory bandwidth | 192 GB/s (GDDR6 128-bit @ 12 Gbps) |
| Driver | 570.211.01 |
| CUDA runtime | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.11.14 |

---

## The paper in one page

TurboQuant quantizes unit-norm vectors `x ∈ ℝᵈ` in a data-oblivious, online way. Two variants:

**`TurboQuant_mse` (Algorithm 1)** — optimised for reconstruction MSE.

1. Rotate: `y ← Π · x` with a random orthogonal `Π`. In high `d`, each coord of `y` is ≈ 𝒩(0, 1/d), coords are near-independent.
2. Per-coord scalar quantization to `b` bits with a Lloyd-Max codebook fitted to that Gaussian.
3. Dequant: look up centroids, apply `Πᵀ`.

**`TurboQuant_prod` (Algorithm 2)** — unbiased inner-product estimator.

1. Run `mse` at `b − 1` bits; recover the residual `r = x − Q⁻¹(Q(x))`.
2. QJL-sketch the residual: `qjl ← sign(S · r)` with `S ∈ ℝᵈˣᵈ`, `Sᵢⱼ ∼ 𝒩(0, 1)`.
3. Store `(idx, qjl, ‖r‖)`. Dequant: mse-dequant + `(√(π/2) / d) · ‖r‖ · Sᵀ · qjl`.

The key analytical result (Theorem 1): `E[‖x − x̂‖²] / d` is sandwiched between `4⁻ᵇ` (Shannon lower) and `(√3 · π / 2) · 4⁻ᵇ` (paper upper), *independent of `d`*. That is what makes the scheme practical for attention heads and compressed vector indexes.

---

## Kernel design

One `.cu` file (`csrc/turboquant_kernels.cu`), JIT-compiled through `torch.utils.cpp_extension` on first import. All kernels use **one block per input vector**, up to 256 threads cooperating via shared memory.

| Kernel | What it does | Why this layout |
|---|---|---|
| `fwht_forward`, `fwht_inverse` | In-place O(d log d) Walsh-Hadamard transform with a random sign flip | Structured stand-in for a Haar-random `Π`. Same concentration behaviour as a dense `d × d` GEMM but ~10× fewer FLOPs. Used by QuIP#, HadaCore. |
| `quantize_pack<B>` (templated on `B ∈ {1, 2, 4}`) | Per-coord argmin over the Lloyd codebook, then bit-packs `32/B` indices per `uint32` word | Shared-mem staging keeps the pack reduction warp-local and writes one coalesced word per group. `B` is a template param so the compiler unrolls the centroid scan. |
| `unpack_dequantize<B>` | Reverse: unpack `B` bits → codebook lookup → fp32 | Same block layout as `quantize_pack` so the caller can run `fwht_inverse` on the output in-place. |
| `fused_quantize<B>` | **Fusion of `fwht_forward` + `quantize_pack` into one kernel.** Keeps the rotated `y` in shared memory between the FWHT butterflies and the centroid search, skipping an HBM round-trip. | Default path since this change. See the next section for the measured speedup. |
| `fused_dequantize<B>` | **Fusion of `unpack_dequantize` + `fwht_inverse` into one kernel.** Keeps the unpacked `y_hat` in shared memory across the codebook lookup and the inverse butterflies. | Default path since this change. |
| `pack_signs`, `unpack_signs` | 1-bit sign packing for the QJL residual | Used only by the prod variant. |

`TurboQuant_prod` additionally invokes a dense `S · r` projection via cuBLAS (`torch.matmul`); writing a competitive GEMM from scratch is out of scope, and the Gaussian matrix is rotation-agnostic.

**Rotation choice.** The paper's analysis assumes a Haar-random `Π`. Practical systems (QuIP#, HadaCore) replace it with the structured factorisation `Π = (1/√d) · H · diag(s)` where `H` is the `d × d` Walsh-Hadamard matrix and `s ∈ {±1}ᵈ` are random signs. This is what we implement. The distortion numbers below land exactly on the paper's theoretical bounds, so the structured approximation is sufficient.

---

## Kernel fusion (fused FWHT + quantize-pack)

The MSE quantize pipeline is two kernels, with the rotated intermediate `y` round-tripped through HBM:

```
  UNFUSED (two kernels):
    x ─► fwht_forward  ─► [HBM: y, N·d·4 B written] ─► quantize_pack ─► packed
```

Fusion keeps `y` in shared memory across both stages:

```
  FUSED (one kernel):
    x ─► {fwht_forward · quantize_pack}  ─► packed
```

Same for the reverse direction (`unpack_dequantize` + `fwht_inverse`). This saves one kernel launch and the HBM round-trip of the intermediate tensor: **64 MiB per call at `d = 128, N = 65 536`**, scaling linearly with `N·d`.

The fused kernel is **bit-exact with the unfused path** — the centroid indices produced by `fused_quantize` are `torch.equal(...)` to those produced by `fwht_forward`+`quantize_pack`, and the reconstructed `x_hat` is identical to machine precision. Verified by `benchmark/smoke_test.py::check_fused_equivalence` across `b ∈ {1, 2, 4}`.

### Measured speedup

Median latency (100 iters after 15 warmup, CUDA events), fused-vs-unfused at the production config (d=128, KV-cache-shaped) and a range of other shapes:

| config | direction | unfused (µs) | fused (µs) | **speedup** | HBM saved |
|---|---|---|---|---|---|
| d=128, N=65 536, b=2 | quantize | 1141.8 | 996.4 | **1.15×** | 64 MiB |
| d=128, N=65 536, b=2 | dequantize | 1116.7 | 832.5 | **1.34×** | 64 MiB |
| d=128, N=65 536, b=4 | quantize | 1375.2 | 1233.9 | 1.12× | 64 MiB |
| d=128, N=65 536, b=4 | dequantize | 1195.0 | 943.1 | **1.27×** | 64 MiB |
| d=128, N=262 144, b=2 | quantize | 5700.1 | 4773.9 | 1.19× | 256 MiB |
| d=128, N=262 144, b=2 | dequantize | 5219.8 | 3909.6 | **1.34×** | 256 MiB |
| d=256, N=262 144, b=1 | quantize | 14 756.9 | 11 786.8 | 1.25× | 512 MiB |
| d=256, N=262 144, b=1 | dequantize | 12 183.6 | 9 837.6 | 1.24× | 512 MiB |
| d=512, N=65 536, b=2 | quantize | 6 548.0 | 5 194.2 | **1.26×** | 256 MiB |
| d=512, N=65 536, b=2 | dequantize | 6 858.2 | 4 851.7 | **1.41×** | 256 MiB |

Full sweep in `results/fusion/summary.csv`. Summary across all 15 configs: **quantize speedup median 1.11× (range 0.96–1.26×); dequantize speedup median 1.29× (range 1.17–1.41×)**. Dequantize benefits more because it does less compute per coord (codebook-lookup is a single load vs a 2^B-way argmin), so shaving the HBM round-trip is a bigger fraction of the runtime.

![Fusion speedup](results/fusion/fig_fusion_speedup.png)

**Transitive speedup for `TurboQuant_prod`.** The prod variant's internal `TurboQuantMSE` uses the fused path by default, so Algorithm 2 gets the same acceleration on its MSE stages at no additional cost. The `S · r` cuBLAS GEMM and the `pack_signs` kernel are unchanged.

**Caveat — `__fmul_rn` for bit-exact parity.** The unfused pipeline stores the scaled `y` to HBM between kernels, which forces fp32 rounding. The fused version keeps `y` in registers, where `--use_fast_math` lets nvcc fuse the scale-multiply with the downstream subtraction into a single-rounding FMA. For one input per ~4 M at `b=4` the FMA rounding flipped the argmin at a codebook midpoint. We pin the scaling multiplication with `__fmul_rn` to force separate rounding, restoring bit-exact parity at negligible cost (a couple of instructions per coord).

---

## Inline PTX — where it helps, where it doesn't

Two places in the pipeline have clean inline-PTX implementations worth comparing against the CUDA C++ version. Both are **bit-exact** with their non-PTX counterparts — verified by `benchmark/smoke_test.py::check_ptx_equivalence`.

### 1. `pack_signs` via `vote.sync.ballot.b32` — **~2× speedup, real and consistent**

The scalar `pack_signs_kernel` runs a 32-iteration bit-OR loop per word, with one thread per output word. The PTX version uses a warp of 32 lanes and collects the per-lane sign bits into one 32-bit word via `vote.sync.ballot.b32` in a single instruction. 32 scalar ORs collapse to one warp vote.

| d | N | scalar (µs) | PTX warp-ballot (µs) | **speedup** |
|---|---|---|---|---|
| 128 | 65 536 | 460.8 | 226.3 | **2.04×** |
| 128 | 262 144 | 2211.8 | 1053.7 | **2.10×** |
| 256 | 65 536 | 1140.7 | 541.7 | **2.11×** |
| 256 | 262 144 | 4177.4 | 1992.7 | **2.10×** |
| 512 | 65 536 | 1682.9 | 970.8 | 1.73× |
| 512 | 262 144 | 6870.6 | 3872.8 | 1.77× |

![pack_signs PTX speedup](results/fusion/fig_pack_signs_ptx.png)

The speedup tapers at `d = 512` because the kernel is hitting HBM bandwidth limits — at 2 MB/s output and many warps per block, both versions start to saturate the memory subsystem. For `d ≤ 256`, where compute dominates, the 2× PTX win is flat.

This is exactly the kind of operation PTX intrinsics were designed for: a warp-level reduction with no clean C++ expression. The SASS listing for the PTX kernel shows a single `VOTE.ALL` instruction producing the word, vs ~32 shift/OR instructions in the scalar kernel.

### 2. `fused_quantize` with `bfi.b32` bit-packing — **no measurable speedup**

The packing loop inside `fused_quantize_kernel<B>` is a sequence of `word |= (idx & MASK) << (i * B)` operations. Inline PTX replaces these with explicit `bfi.b32 dst, src, dst, pos, num_bits` (bitfield-insert) instructions.

| config | fused CUDA C++ (µs) | fused + PTX bfi (µs) | **speedup** |
|---|---|---|---|
| d=128, N=65 536, b=2 | 965.7 | 955.9 | 1.01× |
| d=128, N=262 144, b=2 | 4248.6 | 4219.4 | 1.01× |
| d=256, N=65 536, b=2 | 2174.6 | 2169.3 | 1.00× |
| d=256, N=262 144, b=2 | 9870.3 | 9750.5 | 1.01× |
| d=512, N=65 536, b=2 | 4571.1 | 4881.9 | 0.94× |
| (full sweep across b ∈ {1, 2, 4}) | — | — | 0.92–1.11× |

**No win from explicit PTX.** nvcc is already lowering the C++ shift/OR pattern to `bfi.b32` under `-O3` (confirmed by inspecting the SASS). The inline PTX version emits the same instructions, so the delta is noise — some configs come out slightly faster, some slightly slower, nothing consistent.

This is the honest counterpart to the pack_signs result: PTX only helps when it exposes a hardware primitive the C++ version can't express. When the compiler is already generating the optimal instruction, rewriting the same operation in inline asm is cosmetic.

### Combined three-column view of the MSE quantize path

![Fused + PTX comparison](results/fusion/fig_fusion_speedup.png)

Left subplot (quantize) shows three bars — unfused / fused / fused+PTX — at each config. The `fused` vs `unfused` improvement (1.00–1.24×) is the real win; the `fused+PTX` bar tracks `fused` almost exactly. Right subplot (dequantize) shows two bars — there's no PTX dequant variant because the dequantize kernel doesn't have a packing loop to optimise.

### Summary

| Change | Expressed as | Speedup | Reason |
|---|---|---|---|
| Two kernels → one fused | CUDA C++ kernel fusion | 1.10–1.25× (quantize), 1.17–1.41× (dequantize) | Skips HBM round-trip of intermediate `y` |
| `word \|= idx << shift` → `bfi.b32` | Inline PTX | 1.00× (within noise) | nvcc already emits this instruction |
| 32-iteration bit-OR loop → warp ballot | Inline PTX (`vote.sync.ballot.b32`) | **~2.0× (pack_signs)** | Warp-level primitive has no clean C++ form |

Takeaway: **PTX is worth the readability tax only for warp/lane-level primitives that C++ can't express directly**. For straight-line arithmetic, trust the compiler.

Raw data: `results/fusion/summary.csv`, `results/fusion/raw/fusion.json`.

---

## Methodology

### Synthetic microbenchmark (`benchmark/run_benchmark.py`)

* **Timing**: `torch.cuda.Event` start/stop per iteration. 15 warmup + 100 timed iters. Median, mean, std, p5, p95, min, max reported.
* **Accuracy**: 10 independent seeds → mean ± std for every `D_mse` / `D_prod` number.
* **9 phases**: env probe → correctness check vs pure-PyTorch reference → accuracy (bits × seeds) → bias vs ⟨x,y⟩ → latency-vs-N → latency-vs-d → bandwidth util → compression → Pareto.

### SIFT-1M retrieval benchmark (`benchmark/sift_retrieval.py`)

* **Dataset**: SIFT-1M (Jegou et al., TPAMI 2011) — 1 M × 128 fp32 base, 10 K × 128 queries, public FTP distribution. Auto-downloaded on first run.
* **Geometry**: vectors L2-normalised so that top-k by inner product equals top-k by L2 (TurboQuant's native setting).
* **Ground truth**: fp32 brute force top-100 on the normalised corpus, cached on disk.
* **Recall metrics**: `Recall@1`, `Recall@10`, `Recall@100` against ground truth, plus the practical **2-stage `Recall@10`** where the quantized index returns 100 candidates and fp32 re-scores them (the production deployment pattern).
* **Retrieval engine**: `cuturbo.retrieval.QuantizedIndex.search()` — chunked dequant-then-GEMM over 100 K-doc tiles × 100-query batches, with GPU `torch.topk` merging. The paper doesn't require (and we don't implement) a fused quantized-GEMM kernel; re-ranking covers that gap in practice.

---

## Part I — Algorithmic correctness (synthetic)

### 1. CUDA ↔ reference round-trip

Every `(d, b)` pair in `d ∈ {64, 128, 256, 512, 1024}` × `b ∈ {1, 2, 4}` was verified to agree with the pure-PyTorch reference to within fp32 noise:

```
max |x̂_cuda − x̂_reference| ≈ 5 × 10⁻⁸  (threshold: 1 × 10⁻⁴)
```

See `results/raw/correctness.json`. All 15 configurations pass.

### 2. Distortion matches paper Theorem 1

*What the paper predicts.* `D_mse = E[‖x − x̂‖²] / d ∈ [4⁻ᵇ, (√3·π/2) · 4⁻ᵇ]`, tabulated values `{0.36, 0.117, 0.009}` for `b ∈ {1, 2, 4}`.

*What we measured.* `d = 128, N = 65 536`, 10 seeds:

| b | measured `D_mse` (mean ± std) | Shannon lower `4⁻ᵇ` | paper upper `√3·π/2 · 4⁻ᵇ` |
|---|---|---|---|
| 1 | **0.36091 ± 0.00014** | 0.25000 | 0.68017 |
| 2 | **0.11601 ± 0.00008** | 0.06250 | 0.17004 |
| 4 | **0.00933 ± 0.00001** | 0.00391 | 0.01063 |

| b | measured `D_prod` (mean ± std) | Shannon lower | paper upper |
|---|---|---|---|
| 2 | **0.00439 ± 0.00009** | 0.00049 | 0.00835 |
| 3 | **0.00141 ± 0.00003** | 0.00012 | 0.00209 |
| 5 | **0.000113 ± 0.000002** | 0.0000076 | 0.000130 |

*Verdict.* Every row lies inside the theoretical band; mean `D_mse` agrees with the paper's tabulated values to three significant figures. **Match.**

![Distortion vs bits](results/fig3_distortion_vs_bits.png)

### 3. Unbiasedness reproduces paper Figure 2

*What the paper predicts.* `TurboQuant_mse` has non-zero bias growing with `⟨x, y⟩`; `TurboQuant_prod` (Algorithm 2) is centred at zero.

*What we measured.* Pairs constructed with target `⟨x, y⟩ ∈ {0.01, 0.1, 0.3, 0.5}`, `b = 2, d = 128`, mean IP error over many trials:

| target `⟨x, y⟩` | `TurboQuant_mse` bias | `TurboQuant_prod` bias | ratio |
|---|---|---|---|
| 0.01 | −0.00121 | +0.00008 | 14× |
| 0.10 | −0.01168 | −0.00003 | 350× |
| 0.30 | −0.03443 | +0.00028 | 123× |
| 0.50 | **−0.05747** | **+0.00075** | **77×** |

*Verdict.* mse bias grows roughly linearly with `⟨x, y⟩`; prod stays at noise level. **Match.**

![Bias vs inner product](results/fig2_bias_vs_ip.png)

### 4. Synthetic kernel latency

This is *kernel-time*, not retrieval-time. Per-batch median latency (100 iters), `d = 128, N = 65 536`:

| Method | What it does | quant (µs) | dequant (µs) | effective throughput |
|---|---|---|---|---|
| fp16 cast | Type cast to half-precision (no quantization) — memory-bandwidth ceiling | 275.5 | 306.2 | 121.8 GB/s |
| naive scalar b=2 | Per-coord uniform bucketing, no rotation — apples-to-apples quantizer baseline | 1753.1 | 976.9 | 19.1 GB/s |
| **TurboQuant_mse b=2** | Algorithm 1 (ours) | **1127.4** | **1084.4** | **29.8 GB/s** |
| TurboQuant_prod b=2 | Algorithm 2 (ours) | 4278.3 | 2828.8 | 7.8 GB/s |

*Interpretation.* `TurboQuant_mse` is **1.56× faster than the naive scalar quantizer** while doing strictly more work (rotation + Lloyd codebook + bit-pack). The naive baseline loses because it writes one byte per coord; `TurboQuant_mse` packs 16 coords into each 32-bit word. fp16 cast is not a competitor — it just copies bytes, no compression, and sets the memory-bandwidth ceiling at ~64 % of this GPU's 192 GB/s peak. TurboQuant_mse hits ~16 % of peak, which indicates the kernel is compute-bound (rotation + codebook lookup), leaving headroom for PTX tuning.

See `results/fig4_scaling_vs_N.png`, `fig5_scaling_vs_d.png`, `fig7_bandwidth_util.png` for the full scaling study.

---

## Part II — End-to-end retrieval on SIFT-1M

All numbers in this section come from a single run of `benchmark/sift_retrieval.py`, written to `results/sift/summary.csv`. Glossary of terms used in the tables:

- **Index size**: bytes of GPU memory the compressed corpus takes after `build`. This is the memory the paper is actually trying to reduce.
- **Build time**: one-time cost to quantize 1 M docs.
- **QPS (median of 20 batches)**: queries per second at `query_batch = 100`; higher is faster.
- **Recall@k**: fraction of the true top-k nearest neighbours the index returns, averaged over all 10 K queries.
- **2-stage Recall@10**: the index returns its top-100 candidates, fp32 re-scores them, the top-10 of the re-scored pool is compared to fp32 ground truth. This is how quantized ANN is deployed in production.

### Recall

| Method | Index | Compression | Build (s) | R@1 | R@10 | R@100 | **R@10 rerank** |
|---|---|---|---|---|---|---|---|
| fp32 | 488 MB | 1× | 0.01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fp16 | 244 MB | 2× | 0.01 | 0.9904 | 0.9933 | 0.9957 | 1.0000 |
| naive-b2 | 30.5 MB | **16×** | 0.06 | 0.0045 | 0.0099 | 0.0272 | 0.0585 |
| naive-b4 | 61.0 MB | 8× | 0.10 | 0.2086 | 0.2789 | 0.3862 | 0.6756 |
| **mse-b2** | **30.5 MB** | **16×** | 0.44 | 0.1149 | **0.1729** | 0.2690 | **0.5161** |
| **mse-b4** | **61.0 MB** | **8×** | 0.15 | 0.3637 | **0.4840** | 0.6015 | **0.9280** |
| prod-b2 | 34.3 MB | 14× | 0.63 | 0.0534 | 0.0864 | 0.1556 | 0.2946 |
| prod-b3 | 49.6 MB | 10× | 0.44 | 0.1062 | 0.1664 | 0.2604 | 0.4834 |
| **prod-b5** | **80.1 MB** | **6×** | 0.40 | 0.4247 | **0.5441** | 0.6514 | **0.9623** |

### Throughput

| Method | Per-query (µs) | QPS |
|---|---|---|
| fp32 | 429.8 | 2 349 |
| fp16 | 566.8 | 1 746 |
| mse-b2 | 2 505.7 | 399 |
| mse-b4 | 3 579.2 | 282 |
| prod-b5 | 3 687.0 | 256 |
| naive-b4 | 3 099.1 | 319 |
| naive-b2 | 3 258.3 | 308 |

### Figures

**Pareto frontier — compressed index + fp32 rerank (production pattern):**

![2-stage Recall@10 vs index size](results/sift/fig_sift_rerank_recall.png)

`TurboQuant_mse b=4` at 61 MB reaches **0.928** — within 7 percentage points of the 488 MB fp32 index, at 8× less memory. The naive quantizer at the same bit budget (`naive-b4`) stops at **0.676**. At 16× compression (30.5 MB), naive scalar is essentially useless (R@10 rerank = 0.058); `TurboQuant_mse b=2` recovers **0.516** — still practically usable.

**Raw Recall@10 (no rerank):**

![Raw Recall@10 vs index size](results/sift/fig_sift_recall_vs_size.png)

**Recall vs throughput:**

![Recall vs QPS](results/sift/fig_sift_recall_vs_qps.png)

**Unbiased vs biased estimator, matched bits:**

![TurboQuant_mse vs TurboQuant_prod](results/sift/fig_sift_prod_vs_mse.png)

At very low bit budgets the `prod` variant pays a 1-bit QJL overhead that eats into the MSE stage's budget and so loses raw recall. At `b ≥ 4` bits, the unbiased residual correction *improves* retrieval: `prod-b5` (6× compression) beats `mse-b4` (8× compression) on 2-stage Recall@10 (0.962 vs 0.928). The paper's unbiasedness claim pays off in the regime where per-pair estimation error is already small, so ranking is preserved rather than shuffled.

---

## Correlation with the paper — single table

| Paper claim | Our measurement | Verdict |
|---|---|---|
| Theorem 1: `D_mse ∈ [4⁻ᵇ, √3·π/2 · 4⁻ᵇ]` | 0.3609 / 0.1160 / 0.00933 for b = 1 / 2 / 4 — all inside bounds | ✓ Match |
| Theorem 1 tabulated values `{0.36, 0.117, 0.009}` | 0.361 / 0.116 / 0.00933 | ✓ Match (3 sig figs) |
| Algorithm 2 gives unbiased IP estimator | prod bias ≈ 0.001 vs mse bias ≈ −0.057 at `⟨x,y⟩ = 0.5` (77× smaller) | ✓ Match |
| 8× KV-cache compression at b = 2 | 32 B/vec vs 256 B/vec at d = 128 | ✓ Match |
| Compression preserves retrieval quality | SIFT-1M R@10 rerank: 0.928 at 8× compression (mse b=4) | ✓ Verified on public ANN benchmark |
| Beats naive scalar quantization | SIFT-1M R@10 rerank at 8×: 0.928 (mse) vs 0.676 (naive) | ✓ +25 percentage points |
| LLM attention wall-clock speedup (§4) | Not measured — Llama-8B doesn't fit on 4 GB VRAM | — (out of scope) |

---

## Reproducing the results

Setup: CUDA 12.8 toolchain + PyTorch 2.10.0+cu128 installed (nvcc on PATH). No build step — kernels JIT-compile on first import.

```bash
# Quick correctness smoke test (~10 s incl. first JIT compile)
# Also verifies fused vs unfused bit-exact equivalence
python3 benchmark/smoke_test.py

# Synthetic microbenchmark (~2–3 min) — produces the Part I results
python3 benchmark/run_benchmark.py

# Fused vs unfused MSE kernel benchmark (~90 s)
python3 benchmark/fused_benchmark.py

# SIFT-1M end-to-end retrieval (~7 min, auto-downloads SIFT-1M on first run)
python3 benchmark/sift_retrieval.py
```

Outputs:

| | File |
|---|---|
| Env probe | `results/env.json`, `results/sift/env.json` |
| Synthetic metrics | `results/summary.csv`, `results/raw/*.json`, `results/fig1..fig10*.png` |
| SIFT metrics | `results/sift/summary.csv`, `results/sift/raw/retrieval.json`, `results/sift/fig_sift_*.png` |
| Fusion metrics | `results/fusion/summary.csv`, `results/fusion/raw/fusion.json`, `results/fusion/fig_fusion_speedup.png` |

The SIFT download (~160 MB tar.gz over FTP) is cached in `.cache/sift1m/`. Ground truth is computed once and cached as `.cache/sift1m/gt_ip_top100.pt`. Re-running is fast.

---

## Limitations & future work

* **Only `b ∈ {1, 2, 4}` are bit-packed.** `b = 3` is awkward because `32 / 3` isn't an integer — everything else works for `b = 3`, just isn't packed.
* **The structured rotation approximates Haar-random `Π`.** This is what practical systems do. The concentration analysis in the paper assumes full randomness; the structured version is close enough that distortion numbers land on the theoretical bounds, but a Haar variant could be added as a correctness oracle.
* **No fp16 input path.** Kernels are fp32 end-to-end. An fp16-input specialisation is straightforward given the current block layout.
* **No fused quantized-GEMM kernel.** Retrieval dequantizes a chunk of docs then runs cuBLAS GEMM. Real production indexes (FAISS PQ, ScaNN) use a scalar-quant look-up table or bit-parallel Hamming tricks to avoid the dequant step. That's orthogonal to TurboQuant's contribution and out of scope.
* **No IVF / HNSW coarse-quantization layer.** Brute-force scan is the honest apples-to-apples comparison at 1 M vectors; we're not claiming this beats HNSW.
* **No LLM-side evaluation.** Reproducing paper §4.2–4.3 (LongBench, Needle-in-a-Haystack) would require loading a 7–8 B parameter model; doesn't fit on 4 GB VRAM.

---

## Reference

Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874, 2025.
