"""
Benchmark primitives: timing, statistics, hardware probe, baseline implementations.

Kept separate from run_benchmark.py so that the main driver reads as a narrative.
"""
from __future__ import annotations

import dataclasses
import json
import math
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hardware / environment probe (for the repro dump)
# ---------------------------------------------------------------------------

@dataclass
class EnvInfo:
    gpu_name: str
    gpu_vram_mib: int
    gpu_capability: str
    cuda_runtime: str
    driver: str
    torch_version: str
    python_version: str
    platform: str
    sm_count: int
    theoretical_bw_gbps: float    # RTX 3050 Laptop ≈ 192 GB/s (GDDR6 128-bit @ 12 Gbps)


def detect_env(theoretical_bw_gbps: float = 192.0) -> EnvInfo:
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)

    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        drv = "unknown"

    return EnvInfo(
        gpu_name=props.name,
        gpu_vram_mib=int(props.total_memory / 2**20),
        gpu_capability=f"sm_{props.major}{props.minor}",
        cuda_runtime=torch.version.cuda or "unknown",
        driver=drv,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        sm_count=props.multi_processor_count,
        theoretical_bw_gbps=theoretical_bw_gbps,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    """All times in microseconds. per_iter_us is the full array (for box plots)."""
    per_iter_us: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        a = np.asarray(self.per_iter_us, dtype=np.float64)
        return {
            "median_us": float(np.median(a)),
            "mean_us":   float(np.mean(a)),
            "std_us":    float(np.std(a)),
            "min_us":    float(np.min(a)),
            "max_us":    float(np.max(a)),
            "p05_us":    float(np.percentile(a, 5)),
            "p95_us":    float(np.percentile(a, 95)),
            "iters":     int(a.size),
        }


def time_cuda(fn: Callable[[], None], warmup: int = 15, iters: int = 100) -> TimingStats:
    """Robust CUDA timing using events.

    - Calls `fn()` `warmup` times to warm caches and pipeline.
    - Calls `fn()` `iters` times, measuring each individually with
      cudaEvent start/stop.
    - Returns all per-iteration times (µs) so caller can form distributions.
    """
    # Make sure any previous work has flushed before we begin
    torch.cuda.synchronize()

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops  = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()

    return TimingStats(
        per_iter_us=[starts[i].elapsed_time(stops[i]) * 1000.0 for i in range(iters)]
    )


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def measure_peak_vram_bytes(fn: Callable[[], None]) -> int:
    """Run fn once and report peak allocated VRAM during the call."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


# ---------------------------------------------------------------------------
# Baselines used throughout the benchmark
# ---------------------------------------------------------------------------

class FP16Baseline:
    """Store KV as fp16 (current production practice). Zero extra computation."""
    def __init__(self, d: int): self.d = d
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float16)
    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return q.to(torch.float32)
    def payload_bytes(self, N: int) -> int:
        return 2 * N * self.d


class UniformScalarBaseline:
    """Uniform scalar quantizer to b bits, no rotation. Shows the advantage of
    TurboQuant's random rotation + Lloyd codebook."""
    def __init__(self, d: int, b: int, device):
        self.d, self.b, self.K = d, b, 1 << b
        self.device = device
        # range ±3σ with σ=1/√d  (matches unit-norm vector statistics)
        self.range = 3.0 / math.sqrt(d)
        self.step  = 2.0 * self.range / self.K
        self.centroids = (torch.arange(self.K, device=device, dtype=torch.float32)
                          * self.step - self.range + self.step / 2)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        q = (x + self.range) / self.step
        return q.clamp_(0, self.K - 1).round_().to(torch.uint8)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return self.centroids[q.long()]

    def payload_bytes(self, N: int) -> int:
        # one byte per coord — not bit-packed (that's the inefficiency)
        return N * self.d


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def to_gbps(bytes_processed: int, median_us: float) -> float:
    """Return throughput in GB/s (1e9 bytes/s)."""
    if median_us <= 0:
        return 0.0
    seconds = median_us * 1e-6
    return (bytes_processed / 1e9) / seconds


def ip_errors(x: torch.Tensor, x_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x_hat * y).sum(dim=1) - (x * y).sum(dim=1))


def dump_json(path, obj):
    """Dump obj as JSON, converting numpy/torch scalars."""
    def _default(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if dataclasses.is_dataclass(o): return dataclasses.asdict(o)
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
