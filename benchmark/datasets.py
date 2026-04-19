"""
SIFT-1M loader (Jegou et al., TPAMI 2011).

Provides the canonical ANN benchmark:
    - 1 000 000 base vectors × 128 fp32
    - 10 000 query vectors × 128 fp32
    - 10 000 × 100 int32 groundtruth (true top-100 nearest neighbours by L2)

Idempotent download; fvecs / ivecs parsing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


def read_fvecs(path: Path) -> np.ndarray:
    """Parse fvecs format: for each vector, int32 dim header then `dim` float32 values."""
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = int(raw[0])
    stride = dim + 1
    assert raw.size % stride == 0, f"fvecs size {raw.size} not divisible by stride {stride}"
    n = raw.size // stride
    mat = raw.reshape(n, stride)[:, 1:].copy()
    return mat.view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    """Parse ivecs format: for each vector, int32 dim header then `dim` int32 values."""
    raw = np.fromfile(str(path), dtype=np.int32)
    dim = int(raw[0])
    stride = dim + 1
    assert raw.size % stride == 0
    n = raw.size // stride
    return raw.reshape(n, stride)[:, 1:].copy()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    # Python's urllib has a bug with some FTP extended-passive-mode responses.
    # Use curl, which handles passive FTP reliably, falling back to urllib.
    tmp = dst.with_suffix(dst.suffix + ".partial")
    if shutil.which("curl"):
        cmd = ["curl", "-fSL", "--ftp-pasv", "--connect-timeout", "30", "-o", str(tmp), url]
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.rename(dst); return
        if tmp.exists():
            tmp.unlink()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"    {total / 2**20:7.1f} MiB", end="\r")
    print()
    tmp.rename(dst)


def download_sift1m(cache_dir: Path = Path(".cache/sift1m")) -> Path:
    cache_dir = Path(cache_dir)
    required = {
        "sift_base.fvecs":        cache_dir / "sift_base.fvecs",
        "sift_query.fvecs":       cache_dir / "sift_query.fvecs",
        "sift_groundtruth.ivecs": cache_dir / "sift_groundtruth.ivecs",
    }
    if all(p.exists() for p in required.values()):
        return cache_dir

    tgz = cache_dir / "sift.tar.gz"
    if not tgz.exists():
        _download(SIFT_URL, tgz)

    print(f"  extracting {tgz}")
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if name in required:
                m.name = name
                tf.extract(m, path=cache_dir)

    missing = [n for n, p in required.items() if not p.exists()]
    if missing:
        raise RuntimeError(f"SIFT archive extracted but missing: {missing}")
    return cache_dir


def load_sift1m(
    cache_dir: Path = Path(".cache/sift1m"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (base, query, gt) as CPU tensors. base/query fp32, gt int64."""
    cache_dir = download_sift1m(cache_dir)
    base  = read_fvecs(cache_dir / "sift_base.fvecs")
    query = read_fvecs(cache_dir / "sift_query.fvecs")
    gt    = read_ivecs(cache_dir / "sift_groundtruth.ivecs")
    assert base.shape  == (1_000_000, 128), base.shape
    assert query.shape == (10_000,    128), query.shape
    assert gt.shape    == (10_000,    100), gt.shape
    return (
        torch.from_numpy(base),
        torch.from_numpy(query),
        torch.from_numpy(gt.astype(np.int64)),
    )


if __name__ == "__main__":
    import sys
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".cache/sift1m")
    b, q, g = load_sift1m(cache)
    print(f"base  {tuple(b.shape)} dtype={b.dtype}")
    print(f"query {tuple(q.shape)} dtype={q.dtype}")
    print(f"gt    {tuple(g.shape)} dtype={g.dtype}")
