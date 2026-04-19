import os
from pathlib import Path
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parent.parent
_CSRC = _ROOT / "csrc" / "turboquant_kernels.cu"
_BUILD = _ROOT / ".build"
_BUILD.mkdir(exist_ok=True)

_ext = None


def get_ext():
    global _ext
    if _ext is not None:
        return _ext

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    _ext = load(
        name="cuturbo_kernels",
        sources=[str(_CSRC)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-Xptxas", "-O3",
            "-lineinfo",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(_BUILD),
        verbose=False,
    )
    return _ext
