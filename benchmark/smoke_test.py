"""Quick correctness smoke test: CUDA path vs PyTorch reference path."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
import torch

from cuturbo.api import TurboQuantMSE, TurboQuantProd
from cuturbo import reference as ref
from cuturbo.codebook import build_codebook, paper_codebook


def check_codebooks():
    # Paper says b=1 centroids are ±√(2/π)/√d ≈ ±0.7979/√d  and b=2 are ±0.4528/√d, ±1.51/√d
    cb1 = paper_codebook(1, d=128)
    cb2 = paper_codebook(2, d=128)
    print("b=1 centroids × √d:", [f"{c*math.sqrt(128):+.4f}" for c in cb1])
    print("b=2 centroids × √d:", [f"{c*math.sqrt(128):+.4f}" for c in cb2])


def check_fwht(d=128, N=4, seed=7):
    torch.manual_seed(seed)
    x = torch.randn(N, d, device="cuda", dtype=torch.float32)

    signs_cuda = torch.sign(torch.randn(d, device="cuda"))  # ±1
    signs_cuda = torch.where(signs_cuda == 0, torch.ones_like(signs_cuda), signs_cuda)

    from cuturbo.ext import get_ext
    ext = get_ext()

    # CUDA forward
    y_cuda = ext.fwht_forward(x.contiguous(), signs_cuda)
    # Reference forward via explicit Hadamard matmul
    H = ref.hadamard_matrix(d, "cuda")
    y_ref = (x * signs_cuda) @ H.T / math.sqrt(d)

    err = (y_cuda - y_ref).abs().max().item()
    print(f"FWHT forward  max abs err (cuda vs ref): {err:.2e}")
    assert err < 1e-3, f"FWHT forward mismatch: {err}"

    # Inverse round-trip
    x_back = ext.fwht_inverse(y_cuda, signs_cuda)
    err2 = (x - x_back).abs().max().item()
    print(f"FWHT roundtrip  max abs err: {err2:.2e}")
    assert err2 < 1e-3, f"FWHT roundtrip mismatch: {err2}"


def check_mse(d=128, N=256, b=2, seed=3):
    torch.manual_seed(seed)
    x = torch.randn(N, d, device="cuda", dtype=torch.float32)
    x = x / x.norm(dim=1, keepdim=True)       # unit norm, as paper assumes

    tq = TurboQuantMSE(d, b, "cuda", seed=11)
    code = tq.quantize(x)
    x_hat_cuda = tq.dequantize(code)

    # Reference with matching Π parameters
    signs = tq.signs
    H = ref.hadamard_matrix(d, "cuda")
    cb = tq.codebook
    ref_code = ref.quantize_mse_ref(x, b, signs, H, cb)
    x_hat_ref = ref.dequantize_mse_ref(ref_code, signs, H, cb)

    err = (x_hat_cuda - x_hat_ref).abs().max().item()
    print(f"TurboQuant_mse  b={b}  max abs err (cuda vs ref): {err:.2e}")
    assert err < 5e-3, f"MSE variant mismatch: {err}"

    mse = (x - x_hat_cuda).pow(2).sum(dim=1).mean().item()
    # Paper theorem 1 bound (per-vector MSE) and lower bound — for x on S^{d-1}
    upper = math.sqrt(3) * math.pi / 2.0 * (4 ** -b)
    lower = 4 ** -b
    print(f"  measured  D_mse={mse:.4f}   lower={lower:.4f}   upper={upper:.4f}")
    assert lower * 0.5 < mse < upper * 2.0, "MSE out of expected theoretical band"


def check_fused_equivalence(d=128, N=256, b=2, seed=3):
    """Fused kernel must produce bit-exact packed indices and identical x_hat
    as the unfused two-kernel path (same FWHT, same rounding, same packing)."""
    torch.manual_seed(seed)
    x = torch.randn(N, d, device="cuda", dtype=torch.float32)
    x = x / x.norm(dim=1, keepdim=True)

    unfused = TurboQuantMSE(d, b, "cuda", seed=11, fused=False)
    fused   = TurboQuantMSE(d, b, "cuda", seed=11, fused=True)
    # Parameters must match — same seed → same signs and codebook
    assert torch.equal(unfused.signs, fused.signs)
    assert torch.equal(unfused.codebook, fused.codebook)

    code_unf = unfused.quantize(x)
    code_fus = fused.quantize(x)
    assert torch.equal(code_unf.packed, code_fus.packed), \
        f"fused quantize packed indices differ at b={b}"

    xh_unf = unfused.dequantize(code_unf)
    xh_fus = fused.dequantize(code_fus)
    max_err = (xh_unf - xh_fus).abs().max().item()
    print(f"fused vs unfused  b={b}: packed bit-exact ✓  dequant max abs err = {max_err:.2e}")
    assert max_err < 1e-5, f"fused dequant mismatch at b={b}: {max_err}"


def check_ptx_equivalence(seed=7):
    """fused_quantize_ptx must be bit-exact with fused_quantize (same math,
    different instruction lowering for the pack loop). Likewise pack_signs_ptx
    and pack_signs."""
    from cuturbo.ext import get_ext
    from cuturbo.codebook import build_codebook

    torch.manual_seed(seed)
    ext = get_ext()

    # fused_quantize_ptx vs fused_quantize
    d, N = 128, 4096
    x = torch.randn(N, d, device="cuda", dtype=torch.float32)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    signs = torch.where(torch.randn(d, device="cuda") >= 0, 1.0, -1.0).float()
    for b in (1, 2, 4):
        cb = build_codebook(b, d, "cuda")
        p_cuda = ext.fused_quantize(x, signs, cb, b)
        p_ptx  = ext.fused_quantize_ptx(x, signs, cb, b)
        assert torch.equal(p_cuda, p_ptx), f"fused_quantize_ptx mismatch at b={b}"
        print(f"fused_quantize_ptx  b={b}: bit-exact ✓")

    # pack_signs_ptx vs pack_signs across shapes
    for d_ in (64, 128, 256, 512):
        xx = torch.randn(1024, d_, device="cuda", dtype=torch.float32)
        pk_c = ext.pack_signs(xx)
        pk_p = ext.pack_signs_ptx(xx)
        assert torch.equal(pk_c, pk_p), f"pack_signs_ptx mismatch at d={d_}"
        print(f"pack_signs_ptx     d={d_}: bit-exact ✓")


def check_prod(d=128, N=256, b=2, seed=3):
    torch.manual_seed(seed)
    x = torch.randn(N, d, device="cuda", dtype=torch.float32)
    x = x / x.norm(dim=1, keepdim=True)
    y = torch.randn(N, d, device="cuda", dtype=torch.float32)
    y = y / y.norm(dim=1, keepdim=True)

    tq = TurboQuantProd(d, b, "cuda", seed=19)
    code = tq.quantize(x)
    x_hat = tq.dequantize(code)

    ip_true = (x * y).sum(dim=1)
    ip_est  = (x_hat * y).sum(dim=1)
    bias = (ip_est - ip_true).mean().item()
    var  = (ip_est - ip_true).pow(2).mean().item()
    print(f"TurboQuant_prod b={b}: mean IP err (bias) = {bias:+.5f}   MSE(IP) = {var:.5f}")


if __name__ == "__main__":
    print("=== Codebook sanity ===")
    check_codebooks()
    print()
    print("=== FWHT correctness ===")
    check_fwht()
    print()
    print("=== TurboQuant_mse correctness + bound check ===")
    for b in (1, 2, 4):
        check_mse(b=b)
    print()
    print("=== Fused vs unfused kernel equivalence ===")
    for b in (1, 2, 4):
        check_fused_equivalence(b=b)
    print()
    print("=== PTX kernel equivalence ===")
    check_ptx_equivalence()
    print()
    print("=== TurboQuant_prod unbiasedness ===")
    for b in (2, 3, 4):
        # prod at total bit-width b uses mse at b-1 ∈ {1, 2, 3}
        # b-1=3 is not in our packed set — skip
        if b - 1 in (1, 2, 4):
            check_prod(b=b)
    print()
    print("All smoke tests passed.")
