// CUTurbo — TurboQuant CUDA kernels
//
// Implements the GPU primitives for TurboQuant (Zandieh et al., 2025):
//   - FWHT-based structured random rotation (forward + inverse)
//   - b-bit scalar quantization with packed index storage
//   - b-bit dequantization with unpacking + codebook lookup
//   - 1-bit QJL sign packing / unpacking (for the inner-product variant)
//
// Target: sm_86 (RTX 3050 Laptop). All kernels are row-parallel:
// one CUDA block per input vector, threads cooperate within a row.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// FWHT-based random rotation
// ---------------------------------------------------------------------------
//
// We approximate a Haar-random rotation Π with a structured rotation
//   Π · x  =  (1/√d) · H · diag(s) · x
// where H is the dxd Walsh-Hadamard matrix and s ∈ {±1}^d is a random sign
// vector. This is O(d log d) instead of O(d²). Concentration properties of
// coordinates of Πx (used by TurboQuant's analysis) hold in high dimension.
//
// Forward:  apply signs first, then butterflies, then scale by 1/√d.
// Inverse:  butterflies first, then scale by 1/√d, then apply signs.
// (H is symmetric; H H = d·I, so Πᵀ = diag(s) · (1/√d) H.)

template <bool SIGNS_FIRST>
__global__ void fwht_kernel(
    const float* __restrict__ x,
    const float* __restrict__ signs,   // (d,)
    float* __restrict__ y,
    int d,
    float scale)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int n   = blockIdx.x;

    // Stage 0: load into shared memory
    for (int c = tid; c < d; c += blockDim.x) {
        float v = x[n * d + c];
        if (SIGNS_FIRST) v *= signs[c];
        sdata[c] = v;
    }
    __syncthreads();

    // In-place Walsh-Hadamard butterflies
    // for h in {1, 2, 4, ..., d/2}:
    //   pair up coords at distance h, replace (a,b) with (a+b, a-b)
    for (int h = 1; h < d; h <<= 1) {
        int pairs = d >> 1;  // there are d/2 butterflies per stage
        for (int p = tid; p < pairs; p += blockDim.x) {
            int i = ((p / h) << 1) * h + (p % h);
            int j = i + h;
            float a = sdata[i];
            float b = sdata[j];
            sdata[i] = a + b;
            sdata[j] = a - b;
        }
        __syncthreads();
    }

    // Stage last: scale and (for inverse) apply signs
    for (int c = tid; c < d; c += blockDim.x) {
        float v = sdata[c] * scale;
        if (!SIGNS_FIRST) v *= signs[c];
        y[n * d + c] = v;
    }
}

// ---------------------------------------------------------------------------
// Scalar quantization + bit-packing
// ---------------------------------------------------------------------------
//
// For every coord of y, find the nearest codebook entry and pack that index
// into the output bit-stream. Supports B ∈ {1, 2, 4} so that 32/B indices fit
// cleanly per uint32 word. Coords within each row are packed independently.
//
// Shared memory: d bytes for the per-coord indices before packing.

template <int B>
__global__ void quantize_pack_kernel(
    const float* __restrict__ y,
    const float* __restrict__ codebook,   // (2^B,)
    uint32_t* __restrict__ packed,        // (N, words_per_row)
    int d)
{
    constexpr int K = 1 << B;
    constexpr int IDX_PER_WORD = 32 / B;
    constexpr uint32_t MASK = (1u << B) - 1u;

    extern __shared__ uint8_t sidx[];

    int tid = threadIdx.x;
    int n   = blockIdx.x;

    // Each thread assigns codebook indices for one or more coords
    for (int c = tid; c < d; c += blockDim.x) {
        float v = y[n * d + c];
        float best_dist = 1e30f;
        int   best_idx  = 0;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float diff = v - codebook[k];
            float dist = diff * diff;
            if (dist < best_dist) { best_dist = dist; best_idx = k; }
        }
        sidx[c] = static_cast<uint8_t>(best_idx);
    }
    __syncthreads();

    // Pack groups of IDX_PER_WORD indices into one uint32
    int words_per_row = (d + IDX_PER_WORD - 1) / IDX_PER_WORD;
    for (int w = tid; w < words_per_row; w += blockDim.x) {
        uint32_t word = 0;
        int base = w * IDX_PER_WORD;
        #pragma unroll
        for (int i = 0; i < IDX_PER_WORD; ++i) {
            int c = base + i;
            if (c < d) {
                word |= (static_cast<uint32_t>(sidx[c]) & MASK) << (i * B);
            }
        }
        packed[n * words_per_row + w] = word;
    }
}

// ---------------------------------------------------------------------------
// Dequantize (unpack + codebook lookup). Caller applies the inverse FWHT
// afterwards to recover the reconstruction in the original basis.
// ---------------------------------------------------------------------------

template <int B>
__global__ void unpack_dequantize_kernel(
    const uint32_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int d)
{
    constexpr int IDX_PER_WORD = 32 / B;
    constexpr uint32_t MASK = (1u << B) - 1u;

    int tid = threadIdx.x;
    int n   = blockIdx.x;
    int words_per_row = (d + IDX_PER_WORD - 1) / IDX_PER_WORD;

    for (int c = tid; c < d; c += blockDim.x) {
        int w = c / IDX_PER_WORD;
        int i = c - w * IDX_PER_WORD;
        uint32_t word = packed[n * words_per_row + w];
        int idx = (word >> (i * B)) & MASK;
        y[n * d + c] = codebook[idx];
    }
}

// ---------------------------------------------------------------------------
// Fused FWHT + quantize-pack (MSE forward path, single kernel)
// ---------------------------------------------------------------------------
//
// Combines `fwht_kernel<SIGNS_FIRST=true>` and `quantize_pack_kernel<B>` into
// one block per vector. Saves one kernel launch and the HBM round-trip of
// the intermediate `y` tensor (N·d·4 bytes written then read).
//
// Shared memory layout: d floats (FWHT working buffer) followed by
// d uint8 indices (pre-pack scratch). Total 5·d bytes — well under
// 48 KB for d ≤ 512 on sm_86.

template <int B>
__global__ void fused_quantize_kernel(
    const float* __restrict__ x,
    const float* __restrict__ signs,
    const float* __restrict__ codebook,
    uint32_t* __restrict__ packed,
    int d,
    float scale)
{
    constexpr int K = 1 << B;
    constexpr int IDX_PER_WORD = 32 / B;
    constexpr uint32_t MASK = (1u << B) - 1u;

    extern __shared__ float smem[];
    uint8_t* sidx = reinterpret_cast<uint8_t*>(smem + d);

    int tid = threadIdx.x;
    int n   = blockIdx.x;

    // Load x with sign flip
    for (int c = tid; c < d; c += blockDim.x) {
        smem[c] = x[n * d + c] * signs[c];
    }
    __syncthreads();

    // In-place FWHT butterflies
    for (int h = 1; h < d; h <<= 1) {
        int pairs = d >> 1;
        for (int p = tid; p < pairs; p += blockDim.x) {
            int i = ((p / h) << 1) * h + (p % h);
            int j = i + h;
            float a = smem[i];
            float b = smem[j];
            smem[i] = a + b;
            smem[j] = a - b;
        }
        __syncthreads();
    }

    // Scale + nearest-centroid, write indices to byte scratch.
    // __fmul_rn prevents --use_fast_math from fusing the scale multiply with
    // the downstream (v - codebook[k]) subtraction into an FMA — that fusion
    // would change rounding at tie-break boundaries and break bit-exact
    // equivalence with the unfused two-kernel path (which force-rounds through
    // HBM between kernels).
    for (int c = tid; c < d; c += blockDim.x) {
        float v = __fmul_rn(smem[c], scale);
        float best_dist = 1e30f;
        int   best_idx  = 0;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float diff = v - codebook[k];
            float dist = diff * diff;
            if (dist < best_dist) { best_dist = dist; best_idx = k; }
        }
        sidx[c] = static_cast<uint8_t>(best_idx);
    }
    __syncthreads();

    // Pack IDX_PER_WORD indices per uint32
    int words_per_row = (d + IDX_PER_WORD - 1) / IDX_PER_WORD;
    for (int w = tid; w < words_per_row; w += blockDim.x) {
        uint32_t word = 0;
        int base = w * IDX_PER_WORD;
        #pragma unroll
        for (int i = 0; i < IDX_PER_WORD; ++i) {
            int c = base + i;
            if (c < d) {
                word |= (static_cast<uint32_t>(sidx[c]) & MASK) << (i * B);
            }
        }
        packed[n * words_per_row + w] = word;
    }
}

// ---------------------------------------------------------------------------
// Inline-PTX variant of fused_quantize_kernel.
//
// Uses `bfi.b32` (bitfield insert) for the per-word packing loop. Semantically
// identical to the C++ `word |= (idx & MASK) << (i * B)` pattern, but forces
// the compiler to emit exactly one bfi instruction per insertion — nvcc may
// lower the C++ expression to the same instruction under -O3, but inline PTX
// guarantees it and makes the generated code auditable. Everything else
// (FWHT butterflies, centroid search, shared-mem layout) is copied verbatim
// from fused_quantize_kernel so the benchmark delta is purely the pack path.
// ---------------------------------------------------------------------------

template <int B>
__global__ void fused_quantize_ptx_kernel(
    const float* __restrict__ x,
    const float* __restrict__ signs,
    const float* __restrict__ codebook,
    uint32_t* __restrict__ packed,
    int d,
    float scale)
{
    constexpr int K = 1 << B;
    constexpr int IDX_PER_WORD = 32 / B;

    extern __shared__ float smem[];
    uint8_t* sidx = reinterpret_cast<uint8_t*>(smem + d);

    int tid = threadIdx.x;
    int n   = blockIdx.x;

    for (int c = tid; c < d; c += blockDim.x) {
        smem[c] = x[n * d + c] * signs[c];
    }
    __syncthreads();

    for (int h = 1; h < d; h <<= 1) {
        int pairs = d >> 1;
        for (int p = tid; p < pairs; p += blockDim.x) {
            int i = ((p / h) << 1) * h + (p % h);
            int j = i + h;
            float a = smem[i];
            float b = smem[j];
            smem[i] = a + b;
            smem[j] = a - b;
        }
        __syncthreads();
    }

    for (int c = tid; c < d; c += blockDim.x) {
        float v = __fmul_rn(smem[c], scale);
        float best_dist = 1e30f;
        int   best_idx  = 0;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float diff = v - codebook[k];
            float dist = diff * diff;
            if (dist < best_dist) { best_dist = dist; best_idx = k; }
        }
        sidx[c] = static_cast<uint8_t>(best_idx);
    }
    __syncthreads();

    int words_per_row = (d + IDX_PER_WORD - 1) / IDX_PER_WORD;
    for (int w = tid; w < words_per_row; w += blockDim.x) {
        uint32_t word = 0;
        int base = w * IDX_PER_WORD;
        #pragma unroll
        for (int i = 0; i < IDX_PER_WORD; ++i) {
            int c = base + i;
            if (c < d) {
                uint32_t idx = static_cast<uint32_t>(sidx[c]);
                // bfi.b32 dst, src, base, start_bit, num_bits
                // dst[start_bit + num_bits - 1 : start_bit] = src[num_bits-1 : 0]
                // other bits of dst come from `base`.
                asm("bfi.b32 %0, %1, %0, %2, %3;"
                    : "+r"(word)
                    : "r"(idx), "r"(i * B), "n"(B));
            }
        }
        packed[n * words_per_row + w] = word;
    }
}

// ---------------------------------------------------------------------------
// Fused unpack-dequantize + inverse FWHT (MSE reverse path, single kernel)
// ---------------------------------------------------------------------------
//
// Combines `unpack_dequantize_kernel<B>` and `fwht_kernel<SIGNS_FIRST=false>`.
// Shared memory: d floats.

template <int B>
__global__ void fused_dequantize_kernel(
    const uint32_t* __restrict__ packed,
    const float* __restrict__ signs,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int d,
    float scale)
{
    constexpr int IDX_PER_WORD = 32 / B;
    constexpr uint32_t MASK = (1u << B) - 1u;

    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int n   = blockIdx.x;
    int words_per_row = (d + IDX_PER_WORD - 1) / IDX_PER_WORD;

    // Unpack + codebook lookup into shared memory
    for (int c = tid; c < d; c += blockDim.x) {
        int w = c / IDX_PER_WORD;
        int i = c - w * IDX_PER_WORD;
        uint32_t word = packed[n * words_per_row + w];
        int idx = (word >> (i * B)) & MASK;
        smem[c] = codebook[idx];
    }
    __syncthreads();

    // In-place FWHT butterflies (inverse uses the same butterflies; scaling
    // and sign flip happen on the way out)
    for (int h = 1; h < d; h <<= 1) {
        int pairs = d >> 1;
        for (int p = tid; p < pairs; p += blockDim.x) {
            int i = ((p / h) << 1) * h + (p % h);
            int j = i + h;
            float a = smem[i];
            float b = smem[j];
            smem[i] = a + b;
            smem[j] = a - b;
        }
        __syncthreads();
    }

    // Scale + sign flip, write to HBM.
    // Use __fmul_rn to force full fp32 rounding at each multiplication,
    // matching the unfused path's HBM round-trip semantics exactly.
    for (int c = tid; c < d; c += blockDim.x) {
        float scaled = __fmul_rn(smem[c], scale);
        y[n * d + c]  = __fmul_rn(scaled, signs[c]);
    }
}

// ---------------------------------------------------------------------------
// QJL 1-bit sign packing / unpacking
// ---------------------------------------------------------------------------

__global__ void pack_signs_kernel(
    const float* __restrict__ x,
    uint32_t* __restrict__ packed,
    int d)
{
    int tid = threadIdx.x;
    int n   = blockIdx.x;
    int words_per_row = (d + 31) / 32;

    for (int w = tid; w < words_per_row; w += blockDim.x) {
        uint32_t word = 0;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            int c = w * 32 + b;
            if (c < d) {
                float v = x[n * d + c];
                // Map nonnegative -> 1, negative -> 0
                word |= static_cast<uint32_t>(v >= 0.0f ? 1u : 0u) << b;
            }
        }
        packed[n * words_per_row + w] = word;
    }
}

// ---------------------------------------------------------------------------
// Warp-ballot variant of pack_signs using inline PTX `vote.sync.ballot.b32`.
// One warp (32 lanes) produces one 32-bit packed word in ONE PTX instruction,
// replacing the 32-iteration scalar bit-OR loop. Threads-per-word goes from
// 1 (scalar) to 32 (warp), so we use 32× more threads per word — the outer
// loop iterates warps, not threads.
// ---------------------------------------------------------------------------

__global__ void pack_signs_ptx_kernel(
    const float* __restrict__ x,
    uint32_t* __restrict__ packed,
    int d)
{
    int tid = threadIdx.x;
    int n   = blockIdx.x;
    int words_per_row = (d + 31) / 32;

    int lane = tid & 31;
    int warp_id = tid >> 5;
    int warps_per_block = blockDim.x >> 5;

    for (int w = warp_id; w < words_per_row; w += warps_per_block) {
        int c = w * 32 + lane;
        int pred = 0;
        if (c < d) {
            float v = x[n * d + c];
            pred = (v >= 0.0f) ? 1 : 0;
        }
        // Inline PTX: convert the int predicate to a real predicate register,
        // then ballot across the full warp. The ballot result is bit-i=pred of
        // lane i — exactly the packed sign word we want.
        uint32_t ballot;
        asm volatile(
            "{\n\t"
            "  .reg .pred p;\n\t"
            "  setp.ne.s32 p, %1, 0;\n\t"
            "  vote.sync.ballot.b32 %0, p, 0xffffffff;\n\t"
            "}"
            : "=r"(ballot)
            : "r"(pred)
        );
        if (lane == 0) {
            packed[n * words_per_row + w] = ballot;
        }
    }
}

__global__ void unpack_signs_kernel(
    const uint32_t* __restrict__ packed,
    float* __restrict__ out,
    int d)
{
    int tid = threadIdx.x;
    int n   = blockIdx.x;
    int words_per_row = (d + 31) / 32;

    for (int c = tid; c < d; c += blockDim.x) {
        int w = c / 32;
        int b = c - w * 32;
        uint32_t word = packed[n * words_per_row + w];
        float v = ((word >> b) & 1u) ? 1.0f : -1.0f;
        out[n * d + c] = v;
    }
}

// ===========================================================================
// Host-side dispatch (called from Python via pybind)
// ===========================================================================

static inline int pick_block_size(int d) {
    // Match threads to d, clamp at 256 (one warp × 8 is plenty for d≤256)
    if (d <= 64)  return 64;
    if (d <= 128) return 128;
    if (d <= 256) return 256;
    return 256;
}

static inline void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor fwht_forward(torch::Tensor x, torch::Tensor signs) {
    check_cuda_tensor(x, "x");
    check_cuda_tensor(signs, "signs");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(signs.dtype() == torch::kFloat32, "signs must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be (N, d)");

    int N = x.size(0);
    int d = x.size(1);
    TORCH_CHECK((d & (d - 1)) == 0 && d >= 2, "d must be a power of 2, got ", d);
    TORCH_CHECK(signs.numel() == d, "signs must have length d");

    auto y = torch::empty_like(x);
    float scale = 1.0f / std::sqrt((float)d);
    int block = pick_block_size(d);
    size_t shmem = d * sizeof(float);

    fwht_kernel<true><<<N, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), signs.data_ptr<float>(), y.data_ptr<float>(), d, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor fwht_inverse(torch::Tensor x, torch::Tensor signs) {
    check_cuda_tensor(x, "x");
    check_cuda_tensor(signs, "signs");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && signs.dtype() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2);

    int N = x.size(0);
    int d = x.size(1);
    TORCH_CHECK((d & (d - 1)) == 0 && d >= 2);

    auto y = torch::empty_like(x);
    float scale = 1.0f / std::sqrt((float)d);
    int block = pick_block_size(d);
    size_t shmem = d * sizeof(float);

    fwht_kernel<false><<<N, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), signs.data_ptr<float>(), y.data_ptr<float>(), d, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor quantize_pack(torch::Tensor y, torch::Tensor codebook, int64_t B) {
    check_cuda_tensor(y, "y");
    check_cuda_tensor(codebook, "codebook");
    TORCH_CHECK(y.dtype() == torch::kFloat32 && codebook.dtype() == torch::kFloat32);
    TORCH_CHECK(y.dim() == 2);
    TORCH_CHECK(B == 1 || B == 2 || B == 4, "B must be 1, 2, or 4 (packed), got ", B);
    TORCH_CHECK(codebook.numel() == (1LL << B), "codebook size must be 2^B");

    int N = y.size(0);
    int d = y.size(1);
    int idx_per_word = 32 / (int)B;
    int words_per_row = (d + idx_per_word - 1) / idx_per_word;

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(y.device());
    auto packed = torch::empty({N, words_per_row}, opts);

    int block = pick_block_size(d);
    size_t shmem = d * sizeof(uint8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (B == 1) {
        quantize_pack_kernel<1><<<N, block, shmem, stream>>>(
            y.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d);
    } else if (B == 2) {
        quantize_pack_kernel<2><<<N, block, shmem, stream>>>(
            y.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d);
    } else {
        quantize_pack_kernel<4><<<N, block, shmem, stream>>>(
            y.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor unpack_dequantize(torch::Tensor packed, torch::Tensor codebook,
                                 int64_t B, int64_t d) {
    check_cuda_tensor(packed, "packed");
    check_cuda_tensor(codebook, "codebook");
    TORCH_CHECK(packed.dim() == 2);
    TORCH_CHECK(B == 1 || B == 2 || B == 4);
    TORCH_CHECK(codebook.numel() == (1LL << B));

    int N = packed.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(packed.device());
    auto y = torch::empty({N, (int64_t)d}, opts);

    int block = pick_block_size((int)d);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (B == 1) {
        unpack_dequantize_kernel<1><<<N, block, 0, stream>>>(
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
            codebook.data_ptr<float>(), y.data_ptr<float>(), (int)d);
    } else if (B == 2) {
        unpack_dequantize_kernel<2><<<N, block, 0, stream>>>(
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
            codebook.data_ptr<float>(), y.data_ptr<float>(), (int)d);
    } else {
        unpack_dequantize_kernel<4><<<N, block, 0, stream>>>(
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
            codebook.data_ptr<float>(), y.data_ptr<float>(), (int)d);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor fused_quantize(torch::Tensor x, torch::Tensor signs,
                              torch::Tensor codebook, int64_t B) {
    check_cuda_tensor(x, "x");
    check_cuda_tensor(signs, "signs");
    check_cuda_tensor(codebook, "codebook");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(signs.dtype() == torch::kFloat32, "signs must be float32");
    TORCH_CHECK(codebook.dtype() == torch::kFloat32, "codebook must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be (N, d)");
    TORCH_CHECK(B == 1 || B == 2 || B == 4, "B must be 1, 2, or 4");
    TORCH_CHECK(codebook.numel() == (1LL << B), "codebook size must be 2^B");

    int N = x.size(0);
    int d = x.size(1);
    TORCH_CHECK((d & (d - 1)) == 0 && d >= 2, "d must be a power of 2, got ", d);
    TORCH_CHECK(signs.numel() == d, "signs must have length d");

    int idx_per_word = 32 / (int)B;
    int words_per_row = (d + idx_per_word - 1) / idx_per_word;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto packed = torch::empty({N, words_per_row}, opts);

    float scale = 1.0f / std::sqrt((float)d);
    int block = pick_block_size(d);
    size_t shmem = d * sizeof(float) + d * sizeof(uint8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (B == 1) {
        fused_quantize_kernel<1><<<N, block, shmem, stream>>>(
            x.data_ptr<float>(), signs.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d, scale);
    } else if (B == 2) {
        fused_quantize_kernel<2><<<N, block, shmem, stream>>>(
            x.data_ptr<float>(), signs.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d, scale);
    } else {
        fused_quantize_kernel<4><<<N, block, shmem, stream>>>(
            x.data_ptr<float>(), signs.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d, scale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor fused_quantize_ptx(torch::Tensor x, torch::Tensor signs,
                                  torch::Tensor codebook, int64_t B) {
    check_cuda_tensor(x, "x");
    check_cuda_tensor(signs, "signs");
    check_cuda_tensor(codebook, "codebook");
    TORCH_CHECK(x.dtype() == torch::kFloat32);
    TORCH_CHECK(signs.dtype() == torch::kFloat32);
    TORCH_CHECK(codebook.dtype() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(B == 1 || B == 2 || B == 4);
    TORCH_CHECK(codebook.numel() == (1LL << B));

    int N = x.size(0);
    int d = x.size(1);
    TORCH_CHECK((d & (d - 1)) == 0 && d >= 2);
    TORCH_CHECK(signs.numel() == d);

    int idx_per_word = 32 / (int)B;
    int words_per_row = (d + idx_per_word - 1) / idx_per_word;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto packed = torch::empty({N, words_per_row}, opts);

    float scale = 1.0f / std::sqrt((float)d);
    int block = pick_block_size(d);
    size_t shmem = d * sizeof(float) + d * sizeof(uint8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (B == 1) {
        fused_quantize_ptx_kernel<1><<<N, block, shmem, stream>>>(
            x.data_ptr<float>(), signs.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d, scale);
    } else if (B == 2) {
        fused_quantize_ptx_kernel<2><<<N, block, shmem, stream>>>(
            x.data_ptr<float>(), signs.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d, scale);
    } else {
        fused_quantize_ptx_kernel<4><<<N, block, shmem, stream>>>(
            x.data_ptr<float>(), signs.data_ptr<float>(), codebook.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()), d, scale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor pack_signs_ptx(torch::Tensor x) {
    check_cuda_tensor(x, "x");
    TORCH_CHECK(x.dtype() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2);

    int N = x.size(0);
    int d = x.size(1);
    int words_per_row = (d + 31) / 32;

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto packed = torch::empty({N, words_per_row}, opts);

    // Block must be a multiple of 32 (warp size) and hold enough warps to
    // cover words_per_row in one sweep when possible.
    int target = words_per_row * 32;
    int block;
    if      (target <= 64)   block = 64;
    else if (target <= 128)  block = 128;
    else if (target <= 256)  block = 256;
    else                      block = 256;

    pack_signs_ptx_kernel<<<N, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor fused_dequantize(torch::Tensor packed, torch::Tensor signs,
                                torch::Tensor codebook, int64_t B, int64_t d) {
    check_cuda_tensor(packed, "packed");
    check_cuda_tensor(signs, "signs");
    check_cuda_tensor(codebook, "codebook");
    TORCH_CHECK(packed.dim() == 2);
    TORCH_CHECK(signs.dtype() == torch::kFloat32);
    TORCH_CHECK(codebook.dtype() == torch::kFloat32);
    TORCH_CHECK(B == 1 || B == 2 || B == 4);
    TORCH_CHECK(codebook.numel() == (1LL << B));
    TORCH_CHECK((d & (d - 1)) == 0 && d >= 2);
    TORCH_CHECK(signs.numel() == d);

    int N = packed.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(packed.device());
    auto y = torch::empty({N, (int64_t)d}, opts);

    float scale = 1.0f / std::sqrt((float)d);
    int block = pick_block_size((int)d);
    size_t shmem = (size_t)d * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (B == 1) {
        fused_dequantize_kernel<1><<<N, block, shmem, stream>>>(
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
            signs.data_ptr<float>(), codebook.data_ptr<float>(),
            y.data_ptr<float>(), (int)d, scale);
    } else if (B == 2) {
        fused_dequantize_kernel<2><<<N, block, shmem, stream>>>(
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
            signs.data_ptr<float>(), codebook.data_ptr<float>(),
            y.data_ptr<float>(), (int)d, scale);
    } else {
        fused_dequantize_kernel<4><<<N, block, shmem, stream>>>(
            reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
            signs.data_ptr<float>(), codebook.data_ptr<float>(),
            y.data_ptr<float>(), (int)d, scale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor pack_signs(torch::Tensor x) {
    check_cuda_tensor(x, "x");
    TORCH_CHECK(x.dtype() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2);

    int N = x.size(0);
    int d = x.size(1);
    int words_per_row = (d + 31) / 32;

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto packed = torch::empty({N, words_per_row}, opts);

    int block = pick_block_size(d);
    pack_signs_kernel<<<N, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor unpack_signs(torch::Tensor packed, int64_t d) {
    check_cuda_tensor(packed, "packed");
    TORCH_CHECK(packed.dim() == 2);

    int N = packed.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(packed.device());
    auto out = torch::empty({N, (int64_t)d}, opts);

    int block = pick_block_size((int)d);
    unpack_signs_kernel<<<N, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        out.data_ptr<float>(), (int)d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwht_forward",      &fwht_forward,      "FWHT forward rotation (signs then butterflies)");
    m.def("fwht_inverse",      &fwht_inverse,      "FWHT inverse rotation (butterflies then signs)");
    m.def("quantize_pack",     &quantize_pack,     "Per-coord quantize + bit-pack");
    m.def("unpack_dequantize", &unpack_dequantize, "Bit-unpack + codebook lookup");
    m.def("fused_quantize",    &fused_quantize,    "Fused fwht_forward + quantize_pack (one kernel)");
    m.def("fused_dequantize",  &fused_dequantize,  "Fused unpack_dequantize + fwht_inverse (one kernel)");
    m.def("fused_quantize_ptx",&fused_quantize_ptx,"Fused quantize with inline PTX bfi.b32 packing");
    m.def("pack_signs_ptx",    &pack_signs_ptx,    "Pack sign bits using warp-ballot vote.sync.ballot.b32");
    m.def("pack_signs",        &pack_signs,        "Pack sign bits to uint32 words");
    m.def("unpack_signs",      &unpack_signs,      "Unpack sign bits to ±1 floats");
}
