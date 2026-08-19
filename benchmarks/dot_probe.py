import torch
import triton
import triton.language as tl


@triton.jit
def _dot_probe(
    A,
    B,
    C,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + offs_m[:, None] * BLOCK_K
        + offs_k[None, :]
    )

    b_ptrs = (
        B
        + offs_k[:, None] * BLOCK_N
        + offs_n[None, :]
    )

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    c = tl.dot(a, b)

    c_ptrs = (
        C
        + offs_m[:, None] * BLOCK_N
        + offs_n[None, :]
    )

    tl.store(c_ptrs, c)


M = 32
N = 32
K = 64

a = torch.randn(
    M, K,
    device="cuda",
    dtype=torch.float16,
)

b = torch.randn(
    K, N,
    device="cuda",
    dtype=torch.float16,
)

c = torch.empty(
    M, N,
    device="cuda",
    dtype=torch.float32,
)

_dot_probe[(1,)](
    a, b, c,
    BLOCK_M=M,
    BLOCK_N=N,
    BLOCK_K=K,
    num_warps=4,
)

torch.cuda.synchronize()

# profile 대상
_dot_probe[(1,)](
    a, b, c,
    BLOCK_M=M,
    BLOCK_N=N,
    BLOCK_K=K,
    num_warps=4,
)

torch.cuda.synchronize()