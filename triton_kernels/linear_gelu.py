import torch
import triton
import triton.language as tl


@triton.jit
def _linear_gelu_fwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_xm,
    stride_xk,

    stride_wn,
    stride_wk,

    stride_om,
    stride_on,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,

    HAS_BIAS: tl.constexpr,
):
    # 이 program이 output matrix의 어느 tile을 담당할지
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # output tile의 row / column
    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    # reduction dimension K
    offs_k = tl.arange(0, BLOCK_K)

    # X shape = [M, K]
    #
    # [BLOCK_M, BLOCK_K]
    x_ptrs = (
        x_ptr
        + offs_m[:, None] * stride_xm
        + offs_k[None, :] * stride_xk
    )

    # weight shape = [N, K]
    #
    # 우리가 원하는 건 X @ W.T
    #
    # 따라서 논리적으로
    # B[k, n] = weight[n, k]
    #
    # [BLOCK_K, BLOCK_N]
    weight_ptrs = (
        weight_ptr
        + offs_k[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )

    # matmul 결과를 FP32로 누적
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    # K dimension을 BLOCK_K씩 순회
    for k in range(0, tl.cdiv(K, BLOCK_K)):

        k_mask = offs_k < K - k * BLOCK_K

        x = tl.load(
            x_ptrs,
            mask=(
                (offs_m[:, None] < M)
                & k_mask[None, :]
            ),
            other=0.0,
        )

        weight = tl.load(
            weight_ptrs,
            mask=(
                k_mask[:, None]
                & (offs_n[None, :] < N)
            ),
            other=0.0,
        )

        accumulator = tl.dot(
            x,
            weight,
            accumulator,
        )

        # 다음 K tile로 이동
        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += BLOCK_K * stride_wk

    # -------------------------
    # Linear bias
    # -------------------------

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offs_n,
            mask=offs_n < N,
            other=0.0,
        )

        # [BLOCK_N]
        #      ↓ broadcast
        # [BLOCK_M, BLOCK_N]
        accumulator += bias[None, :]

    # -------------------------
    # GELU
    #
    # 0.5*x*(1 + erf(x/sqrt(2)))
    # -------------------------

    output = 0.5 * accumulator * (
        1.0
        + tl.erf(
            accumulator * 0.7071067811865476
        )
    )

    # -------------------------
    # store
    # -------------------------

    out_ptrs = (
        out_ptr
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )

    out_mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        out_ptrs,
        output,
        mask=out_mask,
    )


def linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
):
    assert x.is_cuda
    assert weight.is_cuda

    assert x.is_contiguous()
    assert weight.is_contiguous()

    # PyTorch nn.Linear weight:
    #
    # weight.shape = [out_features, in_features]
    N, K = weight.shape

    assert x.shape[-1] == K

    if bias is not None:
        assert bias.shape == (N,)
        assert bias.is_contiguous()

    # [B,T,K]
    #
    # 를 논리적으로
    #
    # [M,K]
    #
    # matrix로 봄
    x_2d = x.view(-1, K)

    M = x_2d.shape[0]

    out_2d = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype,
    )

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    bias_ptr = (
        bias if bias is not None
        else weight
    )

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _linear_gelu_fwd_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        out_2d,

        M=M,
        N=N,
        K=K,

        stride_xm=x_2d.stride(0),
        stride_xk=x_2d.stride(1),

        # weight[n,k]
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_om=out_2d.stride(0),
        stride_on=out_2d.stride(1),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,

        HAS_BIAS=bias is not None,

        num_warps=4,
    )

    return out_2d.view(
        *x.shape[:-1],
        N,
    )