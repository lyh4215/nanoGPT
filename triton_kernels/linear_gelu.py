import torch
import triton
import triton.language as tl

from triton_kernels.linear import (
    _linear_tile_from_ptrs,
    _get_linear_fwd_config,
)

@triton.jit
def _gelu(x):
    return 0.5 * x * (
        1.0
        + tl.libdevice.erf(
            x * 0.7071067811865476
        )
    )

@triton.jit
def _linear_gelu_fwd_kernel(
    X,
    W,
    BIAS,
    Y,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_xm,
    stride_xk,

    stride_wn,
    stride_wk,

    stride_ym,
    stride_yn,

    HAS_BIAS: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,

    GROUP_SIZE_M: tl.constexpr,
):
    # ========================================================
    # grouped program ordering
    # ========================================================

    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(
        M,
        BLOCK_M,
    )

    num_pid_n = tl.cdiv(
        N,
        BLOCK_N,
    )

    num_pid_in_group = (
        GROUP_SIZE_M
        * num_pid_n
    )

    group_id = (
        pid
        // num_pid_in_group
    )

    first_pid_m = (
        group_id
        * GROUP_SIZE_M
    )

    group_size_m = min(
        num_pid_m - first_pid_m,
        GROUP_SIZE_M,
    )

    pid_m = (
        first_pid_m
        + (
            pid
            % num_pid_in_group
        )
        % group_size_m
    )

    pid_n = (
        (
            pid
            % num_pid_in_group
        )
        // group_size_m
    )

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    # ========================================================
    # Linear
    # ========================================================

    acc = _linear_tile_from_ptrs(
        X,
        W,

        offs_m,
        offs_n,

        M,
        N,
        K,

        stride_xm,
        stride_xk,

        stride_wn,
        stride_wk,

        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    # ========================================================
    # Bias
    # ========================================================

    if HAS_BIAS:
        bias = tl.load(
            BIAS + offs_n,
            mask=offs_n < N,
            other=0.0,
        )

        acc += bias[None, :]

    # ========================================================
    # GELU
    #
    # 핵심:
    # acc를 HBM에 store하지 않고
    # register 상태에서 바로 GELU
    # ========================================================

    acc = _gelu(acc)

    # ========================================================
    # Store
    # ========================================================

    y_ptrs = (
        Y
        + offs_m[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )

    tl.store(
        y_ptrs,
        acc,
        mask=(
            (offs_m[:, None] < M)
            & (offs_n[None, :] < N)
        ),
    )

def triton_linear_gelu_forward(
    x,
    weight,
    bias=None,
):
    assert x.is_cuda
    assert weight.is_cuda

    assert x.is_contiguous()
    assert weight.is_contiguous()

    K = x.shape[-1]

    N, K_weight = weight.shape

    assert K == K_weight

    original_shape = x.shape

    x_2d = x.view(
        -1,
        K,
    )

    M = x_2d.shape[0]

    y = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype,
    )

    (
        block_m,
        block_n,
        block_k,
        num_warps,
        group_size_m,
    ) = _get_linear_fwd_config(
        K,
        N,
    )

    grid = (
        triton.cdiv(M, block_m)
        * triton.cdiv(N, block_n),
    )

    _linear_gelu_fwd_kernel[grid](
        x_2d,
        weight,
        bias,
        y,

        M=M,
        N=N,
        K=K,

        stride_xm=x_2d.stride(0),
        stride_xk=x_2d.stride(1),

        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_ym=y.stride(0),
        stride_yn=y.stride(1),

        HAS_BIAS=bias is not None,

        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,

        GROUP_SIZE_M=group_size_m,

        num_warps=num_warps,
    )

    return y.view(
        *original_shape[:-1],
        N,
    )