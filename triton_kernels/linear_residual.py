import torch
import triton
import triton.language as tl

from triton_kernels.linear import (
    _linear_tile_from_ptrs,
    _get_linear_fwd_config,

    triton_linear_backward_dx,
    triton_linear_backward_dw,
    triton_linear_backward_db,
)


# ============================================================
# Forward kernel
#
# Y = X @ W.T + bias + residual
#
# X        : [M, K]
# W        : [N, K]
# residual : [M, N]
# Y        : [M, N]
# ============================================================

@triton.jit
def _linear_residual_fwd_kernel(
    X,
    W,
    BIAS,
    RESIDUAL,
    Y,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_xm,
    stride_xk,

    stride_wn,
    stride_wk,

    stride_rm,
    stride_rn,

    stride_ym,
    stride_yn,

    HAS_BIAS: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,

    GROUP_SIZE_M: tl.constexpr,
):
    # ========================================================
    # Program ordering
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

    # ========================================================
    # Output offsets
    # ========================================================

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    # ========================================================
    # Linear
    #
    # acc:
    #   [BLOCK_M, BLOCK_N]
    #
    # FP32 accumulator
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
    # Residual
    #
    # 여기서 Linear output을 HBM에 쓰지 않고
    # FP32 accumulator에 바로 residual을 더함.
    # ========================================================

    residual_ptrs = (
        RESIDUAL
        + offs_m[:, None] * stride_rm
        + offs_n[None, :] * stride_rn
    )

    residual = tl.load(
        residual_ptrs,
        mask=mask,
        other=0.0,
    )

    acc += residual

    # ========================================================
    # Final store
    # ========================================================

    y_ptrs = (
        Y
        + offs_m[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )

    tl.store(
        y_ptrs,
        acc,
        mask=mask,
    )

def triton_linear_residual_forward(
    x,
    weight,
    bias,
    residual,
):
    assert x.is_cuda
    assert weight.is_cuda
    assert residual.is_cuda

    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert residual.is_contiguous()

    K = x.shape[-1]

    N, K_weight = (
        weight.shape
    )

    assert K == K_weight

    original_shape = x.shape

    x_2d = x.view(
        -1,
        K,
    )

    M = x_2d.shape[0]

    # residual은 Linear output과 shape이 같아야 함
    expected_shape = (
        *original_shape[:-1],
        N,
    )

    assert residual.shape == expected_shape, (
        f"residual shape mismatch: "
        f"expected {expected_shape}, "
        f"got {residual.shape}"
    )

    residual_2d = residual.view(
        M,
        N,
    )

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

    _linear_residual_fwd_kernel[grid](
        x_2d,
        weight,
        bias,
        residual_2d,
        y,

        M=M,
        N=N,
        K=K,

        stride_xm=x_2d.stride(0),
        stride_xk=x_2d.stride(1),

        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_rm=residual_2d.stride(0),
        stride_rn=residual_2d.stride(1),

        stride_ym=y.stride(0),
        stride_yn=y.stride(1),

        HAS_BIAS=(
            bias is not None
        ),

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

class TritonLinearResidualFunction(
    torch.autograd.Function
):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual,
    ):
        y = triton_linear_residual_forward(
            x,
            weight,
            bias,
            residual,
        )

        ctx.save_for_backward(
            x,
            weight,
        )

        ctx.has_bias = (
            bias is not None
        )

        return y

    @staticmethod
    def backward(
        ctx,
        dy,
    ):
        x, weight = (
            ctx.saved_tensors
        )

        # 기존 Linear backward wrapper가
        # contiguous input을 기대할 수 있으므로
        dy = dy.contiguous()

        # ====================================================
        # Linear backward
        # ====================================================

        dx = triton_linear_backward_dx(
            dy,
            weight,
        )

        dw = triton_linear_backward_dw(
            dy,
            x,
        )

        db = (
            triton_linear_backward_db(
                dy
            )
            if ctx.has_bias
            else None
        )

        # ====================================================
        # Residual backward
        #
        # y = linear(...) + residual
        #
        # dy / dresidual = 1
        #
        # 따라서:
        #
        # dResidual = dY
        # ====================================================

        dresidual = dy

        return (
            dx,
            dw,
            db,
            dresidual,
        )
    
def triton_linear_residual(
    x,
    weight,
    bias,
    residual,
):
    return (
        TritonLinearResidualFunction.apply(
            x,
            weight,
            bias,
            residual,
        )
    )