import torch
import triton
import triton.language as tl

from triton_kernels.matmul import _matmul_accumulate

@triton.jit
def _linear_tile_from_ptrs(
    X,
    W,

    offs_m,
    offs_n,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_xm,
    stride_xk,

    stride_wn,
    stride_wk,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(
        0,
        BLOCK_K,
    )

    acc = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for k_start in range(
        0,
        K,
        BLOCK_K,
    ):
        k = (
            k_start
            + offs_k
        )

        # ----------------------------------------------------
        # X tile
        #
        # [BLOCK_M, BLOCK_K]
        # ----------------------------------------------------

        x_ptrs = (
            X
            + offs_m[:, None] * stride_xm
            + k[None, :] * stride_xk
        )

        x = tl.load(
            x_ptrs,
            mask=(
                (offs_m[:, None] < M)
                & (k[None, :] < K)
            ),
            other=0.0,
        )

        # ----------------------------------------------------
        # W tile
        #
        # nn.Linear weight shape:
        #
        # [N, K]
        #
        # 여기서는:
        # [BLOCK_N, BLOCK_K]
        # ----------------------------------------------------

        w_ptrs = (
            W
            + offs_n[:, None] * stride_wn
            + k[None, :] * stride_wk
        )

        w = tl.load(
            w_ptrs,
            mask=(
                (offs_n[:, None] < N)
                & (k[None, :] < K)
            ),
            other=0.0,
        )

        # ----------------------------------------------------
        # 실제 matrix multiplication은
        # matmul primitive에게 위임
        # ----------------------------------------------------

        acc = _matmul_accumulate(
            acc,
            x,
            w,
        )

    return acc

@triton.jit
def _linear_fwd_kernel(
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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    # --------------------------------------------------------
    # X @ W.T
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # + bias
    # --------------------------------------------------------

    if HAS_BIAS:
        bias = tl.load(
            BIAS + offs_n,
            mask=offs_n < N,
            other=0.0,
        )

        acc += bias[None, :]

    # --------------------------------------------------------
    # output store
    # --------------------------------------------------------

    y_ptrs = (
        Y
        + offs_m[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        y_ptrs,
        acc,
        mask=mask,
    )

def triton_linear_forward(
    x,
    weight,
    bias=None,
):
    """
    Equivalent to:

        torch.nn.functional.linear(
            x,
            weight,
            bias,
        )

    x:
        [..., K]

    weight:
        [N, K]

    output:
        [..., N]
    """

    assert x.is_cuda
    assert weight.is_cuda

    assert x.is_contiguous()
    assert weight.is_contiguous()

    K = x.shape[-1]

    N, K_weight = weight.shape

    assert K == K_weight

    original_shape = x.shape

    # [B,T,K]
    #       ↓
    # [B*T,K]

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

    # 우선 correctness용.
    # 튜닝은 나중.
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _linear_fwd_kernel[grid](
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

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,

        num_warps=4,
    )

    return y.view(
        *original_shape[:-1],
        N,
    )

@triton.jit
def _linear_bwd_dx_kernel(
    DY,
    W,
    DX,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_dym,
    stride_dyn,

    stride_wn,
    stride_wk,

    stride_dxm,
    stride_dxk,

    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_k = (
        pid_k * BLOCK_K
        + tl.arange(0, BLOCK_K)
    )

    offs_n = tl.arange(
        0,
        BLOCK_N,
    )

    # dX tile
    #
    # [BM, BK]
    acc = tl.zeros(
        (BLOCK_M, BLOCK_K),
        dtype=tl.float32,
    )

    # ========================================================
    # reduction over N
    #
    # dX = dY @ W
    # ========================================================

    for n_start in range(
        0,
        N,
        BLOCK_N,
    ):
        n = (
            n_start
            + offs_n
        )

        # ----------------------------------------------------
        # dY tile
        #
        # [BM, BN]
        # ----------------------------------------------------

        dy_ptrs = (
            DY
            + offs_m[:, None] * stride_dym
            + n[None, :] * stride_dyn
        )

        dy = tl.load(
            dy_ptrs,
            mask=(
                (offs_m[:, None] < M)
                & (n[None, :] < N)
            ),
            other=0.0,
        )

        # ----------------------------------------------------
        # W tile
        #
        # 원래 W:
        #
        # [N, K]
        #
        # 그런데 helper가
        #
        # a @ b.T
        #
        # 를 하므로 여기서는:
        #
        # [BK, BN]
        #
        # 형태로 읽는다.
        # ----------------------------------------------------

        w_ptrs = (
            W
            + n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk
        )

        w = tl.load(
            w_ptrs,
            mask=(
                (n[None, :] < N)
                & (offs_k[:, None] < K)
            ),
            other=0.0,
        )

        # dy : [BM, BN]
        # w  : [BK, BN]
        #
        # helper:
        #
        # dy @ w.T
        #
        # [BM,BN] @ [BN,BK]
        #        ↓
        # [BM,BK]

        acc = _matmul_accumulate(
            acc,
            dy,
            w,
        )

    # ========================================================
    # store dX
    # ========================================================

    dx_ptrs = (
        DX
        + offs_m[:, None] * stride_dxm
        + offs_k[None, :] * stride_dxk
    )

    tl.store(
        dx_ptrs,
        acc,
        mask=(
            (offs_m[:, None] < M)
            & (offs_k[None, :] < K)
        ),
    )

@triton.jit
def _linear_bwd_dw_kernel(
    DY,
    X,
    DW,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_dym,
    stride_dyn,

    stride_xm,
    stride_xk,

    stride_dwn,
    stride_dwk,

    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    offs_k = (
        pid_k * BLOCK_K
        + tl.arange(0, BLOCK_K)
    )

    offs_m = tl.arange(
        0,
        BLOCK_M,
    )

    # dW tile
    #
    # [BN, BK]
    acc = tl.zeros(
        (BLOCK_N, BLOCK_K),
        dtype=tl.float32,
    )

    # ========================================================
    # reduction over M
    #
    # dW = dY.T @ X
    # ========================================================

    for m_start in range(
        0,
        M,
        BLOCK_M,
    ):
        m = (
            m_start
            + offs_m
        )

        # ----------------------------------------------------
        # dY tile
        #
        # 원래 dY:
        # [M, N]
        #
        # register에서는:
        # [BN, BM]
        # ----------------------------------------------------

        dy_ptrs = (
            DY
            + m[None, :] * stride_dym
            + offs_n[:, None] * stride_dyn
        )

        dy = tl.load(
            dy_ptrs,
            mask=(
                (m[None, :] < M)
                & (offs_n[:, None] < N)
            ),
            other=0.0,
        )

        # ----------------------------------------------------
        # X tile
        #
        # 원래 X:
        # [M, K]
        #
        # register에서는:
        # [BK, BM]
        # ----------------------------------------------------

        x_ptrs = (
            X
            + m[None, :] * stride_xm
            + offs_k[:, None] * stride_xk
        )

        x = tl.load(
            x_ptrs,
            mask=(
                (m[None, :] < M)
                & (offs_k[:, None] < K)
            ),
            other=0.0,
        )

        # dy : [BN, BM]
        # x  : [BK, BM]
        #
        # helper:
        #
        # dy @ x.T
        #
        # [BN,BM] @ [BM,BK]
        #        ↓
        # [BN,BK]

        acc = _matmul_accumulate(
            acc,
            dy,
            x,
        )

    # ========================================================
    # store dW
    # ========================================================

    dw_ptrs = (
        DW
        + offs_n[:, None] * stride_dwn
        + offs_k[None, :] * stride_dwk
    )

    tl.store(
        dw_ptrs,
        acc,
        mask=(
            (offs_n[:, None] < N)
            & (offs_k[None, :] < K)
        ),
    )

def triton_linear_backward_dx(
    dy,
    weight,
):
    assert dy.is_cuda
    assert weight.is_cuda

    # dy : [..., N]
    # W  : [N, K]

    N = dy.shape[-1]

    N_weight, K = weight.shape

    assert N == N_weight

    original_shape = dy.shape

    dy_2d = dy.reshape(
        -1,
        N,
    )

    M = dy_2d.shape[0]

    dx = torch.empty(
        (M, K),
        device=dy.device,
        dtype=dy.dtype,
    )

    BLOCK_M = 32
    BLOCK_K = 32
    BLOCK_N = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K, BLOCK_K),
    )

    _linear_bwd_dx_kernel[grid](
        dy_2d,
        weight,
        dx,

        M=M,
        N=N,
        K=K,

        stride_dym=dy_2d.stride(0),
        stride_dyn=dy_2d.stride(1),

        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_dxm=dx.stride(0),
        stride_dxk=dx.stride(1),

        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,

        num_warps=4,
    )

    return dx.view(
        *original_shape[:-1],
        K,
    )

def triton_linear_backward_dw(
    dy,
    x,
):
    assert dy.is_cuda
    assert x.is_cuda

    # dy : [..., N]
    # x  : [..., K]

    N = dy.shape[-1]
    K = x.shape[-1]

    dy_2d = dy.reshape(
        -1,
        N,
    )

    x_2d = x.reshape(
        -1,
        K,
    )

    M = dy_2d.shape[0]

    assert x_2d.shape[0] == M

    dw = torch.empty(
        (N, K),
        device=dy.device,
        dtype=dy.dtype,
    )

    BLOCK_N = 32
    BLOCK_K = 32
    BLOCK_M = 32

    grid = (
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(K, BLOCK_K),
    )

    _linear_bwd_dw_kernel[grid](
        dy_2d,
        x_2d,
        dw,

        M=M,
        N=N,
        K=K,

        stride_dym=dy_2d.stride(0),
        stride_dyn=dy_2d.stride(1),

        stride_xm=x_2d.stride(0),
        stride_xk=x_2d.stride(1),

        stride_dwn=dw.stride(0),
        stride_dwk=dw.stride(1),

        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_M=BLOCK_M,

        num_warps=4,
    )

    return dw

@triton.jit
def _linear_bwd_db_kernel(
    DY,
    DB,

    M: tl.constexpr,
    N: tl.constexpr,

    stride_dym,
    stride_dyn,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    offs_m = tl.arange(
        0,
        BLOCK_M,
    )

    # 각 output feature마다 하나씩
    #
    # [BLOCK_N]
    acc = tl.zeros(
        (BLOCK_N,),
        dtype=tl.float32,
    )

    # ========================================================
    # db_n = sum_m dY[m, n]
    #
    # M dimension reduction
    # ========================================================

    for m_start in range(
        0,
        M,
        BLOCK_M,
    ):
        m = (
            m_start
            + offs_m
        )

        # dY tile:
        #
        # [BLOCK_M, BLOCK_N]
        dy_ptrs = (
            DY
            + m[:, None] * stride_dym
            + offs_n[None, :] * stride_dyn
        )

        dy = tl.load(
            dy_ptrs,
            mask=(
                (m[:, None] < M)
                & (offs_n[None, :] < N)
            ),
            other=0.0,
        )

        # [BM, BN]
        #    ↓ sum over M
        # [BN]
        acc += tl.sum(
            dy.to(tl.float32),
            axis=0,
        )

    tl.store(
        DB + offs_n,
        acc,
        mask=offs_n < N,
    )

def triton_linear_backward_db(
    dy,
):
    assert dy.is_cuda

    N = dy.shape[-1]

    dy_2d = dy.reshape(
        -1,
        N,
    )

    M = dy_2d.shape[0]

    db = torch.empty(
        (N,),
        device=dy.device,
        dtype=dy.dtype,
    )

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (
        triton.cdiv(N, BLOCK_N),
    )

    _linear_bwd_db_kernel[grid](
        dy_2d,
        db,

        M=M,
        N=N,

        stride_dym=dy_2d.stride(0),
        stride_dyn=dy_2d.stride(1),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=4,
    )

    return db

class TritonLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
    ):
        # ----------------------------------------------------
        # Triton Linear forward
        #
        # Y = X W^T + b
        # ----------------------------------------------------

        y = triton_linear_forward(
            x,
            weight,
            bias,
        )

        # backward에 필요한 건:
        #
        # dX = dY W
        # dW = dY^T X
        #
        # 따라서 X, W 저장
        ctx.save_for_backward(
            x,
            weight,
        )

        # bias 값 자체는 backward에 필요 없음.
        # bias가 있었는지만 기억하면 됨.
        ctx.has_bias = (
            bias is not None
        )

        return y

    @staticmethod
    def backward(
        ctx,
        dy,
    ):
        x, weight = ctx.saved_tensors

        # ----------------------------------------------------
        # dX = dY @ W
        # ----------------------------------------------------

        dx = triton_linear_backward_dx(
            dy,
            weight,
        )

        # ----------------------------------------------------
        # dW = dY^T @ X
        # ----------------------------------------------------

        dw = triton_linear_backward_dw(
            dy,
            x,
        )

        # ----------------------------------------------------
        # db = sum(dY, dim=M)
        # ----------------------------------------------------

        if ctx.has_bias:
            db = triton_linear_backward_db(
                dy,
            )
        else:
            db = None

        # forward inputs:
        #
        # x, weight, bias
        #
        # 순서와 정확히 대응해서 gradient 반환
        return (
            dx,
            dw,
            db,
        )


def triton_linear(
    x,
    weight,
    bias=None,
):
    return TritonLinearFunction.apply(
        x,
        weight,
        bias,
    )