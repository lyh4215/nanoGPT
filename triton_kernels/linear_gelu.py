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

def _get_linear_gelu_fwd_config(K, N):
    # GPT-2 MLP c_fc
    if K == 768 and N == 3072:
        return 128, 64, 64, 4, 8

    return 128, 64, 64, 4, 1

@triton.jit
def _gelu(x):
    return 0.5 * x * (
        1.0
        + tl.erf(
            x * 0.7071067811865476
        )
    )

@triton.jit
def _linear_gelu_fwd_kernel(
    X,
    W,
    BIAS,
    Z,
    Y,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_xm,
    stride_xk,

    stride_wn,
    stride_wk,

    stride_zm,
    stride_zn,

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
    # ------------------------------------------------------------
    # Z = Linear output
    # backward에서 GELU'(Z)를 계산하기 위해 저장
    # ------------------------------------------------------------
    z_ptrs = (
        Z
        + offs_m[:, None] * stride_zm
        + offs_n[None, :] * stride_zn
    )

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        z_ptrs,
        acc,
        mask=mask,
    )

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

    z = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype,
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
    ) = _get_linear_gelu_fwd_config(
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
        z,
        y,

        M=M,
        N=N,
        K=K,

        stride_xm=x_2d.stride(0),
        stride_xk=x_2d.stride(1),

        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_zm=z.stride(0),
        stride_zn=z.stride(1),

        stride_ym=y.stride(0),
        stride_yn=y.stride(1),

        HAS_BIAS=bias is not None,

        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,

        GROUP_SIZE_M=group_size_m,

        num_warps=num_warps,
    )

    return (
        y.view(*original_shape[:-1], N),
        z.view(*original_shape[:-1], N),
    )

@triton.jit
def _gelu_backward(dy, z):
    dy = dy.to(tl.float32)
    z = z.to(tl.float32)

    cdf = 0.5 * (
        1.0
        + tl.erf(
            z * 0.7071067811865476
        )
    )

    pdf_term = (
        z
        * 0.3989422804014327
        * tl.exp(-0.5 * z * z)
    )

    return dy * (
        cdf + pdf_term
    )


@triton.jit
def _gelu_bwd_db_partial_kernel(
    DY,
    Z,
    DZ,
    PARTIAL_DB,

    M: tl.constexpr,
    N: tl.constexpr,

    stride_dym,
    stride_dyn,

    stride_zm,
    stride_zn,

    stride_dzm,
    stride_dzn,

    stride_pm,
    stride_pn,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    # ========================================================
    # load dy / z
    # ========================================================

    dy_ptrs = (
        DY
        + offs_m[:, None] * stride_dym
        + offs_n[None, :] * stride_dyn
    )

    z_ptrs = (
        Z
        + offs_m[:, None] * stride_zm
        + offs_n[None, :] * stride_zn
    )

    dy = tl.load(
        dy_ptrs,
        mask=mask,
        other=0.0,
    )

    z = tl.load(
        z_ptrs,
        mask=mask,
        other=0.0,
    )

    # ========================================================
    # dz = dy * GELU'(z)
    #
    # _gelu_backward() 내부에서 FP32로 올림
    # dz logical shape:
    #   [BLOCK_M, BLOCK_N]
    # ========================================================

    dz = _gelu_backward(
        dy,
        z,
    )

    # ========================================================
    # DZ 저장
    #
    # dX / dW kernel이 이후 재사용
    # ========================================================

    dz_ptrs = (
        DZ
        + offs_m[:, None] * stride_dzm
        + offs_n[None, :] * stride_dzn
    )

    tl.store(
        dz_ptrs,
        dz,
        mask=mask,
    )

    # ========================================================
    # 이 M tile 안에서 bias gradient partial reduction
    #
    # dz:
    #   [BLOCK_M, BLOCK_N]
    #
    # axis=0으로 합하면:
    #   [BLOCK_N]
    #
    # partial[n] =
    #   sum_{m in this M tile} dz[m,n]
    # ========================================================

    partial = tl.sum(
        dz,
        axis=0,
    )

    partial_ptrs = (
        PARTIAL_DB
        + pid_m * stride_pm
        + offs_n * stride_pn
    )

    tl.store(
        partial_ptrs,
        partial,
        mask=offs_n < N,
    )

@triton.jit
def _db_reduce_partials_kernel(
    PARTIAL_DB,
    DB,

    NUM_PARTIALS: tl.constexpr,
    N: tl.constexpr,

    stride_pm,
    stride_pn,

    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_r = tl.arange(
        0,
        BLOCK_R,
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    ptrs = (
        PARTIAL_DB
        + offs_r[:, None] * stride_pm
        + offs_n[None, :] * stride_pn
    )

    mask = (
        (offs_r[:, None] < NUM_PARTIALS)
        & (offs_n[None, :] < N)
    )

    partials = tl.load(
        ptrs,
        mask=mask,
        other=0.0,
    )

    # [BLOCK_R, BLOCK_N]
    #      ↓ axis 0
    # [BLOCK_N]

    db = tl.sum(
        partials,
        axis=0,
    )

    tl.store(
        DB + offs_n,
        db,
        mask=offs_n < N,
    )

def triton_gelu_backward_with_db(
    dy,
    z,
):
    assert dy.is_cuda
    assert z.is_cuda

    assert dy.shape == z.shape

    N = dy.shape[-1]

    dy_2d = (
        dy.contiguous()
        .view(-1, N)
    )

    z_2d = (
        z.contiguous()
        .view(-1, N)
    )

    M = dy_2d.shape[0]

    dz = torch.empty_like(
        dy_2d
    )

    # ========================================================
    # 첫 구현용 config
    #
    # 이건 아직 tuning한 값이 아님.
    # correctness + benchmark 후 sweep 가능.
    # ========================================================

    BLOCK_M = 128
    BLOCK_N = 16

    num_pid_m = triton.cdiv(
        M,
        BLOCK_M,
    )

    # partial은 FP32로 유지
    partial_db = torch.empty(
        (num_pid_m, N),
        device=dy.device,
        dtype=torch.float32,
    )

    # ========================================================
    # Kernel 1
    # GELU backward + partial db
    # ========================================================

    grid = (
        num_pid_m,
        triton.cdiv(
            N,
            BLOCK_N,
        ),
    )

    _gelu_bwd_db_partial_kernel[grid](
        dy_2d,
        z_2d,
        dz,
        partial_db,

        M=M,
        N=N,

        stride_dym=dy_2d.stride(0),
        stride_dyn=dy_2d.stride(1),

        stride_zm=z_2d.stride(0),
        stride_zn=z_2d.stride(1),

        stride_dzm=dz.stride(0),
        stride_dzn=dz.stride(1),

        stride_pm=partial_db.stride(0),
        stride_pn=partial_db.stride(1),

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=4,
    )

    # ========================================================
    # Kernel 2
    # partial_db -> final db
    # ========================================================

    db = torch.empty(
        (N,),
        device=dy.device,
        dtype=dy.dtype,
    )

    # tl.arange는 power-of-two 크기가 편하니까
    # num_pid_m 이상인 가장 가까운 2^n
    BLOCK_R = triton.next_power_of_2(
        num_pid_m
    )

    DB_BLOCK_N = 128

    db_grid = (
        triton.cdiv(
            N,
            DB_BLOCK_N,
        ),
    )

    _db_reduce_partials_kernel[db_grid](
        partial_db,
        db,

        NUM_PARTIALS=num_pid_m,
        N=N,

        stride_pm=partial_db.stride(0),
        stride_pn=partial_db.stride(1),

        BLOCK_R=BLOCK_R,
        BLOCK_N=DB_BLOCK_N,

        num_warps=4,
    )

    return (
        dz.view_as(dy),
        db,
    )


@triton.jit
def _gelu_bwd_kernel(
    DY,
    Z,
    DZ,

    N_ELEMENTS: tl.constexpr,

    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = (
        pid * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    mask = offsets < N_ELEMENTS

    dy = tl.load(
        DY + offsets,
        mask=mask,
        other=0.0,
    )

    z = tl.load(
        Z + offsets,
        mask=mask,
        other=0.0,
    )

    dz = _gelu_backward(
        dy,
        z,
    )

    tl.store(
        DZ + offsets,
        dz,
        mask=mask,
    )

def triton_gelu_backward(
    dy,
    z,
):
    assert dy.is_cuda
    assert z.is_cuda
    assert dy.shape == z.shape

    dy_flat = dy.contiguous().view(-1)
    z_flat = z.contiguous().view(-1)

    dz = torch.empty_like(
        dy_flat
    )

    n_elements = dy_flat.numel()

    BLOCK_SIZE = 256

    grid = (
        triton.cdiv(
            n_elements,
            BLOCK_SIZE,
        ),
    )

    _gelu_bwd_kernel[grid](
        dy_flat,
        z_flat,
        dz,

        N_ELEMENTS=n_elements,

        BLOCK_SIZE=BLOCK_SIZE,

        num_warps=4,
    )

    return dz.view_as(dy)

class TritonLinearGELUFunction(
    torch.autograd.Function
):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
    ):
        y, z = triton_linear_gelu_forward(
            x,
            weight,
            bias,
        )

        ctx.save_for_backward(
            x,
            weight,
            z,
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
        x, weight, z = (
            ctx.saved_tensors
        )

        # ====================================================
        # GELU backward
        # ====================================================

        if ctx.has_bias:
            dz, db = (
                triton_gelu_backward_with_db(
                    dy,
                    z,
                )
            )
        else:
            dz = triton_gelu_backward(
                dy,
                z,
            )

            db = None

        # ====================================================
        # Linear backward
        # ====================================================

        dx = triton_linear_backward_dx(
            dz,
            weight,
        )

        dw = triton_linear_backward_dw(
            dz,
            x,
        )

        return (
            dx,
            dw,
            db,
        )

def triton_linear_gelu(
    x,
    weight,
    bias=None,
):
    return TritonLinearGELUFunction.apply(
        x,
        weight,
        bias,
    )