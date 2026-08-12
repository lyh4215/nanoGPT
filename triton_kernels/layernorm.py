import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # program 하나가 row 하나를 담당
    row = tl.program_id(0)

    # 현재 row의 시작 주소
    row_start = row * N

    # 이 program이 다룰 column들
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # [N] row를 HBM에서 load
    # reduction은 FP32로 하는 게 안정적
    x = tl.load(
        x_ptr + row_start + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # mean
    mean = tl.sum(x, axis=0) / N

    # variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    # 1 / sqrt(var + eps)
    rstd = tl.rsqrt(var + eps)

    #backward에서 사용
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # normalize
    x_hat = diff * rstd

    # LayerNorm learnable weight
    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    else:
        bias = 0.0

    # y = gamma * x_hat + beta
    y = x_hat * weight + bias

    # 결과를 HBM에 write
    tl.store(
        y_ptr + row_start + offsets,
        y,
        mask=mask,
    )

@triton.jit
def _layer_norm_bwd_dx_fused_kernel(
    dx_ptr,
    dy_ptr,

    partial_dw_ptr,
    partial_db_ptr,

    x_ptr,
    weight_ptr,

    mean_ptr,
    rstd_ptr,

    locks_ptr,

    N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(
        0,
        BLOCK_SIZE,
    )

    mask = offsets < N

    row_start = row * N

    # ------------------------
    # load
    # ------------------------

    x = tl.load(
        x_ptr + row_start + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    dy = tl.load(
        dy_ptr + row_start + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mean = tl.load(
        mean_ptr + row,
    )

    rstd = tl.load(
        rstd_ptr + row,
    )

    # ------------------------
    # dx
    # ------------------------

    x_hat = tl.where(
        mask,
        (x - mean) * rstd,
        0.0,
    )

    g = tl.where(
        mask,
        dy * weight,
        0.0,
    )

    c1 = tl.sum(
        x_hat * g,
        axis=0,
    ) / N

    c2 = tl.sum(
        g,
        axis=0,
    ) / N

    dx = rstd * (
        g
        - x_hat * c1
        - c2
    )

    tl.store(
        dx_ptr + row_start + offsets,
        dx,
        mask=mask,
    )

    # ==========================================
    # partial dw / db
    # ==========================================

    partial_dw = dy * x_hat
    partial_db = dy

    # row들을 GROUP_SIZE_M개의 buffer로 분산
    lock_id = row % GROUP_SIZE_M

    dw_ptrs = (
        partial_dw_ptr
        + lock_id * N
        + offsets
    )

    db_ptrs = (
        partial_db_ptr
        + lock_id * N
        + offsets
    )

    # locks:
    #
    # [0 : GROUP_SIZE_M]       → mutex
    # [GROUP_SIZE_M : 2G]      → count

    lock_ptr = (
        locks_ptr + lock_id
    )

    count_ptr = (
        locks_ptr
        + GROUP_SIZE_M
        + lock_id
    )

    # 같은 partial buffer를 사용하는
    # 다른 program과 race 방지
    while tl.atomic_cas(
        lock_ptr,
        0,
        1,
    ) == 1:
        pass

    count = tl.load(count_ptr)

    if count == 0:
        # 첫 번째 row
        tl.atomic_xchg(
            count_ptr,
            1,
        )

    else:
        old_dw = tl.load(
            dw_ptrs,
            mask=mask,
            other=0.0,
        )

        old_db = tl.load(
            db_ptrs,
            mask=mask,
            other=0.0,
        )

        partial_dw += old_dw
        partial_db += old_db

    tl.store(
        dw_ptrs,
        partial_dw,
        mask=mask,
    )

    tl.store(
        db_ptrs,
        partial_db,
        mask=mask,
    )

    tl.debug_barrier()

    # lock release
    tl.atomic_xchg(
        lock_ptr,
        0,
    )

@triton.jit
def _layer_norm_bwd_final_dwdb_kernel(
    partial_dw_ptr,
    partial_db_ptr,

    dw_ptr,
    db_ptr,

    M,
    N: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = (
        pid * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    dw = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    db = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    # 여기서 M은 원래 B*T가 아니라
    # partial buffer 개수
    for i in range(
        0,
        M,
        BLOCK_M,
    ):
        rows = (
            i
            + tl.arange(0, BLOCK_M)
        )

        mask = (
            (rows[:, None] < M)
            & (cols[None, :] < N)
        )

        offsets = (
            rows[:, None] * N
            + cols[None, :]
        )

        dw += tl.load(
            partial_dw_ptr + offsets,
            mask=mask,
            other=0.0,
        )

        db += tl.load(
            partial_db_ptr + offsets,
            mask=mask,
            other=0.0,
        )

    final_dw = tl.sum(
        dw,
        axis=0,
    )

    final_db = tl.sum(
        db,
        axis=0,
    )

    tl.store(
        dw_ptr + cols,
        final_dw,
        mask=cols < N,
    )

    tl.store(
        db_ptr + cols,
        final_db,
        mask=cols < N,
    )

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    num_warps=4,
):
    assert x.is_cuda
    assert weight.is_cuda
    assert x.is_contiguous()

    N = x.shape[-1]

    # [B, T, C]를 논리적으로 [B*T, C]로 봄
    M = x.numel() // N

    y = torch.empty_like(x)

    # C=768이면 1024
    BLOCK_SIZE = triton.next_power_of_2(N)

    # bias=None이어도 pointer argument 자체는 하나 넘겨야 하므로
    # 사용하지 않을 weight pointer를 대신 넘김.
    bias_ptr = bias if bias is not None else weight

    grid = (M,)

    _layer_norm_fwd_kernel[grid](
        x,
        weight,
        bias_ptr,
        y,
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
    )

    return y

def layer_norm_autograd(
    x,
    weight,
    bias,
    eps=1e-5,
):
    return TritonLayerNorm.apply(
        x,
        weight,
        bias,
        eps,
    )

class TritonLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        eps,
    ):
        assert x.is_cuda
        assert x.is_contiguous()

        N = x.shape[-1]
        M = x.numel() // N

        y = torch.empty_like(x)

        mean = torch.empty(
            M,
            device=x.device,
            dtype=torch.float32,
        )

        rstd = torch.empty(
            M,
            device=x.device,
            dtype=torch.float32,
        )

        BLOCK_SIZE = triton.next_power_of_2(N)

        bias_ptr = (
            bias
            if bias is not None
            else weight
        )

        _layer_norm_fwd_kernel[(M,)](
            x,
            weight,
            bias_ptr,
            y,

            mean,
            rstd,

            N=N,
            eps=eps,

            BLOCK_SIZE=BLOCK_SIZE,
            HAS_BIAS=bias is not None,

            num_warps=8,
        )

        # backward에서 필요
        ctx.save_for_backward(
            x,
            weight,
            mean,
            rstd,
        )

        ctx.N = N
        ctx.has_bias = bias is not None
        ctx.block_size = BLOCK_SIZE

        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors

        dy = dy.contiguous()

        N = ctx.N
        M = x.numel() // N

        dx = torch.empty_like(x)

        dw = torch.empty_like(weight)
        db = torch.empty_like(weight)

        # C=768에서는 256
        GROUP_SIZE_M = 256

        # M보다 클 필요는 없음
        GROUP_SIZE_M = min(
            GROUP_SIZE_M,
            M,
        )

        partial_dw = torch.zeros(
            (GROUP_SIZE_M, N),
            device=x.device,
            dtype=torch.float32,
        )

        partial_db = torch.zeros(
            (GROUP_SIZE_M, N),
            device=x.device,
            dtype=torch.float32,
        )

        # mutex + count
        locks = torch.zeros(
            2 * GROUP_SIZE_M,
            device=x.device,
            dtype=torch.int32,
        )

        # =============================
        # Stage 1
        # dx + partial dw/db
        # =============================

        _layer_norm_bwd_dx_fused_kernel[(M,)](
            dx,
            dy,

            partial_dw,
            partial_db,

            x,
            weight,

            mean,
            rstd,

            locks,

            N=N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE=ctx.block_size,

            num_warps=8,
        )

        # =============================
        # Stage 2
        # partial → final dw/db
        # =============================

        BLOCK_M = 32
        BLOCK_N = 128

        grid = (
            triton.cdiv(
                N,
                BLOCK_N,
            ),
        )

        _layer_norm_bwd_final_dwdb_kernel[grid](
            partial_dw,
            partial_db,

            dw,
            db,

            GROUP_SIZE_M,
            N=N,

            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,

            num_warps=4,
        )

        if not ctx.has_bias:
            db = None

        return (
            dx,
            dw,
            db,
            None,
        )