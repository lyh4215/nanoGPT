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
def _layer_norm_bwd_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,

    mean_ptr,
    rstd_ptr,

    dx_ptr,

    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    row_start = row * N

    offsets = tl.arange(
        0,
        BLOCK_SIZE,
    )

    mask = offsets < N

    # -----------------------
    # load
    # -----------------------

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

    # -----------------------
    # x_hat
    # -----------------------

    x_hat = tl.where(
        mask,
        (x - mean) * rstd,
        0.0,
    )

    # dL / dx_hat
    g = tl.where(
        mask,
        dy * weight,
        0.0,
    )

    # -----------------------
    # two reductions
    # -----------------------

    c1 = tl.sum(
        g * x_hat,
        axis=0,
    ) / N

    c2 = tl.sum(
        g,
        axis=0,
    ) / N

    # -----------------------
    # dx
    # -----------------------

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

@triton.jit
def _layer_norm_bwd_dwdb_kernel(
    dy_ptr,
    x_ptr,

    mean_ptr,
    rstd_ptr,

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

    col_mask = cols < N

    dw = tl.zeros(
        (BLOCK_N,),
        dtype=tl.float32,
    )

    db = tl.zeros(
        (BLOCK_N,),
        dtype=tl.float32,
    )

    # M(row) 방향으로 순회
    for row_start in range(
        0,
        M,
        BLOCK_M,
    ):
        rows = (
            row_start
            + tl.arange(0, BLOCK_M)
        )

        row_mask = rows < M

        mask = (
            row_mask[:, None]
            & col_mask[None, :]
        )

        offsets = (
            rows[:, None] * N
            + cols[None, :]
        )

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        dy = tl.load(
            dy_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(
            mean_ptr + rows,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        rstd = tl.load(
            rstd_ptr + rows,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        x_hat = (
            x
            - mean[:, None]
        ) * rstd[:, None]

        # 같은 column의 모든 row 합
        dw += tl.sum(
            dy * x_hat,
            axis=0,
        )

        db += tl.sum(
            dy,
            axis=0,
        )

    tl.store(
        dw_ptr + cols,
        dw,
        mask=col_mask,
    )

    tl.store(
        db_ptr + cols,
        db,
        mask=col_mask,
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
    def backward(
        ctx,
        dy,
    ):
        x, weight, mean, rstd = (
            ctx.saved_tensors
        )

        dy = dy.contiguous()

        N = ctx.N
        M = x.numel() // N

        dx = torch.empty_like(x)

        # -------------------
        # dx
        # -------------------

        _layer_norm_bwd_dx_kernel[(M,)](
            dy,
            x,
            weight,

            mean,
            rstd,

            dx,

            N=N,
            BLOCK_SIZE=ctx.block_size,

            num_warps=8,
        )

        # -------------------
        # dw / db
        # -------------------

        dw = torch.empty_like(weight)

        db = torch.empty_like(weight)

        BLOCK_M = 32
        BLOCK_N = 64

        grid = (
            triton.cdiv(
                N,
                BLOCK_N,
            ),
        )

        _layer_norm_bwd_dwdb_kernel[grid](
            dy,
            x,

            mean,
            rstd,

            dw,
            db,

            M,
            N=N,

            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,

            num_warps=4,
        )

        if not ctx.has_bias:
            db = None

        # forward inputs:
        # x, weight, bias, eps
        return (
            dx,
            dw,
            db,
            None,
        )