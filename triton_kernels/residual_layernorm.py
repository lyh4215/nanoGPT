import torch
import triton
import triton.language as tl


@triton.jit
def _residual_layer_norm_fwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)

    row_start = row * N

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # x[row, :]
    x = tl.load(
        x_ptr + row_start + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # residual[row, :]
    residual = tl.load(
        residual_ptr + row_start + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # 핵심: 중간 tensor를 HBM에 저장하지 않음
    z = x + residual

    # LayerNorm mean
    mean = tl.sum(z, axis=0) / N

    # padding 영역이 variance에 들어가지 않게 처리
    diff = tl.where(
        mask,
        z - mean,
        0.0,
    )

    # LayerNorm variance
    var = tl.sum(
        diff * diff,
        axis=0,
    ) / N

    rstd = tl.rsqrt(var + eps)

    x_hat = diff * rstd

    # gamma
    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # beta
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    else:
        bias = 0.0

    y = x_hat * weight + bias

    tl.store(
        y_ptr + row_start + offsets,
        y,
        mask=mask,
    )


def residual_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    num_warps: int = 8,
):
    assert x.is_cuda
    assert residual.is_cuda
    assert weight.is_cuda

    assert x.shape == residual.shape
    assert x.is_contiguous()
    assert residual.is_contiguous()

    N = x.shape[-1]
    M = x.numel() // N

    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)

    bias_ptr = bias if bias is not None else weight

    grid = (M,)

    _residual_layer_norm_fwd_kernel[grid](
        x,
        residual,
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