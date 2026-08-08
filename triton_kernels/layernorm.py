import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
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


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
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
    )

    return y