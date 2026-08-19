import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    row_start = row * N

    x = tl.load(
        x_ptr + row_start + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    # numerical stability
    x_max = tl.max(
        x,
        axis=0,
    )

    x = x - x_max

    numerator = tl.exp(x)

    denominator = tl.sum(
        numerator,
        axis=0,
    )

    y = numerator / denominator

    tl.store(
        y_ptr + row_start + offsets,
        y,
        mask=mask,
    )

def triton_softmax(x):
    assert x.is_cuda
    assert x.is_contiguous()

    N = x.shape[-1]
    M = x.numel() // N

    BLOCK_SIZE = triton.next_power_of_2(N)

    y = torch.empty_like(x)

    _softmax_kernel[(M,)](
        x,
        y,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return y

@triton.jit
def _causal_softmax_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # row = ((b * H + h) * T + query_idx)
    query_idx = row % N

    x = tl.load(
        x_ptr + row * N + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    # scale
    x = x * SCALE

    # causal mask
    causal_mask = offsets <= query_idx

    x = tl.where(
        causal_mask & mask,
        x,
        -float("inf"),
    )

    # softmax
    x_max = tl.max(x, axis=0)

    numerator = tl.exp(
        x - x_max
    )

    denominator = tl.sum(
        numerator,
        axis=0,
    )

    y = numerator / denominator

    tl.store(
        y_ptr + row * N + offsets,
        y,
        mask=mask,
    )

def triton_causal_softmax(scores, scale):
    N = scores.shape[-1]
    M = scores.numel() // N

    BLOCK_SIZE = triton.next_power_of_2(N)

    out = torch.empty_like(scores)

    _causal_softmax_kernel[(M,)](
        scores,
        out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        SCALE=scale,
        num_warps=8,
    )

    return out

def naive_attention(
    q,
    k,
    v,
    causal=True,
):
    """
    q, k, v:
        [B, H, T, D]

    return:
        [B, H, T, D]
    """

    B, H, T, D = q.shape

    # ---------------------------------
    # 1. attention score
    # ---------------------------------

    scores = q @ k.transpose(-2, -1)
    # [B, H, T, T]

    probs = triton_causal_softmax(
        scores,
        1.0 / math.sqrt(D),
    )

    out = probs @ v
    # [B, H,T,D]

    return out