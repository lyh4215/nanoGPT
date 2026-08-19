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

    scores = scores / math.sqrt(D)

    # ---------------------------------
    # 2. causal mask
    # ---------------------------------

    if causal:
        mask = torch.tril(
            torch.ones(
                T,
                T,
                device=q.device,
                dtype=torch.bool,
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # ---------------------------------
    # 3. row-wise softmax
    # ---------------------------------

    probs = F.softmax(
        scores,
        dim=-1,
    )
    # [B, H, T, T]

    # ---------------------------------
    # 4. weighted sum of V
    # ---------------------------------

    out = probs @ v
    # [B, H,T,D]

    return out