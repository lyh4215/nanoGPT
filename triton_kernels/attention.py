import math

import torch
import torch.nn.functional as F


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