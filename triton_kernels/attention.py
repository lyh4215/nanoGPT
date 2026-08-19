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

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_fwd_kernel(
    Q,
    K,
    V,
    O,

    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,

    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,

    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,

    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,

    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    SCALE: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ---------------------------------------
    # program mapping
    #
    # pid_m  : query tile index
    # pid_bh : batch/head index
    # ---------------------------------------

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    start_m = pid_m * BLOCK_M

    offs_m = (
        start_m
        + tl.arange(0, BLOCK_M)
    )

    offs_d = tl.arange(
        0,
        HEAD_DIM,
    )

    # ---------------------------------------
    # Q tile
    #
    # [BLOCK_M, HEAD_DIM]
    #
    # 이 Q tile은 K/V loop 동안 계속 사용
    # ---------------------------------------

    q_ptrs = (
        Q
        + batch * stride_qb
        + head * stride_qh
        + offs_m[:, None] * stride_qt
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(q_ptrs)

    # ---------------------------------------
    # online softmax state
    #
    # query row마다 하나씩:
    #
    # m_i   : 현재까지 max
    # l_i   : exp sum
    #
    # acc   : weighted V numerator
    #         [BLOCK_M, HEAD_DIM]
    # ---------------------------------------

    m_i = tl.full(
        (BLOCK_M,),
        -float("inf"),
        tl.float32,
    )

    l_i = tl.zeros(
        (BLOCK_M,),
        dtype=tl.float32,
    )

    acc = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )

    # ---------------------------------------
    # causal attention이므로
    #
    # query block이 [start_m, start_m+BM)
    # 이면 그 뒤의 key block은 볼 필요 없음
    # ---------------------------------------

    hi = tl.minimum(
        (pid_m + 1) * BLOCK_M,
        N_CTX,
    )

    # =======================================
    # K / V tile loop
    # =======================================

    for start_n in tl.range(
        0,
        hi,
        BLOCK_N,
    ):
        offs_n = (
            start_n
            + tl.arange(0, BLOCK_N)
        )

        # -----------------------------------
        # K tile
        #
        # [BLOCK_N, HEAD_DIM]
        # -----------------------------------

        k_ptrs = (
            K
            + batch * stride_kb
            + head * stride_kh
            + offs_n[:, None] * stride_kt
            + offs_d[None, :] * stride_kd
        )

        k = tl.load(k_ptrs)

        # -----------------------------------
        # score tile
        #
        # [BM,D] @ [D,BN]
        #       ↓
        # [BM,BN]
        #
        # 이 tensor는 HBM에 store 안 함
        # -----------------------------------

        qk = tl.dot(
            q,
            tl.trans(k),
        )

        qk *= SCALE

        # -----------------------------------
        # causal mask
        # -----------------------------------

        causal_mask = (
            offs_m[:, None]
            >= offs_n[None, :]
        )

        qk = tl.where(
            causal_mask,
            qk,
            -float("inf"),
        )

        # ===================================
        # ONLINE SOFTMAX
        # ===================================

        # 이번 tile까지 포함한 새로운 max
        m_ij = tl.maximum(
            m_i,
            tl.max(
                qk,
                axis=1,
            ),
        )

        # 이전 accumulator를
        # 새로운 max 기준으로 rescale
        alpha = tl.exp(
            m_i - m_ij
        )

        # 이번 score tile의 exponent
        p = tl.exp(
            qk - m_ij[:, None]
        )

        # 이번 tile의 exp sum
        l_ij = tl.sum(
            p,
            axis=1,
        )

        # -----------------------------------
        # V tile
        #
        # [BLOCK_N, HEAD_DIM]
        # -----------------------------------

        v_ptrs = (
            V
            + batch * stride_vb
            + head * stride_vh
            + offs_n[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )

        v = tl.load(v_ptrs)

        # ===================================
        # output numerator update
        #
        # old accumulator rescale
        # +
        # P_tile @ V_tile
        # ===================================

        acc *= alpha[:, None]

        acc += tl.dot(
            p.to(tl.float16),
            v,
        )

        # -----------------------------------
        # denominator update
        # -----------------------------------

        l_i = (
            l_i * alpha
            + l_ij
        )

        m_i = m_ij

    # =======================================
    # final normalization
    # =======================================

    out = (
        acc
        / l_i[:, None]
    )

    # ---------------------------------------
    # 이것만 HBM에 최종 store
    # ---------------------------------------

    o_ptrs = (
        O
        + batch * stride_ob
        + head * stride_oh
        + offs_m[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )

    tl.store(
        o_ptrs,
        out,
    )

def triton_flash_attention_forward(
    q,
    k,
    v,
):
    """
    q, k, v:
        [B, H, T, D]

    현재 educational version:
        causal=True
        dtype=float16
        D=64
        T는 64의 배수
    """

    assert q.is_cuda
    assert k.is_cuda
    assert v.is_cuda

    assert q.shape == k.shape
    assert q.shape == v.shape

    B, H, T, D = q.shape

    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16

    assert D == 64
    assert T % 64 == 0

    BLOCK_M = 32
    BLOCK_N = 32

    scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    grid = (
        triton.cdiv(
            T,
            BLOCK_M,
        ),
        B * H,
    )

    _flash_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,

        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),

        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),

        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),

        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,

        SCALE=scale,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=4,
        num_stages=2,
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