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
    LSE,

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

    lse_ptrs = (
        LSE
        + batch * H * N_CTX
        + head * N_CTX
        + offs_m
    )

    tl.store(
        lse_ptrs,
        m_i + tl.log(l_i),
        mask=offs_m < N_CTX,
    )

def triton_flash_attention_forward(
    q,
    k,
    v,
    block_m=32,
    block_n=64,
    num_warps=1,
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
    lse = torch.empty(
        (B, H, T),
        device=q.device,
        dtype=torch.float32,
    )

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
        lse,

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

        BLOCK_M=block_m,
        BLOCK_N=block_n,

        num_warps=num_warps,
        num_stages=2,
    )

    return out, lse

@triton.jit
def _flash_attention_bwd_preprocess_kernel(
    O,
    DO,
    DELTA,

    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,

    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,

    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    o_ptrs = (
        O
        + batch * stride_ob
        + head * stride_oh
        + offs_m[:, None] * stride_ot
        + offs_d[None, :] * stride_od
    )

    do_ptrs = (
        DO
        + batch * stride_dob
        + head * stride_doh
        + offs_m[:, None] * stride_dot
        + offs_d[None, :] * stride_dod
    )

    mask = offs_m[:, None] < N_CTX

    o = tl.load(o_ptrs, mask=mask, other=0.0)
    do = tl.load(do_ptrs, mask=mask, other=0.0)

    # [BLOCK_M, D] → [BLOCK_M]
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = (
        DELTA
        + batch * H * N_CTX
        + head * N_CTX
        + offs_m
    )

    tl.store(
        delta_ptrs,
        delta,
        mask=offs_m < N_CTX,
    )

@triton.jit
def _flash_attention_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,

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

    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,

    stride_dqb,
    stride_dqh,
    stride_dqt,
    stride_dqd,

    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    SCALE: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    #
    # Q [BM,D]
    #
    q_ptrs = (
        Q
        + batch * stride_qb
        + head * stride_qh
        + offs_m[:, None] * stride_qt
        + offs_d[None, :] * stride_qd
    )

    #
    # dO [BM,D]
    #
    do_ptrs = (
        DO
        + batch * stride_dob
        + head * stride_doh
        + offs_m[:, None] * stride_dot
        + offs_d[None, :] * stride_dod
    )

    row_mask = offs_m[:, None] < N_CTX

    q = tl.load(
        q_ptrs,
        mask=row_mask,
        other=0.0,
    )

    do = tl.load(
        do_ptrs,
        mask=row_mask,
        other=0.0,
    )

    #
    # LSE [BM]
    #
    lse_ptrs = (
        LSE
        + batch * H * N_CTX
        + head * N_CTX
        + offs_m
    )

    lse = tl.load(
        lse_ptrs,
        mask=offs_m < N_CTX,
        other=0.0,
    )

    #
    # delta [BM]
    #
    delta_ptrs = (
        DELTA
        + batch * H * N_CTX
        + head * N_CTX
        + offs_m
    )

    delta = tl.load(
        delta_ptrs,
        mask=offs_m < N_CTX,
        other=0.0,
    )

    #
    # dQ accumulator
    #
    dq = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )

    #
    # causal attention:
    #
    # query tile [start_m, start_m + BM)
    # 에서는 start_m+BM 이전의 K만 볼 필요 있음
    #
    hi = tl.minimum(
        (pid_m + 1) * BLOCK_M,
        N_CTX,
    )

    for start_n in tl.range(
        0,
        hi,
        BLOCK_N,
    ):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        #
        # K [BN,D]
        #
        k_ptrs = (
            K
            + batch * stride_kb
            + head * stride_kh
            + offs_n[:, None] * stride_kt
            + offs_d[None, :] * stride_kd
        )

        k = tl.load(
            k_ptrs,
            mask=offs_n[:, None] < N_CTX,
            other=0.0,
        )

        #
        # V [BN,D]
        #
        v_ptrs = (
            V
            + batch * stride_vb
            + head * stride_vh
            + offs_n[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )

        v = tl.load(
            v_ptrs,
            mask=offs_n[:, None] < N_CTX,
            other=0.0,
        )

        #
        # S = scale * Q K^T
        #
        # [BM,D] @ [D,BN]
        # → [BM,BN]
        #
        scores = tl.dot(
            q,
            tl.trans(k),
        )

        scores *= SCALE

        #
        # causal mask
        #
        causal = (
            offs_m[:, None]
            >= offs_n[None, :]
        )

        valid = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
            & causal
        )

        scores = tl.where(
            valid,
            scores,
            -float("inf"),
        )

        #
        # P를 저장해놨던 게 아니라
        # backward에서 복원
        #
        # P_ij = exp(S_ij - LSE_i)
        #
        p = tl.exp(
            scores - lse[:, None]
        )

        #
        # dP = dO @ V^T
        #
        dp = tl.dot(
            do,
            tl.trans(v),
        )

        #
        # dS = P * (dP - delta)
        #
        ds = p * (
            dp - delta[:, None]
        )

        #
        # dQ += scale * dS @ K
        #
        dq += (
            tl.dot(
                ds.to(tl.float16),
                k,
            )
            * SCALE
        )

    #
    # dQ store
    #
    dq_ptrs = (
        DQ
        + batch * stride_dqb
        + head * stride_dqh
        + offs_m[:, None] * stride_dqt
        + offs_d[None, :] * stride_dqd
    )

    tl.store(
        dq_ptrs,
        dq,
        mask=offs_m[:, None] < N_CTX,
    )

@triton.jit
def _flash_attention_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,

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

    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,

    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,

    stride_dvb,
    stride_dvh,
    stride_dvt,
    stride_dvd,

    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    SCALE: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    #
    # 이 program이 담당하는 K/V tile
    #
    k_ptrs = (
        K
        + batch * stride_kb
        + head * stride_kh
        + offs_n[:, None] * stride_kt
        + offs_d[None, :] * stride_kd
    )

    v_ptrs = (
        V
        + batch * stride_vb
        + head * stride_vh
        + offs_n[:, None] * stride_vt
        + offs_d[None, :] * stride_vd
    )

    kv_mask = offs_n[:, None] < N_CTX

    k = tl.load(
        k_ptrs,
        mask=kv_mask,
        other=0.0,
    )

    v = tl.load(
        v_ptrs,
        mask=kv_mask,
        other=0.0,
    )

    dk = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        tl.float32,
    )

    dv = tl.zeros(
        (BLOCK_N, HEAD_DIM),
        tl.float32,
    )

    #
    # correctness 우선:
    # 모든 Q tile을 순회
    #
    start_m_begin = pid_n * BLOCK_N

    for start_m in tl.range(
        start_m_begin,
        N_CTX,
        BLOCK_M,
    ):
        offs_m = (
            start_m
            + tl.arange(0, BLOCK_M)
        )

        q_ptrs = (
            Q
            + batch * stride_qb
            + head * stride_qh
            + offs_m[:, None] * stride_qt
            + offs_d[None, :] * stride_qd
        )

        do_ptrs = (
            DO
            + batch * stride_dob
            + head * stride_doh
            + offs_m[:, None] * stride_dot
            + offs_d[None, :] * stride_dod
        )

        q = tl.load(
            q_ptrs,
            mask=offs_m[:, None] < N_CTX,
            other=0.0,
        )

        do = tl.load(
            do_ptrs,
            mask=offs_m[:, None] < N_CTX,
            other=0.0,
        )

        lse_ptrs = (
            LSE
            + batch * H * N_CTX
            + head * N_CTX
            + offs_m
        )

        delta_ptrs = (
            DELTA
            + batch * H * N_CTX
            + head * N_CTX
            + offs_m
        )

        lse = tl.load(
            lse_ptrs,
            mask=offs_m < N_CTX,
            other=0.0,
        )

        delta = tl.load(
            delta_ptrs,
            mask=offs_m < N_CTX,
            other=0.0,
        )

        #
        # recompute S
        #
        scores = (
            tl.dot(
                q,
                tl.trans(k),
            )
            * SCALE
        )

        causal = (
            offs_m[:, None]
            >= offs_n[None, :]
        )

        valid = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
            & causal
        )

        scores = tl.where(
            valid,
            scores,
            -float("inf"),
        )

        #
        # recompute P
        #
        p = tl.exp(
            scores - lse[:, None]
        )

        #
        # dP
        #
        dp = tl.dot(
            do,
            tl.trans(v),
        )

        #
        # dS
        #
        ds = p * (
            dp - delta[:, None]
        )

        #
        # [BN,BM] @ [BM,D]
        #
        dk += (
            tl.dot(
                tl.trans(ds).to(tl.float16),
                q,
            )
            * SCALE
        )

        #
        # [BN,BM] @ [BM,D]
        #
        dv += tl.dot(
            tl.trans(p).to(tl.float16),
            do,
        )

    dk_ptrs = (
        DK
        + batch * stride_dkb
        + head * stride_dkh
        + offs_n[:, None] * stride_dkt
        + offs_d[None, :] * stride_dkd
    )

    dv_ptrs = (
        DV
        + batch * stride_dvb
        + head * stride_dvh
        + offs_n[:, None] * stride_dvt
        + offs_d[None, :] * stride_dvd
    )

    tl.store(
        dk_ptrs,
        dk,
        mask=offs_n[:, None] < N_CTX,
    )

    tl.store(
        dv_ptrs,
        dv,
        mask=offs_n[:, None] < N_CTX,
    )

def triton_flash_attention_backward(
    q,
    k,
    v,
    out,
    do,
    lse,
):
    B, H, T, D = q.shape

    scale = D ** -0.5

    delta = torch.empty(
        (B, H, T),
        device=q.device,
        dtype=torch.float32,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    BLOCK_M = 32
    BLOCK_N = 32

    #
    # 1. delta
    #
    grid_q = (
        triton.cdiv(T, BLOCK_M),
        B * H,
    )

    _flash_attention_bwd_preprocess_kernel[grid_q](
        out,
        do,
        delta,

        *out.stride(),
        *do.stride(),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,

        num_warps=4,
    )

    #
    # 2. dQ
    #
    _flash_attention_bwd_dq_kernel[grid_q](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,

        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        *dq.stride(),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,
        SCALE=scale,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=4,
    )

    #
    # 3. dK,dV
    #
    grid_kv = (
        triton.cdiv(T, BLOCK_N),
        B * H,
    )

    _flash_attention_bwd_dkdv_kernel[grid_kv](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,

        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        *dk.stride(),
        *dv.stride(),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,
        SCALE=scale,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=4,
    )

    return dq, dk, dv

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

class TritonFlashAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        # 우리가 만든 Triton forward
        out, lse = triton_flash_attention_forward(
            q,
            k,
            v,
        )

        # backward에서 필요한 tensor 저장
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
        )

        # 사용자에게는 attention output만 반환
        return out

    @staticmethod
    def backward(ctx, do):
        # forward에서 저장해둔 tensor 복구
        q, k, v, out, lse = ctx.saved_tensors

        # 우리가 만든 Triton backward
        dq, dk, dv = triton_flash_attention_backward(
            q,
            k,
            v,
            out,
            do,
            lse,
        )

        # forward 입력이 q, k, v 세 개였으므로
        # 각각에 대한 gradient 반환
        return dq, dk, dv

def triton_flash_attention(q, k, v):
    return TritonFlashAttentionFunction.apply(
        q,
        k,
        v,
    )