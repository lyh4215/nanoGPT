import math

import torch
import torch.nn.functional as F
import triton

from triton_kernels.attention import (
    triton_causal_softmax,
    triton_flash_attention_forward,
)


B = 8
H = 12
T = 1024
D = 64

dtype = torch.float16
device = "cuda"


q = torch.randn(
    B, H, T, D,
    device=device,
    dtype=dtype,
)

k = torch.randn_like(q)
v = torch.randn_like(q)


# ==========================================
# 1. PyTorch SDPA
# ==========================================

def sdpa_fn():
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )


# ==========================================
# 2. Fused-naive
#
# QK^T 자체는 HBM에 materialize
# softmax 결과도 HBM에 materialize
#
# 단:
# scale + mask + softmax 는 한 kernel
# ==========================================

def fused_naive_fn():

    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )
    # [B,H,T,T]

    probs = triton_causal_softmax(
        scores,
        1.0 / math.sqrt(D),
    )

    out = torch.matmul(
        probs,
        v,
    )

    return out


# ==========================================
# 3. Triton FlashAttention
#
# score/probs 전체를 만들지 않음
# ==========================================

def flash_fn():
    return triton_flash_attention_forward(
        q,
        k,
        v,
    )


# ==========================================
# correctness 한 번 더
# ==========================================

ref = sdpa_fn()
flash = flash_fn()
naive = fused_naive_fn()

print(
    "Flash max diff:",
    (ref.float() - flash.float())
    .abs()
    .max()
    .item()
)

print(
    "Naive max diff:",
    (ref.float() - naive.float())
    .abs()
    .max()
    .item()
)


# ==========================================
# warmup
# ==========================================

for _ in range(10):
    sdpa_fn()
    fused_naive_fn()
    flash_fn()

torch.cuda.synchronize()


# ==========================================
# benchmark
# ==========================================

sdpa_ms = triton.testing.do_bench(
    sdpa_fn,
    warmup=25,
    rep=100,
)

naive_ms = triton.testing.do_bench(
    fused_naive_fn,
    warmup=25,
    rep=100,
)

flash_ms = triton.testing.do_bench(
    flash_fn,
    warmup=25,
    rep=100,
)


print()
print(
    f"B={B}, H={H}, T={T}, D={D}, dtype={dtype}"
)

print(
    f"SDPA         : {sdpa_ms:.4f} ms"
)

print(
    f"Fused Naive  : {naive_ms:.4f} ms"
)

print(
    f"Triton Flash : {flash_ms:.4f} ms"
)

print()

print(
    f"Flash vs Naive : "
    f"{naive_ms / flash_ms:.2f}x"
)

print(
    f"Flash vs SDPA  : "
    f"{sdpa_ms / flash_ms:.2f}x"
)