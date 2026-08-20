import torch
import torch.nn.functional as F
import triton

from triton_kernels.attention import (
    triton_flash_attention_forward,
    triton_flash_attention_backward,

    _flash_attention_bwd_preprocess_kernel,
    _flash_attention_bwd_dq_kernel,
    _flash_attention_bwd_dkdv_kernel,
)


# ============================================================
# Config
# ============================================================

B = 8
H = 12
T = 1024
D = 64

DTYPE = torch.float16
DEVICE = "cuda"

# 현재 backward에서 쓰는 config와 동일하게 맞춰야 함
BLOCK_M = 32
BLOCK_N = 32
NUM_WARPS = 4


# ============================================================
# Input
# ============================================================

torch.manual_seed(0)

q = torch.randn(
    B, H, T, D,
    device=DEVICE,
    dtype=DTYPE,
)

k = torch.randn_like(q)
v = torch.randn_like(q)

do = torch.randn_like(q)

scale = D ** -0.5


# ============================================================
# 1. Triton Forward
# ============================================================

def bench_triton_forward():
    def fn():
        triton_flash_attention_forward(
            q,
            k,
            v,
        )

    return triton.testing.do_bench(fn)


# ============================================================
# backward에 필요한 forward 결과 준비
# ============================================================

out, lse = triton_flash_attention_forward(
    q,
    k,
    v,
)

torch.cuda.synchronize()


# ============================================================
# backward intermediate / output allocation
# ============================================================

delta = torch.empty(
    (B, H, T),
    device=DEVICE,
    dtype=torch.float32,
)

dq = torch.empty_like(q)
dk = torch.empty_like(k)
dv = torch.empty_like(v)


grid_q = (
    triton.cdiv(T, BLOCK_M),
    B * H,
)

grid_kv = (
    triton.cdiv(T, BLOCK_N),
    B * H,
)


# ============================================================
# 2. Triton Backward Total
# ============================================================

def bench_triton_backward_total():
    def fn():
        triton_flash_attention_backward(
            q,
            k,
            v,
            out,
            do,
            lse,
        )

    return triton.testing.do_bench(fn)


# ============================================================
# 3. Preprocess
#
# delta_i = sum_d O_id * dO_id
# ============================================================

def launch_preprocess():
    _flash_attention_bwd_preprocess_kernel[grid_q](
        out,
        do,
        delta,

        # O strides
        *out.stride(),

        # dO strides
        *do.stride(),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,

        num_warps=NUM_WARPS,
    )


def bench_preprocess():
    return triton.testing.do_bench(
        launch_preprocess
    )


# ============================================================
# delta는 dQ / dK,dV에서 필요하므로
# benchmark 밖에서 한 번 계산
# ============================================================

launch_preprocess()
torch.cuda.synchronize()


# ============================================================
# 4. dQ kernel
# ============================================================

def launch_dq():
    _flash_attention_bwd_dq_kernel[grid_q](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,

        # Q strides
        *q.stride(),

        # K strides
        *k.stride(),

        # V strides
        *v.stride(),

        # dO strides
        *do.stride(),

        # dQ strides
        *dq.stride(),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,
        SCALE=scale,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=NUM_WARPS,
    )


def bench_dq():
    return triton.testing.do_bench(
        launch_dq
    )


# ============================================================
# 5. dK / dV kernel
# ============================================================

def launch_dkdv():
    _flash_attention_bwd_dkdv_kernel[grid_kv](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dk,
        dv,

        # Q strides
        *q.stride(),

        # K strides
        *k.stride(),

        # V strides
        *v.stride(),

        # dO strides
        *do.stride(),

        # dK strides
        *dk.stride(),

        # dV strides
        *dv.stride(),

        H=H,
        N_CTX=T,
        HEAD_DIM=D,
        SCALE=scale,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,

        num_warps=NUM_WARPS,
    )


def bench_dkdv():
    return triton.testing.do_bench(
        launch_dkdv
    )


# ============================================================
# SDPA Forward
# ============================================================

def bench_sdpa_forward():
    def fn():
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

    return triton.testing.do_bench(fn)


# ============================================================
# SDPA Backward only
#
# forward graph는 benchmark 밖에서 한 번 생성.
# torch.autograd.grad()로 backward kernel만 반복 실행.
# ============================================================

q_ref = q.detach().clone().requires_grad_(True)
k_ref = k.detach().clone().requires_grad_(True)
v_ref = v.detach().clone().requires_grad_(True)

out_ref = F.scaled_dot_product_attention(
    q_ref,
    k_ref,
    v_ref,
    dropout_p=0.0,
    is_causal=True,
)


def bench_sdpa_backward():
    def fn():
        torch.autograd.grad(
            outputs=out_ref,
            inputs=(q_ref, k_ref, v_ref),
            grad_outputs=do,
            retain_graph=True,
        )

    return triton.testing.do_bench(fn)


# ============================================================
# Run
# ============================================================

torch.cuda.synchronize()

sdpa_fwd_ms = bench_sdpa_forward()
sdpa_bwd_ms = bench_sdpa_backward()

triton_fwd_ms = bench_triton_forward()
triton_bwd_ms = bench_triton_backward_total()

preprocess_ms = bench_preprocess()
dq_ms = bench_dq()
dkdv_ms = bench_dkdv()


# ============================================================
# Print
# ============================================================

component_sum = (
    preprocess_ms
    + dq_ms
    + dkdv_ms
)

print()
print("=" * 65)
print(
    f"B={B}, H={H}, T={T}, D={D}, "
    f"dtype={DTYPE}"
)
print("=" * 65)

print("\n[SDPA]")

print(
    f"Forward            : "
    f"{sdpa_fwd_ms:.4f} ms"
)

print(
    f"Backward           : "
    f"{sdpa_bwd_ms:.4f} ms"
)

print(
    f"Fwd + Bwd          : "
    f"{sdpa_fwd_ms + sdpa_bwd_ms:.4f} ms"
)


print("\n[Triton FlashAttention]")

print(
    f"Forward            : "
    f"{triton_fwd_ms:.4f} ms"
)

print(
    f"Backward total     : "
    f"{triton_bwd_ms:.4f} ms"
)

print(
    f"  preprocess       : "
    f"{preprocess_ms:.4f} ms"
)

print(
    f"  dQ               : "
    f"{dq_ms:.4f} ms"
)

print(
    f"  dK/dV            : "
    f"{dkdv_ms:.4f} ms"
)

print(
    f"  component sum    : "
    f"{component_sum:.4f} ms"
)

print(
    f"Fwd + Bwd          : "
    f"{triton_fwd_ms + triton_bwd_ms:.4f} ms"
)


print("\n[Ratio]")

print(
    f"Forward speedup    : "
    f"{sdpa_fwd_ms / triton_fwd_ms:.2f}x"
)

print(
    f"Backward speedup   : "
    f"{sdpa_bwd_ms / triton_bwd_ms:.2f}x"
)

print(
    f"Overall speedup    : "
    f"{(sdpa_fwd_ms + sdpa_bwd_ms) / (triton_fwd_ms + triton_bwd_ms):.2f}x"
)


print("\n[Backward breakdown]")

print(
    f"preprocess         : "
    f"{preprocess_ms / component_sum * 100:.1f}%"
)

print(
    f"dQ                 : "
    f"{dq_ms / component_sum * 100:.1f}%"
)

print(
    f"dK/dV              : "
    f"{dkdv_ms / component_sum * 100:.1f}%"
)