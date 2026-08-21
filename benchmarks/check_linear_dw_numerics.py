import torch

from triton_kernels.linear import (
    triton_linear_backward_dw,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024

# QKV shape
K = 768
N = 2304

M = B * T


torch.manual_seed(0)


# ============================================================
# Input
# ============================================================

dy = torch.randn(
    B,
    T,
    N,
    device=DEVICE,
    dtype=DTYPE,
)

x = torch.randn(
    B,
    T,
    K,
    device=DEVICE,
    dtype=DTYPE,
)


dy_2d = dy.reshape(
    -1,
    N,
)

x_2d = x.reshape(
    -1,
    K,
)


# ============================================================
# PyTorch FP16
# ============================================================

ref_fp16 = (
    dy_2d.T
    @ x_2d
)


# ============================================================
# PyTorch FP32 reference
# ============================================================

ref_fp32 = (
    dy_2d.float().T
    @ x_2d.float()
)


# ============================================================
# Triton dW
#
# 우선 대표 config 하나만
# ============================================================

out = triton_linear_backward_dw(
    dy,
    x,

    block_n=128,
    block_k=128,
    block_m=32,

    num_warps=4,
    group_size_n=1,
)


torch.cuda.synchronize()


# ============================================================
# Error 1:
# PyTorch FP16 vs FP32
# ============================================================

torch_err = (
    ref_fp16.float()
    - ref_fp32
).abs()

print()
print("[PyTorch FP16 vs FP32]")

print(
    "max :",
    torch_err.max().item(),
)

print(
    "mean:",
    torch_err.mean().item(),
)


# ============================================================
# Error 2:
# Triton vs FP32
# ============================================================

triton_err = (
    out.float()
    - ref_fp32
).abs()

print()
print("[Triton FP16 vs FP32]")

print(
    "max :",
    triton_err.max().item(),
)

print(
    "mean:",
    triton_err.mean().item(),
)


# ============================================================
# Error 3:
# Triton vs PyTorch FP16
# ============================================================

diff = (
    out.float()
    - ref_fp16.float()
).abs()

print()
print("[Triton vs PyTorch FP16]")

print(
    "max :",
    diff.max().item(),
)

print(
    "mean:",
    diff.mean().item(),
)


# ============================================================
# Magnitudes
# ============================================================

print()
print("[Magnitude]")

print(
    "FP32 ref abs max:",
    ref_fp32.abs().max().item(),
)

print(
    "Triton abs max:",
    out.float().abs().max().item(),
)


# ============================================================
# allclose도 참고용
# ============================================================

print()

print(
    "allclose 1e-2:",
    torch.allclose(
        out,
        ref_fp16,
        atol=1e-2,
        rtol=1e-2,
    )
)

print(
    "allclose 1e-1:",
    torch.allclose(
        out,
        ref_fp16,
        atol=1e-1,
        rtol=1e-2,
    )
)