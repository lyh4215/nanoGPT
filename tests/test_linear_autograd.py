import torch
import torch.nn.functional as F

from triton_kernels.linear import (
    triton_linear,
)


torch.manual_seed(0)

B = 2
T = 128

K = 768
N = 2304

device = "cuda"
dtype = torch.float16


# ============================================================
# Input
# ============================================================

x = torch.randn(
    B,
    T,
    K,
    device=device,
    dtype=dtype,
)

weight = torch.randn(
    N,
    K,
    device=device,
    dtype=dtype,
)

bias = torch.randn(
    N,
    device=device,
    dtype=dtype,
)

dy = torch.randn(
    B,
    T,
    N,
    device=device,
    dtype=dtype,
)


# ============================================================
# PyTorch reference
# ============================================================

x_ref = (
    x.detach()
    .clone()
    .requires_grad_(True)
)

w_ref = (
    weight.detach()
    .clone()
    .requires_grad_(True)
)

b_ref = (
    bias.detach()
    .clone()
    .requires_grad_(True)
)


y_ref = F.linear(
    x_ref,
    w_ref,
    b_ref,
)

y_ref.backward(dy)


# ============================================================
# Triton
# ============================================================

x_tri = (
    x.detach()
    .clone()
    .requires_grad_(True)
)

w_tri = (
    weight.detach()
    .clone()
    .requires_grad_(True)
)

b_tri = (
    bias.detach()
    .clone()
    .requires_grad_(True)
)


y_tri = triton_linear(
    x_tri,
    w_tri,
    b_tri,
)

# 직접 Triton backward 호출 안 함.
#
# PyTorch autograd가
# TritonLinearFunction.backward()
# 호출.
y_tri.backward(dy)


# ============================================================
# Compare
# ============================================================

print(
    "Forward:",
    (y_ref - y_tri)
    .abs()
    .max()
    .item(),
)

print(
    "dX:",
    (x_ref.grad - x_tri.grad)
    .abs()
    .max()
    .item(),
)

print(
    "dW:",
    (w_ref.grad - w_tri.grad)
    .abs()
    .max()
    .item(),
)

print(
    "db:",
    (b_ref.grad - b_tri.grad)
    .abs()
    .max()
    .item(),
)