import torch
import torch.nn.functional as F

from triton_kernels.attention import triton_flash_attention


torch.manual_seed(0)

B = 2
H = 2
T = 128
D = 64

dtype = torch.float16
device = "cuda"


# ============================================================
# 동일한 입력 생성
# ============================================================

q = torch.randn(
    B, H, T, D,
    device=device,
    dtype=dtype,
)

k = torch.randn_like(q)
v = torch.randn_like(q)

do = torch.randn_like(q)


# ============================================================
# PyTorch reference
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

out_ref.backward(do)

dq_ref = q_ref.grad
dk_ref = k_ref.grad
dv_ref = v_ref.grad


# ============================================================
# Triton custom autograd.Function
# ============================================================

q_tri = q.detach().clone().requires_grad_(True)
k_tri = k.detach().clone().requires_grad_(True)
v_tri = v.detach().clone().requires_grad_(True)

out_tri = triton_flash_attention(
    q_tri,
    k_tri,
    v_tri,
)

# 중요:
# 우리가 backward()를 직접 호출하지 않는다.
out_tri.backward(do)

dq_tri = q_tri.grad
dk_tri = k_tri.grad
dv_tri = v_tri.grad


# ============================================================
# Compare
# ============================================================

print(
    "Forward max diff:",
    (out_tri - out_ref).abs().max().item(),
)

print(
    "dQ max diff:",
    (dq_tri - dq_ref).abs().max().item(),
)

print(
    "dK max diff:",
    (dk_tri - dk_ref).abs().max().item(),
)

print(
    "dV max diff:",
    (dv_tri - dv_ref).abs().max().item(),
)