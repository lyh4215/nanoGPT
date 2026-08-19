import torch

from triton_kernels.attention import (
    triton_flash_attention_forward,
)


B = 8
H = 12
T = 1024
D = 64

q = torch.randn(
    B, H, T, D,
    device="cuda",
    dtype=torch.float16,
)

k = torch.randn_like(q)
v = torch.randn_like(q)

# compile 먼저
triton_flash_attention_forward(q, k, v)

torch.cuda.synchronize()

# profiling 대상 1회
triton_flash_attention_forward(q, k, v)

torch.cuda.synchronize()