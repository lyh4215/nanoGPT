import torch
import torch.nn.functional as F

from triton_kernels.attention import (
    triton_flash_attention_forward,
)


torch.manual_seed(1337)


B = 2
H = 4
T = 128
D = 64


q = torch.randn(
    B,
    H,
    T,
    D,
    device="cuda",
    dtype=torch.float16,
)

k = torch.randn_like(q)
v = torch.randn_like(q)


ref = F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)


out = triton_flash_attention_forward(
    q,
    k,
    v,
)


diff = (
    ref.float()
    - out.float()
).abs()


print(
    "max diff:",
    diff.max().item(),
)

print(
    "mean diff:",
    diff.mean().item(),
)

print(
    "allclose:",
    torch.allclose(
        ref,
        out,
        atol=1e-2,
        rtol=0.0,
    ),
)