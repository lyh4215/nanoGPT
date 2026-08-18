import torch
import torch.nn.functional as F

from triton_kernels.attention import naive_attention


B = 2
H = 4
T = 128
D = 64

q = torch.randn(
    B, H, T, D,
    device="cuda",
    dtype=torch.float32,
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

out = naive_attention(
    q,
    k,
    v,
    causal=True,
)


print(
    "max diff:",
    (ref - out).abs().max().item(),
)

print(
    "allclose:",
    torch.allclose(
        ref,
        out,
        atol=1e-5,
        rtol=1e-5,
    ),
)