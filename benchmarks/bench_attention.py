import torch
import triton
import torch.nn.functional as F

from triton_kernels.attention import naive_attention


B = 8
H = 12
T = 1024
D = 64

q = torch.randn(
    B, H, T, D,
    device="cuda",
    dtype=torch.float32,
)

k = torch.randn_like(q)
v = torch.randn_like(q)


def torch_sdpa():
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )


def torch_naive():
    return naive_attention(
        q,
        k,
        v,
        causal=True,
    )


# warmup
for _ in range(5):
    torch_sdpa()
    torch_naive()

torch.cuda.synchronize()


sdpa_ms = triton.testing.do_bench(
    torch_sdpa,
    warmup=25,
    rep=100,
)

naive_ms = triton.testing.do_bench(
    torch_naive,
    warmup=25,
    rep=100,
)


print(f"SDPA  : {sdpa_ms:.4f} ms")
print(f"Naive : {naive_ms:.4f} ms")
print(f"Ratio : {naive_ms / sdpa_ms:.2f}x slower")