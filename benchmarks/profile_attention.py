import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from triton_kernels.attention import naive_attention


B = 8
H = 12
T = 1024
D = 64

q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)


def sdpa():
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )


def naive():
    return naive_attention(
        q,
        k,
        v,
        causal=True,
    )


for _ in range(5):
    sdpa()
    naive()

torch.cuda.synchronize()


print("===== SDPA =====")

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]
) as prof:
    for _ in range(20):
        sdpa()

    torch.cuda.synchronize()

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
)


print("\n===== NAIVE =====")

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]
) as prof:
    for _ in range(20):
        naive()

    torch.cuda.synchronize()

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
)