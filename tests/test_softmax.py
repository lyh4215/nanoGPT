import torch

from triton_kernels.attention import triton_softmax


B = 8
H = 12
T = 1024

x = torch.randn(
    B,
    H,
    T,
    T,
    device="cuda",
    dtype=torch.float32,
)

ref = torch.softmax(
    x,
    dim=-1,
)

out = triton_softmax(x)

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