import torch
import torch.nn.functional as F

from triton_kernels.linear import (
    triton_linear_forward,
)


torch.manual_seed(0)

B = 2
T = 128

K = 768
N = 2304

dtype = torch.float16
device = "cuda"


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


ref = F.linear(
    x,
    weight,
    bias,
)

out = triton_linear_forward(
    x,
    weight,
    bias,
)


diff = (
    ref.float()
    - out.float()
).abs()


print(
    "max diff :",
    diff.max().item(),
)

print(
    "mean diff:",
    diff.mean().item(),
)

print(
    "allclose :",
    torch.allclose(
        ref,
        out,
        atol=1e-2,
        rtol=1e-2,
    ),
)