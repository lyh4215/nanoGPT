import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from triton_kernels.layernorm import layer_norm_autograd


B = 8
T = 1024
C = 768
eps = 1e-5

x0 = torch.randn(
    B, T, C,
    device="cuda",
    dtype=torch.float32,
)

w0 = torch.randn(
    C,
    device="cuda",
    dtype=torch.float32,
)

b0 = torch.randn(
    C,
    device="cuda",
    dtype=torch.float32,
)

dy = torch.randn_like(x0)


def torch_fn():
    x = x0.detach().requires_grad_(True)
    w = w0.detach().requires_grad_(True)
    b = b0.detach().requires_grad_(True)

    y = F.layer_norm(
        x,
        (C,),
        w,
        b,
        eps,
    )

    torch.autograd.grad(
        y,
        (x, w, b),
        grad_outputs=dy,
    )


def triton_fn():
    x = x0.detach().requires_grad_(True)
    w = w0.detach().requires_grad_(True)
    b = b0.detach().requires_grad_(True)

    y = layer_norm_autograd(
        x,
        w,
        b,
        eps,
    )

    torch.autograd.grad(
        y,
        (x, w, b),
        grad_outputs=dy,
    )


# warmup
for _ in range(10):
    torch_fn()
    triton_fn()

torch.cuda.synchronize()


print("===== PyTorch LayerNorm Forward + Backward =====")

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]
) as prof:

    for _ in range(20):
        torch_fn()

    torch.cuda.synchronize()

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
)


print("\n===== Triton LayerNorm Forward + Backward =====")

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]
) as prof:

    for _ in range(20):
        triton_fn()

    torch.cuda.synchronize()

print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
)