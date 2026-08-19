import torch
import triton

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


def torch_fn():
    return torch.softmax(
        x,
        dim=-1,
    )


def triton_fn():
    return triton_softmax(x)


for _ in range(5):
    torch_fn()
    triton_fn()

torch.cuda.synchronize()


torch_ms = triton.testing.do_bench(
    torch_fn,
    warmup=25,
    rep=100,
)

triton_ms = triton.testing.do_bench(
    triton_fn,
    warmup=25,
    rep=100,
)


print(f"PyTorch : {torch_ms:.4f} ms")
print(f"Triton  : {triton_ms:.4f} ms")
print(f"Speedup : {torch_ms / triton_ms:.2f}x")