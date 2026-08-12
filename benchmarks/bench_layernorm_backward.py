import torch
import torch.nn.functional as F
import triton

from triton_kernels.layernorm import layer_norm_autograd


def main():
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

    # backward를 반복하려면 매번 새 graph 필요
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

    torch_ms = triton.testing.do_bench(
        torch_fn,
        return_mode="median",
    )

    triton_ms = triton.testing.do_bench(
        triton_fn,
        return_mode="median",
    )

    print(f"B={B}, T={T}, C={C}")
    print(f"PyTorch : {torch_ms:.4f} ms")
    print(f"Triton  : {triton_ms:.4f} ms")
    print(f"Speedup : {torch_ms / triton_ms:.2f}x")


if __name__ == "__main__":
    main()