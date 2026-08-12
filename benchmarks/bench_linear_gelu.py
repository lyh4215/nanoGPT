# benchmarks/bench_linear_gelu.py

import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear_gelu import linear_gelu


def bench_linear_gelu(
    B=8,
    T=1024,
    C=768,
    dtype=torch.float16,
):
    N = 4 * C

    x = torch.randn(
        B, T, C,
        device="cuda",
        dtype=dtype,
    )

    weight = torch.randn(
        N, C,
        device="cuda",
        dtype=dtype,
    )

    bias = torch.randn(
        N,
        device="cuda",
        dtype=dtype,
    )

    def torch_fn():
        y = F.linear(
            x,
            weight,
            bias,
        )

        return F.gelu(
            y,
            approximate="none",
        )

    def triton_fn():
        return linear_gelu(
            x,
            weight,
            bias,
        )

    y_torch = torch_fn()
    y_triton = triton_fn()

    torch.testing.assert_close(
        y_triton,
        y_torch,
        rtol=1e-2,
        atol=1e-2,
    )

    torch_ms = triton.testing.do_bench(
        torch_fn,
        return_mode="median",
    )

    triton_ms = triton.testing.do_bench(
        triton_fn,
        return_mode="median",
    )

    print(
        f"B={B}, T={T}, C={C}, dtype={dtype}"
    )
    print(f"PyTorch : {torch_ms:.4f} ms")
    print(f"Triton  : {triton_ms:.4f} ms")
    print(
        f"Speedup : {torch_ms / triton_ms:.2f}x"
    )


if __name__ == "__main__":
    bench_linear_gelu()