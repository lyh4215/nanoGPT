import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear_gelu import (
    triton_linear_gelu_forward,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024
K = 768
N = 3072


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    x = torch.randn(
        B, T, K,
        device=DEVICE,
        dtype=DTYPE,
    )

    weight = torch.randn(
        N, K,
        device=DEVICE,
        dtype=DTYPE,
    )

    bias = torch.randn(
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # compile / warmup
    for _ in range(3):
        triton_linear_gelu_forward(
            x,
            weight,
            bias,
        )

        F.gelu(
            F.linear(
                x,
                weight,
                bias,
            )
        )

    torch.cuda.synchronize()

    def torch_fn():
        return F.gelu(
            F.linear(
                x,
                weight,
                bias,
            )
        )

    def triton_fn():
        return triton_linear_gelu_forward(
            x,
            weight,
            bias,
        )

    torch_ms = triton.testing.do_bench(
        torch_fn
    )

    triton_ms = triton.testing.do_bench(
        triton_fn
    )

    print()
    print("=" * 70)
    print("Linear + GELU Forward Benchmark")
    print("=" * 70)

    print(
        f"PyTorch : {torch_ms:.4f} ms"
    )

    print(
        f"Triton  : {triton_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_ms / triton_ms:.2f}x"
    )


if __name__ == "__main__":
    main()