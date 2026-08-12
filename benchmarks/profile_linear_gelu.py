import torch
import torch.nn.functional as F

from torch.profiler import profile, ProfilerActivity

from triton_kernels.linear_gelu import linear_gelu


def profile_linear_gelu(
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

    # ---------------------------
    # Warmup
    # ---------------------------

    for _ in range(10):
        y = F.linear(x, weight, bias)
        y = F.gelu(y, approximate="none")

        y = linear_gelu(
            x,
            weight,
            bias,
        )

    torch.cuda.synchronize()

    # ===========================
    # PyTorch
    # ===========================

    print("\n===== PyTorch Linear + GELU =====")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    ) as prof:

        for _ in range(20):

            y = F.linear(
                x,
                weight,
                bias,
            )

            y = F.gelu(
                y,
                approximate="none",
            )

        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )

    # ===========================
    # Triton
    # ===========================

    print("\n===== Triton Fused Linear + GELU =====")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    ) as prof:

        for _ in range(20):

            y = linear_gelu(
                x,
                weight,
                bias,
            )

        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )


if __name__ == "__main__":
    profile_linear_gelu()