import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from triton_kernels.layernorm import layer_norm


def profile_layernorm(B=8, T=1024, C=768):
    x = torch.randn(B, T, C, device="cuda")
    weight = torch.randn(C, device="cuda")
    bias = torch.randn(C, device="cuda")

    # JIT compile + GPU warmup
    for _ in range(10):
        F.layer_norm(x, (C,), weight, bias)
        layer_norm(x, weight, bias, num_warps=8)

    torch.cuda.synchronize()

    print("\n===== PyTorch LayerNorm =====")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(20):
            F.layer_norm(x, (C,), weight, bias)

        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=15,
        )
    )

    print("\n===== Triton LayerNorm =====")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(20):
            layer_norm(
                x,
                weight,
                bias,
                num_warps=8,
            )

        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=15,
        )
    )


if __name__ == "__main__":
    profile_layernorm()