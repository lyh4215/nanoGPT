import torch
import torch.nn.functional as F

from torch.profiler import profile, ProfilerActivity

from triton_kernels.residual_layernorm import residual_layer_norm


def profile_residual_layernorm(
    B=8,
    T=1024,
    C=768,
    dtype=torch.float32,
):
    x = torch.randn(
        B, T, C,
        device="cuda",
        dtype=dtype,
    )

    residual = torch.randn(
        B, T, C,
        device="cuda",
        dtype=dtype,
    )

    weight = torch.randn(
        C,
        device="cuda",
        dtype=dtype,
    )

    bias = torch.randn(
        C,
        device="cuda",
        dtype=dtype,
    )

    # warmup
    for _ in range(10):
        z = x + residual
        F.layer_norm(z, (C,), weight, bias)

        residual_layer_norm(
            x,
            residual,
            weight,
            bias,
            num_warps=8,
        )

    torch.cuda.synchronize()

    # -------------------------
    # PyTorch
    # -------------------------

    print("\n===== PyTorch Residual + LayerNorm =====")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    ) as prof:

        for _ in range(20):
            z = x + residual

            y = F.layer_norm(
                z,
                (C,),
                weight,
                bias,
                eps=1e-5,
            )

        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )

    # -------------------------
    # Triton
    # -------------------------

    print("\n===== Triton Fused Residual + LayerNorm =====")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    ) as prof:

        for _ in range(20):
            y = residual_layer_norm(
                x,
                residual,
                weight,
                bias,
                eps=1e-5,
                num_warps=8,
            )

        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )


if __name__ == "__main__":
    profile_residual_layernorm()