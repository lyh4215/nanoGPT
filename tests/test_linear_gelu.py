import torch
import torch.nn.functional as F

from triton_kernels.linear_gelu import (
    triton_linear_gelu_forward,
)


DEVICE = "cuda"
DTYPE = torch.float16


def test_case(use_bias):
    torch.manual_seed(0)

    # GPT-2 MLP c_fc shape
    B = 2
    T = 128

    K = 768
    N = 3072

    x = torch.randn(
        B,
        T,
        K,
        device=DEVICE,
        dtype=DTYPE,
    )

    weight = torch.randn(
        N,
        K,
        device=DEVICE,
        dtype=DTYPE,
    )

    bias = (
        torch.randn(
            N,
            device=DEVICE,
            dtype=DTYPE,
        )
        if use_bias
        else None
    )

    # ========================================================
    # PyTorch reference
    # ========================================================

    ref = F.gelu(
        F.linear(
            x,
            weight,
            bias,
        )
    )

    # ========================================================
    # Triton fused Linear + GELU
    # ========================================================

    out_y, out_z = triton_linear_gelu_forward(
        x,
        weight,
        bias,
    )

    # ========================================================
    # Compare
    # ========================================================

    diff = (
        out_y.float()
        - ref.float()
    ).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print()
    print(
        f"use_bias={use_bias}"
    )

    print(
        f"max diff  : {max_diff:.6e}"
    )

    print(
        f"mean diff : {mean_diff:.6e}"
    )

    print(
        "allclose  :",
        torch.allclose(
            out_y,
            ref,
            atol=1e-2,
            rtol=1e-2,
        )
    )


def main():
    print("=" * 70)
    print("Triton Linear + GELU Forward Correctness")
    print("=" * 70)

    test_case(
        use_bias=True,
    )

    test_case(
        use_bias=False,
    )


if __name__ == "__main__":
    main()