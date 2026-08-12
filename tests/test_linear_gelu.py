import torch
import torch.nn.functional as F

from triton_kernels.linear_gelu import linear_gelu


def test_linear_gelu():
    torch.manual_seed(0)

    B = 2
    T = 128

    C = 768
    N = 4 * C

    x = torch.randn(
        B,
        T,
        C,
        device="cuda",
        dtype=torch.float16,
    )

    weight = torch.randn(
        N,
        C,
        device="cuda",
        dtype=torch.float16,
    )

    bias = torch.randn(
        N,
        device="cuda",
        dtype=torch.float16,
    )

    # PyTorch reference
    y_torch = F.linear(
        x,
        weight,
        bias,
    )

    y_torch = F.gelu(
        y_torch,
        approximate="none",
    )

    # Triton fused
    y_triton = linear_gelu(
        x,
        weight,
        bias,
    )

    print("shape:", y_triton.shape)

    print(
        "max error:",
        (y_torch - y_triton)
        .abs()
        .max()
        .item()
    )

    torch.testing.assert_close(
        y_triton,
        y_torch,
        rtol=1e-2,
        atol=1e-2,
    )

    print("Linear + GELU correctness: PASS")


if __name__ == "__main__":
    test_linear_gelu()