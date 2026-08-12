import torch
import torch.nn.functional as F

from triton_kernels.gelu import gelu


def test_gelu():
    torch.manual_seed(0)

    B = 8
    T = 1024
    C = 3072

    x = torch.randn(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
    )

    y_torch = F.gelu(
        x,
        approximate="none",
    )

    y_triton = gelu(x)

    print(
        "max error:",
        (y_torch - y_triton).abs().max().item(),
    )

    torch.testing.assert_close(
        y_triton,
        y_torch,
        rtol=1e-4,
        atol=1e-4,
    )

    print("GELU correctness: PASS")


if __name__ == "__main__":
    test_gelu()