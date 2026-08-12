import torch
import torch.nn.functional as F

from triton_kernels.residual_layernorm import residual_layer_norm


def test_residual_layer_norm():
    torch.manual_seed(0)

    B = 4
    T = 128
    C = 768

    x = torch.randn(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
    )

    residual = torch.randn(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.randn(
        C,
        device="cuda",
        dtype=torch.float32,
    )

    bias = torch.randn(
        C,
        device="cuda",
        dtype=torch.float32,
    )

    # PyTorch reference
    z = x + residual

    y_torch = F.layer_norm(
        z,
        (C,),
        weight,
        bias,
        eps=1e-5,
    )

    # Triton fused
    y_triton = residual_layer_norm(
        x,
        residual,
        weight,
        bias,
        eps=1e-5,
        num_warps=8,
    )

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

    print("Residual + LayerNorm correctness: PASS")


if __name__ == "__main__":
    test_residual_layer_norm()