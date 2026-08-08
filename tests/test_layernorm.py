import torch
import torch.nn.functional as F

from triton_kernels.layernorm import layer_norm


def test_layer_norm():
    torch.manual_seed(0)

    B = 4
    T = 128
    C = 768

    x = torch.randn(
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

    # PyTorch
    y_torch = F.layer_norm(
        x,
        normalized_shape=(C,),
        weight=weight,
        bias=bias,
        eps=1e-5,
    )

    # 우리가 만든 Triton
    y_triton = layer_norm(
        x,
        weight,
        bias,
        eps=1e-5,
    )

    print("torch:")
    print(y_torch[0, 0, :10])

    print("\ntriton:")
    print(y_triton[0, 0, :10])

    print("\nmax error:")
    print((y_torch - y_triton).abs().max().item())

    torch.testing.assert_close(
        y_triton,
        y_torch,
        rtol=1e-4,
        atol=1e-4,
    )

    print("\nPASS")


if __name__ == "__main__":
    test_layer_norm()