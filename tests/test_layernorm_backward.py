import torch
import torch.nn.functional as F

from triton_kernels.layernorm import (
    layer_norm_autograd,
)


def main():
    torch.manual_seed(0)

    B = 2
    T = 128
    C = 768

    eps = 1e-5

    # ---------------------
    # Triton inputs
    # ---------------------

    x_tri = torch.randn(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    w_tri = torch.randn(
        C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    b_tri = torch.randn(
        C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    # 동일한 PyTorch inputs
    x_ref = (
        x_tri.detach()
        .clone()
        .requires_grad_(True)
    )

    w_ref = (
        w_tri.detach()
        .clone()
        .requires_grad_(True)
    )

    b_ref = (
        b_tri.detach()
        .clone()
        .requires_grad_(True)
    )

    # upstream gradient
    dy = torch.randn_like(x_tri)

    # ---------------------
    # forward
    # ---------------------

    y_tri = layer_norm_autograd(
        x_tri,
        w_tri,
        b_tri,
        eps,
    )

    y_ref = F.layer_norm(
        x_ref,
        (C,),
        w_ref,
        b_ref,
        eps,
    )

    # ---------------------
    # backward
    # ---------------------

    y_tri.backward(
        dy,
    )

    y_ref.backward(
        dy,
    )

    # ---------------------
    # errors
    # ---------------------

    print(
        "forward max error:",
        (y_tri - y_ref)
        .abs()
        .max()
        .item()
    )

    print(
        "dx max error:",
        (x_tri.grad - x_ref.grad)
        .abs()
        .max()
        .item()
    )

    print(
        "dw max error:",
        (w_tri.grad - w_ref.grad)
        .abs()
        .max()
        .item()
    )

    print(
        "db max error:",
        (b_tri.grad - b_ref.grad)
        .abs()
        .max()
        .item()
    )

    torch.testing.assert_close(
        y_tri,
        y_ref,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        x_tri.grad,
        x_ref.grad,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        w_tri.grad,
        w_ref.grad,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        b_tri.grad,
        b_ref.grad,
        rtol=1e-4,
        atol=1e-4,
    )

    print("LayerNorm backward: PASS")


if __name__ == "__main__":
    main()