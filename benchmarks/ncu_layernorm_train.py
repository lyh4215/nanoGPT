import argparse

import torch
import torch.nn.functional as F

from triton_kernels.layernorm import layer_norm_autograd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--impl",
        choices=["torch", "triton"],
        required=True,
    )

    args = parser.parse_args()

    B = 8
    T = 1024
    C = 768
    eps = 1e-5

    x = torch.randn(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    weight = torch.randn(
        C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    bias = torch.randn(
        C,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    dy = torch.randn_like(x)

    # tensor 생성/RNG 작업 끝내기
    torch.cuda.synchronize()

    if args.impl == "torch":

        y = F.layer_norm(
            x,
            (C,),
            weight,
            bias,
            eps,
        )

    else:

        y = layer_norm_autograd(
            x,
            weight,
            bias,
            eps,
        )

    torch.autograd.grad(
        y,
        (x, weight, bias),
        grad_outputs=dy,
    )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()