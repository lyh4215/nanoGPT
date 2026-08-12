import argparse

import torch
import torch.nn.functional as F

from triton_kernels.residual_layernorm import residual_layer_norm


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

    x = torch.empty(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
    )

    residual = torch.empty_like(x)

    weight = torch.empty(
        C,
        device="cuda",
        dtype=torch.float32,
    )

    bias = torch.empty(
        C,
        device="cuda",
        dtype=torch.float32,
    )

    if args.impl == "torch":
        z = x + residual

        y = F.layer_norm(
            z,
            (C,),
            weight,
            bias,
            eps=1e-5,
        )

    else:
        y = residual_layer_norm(
            x,
            residual,
            weight,
            bias,
            eps=1e-5,
            num_warps=8,
        )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()