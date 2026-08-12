import argparse

import torch
import torch.nn.functional as F

from triton_kernels.linear_gelu import linear_gelu


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
    N = 4 * C

    dtype = torch.float16

    # profiling 대상 이외의 RNG kernel을 만들지 않으려고 empty 사용
    x = torch.empty(
        B, T, C,
        device="cuda",
        dtype=dtype,
    )

    weight = torch.empty(
        N, C,
        device="cuda",
        dtype=dtype,
    )

    bias = torch.empty(
        N,
        device="cuda",
        dtype=dtype,
    )

    if args.impl == "torch":

        y = F.linear(
            x,
            weight,
            bias,
        )

        y = F.gelu(
            y,
            approximate="none",
        )

    else:

        y = linear_gelu(
            x,
            weight,
            bias,
        )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()