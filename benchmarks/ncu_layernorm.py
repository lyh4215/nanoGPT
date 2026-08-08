import argparse

import torch
import torch.nn.functional as F

from triton_kernels.layernorm import layer_norm


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

    # randn을 쓰면 RNG CUDA kernel도 profiler에 잡히니까
    # 데이터 값 자체가 중요하지 않은 성능 profiling에서는 empty 사용
    x = torch.empty(
        B, T, C,
        device="cuda",
        dtype=torch.float32,
    )
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
        y = F.layer_norm(
            x,
            (C,),
            weight,
            bias,
            eps=1e-5,
        )

    else:
        y = layer_norm(
            x,
            weight,
            bias,
            eps=1e-5,
            num_warps=8,
        )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()