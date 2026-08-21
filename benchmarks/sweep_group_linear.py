import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear import (
    triton_linear_forward,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


SHAPES = [
    ("QKV",       768, 2304),
    ("attn_proj", 768, 768),
    ("mlp_fc",    768, 3072),
    ("mlp_proj", 3072, 768),
]


# tile은 이제 고정
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 4


GROUP_SIZES = [
    1,
    2,
    4,
    8,
    16,
]


def main():
    torch.manual_seed(0)

    totals = {
        g: 0.0
        for g in GROUP_SIZES
    }

    for name, K, N in SHAPES:

        print()
        print("=" * 80)
        print(
            f"{name}: "
            f"M={B*T}, K={K}, N={N}"
        )
        print("=" * 80)

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

        bias = torch.randn(
            N,
            device=DEVICE,
            dtype=DTYPE,
        )

        ref = F.linear(
            x,
            weight,
            bias,
        )

        def torch_fn():
            F.linear(
                x,
                weight,
                bias,
            )

        torch_ms = triton.testing.do_bench(
            torch_fn
        )

        print(
            f"PyTorch: {torch_ms:.4f} ms"
        )

        print()
        print(
            f"{'GROUP_M':>10}"
            f"{'Triton ms':>15}"
            f"{'vs Torch':>12}"
            f"{'max diff':>15}"
        )

        print("-" * 55)

        for group_size in GROUP_SIZES:

            out = triton_linear_forward(
                x,
                weight,
                bias,

                block_m=BLOCK_M,
                block_n=BLOCK_N,
                block_k=BLOCK_K,
                num_warps=NUM_WARPS,

                group_size_m=group_size,
            )

            max_diff = (
                out.float()
                - ref.float()
            ).abs().max().item()

            def fn():
                triton_linear_forward(
                    x,
                    weight,
                    bias,

                    block_m=BLOCK_M,
                    block_n=BLOCK_N,
                    block_k=BLOCK_K,
                    num_warps=NUM_WARPS,

                    group_size_m=group_size,
                )

            ms = triton.testing.do_bench(
                fn
            )

            totals[group_size] += ms

            print(
                f"{group_size:10d}"
                f"{ms:15.4f}"
                f"{torch_ms / ms:11.2f}x"
                f"{max_diff:15.6e}"
            )

    print()
    print("=" * 60)
    print("GLOBAL")
    print("=" * 60)

    ranking = sorted(
        (
            total,
            group_size,
        )
        for group_size, total
        in totals.items()
    )

    for rank, (
        total,
        group_size,
    ) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{rank}. "
            f"GROUP_SIZE_M={group_size:<2} "
            f"total={total:.4f} ms"
        )


if __name__ == "__main__":
    main()