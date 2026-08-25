import torch
import triton

from triton_kernels.linear import (
    _linear_bwd_dx_kernel,
)


DEVICE = "cuda"
DTYPE = torch.float16

# ============================================================
# GPT-2 MLP c_fc backward dX
#
# Forward:
#   X  [M, K]     = [8192, 768]
#   W  [N, K]     = [3072, 768]
#   Y  [M, N]     = [8192, 3072]
#
# Backward:
#   dY [M, N]
#   W  [N, K]
#
#   dX = dY @ W
#
#   dX [M, K]
# ============================================================

B = 8
T = 1024

M = B * T
N = 3072
K = 768


# ============================================================
# Config
#
# dX output tile:
#   [BLOCK_M, BLOCK_K]
#
# reduction:
#   BLOCK_N
#
# (BM, BK, BN, num_warps)
# ============================================================

CONFIGS = [
    # --------------------------------------------------------
    # 기존 best 주변
    # --------------------------------------------------------
    (128, 128, 32, 4),
    (128, 128, 64, 4),
    (128, 128, 128, 4),

    # --------------------------------------------------------
    # smaller K output tile
    # --------------------------------------------------------
    (128, 64, 32, 4),
    (128, 64, 64, 4),
    (128, 64, 128, 4),

    # --------------------------------------------------------
    # smaller M output tile
    # --------------------------------------------------------
    (64, 128, 32, 4),
    (64, 128, 64, 4),
    (64, 128, 128, 4),

    # --------------------------------------------------------
    # both smaller
    # --------------------------------------------------------
    (64, 64, 32, 4),
    (64, 64, 64, 4),
    (64, 64, 128, 4),

    # --------------------------------------------------------
    # 8 warps
    # --------------------------------------------------------
    (128, 128, 32, 8),
    (128, 128, 64, 8),

    (128, 64, 32, 8),
    (128, 64, 64, 8),

    (64, 128, 32, 8),
    (64, 128, 64, 8),
]


GROUPS = [
    1,
    2,
    4,
    8,
]


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Inputs
    # ========================================================

    dy = torch.randn(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    weight = torch.randn(
        N,
        K,
        device=DEVICE,
        dtype=DTYPE,
    )

    dx = torch.empty(
        M,
        K,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # PyTorch reference
    #
    # dX = dY @ W
    # ========================================================

    def torch_fn():
        return torch.mm(
            dy,
            weight,
        )

    for _ in range(3):
        torch_fn()

    torch.cuda.synchronize()

    torch_ms = triton.testing.do_bench(
        torch_fn
    )

    print()
    print("=" * 105)
    print(
        f"Linear dX MLP c_fc sweep "
        f"M={M}, N={N}, K={K}"
    )
    print("=" * 105)

    print()
    print(
        f"PyTorch dX : {torch_ms:.4f} ms"
    )

    print()

    print(
        f"{'BM':>5} "
        f"{'BK':>5} "
        f"{'BN':>5} "
        f"{'W':>4} "
        f"{'G':>4} "
        f"{'ms':>10} "
        f"{'vs Torch':>10}"
    )

    print("-" * 72)

    results = []

    # ========================================================
    # Sweep
    # ========================================================

    for (
        block_m,
        block_k,
        block_n,
        num_warps,
    ) in CONFIGS:

        for group_m in GROUPS:

            num_pid_m = triton.cdiv(
                M,
                block_m,
            )

            num_pid_k = triton.cdiv(
                K,
                block_k,
            )

            grid = (
                num_pid_m
                * num_pid_k,
            )

            def triton_fn():
                _linear_bwd_dx_kernel[grid](
                    dy,
                    weight,
                    dx,

                    M=M,
                    N=N,
                    K=K,

                    stride_dym=dy.stride(0),
                    stride_dyn=dy.stride(1),

                    stride_wn=weight.stride(0),
                    stride_wk=weight.stride(1),

                    stride_dxm=dx.stride(0),
                    stride_dxk=dx.stride(1),

                    BLOCK_M=block_m,
                    BLOCK_K=block_k,

                    # reduction dimension
                    BLOCK_N=block_n,

                    GROUP_SIZE_M=group_m,

                    num_warps=num_warps,
                )

            try:
                # compile
                triton_fn()
                torch.cuda.synchronize()

                ms = triton.testing.do_bench(
                    triton_fn
                )

                speedup = (
                    torch_ms
                    / ms
                )

                results.append(
                    (
                        ms,
                        block_m,
                        block_k,
                        block_n,
                        num_warps,
                        group_m,
                        speedup,
                    )
                )

                print(
                    f"{block_m:>5} "
                    f"{block_k:>5} "
                    f"{block_n:>5} "
                    f"{num_warps:>4} "
                    f"{group_m:>4} "
                    f"{ms:>10.4f} "
                    f"{speedup:>9.2f}x"
                )

            except Exception as e:
                print(
                    f"{block_m:>5} "
                    f"{block_k:>5} "
                    f"{block_n:>5} "
                    f"{num_warps:>4} "
                    f"{group_m:>4} "
                    f"{'FAIL':>10} "
                    f"{type(e).__name__}"
                )

    # ========================================================
    # Ranking
    # ========================================================

    results.sort(
        key=lambda x: x[0]
    )

    print()
    print("=" * 105)
    print("Top configurations")
    print("=" * 105)

    for rank, result in enumerate(
        results[:12],
        start=1,
    ):
        (
            ms,
            block_m,
            block_k,
            block_n,
            num_warps,
            group_m,
            speedup,
        ) = result

        print(
            f"[{rank:02d}] "
            f"BM={block_m:<3} "
            f"BK={block_k:<3} "
            f"BN={block_n:<3} "
            f"warps={num_warps} "
            f"G={group_m:<2} "
            f": {ms:.4f} ms "
            f"({speedup:.2f}x Torch)"
        )

    # ========================================================
    # Best
    # ========================================================

    if results:
        (
            best_ms,
            best_bm,
            best_bk,
            best_bn,
            best_warps,
            best_group,
            best_speedup,
        ) = results[0]

        print()
        print("=" * 105)
        print("BEST")
        print("=" * 105)

        print(
            f"BM={best_bm}, "
            f"BK={best_bk}, "
            f"BN={best_bn}, "
            f"warps={best_warps}, "
            f"GROUP_M={best_group}"
        )

        print(
            f"Triton  : {best_ms:.4f} ms"
        )

        print(
            f"PyTorch : {torch_ms:.4f} ms"
        )

        print(
            f"Speedup : {best_speedup:.2f}x"
        )

        print()

        print(
            "Current config for comparison:"
        )

        print(
            "BM=128, BK=128, BN=64, "
            "warps=4, GROUP_M=2"
        )


if __name__ == "__main__":
    main()