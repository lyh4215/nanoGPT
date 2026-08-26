import torch
import triton

from triton_kernels.linear_residual import (
    _linear_residual_fwd_kernel,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024
M = B * T


# ============================================================
# Candidate configs
#
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
# ============================================================

CONFIGS = [
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (128, 64, 32, 4),
    (128, 128, 32, 4),

    (64, 64, 64, 4),
    (64, 128, 64, 4),
    (128, 64, 64, 4),
    (128, 128, 64, 4),

    (64, 64, 32, 8),
    (64, 128, 32, 8),
    (128, 64, 32, 8),
    (128, 128, 32, 8),

    (64, 64, 64, 8),
    (64, 128, 64, 8),
    (128, 64, 64, 8),
    (128, 128, 64, 8),
]

GROUPS = [
    1,
    2,
    4,
    8,
]


SHAPES = [
    (
        "Attention c_proj + residual",
        768,   # K
        768,   # N
    ),
    (
        "MLP c_proj + residual",
        3072,  # K
        768,   # N
    ),
]


def run_shape(
    name,
    K,
    N,
):
    torch.manual_seed(0)

    # ========================================================
    # Inputs
    # ========================================================

    x = torch.randn(
        M,
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

    residual = torch.randn(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    y = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # PyTorch baseline
    #
    # Linear + residual
    # ========================================================

    def torch_fn():
        return (
            torch.nn.functional.linear(
                x,
                weight,
                bias,
            )
            + residual
        )

    for _ in range(3):
        torch_fn()

    torch.cuda.synchronize()

    torch_ms = triton.testing.do_bench(
        torch_fn
    )

    print()
    print("=" * 100)
    print(
        f"{name} sweep "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 100)

    print()
    print(
        f"PyTorch Linear + Residual : "
        f"{torch_ms:.4f} ms"
    )

    print()
    print(
        f"{'BM':>5} "
        f"{'BN':>5} "
        f"{'BK':>5} "
        f"{'W':>4} "
        f"{'G':>4} "
        f"{'ms':>10} "
        f"{'vs Torch':>10}"
    )

    print("-" * 70)

    results = []

    # ========================================================
    # Sweep
    # ========================================================

    for (
        block_m,
        block_n,
        block_k,
        num_warps,
    ) in CONFIGS:

        for group_m in GROUPS:
            grid = (
                triton.cdiv(
                    M,
                    block_m,
                )
                * triton.cdiv(
                    N,
                    block_n,
                ),
            )

            def triton_fn():
                _linear_residual_fwd_kernel[grid](
                    x,
                    weight,
                    bias,
                    residual,
                    y,

                    M=M,
                    N=N,
                    K=K,

                    stride_xm=x.stride(0),
                    stride_xk=x.stride(1),

                    stride_wn=weight.stride(0),
                    stride_wk=weight.stride(1),

                    stride_rm=residual.stride(0),
                    stride_rn=residual.stride(1),

                    stride_ym=y.stride(0),
                    stride_yn=y.stride(1),

                    HAS_BIAS=True,

                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,

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
                        block_n,
                        block_k,
                        num_warps,
                        group_m,
                        speedup,
                    )
                )

                print(
                    f"{block_m:>5} "
                    f"{block_n:>5} "
                    f"{block_k:>5} "
                    f"{num_warps:>4} "
                    f"{group_m:>4} "
                    f"{ms:>10.4f} "
                    f"{speedup:>9.2f}x"
                )

            except Exception as e:
                print(
                    f"{block_m:>5} "
                    f"{block_n:>5} "
                    f"{block_k:>5} "
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
    print("=" * 100)
    print("Top configurations")
    print("=" * 100)

    for rank, result in enumerate(
        results[:10],
        start=1,
    ):
        (
            ms,
            block_m,
            block_n,
            block_k,
            num_warps,
            group_m,
            speedup,
        ) = result

        print(
            f"[{rank:02d}] "
            f"BM={block_m:<3} "
            f"BN={block_n:<3} "
            f"BK={block_k:<3} "
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
            best_bn,
            best_bk,
            best_warps,
            best_group,
            best_speedup,
        ) = results[0]

        print()
        print("=" * 100)
        print("BEST")
        print("=" * 100)

        print(
            f"BM={best_bm}, "
            f"BN={best_bn}, "
            f"BK={best_bk}, "
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

    return results


def main():
    torch.cuda.init()

    all_results = {}

    for (
        name,
        K,
        N,
    ) in SHAPES:
        results = run_shape(
            name,
            K,
            N,
        )

        all_results[
            (K, N)
        ] = results

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for (
        name,
        K,
        N,
    ) in SHAPES:
        results = all_results[
            (K, N)
        ]

        if not results:
            continue

        (
            ms,
            bm,
            bn,
            bk,
            warps,
            group,
            speedup,
        ) = results[0]

        print(
            f"{name:<32} "
            f"BM={bm:<3} "
            f"BN={bn:<3} "
            f"BK={bk:<3} "
            f"W={warps} "
            f"G={group:<2} "
            f"{ms:.4f} ms "
            f"({speedup:.2f}x Torch)"
        )


if __name__ == "__main__":
    main()