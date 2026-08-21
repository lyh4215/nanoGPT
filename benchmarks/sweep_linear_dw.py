import torch
import triton

from triton_kernels.linear import (
    triton_linear_backward_dw,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


# ============================================================
# 실제 GPT Linear들의 backward dW
#
# forward:
# X [M,K] @ W^T [K,N]
# -> Y [M,N]
#
# backward:
# dW = dY^T @ X
#
# dY^T [N,M] @ X [M,K]
# -> dW [N,K]
# ============================================================

SHAPES = [
    ("QKV",       768, 2304),
    ("attn_proj", 768, 768),
    ("mlp_fc",    768, 3072),
    ("mlp_proj", 3072, 768),
]


# ============================================================
# BLOCK_N, BLOCK_K, BLOCK_M, warps
#
# output tile:
#   [BLOCK_N, BLOCK_K]
#
# reduction:
#   BLOCK_M
# ============================================================

CONFIGS = [
    # baseline
    (32,  32,  32, 4),

    # output tile 확대
    (64,  64,  32, 4),
    (64, 128,  32, 4),
    (128, 64,  32, 4),
    (128, 128, 32, 4),

    # reduction tile = 64
    (64, 128,  64, 4),
    (128, 64,  64, 4),
    (128, 128, 64, 4),

    # reduction tile = 128
    (64, 128,  128, 4),
    (128, 64,  128, 4),
    (128, 128, 128, 4),

    # warp 영향
    (128, 128, 32, 8),
    (128, 128, 64, 8),
]


GROUP_SIZES = [
    1,
    2,
    4,
    8,
    16,
]


# ============================================================
# Benchmark helper
# ============================================================

def bench_dw(
    dy,
    x,
    bn,
    bk,
    bm,
    warps,
    group_size,
):
    def fn():
        triton_linear_backward_dw(
            dy,
            x,

            block_n=bn,
            block_k=bk,
            block_m=bm,

            num_warps=warps,
            group_size_n=group_size,
        )

    return triton.testing.do_bench(
        fn
    )


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(0)

    # ========================================================
    # PHASE 1
    #
    # Tile sweep
    #
    # GROUP_SIZE_N = 1 고정
    # ========================================================

    print()
    print("#" * 100)
    print("PHASE 1: dW TILE SWEEP")
    print("#" * 100)

    best_configs = {}

    for name, K, N in SHAPES:

        M = B * T

        print()
        print("=" * 95)

        print(
            f"{name}: "
            f"dY^T [{N},{M}] @ "
            f"X [{M},{K}] "
            f"-> dW [{N},{K}]"
        )

        print("=" * 95)

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        dy = torch.randn(
            B,
            T,
            N,
            device=DEVICE,
            dtype=DTYPE,
        )

        x = torch.randn(
            B,
            T,
            K,
            device=DEVICE,
            dtype=DTYPE,
        )

        dy_2d = dy.reshape(
            -1,
            N,
        )

        x_2d = x.reshape(
            -1,
            K,
        )

        # ----------------------------------------------------
        # PyTorch reference
        #
        # [N,M] @ [M,K]
        # -> [N,K]
        # ----------------------------------------------------

        ref = (
            dy_2d.T
            @ x_2d
        )

        def torch_fn():
            dy_2d.T @ x_2d

        torch_ms = (
            triton.testing.do_bench(
                torch_fn
            )
        )

        print(
            f"PyTorch: {torch_ms:.4f} ms"
        )

        print()

        print(
            f"{'BN':>5}"
            f"{'BK':>5}"
            f"{'BM':>5}"
            f"{'W':>4}"
            f"{'ms':>12}"
            f"{'vs torch':>12}"
            f"{'max diff':>15}"
        )

        print("-" * 65)

        results = []

        # ----------------------------------------------------
        # Config sweep
        # ----------------------------------------------------

        for (
            bn,
            bk,
            bm,
            warps,
        ) in CONFIGS:

            try:
                # --------------------------------------------
                # correctness
                # --------------------------------------------

                out = (
                    triton_linear_backward_dw(
                        dy,
                        x,

                        block_n=bn,
                        block_k=bk,
                        block_m=bm,

                        num_warps=warps,

                        # tile 효과만 보기
                        group_size_n=1,
                    )
                )

                diff = (
                    out.float()
                    - ref.float()
                ).abs()

                max_diff = (
                    diff.max().item()
                )

                ok = torch.allclose(
                    out,
                    ref,
                    atol=1e-2,
                    rtol=1e-2,
                )

                if not ok:
                    print(
                        f"{bn:5d}"
                        f"{bk:5d}"
                        f"{bm:5d}"
                        f"{warps:4d}"
                        f"{'FAIL':>12}"
                        f"{'':>12}"
                        f"{max_diff:15.6e}"
                    )

                    continue

                # --------------------------------------------
                # benchmark
                # --------------------------------------------

                ms = bench_dw(
                    dy,
                    x,

                    bn,
                    bk,
                    bm,
                    warps,

                    group_size=1,
                )

                print(
                    f"{bn:5d}"
                    f"{bk:5d}"
                    f"{bm:5d}"
                    f"{warps:4d}"
                    f"{ms:12.4f}"
                    f"{torch_ms / ms:11.2f}x"
                    f"{max_diff:15.6e}"
                )

                results.append(
                    (
                        ms,
                        bn,
                        bk,
                        bm,
                        warps,
                    )
                )

            except Exception as e:
                print(
                    f"{bn:5d}"
                    f"{bk:5d}"
                    f"{bm:5d}"
                    f"{warps:4d}"
                    f"{'ERROR':>12} "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

        # ----------------------------------------------------
        # best tile
        # ----------------------------------------------------

        if not results:
            print(
                f"\nNo valid config for {name}"
            )
            continue

        results.sort()

        (
            best_ms,
            best_bn,
            best_bk,
            best_bm,
            best_warps,
        ) = results[0]

        best_configs[name] = (
            best_bn,
            best_bk,
            best_bm,
            best_warps,
        )

        print()

        print(
            f"BEST TILE {name}: "
            f"BN={best_bn}, "
            f"BK={best_bk}, "
            f"BM={best_bm}, "
            f"W={best_warps}"
        )

        print(
            f"{best_ms:.4f} ms "
            f"({torch_ms / best_ms:.2f}x)"
        )

    # ========================================================
    # PHASE 2
    #
    # Best tile 고정
    # GROUP_SIZE_N sweep
    # ========================================================

    print()
    print()
    print("#" * 100)
    print("PHASE 2: dW PROGRAM ORDERING SWEEP")
    print("#" * 100)

    for name, K, N in SHAPES:

        if name not in best_configs:
            continue

        M = B * T

        (
            bn,
            bk,
            bm,
            warps,
        ) = best_configs[name]

        print()
        print("=" * 95)

        print(
            f"{name}: "
            f"BN={bn}, "
            f"BK={bk}, "
            f"BM={bm}, "
            f"W={warps}"
        )

        print("=" * 95)

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        dy = torch.randn(
            B,
            T,
            N,
            device=DEVICE,
            dtype=DTYPE,
        )

        x = torch.randn(
            B,
            T,
            K,
            device=DEVICE,
            dtype=DTYPE,
        )

        dy_2d = dy.reshape(
            -1,
            N,
        )

        x_2d = x.reshape(
            -1,
            K,
        )

        ref = (
            dy_2d.T
            @ x_2d
        )

        def torch_fn():
            dy_2d.T @ x_2d

        torch_ms = (
            triton.testing.do_bench(
                torch_fn
            )
        )

        print(
            f"PyTorch: {torch_ms:.4f} ms"
        )

        print()

        print(
            f"{'GROUP_N':>10}"
            f"{'Triton ms':>15}"
            f"{'vs Torch':>12}"
            f"{'max diff':>15}"
        )

        print("-" * 55)

        results = []

        # ----------------------------------------------------
        # Group sweep
        # ----------------------------------------------------

        for group_size in GROUP_SIZES:

            try:
                out = (
                    triton_linear_backward_dw(
                        dy,
                        x,

                        block_n=bn,
                        block_k=bk,
                        block_m=bm,

                        num_warps=warps,
                        group_size_n=group_size,
                    )
                )

                max_diff = (
                    (
                        out.float()
                        - ref.float()
                    )
                    .abs()
                    .max()
                    .item()
                )

                ok = torch.allclose(
                    out,
                    ref,
                    atol=1e-1,
                    rtol=1e-2,
                )

                if not ok:
                    print(
                        f"{group_size:10d}"
                        f"{'FAIL':>15}"
                        f"{'':>12}"
                        f"{max_diff:15.6e}"
                    )

                    continue

                ms = bench_dw(
                    dy,
                    x,

                    bn,
                    bk,
                    bm,
                    warps,

                    group_size,
                )

                print(
                    f"{group_size:10d}"
                    f"{ms:15.4f}"
                    f"{torch_ms / ms:11.2f}x"
                    f"{max_diff:15.6e}"
                )

                results.append(
                    (
                        ms,
                        group_size,
                    )
                )

            except Exception as e:
                print(
                    f"{group_size:10d}"
                    f"{'ERROR':>15} "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

        # ----------------------------------------------------
        # best group
        # ----------------------------------------------------

        if not results:
            continue

        results.sort()

        (
            best_ms,
            best_group,
        ) = results[0]

        print()

        print(
            f"BEST GROUP {name}: "
            f"GROUP_SIZE_N={best_group}"
        )

        print(
            f"{best_ms:.4f} ms "
            f"({torch_ms / best_ms:.2f}x)"
        )


if __name__ == "__main__":
    main()