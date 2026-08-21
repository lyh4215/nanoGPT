import torch
import triton

from triton_kernels.linear import (
    triton_linear_backward_dx,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


# ============================================================
# 실제 GPT Linear들의 backward dX
#
# forward:
# [M,K] -> [M,N]
#
# backward:
# dY [M,N] @ W [N,K]
# -> dX [M,K]
# ============================================================

SHAPES = [
    ("QKV",       768, 2304),
    ("attn_proj", 768, 768),
    ("mlp_fc",    768, 3072),
    ("mlp_proj", 3072, 768),
]


# ============================================================
# BLOCK_M, BLOCK_K, BLOCK_N, warps
#
# 주의:
# BK = output K tile
# BN = reduction N tile
# ============================================================

CONFIGS = [
    (32,  32,  32, 4),

    (64,  64,  32, 4),
    (64, 128,  32, 4),

    (128, 64,  32, 4),
    (128, 128, 32, 4),

    (64, 128, 32, 8),
    (128, 128, 32, 8),

    # reduction tile도 확인
    (64,  64,  64, 4),
    (64, 128,  64, 4),
    (128, 128, 64, 4),
]


GROUP_SIZES = [
    1,
    2,
    4,
    8,
    16,
]


def bench_dx(
    dy,
    weight,
    bm,
    bk,
    bn,
    warps,
    group_size,
):
    def fn():
        triton_linear_backward_dx(
            dy,
            weight,

            block_m=bm,
            block_k=bk,
            block_n=bn,

            num_warps=warps,
            group_size_m=group_size,
        )

    return triton.testing.do_bench(
        fn
    )


def main():
    torch.manual_seed(0)

    # ========================================================
    # Phase 1:
    # Tile sweep with GROUP_SIZE_M = 1
    # ========================================================

    print()
    print("#" * 100)
    print("PHASE 1: dX TILE SWEEP")
    print("#" * 100)

    best_configs = {}

    for name, K, N in SHAPES:

        M = B * T

        print()
        print("=" * 90)
        print(
            f"{name}: "
            f"dY [{M},{N}] @ "
            f"W [{N},{K}] "
            f"-> dX [{M},{K}]"
        )
        print("=" * 90)

        dy = torch.randn(
            B,
            T,
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

        # ----------------------------------------------------
        # reference
        # ----------------------------------------------------

        dy_2d = dy.reshape(
            -1,
            N,
        )

        ref = (
            dy_2d
            @ weight
        ).view(
            B,
            T,
            K,
        )

        def torch_fn():
            dy_2d @ weight

        torch_ms = triton.testing.do_bench(
            torch_fn
        )

        print(
            f"PyTorch: {torch_ms:.4f} ms"
        )

        print()

        print(
            f"{'BM':>5}"
            f"{'BK':>5}"
            f"{'BN':>5}"
            f"{'W':>4}"
            f"{'ms':>12}"
            f"{'vs torch':>12}"
            f"{'max diff':>15}"
        )

        print("-" * 65)

        results = []

        for bm, bk, bn, warps in CONFIGS:

            out = triton_linear_backward_dx(
                dy,
                weight,

                block_m=bm,
                block_k=bk,
                block_n=bn,

                num_warps=warps,

                # tile 효과만 보기
                group_size_m=1,
            )

            diff = (
                out.float()
                - ref.float()
            ).abs()

            max_diff = (
                diff.max().item()
            )

            ms = bench_dx(
                dy,
                weight,

                bm,
                bk,
                bn,
                warps,

                group_size=1,
            )

            print(
                f"{bm:5d}"
                f"{bk:5d}"
                f"{bn:5d}"
                f"{warps:4d}"
                f"{ms:12.4f}"
                f"{torch_ms / ms:11.2f}x"
                f"{max_diff:15.6e}"
            )

            results.append(
                (
                    ms,
                    bm,
                    bk,
                    bn,
                    warps,
                )
            )

        results.sort()

        best = results[0]

        (
            best_ms,
            best_bm,
            best_bk,
            best_bn,
            best_warps,
        ) = best

        best_configs[name] = (
            best_bm,
            best_bk,
            best_bn,
            best_warps,
        )

        print()

        print(
            f"BEST TILE {name}: "
            f"BM={best_bm}, "
            f"BK={best_bk}, "
            f"BN={best_bn}, "
            f"W={best_warps}"
        )

        print(
            f"{best_ms:.4f} ms "
            f"({torch_ms / best_ms:.2f}x)"
        )

    # ========================================================
    # Phase 2:
    # 각 shape의 best tile을 고정하고 GROUP sweep
    # ========================================================

    print()
    print()
    print("#" * 100)
    print("PHASE 2: dX PROGRAM ORDERING SWEEP")
    print("#" * 100)

    for name, K, N in SHAPES:

        M = B * T

        (
            bm,
            bk,
            bn,
            warps,
        ) = best_configs[name]

        print()
        print("=" * 90)

        print(
            f"{name}: "
            f"BM={bm}, "
            f"BK={bk}, "
            f"BN={bn}, "
            f"W={warps}"
        )

        print("=" * 90)

        dy = torch.randn(
            B,
            T,
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

        dy_2d = dy.reshape(
            -1,
            N,
        )

        ref = (
            dy_2d
            @ weight
        ).view(
            B,
            T,
            K,
        )

        def torch_fn():
            dy_2d @ weight

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

        results = []

        for group_size in GROUP_SIZES:

            out = triton_linear_backward_dx(
                dy,
                weight,

                block_m=bm,
                block_k=bk,
                block_n=bn,

                num_warps=warps,
                group_size_m=group_size,
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

            ms = bench_dx(
                dy,
                weight,

                bm,
                bk,
                bn,
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

        results.sort()

        best_ms, best_group = (
            results[0]
        )

        print()

        print(
            f"BEST GROUP {name}: "
            f"GROUP_SIZE_M={best_group}"
        )

        print(
            f"{best_ms:.4f} ms "
            f"({torch_ms / best_ms:.2f}x)"
        )


if __name__ == "__main__":
    main()