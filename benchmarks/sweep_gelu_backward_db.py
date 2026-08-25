import torch
import triton

from triton_kernels.linear_gelu import (
    _gelu_bwd_db_partial_kernel,
    _db_reduce_partials_kernel,
    triton_gelu_backward,
)

from triton_kernels.linear import (
    triton_linear_backward_db,
)


DEVICE = "cuda"
DTYPE = torch.float16

# GPT-2 MLP c_fc
B = 8
T = 1024

M = B * T
N = 3072


# ============================================================
# Kernel 1 configs
#
# (BLOCK_M, BLOCK_N, num_warps)
# ============================================================

CONFIGS = [
    # BLOCK_M = 32
    (32, 16, 2),
    (32, 32, 2),
    (32, 64, 4),
    (32, 128, 4),

    # BLOCK_M = 64
    (64, 16, 2),
    (64, 32, 4),
    (64, 64, 4),
    (64, 128, 4),

    # BLOCK_M = 128
    (128, 16, 4),
    (128, 32, 4),
    (128, 64, 4),
    (128, 128, 4),

    # BLOCK_M = 256
    (256, 16, 4),
    (256, 32, 4),
    (256, 64, 4),

    # warp variation
    (64, 64, 8),
    (64, 128, 8),

    (128, 32, 8),
    (128, 64, 8),
    (128, 128, 8),

    (256, 32, 8),
    (256, 64, 8),
]


# final partial reduction의 N tile
DB_BLOCK_N = 128


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Input
    # ========================================================

    dy = torch.randn(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    z = torch.randn(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # Separate baseline
    #
    # GELU backward
    #   ↓
    # dZ
    #   ↓
    # db reduction
    # ========================================================

    def separate():
        dz = triton_gelu_backward(
            dy,
            z,
        )

        db = triton_linear_backward_db(
            dz,
        )

        return dz, db

    for _ in range(3):
        separate()

    torch.cuda.synchronize()

    separate_ms = triton.testing.do_bench(
        separate
    )

    # reference correctness
    dz_ref, db_ref = separate()

    torch.cuda.synchronize()

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 110)
    print(
        f"GELU backward + db sweep "
        f"M={M}, N={N}"
    )
    print("=" * 110)

    print()

    print(
        f"Separate baseline : "
        f"{separate_ms:.4f} ms"
    )

    print()

    print(
        f"{'BM':>5} "
        f"{'BN':>5} "
        f"{'W':>4} "
        f"{'partials':>9} "
        f"{'ms':>10} "
        f"{'vs separate':>12} "
        f"{'db max diff':>14}"
    )

    print("-" * 90)

    results = []

    # ========================================================
    # Sweep
    # ========================================================

    for (
        block_m,
        block_n,
        num_warps,
    ) in CONFIGS:

        num_pid_m = triton.cdiv(
            M,
            block_m,
        )

        # GELU backward output
        dz = torch.empty(
            M,
            N,
            device=DEVICE,
            dtype=DTYPE,
        )

        # partial db는 FP32
        partial_db = torch.empty(
            num_pid_m,
            N,
            device=DEVICE,
            dtype=torch.float32,
        )

        db = torch.empty(
            N,
            device=DEVICE,
            dtype=DTYPE,
        )

        # ----------------------------------------------------
        # Kernel 1 grid
        # ----------------------------------------------------

        grid1 = (
            num_pid_m,
            triton.cdiv(
                N,
                block_n,
            ),
        )

        # ----------------------------------------------------
        # Kernel 2
        #
        # NUM_PARTIALS 이상 power-of-two
        # ----------------------------------------------------

        block_r = triton.next_power_of_2(
            num_pid_m
        )

        grid2 = (
            triton.cdiv(
                N,
                DB_BLOCK_N,
            ),
        )

        def fused():
            # =================================================
            # Kernel 1:
            #
            # dy + z
            #   ↓
            # dz
            # ├→ dZ store
            # └→ partial db
            # =================================================

            _gelu_bwd_db_partial_kernel[grid1](
                dy,
                z,
                dz,
                partial_db,

                M=M,
                N=N,

                stride_dym=dy.stride(0),
                stride_dyn=dy.stride(1),

                stride_zm=z.stride(0),
                stride_zn=z.stride(1),

                stride_dzm=dz.stride(0),
                stride_dzn=dz.stride(1),

                stride_pm=partial_db.stride(0),
                stride_pn=partial_db.stride(1),

                BLOCK_M=block_m,
                BLOCK_N=block_n,

                num_warps=num_warps,
            )

            # =================================================
            # Kernel 2:
            #
            # partial_db
            #     ↓
            # final db
            # =================================================

            _db_reduce_partials_kernel[grid2](
                partial_db,
                db,

                NUM_PARTIALS=num_pid_m,
                N=N,

                stride_pm=partial_db.stride(0),
                stride_pn=partial_db.stride(1),

                BLOCK_R=block_r,
                BLOCK_N=DB_BLOCK_N,

                num_warps=4,
            )

            return dz, db

        try:
            # compile / warmup
            for _ in range(2):
                fused()

            torch.cuda.synchronize()

            ms = triton.testing.do_bench(
                fused
            )

            speedup = (
                separate_ms
                / ms
            )

            # =================================================
            # Correctness
            # =================================================

            dz_out, db_out = fused()

            torch.cuda.synchronize()

            dz_diff = (
                dz_out.float()
                - dz_ref.float()
            ).abs()

            db_diff = (
                db_out.float()
                - db_ref.float()
            ).abs()

            dz_max = (
                dz_diff.max().item()
            )

            db_max = (
                db_diff.max().item()
            )

            # dZ는 같은 elementwise 계산이므로
            # 사실상 동일해야 함
            if dz_max != 0.0:
                print(
                    f"{block_m:>5} "
                    f"{block_n:>5} "
                    f"{num_warps:>4} "
                    f"{num_pid_m:>9} "
                    f"{'BAD DZ':>10} "
                    f"{'':>12} "
                    f"{db_max:>14.6e}"
                )

                continue

            results.append(
                (
                    ms,
                    block_m,
                    block_n,
                    num_warps,
                    num_pid_m,
                    speedup,
                    db_max,
                )
            )

            print(
                f"{block_m:>5} "
                f"{block_n:>5} "
                f"{num_warps:>4} "
                f"{num_pid_m:>9} "
                f"{ms:>10.4f} "
                f"{speedup:>11.2f}x "
                f"{db_max:>14.6e}"
            )

        except Exception as e:
            print(
                f"{block_m:>5} "
                f"{block_n:>5} "
                f"{num_warps:>4} "
                f"{num_pid_m:>9} "
                f"{'FAIL':>10} "
                f"{type(e).__name__:>12}"
            )

    # ========================================================
    # Ranking
    # ========================================================

    results.sort(
        key=lambda x: x[0]
    )

    print()
    print("=" * 110)
    print("Top configurations")
    print("=" * 110)

    for rank, result in enumerate(
        results[:12],
        start=1,
    ):
        (
            ms,
            block_m,
            block_n,
            num_warps,
            num_partials,
            speedup,
            db_max,
        ) = result

        print(
            f"[{rank:02d}] "
            f"BM={block_m:<3} "
            f"BN={block_n:<3} "
            f"warps={num_warps} "
            f"partials={num_partials:<3} "
            f": {ms:.4f} ms "
            f"({speedup:.2f}x separate) "
            f"db_max={db_max:.3e}"
        )

    # ========================================================
    # BEST
    # ========================================================

    if results:
        (
            best_ms,
            best_bm,
            best_bn,
            best_warps,
            best_num_partials,
            best_speedup,
            best_db_max,
        ) = results[0]

        print()
        print("=" * 110)
        print("BEST")
        print("=" * 110)

        print(
            f"BLOCK_M={best_bm}, "
            f"BLOCK_N={best_bn}, "
            f"warps={best_warps}"
        )

        print(
            f"NUM_PARTIALS="
            f"{best_num_partials}"
        )

        print()

        print(
            f"Separate : "
            f"{separate_ms:.4f} ms"
        )

        print(
            f"Fused    : "
            f"{best_ms:.4f} ms"
        )

        print(
            f"Speedup  : "
            f"{best_speedup:.2f}x"
        )

        print(
            f"db max diff : "
            f"{best_db_max:.6e}"
        )

        print()

        print(
            "Current config:"
        )

        print(
            "BLOCK_M=128, "
            "BLOCK_N=16, "
            "warps=4"
        )


if __name__ == "__main__":
    main()