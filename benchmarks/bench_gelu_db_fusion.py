import torch
import triton

from triton_kernels.linear_gelu import (
    triton_gelu_backward,
    triton_gelu_backward_with_db,
)

from triton_kernels.linear import (
    triton_linear_backward_db,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024

M = B * T
N = 3072


def main():
    torch.manual_seed(0)
    torch.cuda.init()

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
    # 기존 경로
    #
    # GELU backward
    #   ↓
    # DZ
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

    # ========================================================
    # 새 fused 경로
    #
    # GELU backward
    # ├─ DZ
    # └─ partial db
    #
    # partial db
    #   ↓
    # db
    # ========================================================

    def fused():
        return triton_gelu_backward_with_db(
            dy,
            z,
        )

    # ========================================================
    # Warmup
    # ========================================================

    for _ in range(5):
        separate()
        fused()

    torch.cuda.synchronize()

    # ========================================================
    # Benchmark
    # ========================================================

    separate_ms = triton.testing.do_bench(
        separate
    )

    fused_ms = triton.testing.do_bench(
        fused
    )

    # ========================================================
    # Correctness
    # ========================================================

    dz_ref, db_ref = separate()
    dz_tri, db_tri = fused()

    torch.cuda.synchronize()

    dz_diff = (
        dz_ref.float()
        - dz_tri.float()
    ).abs()

    db_diff = (
        db_ref.float()
        - db_tri.float()
    ).abs()

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"GELU backward + db Fusion Benchmark "
        f"M={M}, N={N}"
    )
    print("=" * 80)

    print()

    print("[Performance]")

    print(
        f"Separate : {separate_ms:.4f} ms"
    )

    print(
        f"Fused    : {fused_ms:.4f} ms"
    )

    print(
        f"Speedup  : "
        f"{separate_ms / fused_ms:.2f}x"
    )

    print()
    print("[Correctness]")

    print(
        f"dZ max diff : "
        f"{dz_diff.max().item():.6e}"
    )

    print(
        f"dZ mean diff: "
        f"{dz_diff.mean().item():.6e}"
    )

    print(
        f"db max diff : "
        f"{db_diff.max().item():.6e}"
    )

    print(
        f"db mean diff: "
        f"{db_diff.mean().item():.6e}"
    )


if __name__ == "__main__":
    main()