import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear_gelu import (
    _linear_gelu_fwd_kernel,
)


DEVICE = "cuda"
DTYPE = torch.float16

# GPT-2 MLP c_fc
B = 8
T = 1024

M = B * T
K = 768
N = 3072


# ============================================================
# Candidate configs
#
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
# ============================================================

CONFIGS = [
    (32, 32, 32, 4),
    (32, 64, 32, 4),
    (64, 32, 32, 4),
    (64, 64, 32, 4),

    (64, 128, 32, 4),
    (128, 64, 32, 4),

    (128, 128, 32, 4),

    # reduction tile도 조금 확인
    (64, 64, 64, 4),
    (64, 128, 64, 4),
    (128, 64, 64, 4),
    (128, 128, 64, 4),

    # warp 수 변화
    (64, 64, 32, 8),
    (64, 128, 32, 8),
    (128, 64, 32, 8),
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
    # Input
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

    # training용 fused forward는 둘 다 생성
    z = torch.empty(
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
    # PyTorch reference performance
    # ========================================================

    def torch_fn():
        z_ref = F.linear(
            x,
            weight,
            bias,
        )

        return F.gelu(
            z_ref
        )

    # warmup
    for _ in range(3):
        torch_fn()

    torch.cuda.synchronize()

    torch_ms = triton.testing.do_bench(
        torch_fn
    )

    print()
    print("=" * 100)
    print(
        f"Linear + GELU config sweep "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 100)

    print()
    print(
        f"PyTorch Linear + GELU : "
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

    total = len(CONFIGS) * len(GROUPS)
    idx = 0

    for (
        block_m,
        block_n,
        block_k,
        num_warps,
    ) in CONFIGS:

        for group_m in GROUPS:
            idx += 1

            grid = (
                triton.cdiv(M, block_m)
                * triton.cdiv(N, block_n),
            )

            def triton_fn():
                _linear_gelu_fwd_kernel[grid](
                    x,
                    weight,
                    bias,
                    z,
                    y,

                    M=M,
                    N=N,
                    K=K,

                    stride_xm=x.stride(0),
                    stride_xk=x.stride(1),

                    stride_wn=weight.stride(0),
                    stride_wk=weight.stride(1),

                    stride_zm=z.stride(0),
                    stride_zn=z.stride(1),

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
                # compile + warmup
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
    # Best results
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

    if results:
        best = results[0]

        (
            best_ms,
            best_bm,
            best_bn,
            best_bk,
            best_warps,
            best_group,
            best_speedup,
        ) = best

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


if __name__ == "__main__":
    main()