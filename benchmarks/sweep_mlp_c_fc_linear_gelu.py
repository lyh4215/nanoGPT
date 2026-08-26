import torch
import torch.nn.functional as F

import triton

from triton_kernels.linear_gelu import (
    _linear_gelu_fwd_kernel,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024

M = B * T
K = 768
N = 3072


# ============================================================
# 새 plain GEMM best 주변만 탐색
# ============================================================

CONFIGS = [
    # 기존 강한 영역
    (128, 128, 32, 4, 4, 2),
    (128, 128, 32, 4, 8, 2),

    (128, 128, 64, 4, 4, 2),
    (128, 128, 64, 4, 8, 2),

    # 새 plain GEMM best
    (256, 128, 64, 8, 8, 2),

    # 주변
    (256, 128, 32, 8, 8, 2),

    (256, 128, 64, 8, 4, 2),
    (256, 128, 64, 8, 8, 3),

    # BN 방향 확인
    (256, 64, 64, 8, 8, 2),
    (128, 256, 64, 8, 8, 2),
]


def main():
    torch.manual_seed(0)
    torch.cuda.init()

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

    y = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # training backward를 위해 Z도 저장하는
    # 현재 linear_gelu kernel 구조 기준.
    z = torch.empty_like(
        y
    )

    # ========================================================
    # PyTorch reference
    # ========================================================

    def torch_fn():
        out = F.linear(
            x,
            weight,
            bias,
        )

        return F.gelu(
            out,
            approximate="none",
        )

    with torch.inference_mode():
        for _ in range(10):
            torch_fn()

    torch.cuda.synchronize()

    with torch.inference_mode():
        torch_ms = triton.testing.do_bench(
            torch_fn
        )

    print()
    print("=" * 105)
    print(
        f"MLP c_fc Linear + GELU Sweep "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 105)

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
        f"{'S':>4} "
        f"{'ms':>10} "
        f"{'vs Torch':>10}"
    )

    print("-" * 75)

    results = []

    for (
        bm,
        bn,
        bk,
        warps,
        group,
        stages,
    ) in CONFIGS:

        grid = (
            triton.cdiv(
                M,
                bm,
            )
            * triton.cdiv(
                N,
                bn,
            ),
        )

        def tri_fn():
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

                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,

                GROUP_SIZE_M=group,

                num_warps=warps,
                num_stages=stages,
            )

        try:
            with torch.inference_mode():
                tri_fn()

            torch.cuda.synchronize()

            with torch.inference_mode():
                ms = triton.testing.do_bench(
                    tri_fn
                )

            results.append(
                (
                    ms,
                    bm,
                    bn,
                    bk,
                    warps,
                    group,
                    stages,
                )
            )

            print(
                f"{bm:>5} "
                f"{bn:>5} "
                f"{bk:>5} "
                f"{warps:>4} "
                f"{group:>4} "
                f"{stages:>4} "
                f"{ms:>10.4f} "
                f"{torch_ms / ms:>9.2f}x"
            )

        except Exception as e:
            print(
                f"{bm:>5} "
                f"{bn:>5} "
                f"{bk:>5} "
                f"{warps:>4} "
                f"{group:>4} "
                f"{stages:>4} "
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
    print("Ranking")
    print("=" * 105)

    print()

    for rank, (
        ms,
        bm,
        bn,
        bk,
        warps,
        group,
        stages,
    ) in enumerate(
        results,
        start=1,
    ):
        print(
            f"[{rank:02d}] "
            f"BM={bm:<3} "
            f"BN={bn:<3} "
            f"BK={bk:<3} "
            f"W={warps} "
            f"G={group:<2} "
            f"S={stages} "
            f"| {ms:.4f} ms "
            f"| {torch_ms / ms:.3f}x Torch"
        )


if __name__ == "__main__":
    main()