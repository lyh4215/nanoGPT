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


# ============================================================
# GPT-2 small에서 실제 사용하는 4개 Linear
# ============================================================

SHAPES = [
    ("QKV",       768, 2304),
    ("attn_proj", 768, 768),
    ("mlp_fc",    768, 3072),
    ("mlp_proj", 3072, 768),
]


# ============================================================
# BM, BN, BK, warps
# ============================================================

CONFIGS = [
    (32,  32,  32, 4),

    (64,  64,  32, 4),
    (64, 128,  32, 4),
    (64, 128,  32, 8),

    (128, 64,  32, 4),
    (128, 64,  32, 8),

    (128, 128, 32, 4),
    (128, 128, 32, 8),

    (64,  64,  64, 4),
    (64, 128,  64, 4),
    (128, 64,  64, 4),
]


def benchmark_config(
    x,
    weight,
    bias,
    ref,
    bm,
    bn,
    bk,
    warps,
):
    # --------------------------------------------------------
    # compile + correctness
    # --------------------------------------------------------

    out = triton_linear_forward(
        x,
        weight,
        bias,
        block_m=bm,
        block_n=bn,
        block_k=bk,
        num_warps=warps,
    )

    torch.cuda.synchronize()

    max_diff = (
        out.float()
        - ref.float()
    ).abs().max().item()

    ok = torch.allclose(
        out,
        ref,
        atol=1e-2,
        rtol=1e-2,
    )

    if not ok:
        return None, max_diff

    # --------------------------------------------------------
    # benchmark
    # --------------------------------------------------------

    def fn():
        triton_linear_forward(
            x,
            weight,
            bias,
            block_m=bm,
            block_n=bn,
            block_k=bk,
            num_warps=warps,
        )

    ms = triton.testing.do_bench(fn)

    return ms, max_diff


def main():
    torch.manual_seed(0)

    # config별로 네 GEMM latency 합산
    totals = {
        config: 0.0
        for config in CONFIGS
    }

    valid_for_all = {
        config: True
        for config in CONFIGS
    }

    all_results = {}

    # ========================================================
    # shape별 sweep
    # ========================================================

    for name, K, N in SHAPES:

        M = B * T

        print()
        print("=" * 95)
        print(
            f"{name}: "
            f"M={M}, K={K}, N={N}"
        )
        print("=" * 95)

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

        # PyTorch reference
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
            f"{'BM':>5}"
            f"{'BN':>5}"
            f"{'BK':>5}"
            f"{'W':>4}"
            f"{'ms':>12}"
            f"{'vs torch':>12}"
            f"{'max diff':>15}"
        )

        print("-" * 65)

        results = []

        for config in CONFIGS:
            bm, bn, bk, warps = config

            try:
                ms, max_diff = benchmark_config(
                    x,
                    weight,
                    bias,
                    ref,
                    bm,
                    bn,
                    bk,
                    warps,
                )

                if ms is None:
                    print(
                        f"{bm:5d}"
                        f"{bn:5d}"
                        f"{bk:5d}"
                        f"{warps:4d}"
                        f"{'FAIL':>12}"
                        f"{'':>12}"
                        f"{max_diff:15.6e}"
                    )

                    valid_for_all[config] = False
                    continue

                speedup = (
                    torch_ms / ms
                )

                print(
                    f"{bm:5d}"
                    f"{bn:5d}"
                    f"{bk:5d}"
                    f"{warps:4d}"
                    f"{ms:12.4f}"
                    f"{speedup:11.2f}x"
                    f"{max_diff:15.6e}"
                )

                results.append(
                    (
                        ms,
                        config,
                        speedup,
                    )
                )

                totals[config] += ms

            except Exception as e:
                print(
                    f"{bm:5d}"
                    f"{bn:5d}"
                    f"{bk:5d}"
                    f"{warps:4d}"
                    f"{'ERROR':>12} "
                    f"{type(e).__name__}"
                )

                valid_for_all[config] = False

        results.sort()

        if results:
            best_ms, best_config, best_speedup = results[0]

            print()
            print(
                f"BEST {name}: "
                f"BM={best_config[0]} "
                f"BN={best_config[1]} "
                f"BK={best_config[2]} "
                f"W={best_config[3]}"
            )

            print(
                f"  {best_ms:.4f} ms "
                f"({best_speedup:.2f}x vs PyTorch)"
            )

            all_results[name] = results

        # config 실패 후 CUDA context가 망가지는 경우를
        # 빠르게 드러내기 위한 sync
        torch.cuda.synchronize()

    # ========================================================
    # 하나의 config를 네 Linear에 공통 사용한다면?
    #
    # 각 op은 Block 안에서 1번씩 실행되므로
    # 단순 latency sum으로 비교
    # ========================================================

    print()
    print("=" * 95)
    print("GLOBAL CONFIG RANKING")
    print("=" * 95)

    ranking = []

    for config in CONFIGS:
        if not valid_for_all[config]:
            continue

        ranking.append(
            (
                totals[config],
                config,
            )
        )

    ranking.sort()

    print(
        f"{'rank':>5}"
        f"{'BM':>6}"
        f"{'BN':>6}"
        f"{'BK':>6}"
        f"{'W':>5}"
        f"{'4-op sum ms':>15}"
    )

    print("-" * 50)

    for rank, (total, config) in enumerate(
        ranking,
        start=1,
    ):
        bm, bn, bk, warps = config

        print(
            f"{rank:5d}"
            f"{bm:6d}"
            f"{bn:6d}"
            f"{bk:6d}"
            f"{warps:5d}"
            f"{total:15.4f}"
        )

    if ranking:
        total, best = ranking[0]

        print()
        print(
            "BEST GLOBAL CONFIG:"
        )

        print(
            f"BM={best[0]}, "
            f"BN={best[1]}, "
            f"BK={best[2]}, "
            f"num_warps={best[3]}"
        )

        print(
            f"4-op total: {total:.4f} ms"
        )


if __name__ == "__main__":
    main()