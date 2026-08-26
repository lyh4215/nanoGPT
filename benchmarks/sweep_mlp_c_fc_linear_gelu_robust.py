import statistics

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

ROUNDS = 5


# ============================================================
# 현재 Linear + GELU production baseline
#
# 네가 기존에 쓰던 config가 다르면
# 여기만 현재 값으로 바꾸면 됨.
# ============================================================

BASELINE = {
    "BM": 128,
    "BN": 64,
    "BK": 64,
    "W": 4,
    "G": 8,
    "S": 2,
}


# ============================================================
# Narrow candidates
#
# plain GEMM에서 확정된:
#
#   128 x 128 x 64
#   W4 G8 S3
#
# 주변 + 기존 Linear-GELU 강한 영역
# ============================================================

CONFIGS = [
    # current neighborhood
    {
        "BM": 128,
        "BN": 64,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 2,
    },
    {
        "BM": 128,
        "BN": 64,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 3,
    },
    {
        "BM": 128,
        "BN": 64,
        "BK": 32,
        "W": 4,
        "G": 8,
        "S": 2,
    },
    {
        "BM": 128,
        "BN": 64,
        "BK": 32,
        "W": 4,
        "G": 8,
        "S": 3,
    },

    # plain GEMM winner neighborhood
    {
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 2,
    },
    {
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 3,
    },
    {
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 4,
    },

    # BK32 비교
    {
        "BM": 128,
        "BN": 128,
        "BK": 32,
        "W": 4,
        "G": 8,
        "S": 2,
    },
    {
        "BM": 128,
        "BN": 128,
        "BK": 32,
        "W": 4,
        "G": 8,
        "S": 3,
    },

    # grouping 영향
    {
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "W": 4,
        "G": 4,
        "S": 3,
    },
    {
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "W": 4,
        "G": 2,
        "S": 3,
    },

    # larger tile 후보
    {
        "BM": 256,
        "BN": 128,
        "BK": 64,
        "W": 8,
        "G": 8,
        "S": 2,
    },
]


def config_name(cfg):
    return (
        f"BM={cfg['BM']} "
        f"BN={cfg['BN']} "
        f"BK={cfg['BK']} "
        f"W={cfg['W']} "
        f"G={cfg['G']} "
        f"S={cfg['S']}"
    )


# ============================================================
# Kernel launcher
#
# 현재 linear_gelu forward가:
#
#   X, W, BIAS, Z, Y
#
# 순서라고 가정.
#
# Z = pre-GELU value
# Y = GELU(Z)
#
# 네 kernel signature가 다르면
# triton_linear_gelu_forward()에서 호출하는 순서 그대로
# 여기만 맞추면 됨.
# ============================================================

def launch(
    x,
    weight,
    bias,
    z,
    y,
    cfg,
):
    bm = cfg["BM"]
    bn = cfg["BN"]

    grid = (
        triton.cdiv(M, bm)
        * triton.cdiv(N, bn),
    )

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

        BLOCK_M=cfg["BM"],
        BLOCK_N=cfg["BN"],
        BLOCK_K=cfg["BK"],

        GROUP_SIZE_M=cfg["G"],

        num_warps=cfg["W"],
        num_stages=cfg["S"],
    )

    return y


def bench_once(fn):
    with torch.inference_mode():
        return triton.testing.do_bench(
            fn
        )


def compare_candidate(
    baseline_fn,
    candidate_fn,
):
    baseline_times = []
    candidate_times = []
    paired_speedups = []

    for i in range(ROUNDS):

        if i % 2 == 0:
            base_ms = bench_once(
                baseline_fn
            )

            cand_ms = bench_once(
                candidate_fn
            )

        else:
            cand_ms = bench_once(
                candidate_fn
            )

            base_ms = bench_once(
                baseline_fn
            )

        baseline_times.append(
            base_ms
        )

        candidate_times.append(
            cand_ms
        )

        paired_speedups.append(
            base_ms / cand_ms
        )

    return {
        "baseline_median":
            statistics.median(
                baseline_times
            ),

        "candidate_median":
            statistics.median(
                candidate_times
            ),

        "candidate_mean":
            statistics.mean(
                candidate_times
            ),

        "paired_median":
            statistics.median(
                paired_speedups
            ),

        "paired_mean":
            statistics.mean(
                paired_speedups
            ),

        "min":
            min(candidate_times),

        "max":
            max(candidate_times),
    }


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Data
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

    z_base = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    y_base = torch.empty_like(
        z_base
    )

    z_candidate = torch.empty_like(
        z_base
    )

    y_candidate = torch.empty_like(
        y_base
    )

    # ========================================================
    # PyTorch reference
    # ========================================================

    def torch_fn():
        z = F.linear(
            x,
            weight,
            bias,
        )

        return F.gelu(
            z,
            approximate="none",
        )

    # ========================================================
    # Baseline Triton
    # ========================================================

    def baseline_fn():
        return launch(
            x,
            weight,
            bias,
            z_base,
            y_base,
            BASELINE,
        )

    # ========================================================
    # Warmup
    # ========================================================

    with torch.inference_mode():

        for _ in range(10):
            torch_fn()
            baseline_fn()

    torch.cuda.synchronize()

    # ========================================================
    # Robust PyTorch number
    # ========================================================

    torch_times = [
        bench_once(torch_fn)
        for _ in range(10)
    ]

    torch_median = (
        statistics.median(
            torch_times
        )
    )

    print()
    print("=" * 115)
    print(
        f"Robust MLP c_fc Linear + GELU Sweep "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 115)

    print()

    print(
        f"Baseline        : "
        f"{config_name(BASELINE)}"
    )

    print(
        f"PyTorch median  : "
        f"{torch_median:.4f} ms"
    )

    print(
        f"Rounds/config   : "
        f"{ROUNDS}"
    )

    print()

    print(
        f"{'#':>3} "
        f"{'BM':>4} "
        f"{'BN':>4} "
        f"{'BK':>4} "
        f"{'W':>3} "
        f"{'G':>3} "
        f"{'S':>3} "
        f"{'cand':>10} "
        f"{'base':>10} "
        f"{'paired':>9} "
        f"{'vsTorch':>9}"
    )

    print("-" * 90)

    results = []

    # ========================================================
    # Sweep
    # ========================================================

    for idx, cfg in enumerate(
        CONFIGS,
        start=1,
    ):

        def candidate_fn(
            cfg=cfg,
        ):
            return launch(
                x,
                weight,
                bias,
                z_candidate,
                y_candidate,
                cfg,
            )

        try:
            # compile + local warmup
            with torch.inference_mode():

                candidate_fn()

                for _ in range(3):
                    baseline_fn()
                    candidate_fn()

            torch.cuda.synchronize()

            stat = compare_candidate(
                baseline_fn,
                candidate_fn,
            )

            vs_torch = (
                torch_median
                / stat["candidate_median"]
            )

            results.append(
                {
                    "config": cfg,
                    **stat,
                    "vs_torch": vs_torch,
                }
            )

            print(
                f"{idx:>3} "
                f"{cfg['BM']:>4} "
                f"{cfg['BN']:>4} "
                f"{cfg['BK']:>4} "
                f"{cfg['W']:>3} "
                f"{cfg['G']:>3} "
                f"{cfg['S']:>3} "
                f"{stat['candidate_median']:>8.4f} ms "
                f"{stat['baseline_median']:>8.4f} ms "
                f"{stat['paired_median']:>8.3f}x "
                f"{vs_torch:>8.3f}x"
            )

        except Exception as e:
            print(
                f"{idx:>3} "
                f"{cfg['BM']:>4} "
                f"{cfg['BN']:>4} "
                f"{cfg['BK']:>4} "
                f"{cfg['W']:>3} "
                f"{cfg['G']:>3} "
                f"{cfg['S']:>3} "
                f"{'FAIL':>10} "
                f"{type(e).__name__}"
            )

    # ========================================================
    # Ranking
    # ========================================================

    results.sort(
        key=lambda r:
            r["paired_median"],
        reverse=True,
    )

    print()
    print("=" * 115)
    print(
        "Ranking — paired speedup "
        "vs current Linear+GELU baseline"
    )
    print("=" * 115)

    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        cfg = result["config"]

        print(
            f"[{rank:02d}] "
            f"{config_name(cfg)} "
            f"| cand={result['candidate_median']:.4f} ms "
            f"| base={result['baseline_median']:.4f} ms "
            f"| paired={result['paired_median']:.3f}x "
            f"| Torch={result['vs_torch']:.3f}x"
        )

    if not results:
        return

    best = results[0]
    cfg = best["config"]

    # ========================================================
    # Best
    # ========================================================

    print()
    print("=" * 115)
    print("BEST")
    print("=" * 115)

    print()

    print(
        config_name(cfg)
    )

    print()

    print(
        f"Candidate median      : "
        f"{best['candidate_median']:.4f} ms"
    )

    print(
        f"Baseline median       : "
        f"{best['baseline_median']:.4f} ms"
    )

    print(
        f"Median paired speedup : "
        f"{best['paired_median']:.3f}x"
    )

    print(
        f"Mean paired speedup   : "
        f"{best['paired_mean']:.3f}x"
    )

    print()

    print(
        f"PyTorch median        : "
        f"{torch_median:.4f} ms"
    )

    print(
        f"Best vs PyTorch       : "
        f"{best['vs_torch']:.3f}x"
    )

    # ========================================================
    # Correctness
    # ========================================================

    with torch.inference_mode():

        ref = torch_fn()

        launch(
            x,
            weight,
            bias,
            z_candidate,
            y_candidate,
            cfg,
        )

    torch.cuda.synchronize()

    diff = (
        ref.float()
        - y_candidate.float()
    ).abs()

    print()
    print("=" * 115)
    print("Correctness")
    print("=" * 115)

    print()

    print(
        f"max diff  : "
        f"{diff.max().item():.6e}"
    )

    print(
        f"mean diff : "
        f"{diff.mean().item():.6e}"
    )

    print(
        f"allclose  : "
        f"{torch.allclose(ref, y_candidate, atol=5e-2, rtol=1e-2)}"
    )


if __name__ == "__main__":
    main()