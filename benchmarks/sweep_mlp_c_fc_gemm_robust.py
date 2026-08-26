import statistics

import torch
import torch.nn.functional as F
import triton

from benchmarks.sweep_mlp_c_fc_gemm import (
    _c_fc_kernel,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024

M = B * T
K = 768
N = 3072


# ============================================================
# Robust benchmark settings
# ============================================================

ROUNDS = 5


# ============================================================
# Current production baseline
# ============================================================

BASELINE = {
    "BM": 128,
    "BN": 128,
    "BK": 32,
    "W": 4,
    "G": 8,
    "S": 2,
}


# ============================================================
# Narrow search space
#
# BM / BN은 현재 강한 128 x 128로 고정.
#
# 집중해서 보는 것:
#
# BK
# warps
# GROUP_SIZE_M
# num_stages
#
# 2 * 2 * 4 * 3 = 48 configs
# ============================================================

BLOCK_KS = [
    32,
    64,
]

WARPS = [
    4,
    8,
]

GROUPS = [
    1,
    2,
    4,
    8,
]

STAGES = [
    2,
    3,
    4,
]


CONFIGS = []

for bk in BLOCK_KS:
    for warps in WARPS:
        for group in GROUPS:
            for stages in STAGES:
                CONFIGS.append(
                    {
                        "BM": 128,
                        "BN": 128,
                        "BK": bk,
                        "W": warps,
                        "G": group,
                        "S": stages,
                    }
                )


# ============================================================
# Helpers
# ============================================================

def config_name(config):
    return (
        f"BM={config['BM']} "
        f"BN={config['BN']} "
        f"BK={config['BK']} "
        f"W={config['W']} "
        f"G={config['G']} "
        f"S={config['S']}"
    )


def launch(
    x,
    weight,
    bias,
    y,
    config,
):
    bm = config["BM"]
    bn = config["BN"]
    bk = config["BK"]

    grid = (
        triton.cdiv(M, bm)
        * triton.cdiv(N, bn),
    )

    _c_fc_kernel[grid](
        x,
        weight,
        bias,
        y,

        M=M,
        N=N,
        K=K,

        stride_xm=x.stride(0),
        stride_xk=x.stride(1),

        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_ym=y.stride(0),
        stride_yn=y.stride(1),

        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,

        GROUP_SIZE_M=config["G"],

        num_warps=config["W"],
        num_stages=config["S"],
    )

    return y


def bench_once(fn):
    with torch.inference_mode():
        return triton.testing.do_bench(
            fn
        )


def median(values):
    return statistics.median(values)


def mean(values):
    return statistics.mean(values)


def tflops(ms):
    total_flops = (
        2
        * M
        * N
        * K
    )

    return (
        total_flops
        / (ms * 1e-3)
        / 1e12
    )


# ============================================================
# Robust candidate vs baseline benchmark
#
# 매 round마다:
#
# even:
#   baseline -> candidate
#
# odd:
#   candidate -> baseline
#
# 측정 순서 bias를 줄인다.
# ============================================================

def compare_candidate(
    baseline_fn,
    candidate_fn,
):
    baseline_times = []
    candidate_times = []

    paired_speedups = []

    for round_idx in range(
        ROUNDS
    ):

        if round_idx % 2 == 0:

            baseline_ms = bench_once(
                baseline_fn
            )

            candidate_ms = bench_once(
                candidate_fn
            )

        else:

            candidate_ms = bench_once(
                candidate_fn
            )

            baseline_ms = bench_once(
                baseline_fn
            )

        baseline_times.append(
            baseline_ms
        )

        candidate_times.append(
            candidate_ms
        )

        paired_speedups.append(
            baseline_ms
            / candidate_ms
        )

    return {
        "baseline_times": baseline_times,
        "candidate_times": candidate_times,

        "baseline_median": median(
            baseline_times
        ),

        "candidate_median": median(
            candidate_times
        ),

        "candidate_mean": mean(
            candidate_times
        ),

        "paired_median": median(
            paired_speedups
        ),

        "paired_mean": mean(
            paired_speedups
        ),

        "candidate_min": min(
            candidate_times
        ),

        "candidate_max": max(
            candidate_times
        ),
    }


def main():
    torch.manual_seed(0)
    torch.cuda.init()

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

    y_baseline = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    y_candidate = torch.empty_like(
        y_baseline
    )

    # ========================================================
    # PyTorch baseline
    # ========================================================

    def torch_fn():
        return F.linear(
            x,
            weight,
            bias,
        )

    # ========================================================
    # Triton production baseline
    # ========================================================

    def baseline_fn():
        return launch(
            x,
            weight,
            bias,
            y_baseline,
            BASELINE,
        )

    # ========================================================
    # Compile / warmup baseline
    # ========================================================

    with torch.inference_mode():

        for _ in range(10):
            torch_fn()
            baseline_fn()

    torch.cuda.synchronize()

    # ========================================================
    # Robust PyTorch baseline
    #
    # PyTorch도 한 번만 재지 않고 여러 번.
    # ========================================================

    torch_results = []

    for _ in range(10):
        torch_results.append(
            bench_once(
                torch_fn
            )
        )

    torch_median = median(
        torch_results
    )

    print()
    print("=" * 120)
    print(
        f"Robust MLP c_fc GEMM Sweep "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 120)

    print()

    print(
        f"Current baseline : "
        f"{config_name(BASELINE)}"
    )

    print()

    print(
        f"PyTorch median   : "
        f"{torch_median:.4f} ms"
    )

    print(
        f"PyTorch TFLOPS   : "
        f"{tflops(torch_median):.2f}"
    )

    print()

    print(
        f"Rounds / config  : "
        f"{ROUNDS}"
    )

    print(
        f"Candidate count  : "
        f"{len(CONFIGS)}"
    )

    print()

    print(
        f"{'#':>3} "
        f"{'BK':>4} "
        f"{'W':>3} "
        f"{'G':>3} "
        f"{'S':>3} "
        f"{'cand med':>10} "
        f"{'base med':>10} "
        f"{'paired':>9} "
        f"{'vs Torch':>9}"
    )

    print("-" * 85)

    results = []

    # ========================================================
    # Sweep
    # ========================================================

    for idx, config in enumerate(
        CONFIGS,
        start=1,
    ):

        # candidate closure
        def candidate_fn(
            config=config,
        ):
            return launch(
                x,
                weight,
                bias,
                y_candidate,
                config,
            )

        try:
            # ------------------------------------------------
            # Compile candidate first
            # ------------------------------------------------

            with torch.inference_mode():

                candidate_fn()

                for _ in range(3):
                    baseline_fn()
                    candidate_fn()

            torch.cuda.synchronize()

            # ------------------------------------------------
            # Robust paired benchmark
            # ------------------------------------------------

            stat = compare_candidate(
                baseline_fn,
                candidate_fn,
            )

            candidate_median = (
                stat["candidate_median"]
            )

            paired = (
                stat["paired_median"]
            )

            vs_torch = (
                torch_median
                / candidate_median
            )

            results.append(
                {
                    "config": config,
                    **stat,

                    "vs_torch": vs_torch,

                    "tflops": tflops(
                        candidate_median
                    ),
                }
            )

            print(
                f"{idx:>3} "
                f"{config['BK']:>4} "
                f"{config['W']:>3} "
                f"{config['G']:>3} "
                f"{config['S']:>3} "
                f"{candidate_median:>8.4f} ms "
                f"{stat['baseline_median']:>8.4f} ms "
                f"{paired:>8.3f}x "
                f"{vs_torch:>8.3f}x"
            )

        except Exception as e:

            print(
                f"{idx:>3} "
                f"{config['BK']:>4} "
                f"{config['W']:>3} "
                f"{config['G']:>3} "
                f"{config['S']:>3} "
                f"{'FAIL':>10} "
                f"{type(e).__name__}"
            )

    # ========================================================
    # Ranking
    #
    # 핵심:
    #
    # single candidate latency가 아니라
    # paired median speedup으로 정렬.
    # ========================================================

    results.sort(
        key=lambda r: r[
            "paired_median"
        ],
        reverse=True,
    )

    # ========================================================
    # Top 15
    # ========================================================

    print()
    print("=" * 120)
    print("TOP 15 — ranked by median paired speedup vs current baseline")
    print("=" * 120)

    print()

    for rank, result in enumerate(
        results[:15],
        start=1,
    ):

        config = result[
            "config"
        ]

        print(
            f"[{rank:02d}] "
            f"{config_name(config)} "
            f"| cand={result['candidate_median']:.4f} ms "
            f"| base={result['baseline_median']:.4f} ms "
            f"| paired={result['paired_median']:.3f}x "
            f"| Torch={result['vs_torch']:.3f}x "
            f"| {result['tflops']:.2f} TFLOPS"
        )

    # ========================================================
    # Best
    # ========================================================

    if not results:
        print()
        print("No valid configuration.")
        return

    best = results[0]

    best_config = best[
        "config"
    ]

    print()
    print("=" * 120)
    print("BEST ROBUST CONFIG")
    print("=" * 120)

    print()

    print(
        config_name(
            best_config
        )
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
        f"Candidate min/max     : "
        f"{best['candidate_min']:.4f} / "
        f"{best['candidate_max']:.4f} ms"
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

    print(
        f"Best TFLOPS           : "
        f"{best['tflops']:.2f}"
    )

    # ========================================================
    # Is it actually worth changing?
    # ========================================================

    print()
    print("=" * 120)
    print("Decision")
    print("=" * 120)

    print()

    improvement = (
        best["paired_median"]
        - 1.0
    )

    if improvement >= 0.03:

        print(
            "Meaningful win: "
            "candidate is >= 3% faster than current baseline."
        )

        print(
            "→ production config 변경 후보."
        )

    elif improvement >= 0.01:

        print(
            "Small win: "
            "candidate is only 1~3% faster."
        )

        print(
            "→ 한 번 더 focused A/B 후 결정."
        )

    else:

        print(
            "No meaningful tile-level improvement."
        )

        print(
            "→ 현재 config 유지하고 "
            "SASS / register / occupancy 분석으로 이동."
        )

    # ========================================================
    # Correctness of best candidate
    # ========================================================

    with torch.inference_mode():

        ref = torch_fn()

        launch(
            x,
            weight,
            bias,
            y_candidate,
            best_config,
        )

    torch.cuda.synchronize()

    diff = (
        ref.float()
        - y_candidate.float()
    ).abs()

    print()
    print("=" * 120)
    print("Correctness")
    print("=" * 120)

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