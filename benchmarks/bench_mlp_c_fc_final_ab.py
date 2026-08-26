import statistics

import torch
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

ROUNDS = 20


BASELINE = {
    "BM": 128,
    "BN": 128,
    "BK": 32,
    "W": 4,
    "G": 8,
    "S": 2,
}


CANDIDATE = {
    "BM": 128,
    "BN": 128,
    "BK": 64,
    "W": 4,
    "G": 8,
    "S": 3,
}


def launch(
    x,
    weight,
    bias,
    y,
    cfg,
):
    bm = cfg["BM"]
    bn = cfg["BN"]

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

        BLOCK_M=cfg["BM"],
        BLOCK_N=cfg["BN"],
        BLOCK_K=cfg["BK"],

        GROUP_SIZE_M=cfg["G"],

        num_warps=cfg["W"],
        num_stages=cfg["S"],
    )


def bench(fn):
    with torch.inference_mode():
        return triton.testing.do_bench(
            fn
        )


def main():
    torch.manual_seed(0)

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

    y_a = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    y_b = torch.empty_like(
        y_a
    )

    def baseline():
        launch(
            x,
            weight,
            bias,
            y_a,
            BASELINE,
        )

    def candidate():
        launch(
            x,
            weight,
            bias,
            y_b,
            CANDIDATE,
        )

    # compile + warmup
    with torch.inference_mode():
        for _ in range(20):
            baseline()
            candidate()

    torch.cuda.synchronize()

    ratios = []
    baseline_refs = []
    candidate_times = []

    print()
    print("=" * 90)
    print("Final c_fc GEMM A/B")
    print("=" * 90)
    print()

    for i in range(ROUNDS):

        # ----------------------------------------------------
        # A → B → A
        #
        # 다음 round는 B → A → B처럼 뒤집지 않고
        # baseline을 candidate 양쪽에서 감싸서
        # clock drift를 보정.
        # ----------------------------------------------------

        a_before = bench(
            baseline
        )

        b = bench(
            candidate
        )

        a_after = bench(
            baseline
        )

        a_ref = (
            a_before
            + a_after
        ) / 2.0

        ratio = (
            a_ref
            / b
        )

        baseline_refs.append(
            a_ref
        )

        candidate_times.append(
            b
        )

        ratios.append(
            ratio
        )

        print(
            f"[{i + 1:02d}/{ROUNDS}] "
            f"A_before={a_before:.4f}  "
            f"B={b:.4f}  "
            f"A_after={a_after:.4f}  "
            f"A_ref={a_ref:.4f}  "
            f"speedup={ratio:.3f}x"
        )

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)
    print()

    a_median = statistics.median(
        baseline_refs
    )

    b_median = statistics.median(
        candidate_times
    )

    ratio_median = statistics.median(
        ratios
    )

    ratio_mean = statistics.mean(
        ratios
    )

    print(
        f"Baseline median reference : "
        f"{a_median:.4f} ms"
    )

    print(
        f"Candidate median          : "
        f"{b_median:.4f} ms"
    )

    print()

    print(
        f"Median paired speedup     : "
        f"{ratio_median:.3f}x"
    )

    print(
        f"Mean paired speedup       : "
        f"{ratio_mean:.3f}x"
    )

    print()

    wins = sum(
        ratio > 1.0
        for ratio in ratios
    )

    print(
        f"Candidate wins            : "
        f"{wins}/{ROUNDS}"
    )

    print()

    if (
        ratio_median >= 1.03
        and wins >= 15
    ):
        print(
            "Decision: candidate wins robustly."
        )
        print(
            "→ production config를 BK=64, S=3로 변경."
        )

    elif ratio_median >= 1.01:
        print(
            "Decision: small / marginal improvement."
        )
        print(
            "→ 현재 config 유지해도 무방."
        )

    else:
        print(
            "Decision: no robust improvement."
        )
        print(
            "→ 기존 config 유지."
        )


if __name__ == "__main__":
    main()