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

ROUNDS = 10


OLD_CONFIG = {
    "BM": 128,
    "BN": 128,
    "BK": 32,
    "W": 4,
    "G": 8,
    "S": 2,
}


NEW_CONFIG = {
    "BM": 256,
    "BN": 128,
    "BK": 64,
    "W": 8,
    "G": 8,
    "S": 2,
}


def launch(
    x,
    weight,
    bias,
    y,
    config,
):
    BM = config["BM"]
    BN = config["BN"]
    BK = config["BK"]

    grid = (
        triton.cdiv(M, BM)
        * triton.cdiv(N, BN),
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

        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,

        GROUP_SIZE_M=config["G"],

        num_warps=config["W"],
        num_stages=config["S"],
    )


def bench_once(fn):
    with torch.inference_mode():
        return triton.testing.do_bench(
            fn
        )


def print_stats(
    name,
    values,
):
    print(
        f"{name:<16} "
        f"median={statistics.median(values):.4f} ms  "
        f"mean={statistics.mean(values):.4f} ms  "
        f"min={min(values):.4f} ms  "
        f"max={max(values):.4f} ms"
    )


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

    y_old = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    y_new = torch.empty_like(
        y_old
    )

    # ========================================================
    # Functions
    # ========================================================

    def torch_fn():
        return F.linear(
            x,
            weight,
            bias,
        )

    def old_fn():
        launch(
            x,
            weight,
            bias,
            y_old,
            OLD_CONFIG,
        )

        return y_old

    def new_fn():
        launch(
            x,
            weight,
            bias,
            y_new,
            NEW_CONFIG,
        )

        return y_new

    # ========================================================
    # Correctness
    # ========================================================

    with torch.inference_mode():

        ref = torch_fn()

        old_fn()
        new_fn()

    torch.cuda.synchronize()

    old_diff = (
        ref.float()
        - y_old.float()
    ).abs()

    new_diff = (
        ref.float()
        - y_new.float()
    ).abs()

    print()
    print("=" * 90)
    print("Correctness")
    print("=" * 90)

    print(
        f"OLD max diff : "
        f"{old_diff.max().item():.6e}"
    )

    print(
        f"NEW max diff : "
        f"{new_diff.max().item():.6e}"
    )

    # ========================================================
    # Warmup
    # ========================================================

    with torch.inference_mode():
        for _ in range(10):
            torch_fn()
            old_fn()
            new_fn()

    torch.cuda.synchronize()

    # ========================================================
    # Repeated A/B/C
    #
    # 순서를 계속 회전시켜 cache/order bias를 줄인다.
    # ========================================================

    torch_results = []
    old_results = []
    new_results = []

    orders = [
        ("torch", "old", "new"),
        ("old", "new", "torch"),
        ("new", "torch", "old"),
    ]

    funcs = {
        "torch": torch_fn,
        "old": old_fn,
        "new": new_fn,
    }

    results = {
        "torch": torch_results,
        "old": old_results,
        "new": new_results,
    }

    print()
    print("=" * 90)
    print(
        f"Repeated c_fc GEMM Benchmark "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 90)

    print()

    for round_idx in range(
        ROUNDS
    ):
        order = orders[
            round_idx % len(orders)
        ]

        current = {}

        for name in order:
            ms = bench_once(
                funcs[name]
            )

            results[name].append(
                ms
            )

            current[name] = ms

        print(
            f"[{round_idx + 1:02d}/{ROUNDS}] "
            f"Torch={current['torch']:.4f}  "
            f"OLD={current['old']:.4f}  "
            f"NEW={current['new']:.4f} ms  "
            f"| NEW/OLD="
            f"{current['old'] / current['new']:.3f}x"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("Statistics")
    print("=" * 90)

    print()

    print_stats(
        "PyTorch",
        torch_results,
    )

    print_stats(
        "OLD Triton",
        old_results,
    )

    print_stats(
        "NEW Triton",
        new_results,
    )

    torch_median = statistics.median(
        torch_results
    )

    old_median = statistics.median(
        old_results
    )

    new_median = statistics.median(
        new_results
    )

    print()
    print("=" * 90)
    print("Median Comparison")
    print("=" * 90)

    print()

    print(
        f"PyTorch       : "
        f"{torch_median:.4f} ms"
    )

    print(
        f"OLD Triton    : "
        f"{old_median:.4f} ms"
    )

    print(
        f"NEW Triton    : "
        f"{new_median:.4f} ms"
    )

    print()

    print(
        f"NEW vs OLD    : "
        f"{old_median / new_median:.3f}x"
    )

    print(
        f"NEW vs PyTorch: "
        f"{torch_median / new_median:.3f}x"
    )

    # ========================================================
    # Paired NEW / OLD
    # ========================================================

    paired_speedups = [
        old / new
        for old, new in zip(
            old_results,
            new_results,
        )
    ]

    print()

    print(
        f"Median paired NEW vs OLD: "
        f"{statistics.median(paired_speedups):.3f}x"
    )

    print(
        f"Mean paired NEW vs OLD  : "
        f"{statistics.mean(paired_speedups):.3f}x"
    )


if __name__ == "__main__":
    main()