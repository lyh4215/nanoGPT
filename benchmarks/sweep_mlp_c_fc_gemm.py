import torch
import torch.nn.functional as F

import triton
import triton.language as tl


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024

M = B * T
K = 768
N = 3072


# ============================================================
# GEMM
#
# Y = X @ W.T + bias
#
# X : [M, K]
# W : [N, K]
# Y : [M, N]
#
# c_fc:
#
# [8192, 768] @ [3072, 768].T
#       ->
# [8192, 3072]
# ============================================================


@triton.jit
def _c_fc_kernel(
    X,
    W,
    BIAS,
    Y,

    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    stride_xm,
    stride_xk,

    stride_wn,
    stride_wk,

    stride_ym,
    stride_yn,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,

    GROUP_SIZE_M: tl.constexpr,
):
    # ========================================================
    # Program ordering
    # ========================================================

    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(
        M,
        BLOCK_M,
    )

    num_pid_n = tl.cdiv(
        N,
        BLOCK_N,
    )

    num_pid_in_group = (
        GROUP_SIZE_M
        * num_pid_n
    )

    group_id = (
        pid
        // num_pid_in_group
    )

    first_pid_m = (
        group_id
        * GROUP_SIZE_M
    )

    group_size_m = min(
        num_pid_m - first_pid_m,
        GROUP_SIZE_M,
    )

    pid_m = (
        first_pid_m
        + (
            pid
            % num_pid_in_group
        )
        % group_size_m
    )

    pid_n = (
        (
            pid
            % num_pid_in_group
        )
        // group_size_m
    )

    # ========================================================
    # Output offsets
    # ========================================================

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    offs_k = tl.arange(
        0,
        BLOCK_K,
    )

    # ========================================================
    # Initial pointers
    #
    # X tile:
    #   [BLOCK_M, BLOCK_K]
    #
    # W tile:
    #   [BLOCK_N, BLOCK_K]
    #
    # 이후 tl.trans(W tile)해서
    #
    # [BM,BK] @ [BK,BN]
    # ========================================================

    x_ptrs = (
        X
        + offs_m[:, None] * stride_xm
        + offs_k[None, :] * stride_xk
    )

    w_ptrs = (
        W
        + offs_n[:, None] * stride_wn
        + offs_k[None, :] * stride_wk
    )

    # ========================================================
    # Accumulator
    # ========================================================

    acc = tl.zeros(
        (
            BLOCK_M,
            BLOCK_N,
        ),
        dtype=tl.float32,
    )

    # ========================================================
    # K reduction
    # ========================================================

    for k_start in range(
        0,
        K,
        BLOCK_K,
    ):
        k_mask = (
            k_start
            + offs_k
            < K
        )

        x = tl.load(
            x_ptrs,
            mask=(
                (offs_m[:, None] < M)
                & k_mask[None, :]
            ),
            other=0.0,
        )

        w = tl.load(
            w_ptrs,
            mask=(
                (offs_n[:, None] < N)
                & k_mask[None, :]
            ),
            other=0.0,
        )

        acc += tl.dot(
            x,
            tl.trans(w),
        )

        x_ptrs += (
            BLOCK_K
            * stride_xk
        )

        w_ptrs += (
            BLOCK_K
            * stride_wk
        )

    # ========================================================
    # Bias epilogue
    # ========================================================

    bias = tl.load(
        BIAS + offs_n,
        mask=offs_n < N,
        other=0.0,
    )

    acc += bias[None, :]

    # ========================================================
    # Store
    # ========================================================

    y_ptrs = (
        Y
        + offs_m[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )

    mask = (
        (offs_m[:, None] < M)
        & (offs_n[None, :] < N)
    )

    tl.store(
        y_ptrs,
        acc,
        mask=mask,
    )


# ============================================================
# Candidate tile configurations
#
# 일부러 무작정 Cartesian product 하지 않고,
# T4에서 가능성 있는 tile들 위주로 잡음.
#
# tuple:
#
# (BM, BN, BK, warps)
# ============================================================

TILES = [
    # smaller tiles
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (128, 64, 32, 4),

    # 우리가 기존에 잘 쓰던 영역
    (128, 128, 32, 4),

    # wider N tile
    (64, 256, 32, 8),
    (128, 256, 32, 8),

    # larger BK
    (64, 64, 64, 4),
    (64, 128, 64, 4),
    (128, 64, 64, 4),
    (128, 128, 64, 4),

    # wide N + BK64
    (64, 256, 64, 8),
    (128, 256, 64, 8),

    # larger M tile
    (256, 64, 32, 8),
    (256, 128, 32, 8),

    (256, 64, 64, 8),
    (256, 128, 64, 8),
]


GROUPS = [
    1,
    2,
    4,
    8,
]


NUM_STAGES = [
    2,
    3,
    4,
]


def tflops(
    ms,
):
    flops = (
        2
        * M
        * N
        * K
    )

    return (
        flops
        / (ms * 1e-3)
        / 1e12
    )


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

    y = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # PyTorch / cuBLAS baseline
    # ========================================================

    def torch_fn():
        return F.linear(
            x,
            weight,
            bias,
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
    print("=" * 110)
    print(
        f"MLP c_fc GEMM Sweep "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 110)

    print()

    print(
        f"PyTorch F.linear : "
        f"{torch_ms:.4f} ms"
    )

    print(
        f"PyTorch TFLOPS   : "
        f"{tflops(torch_ms):.2f}"
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
        f"{'TFLOPS':>10} "
        f"{'vs Torch':>10}"
    )

    print("-" * 90)

    results = []

    # ========================================================
    # Sweep
    # ========================================================

    for (
        block_m,
        block_n,
        block_k,
        num_warps,
    ) in TILES:

        for group_size_m in GROUPS:

            for num_stages in NUM_STAGES:

                grid = (
                    triton.cdiv(
                        M,
                        block_m,
                    )
                    * triton.cdiv(
                        N,
                        block_n,
                    ),
                )

                def triton_fn():
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

                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,

                        GROUP_SIZE_M=group_size_m,

                        num_warps=num_warps,
                        num_stages=num_stages,
                    )

                try:
                    # ----------------------------------------
                    # Compile + first launch
                    # ----------------------------------------

                    with torch.inference_mode():
                        triton_fn()

                    torch.cuda.synchronize()

                    # ----------------------------------------
                    # Benchmark
                    # ----------------------------------------

                    with torch.inference_mode():
                        ms = triton.testing.do_bench(
                            triton_fn
                        )

                    perf = tflops(
                        ms
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
                            group_size_m,
                            num_stages,
                            perf,
                            speedup,
                        )
                    )

                    print(
                        f"{block_m:>5} "
                        f"{block_n:>5} "
                        f"{block_k:>5} "
                        f"{num_warps:>4} "
                        f"{group_size_m:>4} "
                        f"{num_stages:>4} "
                        f"{ms:>10.4f} "
                        f"{perf:>10.2f} "
                        f"{speedup:>9.2f}x"
                    )

                except Exception as e:
                    print(
                        f"{block_m:>5} "
                        f"{block_n:>5} "
                        f"{block_k:>5} "
                        f"{num_warps:>4} "
                        f"{group_size_m:>4} "
                        f"{num_stages:>4} "
                        f"{'FAIL':>10} "
                        f"{'':>10} "
                        f"{type(e).__name__}"
                    )

    # ========================================================
    # Sort
    # ========================================================

    results.sort(
        key=lambda x: x[0]
    )

    # ========================================================
    # Top 20
    # ========================================================

    print()
    print("=" * 110)
    print("TOP 20")
    print("=" * 110)

    print()

    for rank, result in enumerate(
        results[:20],
        start=1,
    ):
        (
            ms,
            bm,
            bn,
            bk,
            warps,
            group,
            stages,
            perf,
            speedup,
        ) = result

        print(
            f"[{rank:02d}] "
            f"BM={bm:<3} "
            f"BN={bn:<3} "
            f"BK={bk:<3} "
            f"W={warps} "
            f"G={group:<2} "
            f"S={stages} "
            f"| {ms:.4f} ms "
            f"| {perf:.2f} TFLOPS "
            f"| {speedup:.2f}x Torch"
        )

    # ========================================================
    # Best
    # ========================================================

    if not results:
        print()
        print("No valid configurations.")
        return

    (
        best_ms,
        best_bm,
        best_bn,
        best_bk,
        best_warps,
        best_group,
        best_stages,
        best_perf,
        best_speedup,
    ) = results[0]

    print()
    print("=" * 110)
    print("BEST")
    print("=" * 110)

    print()

    print(
        f"BM          : {best_bm}"
    )

    print(
        f"BN          : {best_bn}"
    )

    print(
        f"BK          : {best_bk}"
    )

    print(
        f"num_warps   : {best_warps}"
    )

    print(
        f"GROUP_SIZE_M: {best_group}"
    )

    print(
        f"num_stages  : {best_stages}"
    )

    print()

    print(
        f"Triton      : "
        f"{best_ms:.4f} ms"
    )

    print(
        f"PyTorch     : "
        f"{torch_ms:.4f} ms"
    )

    print()

    print(
        f"Triton      : "
        f"{best_perf:.2f} TFLOPS"
    )

    print(
        f"PyTorch     : "
        f"{tflops(torch_ms):.2f} TFLOPS"
    )

    print()

    print(
        f"Speedup     : "
        f"{best_speedup:.3f}x"
    )

    # ========================================================
    # Correctness of best config
    # ========================================================

    best_grid = (
        triton.cdiv(
            M,
            best_bm,
        )
        * triton.cdiv(
            N,
            best_bn,
        ),
    )

    with torch.inference_mode():

        _c_fc_kernel[best_grid](
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

            BLOCK_M=best_bm,
            BLOCK_N=best_bn,
            BLOCK_K=best_bk,

            GROUP_SIZE_M=best_group,

            num_warps=best_warps,
            num_stages=best_stages,
        )

        ref = F.linear(
            x,
            weight,
            bias,
        )

    torch.cuda.synchronize()

    diff = (
        ref.float()
        - y.float()
    ).abs()

    max_diff = (
        diff.max().item()
    )

    mean_diff = (
        diff.mean().item()
    )

    allclose = torch.allclose(
        ref,
        y,
        atol=5e-2,
        rtol=1e-2,
    )

    print()
    print("=" * 110)
    print("Correctness")
    print("=" * 110)

    print()

    print(
        f"max diff  : "
        f"{max_diff:.6e}"
    )

    print(
        f"mean diff : "
        f"{mean_diff:.6e}"
    )

    print(
        f"allclose  : "
        f"{allclose}"
    )


if __name__ == "__main__":
    main()