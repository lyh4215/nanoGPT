import statistics

import torch
import triton

from model import (
    GPTConfig,
    Block,
)

from triton_model import (
    TritonBlock,
)

from triton_kernels.linear import (
    triton_linear,
)

from triton_kernels.linear_residual import (
    triton_linear_residual,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024
C = 768

ROUNDS = 10


def bench_once(fn):
    with torch.inference_mode():
        return triton.testing.do_bench(
            fn
        )


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Config
    # ========================================================

    config = GPTConfig(
        block_size=T,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=C,
        dropout=0.0,
        bias=True,
    )

    # ========================================================
    # Model
    # ========================================================

    ref_block = Block(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri_block = TritonBlock(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri_block.load_state_dict(
        ref_block.state_dict()
    )

    ref_block.eval()
    tri_block.eval()

    tri_mlp = tri_block.mlp

    # ========================================================
    # Input
    #
    # c_proj input:
    #
    # [B, T, 3072]
    #
    # 실제 MLP에서는 c_fc + GELU 결과가 들어오지만,
    # 이번 benchmark의 목적은 c_proj + residual 자체의
    # fused / unfused A/B 비교이므로 hidden을 직접 생성.
    # ========================================================

    hidden_size = 4 * C

    hidden = torch.randn(
        B,
        T,
        hidden_size,
        device=DEVICE,
        dtype=DTYPE,
    )

    residual = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # Unfused
    #
    # GEMM
    #   ↓
    # temporary HBM store
    #   ↓
    # separate residual add
    # ========================================================

    def unfused():
        y = triton_linear(
            hidden,
            tri_mlp.c_proj.weight,
            tri_mlp.c_proj.bias,
        )

        return (
            y
            + residual
        )

    # ========================================================
    # Fused
    #
    # GEMM accumulator
    #   ↓
    # residual load + add
    #   ↓
    # final store
    # ========================================================

    def fused():
        return triton_linear_residual(
            hidden,
            tri_mlp.c_proj.weight,
            tri_mlp.c_proj.bias,
            residual,
        )

    # ========================================================
    # Correctness
    # ========================================================

    with torch.inference_mode():
        y_unfused = unfused()
        y_fused = fused()

    torch.cuda.synchronize()

    diff = (
        y_unfused.float()
        - y_fused.float()
    ).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print()
    print("=" * 90)
    print("MLP c_proj + Residual Correctness")
    print("=" * 90)

    print(
        f"max diff  : "
        f"{max_diff:.6e}"
    )

    print(
        f"mean diff : "
        f"{mean_diff:.6e}"
    )

    # ========================================================
    # Warmup
    #
    # 두 kernel 모두 compile + 충분히 warmup.
    # ========================================================

    with torch.inference_mode():
        for _ in range(10):
            unfused()
            fused()

    torch.cuda.synchronize()

    # ========================================================
    # Repeated alternating benchmark
    #
    # 측정 순서 자체가 결과에 영향을 주는 걸 줄이기 위해:
    #
    # round 0:
    #   unfused → fused
    #
    # round 1:
    #   fused → unfused
    #
    # ...
    # ========================================================

    unfused_results = []
    fused_results = []

    for i in range(ROUNDS):

        if i % 2 == 0:

            unfused_ms = bench_once(
                unfused
            )

            fused_ms = bench_once(
                fused
            )

        else:

            fused_ms = bench_once(
                fused
            )

            unfused_ms = bench_once(
                unfused
            )

        unfused_results.append(
            unfused_ms
        )

        fused_results.append(
            fused_ms
        )

        print(
            f"[{i + 1:02d}/{ROUNDS}] "
            f"unfused={unfused_ms:.4f} ms  "
            f"fused={fused_ms:.4f} ms  "
            f"speedup={unfused_ms / fused_ms:.3f}x"
        )

    # ========================================================
    # Statistics
    # ========================================================

    unfused_median = statistics.median(
        unfused_results
    )

    fused_median = statistics.median(
        fused_results
    )

    unfused_mean = statistics.mean(
        unfused_results
    )

    fused_mean = statistics.mean(
        fused_results
    )

    unfused_min = min(
        unfused_results
    )

    fused_min = min(
        fused_results
    )

    unfused_max = max(
        unfused_results
    )

    fused_max = max(
        fused_results
    )

    # round별 paired speedup도 같이 계산
    speedups = [
        u / f
        for u, f in zip(
            unfused_results,
            fused_results,
        )
    ]

    speedup_median = statistics.median(
        speedups
    )

    speedup_mean = statistics.mean(
        speedups
    )

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 90)
    print(
        f"MLP c_proj + Residual A/B "
        f"B={B}, T={T}, "
        f"K={hidden_size}, N={C}"
    )
    print("=" * 90)

    print()

    print(
        f"{'Metric':<20} "
        f"{'Unfused':>12} "
        f"{'Fused':>12}"
    )

    print("-" * 48)

    print(
        f"{'Median':<20} "
        f"{unfused_median:>10.4f} ms "
        f"{fused_median:>10.4f} ms"
    )

    print(
        f"{'Mean':<20} "
        f"{unfused_mean:>10.4f} ms "
        f"{fused_mean:>10.4f} ms"
    )

    print(
        f"{'Min':<20} "
        f"{unfused_min:>10.4f} ms "
        f"{fused_min:>10.4f} ms"
    )

    print(
        f"{'Max':<20} "
        f"{unfused_max:>10.4f} ms "
        f"{fused_max:>10.4f} ms"
    )

    # ========================================================
    # Speedup
    # ========================================================

    print()
    print("=" * 90)
    print("Speedup")
    print("=" * 90)

    print()

    print(
        f"median(unfused) / median(fused) : "
        f"{unfused_median / fused_median:.3f}x"
    )

    print(
        f"median paired speedup            : "
        f"{speedup_median:.3f}x"
    )

    print(
        f"mean paired speedup              : "
        f"{speedup_mean:.3f}x"
    )

    print()

    saved_ms = (
        unfused_median
        - fused_median
    )

    print(
        f"Median saved per MLP : "
        f"{saved_ms:+.4f} ms"
    )

    print()

    if speedup_median > 1.03:
        print(
            "Conclusion: fused path is consistently faster."
        )

    elif speedup_median < 0.97:
        print(
            "Conclusion: fused path is consistently slower."
        )

    else:
        print(
            "Conclusion: performance is effectively tied "
            "within a small margin."
        )


if __name__ == "__main__":
    main()