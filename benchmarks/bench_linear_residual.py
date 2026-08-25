import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear_residual import (
    triton_linear_residual,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def bench_case(
    name,
    K,
    N,
):
    torch.manual_seed(0)

    # ========================================================
    # Inputs
    # ========================================================

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

    residual = torch.randn(
        B,
        T,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    dy = torch.randn(
        B,
        T,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # Separate tensors for backward
    # ========================================================

    x_ref = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    w_ref = (
        weight.detach()
        .clone()
        .requires_grad_(True)
    )

    b_ref = (
        bias.detach()
        .clone()
        .requires_grad_(True)
    )

    r_ref = (
        residual.detach()
        .clone()
        .requires_grad_(True)
    )

    x_tri = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    w_tri = (
        weight.detach()
        .clone()
        .requires_grad_(True)
    )

    b_tri = (
        bias.detach()
        .clone()
        .requires_grad_(True)
    )

    r_tri = (
        residual.detach()
        .clone()
        .requires_grad_(True)
    )

    # ========================================================
    # Forward
    # ========================================================

    def torch_forward():
        return (
            F.linear(
                x_ref,
                w_ref,
                b_ref,
            )
            + r_ref
        )

    def triton_forward():
        return triton_linear_residual(
            x_tri,
            w_tri,
            b_tri,
            r_tri,
        )

    # ========================================================
    # Forward + Backward
    # ========================================================

    def torch_fb():
        x_ref.grad = None
        w_ref.grad = None
        b_ref.grad = None
        r_ref.grad = None

        y = (
            F.linear(
                x_ref,
                w_ref,
                b_ref,
            )
            + r_ref
        )

        y.backward(
            dy
        )

    def triton_fb():
        x_tri.grad = None
        w_tri.grad = None
        b_tri.grad = None
        r_tri.grad = None

        y = triton_linear_residual(
            x_tri,
            w_tri,
            b_tri,
            r_tri,
        )

        y.backward(
            dy
        )

    # ========================================================
    # Warmup
    # ========================================================

    for _ in range(3):
        torch_forward()
        triton_forward()

    torch_fb()
    triton_fb()

    torch.cuda.synchronize()

    # ========================================================
    # Benchmark
    # ========================================================

    torch_fwd_ms = triton.testing.do_bench(
        torch_forward
    )

    triton_fwd_ms = triton.testing.do_bench(
        triton_forward
    )

    torch_fb_ms = triton.testing.do_bench(
        torch_fb
    )

    triton_fb_ms = triton.testing.do_bench(
        triton_fb
    )

    torch_bwd_ms = (
        torch_fb_ms
        - torch_fwd_ms
    )

    triton_bwd_ms = (
        triton_fb_ms
        - triton_fwd_ms
    )

    return {
        "name": name,

        "torch_fwd": torch_fwd_ms,
        "triton_fwd": triton_fwd_ms,

        "torch_fb": torch_fb_ms,
        "triton_fb": triton_fb_ms,

        "torch_bwd": torch_bwd_ms,
        "triton_bwd": triton_bwd_ms,
    }


def print_result(
    result,
):
    print()
    print("=" * 90)
    print(result["name"])
    print("=" * 90)

    print()
    print("[Forward]")

    print(
        f"PyTorch : "
        f"{result['torch_fwd']:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{result['triton_fwd']:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{result['torch_fwd'] / result['triton_fwd']:.2f}x"
    )

    print()
    print("[Forward + Backward]")

    print(
        f"PyTorch : "
        f"{result['torch_fb']:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{result['triton_fb']:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{result['torch_fb'] / result['triton_fb']:.2f}x"
    )

    print()
    print("[Approx. Backward]")

    print(
        f"PyTorch : "
        f"{result['torch_bwd']:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{result['triton_bwd']:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{result['torch_bwd'] / result['triton_bwd']:.2f}x"
    )


def main():
    torch.cuda.init()

    # ========================================================
    # Attention projection
    #
    # K=768 → N=768
    # ========================================================

    attn = bench_case(
        name=(
            "Attention c_proj + Residual "
            "B=8 T=1024 K=768 N=768"
        ),
        K=768,
        N=768,
    )

    # ========================================================
    # MLP projection
    #
    # K=3072 → N=768
    # ========================================================

    mlp = bench_case(
        name=(
            "MLP c_proj + Residual "
            "B=8 T=1024 K=3072 N=768"
        ),
        K=3072,
        N=768,
    )

    # ========================================================
    # Results
    # ========================================================

    print_result(
        attn
    )

    print_result(
        mlp
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print()

    print(
        f"{'Case':<24} "
        f"{'Fwd':>10} "
        f"{'F+B':>10}"
    )

    print("-" * 48)

    print(
        f"{'Attention c_proj':<24} "
        f"{attn['torch_fwd'] / attn['triton_fwd']:>9.2f}x "
        f"{attn['torch_fb'] / attn['triton_fb']:>9.2f}x"
    )

    print(
        f"{'MLP c_proj':<24} "
        f"{mlp['torch_fwd'] / mlp['triton_fwd']:>9.2f}x "
        f"{mlp['torch_fb'] / mlp['triton_fb']:>9.2f}x"
    )


if __name__ == "__main__":
    main()