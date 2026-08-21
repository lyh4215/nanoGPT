import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear import triton_linear


DEVICE = "cuda"
DTYPE = torch.float16


# ============================================================
# Helpers
# ============================================================

def print_diff(name, a, b):
    diff = (a.float() - b.float()).abs()

    print(
        f"{name:>8} | "
        f"max={diff.max().item():.6e} | "
        f"mean={diff.mean().item():.6e}"
    )


# ============================================================
# 1. Autograd correctness
# ============================================================

def test_correctness(
    K,
    N,
    B=2,
    T=128,
):
    print()
    print("=" * 75)
    print(
        f"Correctness: "
        f"[{B},{T},{K}] -> [{B},{T},{N}]"
    )
    print("=" * 75)

    torch.manual_seed(0)

    # --------------------------------------------------------
    # common values
    # --------------------------------------------------------

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

    dy = torch.randn(
        B,
        T,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------

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

    y_ref = F.linear(
        x_ref,
        w_ref,
        b_ref,
    )

    y_ref.backward(dy)

    # --------------------------------------------------------
    # Triton
    # --------------------------------------------------------

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

    y_tri = triton_linear(
        x_tri,
        w_tri,
        b_tri,
    )

    y_tri.backward(dy)

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    print_diff(
        "Forward",
        y_ref,
        y_tri,
    )

    print_diff(
        "dX",
        x_ref.grad,
        x_tri.grad,
    )

    print_diff(
        "dW",
        w_ref.grad,
        w_tri.grad,
    )

    print_diff(
        "db",
        b_ref.grad,
        b_tri.grad,
    )


# ============================================================
# 2. Benchmark one Linear shape
# ============================================================

def benchmark_shape(
    name,
    K,
    N,
    B=8,
    T=1024,
):
    print()
    print("=" * 75)
    print(
        f"{name}: "
        f"[{B},{T},{K}] -> [{B},{T},{N}]"
    )
    print("=" * 75)

    torch.manual_seed(0)

    # --------------------------------------------------------
    # PyTorch tensors
    # --------------------------------------------------------

    x_ref = torch.randn(
        B,
        T,
        K,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    w_ref = torch.randn(
        N,
        K,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    b_ref = torch.randn(
        N,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # --------------------------------------------------------
    # Triton tensors
    #
    # 값은 PyTorch와 동일하게 맞춘다.
    # --------------------------------------------------------

    x_tri = (
        x_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    w_tri = (
        w_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    b_tri = (
        b_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    # 동일한 upstream gradient
    dy = torch.randn(
        B,
        T,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # Forward only
    # --------------------------------------------------------

    def torch_forward():
        F.linear(
            x_ref,
            w_ref,
            b_ref,
        )

    def triton_forward():
        triton_linear(
            x_tri,
            w_tri,
            b_tri,
        )

    # --------------------------------------------------------
    # Forward + Backward
    # --------------------------------------------------------

    def torch_fb():
        x_ref.grad = None
        w_ref.grad = None
        b_ref.grad = None

        y = F.linear(
            x_ref,
            w_ref,
            b_ref,
        )

        y.backward(dy)

    def triton_fb():
        x_tri.grad = None
        w_tri.grad = None
        b_tri.grad = None

        y = triton_linear(
            x_tri,
            w_tri,
            b_tri,
        )

        y.backward(dy)

    # --------------------------------------------------------
    # Warmup / compile
    # --------------------------------------------------------

    triton_forward()
    triton_fb()

    x_tri.grad = None
    w_tri.grad = None
    b_tri.grad = None

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    torch_fwd_ms = triton.testing.do_bench(
        torch_forward,
    )

    triton_fwd_ms = triton.testing.do_bench(
        triton_forward,
    )

    torch_fb_ms = triton.testing.do_bench(
        torch_fb,
    )

    triton_fb_ms = triton.testing.do_bench(
        triton_fb,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("[Forward]")

    print(
        f"PyTorch : "
        f"{torch_fwd_ms:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{triton_fwd_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_fwd_ms / triton_fwd_ms:.2f}x"
    )

    print()

    print("[Forward + Backward]")

    print(
        f"PyTorch : "
        f"{torch_fb_ms:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{triton_fb_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_fb_ms / triton_fb_ms:.2f}x"
    )

    return {
        "name": name,

        "torch_fwd": torch_fwd_ms,
        "triton_fwd": triton_fwd_ms,

        "torch_fb": torch_fb_ms,
        "triton_fb": triton_fb_ms,
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # GPT-2 small Linear shapes
    # ========================================================

    shapes = [
        (
            "QKV projection",
            768,
            2304,
        ),

        (
            "Attention c_proj",
            768,
            768,
        ),

        (
            "MLP c_fc",
            768,
            3072,
        ),

        (
            "MLP c_proj",
            3072,
            768,
        ),
    ]

    # ========================================================
    # Correctness
    #
    # 먼저 대표 shape(QKV)로 전체 autograd 확인
    # ========================================================

    test_correctness(
        K=768,
        N=2304,
        B=2,
        T=128,
    )

    # ========================================================
    # Benchmark
    # ========================================================

    results = []

    for name, K, N in shapes:
        result = benchmark_shape(
            name=name,
            K=K,
            N=N,
            B=8,
            T=1024,
        )

        results.append(result)

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"{'Operation':22s}"
        f"{'Torch Fwd':>12s}"
        f"{'Triton Fwd':>14s}"
        f"{'Fwd x':>9s}"
        f"{'Torch F+B':>12s}"
        f"{'Triton F+B':>14s}"
        f"{'F+B x':>9s}"
    )

    print("-" * 90)

    for r in results:

        fwd_speedup = (
            r["torch_fwd"]
            / r["triton_fwd"]
        )

        fb_speedup = (
            r["torch_fb"]
            / r["triton_fb"]
        )

        print(
            f"{r['name']:22s}"
            f"{r['torch_fwd']:12.4f}"
            f"{r['triton_fwd']:14.4f}"
            f"{fwd_speedup:9.2f}"
            f"{r['torch_fb']:12.4f}"
            f"{r['triton_fb']:14.4f}"
            f"{fb_speedup:9.2f}"
        )


if __name__ == "__main__":
    main()