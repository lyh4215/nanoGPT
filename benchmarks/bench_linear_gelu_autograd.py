import torch
import torch.nn.functional as F
import triton

from triton_kernels.linear_gelu import (
    triton_linear_gelu,
)


DEVICE = "cuda"
DTYPE = torch.float16

# GPT-2 MLP c_fc 실제 shape
B = 8
T = 1024
K = 768
N = 3072


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Shared initial values
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

    dy = torch.randn(
        B,
        T,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # Separate tensors
    #
    # PyTorch / Triton이 grad를 서로 공유하지 않도록 분리
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

    # ========================================================
    # Forward functions
    # ========================================================

    def torch_forward():
        z = F.linear(
            x_ref,
            w_ref,
            b_ref,
        )

        return F.gelu(z)

    def triton_forward():
        return triton_linear_gelu(
            x_tri,
            w_tri,
            b_tri,
        )

    # ========================================================
    # Forward + Backward
    # ========================================================

    def torch_fb():
        x_ref.grad = None
        w_ref.grad = None
        b_ref.grad = None

        y = F.gelu(
            F.linear(
                x_ref,
                w_ref,
                b_ref,
            )
        )

        y.backward(dy)

    def triton_fb():
        x_tri.grad = None
        w_tri.grad = None
        b_tri.grad = None

        y = triton_linear_gelu(
            x_tri,
            w_tri,
            b_tri,
        )

        y.backward(dy)

    # ========================================================
    # Warmup
    # ========================================================

    for _ in range(3):
        torch_forward()
        triton_forward()

    torch_fb()
    triton_fb()

    torch.cuda.synchronize()

    # Clear leftover grads
    x_ref.grad = None
    w_ref.grad = None
    b_ref.grad = None

    x_tri.grad = None
    w_tri.grad = None
    b_tri.grad = None

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

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"Linear + GELU Autograd Benchmark "
        f"B={B}, T={T}, K={K}, N={N}"
    )
    print("=" * 80)

    print()
    print("[Forward]")

    print(
        f"PyTorch : {torch_fwd_ms:.4f} ms"
    )

    print(
        f"Triton  : {triton_fwd_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_fwd_ms / triton_fwd_ms:.2f}x"
    )

    print()
    print("[Forward + Backward]")

    print(
        f"PyTorch : {torch_fb_ms:.4f} ms"
    )

    print(
        f"Triton  : {triton_fb_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_fb_ms / triton_fb_ms:.2f}x"
    )

    print()

    torch_backward_ms = (
        torch_fb_ms
        - torch_fwd_ms
    )

    triton_backward_ms = (
        triton_fb_ms
        - triton_fwd_ms
    )

    print("[Approx. Backward]")

    print(
        f"PyTorch : {torch_backward_ms:.4f} ms"
    )

    print(
        f"Triton  : {triton_backward_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_backward_ms / triton_backward_ms:.2f}x"
    )


if __name__ == "__main__":
    main()