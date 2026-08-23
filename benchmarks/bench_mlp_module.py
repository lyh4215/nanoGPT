import torch
import triton

from model import MLP, GPTConfig
from triton_model import TritonMLP


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    C = config.n_embd

    # ========================================================
    # Modules
    # ========================================================

    ref = MLP(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonMLP(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri.load_state_dict(
        ref.state_dict()
    )

    ref.train()
    tri.train()

    # ========================================================
    # Input
    # ========================================================

    x_ref = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    x_tri = (
        x_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    dy = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # Forward
    # ========================================================

    def torch_forward():
        return ref(x_ref)

    def triton_forward():
        return tri(x_tri)

    # ========================================================
    # Forward + Backward
    # ========================================================

    def torch_fb():
        x_ref.grad = None
        ref.zero_grad(set_to_none=True)

        y = ref(x_ref)
        y.backward(dy)

    def triton_fb():
        x_tri.grad = None
        tri.zero_grad(set_to_none=True)

        y = tri(x_tri)
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

    x_ref.grad = None
    x_tri.grad = None

    ref.zero_grad(set_to_none=True)
    tri.zero_grad(set_to_none=True)

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
        f"MLP Module Benchmark "
        f"B={B}, T={T}, C={C}"
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


if __name__ == "__main__":
    main()