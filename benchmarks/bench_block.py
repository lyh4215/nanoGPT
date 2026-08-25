import torch
import triton

from model import (
    Block,
    GPTConfig,
)

from triton_model import (
    TritonBlock,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Config
    # ========================================================

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

    ref = Block(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonBlock(
        config
    ).to(
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

    x = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    dy = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    x_ref = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    x_tri = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    # ========================================================
    # Forward
    # ========================================================

    def torch_forward():
        return ref(
            x_ref
        )

    def triton_forward():
        return tri(
            x_tri
        )

    # ========================================================
    # Forward + Backward
    # ========================================================

    def torch_fb():
        x_ref.grad = None

        ref.zero_grad(
            set_to_none=True
        )

        y = ref(
            x_ref
        )

        y.backward(
            dy
        )

    def triton_fb():
        x_tri.grad = None

        tri.zero_grad(
            set_to_none=True
        )

        y = tri(
            x_tri
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

    x_ref.grad = None
    x_tri.grad = None

    ref.zero_grad(
        set_to_none=True
    )

    tri.zero_grad(
        set_to_none=True
    )

    # ========================================================
    # Benchmark
    # ========================================================

    torch_fwd_ms = (
        triton.testing.do_bench(
            torch_forward
        )
    )

    triton_fwd_ms = (
        triton.testing.do_bench(
            triton_forward
        )
    )

    torch_fb_ms = (
        triton.testing.do_bench(
            torch_fb
        )
    )

    triton_fb_ms = (
        triton.testing.do_bench(
            triton_fb
        )
    )

    # ========================================================
    # Approx backward
    # ========================================================

    torch_bwd_ms = (
        torch_fb_ms
        - torch_fwd_ms
    )

    triton_bwd_ms = (
        triton_fb_ms
        - triton_fwd_ms
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 90)

    print(
        f"Transformer Block Benchmark "
        f"B={B}, T={T}, C={C}"
    )

    print("=" * 90)

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

    print()
    print("[Approx. Backward]")

    print(
        f"PyTorch : "
        f"{torch_bwd_ms:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{triton_bwd_ms:.4f} ms"
    )

    print(
        f"Speedup : "
        f"{torch_bwd_ms / triton_bwd_ms:.2f}x"
    )


if __name__ == "__main__":
    main()