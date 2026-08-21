import torch
import triton

from model import (
    CausalSelfAttention,
    GPTConfig,
)

from triton_model import (
    TritonCausalSelfAttention,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def main():
    torch.manual_seed(0)

    # CUDA context 미리 생성
    torch.cuda.init()

    # ============================================================
    # Config
    # ============================================================

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

    # ============================================================
    # Modules
    # ============================================================

    ref = CausalSelfAttention(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonCausalSelfAttention(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    # 동일한 weight 사용
    tri.load_state_dict(
        ref.state_dict()
    )

    ref.train()
    tri.train()

    # ============================================================
    # Input
    # ============================================================

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

    # ============================================================
    # Forward closures
    # ============================================================

    def torch_forward():
        return ref(
            x_ref
        )

    def triton_forward():
        return tri(
            x_tri
        )

    # ============================================================
    # Forward + Backward closures
    # ============================================================

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

    # ============================================================
    # Warmup / Triton compile
    # ============================================================

    for _ in range(3):
        torch_forward()
        triton_forward()

    torch.cuda.synchronize()

    torch_fb()
    triton_fb()

    torch.cuda.synchronize()

    # grad cleanup
    x_ref.grad = None
    x_tri.grad = None

    ref.zero_grad(
        set_to_none=True
    )

    tri.zero_grad(
        set_to_none=True
    )

    # ============================================================
    # Forward benchmark
    # ============================================================

    torch_fwd_ms = triton.testing.do_bench(
        torch_forward,
    )

    triton_fwd_ms = triton.testing.do_bench(
        triton_forward,
    )

    # ============================================================
    # Forward + Backward benchmark
    # ============================================================

    torch_fb_ms = triton.testing.do_bench(
        torch_fb,
    )

    triton_fb_ms = triton.testing.do_bench(
        triton_fb,
    )

    # ============================================================
    # Results
    # ============================================================

    print()
    print("=" * 80)
    print(
        f"Attention Module Benchmark "
        f"B={B}, T={T}, C={C}"
    )
    print("=" * 80)

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


if __name__ == "__main__":
    main()