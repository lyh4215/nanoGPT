import torch
import triton

from model import (
    CausalSelfAttention as TorchAttention,
    GPTConfig,
)

from triton_model import (
    TritonCausalSelfAttention as TritonAttention,
)


DEVICE = "cuda"
DTYPE = torch.float16


def print_diff(
    name,
    a,
    b,
):
    diff = (
        a.float()
        - b.float()
    ).abs()

    print(
        f"{name:>12} | "
        f"max={diff.max().item():.6e} | "
        f"mean={diff.mean().item():.6e}"
    )


def main():

    torch.manual_seed(0)

    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,

        n_layer=12,
        n_head=12,
        n_embd=768,

        dropout=0.0,
        bias=True,
    )

    # ========================================================
    # Modules
    # ========================================================

    ref = TorchAttention(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonAttention(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    # 정확히 같은 parameter
    tri.load_state_dict(
        ref.state_dict()
    )

    ref.train()
    tri.train()

    # ========================================================
    # Input
    # ========================================================

    B = 2
    T = 128
    C = config.n_embd

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

    # ========================================================
    # Reference
    # ========================================================

    x_ref = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    ref.zero_grad(
        set_to_none=True
    )

    y_ref = ref(
        x_ref
    )

    y_ref.backward(
        dy
    )

    # ========================================================
    # Triton
    # ========================================================

    x_tri = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    tri.zero_grad(
        set_to_none=True
    )

    y_tri = tri(
        x_tri
    )

    y_tri.backward(
        dy
    )

    # ========================================================
    # Correctness
    # ========================================================

    print()
    print("=" * 80)
    print("Attention module correctness")
    print("=" * 80)

    print_diff(
        "output",
        y_ref,
        y_tri,
    )

    print_diff(
        "dx",
        x_ref.grad,
        x_tri.grad,
    )

    # ========================================================
    # Parameter gradients
    # ========================================================

    ref_params = dict(
        ref.named_parameters()
    )

    tri_params = dict(
        tri.named_parameters()
    )

    print()
    print("[Parameter gradients]")

    for name in ref_params:

        grad_ref = (
            ref_params[name].grad
        )

        grad_tri = (
            tri_params[name].grad
        )

        print_diff(
            name,
            grad_ref,
            grad_tri,
        )


if __name__ == "__main__":
    main()