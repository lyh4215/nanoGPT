import torch

from model import (
    MLP,
    GPTConfig,
)

from triton_model import (
    TritonMLP,
)


DEVICE = "cuda"
DTYPE = torch.float16


def print_diff(
    name,
    ref,
    out,
    atol,
    rtol,
):
    diff = (
        ref.float()
        - out.float()
    ).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    allclose = torch.allclose(
        ref,
        out,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"{name:<24} "
        f"max={max_diff:.6e} "
        f"mean={mean_diff:.6e} "
        f"allclose={allclose}"
    )

    return allclose


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

    B = 2
    T = 128
    C = config.n_embd

    # ========================================================
    # Modules
    # ========================================================

    ref = MLP(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonMLP(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    # 동일한 parameter 사용
    tri.load_state_dict(
        ref.state_dict()
    )

    ref.train()
    tri.train()

    # ========================================================
    # Input / upstream gradient
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
    # PyTorch reference
    # ========================================================

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

    tri.zero_grad(
        set_to_none=True
    )

    y_tri = tri(
        x_tri
    )

    y_tri.backward(
        dy
    )

    torch.cuda.synchronize()

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 90)
    print(
        "Triton MLP Correctness "
        f"B={B}, T={T}, C={C}"
    )
    print("=" * 90)

    print()
    print("[Forward]")

    forward_ok = print_diff(
        "output",
        y_ref,
        y_tri,
        atol=1e-2,
        rtol=1e-2,
    )

    print()
    print("[Input Gradient]")

    dx_ok = print_diff(
        "dX",
        x_ref.grad,
        x_tri.grad,
        atol=1e-1,
        rtol=1e-2,
    )

    # ========================================================
    # Parameter gradients
    # ========================================================

    print()
    print("[Parameter Gradients]")

    ref_params = dict(
        ref.named_parameters()
    )

    tri_params = dict(
        tri.named_parameters()
    )

    param_ok = True

    for name in ref_params:
        ref_grad = (
            ref_params[name].grad
        )

        tri_grad = (
            tri_params[name].grad
        )

        if ref_grad is None:
            print(
                f"{name:<24} "
                f"reference grad is None"
            )

            param_ok = False
            continue

        if tri_grad is None:
            print(
                f"{name:<24} "
                f"triton grad is None"
            )

            param_ok = False
            continue

        ok = print_diff(
            name,
            ref_grad,
            tri_grad,

            # FP16 GEMM reduction 순서 차이 때문에
            # parameter grad는 조금 더 여유 있게 본다.
            atol=1e-1,
            rtol=1e-2,
        )

        param_ok = (
            param_ok
            and ok
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Forward        : "
        f"{'PASS' if forward_ok else 'FAIL'}"
    )

    print(
        f"Input gradient : "
        f"{'PASS' if dx_ok else 'FAIL'}"
    )

    print(
        f"Parameter grads: "
        f"{'PASS' if param_ok else 'FAIL'}"
    )

    all_ok = (
        forward_ok
        and dx_ok
        and param_ok
    )

    print()

    if all_ok:
        print(
            "Overall: PASS"
        )
    else:
        print(
            "Overall: FAIL"
        )


if __name__ == "__main__":
    main()