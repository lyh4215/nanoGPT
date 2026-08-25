import torch

from model import (
    Block,
    GPTConfig,
)

from triton_model import (
    TritonBlock,
)


DEVICE = "cuda"
DTYPE = torch.float16


def compare(
    name,
    ref,
    tri,
    atol,
    rtol,
):
    diff = (
        ref.float()
        - tri.float()
    ).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ok = torch.allclose(
        ref,
        tri,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"{name:<32} "
        f"max={max_diff:.6e} "
        f"mean={mean_diff:.6e} "
        f"allclose={ok}"
    )

    return ok


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

    # 원본 Block과 동일한 parameter 사용
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
    # PyTorch
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
    # Forward
    # ========================================================

    print()
    print("=" * 110)
    print(
        f"Triton Block Correctness "
        f"B={B}, T={T}, C={C}"
    )
    print("=" * 110)

    print()
    print("[Forward]")

    forward_ok = compare(
        "output",
        y_ref,
        y_tri,
        atol=2e-2,
        rtol=1e-2,
    )

    # ========================================================
    # Input gradient
    # ========================================================

    print()
    print("[Input Gradient]")

    dx_ok = compare(
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

    # state_dict 구조 자체도 동일한지 확인
    assert (
        ref_params.keys()
        == tri_params.keys()
    ), (
        "Parameter names do not match\n"
        f"ref={list(ref_params.keys())}\n"
        f"tri={list(tri_params.keys())}"
    )

    for name in ref_params:
        ref_grad = ref_params[name].grad
        tri_grad = tri_params[name].grad

        if ref_grad is None:
            print(
                f"{name:<32} "
                "REF GRAD NONE"
            )

            param_ok = False
            continue

        if tri_grad is None:
            print(
                f"{name:<32} "
                "TRITON GRAD NONE"
            )

            param_ok = False
            continue

        ok = compare(
            name,
            ref_grad,
            tri_grad,

            # Block에서는 여러 FP16 reduction 차이가
            # 누적될 수 있으므로 우선 진단용으로 이 정도.
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
    print("=" * 110)
    print("Summary")
    print("=" * 110)

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

    print(
        "Overall: "
        + (
            "PASS"
            if all_ok
            else "FAIL"
        )
    )


if __name__ == "__main__":
    main()