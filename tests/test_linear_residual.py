import torch
import torch.nn.functional as F

from triton_kernels.linear_residual import (
    triton_linear_residual,
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
        f"{name:<20} "
        f"max={max_diff:.6e} "
        f"mean={mean_diff:.6e} "
        f"allclose={ok}"
    )

    return ok


def test_case(
    name,
    B,
    T,
    K,
    N,
):
    torch.manual_seed(0)

    print()
    print("=" * 90)
    print(
        f"{name}: "
        f"B={B}, T={T}, "
        f"K={K}, N={N}"
    )
    print("=" * 90)

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
    # PyTorch tensors
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

    # ========================================================
    # Triton tensors
    # ========================================================

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
    # PyTorch
    #
    # Y = Linear(X) + Residual
    # ========================================================

    y_ref = (
        F.linear(
            x_ref,
            w_ref,
            b_ref,
        )
        + r_ref
    )

    y_ref.backward(
        dy
    )

    # ========================================================
    # Triton
    # ========================================================

    y_tri = triton_linear_residual(
        x_tri,
        w_tri,
        b_tri,
        r_tri,
    )

    y_tri.backward(
        dy
    )

    torch.cuda.synchronize()

    # ========================================================
    # Forward
    # ========================================================

    print()
    print("[Forward]")

    forward_ok = compare(
        "output",
        y_ref,
        y_tri,
        atol=1e-2,
        rtol=1e-2,
    )

    # ========================================================
    # Backward
    # ========================================================

    print()
    print("[Backward]")

    dx_ok = compare(
        "dX",
        x_ref.grad,
        x_tri.grad,
        atol=1e-1,
        rtol=1e-2,
    )

    dw_ok = compare(
        "dW",
        w_ref.grad,
        w_tri.grad,
        atol=1e-1,
        rtol=1e-2,
    )

    db_ok = compare(
        "db",
        b_ref.grad,
        b_tri.grad,
        atol=1e-1,
        rtol=1e-2,
    )

    dr_ok = compare(
        "dResidual",
        r_ref.grad,
        r_tri.grad,
        atol=0.0,
        rtol=0.0,
    )

    # ========================================================
    # Summary
    # ========================================================

    all_ok = (
        forward_ok
        and dx_ok
        and dw_ok
        and db_ok
        and dr_ok
    )

    print()
    print(
        "Result: "
        + (
            "PASS"
            if all_ok
            else "FAIL"
        )
    )

    return all_ok


def main():
    torch.cuda.init()

    print()
    print("=" * 90)
    print("Triton Linear + Residual Correctness")
    print("=" * 90)

    # ========================================================
    # Attention c_proj
    #
    # [B,T,768] -> [B,T,768]
    # ========================================================

    attn_ok = test_case(
        name="Attention c_proj + residual",
        B=2,
        T=128,
        K=768,
        N=768,
    )

    # ========================================================
    # MLP c_proj
    #
    # [B,T,3072] -> [B,T,768]
    # ========================================================

    mlp_ok = test_case(
        name="MLP c_proj + residual",
        B=2,
        T=128,
        K=3072,
        N=768,
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Attention c_proj + residual : "
        f"{'PASS' if attn_ok else 'FAIL'}"
    )

    print(
        f"MLP c_proj + residual       : "
        f"{'PASS' if mlp_ok else 'FAIL'}"
    )

    print()

    print(
        "Overall: "
        + (
            "PASS"
            if attn_ok and mlp_ok
            else "FAIL"
        )
    )


if __name__ == "__main__":
    main()