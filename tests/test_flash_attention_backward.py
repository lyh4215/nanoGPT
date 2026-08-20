import torch
import torch.nn.functional as F

from triton_kernels.attention import (
    triton_flash_attention_forward,
    triton_flash_attention_backward,
)


def check_close(name, actual, expected, atol=1e-2, rtol=1e-2):
    diff = (actual - expected).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ok = torch.allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"{name:>3} | "
        f"{'PASS' if ok else 'FAIL'} | "
        f"max diff={max_diff:.6e} | "
        f"mean diff={mean_diff:.6e}"
    )

    return ok


def main():
    torch.manual_seed(0)

    # 처음에는 작은 shape로 correctness 확인
    B = 2
    H = 2
    T = 128
    D = 64

    dtype = torch.float16
    device = "cuda"

    print(
        f"B={B}, H={H}, T={T}, D={D}, "
        f"dtype={dtype}"
    )

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    q = torch.randn(
        B, H, T, D,
        device=device,
        dtype=dtype,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # backward에서 들어오는 upstream gradient
    do = torch.randn_like(q)

    # ------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        is_causal=True,
    )

    # do = dL/dO 라고 생각하면 됨.
    out_ref.backward(do)

    dq_ref = q_ref.grad
    dk_ref = k_ref.grad
    dv_ref = v_ref.grad

    # ------------------------------------------------------------
    # Triton forward
    #
    # forward가 이제 out, lse를 반환한다고 가정
    # ------------------------------------------------------------

    out, lse = triton_flash_attention_forward(
        q,
        k,
        v,
    )

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Triton backward
    # ------------------------------------------------------------

    dq, dk, dv = triton_flash_attention_backward(
        q,
        k,
        v,
        out,
        do,
        lse,
    )

    torch.cuda.synchronize()

    # ------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------

    print("\n[Forward]")

    out_ok = check_close(
        "O",
        out,
        out_ref.detach(),
        atol=5e-3,
        rtol=5e-3,
    )

    print("\n[Backward]")

    dq_ok = check_close(
        "dQ",
        dq,
        dq_ref,
        atol=1e-2,
        rtol=1e-2,
    )

    dk_ok = check_close(
        "dK",
        dk,
        dk_ref,
        atol=1e-2,
        rtol=1e-2,
    )

    dv_ok = check_close(
        "dV",
        dv,
        dv_ref,
        atol=1e-2,
        rtol=1e-2,
    )

    print()

    if out_ok and dq_ok and dk_ok and dv_ok:
        print("ALL PASS ✅")
    else:
        print("FAILED ❌")


if __name__ == "__main__":
    main()