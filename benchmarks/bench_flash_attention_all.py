import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from triton_kernels.attention import triton_flash_attention


class SDPA(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )


class TritonFlashAttention(nn.Module):
    def forward(self, q, k, v):
        return triton_flash_attention(q, k, v)


def bench_module(
    module,
    q,
    k,
    v,
    do,
):
    def fn():
        # 이전 iteration grad 제거
        q.grad = None
        k.grad = None
        v.grad = None

        out = module(q, k, v)

        out.backward(do)

    return triton.testing.do_bench(fn)


def main():
    torch.manual_seed(0)

    B = 8
    H = 12
    T = 1024
    D = 64

    dtype = torch.float16
    device = "cuda"

    q = torch.randn(
        B, H, T, D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    k = torch.randn(
        B, H, T, D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    v = torch.randn(
        B, H, T, D,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    do = torch.randn_like(q)

    sdpa = SDPA()
    flash = TritonFlashAttention()

    # --------------------------------------------------------
    # correctness 한 번 확인
    # --------------------------------------------------------

    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)

    out_sdpa = sdpa(q1, k1, v1)
    out_flash = flash(q2, k2, v2)

    out_sdpa.backward(do)
    out_flash.backward(do)

    print("Correctness")
    print(
        "O :",
        (out_sdpa - out_flash).abs().max().item(),
    )
    print(
        "dQ:",
        (q1.grad - q2.grad).abs().max().item(),
    )
    print(
        "dK:",
        (k1.grad - k2.grad).abs().max().item(),
    )
    print(
        "dV:",
        (v1.grad - v2.grad).abs().max().item(),
    )

    # --------------------------------------------------------
    # benchmark
    # --------------------------------------------------------

    torch.cuda.synchronize()

    sdpa_ms = bench_module(
        sdpa,
        q,
        k,
        v,
        do,
    )

    flash_ms = bench_module(
        flash,
        q,
        k,
        v,
        do,
    )

    print()
    print(
        f"B={B}, H={H}, T={T}, D={D}, "
        f"dtype={dtype}"
    )

    print(
        f"SDPA forward+backward  : "
        f"{sdpa_ms:.4f} ms"
    )

    print(
        f"Triton forward+backward: "
        f"{flash_ms:.4f} ms"
    )

    print(
        f"Speedup: "
        f"{sdpa_ms / flash_ms:.2f}x"
    )


if __name__ == "__main__":
    main()