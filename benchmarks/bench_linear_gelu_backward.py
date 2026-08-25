import torch
import triton

from triton_kernels.linear_gelu import (
    triton_gelu_backward,
)

from triton_kernels.linear import (
    triton_linear_backward_dx,
    triton_linear_backward_dw,
    triton_linear_backward_db,
)


DEVICE = "cuda"
DTYPE = torch.float16

# GPT-2 MLP c_fc shape
B = 8
T = 1024

M = B * T
K = 768
N = 3072


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Inputs
    #
    # Linear:
    #   X [M, K]
    #   W [N, K]
    #   Z [M, N]
    #
    # GELU backward:
    #   dY [M, N]
    #   Z  [M, N]
    #   -> dZ [M, N]
    # ========================================================

    x = torch.randn(
        M,
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

    z = torch.randn(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    dy = torch.randn(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    # Linear backward 단독 benchmark에서 사용할 dZ
    dz = triton_gelu_backward(
        dy,
        z,
    )

    torch.cuda.synchronize()

    # ========================================================
    # Individual backward components
    # ========================================================

    def gelu_backward():
        return triton_gelu_backward(
            dy,
            z,
        )

    def linear_dx():
        return triton_linear_backward_dx(
            dz,
            weight,
        )

    def linear_dw():
        return triton_linear_backward_dw(
            dz,
            x,
        )

    def linear_db():
        return triton_linear_backward_db(
            dz,
        )

    # ========================================================
    # Current full backward path
    #
    # dy
    #  ↓
    # GELU backward
    #  ↓
    # dz
    #  ├─ dX
    #  ├─ dW
    #  └─ db
    # ========================================================

    def full_backward():
        dz_local = triton_gelu_backward(
            dy,
            z,
        )

        dx = triton_linear_backward_dx(
            dz_local,
            weight,
        )

        dw = triton_linear_backward_dw(
            dz_local,
            x,
        )

        db = triton_linear_backward_db(
            dz_local,
        )

        return dx, dw, db

    # ========================================================
    # Warmup
    # ========================================================

    for _ in range(3):
        gelu_backward()
        linear_dx()
        linear_dw()
        linear_db()
        full_backward()

    torch.cuda.synchronize()

    # ========================================================
    # Benchmark
    # ========================================================

    gelu_ms = triton.testing.do_bench(
        gelu_backward
    )

    dx_ms = triton.testing.do_bench(
        linear_dx
    )

    dw_ms = triton.testing.do_bench(
        linear_dw
    )

    db_ms = triton.testing.do_bench(
        linear_db
    )

    full_ms = triton.testing.do_bench(
        full_backward
    )

    component_sum = (
        gelu_ms
        + dx_ms
        + dw_ms
        + db_ms
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"Linear + GELU Backward Breakdown "
        f"M={M}, K={K}, N={N}"
    )
    print("=" * 80)

    print()

    print(
        f"GELU backward : {gelu_ms:.4f} ms"
    )

    print(
        f"Linear dX     : {dx_ms:.4f} ms"
    )

    print(
        f"Linear dW     : {dw_ms:.4f} ms"
    )

    print(
        f"Linear db     : {db_ms:.4f} ms"
    )

    print("-" * 80)

    print(
        f"Component sum : {component_sum:.4f} ms"
    )

    print(
        f"Full backward : {full_ms:.4f} ms"
    )

    print()

    print("[Share of component sum]")

    print(
        f"GELU backward : "
        f"{gelu_ms / component_sum * 100:.1f}%"
    )

    print(
        f"Linear dX     : "
        f"{dx_ms / component_sum * 100:.1f}%"
    )

    print(
        f"Linear dW     : "
        f"{dw_ms / component_sum * 100:.1f}%"
    )

    print(
        f"Linear db     : "
        f"{db_ms / component_sum * 100:.1f}%"
    )


if __name__ == "__main__":
    main()