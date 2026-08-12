import torch
import torch.nn.functional as F
import triton

from triton_kernels.residual_layernorm import residual_layer_norm


def bench_residual_layernorm(
    B=8,
    T=1024,
    C=768,
    dtype=torch.float32,
):
    x = torch.randn(
        B, T, C,
        device="cuda",
        dtype=dtype,
    )

    residual = torch.randn(
        B, T, C,
        device="cuda",
        dtype=dtype,
    )

    weight = torch.randn(
        C,
        device="cuda",
        dtype=dtype,
    )

    bias = torch.randn(
        C,
        device="cuda",
        dtype=dtype,
    )

    def torch_fn():
        z = x + residual

        return F.layer_norm(
            z,
            (C,),
            weight,
            bias,
            eps=1e-5,
        )

    def triton_fn():
        return residual_layer_norm(
            x,
            residual,
            weight,
            bias,
            eps=1e-5,
            num_warps=8,
        )

    # 먼저 correctness 한 번 확인
    y_torch = torch_fn()
    y_triton = triton_fn()

    torch.testing.assert_close(
        y_triton,
        y_torch,
        rtol=1e-4,
        atol=1e-4,
    )

    # benchmark
    torch_ms = triton.testing.do_bench(
        torch_fn,
        return_mode="median",
    )

    triton_ms = triton.testing.do_bench(
        triton_fn,
        return_mode="median",
    )

    print(
        f"B={B}, T={T}, C={C}, dtype={dtype}"
    )

    print(f"PyTorch : {torch_ms:.4f} ms")
    print(f"Triton  : {triton_ms:.4f} ms")
    print(
        f"Speedup : {torch_ms / triton_ms:.2f}x"
    )


if __name__ == "__main__":
    bench_residual_layernorm()