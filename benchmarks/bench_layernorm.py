import torch
import torch.nn.functional as F
import triton

from triton_kernels.layernorm import layer_norm


def bench_once(B, T, C, dtype=torch.float32):
    x = torch.randn(B, T, C, device="cuda", dtype=dtype)
    weight = torch.randn(C, device="cuda", dtype=dtype)
    bias = torch.randn(C, device="cuda", dtype=dtype)

    def torch_fn():
        return F.layer_norm(
            x,
            (C,),
            weight,
            bias,
            eps=1e-5,
        )

    def triton_fn():
        return layer_norm(
            x,
            weight,
            bias,
            eps=1e-5,
        )

    torch_ms = triton.testing.do_bench(
        torch_fn,
        return_mode="median",
    )

    triton_ms = triton.testing.do_bench(
        triton_fn,
        return_mode="median",
    )

    print(f"B={B}, T={T}, C={C}, dtype={dtype}")
    print(f"PyTorch : {torch_ms:.4f} ms")
    print(f"Triton  : {triton_ms:.4f} ms")
    print(f"speedup : {torch_ms / triton_ms:.2f}x")
    
def bench_num_warps(
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

    print(f"B={B}, T={T}, C={C}, dtype={dtype}")

    for num_warps in [2, 4, 8]:

        def triton_fn():
            return layer_norm(
                x,
                weight,
                bias,
                eps=1e-5,
                num_warps=num_warps,
            )

        triton_ms = triton.testing.do_bench(
            triton_fn,
            return_mode="median",
        )

        print(
            f"num_warps={num_warps}: "
            f"{triton_ms:.4f} ms"
        )

if __name__ == "__main__":
    bench_once(
        B=8,
        T=1024,
        C=768,
        dtype=torch.float32,
    )