import torch
import triton

from model import (
    GPTConfig,
    Block,
)

from triton_model import (
    TritonBlock,
)

from triton_kernels.linear_gelu import (
    triton_linear_gelu,
)

from triton_kernels.linear_residual import (
    triton_linear_residual,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024
C = 768


def bench(fn):
    for _ in range(3):
        fn()

    torch.cuda.synchronize()

    return triton.testing.do_bench(fn)


def print_row(
    name,
    torch_ms,
    triton_ms,
):
    print(
        f"{name:<32} "
        f"{torch_ms:>10.4f} "
        f"{triton_ms:>10.4f} "
        f"{torch_ms / triton_ms:>9.2f}x"
    )


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # Config
    # ========================================================

    config = GPTConfig(
        block_size=T,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=C,
        dropout=0.0,
        bias=True,
    )

    # ========================================================
    # Blocks
    #
    # MLP parameter 구조가 동일하므로 Block 하나만 생성
    # ========================================================

    ref_block = Block(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri_block = TritonBlock(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri_block.load_state_dict(
        ref_block.state_dict()
    )

    ref_block.eval()
    tri_block.eval()

    ref_mlp = ref_block.mlp
    tri_mlp = tri_block.mlp

    # ========================================================
    # Input
    #
    # 실제 MLP가 받는 LN 출력과 동일한 shape
    #
    # [B, T, 768]
    # ========================================================

    x = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    residual = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    # 동일한 입력 사용
    ref_x = x.clone()
    tri_x = x.clone()

    ref_residual = residual.clone()
    tri_residual = residual.clone()

    # ========================================================
    # Prepare hidden
    #
    # stage 2를 측정할 때 stage 1 비용이 섞이지 않도록
    # 미리 c_fc + GELU 결과 계산
    # ========================================================

    with torch.no_grad():

        ref_hidden = ref_mlp.gelu(
            ref_mlp.c_fc(
                ref_x
            )
        )

        tri_hidden = triton_linear_gelu(
            tri_x,
            tri_mlp.c_fc.weight,
            tri_mlp.c_fc.bias,
        )

    torch.cuda.synchronize()

    # ========================================================
    # ① c_fc + GELU
    #
    # PyTorch:
    # Linear(768 -> 3072)
    # +
    # exact GELU
    #
    # Triton:
    # fused Linear + exact GELU
    # ========================================================

    def ref_fc_gelu():
        x = ref_mlp.c_fc(
            ref_x
        )

        x = ref_mlp.gelu(
            x
        )

        return x

    def tri_fc_gelu():
        return triton_linear_gelu(
            tri_x,
            tri_mlp.c_fc.weight,
            tri_mlp.c_fc.bias,
        )

    # ========================================================
    # ② c_proj + Residual
    #
    # PyTorch:
    #
    # Linear(3072 -> 768)
    # → residual add
    #
    # Triton:
    #
    # Linear + residual fused epilogue
    # ========================================================

    def ref_proj_residual():
        x = ref_mlp.c_proj(
            ref_hidden
        )

        return (
            x
            + ref_residual
        )

    def tri_proj_residual():
        return triton_linear_residual(
            tri_hidden,
            tri_mlp.c_proj.weight,
            tri_mlp.c_proj.bias,
            tri_residual,
        )

    # ========================================================
    # Full MLP + Residual
    # ========================================================

    def ref_full():
        x = ref_mlp(
            ref_x
        )

        return (
            ref_residual
            + x
        )

    def tri_full():
        return tri_mlp(
            tri_x,
            residual=tri_residual,
        )

    # ========================================================
    # Optional finer PyTorch breakdown
    #
    # c_fc 자체와 GELU 자체도 따로 측정해서
    # ①이 느리다면 GEMM인지 GELU인지 확인.
    # ========================================================

    with torch.no_grad():
        ref_fc_output = ref_mlp.c_fc(
            ref_x
        )

    torch.cuda.synchronize()

    def ref_fc_only():
        return ref_mlp.c_fc(
            ref_x
        )

    def ref_gelu_only():
        return ref_mlp.gelu(
            ref_fc_output
        )

    # ========================================================
    # Benchmark
    # ========================================================

    ref_fc_gelu_ms = bench(
        ref_fc_gelu
    )

    tri_fc_gelu_ms = bench(
        tri_fc_gelu
    )

    ref_proj_residual_ms = bench(
        ref_proj_residual
    )

    tri_proj_residual_ms = bench(
        tri_proj_residual
    )

    ref_full_ms = bench(
        ref_full
    )

    tri_full_ms = bench(
        tri_full
    )

    ref_fc_only_ms = bench(
        ref_fc_only
    )

    ref_gelu_only_ms = bench(
        ref_gelu_only
    )

    # ========================================================
    # Main breakdown
    # ========================================================

    print()
    print("=" * 92)
    print(
        f"MLP Forward Breakdown "
        f"B={B}, T={T}, C={C}"
    )
    print("=" * 92)

    print()

    print(
        f"{'Component':<32} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10} "
        f"{'Speedup':>9}"
    )

    print("-" * 66)

    print_row(
        "c_fc + GELU",
        ref_fc_gelu_ms,
        tri_fc_gelu_ms,
    )

    print_row(
        "c_proj + Residual",
        ref_proj_residual_ms,
        tri_proj_residual_ms,
    )

    print("-" * 66)

    print_row(
        "Full MLP + Residual",
        ref_full_ms,
        tri_full_ms,
    )

    # ========================================================
    # Component sum
    # ========================================================

    ref_sum = (
        ref_fc_gelu_ms
        + ref_proj_residual_ms
    )

    tri_sum = (
        tri_fc_gelu_ms
        + tri_proj_residual_ms
    )

    print()
    print("=" * 92)
    print("Component Sum vs Full")
    print("=" * 92)

    print()

    print(
        f"PyTorch component sum : "
        f"{ref_sum:.4f} ms"
    )

    print(
        f"PyTorch full          : "
        f"{ref_full_ms:.4f} ms"
    )

    print()

    print(
        f"Triton component sum  : "
        f"{tri_sum:.4f} ms"
    )

    print(
        f"Triton full           : "
        f"{tri_full_ms:.4f} ms"
    )

    # ========================================================
    # PyTorch c_fc / GELU detail
    # ========================================================

    print()
    print("=" * 92)
    print("PyTorch c_fc + GELU Detail")
    print("=" * 92)

    print()

    print(
        f"c_fc only     : "
        f"{ref_fc_only_ms:.4f} ms"
    )

    print(
        f"GELU only     : "
        f"{ref_gelu_only_ms:.4f} ms"
    )

    print(
        f"Separate sum  : "
        f"{ref_fc_only_ms + ref_gelu_only_ms:.4f} ms"
    )

    print(
        f"Combined bench: "
        f"{ref_fc_gelu_ms:.4f} ms"
    )


if __name__ == "__main__":
    main()