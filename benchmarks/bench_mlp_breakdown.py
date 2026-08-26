import torch
import triton

from model import (
    GPTConfig,
    Block,
)

from triton_model import (
    TritonBlock,
)

from triton_kernels.linear import (
    triton_linear,
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


# ============================================================
# Benchmark helper
#
# Forward 성능만 보고 있으므로 autograd / graph 생성 완전히 제거
# ============================================================

def bench(fn):
    def run():
        with torch.inference_mode():
            return fn()

    # warmup
    for _ in range(10):
        run()

    torch.cuda.synchronize()

    return triton.testing.do_bench(
        run
    )


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
    # 같은 weight를 사용하도록 state_dict 복사
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
    # 실제 MLP input shape:
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

    ref_x = x.clone()
    tri_x = x.clone()

    ref_residual = residual.clone()
    tri_residual = residual.clone()

    # ========================================================
    # Prepare intermediate values
    #
    # 각 component benchmark에서 이전 stage 비용이
    # 섞이지 않도록 미리 결과를 생성.
    # ========================================================

    with torch.inference_mode():

        # ----------------------------------------------------
        # c_fc outputs
        #
        # [B,T,768] -> [B,T,3072]
        # ----------------------------------------------------

        ref_fc_output = ref_mlp.c_fc(
            ref_x
        )

        tri_fc_output = triton_linear(
            tri_x,
            tri_mlp.c_fc.weight,
            tri_mlp.c_fc.bias,
        )

        # ----------------------------------------------------
        # c_fc + GELU outputs
        #
        # c_proj의 입력으로 사용
        # ----------------------------------------------------

        ref_hidden = ref_mlp.gelu(
            ref_fc_output
        )

        tri_hidden = triton_linear_gelu(
            tri_x,
            tri_mlp.c_fc.weight,
            tri_mlp.c_fc.bias,
        )

        # ----------------------------------------------------
        # c_proj outputs
        #
        # [B,T,3072] -> [B,T,768]
        # ----------------------------------------------------

        ref_proj_output = ref_mlp.c_proj(
            ref_hidden
        )

        tri_proj_output = triton_linear(
            tri_hidden,
            tri_mlp.c_proj.weight,
            tri_mlp.c_proj.bias,
        )

    torch.cuda.synchronize()

    # ========================================================
    #
    # 1. c_fc + GELU
    #
    # ========================================================

    def ref_fc_gelu():
        y = ref_mlp.c_fc(
            ref_x
        )

        y = ref_mlp.gelu(
            y
        )

        return y

    def tri_fc_gelu():
        return triton_linear_gelu(
            tri_x,
            tri_mlp.c_fc.weight,
            tri_mlp.c_fc.bias,
        )

    # ========================================================
    #
    # 2. c_proj + Residual
    #
    # PyTorch:
    #
    #   c_proj
    #       ↓
    #   temporary output
    #       ↓
    #   residual add
    #
    # Triton:
    #
    #   GEMM accumulator
    #       ↓
    #   residual add
    #       ↓
    #   one final store
    #
    # ========================================================

    def ref_proj_residual():
        y = ref_mlp.c_proj(
            ref_hidden
        )

        return (
            y
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
    #
    # 3. Full MLP + Residual
    #
    # ========================================================

    def ref_full():
        y = ref_mlp(
            ref_x
        )

        return (
            y
            + ref_residual
        )

    def tri_full():
        return tri_mlp(
            tri_x,
            residual=tri_residual,
        )

    # ========================================================
    #
    # c_fc detail
    #
    # ========================================================

    def ref_fc_only():
        return ref_mlp.c_fc(
            ref_x
        )

    def tri_fc_only():
        return triton_linear(
            tri_x,
            tri_mlp.c_fc.weight,
            tri_mlp.c_fc.bias,
        )

    def ref_gelu_only():
        return ref_mlp.gelu(
            ref_fc_output
        )

    # ========================================================
    #
    # c_proj detail
    #
    # ========================================================

    def ref_proj_only():
        return ref_mlp.c_proj(
            ref_hidden
        )

    def tri_proj_only():
        return triton_linear(
            tri_hidden,
            tri_mlp.c_proj.weight,
            tri_mlp.c_proj.bias,
        )

    # ========================================================
    #
    # Residual add only
    #
    # 둘 다 torch add를 사용.
    #
    # 입력 tensor만 각각 자기 implementation에서 만들어진
    # projection output을 사용.
    #
    # ========================================================

    def ref_residual_only():
        return (
            ref_proj_output
            + ref_residual
        )

    def tri_residual_only():
        return (
            tri_proj_output
            + tri_residual
        )

    # ========================================================
    #
    # Triton unfused c_proj + residual
    #
    # 기존 구조:
    #
    # triton_linear
    #       ↓
    # temporary HBM store
    #       ↓
    # torch residual add
    #
    # ========================================================

    def tri_proj_residual_unfused():
        y = triton_linear(
            tri_hidden,
            tri_mlp.c_proj.weight,
            tri_mlp.c_proj.bias,
        )

        return (
            y
            + tri_residual
        )

    # ========================================================
    # Benchmarks
    # ========================================================

    # --------------------------------------------------------
    # Main
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # c_fc detail
    # --------------------------------------------------------

    ref_fc_only_ms = bench(
        ref_fc_only
    )

    tri_fc_only_ms = bench(
        tri_fc_only
    )

    ref_gelu_only_ms = bench(
        ref_gelu_only
    )

    # --------------------------------------------------------
    # c_proj detail
    # --------------------------------------------------------

    ref_proj_only_ms = bench(
        ref_proj_only
    )

    tri_proj_only_ms = bench(
        tri_proj_only
    )

    ref_residual_only_ms = bench(
        ref_residual_only
    )

    tri_residual_only_ms = bench(
        tri_residual_only
    )

    tri_proj_residual_unfused_ms = bench(
        tri_proj_residual_unfused
    )

    # ========================================================
    #
    # Main breakdown
    #
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
    #
    # Component sum
    #
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
    #
    # c_fc + GELU detail
    #
    # ========================================================

    print()
    print("=" * 92)
    print("c_fc + GELU Detail")
    print("=" * 92)

    print()

    print(
        f"{'Operation':<32} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10} "
        f"{'Speedup':>9}"
    )

    print("-" * 66)

    print_row(
        "c_fc only",
        ref_fc_only_ms,
        tri_fc_only_ms,
    )

    print()

    print(
        f"PyTorch GELU only     : "
        f"{ref_gelu_only_ms:.4f} ms"
    )

    print()

    print(
        f"PyTorch c_fc + GELU separate sum : "
        f"{ref_fc_only_ms + ref_gelu_only_ms:.4f} ms"
    )

    print(
        f"PyTorch combined benchmark        : "
        f"{ref_fc_gelu_ms:.4f} ms"
    )

    print(
        f"Triton fused c_fc + GELU          : "
        f"{tri_fc_gelu_ms:.4f} ms"
    )

    # ========================================================
    #
    # c_proj detail
    #
    # ========================================================

    print()
    print("=" * 92)
    print("c_proj + Residual Detail")
    print("=" * 92)

    print()

    print(
        f"{'Operation':<32} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10} "
        f"{'Speedup':>9}"
    )

    print("-" * 66)

    print_row(
        "c_proj only",
        ref_proj_only_ms,
        tri_proj_only_ms,
    )

    print_row(
        "Residual add only",
        ref_residual_only_ms,
        tri_residual_only_ms,
    )

    # ========================================================
    # Separate vs fused
    # ========================================================

    print()
    print("-" * 92)
    print("c_proj + Residual Paths")
    print("-" * 92)

    print()

    ref_separate_sum = (
        ref_proj_only_ms
        + ref_residual_only_ms
    )

    tri_separate_sum = (
        tri_proj_only_ms
        + tri_residual_only_ms
    )

    print(
        f"PyTorch separate component sum : "
        f"{ref_separate_sum:.4f} ms"
    )

    print(
        f"PyTorch combined benchmark      : "
        f"{ref_proj_residual_ms:.4f} ms"
    )

    print()

    print(
        f"Triton separate component sum  : "
        f"{tri_separate_sum:.4f} ms"
    )

    print(
        f"Triton unfused actual path      : "
        f"{tri_proj_residual_unfused_ms:.4f} ms"
    )

    print(
        f"Triton fused actual path        : "
        f"{tri_proj_residual_ms:.4f} ms"
    )

    print()

    fusion_saved = (
        tri_proj_residual_unfused_ms
        - tri_proj_residual_ms
    )

    fusion_speedup = (
        tri_proj_residual_unfused_ms
        / tri_proj_residual_ms
    )

    print(
        f"Triton fusion speedup : "
        f"{fusion_speedup:.2f}x"
    )

    print(
        f"Triton fusion saved   : "
        f"{fusion_saved:.4f} ms"
    )


if __name__ == "__main__":
    main()