import torch
import triton

from model import (
    GPTConfig,
    Block,
)

from triton_model import (
    TritonBlock,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024
C = 768


def bench(fn):
    # warmup
    for _ in range(3):
        fn()

    torch.cuda.synchronize()

    return triton.testing.do_bench(
        fn
    )


def print_row(
    name,
    torch_ms,
    triton_ms,
):
    print(
        f"{name:<34} "
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

    tri.load_state_dict(
        ref.state_dict()
    )

    ref.eval()
    tri.eval()

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

    ref_x = x.clone()
    tri_x = x.clone()

    # ========================================================
    # Prepare intermediate tensors
    #
    # measurement 안에서 upstream 연산까지 같이 측정되지 않게
    # 각 stage의 입력을 미리 계산해둠.
    # ========================================================

    with torch.no_grad():

        # ----------------------------------------------------
        # LN1 outputs
        # ----------------------------------------------------

        ref_ln1_out = ref.ln_1(
            ref_x
        )

        tri_ln1_out = tri.ln_1(
            tri_x
        )

        # ----------------------------------------------------
        # Attention branch output
        #
        # PyTorch:
        # x1 = x + attn(ln1)
        #
        # Triton:
        # x1 = attn(ln1, residual=x)
        # ----------------------------------------------------

        ref_x1 = (
            ref_x
            + ref.attn(
                ref_ln1_out
            )
        )

        tri_x1 = tri.attn(
            tri_ln1_out,
            residual=tri_x,
        )

        # ----------------------------------------------------
        # LN2 outputs
        # ----------------------------------------------------

        ref_ln2_out = ref.ln_2(
            ref_x1
        )

        tri_ln2_out = tri.ln_2(
            tri_x1
        )

    torch.cuda.synchronize()

    # ========================================================
    # 1. LayerNorm 1
    # ========================================================

    def ref_ln1():
        return ref.ln_1(
            ref_x
        )

    def tri_ln1():
        return tri.ln_1(
            tri_x
        )

    # ========================================================
    # 2. Attention + Residual
    #
    # PyTorch:
    #
    #   attn output
    #   +
    #   residual add
    #
    # Triton:
    #
    #   c_proj + residual
    #   fused
    # ========================================================

    def ref_attn_residual():
        return (
            ref_x
            + ref.attn(
                ref_ln1_out
            )
        )

    def tri_attn_residual():
        return tri.attn(
            tri_ln1_out,
            residual=tri_x,
        )

    # ========================================================
    # 3. LayerNorm 2
    # ========================================================

    def ref_ln2():
        return ref.ln_2(
            ref_x1
        )

    def tri_ln2():
        return tri.ln_2(
            tri_x1
        )

    # ========================================================
    # 4. MLP + Residual
    # ========================================================

    def ref_mlp_residual():
        return (
            ref_x1
            + ref.mlp(
                ref_ln2_out
            )
        )

    def tri_mlp_residual():
        return tri.mlp(
            tri_ln2_out,
            residual=tri_x1,
        )

    # ========================================================
    # Full Block
    # ========================================================

    def ref_full():
        return ref(
            ref_x
        )

    def tri_full():
        return tri(
            tri_x
        )

    # ========================================================
    # Benchmark main components
    # ========================================================

    ref_ln1_ms = bench(
        ref_ln1
    )

    tri_ln1_ms = bench(
        tri_ln1
    )

    ref_attn_ms = bench(
        ref_attn_residual
    )

    tri_attn_ms = bench(
        tri_attn_residual
    )

    ref_ln2_ms = bench(
        ref_ln2
    )

    tri_ln2_ms = bench(
        tri_ln2
    )

    ref_mlp_ms = bench(
        ref_mlp_residual
    )

    tri_mlp_ms = bench(
        tri_mlp_residual
    )

    ref_full_ms = bench(
        ref_full
    )

    tri_full_ms = bench(
        tri_full
    )

    # ========================================================
    # Print breakdown
    # ========================================================

    print()
    print("=" * 94)
    print(
        f"Transformer Block Forward Breakdown "
        f"B={B}, T={T}, C={C}"
    )
    print("=" * 94)

    print()

    print(
        f"{'Component':<34} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10} "
        f"{'Speedup':>9}"
    )

    print("-" * 68)

    print_row(
        "LayerNorm 1",
        ref_ln1_ms,
        tri_ln1_ms,
    )

    print_row(
        "Attention + Residual",
        ref_attn_ms,
        tri_attn_ms,
    )

    print_row(
        "LayerNorm 2",
        ref_ln2_ms,
        tri_ln2_ms,
    )

    print_row(
        "MLP + Residual",
        ref_mlp_ms,
        tri_mlp_ms,
    )

    print("-" * 68)

    print_row(
        "Full Block",
        ref_full_ms,
        tri_full_ms,
    )

    # ========================================================
    # Component sums
    # ========================================================

    ref_sum = (
        ref_ln1_ms
        + ref_attn_ms
        + ref_ln2_ms
        + ref_mlp_ms
    )

    tri_sum = (
        tri_ln1_ms
        + tri_attn_ms
        + tri_ln2_ms
        + tri_mlp_ms
    )

    print()
    print("=" * 94)
    print("Component Sum vs Full Block")
    print("=" * 94)

    print()

    print(
        f"PyTorch component sum : "
        f"{ref_sum:.4f} ms"
    )

    print(
        f"PyTorch full block    : "
        f"{ref_full_ms:.4f} ms"
    )

    print()

    print(
        f"Triton component sum  : "
        f"{tri_sum:.4f} ms"
    )

    print(
        f"Triton full block     : "
        f"{tri_full_ms:.4f} ms"
    )

    # ========================================================
    # Triton fusion A/B
    #
    # 현재 fused path와
    #
    #   triton attention
    #   +
    #   separate residual
    #
    # 를 같은 Triton implementation끼리 직접 비교.
    # ========================================================

    def tri_attn_unfused():
        y = tri.attn(
            tri_ln1_out
        )

        return (
            tri_x
            + y
        )

    def tri_attn_fused():
        return tri.attn(
            tri_ln1_out,
            residual=tri_x,
        )

    def tri_mlp_unfused():
        y = tri.mlp(
            tri_ln2_out
        )

        return (
            tri_x1
            + y
        )

    def tri_mlp_fused():
        return tri.mlp(
            tri_ln2_out,
            residual=tri_x1,
        )

    tri_attn_unfused_ms = bench(
        tri_attn_unfused
    )

    tri_attn_fused_ms = bench(
        tri_attn_fused
    )

    tri_mlp_unfused_ms = bench(
        tri_mlp_unfused
    )

    tri_mlp_fused_ms = bench(
        tri_mlp_fused
    )

    # ========================================================
    # Fusion result
    # ========================================================

    print()
    print("=" * 94)
    print("Triton Residual Fusion A/B")
    print("=" * 94)

    print()

    print(
        f"{'Branch':<24} "
        f"{'Unfused':>10} "
        f"{'Fused':>10} "
        f"{'Speedup':>10} "
        f"{'Saved':>10}"
    )

    print("-" * 70)

    print(
        f"{'Attention':<24} "
        f"{tri_attn_unfused_ms:>8.4f} ms "
        f"{tri_attn_fused_ms:>8.4f} ms "
        f"{tri_attn_unfused_ms / tri_attn_fused_ms:>9.2f}x "
        f"{tri_attn_unfused_ms - tri_attn_fused_ms:>8.4f} ms"
    )

    print(
        f"{'MLP':<24} "
        f"{tri_mlp_unfused_ms:>8.4f} ms "
        f"{tri_mlp_fused_ms:>8.4f} ms "
        f"{tri_mlp_unfused_ms / tri_mlp_fused_ms:>9.2f}x "
        f"{tri_mlp_unfused_ms - tri_mlp_fused_ms:>8.4f} ms"
    )

    total_saved = (
        tri_attn_unfused_ms
        - tri_attn_fused_ms
        + tri_mlp_unfused_ms
        - tri_mlp_fused_ms
    )

    print()

    print(
        f"Estimated saved per Block: "
        f"{total_saved:.4f} ms"
    )


if __name__ == "__main__":
    main()