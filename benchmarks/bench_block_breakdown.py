import torch
import triton

from model import (
    Block,
    GPTConfig,
)

from triton_model import (
    TritonBlock,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def bench(fn):
    for _ in range(3):
        fn()

    torch.cuda.synchronize()

    return triton.testing.do_bench(fn)


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    C = config.n_embd

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

    x = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    # ========================================================
    # 미리 intermediate 생성
    #
    # component benchmark가 앞 component의 시간까지
    # 포함하지 않게 하기 위함
    # ========================================================

    with torch.no_grad():
        ref_ln1_out = ref.ln_1(x)
        tri_ln1_out = tri.ln_1(x)

        ref_attn_out = ref.attn(
            ref_ln1_out
        )

        tri_attn_out = tri.attn(
            tri_ln1_out
        )

        ref_res1 = (
            x + ref_attn_out
        )

        tri_res1 = (
            x + tri_attn_out
        )

        ref_ln2_out = ref.ln_2(
            ref_res1
        )

        tri_ln2_out = tri.ln_2(
            tri_res1
        )

        ref_mlp_out = ref.mlp(
            ref_ln2_out
        )

        tri_mlp_out = tri.mlp(
            tri_ln2_out
        )

    torch.cuda.synchronize()

    # ========================================================
    # LayerNorm 1
    # ========================================================

    ref_ln1_ms = bench(
        lambda: ref.ln_1(x)
    )

    tri_ln1_ms = bench(
        lambda: tri.ln_1(x)
    )

    # ========================================================
    # Attention
    # ========================================================

    ref_attn_ms = bench(
        lambda: ref.attn(
            ref_ln1_out
        )
    )

    tri_attn_ms = bench(
        lambda: tri.attn(
            tri_ln1_out
        )
    )

    # ========================================================
    # Residual 1
    # ========================================================

    ref_res1_ms = bench(
        lambda: x + ref_attn_out
    )

    tri_res1_ms = bench(
        lambda: x + tri_attn_out
    )

    # ========================================================
    # LayerNorm 2
    # ========================================================

    ref_ln2_ms = bench(
        lambda: ref.ln_2(
            ref_res1
        )
    )

    tri_ln2_ms = bench(
        lambda: tri.ln_2(
            tri_res1
        )
    )

    # ========================================================
    # MLP
    # ========================================================

    ref_mlp_ms = bench(
        lambda: ref.mlp(
            ref_ln2_out
        )
    )

    tri_mlp_ms = bench(
        lambda: tri.mlp(
            tri_ln2_out
        )
    )

    # ========================================================
    # Residual 2
    # ========================================================

    ref_res2_ms = bench(
        lambda: ref_res1 + ref_mlp_out
    )

    tri_res2_ms = bench(
        lambda: tri_res1 + tri_mlp_out
    )

    # ========================================================
    # Full block
    # ========================================================

    ref_block_ms = bench(
        lambda: ref(x)
    )

    tri_block_ms = bench(
        lambda: tri(x)
    )

    # ========================================================
    # Print
    # ========================================================

    def row(
        name,
        torch_ms,
        triton_ms,
    ):
        print(
            f"{name:<18} "
            f"{torch_ms:>9.4f} "
            f"{triton_ms:>9.4f} "
            f"{torch_ms / triton_ms:>8.2f}x"
        )

    print()
    print("=" * 70)
    print(
        f"Transformer Block Forward Breakdown "
        f"B={B}, T={T}, C={C}"
    )
    print("=" * 70)

    print()
    print(
        f"{'Component':<18} "
        f"{'PyTorch':>9} "
        f"{'Triton':>9} "
        f"{'Speedup':>8}"
    )

    print("-" * 55)

    row(
        "LayerNorm 1",
        ref_ln1_ms,
        tri_ln1_ms,
    )

    row(
        "Attention",
        ref_attn_ms,
        tri_attn_ms,
    )

    row(
        "Residual 1",
        ref_res1_ms,
        tri_res1_ms,
    )

    row(
        "LayerNorm 2",
        ref_ln2_ms,
        tri_ln2_ms,
    )

    row(
        "MLP",
        ref_mlp_ms,
        tri_mlp_ms,
    )

    row(
        "Residual 2",
        ref_res2_ms,
        tri_res2_ms,
    )

    print("-" * 55)

    row(
        "Full Block",
        ref_block_ms,
        tri_block_ms,
    )

    ref_sum = (
        ref_ln1_ms
        + ref_attn_ms
        + ref_res1_ms
        + ref_ln2_ms
        + ref_mlp_ms
        + ref_res2_ms
    )

    tri_sum = (
        tri_ln1_ms
        + tri_attn_ms
        + tri_res1_ms
        + tri_ln2_ms
        + tri_mlp_ms
        + tri_res2_ms
    )

    print()
    print(
        f"Component sum PyTorch : "
        f"{ref_sum:.4f} ms"
    )

    print(
        f"Component sum Triton  : "
        f"{tri_sum:.4f} ms"
    )


if __name__ == "__main__":
    main()