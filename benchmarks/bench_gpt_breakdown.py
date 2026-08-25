import torch
import triton

from model import (
    GPT,
    GPTConfig,
)

from triton_model import (
    TritonGPT,
)

from triton_kernels.linear import (
    triton_linear,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def bench(fn):
    # compile / warmup
    for _ in range(3):
        fn()

    torch.cuda.synchronize()

    return triton.testing.do_bench(
        fn
    )


def row(
    name,
    torch_ms,
    triton_ms,
):
    print(
        f"{name:<24} "
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
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    C = config.n_embd

    # ========================================================
    # Models
    # ========================================================

    ref = GPT(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonGPT(
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

    idx = torch.randint(
        0,
        config.vocab_size,
        (B, T),
        device=DEVICE,
        dtype=torch.long,
    )

    pos = torch.arange(
        0,
        T,
        device=DEVICE,
        dtype=torch.long,
    )

    # ========================================================
    # Embedding intermediate
    # ========================================================

    with torch.no_grad():
        ref_tok = ref.transformer.wte(
            idx
        )

        ref_pos = ref.transformer.wpe(
            pos
        )

        ref_x0 = ref.transformer.drop(
            ref_tok + ref_pos
        )

        tri_tok = tri.transformer.wte(
            idx
        )

        tri_pos = tri.transformer.wpe(
            pos
        )

        tri_x0 = tri.transformer.drop(
            tri_tok + tri_pos
        )

    # ========================================================
    # Embedding benchmark
    # ========================================================

    def ref_embedding():
        tok = ref.transformer.wte(
            idx
        )

        pos_emb = ref.transformer.wpe(
            pos
        )

        return ref.transformer.drop(
            tok + pos_emb
        )

    def tri_embedding():
        tok = tri.transformer.wte(
            idx
        )

        pos_emb = tri.transformer.wpe(
            pos
        )

        return tri.transformer.drop(
            tok + pos_emb
        )

    ref_embedding_ms = bench(
        ref_embedding
    )

    tri_embedding_ms = bench(
        tri_embedding
    )

    # ========================================================
    # Build per-block inputs
    #
    # 각 block을 독립 benchmark할 수 있도록
    # block 진입 시점의 x를 미리 저장.
    # ========================================================

    ref_block_inputs = []
    tri_block_inputs = []

    with torch.no_grad():
        x_ref = ref_x0
        x_tri = tri_x0

        for i in range(
            config.n_layer
        ):
            ref_block_inputs.append(
                x_ref
            )

            tri_block_inputs.append(
                x_tri
            )

            x_ref = (
                ref.transformer.h[i](
                    x_ref
                )
            )

            x_tri = (
                tri.transformer.h[i](
                    x_tri
                )
            )

        ref_after_blocks = x_ref
        tri_after_blocks = x_tri

    torch.cuda.synchronize()

    # ========================================================
    # Per-block benchmark
    # ========================================================

    ref_block_times = []
    tri_block_times = []

    for i in range(
        config.n_layer
    ):
        ref_input = (
            ref_block_inputs[i]
        )

        tri_input = (
            tri_block_inputs[i]
        )

        ref_block = (
            ref.transformer.h[i]
        )

        tri_block = (
            tri.transformer.h[i]
        )

        ref_ms = bench(
            lambda block=ref_block,
                   x=ref_input:
                block(x)
        )

        tri_ms = bench(
            lambda block=tri_block,
                   x=tri_input:
                block(x)
        )

        ref_block_times.append(
            ref_ms
        )

        tri_block_times.append(
            tri_ms
        )

    # ========================================================
    # All 12 blocks as one region
    #
    # per-block sum과 실제 연속 실행을 비교하기 위함
    # ========================================================

    def ref_all_blocks():
        x = ref_x0

        for block in ref.transformer.h:
            x = block(x)

        return x

    def tri_all_blocks():
        x = tri_x0

        for block in tri.transformer.h:
            x = block(x)

        return x

    ref_blocks_ms = bench(
        ref_all_blocks
    )

    tri_blocks_ms = bench(
        tri_all_blocks
    )

    # ========================================================
    # Final LayerNorm
    # ========================================================

    ref_lnf_ms = bench(
        lambda:
            ref.transformer.ln_f(
                ref_after_blocks
            )
    )

    tri_lnf_ms = bench(
        lambda:
            tri.transformer.ln_f(
                tri_after_blocks
            )
    )

    # final LN output을 미리 만들어서
    # LM head benchmark에 LN 시간이 섞이지 않게 함.
    with torch.no_grad():
        ref_final = (
            ref.transformer.ln_f(
                ref_after_blocks
            )
        )

        tri_final = (
            tri.transformer.ln_f(
                tri_after_blocks
            )
        )

        # nanoGPT inference와 동일:
        # 마지막 token만 projection
        ref_last = (
            ref_final[:, -1:, :]
            .contiguous()
        )

        tri_last = (
            tri_final[:, -1:, :]
            .contiguous()
        )

    # ========================================================
    # LM Head
    #
    # M = B = 8
    # K = 768
    # N = 50304
    # ========================================================

    def ref_lm_head():
        return ref.lm_head(
            ref_last
        )

    def tri_lm_head():
        return triton_linear(
            tri_last,
            tri.lm_head.weight,
            None,
        )

    ref_lm_ms = bench(
        ref_lm_head
    )

    tri_lm_ms = bench(
        tri_lm_head
    )

    # ========================================================
    # Full GPT inference
    #
    # targets=None
    # → 마지막 token logits만 계산
    # ========================================================

    def ref_full():
        return ref(
            idx
        )

    def tri_full():
        return tri(
            idx
        )

    ref_full_ms = bench(
        ref_full
    )

    tri_full_ms = bench(
        tri_full
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 90)
    print(
        f"GPT Forward Breakdown "
        f"B={B}, T={T}, "
        f"C={C}, "
        f"layers={config.n_layer}"
    )
    print("=" * 90)

    print()

    print(
        f"{'Component':<24} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10} "
        f"{'Speedup':>9}"
    )

    print("-" * 60)

    row(
        "Embedding",
        ref_embedding_ms,
        tri_embedding_ms,
    )

    print("-" * 60)

    for i in range(
        config.n_layer
    ):
        row(
            f"Block {i}",
            ref_block_times[i],
            tri_block_times[i],
        )

    print("-" * 60)

    row(
        "12 Blocks region",
        ref_blocks_ms,
        tri_blocks_ms,
    )

    row(
        "Final LayerNorm",
        ref_lnf_ms,
        tri_lnf_ms,
    )

    row(
        "LM Head",
        ref_lm_ms,
        tri_lm_ms,
    )

    print("-" * 60)

    row(
        "Full GPT",
        ref_full_ms,
        tri_full_ms,
    )

    # ========================================================
    # Block stats
    # ========================================================

    ref_block_sum = sum(
        ref_block_times
    )

    tri_block_sum = sum(
        tri_block_times
    )

    ref_block_avg = (
        ref_block_sum
        / config.n_layer
    )

    tri_block_avg = (
        tri_block_sum
        / config.n_layer
    )

    print()
    print("=" * 90)
    print("Block Statistics")
    print("=" * 90)

    print()

    print(
        f"PyTorch block avg : "
        f"{ref_block_avg:.4f} ms"
    )

    print(
        f"Triton block avg  : "
        f"{tri_block_avg:.4f} ms"
    )

    print(
        f"Average speedup    : "
        f"{ref_block_avg / tri_block_avg:.2f}x"
    )

    print()

    print(
        f"Per-block sum PyTorch : "
        f"{ref_block_sum:.4f} ms"
    )

    print(
        f"12-block region PyTorch: "
        f"{ref_blocks_ms:.4f} ms"
    )

    print()

    print(
        f"Per-block sum Triton  : "
        f"{tri_block_sum:.4f} ms"
    )

    print(
        f"12-block region Triton : "
        f"{tri_blocks_ms:.4f} ms"
    )

    # ========================================================
    # Major component sum
    # ========================================================

    ref_component_sum = (
        ref_embedding_ms
        + ref_blocks_ms
        + ref_lnf_ms
        + ref_lm_ms
    )

    tri_component_sum = (
        tri_embedding_ms
        + tri_blocks_ms
        + tri_lnf_ms
        + tri_lm_ms
    )

    print()
    print("=" * 90)
    print("Component Sum vs Full GPT")
    print("=" * 90)

    print()

    print(
        f"PyTorch component sum : "
        f"{ref_component_sum:.4f} ms"
    )

    print(
        f"PyTorch full GPT      : "
        f"{ref_full_ms:.4f} ms"
    )

    print()

    print(
        f"Triton component sum  : "
        f"{tri_component_sum:.4f} ms"
    )

    print(
        f"Triton full GPT       : "
        f"{tri_full_ms:.4f} ms"
    )


if __name__ == "__main__":
    main()