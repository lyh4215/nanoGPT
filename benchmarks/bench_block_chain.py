import torch
import triton

from model import (
    GPT,
    GPTConfig,
)

from triton_model import (
    TritonGPT,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024


def bench(fn):
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
        f"{name:<28} "
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
    # Initial embedding output
    #
    # Block chain만 재기 위해 embedding 시간은 제거
    # ========================================================

    with torch.no_grad():
        ref_x0 = (
            ref.transformer.wte(idx)
            + ref.transformer.wpe(pos)
        )

        tri_x0 = (
            tri.transformer.wte(idx)
            + tri.transformer.wpe(pos)
        )

    torch.cuda.synchronize()

    # ========================================================
    # Same block × 12
    #
    # 같은 weight를 계속 재사용
    # ========================================================

    def ref_same_block():
        x = ref_x0

        block = (
            ref.transformer.h[0]
        )

        for _ in range(
            config.n_layer
        ):
            x = block(x)

        return x

    def tri_same_block():
        x = tri_x0

        block = (
            tri.transformer.h[0]
        )

        for _ in range(
            config.n_layer
        ):
            x = block(x)

        return x

    # ========================================================
    # Real 12 blocks
    #
    # 매 layer마다 다른 weight 사용
    # ========================================================

    def ref_real_blocks():
        x = ref_x0

        for block in (
            ref.transformer.h
        ):
            x = block(x)

        return x

    def tri_real_blocks():
        x = tri_x0

        for block in (
            tri.transformer.h
        ):
            x = block(x)

        return x

    # ========================================================
    # Benchmark
    # ========================================================

    ref_same_ms = bench(
        ref_same_block
    )

    tri_same_ms = bench(
        tri_same_block
    )

    ref_real_ms = bench(
        ref_real_blocks
    )

    tri_real_ms = bench(
        tri_real_blocks
    )

    # ========================================================
    # Per-layer averages
    # ========================================================

    ref_same_per = (
        ref_same_ms
        / config.n_layer
    )

    tri_same_per = (
        tri_same_ms
        / config.n_layer
    )

    ref_real_per = (
        ref_real_ms
        / config.n_layer
    )

    tri_real_per = (
        tri_real_ms
        / config.n_layer
    )

    # ========================================================
    # Penalty:
    #
    # real / same
    #
    # > 1이면
    # 서로 다른 weight를 사용할 때 느려지는 것
    # ========================================================

    ref_penalty = (
        ref_real_ms
        / ref_same_ms
    )

    tri_penalty = (
        tri_real_ms
        / tri_same_ms
    )

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 90)
    print(
        f"Block Chain Benchmark "
        f"B={B}, T={T}, "
        f"layers={config.n_layer}"
    )
    print("=" * 90)

    print()

    print(
        f"{'Case':<28} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10} "
        f"{'Speedup':>9}"
    )

    print("-" * 62)

    row(
        "Same Block x12",
        ref_same_ms,
        tri_same_ms,
    )

    row(
        "Real 12 Blocks",
        ref_real_ms,
        tri_real_ms,
    )

    # ========================================================
    # Per-layer
    # ========================================================

    print()
    print("=" * 90)
    print("Per-layer average")
    print("=" * 90)

    print()

    print(
        f"{'Case':<28} "
        f"{'PyTorch':>10} "
        f"{'Triton':>10}"
    )

    print("-" * 52)

    print(
        f"{'Same Block':<28} "
        f"{ref_same_per:>8.4f} ms "
        f"{tri_same_per:>8.4f} ms"
    )

    print(
        f"{'Real Blocks':<28} "
        f"{ref_real_per:>8.4f} ms "
        f"{tri_real_per:>8.4f} ms"
    )

    # ========================================================
    # Different-weight penalty
    # ========================================================

    print()
    print("=" * 90)
    print("Different-weight penalty")
    print("=" * 90)

    print()

    print(
        f"PyTorch real / same : "
        f"{ref_penalty:.3f}x"
    )

    print(
        f"Triton  real / same : "
        f"{tri_penalty:.3f}x"
    )

    print()

    print(
        "Additional time from using "
        "different blocks:"
    )

    print(
        f"PyTorch : "
        f"{ref_real_ms - ref_same_ms:.4f} ms"
    )

    print(
        f"Triton  : "
        f"{tri_real_ms - tri_same_ms:.4f} ms"
    )

    # ========================================================
    # Relative extra penalty
    # ========================================================

    extra_gap = (
        (tri_real_ms - tri_same_ms)
        - (ref_real_ms - ref_same_ms)
    )

    print()

    print(
        f"Extra Triton penalty vs PyTorch: "
        f"{extra_gap:.4f} ms"
    )


if __name__ == "__main__":
    main()