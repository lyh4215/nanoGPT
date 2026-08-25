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

    eager_model = GPT(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    triton_model = TritonGPT(
        config
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    triton_model.load_state_dict(
        eager_model.state_dict()
    )

    eager_model.eval()
    triton_model.eval()

    # ========================================================
    # torch.compile
    #
    # 같은 eager_model의 parameter를 사용.
    # compile wrapper만 별도로 만든다.
    # ========================================================

    compiled_default = torch.compile(
        eager_model,
        backend="inductor",
        mode="default",
        dynamic=False,
    )

    compiled_max = torch.compile(
        eager_model,
        backend="inductor",
        mode="max-autotune",
        dynamic=False,
    )

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

    # ========================================================
    # Benchmark callables
    #
    # targets=None:
    #
    # nanoGPT inference path
    # → 마지막 token에 대해서만 LM Head 계산
    # ========================================================

    @torch.inference_mode()
    def eager_fn():
        return eager_model(
            idx
        )

    @torch.inference_mode()
    def compile_default_fn():
        return compiled_default(
            idx
        )

    @torch.inference_mode()
    def compile_max_fn():
        return compiled_max(
            idx
        )

    @torch.inference_mode()
    def triton_fn():
        return triton_model(
            idx
        )

    # ========================================================
    # Compile + warmup
    #
    # 중요:
    # torch.compile 첫 호출에는 실제 compilation이 발생한다.
    # 그 시간은 runtime benchmark에 넣지 않는다.
    # ========================================================

    print()
    print("Compiling / warming up...")

    eager_fn()

    # first call triggers compile
    compile_default_fn()
    torch.cuda.synchronize()

    compile_max_fn()
    torch.cuda.synchronize()

    # Triton JIT compilation
    triton_fn()
    torch.cuda.synchronize()

    # runtime warmup
    for _ in range(5):
        eager_fn()
        compile_default_fn()
        compile_max_fn()
        triton_fn()

    torch.cuda.synchronize()

    # ========================================================
    # Benchmark
    # ========================================================

    eager_ms = triton.testing.do_bench(
        eager_fn
    )

    default_ms = triton.testing.do_bench(
        compile_default_fn
    )

    max_ms = triton.testing.do_bench(
        compile_max_fn
    )

    triton_ms = triton.testing.do_bench(
        triton_fn
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 100)

    print(
        f"GPT Inference Benchmark "
        f"B={B}, T={T}, "
        f"C={config.n_embd}, "
        f"layers={config.n_layer}"
    )

    print("=" * 100)

    print()

    print(
        f"{'Implementation':<32} "
        f"{'Latency':>12} "
        f"{'vs Eager':>12}"
    )

    print("-" * 60)

    print(
        f"{'PyTorch eager':<32} "
        f"{eager_ms:>9.4f} ms "
        f"{1.0:>11.2f}x"
    )

    print(
        f"{'torch.compile(default)':<32} "
        f"{default_ms:>9.4f} ms "
        f"{eager_ms / default_ms:>11.2f}x"
    )

    print(
        f"{'torch.compile(max-autotune)':<32} "
        f"{max_ms:>9.4f} ms "
        f"{eager_ms / max_ms:>11.2f}x"
    )

    print(
        f"{'TritonGPT':<32} "
        f"{triton_ms:>9.4f} ms "
        f"{eager_ms / triton_ms:>11.2f}x"
    )

    # ========================================================
    # Compare Triton directly against compiled PyTorch
    # ========================================================

    print()
    print("=" * 100)
    print("TritonGPT vs torch.compile")
    print("=" * 100)

    print()

    print(
        "vs compile(default)      : "
        f"{default_ms / triton_ms:.2f}x"
    )

    print(
        "vs compile(max-autotune) : "
        f"{max_ms / triton_ms:.2f}x"
    )

    # ========================================================
    # Winner
    # ========================================================

    results = {
        "PyTorch eager":
            eager_ms,

        "torch.compile(default)":
            default_ms,

        "torch.compile(max-autotune)":
            max_ms,

        "TritonGPT":
            triton_ms,
    }

    winner = min(
        results,
        key=results.get,
    )

    print()
    print("=" * 100)
    print("BEST")
    print("=" * 100)

    print(
        f"{winner}: "
        f"{results[winner]:.4f} ms"
    )


if __name__ == "__main__":
    main()