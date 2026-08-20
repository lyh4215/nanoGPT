import torch
import triton

from model import GPT, GPTConfig, CausalSelfAttention


# ============================================================
# Config
# ============================================================

B = 8
T = 1024

DEVICE = "cuda"
DTYPE = torch.float16

torch.manual_seed(0)


# ============================================================
# Model
# ============================================================

config = GPTConfig(
    block_size=T,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
    use_triton_flash=False,
)

model = GPT(config).to(DEVICE)

model.train()


# ============================================================
# Input
# ============================================================

x = torch.randint(
    0,
    config.vocab_size,
    (B, T),
    device=DEVICE,
)

targets = torch.randint(
    0,
    config.vocab_size,
    (B, T),
    device=DEVICE,
)


# ============================================================
# Attention backend switch
#
# 같은 model / 같은 weight에서
# attention implementation만 교체
# ============================================================

def set_triton_flash(model, enabled):
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.use_triton_flash = enabled


# ============================================================
# Gradient 저장
# ============================================================

def save_grads(model):
    grads = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()

    return grads


# ============================================================
# Correctness
# ============================================================

def run_correctness():
    print("=" * 70)
    print("Correctness")
    print("=" * 70)

    #
    # --------------------------------------------------------
    # SDPA
    # --------------------------------------------------------
    #

    set_triton_flash(
        model,
        False,
    )

    model.zero_grad(
        set_to_none=True,
    )

    with torch.autocast(
        device_type="cuda",
        dtype=DTYPE,
    ):
        logits_sdpa, loss_sdpa = model(
            x,
            targets,
        )

    loss_sdpa.backward()

    grads_sdpa = save_grads(model)

    logits_sdpa = logits_sdpa.detach()
    loss_sdpa_value = loss_sdpa.detach().item()

    #
    # --------------------------------------------------------
    # Triton
    # --------------------------------------------------------
    #

    set_triton_flash(
        model,
        True,
    )

    model.zero_grad(
        set_to_none=True,
    )

    with torch.autocast(
        device_type="cuda",
        dtype=DTYPE,
    ):
        logits_triton, loss_triton = model(
            x,
            targets,
        )

    loss_triton.backward()

    grads_triton = save_grads(model)

    logits_triton = logits_triton.detach()
    loss_triton_value = loss_triton.detach().item()

    #
    # --------------------------------------------------------
    # Compare output / loss
    # --------------------------------------------------------
    #

    print()

    print(
        f"SDPA loss   : "
        f"{loss_sdpa_value:.8f}"
    )

    print(
        f"Triton loss : "
        f"{loss_triton_value:.8f}"
    )

    print(
        f"Loss diff   : "
        f"{abs(loss_sdpa_value - loss_triton_value):.8e}"
    )

    print(
        f"Logits max diff : "
        f"{(logits_sdpa - logits_triton).abs().max().item():.8e}"
    )

    print()

    #
    # --------------------------------------------------------
    # Parameter gradient compare
    # --------------------------------------------------------
    #

    max_grad_diff = 0.0
    max_grad_name = None

    print("[Gradient diff]")

    for name in grads_sdpa:
        if name not in grads_triton:
            continue

        diff = (
            grads_sdpa[name]
            - grads_triton[name]
        ).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if max_diff > max_grad_diff:
            max_grad_diff = max_diff
            max_grad_name = name

        print(
            f"{name:50s} "
            f"max={max_diff:.6e} "
            f"mean={mean_diff:.6e}"
        )

    print()

    print(
        f"Largest grad diff: "
        f"{max_grad_name} "
        f"({max_grad_diff:.6e})"
    )


# ============================================================
# Benchmark
# ============================================================

def benchmark_backend(
    use_triton,
):
    set_triton_flash(
        model,
        use_triton,
    )

    def fn():
        model.zero_grad(
            set_to_none=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=DTYPE,
        ):
            _, loss = model(
                x,
                targets,
            )

        loss.backward()

    return triton.testing.do_bench(
        fn,
    )


# ============================================================
# Optional: forward only
# ============================================================

def benchmark_forward(
    use_triton,
):
    set_triton_flash(
        model,
        use_triton,
    )

    def fn():
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=DTYPE,
            ):
                model(
                    x,
                    targets,
                )

    return triton.testing.do_bench(
        fn,
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    #
    # correctness
    #
    run_correctness()

    model.zero_grad(
        set_to_none=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    #
    # forward only
    #
    sdpa_fwd_ms = benchmark_forward(
        False,
    )

    triton_fwd_ms = benchmark_forward(
        True,
    )

    #
    # forward + backward
    #
    sdpa_fb_ms = benchmark_backend(
        False,
    )

    triton_fb_ms = benchmark_backend(
        True,
    )

    print()
    print("=" * 70)
    print(
        f"nanoGPT benchmark "
        f"B={B}, T={T}"
    )
    print("=" * 70)

    print("\n[Forward only]")

    print(
        f"SDPA             : "
        f"{sdpa_fwd_ms:.4f} ms"
    )

    print(
        f"Triton Flash     : "
        f"{triton_fwd_ms:.4f} ms"
    )

    print(
        f"Speedup          : "
        f"{sdpa_fwd_ms / triton_fwd_ms:.2f}x"
    )

    print("\n[Forward + Backward]")

    print(
        f"SDPA             : "
        f"{sdpa_fb_ms:.4f} ms"
    )

    print(
        f"Triton Flash     : "
        f"{triton_fb_ms:.4f} ms"
    )

    print(
        f"Speedup          : "
        f"{sdpa_fb_ms / triton_fb_ms:.2f}x"
    )