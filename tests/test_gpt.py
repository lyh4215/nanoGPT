import torch

from model import (
    GPT,
    GPTConfig,
)

from triton_model import (
    TritonGPT,
)


DEVICE = "cuda"
DTYPE = torch.float16


def compare(
    name,
    ref,
    tri,
    atol,
    rtol,
):
    diff = (
        ref.float()
        - tri.float()
    ).abs()

    max_diff = (
        diff.max().item()
    )

    mean_diff = (
        diff.mean().item()
    )

    ok = torch.allclose(
        ref,
        tri,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"{name:<35} "
        f"max={max_diff:.6e} "
        f"mean={mean_diff:.6e} "
        f"allclose={ok}"
    )

    return ok


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # correctness에서는 가볍게 2 blocks
    config = GPTConfig(
        block_size=128,
        vocab_size=50304,
        n_layer=2,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    B = 1
    T = 64

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

    ref.train()
    tri.train()

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

    targets = torch.randint(
        0,
        config.vocab_size,
        (B, T),
        device=DEVICE,
        dtype=torch.long,
    )

    # ========================================================
    # PyTorch
    # ========================================================

    ref.zero_grad(
        set_to_none=True
    )

    logits_ref, loss_ref = ref(
        idx,
        targets,
    )

    loss_ref.backward()

    # ========================================================
    # Triton
    # ========================================================

    tri.zero_grad(
        set_to_none=True
    )

    logits_tri, loss_tri = tri(
        idx,
        targets,
    )

    loss_tri.backward()

    torch.cuda.synchronize()

    # ========================================================
    # Forward
    # ========================================================

    print()
    print("=" * 110)
    print(
        f"Triton GPT Correctness "
        f"B={B}, T={T}, "
        f"layers={config.n_layer}"
    )
    print("=" * 110)

    print()
    print("[Forward]")

    logits_ok = compare(
        "logits",
        logits_ref,
        logits_tri,
        atol=5e-2,
        rtol=1e-2,
    )

    loss_diff = abs(
        loss_ref.item()
        - loss_tri.item()
    )

    print(
        f"{'loss':<35} "
        f"ref={loss_ref.item():.6f} "
        f"tri={loss_tri.item():.6f} "
        f"diff={loss_diff:.6e}"
    )

    # ========================================================
    # Representative gradients
    #
    # 전체 50304x768 embedding까지 매번 diff tensor를
    # 만드는 것보다 중요한 경로들을 먼저 확인.
    # ========================================================

    print()
    print("[Representative Parameter Gradients]")

    ref_params = dict(
        ref.named_parameters()
    )

    tri_params = dict(
        tri.named_parameters()
    )

    names = [
        "transformer.wte.weight",

        "transformer.h.0.ln_1.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.ln_2.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",

        "transformer.h.1.attn.c_attn.weight",
        "transformer.h.1.mlp.c_fc.weight",

        "transformer.ln_f.weight",
    ]

    grad_ok = True

    for name in names:
        ok = compare(
            name,
            ref_params[name].grad,
            tri_params[name].grad,
            atol=1e-1,
            rtol=2e-2,
        )

        grad_ok = (
            grad_ok
            and ok
        )

    print()
    print("=" * 110)
    print("Summary")
    print("=" * 110)

    print(
        f"Logits : "
        f"{'PASS' if logits_ok else 'FAIL'}"
    )

    print(
        f"Grad   : "
        f"{'PASS' if grad_ok else 'FAIL'}"
    )

    print(
        "Overall: "
        + (
            "PASS"
            if logits_ok and grad_ok
            else "FAIL"
        )
    )


if __name__ == "__main__":
    main()