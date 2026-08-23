import pytest
import torch
import torch.nn.functional as F

from triton_kernels.linear_gelu import (
    triton_linear_gelu_forward,
)


DEVICE = "cuda"
DTYPE = torch.float16


@pytest.mark.parametrize(
    "use_bias",
    [True, False],
)
def test_triton_linear_gelu_forward(use_bias):
    torch.manual_seed(0)

    # GPT-2 MLP c_fc 실제 shape
    B = 2
    T = 128

    K = 768
    N = 3072

    # ========================================================
    # Input
    # ========================================================

    x = torch.randn(
        B,
        T,
        K,
        device=DEVICE,
        dtype=DTYPE,
    )

    weight = torch.randn(
        N,
        K,
        device=DEVICE,
        dtype=DTYPE,
    )

    if use_bias:
        bias = torch.randn(
            N,
            device=DEVICE,
            dtype=DTYPE,
        )
    else:
        bias = None

    # ========================================================
    # PyTorch reference
    #
    # Linear
    #   ↓
    # GELU
    # ========================================================

    ref = F.gelu(
        F.linear(
            x,
            weight,
            bias,
        )
    )

    # ========================================================
    # Triton fused
    #
    # Linear + GELU
    # ========================================================

    out = triton_linear_gelu_forward(
        x,
        weight,
        bias,
    )

    # ========================================================
    # Basic checks
    # ========================================================

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert out.device == ref.device

    # ========================================================
    # Numerical correctness
    # ========================================================

    diff = (
        out.float()
        - ref.float()
    ).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    assert torch.allclose(
        out,
        ref,
        atol=1e-2,
        rtol=1e-2,
    ), (
        f"Linear+GELU mismatch\n"
        f"use_bias={use_bias}\n"
        f"max_diff={max_diff:.6e}\n"
        f"mean_diff={mean_diff:.6e}"
    )