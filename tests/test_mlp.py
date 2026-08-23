import torch

from model import MLP, GPTConfig
from triton_model import TritonMLP


DEVICE = "cuda"
DTYPE = torch.float16


def test_triton_mlp_correctness():
    torch.manual_seed(0)

    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    ref = MLP(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri = TritonMLP(config).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tri.load_state_dict(
        ref.state_dict()
    )

    B = 2
    T = 128
    C = config.n_embd

    x = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    dy = torch.randn(
        B,
        T,
        C,
        device=DEVICE,
        dtype=DTYPE,
    )

    # PyTorch
    x_ref = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    ref.zero_grad(set_to_none=True)

    y_ref = ref(x_ref)
    y_ref.backward(dy)

    # Triton
    x_tri = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    tri.zero_grad(set_to_none=True)

    y_tri = tri(x_tri)
    y_tri.backward(dy)

    # Forward
    assert torch.allclose(
        y_ref,
        y_tri,
        atol=1e-2,
        rtol=1e-2,
    )

    # dX
    assert torch.allclose(
        x_ref.grad,
        x_tri.grad,
        atol=1e-2,
        rtol=1e-2,
    )

    # Parameter grads
    ref_params = dict(ref.named_parameters())
    tri_params = dict(tri.named_parameters())

    for name in ref_params:
        assert torch.allclose(
            ref_params[name].grad,
            tri_params[name].grad,
            atol=1e-1,
            rtol=1e-2,
        ), name