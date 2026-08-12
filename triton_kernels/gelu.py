import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = (
        pid * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    mask = offsets < n_elements

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # exact GELU
    # 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (
        1.0 + tl.erf(x * 0.7071067811865476)
    )

    tl.store(
        y_ptr + offsets,
        y,
        mask=mask,
    )


def gelu(
    x: torch.Tensor,
    block_size: int = 1024,
):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)

    n_elements = x.numel()

    grid = (
        triton.cdiv(n_elements, block_size),
    )

    _gelu_fwd_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return y