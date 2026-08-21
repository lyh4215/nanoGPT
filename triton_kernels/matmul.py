import triton
import triton.language as tl


@triton.jit
def _matmul_accumulate(
    acc,
    a,
    b,
):
    """
    Tile-level matrix multiplication primitive.

    a   : [BLOCK_M, BLOCK_K]
    b   : [BLOCK_N, BLOCK_K]
    acc : [BLOCK_M, BLOCK_N]

    Computes:
        acc += a @ b.T
    """

    acc += tl.dot(
        a,
        tl.trans(b),
    )

    return acc