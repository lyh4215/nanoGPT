# benchmarks/torch_sdpa_probe.py

import torch
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel


B = 8
H = 12
T = 1024
D = 64

device = "cuda"
dtype = torch.float16


print("torch:", torch.__version__)
print("gpu:", torch.cuda.get_device_name())
print("cc:", torch.cuda.get_device_capability())


q = torch.randn(
    B, H, T, D,
    device=device,
    dtype=dtype,
)

k = torch.randn(
    B, H, T, D,
    device=device,
    dtype=dtype,
)

v = torch.randn(
    B, H, T, D,
    device=device,
    dtype=dtype,
)


def run_sdpa():
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )


# warmup
for _ in range(5):
    out = run_sdpa()

torch.cuda.synchronize()

print("warmup done")
print("out:", out.shape, out.dtype)


# NCU는 --profile-from-start off 로 실행할 것
torch.cuda.cudart().cudaProfilerStart()

out = run_sdpa()

torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStop()

print("profiled")