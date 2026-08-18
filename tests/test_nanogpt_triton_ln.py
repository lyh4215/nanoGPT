# tests/test_nanogpt_triton_ln.py

import torch

from model import GPT, GPTConfig


device = "cuda"

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


base_config = dict(
    block_size=128,   # correctness test니까 작게
    vocab_size=1000,
    n_layer=2,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=True,
)


torch_config = GPTConfig(
    **base_config,
    use_triton_ln=False,
)

triton_config = GPTConfig(
    **base_config,
    use_triton_ln=True,
)


model_torch = GPT(torch_config).to(device)
model_triton = GPT(triton_config).to(device)


# 완전히 같은 parameter 사용
model_triton.load_state_dict(
    model_torch.state_dict()
)


idx = torch.randint(
    0,
    base_config["vocab_size"],
    (2, 128),
    device=device,
)

targets = torch.randint(
    0,
    base_config["vocab_size"],
    (2, 128),
    device=device,
)


# -------------------------
# forward
# -------------------------

logits_torch, loss_torch = model_torch(
    idx,
    targets,
)

logits_triton, loss_triton = model_triton(
    idx,
    targets,
)


print(
    "loss torch :",
    loss_torch.item(),
)

print(
    "loss triton:",
    loss_triton.item(),
)

print(
    "max logits diff:",
    (logits_torch - logits_triton)
        .abs()
        .max()
        .item(),
)


# -------------------------
# backward
# -------------------------

model_torch.zero_grad(
    set_to_none=True,
)

model_triton.zero_grad(
    set_to_none=True,
)


loss_torch.backward()
loss_triton.backward()


max_grad_diff = 0.0
max_grad_name = None

for (
    name_t,
    param_t
), (
    name_r,
    param_r
) in zip(
    model_torch.named_parameters(),
    model_triton.named_parameters(),
):

    assert name_t == name_r

    if param_t.grad is None:
        continue

    diff = (
        param_t.grad
        - param_r.grad
    ).abs().max().item()

    if diff > max_grad_diff:
        max_grad_diff = diff
        max_grad_name = name_t


print(
    "max grad diff:",
    max_grad_diff,
)

print(
    "max grad param:",
    max_grad_name,
)