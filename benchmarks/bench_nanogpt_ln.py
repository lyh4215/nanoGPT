import torch
import triton

from model import GPT, GPTConfig


device = "cuda"

B = 4
T = 1024


base = dict(
    block_size=T,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
)


torch_config = GPTConfig(
    **base,
    use_triton_ln=False,
)

triton_config = GPTConfig(
    **base,
    use_triton_ln=True,
)


model_torch = GPT(torch_config).to(device)
model_triton = GPT(triton_config).to(device)

model_triton.load_state_dict(
    model_torch.state_dict()
)

model_torch.train()
model_triton.train()


idx = torch.randint(
    0,
    base["vocab_size"],
    (B, T),
    device=device,
)

targets = torch.randint(
    0,
    base["vocab_size"],
    (B, T),
    device=device,
)


def torch_step():
    model_torch.zero_grad(set_to_none=True)

    _, loss = model_torch(
        idx,
        targets,
    )

    loss.backward()


def triton_step():
    model_triton.zero_grad(set_to_none=True)

    _, loss = model_triton(
        idx,
        targets,
    )

    loss.backward()


# warmup
for _ in range(5):
    torch_step()
    triton_step()

torch.cuda.synchronize()


torch_ms = triton.testing.do_bench(
    torch_step,
    warmup=25,
    rep=100,
)

triton_ms = triton.testing.do_bench(
    triton_step,
    warmup=25,
    rep=100,
)


print(f"PyTorch : {torch_ms:.4f} ms")
print(f"Triton  : {triton_ms:.4f} ms")
print(f"Speedup : {torch_ms / triton_ms:.2f}x")