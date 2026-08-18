import argparse

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)

from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--impl",
        choices=["torch", "triton"],
        required=True,
    )

    args = parser.parse_args()

    device = "cuda"

    B = 8
    T = 1024

    config = GPTConfig(
        block_size=T,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        use_triton_ln=(args.impl == "triton"),
    )

    model = GPT(config).to(device)
    model.train()

    idx = torch.randint(
        0,
        config.vocab_size,
        (B, T),
        device=device,
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (B, T),
        device=device,
    )

    # -------------------------
    # warmup
    # -------------------------

    for _ in range(3):

        model.zero_grad(set_to_none=True)

        _, loss = model(
            idx,
            targets,
        )

        loss.backward()

    torch.cuda.synchronize()

    # -------------------------
    # profiler
    # -------------------------

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],

        # 1 iter는 profiler warmup
        # 다음 3 iter 기록
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=3,
            repeat=1,
        ),
    ) as prof:

        for _ in range(4):

            model.zero_grad(
                set_to_none=True,
            )

            with record_function("GPT_FORWARD"):

                _, loss = model(
                    idx,
                    targets,
                )

            with record_function("GPT_BACKWARD"):

                loss.backward()

            prof.step()

    torch.cuda.synchronize()

    # -------------------------
    # CUDA 기준 TOP operations
    # -------------------------

    print(
        "\n"
        "=========================================="
    )
    print(
        f"{args.impl.upper()} - CUDA TIME"
    )
    print(
        "=========================================="
    )

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
    )

    # -------------------------
    # CPU 기준도 참고용
    # -------------------------

    print(
        "\n"
        "=========================================="
    )
    print(
        f"{args.impl.upper()} - CPU TIME"
    )
    print(
        "=========================================="
    )

    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=30,
        )
    )

    # timeline 저장
    prof.export_chrome_trace(
        f"profile_nanogpt_{args.impl}.json"
    )


if __name__ == "__main__":
    main()