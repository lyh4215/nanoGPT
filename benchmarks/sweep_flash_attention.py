import csv
import itertools
import math

import torch
import triton

from triton_kernels.attention import triton_flash_attention_forward


B = 8
H = 12
T = 1024
D = 64

DTYPE = torch.float16


# 너무 터무니없는 조합까지 돌리지 않도록 적당히 제한
BLOCK_MS = [16, 32, 64]
BLOCK_NS = [16, 32, 64]
NUM_WARPS_LIST = [2, 4, 8]


def reference_sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
    )


def benchmark(fn):
    # warmup / rep 단위는 ms
    return triton.testing.do_bench(
        fn,
        warmup=100,
        rep=300,
    )


def main():
    torch.manual_seed(0)

    q = torch.randn(B, H, T, D, device="cuda", dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = reference_sdpa(q, k, v)

    results = []

    configs = list(
        itertools.product(
            BLOCK_MS,
            BLOCK_NS,
            NUM_WARPS_LIST,
        )
    )

    print(f"Testing {len(configs)} configurations")
    print()

    for i, (bm, bn, nw) in enumerate(configs, 1):

        print(
            f"[{i:02d}/{len(configs)}] "
            f"BM={bm:2d} BN={bn:2d} warps={nw}",
            end=" ... ",
            flush=True,
        )

        try:
            # 한 번 호출해서 compile + correctness 확인
            out = triton_flash_attention_forward(
                q,
                k,
                v,
                block_m=bm,
                block_n=bn,
                num_warps=nw,
            )

            torch.cuda.synchronize()

            max_diff = (out - ref).abs().max().item()

            # FP16이므로 너무 엄격하게 잡지 않음
            if not math.isfinite(max_diff) or max_diff > 0.01:
                print(f"FAIL correctness diff={max_diff}")

                results.append({
                    "block_m": bm,
                    "block_n": bn,
                    "num_warps": nw,
                    "ms": float("inf"),
                    "max_diff": max_diff,
                    "status": "incorrect",
                })
                continue

            ms = benchmark(
                lambda: triton_flash_attention_forward(
                    q,
                    k,
                    v,
                    block_m=bm,
                    block_n=bn,
                    num_warps=nw,
                )
            )

            print(
                f"{ms:.4f} ms "
                f"(diff={max_diff:.6f})"
            )

            results.append({
                "block_m": bm,
                "block_n": bn,
                "num_warps": nw,
                "ms": ms,
                "max_diff": max_diff,
                "status": "ok",
            })

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

            results.append({
                "block_m": bm,
                "block_n": bn,
                "num_warps": nw,
                "ms": float("inf"),
                "max_diff": float("nan"),
                "status": "error",
            })

    valid = [
        r for r in results
        if r["status"] == "ok"
    ]

    valid.sort(key=lambda x: x["ms"])

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"{'rank':>4} "
        f"{'BM':>4} "
        f"{'BN':>4} "
        f"{'warps':>6} "
        f"{'time(ms)':>10} "
        f"{'max diff':>12}"
    )

    for rank, r in enumerate(valid, 1):
        print(
            f"{rank:4d} "
            f"{r['block_m']:4d} "
            f"{r['block_n']:4d} "
            f"{r['num_warps']:6d} "
            f"{r['ms']:10.4f} "
            f"{r['max_diff']:12.6f}"
        )

    with open(
        "flash_attention_sweep.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "block_m",
                "block_n",
                "num_warps",
                "ms",
                "max_diff",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("saved: flash_attention_sweep.csv")

    if valid:
        best = valid[0]

        print()
        print("BEST CONFIG")
        print(
            f"BLOCK_M={best['block_m']}, "
            f"BLOCK_N={best['block_n']}, "
            f"num_warps={best['num_warps']}"
        )
        print(f"{best['ms']:.4f} ms")


if __name__ == "__main__":
    main()