from pathlib import Path

import torch
import triton

from triton.runtime import driver

from triton_kernels.linear_gelu import (
    _linear_gelu_fwd_kernel,
)


DEVICE = "cuda"
DTYPE = torch.float16

B = 8
T = 1024

M = B * T
K = 768
N = 3072


CONFIGS = {
    "baseline_bn64_s2": {
        "BM": 128,
        "BN": 64,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 2,
    },

    "bn64_s3": {
        "BM": 128,
        "BN": 64,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 3,
    },

    "bn128_s2": {
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "W": 4,
        "G": 8,
        "S": 2,
    },
}


def compile_kernel(
    x,
    weight,
    bias,
    z,
    y,
    cfg,
):
    grid = (
        triton.cdiv(
            M,
            cfg["BM"],
        )
        * triton.cdiv(
            N,
            cfg["BN"],
        ),
    )

    compiled = _linear_gelu_fwd_kernel[grid](
        x,
        weight,
        bias,
        z,
        y,

        M=M,
        N=N,
        K=K,

        stride_xm=x.stride(0),
        stride_xk=x.stride(1),

        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),

        stride_zm=z.stride(0),
        stride_zn=z.stride(1),

        stride_ym=y.stride(0),
        stride_yn=y.stride(1),

        HAS_BIAS=True,

        BLOCK_M=cfg["BM"],
        BLOCK_N=cfg["BN"],
        BLOCK_K=cfg["BK"],

        GROUP_SIZE_M=cfg["G"],

        num_warps=cfg["W"],
        num_stages=cfg["S"],
    )

    torch.cuda.synchronize()

    # register 정보 등이 lazy init이면 초기화
    if hasattr(
        compiled,
        "_init_handles",
    ):
        compiled._init_handles()

    return compiled


def count_instruction(
    sass,
    keyword,
):
    return sum(
        keyword in line
        for line in sass.splitlines()
    )


def get_instruction_lines(
    sass,
    keyword,
):
    return [
        line.strip()
        for line in sass.splitlines()
        if keyword in line
    ]


def main():
    torch.manual_seed(0)
    torch.cuda.init()

    # ========================================================
    # GPU properties
    # ========================================================

    device = torch.cuda.current_device()

    props = (
        driver.active.utils
        .get_device_properties(
            device
        )
    )

    torch_props = (
        torch.cuda.get_device_properties(
            device
        )
    )

    print()
    print("=" * 100)
    print("GPU")
    print("=" * 100)

    print(
        torch.cuda.get_device_name(
            device
        )
    )

    print()

    for key, value in props.items():
        print(
            f"{key:<30}: {value}"
        )

    # Triton properties
    num_sm = props[
        "multiprocessor_count"
    ]

    regs_per_sm = props[
        "max_num_regs"
    ]

    max_shared = props[
        "max_shared_mem"
    ]

    warp_size = props[
        "warpSize"
    ]

    # PyTorch CUDA device properties
    max_threads_per_sm = (
        torch_props
        .max_threads_per_multi_processor
    )

    print()
    print(
        f"{'max_threads_per_sm':<30}: "
        f"{max_threads_per_sm}"
    )

    # ========================================================
    # Tensors
    # ========================================================

    x = torch.randn(
        M,
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

    bias = torch.randn(
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    z = torch.empty(
        M,
        N,
        device=DEVICE,
        dtype=DTYPE,
    )

    y = torch.empty_like(
        z
    )

    # ========================================================
    # output directory
    # ========================================================

    output_dir = Path(
        "kernel_dumps/c_fc_gelu"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # ========================================================
    # Compile / inspect
    # ========================================================

    for name, cfg in CONFIGS.items():

        print()
        print("=" * 100)
        print(name)
        print("=" * 100)

        compiled = compile_kernel(
            x,
            weight,
            bias,
            z,
            y,
            cfg,
        )

        # ----------------------------------------------------
        # Register / shared memory
        # ----------------------------------------------------

        n_regs = getattr(
            compiled,
            "n_regs",
            None,
        )

        shared = getattr(
            compiled.metadata,
            "shared",
            0,
        )

        threads_per_program = (
            cfg["W"]
            * warp_size
        )

        # ----------------------------------------------------
        # Rough occupancy
        #
        # resident CTAs / SM
        #
        # 실제 HW register allocation granularity 때문에
        # 정확한 profiler occupancy와 약간 다를 수 있음.
        # ----------------------------------------------------

        if (
            n_regs is not None
            and n_regs > 0
        ):
            register_occupancy = (
                regs_per_sm
                // (
                    n_regs
                    * threads_per_program
                )
            )
        else:
            register_occupancy = None

        thread_occupancy = (
            max_threads_per_sm
            // threads_per_program
        )

        if shared > 0:
            shared_occupancy = (
                max_shared
                // shared
            )
        else:
            shared_occupancy = (
                1 << 30
            )

        occupancy_candidates = [
            thread_occupancy,
            shared_occupancy,
        ]

        if register_occupancy is not None:
            occupancy_candidates.append(
                register_occupancy
            )

        occupancy = min(
            occupancy_candidates
        )

        programs_total = (
            num_sm
            * occupancy
        )

        # ----------------------------------------------------
        # ASM
        # ----------------------------------------------------

        ptx = compiled.asm[
            "ptx"
        ]

        try:
            sass = compiled.asm[
                "sass"
            ]
        except Exception as e:
            sass = (
                f"SASS unavailable: "
                f"{type(e).__name__}: {e}"
            )

        # save
        (
            output_dir
            / f"{name}.ptx"
        ).write_text(
            ptx
        )

        (
            output_dir
            / f"{name}.sass"
        ).write_text(
            sass
        )

        # ----------------------------------------------------
        # instruction counts
        # ----------------------------------------------------

        if not sass.startswith(
            "SASS unavailable"
        ):
            hmma = count_instruction(
                sass,
                "HMMA"
            )

            ffma = count_instruction(
                sass,
                "FFMA"
            )

            ldg = count_instruction(
                sass,
                "LDG"
            )

            stg = count_instruction(
                sass,
                "STG"
            )

            lds = count_instruction(
                sass,
                "LDS"
            )

            sts = count_instruction(
                sass,
                "STS"
            )

            bar = count_instruction(
                sass,
                "BAR"
            )
            ldl = count_instruction(
                sass,
                "LDL"
            )

            stl = count_instruction(
                sass,
                "STL"
            )

        else:
            hmma = ffma = None
            ldg = stg = None
            lds = sts = None
            bar = None

        result = {
            "name": name,

            "BM": cfg["BM"],
            "BN": cfg["BN"],
            "BK": cfg["BK"],
            "W": cfg["W"],
            "S": cfg["S"],

            "n_regs": n_regs,
            "shared": shared,

            "reg_occ":
                register_occupancy,

            "smem_occ":
                shared_occupancy,

            "thread_occ":
                thread_occupancy,

            "occupancy":
                occupancy,

            "programs_total":
                programs_total,

            "HMMA": hmma,
            "FFMA": ffma,
            "LDG": ldg,
            "STG": stg,
            "LDS": lds,
            "STS": sts,
            "BAR": bar,
            "LDL": ldl,
            "STL": stl,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"config              : "
            f"BM={cfg['BM']} "
            f"BN={cfg['BN']} "
            f"BK={cfg['BK']} "
            f"W={cfg['W']} "
            f"G={cfg['G']} "
            f"S={cfg['S']}"
        )

        print()

        print(
            f"registers/thread    : "
            f"{n_regs}"
        )

        print(
            f"shared memory/CTA   : "
            f"{shared} bytes"
        )

        print()

        print(
            f"register CTA limit  : "
            f"{register_occupancy}"
        )

        print(
            f"shared CTA limit    : "
            f"{shared_occupancy}"
        )

        print(
            f"thread CTA limit    : "
            f"{thread_occupancy}"
        )

        print()

        print(
            f"estimated CTA/SM    : "
            f"{occupancy}"
        )

        print(
            f"resident programs   : "
            f"{programs_total}"
        )

        print()

        print(
            f"HMMA instructions   : "
            f"{hmma}"
        )

        print(
            f"FFMA instructions   : "
            f"{ffma}"
        )

        print(
            f"LDG                 : "
            f"{ldg}"
        )

        print(
            f"STG                 : "
            f"{stg}"
        )

        print(
            f"LDS                 : "
            f"{lds}"
        )

        print(
            f"STS                 : "
            f"{sts}"
        )

        print(
            f"BAR                 : "
            f"{bar}"
        )
        print(
            f"LDL                 : "
            f"{ldl}"
        )

        print(
            f"STL                 : "
            f"{stl}"
        )

        # ----------------------------------------------------
        # 실제 HMMA 몇 줄 출력
        # ----------------------------------------------------

        if (
            not sass.startswith(
                "SASS unavailable"
            )
        ):
            hmma_lines = (
                get_instruction_lines(
                    sass,
                    "HMMA",
                )
            )

            print()
            print(
                "First HMMA instructions:"
            )

            for line in (
                hmma_lines[:12]
            ):
                print(
                    "  " + line
                )

    # ========================================================
    # Comparison table
    # ========================================================

    print()
    print("=" * 120)
    print("Comparison")
    print("=" * 120)

    print()

    print(
        f"{'Config':<20} "
        f"{'Regs':>6} "
        f"{'SMEM':>8} "
        f"{'RegOcc':>7} "
        f"{'SmemOcc':>8} "
        f"{'CTA/SM':>7} "
        f"{'HMMA':>7} "
        f"{'FFMA':>7} "
        f"{'LDG':>7} "
        f"{'STG':>7}"
        f"{'LDL':>7} "
        f"{'STL':>7} "
    )

    print("-" * 100)

    for r in results:
        print(
            f"{r['name']:<20} "
            f"{str(r['n_regs']):>6} "
            f"{r['shared']:>8} "
            f"{str(r['reg_occ']):>7} "
            f"{r['smem_occ']:>8} "
            f"{r['occupancy']:>7} "
            f"{str(r['HMMA']):>7} "
            f"{str(r['FFMA']):>7} "
            f"{str(r['LDG']):>7} "
            f"{str(r['STG']):>7}"
            f"{str(r['LDL']):>7} "
            f"{str(r['STL']):>7} "
        )

    print()
    local_lines = get_local_memory_lines(
        sass
    )

    print()
    print(
        "Local-memory instructions:"
    )

    for line in local_lines[:30]:
        print(
            "  " + line
        )
    print(
        f"ASM dumps written to: "
        f"{output_dir}"
    )


def get_local_memory_lines(
    sass,
):
    keywords = (
        "LDL",
        "STL",
    )

    return [
        line.strip()
        for line in sass.splitlines()
        if any(
            keyword in line
            for keyword in keywords
        )
    ]

if __name__ == "__main__":
    main()