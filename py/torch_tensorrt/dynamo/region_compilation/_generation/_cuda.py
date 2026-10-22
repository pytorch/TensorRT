# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA C++ proposals compiled through cuda-python, with a fixed tensor ABI.

These structural checks reject unsupported source and launch contracts; they
are not a hostile-code sandbox. NVRTC and the evaluator remain authoritative
for compilation, numerical correctness, and measurement.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec

_FIELDS = {"source", "kernel_name", "grid", "block", "shared_mem_bytes"}
_HEADERS = {"cuda_fp16.h", "cuda_bf16.h", "math.h", "cmath", "stdint.h"}
_PARAMETER = re.compile(
    r"\s*(?P<const>const\s+)?(?P<type>float|half|__half|__nv_bfloat16)"
    r"\s*\*\s*(?:__restrict__\s+)?(?P<name>[A-Za-z_]\w*)\s*"
)
_DTYPES = {
    "float": "float32",
    "half": "float16",
    "__half": "float16",
    "__nv_bfloat16": "bfloat16",
}


def _source_parameters(
    source: str, name: str, num_inputs: int, num_outputs: int
) -> list[re.Match[str]]:
    # Comments are ignored for structural matching, not for compilation. CUDA
    # macros, helper definitions, and host wrappers are outside this first ABI.
    clean = re.sub(r"//[^\r\n]*|/\*.*?\*/", " ", source, flags=re.DOTALL)
    for include in re.findall(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]\s*$", clean, re.M):
        if include not in _HEADERS:
            raise ValueError(f"Generated CUDA source cannot include {include!r}")
    clean = re.sub(r"^\s*#include\s*[<\"][^>\"]+[>\"]\s*$", "", clean, flags=re.M)
    if "#" in clean:
        raise ValueError(
            "Generated CUDA source permits only supported #include directives"
        )
    kernel = re.fullmatch(
        rf'\s*extern\s+"C"\s+__global__\s+void\s+{re.escape(name)}'
        r"\s*\(([^()]*)\)\s*\{(.*)\}\s*",
        clean,
        flags=re.DOTALL,
    )
    if kernel is None or re.search(r"\b(?:__global__|__host__|asm)\b", kernel[2]):
        raise ValueError(
            "Generated CUDA source requires exactly one extern C __global__ kernel"
        )
    parameters = [_PARAMETER.fullmatch(value) for value in kernel[1].split(",")]
    names = [f"arg{i}" for i in range(num_inputs)] + [
        f"out{i}" for i in range(num_outputs)
    ]
    if len(parameters) != len(names) or any(
        parameter is None or parameter["name"] != expected
        for parameter, expected in zip(parameters, names)
    ):
        raise ValueError(
            f"Generated CUDA parameters must be typed tensor pointers {names}"
        )
    checked = [parameter for parameter in parameters if parameter is not None]
    if any(parameter["const"] for parameter in checked[num_inputs:]):
        raise ValueError("Generated CUDA output pointers cannot be const")
    return checked


def validate_cuda_candidate(
    candidate: dict[str, Any], num_inputs: int, num_outputs: int
) -> dict[str, Any]:
    if any(type(count) is not int or count < 1 for count in (num_inputs, num_outputs)):
        raise ValueError("Generated kernels need at least one tensor input and output")
    if not isinstance(candidate, dict) or set(candidate) != _FIELDS:
        raise ValueError(
            f"Generated CUDA candidate must contain exactly {sorted(_FIELDS)}"
        )
    source, name = candidate["source"], candidate["kernel_name"]
    if not isinstance(source, str) or not source.strip() or len(source) > 131072:
        raise ValueError("Generated CUDA source must be nonempty text under 128 KiB")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Generated CUDA kernel_name must be a nonreserved identifier")
    _source_parameters(source, name, num_inputs, num_outputs)
    for field, limits in (
        ("grid", (2147483647, 65535, 65535)),
        ("block", (1024, 1024, 64)),
    ):
        values = candidate[field]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 3
            or any(
                type(value) is not int or not 1 <= value <= limit
                for value, limit in zip(values, limits)
            )
        ):
            raise ValueError(
                f"Generated CUDA {field} must contain valid positive CUDA dimensions"
            )
    if math.prod(candidate["block"]) > 1024:
        raise ValueError("Generated CUDA block cannot exceed 1024 threads")
    shared = candidate["shared_mem_bytes"]
    if type(shared) is not int or not 0 <= shared <= 49152:
        raise ValueError(
            "Generated CUDA shared_mem_bytes must be an integer from 0 to 49152"
        )
    return {
        **candidate,
        "grid": list(candidate["grid"]),
        "block": list(candidate["block"]),
    }


def _validate_ptx(ptx: bytes, name: str, count: int, block: tuple[int, ...]) -> None:
    from torch_tensorrt.kernels._cutile import (
        parse_entry,
        parse_reqntid,
        unsupported_launch_directive,
    )

    text = ptx.decode("utf-8")
    entry, actual_name, parameters = parse_entry(text)
    pointer = re.compile(
        r"\.param\s+\.u64\s+(?:\.ptr\s+(?:\.global\s+)?(?:\.align\s+\d+\s+)?)?[\w$]+"
    )
    if (
        entry is None
        or actual_name != name
        or len(parameters) != count
        or any(pointer.fullmatch(parameter) is None for parameter in parameters)
    ):
        raise ValueError("Generated CUDA PTX does not match the tensor-pointer ABI")
    required = parse_reqntid(text)
    if required is not None and required != block:
        raise ValueError(
            "Generated CUDA block disagrees with the compiled launch requirement"
        )
    if unsupported_launch_directive(text) is not None:
        raise ValueError("Generated CUDA PTX requires an unsupported launch contract")


def make_cuda_registration(
    candidate: dict[str, Any],
    input_specs: list[Any],
    output_specs: list[Any],
    directory: Path,
) -> KernelSpec:
    import tensorrt.plugin as trtp
    import torch

    from torch_tensorrt.kernels import Custom, KernelSpec
    from torch_tensorrt.kernels._nvrtc import compile_to_ptx

    candidate = validate_cuda_candidate(candidate, len(input_specs), len(output_specs))
    parameters = _source_parameters(
        candidate["source"],
        candidate["kernel_name"],
        len(input_specs),
        len(output_specs),
    )
    for parameter, spec in zip(parameters, input_specs + output_specs):
        dtype = spec["dtype"] if isinstance(spec, dict) else spec.dtype
        if str(dtype).removeprefix("torch.") != _DTYPES[parameter["type"]]:
            raise ValueError(
                f"Generated CUDA parameter {parameter['name']} has the wrong boundary dtype"
            )
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(candidate["source"].encode()).hexdigest()
    (directory / f"{digest}.cu").write_text(candidate["source"], encoding="utf-8")
    grid = tuple(candidate["grid"] + [1] * (3 - len(candidate["grid"])))
    block = tuple(candidate["block"] + [1] * (3 - len(candidate["block"])))
    device = (
        input_specs[0]["device"]
        if isinstance(input_specs[0], dict)
        else input_specs[0].device
    )
    with torch.cuda.device(device):
        properties = torch.cuda.get_device_properties(device)
        ptx, _, kernel = compile_to_ptx(
            candidate["source"], candidate["kernel_name"], None, load_kernel=True
        )
        _validate_ptx(ptx, candidate["kernel_name"], len(parameters), block)
        if math.prod(block) > min(
            properties.max_threads_per_block, kernel.attributes.max_threads_per_block
        ):
            raise ValueError(
                "Generated CUDA block exceeds the compiled kernel's thread limit"
            )
        if (
            candidate["shared_mem_bytes"] + kernel.attributes.shared_size_bytes
            > properties.shared_memory_per_block
        ):
            raise ValueError(
                "Generated CUDA kernel exceeds the shared memory launch limit"
            )

    def launch(inputs: Any, outputs: Any, tactic: Any) -> Any:
        params = trtp.KernelLaunchParams()
        params.grid_x, params.grid_y, params.grid_z = grid
        params.block_x, params.block_y, params.block_z = block
        params.shared_mem = candidate["shared_mem_bytes"]
        return params, trtp.SymIntExprs(0)

    return KernelSpec(
        kernel_source=candidate["source"],
        kernel_name=candidate["kernel_name"],
        geometry=Custom(launch),
    )
