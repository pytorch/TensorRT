# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA C++ proposals compiled through cuda-python, with a fixed tensor ABI.

These structural checks reject unsupported source and launch contracts; they
are not a hostile-code sandbox. NVRTC and the evaluator remain authoritative
for compilation, numerical correctness, and measurement.
"""

from __future__ import annotations

import copy
import math
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch

from ._backends import _dimensions, _proposal_schema, _target_metadata
from ._util import sha256_text

# NVRTC ships no C or C++ standard headers; device math needs no include.
_HEADERS = {"cuda_fp16.h", "cuda_bf16.h"}
_SCHEMA = _proposal_schema(
    source={"type": "string"},
    kernel_name={"type": "string"},
    grid=_dimensions(2147483647),
    block=_dimensions(1024),
    shared_mem_bytes={"type": "integer", "minimum": 0, "maximum": 49152},
)
# Literals first, so comment markers inside them do not start comments.
_LEXEME = re.compile(
    r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.DOTALL
)
_DIRECTIVE = re.compile(
    r"^[ \t]*#[ \t]*(?:include[ \t]*[<\"](?P<header>[^>\"\n]*)[>\"]"
    r"|pragma[ \t]+(?:unroll(?:[ \t]+\d+)?|nounroll))[ \t\r]*$",
    re.M,
)
_PARAMETER = re.compile(
    r"\s*(?P<const>const\s+)?(?P<type>float|half|__half|__nv_bfloat16)"
    r"\s*\*\s*(?:__restrict__\s+)?(?P<name>[A-Za-z_]\w*)\s*"
)
_DTYPES = {
    "float": torch.float32,
    "half": torch.float16,
    "__half": torch.float16,
    "__nv_bfloat16": torch.bfloat16,
}
# Compiled kernels by (source digest, kernel name, device), so registration and
# launch in one process compile each candidate once.
_KERNELS: dict[tuple[str, str, torch.device], tuple[bytes, Any]] = {}


def _source_parameters(
    source: str, name: str, num_inputs: int, num_outputs: int
) -> list[re.Match[str]]:
    # Splice lines and drop comments as the compiler does before checking, so
    # neither can hide a directive or an asm statement from the checks below.
    code = _LEXEME.sub(
        lambda lexeme: lexeme[0] if lexeme[0][0] in "\"'" else " ",
        re.sub(r"\\\r?\n", "", source),
    )
    for directive in _DIRECTIVE.finditer(code):
        header = directive["header"]
        if header is not None and header not in _HEADERS:
            raise ValueError(f"Generated CUDA source cannot include {header!r}")
    code = _DIRECTIVE.sub("", code)
    if re.search(r"#|%:|<%|%>", code):
        raise ValueError(
            "Generated CUDA source allows '#' only in supported #include and "
            "#pragma unroll/nounroll lines, and no digraphs"
        )
    kernel = re.match(
        rf'\s*extern\s+"C"\s+__global__\s+void\s+{re.escape(name)}'
        r"\s*\(([^()]*)\)\s*\{",
        code,
    )
    if kernel is None:
        raise ValueError(
            "Generated CUDA source must contain only allowed directives and one "
            f'extern "C" __global__ void {name}(...) kernel'
        )
    rest = code[kernel.end() :]
    if '"' in rest or "'" in rest:
        raise ValueError(
            "Generated CUDA kernels cannot contain string or char literals"
        )
    depth = 1
    for brace in re.finditer(r"[{}]", rest):
        depth += 1 if brace[0] == "{" else -1
        if depth == 0:
            body, tail = rest[: brace.start()], rest[brace.end() :]
            break
    else:
        raise ValueError("Generated CUDA kernel has unbalanced braces")
    if tail.strip():
        raise ValueError("Generated CUDA source cannot contain code after the kernel")
    if re.search(r"\b(?:__global__|__host__|(?:__)?asm(?:__)?)\b", body):
        raise ValueError(
            "Generated CUDA source requires exactly one extern C __global__ kernel "
            "without inline assembly"
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


def _checked(
    candidate: dict[str, Any], num_inputs: int, num_outputs: int
) -> tuple[dict[str, Any], list[re.Match[str]]]:
    """Return the normalized candidate and its parsed kernel parameters."""
    if any(type(count) is not int or count < 1 for count in (num_inputs, num_outputs)):
        raise ValueError("Generated kernels need at least one tensor input and output")
    if not isinstance(candidate, dict) or set(candidate) != set(_SCHEMA["required"]):
        raise ValueError(
            f"Generated CUDA candidate must contain exactly {sorted(_SCHEMA['required'])}"
        )
    source, name = candidate["source"], candidate["kernel_name"]
    if not isinstance(source, str) or not source.strip() or len(source) > 131072:
        raise ValueError("Generated CUDA source must be nonempty text under 128 KiB")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("Generated CUDA kernel_name must be a nonreserved identifier")
    parameters = _source_parameters(source, name, num_inputs, num_outputs)
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
    grid, block = list(candidate["grid"]), list(candidate["block"])
    return {**candidate, "grid": grid, "block": block}, parameters


def _validate_ptx(ptx: bytes, name: str, count: int, block: tuple[int, ...]) -> None:
    from torch_tensorrt.kernels._common import parse_entry, parse_reqntid
    from torch_tensorrt.kernels._cutile import unsupported_launch_directive

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


def _driver_rejection(ptx: bytes, name: str) -> str | None:
    """Why the driver cannot JIT ``ptx``, or None when it loads."""
    from cuda.bindings import driver

    from torch_tensorrt.kernels._common import load_ptx

    error = load_ptx(ptx.decode("utf-8"), name)
    if error == driver.CUresult.CUDA_SUCCESS:
        return None
    return str(getattr(error, "name", error))


def _ptx_kernel(ptx: bytes, name: str) -> Any:
    try:
        from cuda.core import ObjectCode
    except ImportError:  # cuda-core 0.x
        from cuda.core.experimental import ObjectCode
    return ObjectCode.from_ptx(ptx).get_kernel(name)


def _compiled(source: str, name: str, device: torch.device) -> tuple[bytes, Any]:
    """Return PTX and the kernel loaded from that PTX, compiling once per process.

    The eager launch uses the same PTX that the plugin embeds, so the driver's JIT
    rejects in the eager checks exactly what would fail inside the engine.
    """
    key = (sha256_text(source), name, device)
    if key not in _KERNELS:
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        with torch.cuda.device(device):
            ptx, _, _ = compile_to_ptx(source, name, None)
            rejection = _driver_rejection(ptx, name)
            if rejection is not None:
                raise RuntimeError(
                    f"The CUDA driver could not load the PTX that NVRTC produced "
                    f"({rejection})"
                )
            _KERNELS[key] = ptx, _ptx_kernel(ptx, name)
    return _KERNELS[key]


# The trivial kernel compiled to check that the driver loads NVRTC's PTX
_PROBE = 'extern "C" __global__ void torch_tensorrt_probe(float* out) {}'
# Load errors that every PTX from this NVRTC gets from this driver. Any other
# error, such as a full GPU, may not recur, so it is reported but not cached.
_PTX_REJECTIONS = frozenset(
    {
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        "CUDA_ERROR_INVALID_PTX",
        "CUDA_ERROR_NO_BINARY_FOR_GPU",
        "CUDA_ERROR_JIT_COMPILER_NOT_FOUND",
        "CUDA_ERROR_JIT_COMPILATION_DISABLED",
    }
)
# Probe results by device: None when the driver loads NVRTC's PTX
_TOOLCHAIN: dict[torch.device, str | None] = {}


def _toolchain_rejection(device: torch.device) -> str | None:
    if device not in _TOOLCHAIN:
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        with torch.cuda.device(device):
            ptx, _, _ = compile_to_ptx(_PROBE, "torch_tensorrt_probe", None)
            rejection = _driver_rejection(ptx, "torch_tensorrt_probe")
        if rejection is not None and rejection not in _PTX_REJECTIONS:
            raise RuntimeError(
                f"The CUDA driver could not load a probe kernel ({rejection})"
            )
        _TOOLCHAIN[device] = rejection
    return _TOOLCHAIN[device]


def verify_toolchain(device: Any) -> None:
    """Raise RegionError unless the driver loads the PTX that NVRTC produces.

    A driver older than NVRTC rejects every candidate's PTX, so generation
    checks this before it calls the author. Any other failure to load the probe
    raises RuntimeError and is probed again on the next call.
    """
    from cuda.bindings import driver, nvrtc

    from torch_tensorrt.region import RegionError

    rejection = _toolchain_rejection(torch.device(device))
    if rejection is not None:
        _, version = driver.cuDriverGetVersion()
        driver_version = (version // 1000, version % 1000 // 10)
        _, major, minor = nvrtc.nvrtcVersion()
        message = (
            f"CUDA C++ generation needs a driver that loads NVRTC {major}.{minor} "
            f"PTX; the CUDA {driver_version[0]}.{driver_version[1]} driver "
            f"reported {rejection}."
        )
        if (major, minor) > driver_version:
            message += " Install an NVRTC no newer than the driver, or a newer driver."
        raise RegionError(message)


def _kernel_attribute(kernel: Any, name: str) -> int:
    # cuda-core 0.x exposes kernel attributes as methods, 1.x as properties.
    value = getattr(kernel.attributes, name)
    return int(value() if callable(value) else value)


def _dim3(values: list[int]) -> tuple[int, int, int]:
    x, y, z = values + [1] * (3 - len(values))
    return x, y, z


class _CudaBackend:
    name = "cuda_cpp"
    version = "1"
    instruction_resources = ("references/cuda_cpp.md",)

    @property
    def proposal_schema(self) -> dict[str, Any]:
        return copy.deepcopy(_SCHEMA)

    def target_metadata(self, device: Any) -> dict[str, Any]:
        from cuda.bindings import driver, nvrtc

        from torch_tensorrt.kernels._nvrtc import _cuda_core_imports
        from torch_tensorrt.region import RegionError

        _cuda_core_imports()
        status, driver_version = driver.cuDriverGetVersion()
        rtc_status, major, minor = nvrtc.nvrtcVersion()
        if int(status) or int(rtc_status):
            raise RegionError("CUDA C++ generation cannot query the CUDA toolchain")
        verify_toolchain(device)
        return _target_metadata(
            device,
            {"driver": driver_version, "nvrtc": [major, minor]},
            ("cuda-core", "cuda-bindings"),
        )

    def validate_candidate(
        self, proposal: dict[str, Any], num_inputs: int, num_outputs: int
    ) -> dict[str, Any]:
        return _checked(proposal, num_inputs, num_outputs)[0]

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None = None,
    ) -> partial[None]:
        """Register the checked PTX of a CUDA C++ candidate.

        ``options["aligned_io"]`` changes no compilation: CUDA C++ source states
        its own alignment assumptions, which the backend guide ties to it.
        """
        from torch_tensorrt.kernels import ptx_op

        candidate, parameters = _checked(candidate, len(input_specs), len(output_specs))
        for parameter, spec in zip(parameters, input_specs + output_specs):
            if _DTYPES[parameter["type"]] != spec.dtype:
                raise ValueError(
                    f"Generated CUDA parameter {parameter['name']} has the wrong "
                    f"boundary dtype; the captured tensor is {spec.dtype}"
                )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{sha256_text(candidate['source'])}.cu").write_text(
            candidate["source"], encoding="utf-8"
        )
        grid, block = _dim3(candidate["grid"]), _dim3(candidate["block"])
        device = input_specs[0].device
        ptx, kernel = _compiled(candidate["source"], candidate["kernel_name"], device)
        _validate_ptx(ptx, candidate["kernel_name"], len(parameters), block)
        properties = torch.cuda.get_device_properties(device)
        if math.prod(block) > min(
            properties.max_threads_per_block,
            _kernel_attribute(kernel, "max_threads_per_block"),
        ):
            raise ValueError(
                "Generated CUDA block exceeds the compiled kernel's thread limit"
            )
        if (
            candidate["shared_mem_bytes"]
            + _kernel_attribute(kernel, "shared_size_bytes")
            > properties.shared_memory_per_block
        ):
            raise ValueError(
                "Generated CUDA kernel exceeds the shared memory launch limit"
            )

        def launch(inputs: Any, outputs: Any, tactic: Any) -> Any:
            # TensorRT calls this while building an engine, so validation and
            # compilation above do not need QDP support.
            import tensorrt.plugin as trtp

            params = trtp.KernelLaunchParams()
            params.grid_x, params.grid_y, params.grid_z = grid
            params.block_x, params.block_y, params.block_z = block
            params.shared_mem = candidate["shared_mem_bytes"]
            return params, trtp.SymIntExprs(0)

        # Reuse the artifact whose ABI and resource limits were checked above.
        # Returning source in a KernelSpec would compile it again during registration.
        return partial(
            ptx_op,
            ptx=ptx,
            kernel_name=candidate["kernel_name"],
            aot_fn=launch,
        )

    def launch(
        self,
        candidate: dict[str, Any],
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        directory: Path,
    ) -> None:
        from torch_tensorrt.kernels._nvrtc import _cuda_core_imports

        Device, _, _, launch, LaunchConfig = _cuda_core_imports()
        device = inputs[0].device
        _, kernel = _compiled(candidate["source"], candidate["kernel_name"], device)
        config = LaunchConfig(
            grid=_dim3(candidate["grid"]),
            block=_dim3(candidate["block"]),
            shmem_size=candidate["shared_mem_bytes"],
        )
        with torch.cuda.device(device):
            cuda_device = Device(device.index)
            cuda_device.set_current()
            launch(
                cuda_device.create_stream(torch.cuda.current_stream(device)),
                config,
                kernel,
                *(tensor.data_ptr() for tensor in (*inputs, *outputs)),
            )


CUDA_CPP = _CudaBackend()
