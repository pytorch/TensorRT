# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""NVIDIA CuTe DSL proposals compiled to PTX for the pointer-ABI plugin path.

A candidate is one ``@cute.kernel`` taking pointers ``arg0..``, ``out0..`` and
then ``cutlass.Constexpr`` constants. It compiles once per process with the
frontend behind ``kernels.cutedsl_op``, registers that PTX through
``kernels.ptx_op``, and screens by launching the same PTX through cuda.core.
These checks reject unsupported source and launch contracts; they are not a
hostile-code sandbox.
"""

from __future__ import annotations

import ast
import json
import math
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch

from ._backends import _dimensions, _target_metadata
from ._candidate import _aligned_io, _constants, _DslBackend, _dsl_schema
from ._util import sha256_text

_SIGNATURE_DTYPES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
# Call paths rooted at an alias, after ``cutlass.cute.X`` is normalized to ``cute.X``.
_NAMESPACES = frozenset(
    {
        ("cute",),
        ("cute", "arch"),
        ("cute", "math"),
        ("cute", "nvgpu"),
        ("cute", "nvgpu", "warp"),
        ("cute", "nvgpu", "cpasync"),
        ("cutlass",),
    }
)
# Host-only or debugging APIs, and features that the plugin's pointer ABI or a
# non-datacenter GPU cannot provide. TMA names are matched separately so that
# StMatrix ("stmatrix") stays allowed.
_DENIED_WORDS = (
    "tmem",
    "bulk",
    "cluster",
    "dsmem",
    "multicast",
    "warpgroup",
    "tcgen05",
    "umma",
    "reg_alloc",
    "reg_dealloc",
    "printf",
    "print_",
    "compile",
    "jit",
    "kernel",
    "make_ptr",
    "runtime",
    "torch",
    "testing",
    "export",
    "dump",
)
# Methods that would weaken the pointer contract or reach host-only helpers.
_DENIED_METHODS = frozenset(
    {
        "align",
        "to_llvm_ptr",
        "mark_layout_dynamic",
        "mark_compact_shape_dynamic",
        "dump",
    }
)
# Compiled kernels by candidate identity, device and alignment:
# (artifact, signature layout, kernel loaded from the artifact's PTX).
_ARTIFACTS: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
# The pointer alignment each candidate identity was last registered with, so
# that the screening launch runs the PTX its plugin embeds.
_REGISTERED: dict[tuple[Any, ...], bool] = {}
# Devices whose toolchain compiled and loaded the probe kernel.
_TOOLCHAIN: set[torch.device] = set()
_PROBE_SOURCE = """import cutlass
import cutlass.cute as cute


@cute.kernel
def torch_tensorrt_probe(arg0, out0):
    tidx, _, _ = cute.arch.thread_idx()
    out0[tidx] = arg0[tidx]
"""


def _denied(part: str) -> bool:
    lower = part.lower()
    return (
        lower.startswith("tma")
        or "_tma" in lower
        or any(word in lower for word in _DENIED_WORDS)
    )


def _dim3(values: list[int]) -> tuple[int, int, int]:
    x, y, z = list(values) + [1] * (3 - len(values))
    return x, y, z


def _kernel_attribute(kernel: Any, name: str) -> int:
    # cuda-core 0.x exposes kernel attributes as methods, 1.x as properties.
    value = getattr(kernel.attributes, name)
    return int(value() if callable(value) else value)


class _CuteDslBackend(_DslBackend):
    """A pointer-ABI CuTe DSL kernel registered through the shared PTX plugin path."""

    name = "cutedsl"
    schema = _dsl_schema(
        block=_dimensions(1024),
        shared_mem_bytes={"type": "integer", "minimum": 0, "maximum": 49152},
    )
    packages = ("nvidia-cutlass-dsl",)
    imports = frozenset({("cutlass", None), ("cutlass.cute", "cute")})
    decorator = "cute.kernel"
    annotation = "cutlass.Constexpr"
    aliases = frozenset({"cutlass", "cute"})
    # CuTe DSL's rewrite of a kernel tests `and` and `or` operands with type()
    # and passes the variables of loops and branches through locals().
    compiler_builtins = frozenset({"locals", "type"})
    namespaces = _NAMESPACES
    functions = frozenset({("cutlass", "utils", "SmemAllocator")})
    tensor_types = (
        "cutlass.cute.typing.Tensor",
        "cutlass.cute.TensorSSA",
        "cutlass.cute.typing.Layout",
        "cutlass.cute.typing.Pointer",
        "cutlass.cute.typing.Numeric",
        "cutlass.cute.atom.TiledMma",
        "cutlass.cute.atom.ThrMma",  # spellchecker:disable-line
        "cutlass.cute.atom.TiledCopy",
        "cutlass.cute.atom.ThrCopy",  # spellchecker:disable-line
        "cutlass.cute.atom.MmaAtom",
        "cutlass.cute.atom.CopyAtom",
        "cutlass.utils.SmemAllocator",
    )

    def validate_candidate(
        self, proposal: dict[str, Any], num_inputs: int, num_outputs: int
    ) -> dict[str, Any]:
        candidate = super().validate_candidate(proposal, num_inputs, num_outputs)
        block = candidate["block"]
        if (
            not isinstance(block, list)
            or not 1 <= len(block) <= 3
            or any(
                type(value) is not int or not 1 <= value <= limit
                for value, limit in zip(block, (1024, 1024, 64))
            )
            or math.prod(block) > 1024
        ):
            raise ValueError(
                "Generated CuTe DSL block must contain one to three positive CUDA "
                "dimensions with at most 1024 threads and a z dimension of at most 64"
            )
        shared = candidate["shared_mem_bytes"]
        if type(shared) is not int or not 0 <= shared <= 49152:
            raise ValueError(
                "Generated CuTe DSL shared_mem_bytes must be an integer from 0 to 49152"
            )
        return {**candidate, "block": list(block)}

    @property
    def methods(self) -> frozenset[str]:
        return frozenset(super().methods - _DENIED_METHODS)

    def _dsl_path(self, node: ast.Attribute) -> tuple[str, ...] | None:
        qualified = super()._dsl_path(node)
        if qualified is not None and qualified[:2] == ("cutlass", "cute"):
            qualified = ("cute", *qualified[2:])
        return qualified

    def _dsl_name(self, qualified: tuple[str, ...], call: bool) -> bool:
        # The denials hold for every use of a name, not only for calls.
        return not any(_denied(part) for part in qualified) and super()._dsl_name(
            qualified, call
        )

    def _toolchain(self) -> dict[str, Any]:
        from torch_tensorrt.kernels import _cutedsl

        return {"cutlass_dsl": str(_cutedsl._cutedsl_import().cutlass.__version__)}

    def _identity(
        self,
        candidate: dict[str, Any],
        input_dtypes: list[torch.dtype],
        output_dtypes: list[torch.dtype],
        device: torch.device,
    ) -> tuple[Any, ...]:
        return (
            sha256_text(candidate["source"]),
            candidate["kernel_name"],
            json.dumps(_constants(candidate), sort_keys=True),
            tuple(candidate["block"]),
            candidate["shared_mem_bytes"],
            tuple(map(str, input_dtypes)),
            tuple(map(str, output_dtypes)),
            device,
        )

    def _artifact(
        self,
        candidate: dict[str, Any],
        input_dtypes: list[torch.dtype],
        output_dtypes: list[torch.dtype],
        device: torch.device,
        directory: Path,
        aligned_io: bool,
    ) -> tuple[Any, Any, Any]:
        """Compile, check and load a candidate once per process and alignment."""
        identity = self._identity(candidate, input_dtypes, output_dtypes, device)
        key = (*identity, aligned_io)
        if key not in _ARTIFACTS:
            from torch_tensorrt.kernels import _cutedsl

            from ._cuda import _ptx_kernel

            dtypes = [*input_dtypes, *output_dtypes]
            unsupported = [dtype for dtype in dtypes if dtype not in _SIGNATURE_DTYPES]
            if unsupported:
                raise ValueError(
                    "CuTe DSL generation supports float32, float16 and bfloat16 "
                    f"boundaries; got {unsupported[0]}"
                )
            names = [f"arg{i}" for i in range(len(input_dtypes))] + [
                f"out{i}" for i in range(len(output_dtypes))
            ]
            signature = {
                name: "*" + _SIGNATURE_DTYPES[dtype]
                for name, dtype in zip(names, dtypes)
            }
            constants = _constants(candidate)
            kernel = self._kernel(
                candidate, len(input_dtypes), len(output_dtypes), directory
            )
            label = f"generated_cutedsl::{candidate['kernel_name']}"
            layout = _cutedsl.validate_config(
                label,
                kernel,
                signature,
                constants,
                (len(input_dtypes), len(output_dtypes)),
                None,
            )
            # Element alignment holds for every tensor; 16 bytes is the user's
            # promise about deployed buffers, which the evaluator cannot check.
            alignments = _cutedsl.normalize_pointer_alignment(
                label, layout, 16 if aligned_io else None
            )
            with torch.cuda.device(device):
                artifact = _cutedsl.compile_to_ptx(
                    kernel,
                    constants,
                    layout,
                    _dim3(candidate["block"]),
                    candidate["shared_mem_bytes"],
                    alignments,
                )
                loaded = _ptx_kernel(artifact.ptx, artifact.kernel_name)
            _ARTIFACTS[key] = artifact, layout, loaded
        return _ARTIFACTS[key]

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None = None,
    ) -> partial[None]:
        """Register the checked PTX.

        ``options["aligned_io"]`` compiles every pointer as 16-byte aligned.
        """
        from torch_tensorrt.kernels import _common, _cutedsl, ptx_op

        candidate = self.validate_candidate(
            candidate, len(input_specs), len(output_specs)
        )
        aligned_io = _aligned_io(options)
        device = input_specs[0].device
        dtypes = (
            [spec.dtype for spec in input_specs],
            [spec.dtype for spec in output_specs],
        )
        artifact, layout, loaded = self._artifact(
            candidate, *dtypes, device, directory, aligned_io
        )
        _REGISTERED[self._identity(candidate, *dtypes, device)] = aligned_io
        properties = torch.cuda.get_device_properties(device)
        if math.prod(artifact.block) > min(
            properties.max_threads_per_block,
            _kernel_attribute(loaded, "max_threads_per_block"),
        ):
            raise ValueError(
                "Generated CuTe DSL block exceeds the compiled kernel's thread limit"
            )
        static = _kernel_attribute(loaded, "shared_size_bytes")
        if static + artifact.shared_mem > properties.shared_memory_per_block:
            raise ValueError(
                f"Generated CuTe DSL kernel needs {static} bytes of static plus "
                f"{artifact.shared_mem} bytes of dynamic shared memory, but a "
                "TensorRT plugin launch allows at most "
                f"{properties.shared_memory_per_block}; use smaller tiles"
            )
        grid = _dim3(candidate["grid"])
        aot_fn = _common.make_pointer_aot_fn(
            f"generated_cutedsl::{artifact.kernel_name}",
            layout,
            lambda inputs, outputs: grid,
            artifact.block,
            artifact.shared_mem,
            None,
            lambda: _cutedsl.validate_target(artifact),
            frontend="cutedsl_op",
        )
        # Reuse the PTX checked above instead of compiling again in cutedsl_op.
        return partial(
            ptx_op, ptx=artifact.ptx, kernel_name=artifact.kernel_name, aot_fn=aot_fn
        )

    def _registration(
        self,
        kernel: Any,
        candidate: dict[str, Any],
        grid: Callable[..., Any],
        options: dict[str, Any],
    ) -> partial[None]:
        raise NotImplementedError("The CuTe DSL backend overrides make_registration")

    def launch(
        self,
        candidate: dict[str, Any],
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        directory: Path,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Launch the PTX the latest registration embedded.

        Without ``options`` or a registration, the PTX assumes element alignment.
        """
        from torch_tensorrt.kernels._nvrtc import _cuda_core_imports

        device = inputs[0].device
        dtypes = (
            [tensor.dtype for tensor in inputs],
            [tensor.dtype for tensor in outputs],
        )
        aligned_io = (
            _aligned_io(options)
            if options is not None
            else _REGISTERED.get(self._identity(candidate, *dtypes, device), False)
        )
        artifact, _, loaded = self._artifact(
            candidate, *dtypes, device, directory, aligned_io
        )
        Device, _, _, launch, LaunchConfig = _cuda_core_imports()
        config = LaunchConfig(
            grid=_dim3(candidate["grid"]),
            block=artifact.block,
            shmem_size=artifact.shared_mem,
        )
        with torch.cuda.device(device):
            cuda_device = Device(device.index)
            cuda_device.set_current()
            launch(
                cuda_device.create_stream(torch.cuda.current_stream(device)),
                config,
                loaded,
                *(tensor.data_ptr() for tensor in (*inputs, *outputs)),
            )

    def target_metadata(self, device: Any) -> dict[str, Any]:
        from torch_tensorrt.kernels import _cutedsl
        from torch_tensorrt.region import RegionError

        device = torch.device(device)
        try:
            api = _cutedsl._cutedsl_import()
        except ImportError as error:
            raise RegionError(f"CuTe DSL generation is unavailable: {error}") from None
        with torch.cuda.device(device):
            try:
                target = _cutedsl._target_arch(torch.cuda.get_device_capability(device))
                ceiling = _cutedsl._driver_ptx_version()
            except (ImportError, RuntimeError) as error:
                raise RegionError(
                    f"CuTe DSL generation cannot target this GPU: {error}"
                ) from None
            self._verify_toolchain(device)
        return _target_metadata(
            device,
            {
                "cutlass_dsl": str(api.cutlass.__version__),
                "cute_target": target,
                "driver_ptx": ceiling,
            },
            self.packages,
        )

    def _verify_toolchain(self, device: torch.device) -> None:
        """Compile and load a probe kernel before any author call."""
        if device in _TOOLCHAIN:
            return
        from torch_tensorrt.region import RegionError

        probe = {
            "source": _PROBE_SOURCE,
            "kernel_name": "torch_tensorrt_probe",
            "grid": [1],
            "block": [32],
            "shared_mem_bytes": 0,
            "constants": [],
        }
        try:
            with tempfile.TemporaryDirectory(prefix="torch_tensorrt_cutedsl_") as path:
                self._artifact(
                    self.validate_candidate(probe, 1, 1),
                    [torch.float32],
                    [torch.float32],
                    device,
                    Path(path),
                    False,
                )
        except Exception as error:
            raise RegionError(
                f"CuTe DSL generation cannot compile and load a probe kernel: {error}"
            ) from error
        _TOOLCHAIN.add(device)


CUTEDSL = _CuteDslBackend()
