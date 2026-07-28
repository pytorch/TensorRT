from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


def _cuda_core_imports() -> Tuple[Any, Any, Any, Any, Any]:
    """Import cuda.core symbols, accepting both the stable and legacy namespaces."""
    try:
        from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

        return Device, Program, ProgramOptions, launch, LaunchConfig
    except ImportError:
        pass
    try:
        from cuda.core.experimental import (
            Device,
            LaunchConfig,
            Program,
            ProgramOptions,
            launch,
        )

        return Device, Program, ProgramOptions, launch, LaunchConfig
    except ImportError:
        raise ImportError(
            "cuda-python is required for cuda_python plugins. "
            "Install it with: pip install cuda-python"
        )


def compile_to_ptx(
    kernel_source: str,
    kernel_name: str,
    include_paths: List[str],
    compile_std: str = "c++17",
    arch_override: Optional[str] = None,
) -> Tuple[bytes, Any, Any]:
    """Compile CUDA C++ source to PTX using NVRTC via cuda-python.

    Returns:
        (ptx_bytes, device, None) — the third slot is reserved for a loadable
        kernel handle but is intentionally not materialized here; see
        ``_derive._compile_kernel`` if you need one (it compiles to CUBIN to
        avoid driver-side PTX JIT).
    """
    Device, Program, ProgramOptions, _launch, _LaunchConfig = _cuda_core_imports()

    device = Device()
    device.set_current()
    arch = arch_override if arch_override else f"sm_{device.arch}"

    options = ProgramOptions(
        std=compile_std,
        arch=arch,
        include_path=include_paths,
    )
    program = Program(kernel_source, code_type="c++", options=options)
    module = program.compile("ptx", name_expressions=(kernel_name,))
    ptx: bytes = module.code
    _LOGGER.debug(
        "Compiled kernel '%s' to PTX for %s (%d bytes)", kernel_name, arch, len(ptx)
    )
    # Materializing a Kernel from the PTX module triggers the driver's PTX JIT,
    # which fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION when the host driver
    # is older than the PTX ISA NVRTC emits. Callers only consume ``ptx``.
    return ptx, device, None
