# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

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


def _default_cuda_include_paths() -> List[str]:
    """Resolve CUDA include dir from CUDA_HOME / CUDA_PATH, else default."""
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            return [os.path.join(root, "include")]
    return ["/usr/local/cuda/include"]


def compile_to_ptx(
    kernel_source: str,
    kernel_name: str,
    include_paths: Optional[Sequence[str]],
    compile_std: str = "c++17",
    arch_override: Optional[str] = None,
    *,
    load_kernel: bool = False,
) -> Tuple[bytes, Any, Any]:
    """Compile CUDA C++ to PTX and optionally load a CUBIN for eager use.

    Returns ``(ptx_bytes, device, kernel)``. ``kernel`` is None unless
    ``load_kernel=True``; its separate CUBIN compilation avoids driver PTX JIT.
    """
    Device, Program, ProgramOptions, _launch, _LaunchConfig = _cuda_core_imports()

    device = Device()
    device.set_current()
    arch = arch_override if arch_override else f"sm_{device.arch}"

    options = ProgramOptions(
        std=compile_std,
        arch=arch,
        include_path=(
            list(include_paths)
            if include_paths is not None
            else _default_cuda_include_paths()
        ),
    )
    program = Program(kernel_source, code_type="c++", options=options)
    module = program.compile("ptx", name_expressions=(kernel_name,))
    ptx: bytes = module.code
    _LOGGER.debug(
        "Compiled kernel '%s' to PTX for %s (%d bytes)", kernel_name, arch, len(ptx)
    )
    kernel = None
    if load_kernel:
        # Loading PTX invokes the driver's JIT, which may reject NVRTC's newer
        # PTX ISA. Load CUBIN instead, using a fresh Program because NVRTC does
        # not accept name_expressions after a Program has already compiled.
        cubin_program = Program(kernel_source, code_type="c++", options=options)
        cubin = cubin_program.compile("cubin", name_expressions=(kernel_name,))
        kernel = cubin.get_kernel(kernel_name)
    return ptx, device, kernel
