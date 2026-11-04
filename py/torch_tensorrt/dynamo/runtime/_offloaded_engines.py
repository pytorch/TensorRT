# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled engines kept in files while compilation runs (``offload_compiled_engines_to_disk``).

``convert_module`` writes each engine to a temporary file as soon as it is built, and gives its
``TorchTensorRTModule`` the file memory-mapped as the plan. The engines built so far are then held
neither on the GPU nor in host memory while the next ones build. As with ``lazy_engine_init``,
``compile_module`` sets every engine up once the last one is built, which releases the mapping. The
file is deleted as soon as it is mapped, so its space is freed with the mapping.

``torch_tensorrt.executorch.compile_and_save`` keeps the files instead, and the save streams each one
into the program.
"""

import contextlib
import logging
import os
import tempfile
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)

# The engine files of the enclosing compile_and_save, when compilation must leave them in place.
_kept_files: ContextVar[Optional[List[str]]] = ContextVar(
    "kept_engine_files", default=None
)

# The kept engine files, by the address of their mapping. Lowering hands the ExecuTorch backend another
# tensor object over the same storage, so an attribute would be lost.
_mapped_files: Dict[int, str] = {}


def _remove(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


@contextlib.contextmanager
def keep_offloaded_engines() -> Iterator[None]:
    """Keep the engine files of the compilations inside the block until it ends, then delete them."""
    files: List[str] = []
    token = _kept_files.set(files)
    try:
        yield
    finally:
        _kept_files.reset(token)
        for path in files:
            _remove(path)


def keeping_offloaded_engines() -> bool:
    return _kept_files.get() is not None


def offload_engine(serialized_engine: bytes, name: str) -> torch.Tensor:
    """Write an engine to a temporary file and return the file memory-mapped as a uint8 tensor."""
    fd, path = tempfile.mkstemp(suffix=".engine", prefix=f"{name or 'engine'}_")
    with os.fdopen(fd, "wb") as f:
        f.write(serialized_engine)
    plan = torch.from_file(
        path, shared=False, size=os.path.getsize(path), dtype=torch.uint8
    )
    kept = _kept_files.get()
    if kept is None:
        os.remove(path)  # the mapping keeps the data until it is released
    else:
        kept.append(path)
        _mapped_files[plan.data_ptr()] = path
    logger.debug(f"Offloaded the engine of {name} to {path}")
    return plan


def engine_file_of(tensor: Any) -> Optional[str]:
    """The kept engine file ``tensor`` maps, else None."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        return None
    path = _mapped_files.get(tensor.data_ptr())
    # The size check keeps a reused address from naming a file it did not come from.
    if (
        path is None
        or not os.path.exists(path)
        or os.path.getsize(path) != tensor.numel()
    ):
        return None
    return path
