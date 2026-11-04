# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled engines kept in files while ``torch_tensorrt.executorch.compile_and_save`` runs.

Inside ``keep_offloaded_engines()``, ``convert_module`` writes each engine to a temporary file as soon as it is
built, and gives its ``TorchTensorRTModule`` the file memory-mapped as the plan. The engines built so far are
then held neither on the GPU nor in host memory while the next ones build. ``compile_module`` leaves them in their
files, the save streams each file into the program, and the files are deleted when the block ends.
"""

import contextlib
import logging
import os
import tempfile
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)

# The engine files of the enclosing keep_offloaded_engines() block.
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
    """Keep the engines of the compilations inside the block in files until it ends, then delete them."""
    files: List[str] = []
    token = _kept_files.set(files)
    try:
        yield
    finally:
        _kept_files.reset(token)
        for address in [a for a, path in _mapped_files.items() if path in files]:
            del _mapped_files[address]
        for path in files:
            _remove(path)


def keeping_offloaded_engines() -> bool:
    return _kept_files.get() is not None


def offload_engine(serialized_engine: bytes, name: str) -> torch.Tensor:
    """Write an engine to a kept temporary file and return the file memory-mapped as a uint8 tensor."""
    kept = _kept_files.get()
    assert (
        kept is not None
    ), "engines are offloaded only inside keep_offloaded_engines()"
    fd, path = tempfile.mkstemp(suffix=".engine", prefix=f"{name or 'engine'}_")
    kept.append(path)  # before the write, so a partly written file is deleted too
    with os.fdopen(fd, "wb") as f:
        f.write(serialized_engine)
    plan = torch.from_file(
        path, shared=False, size=os.path.getsize(path), dtype=torch.uint8
    )
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
