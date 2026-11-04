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
from typing import Any, Dict, Iterator, Optional

import torch

logger = logging.getLogger(__name__)

# The engine files of the enclosing keep_offloaded_engines() block, each with its mapping once it is written.
# Holding the mapping keeps its address from naming another tensor until the block ends.
_kept_engines: ContextVar[Optional[Dict[str, Optional[torch.Tensor]]]] = ContextVar(
    "kept_engines", default=None
)


@contextlib.contextmanager
def keep_offloaded_engines() -> Iterator[None]:
    """Keep the engines of the compilations inside the block in files until it ends, then delete them."""
    files: Dict[str, Optional[torch.Tensor]] = {}
    token = _kept_engines.set(files)
    try:
        yield
    finally:
        _kept_engines.reset(token)
        paths = list(files)
        files.clear()
        for path in paths:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)


def keeping_offloaded_engines() -> bool:
    return _kept_engines.get() is not None


def offload_engine(serialized_engine: bytes, name: str) -> torch.Tensor:
    """Write an engine to a kept temporary file and return the file memory-mapped as a uint8 tensor."""
    files = _kept_engines.get()
    assert (
        files is not None
    ), "engines are offloaded only inside keep_offloaded_engines()"
    fd, path = tempfile.mkstemp(suffix=".engine", prefix=f"{name or 'engine'}_")
    files[path] = None  # before the write, so a partly written file is deleted too
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(serialized_engine)
    except OSError as e:
        raise OSError(
            e.errno,
            f"Could not write the engine of {name} to {path} ({e.strerror}). compile_and_save keeps every "
            "engine in the temporary directory until the save; set TMPDIR to a disk with room for them.",
        ) from e
    plan = torch.from_file(
        path, shared=False, size=os.path.getsize(path), dtype=torch.uint8
    )
    files[path] = plan
    logger.debug(f"Offloaded the engine of {name} to {path}")
    return plan


def engine_file_of(tensor: Any) -> Optional[str]:
    """The kept engine file ``tensor`` maps, else None."""
    files = _kept_engines.get()
    if not files or not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        return None
    return next(
        (
            path
            for path, plan in files.items()
            if plan is not None
            and plan.data_ptr() == tensor.data_ptr()
            and plan.numel() == tensor.numel()
        ),
        None,
    )
