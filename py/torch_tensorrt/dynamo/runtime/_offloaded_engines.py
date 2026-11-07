# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled engines kept on disk while ``torch_tensorrt.compile_and_save`` runs.

Inside ``offload_engines_to_disk()``, ``convert_module`` writes each engine to a file in Torch-TensorRT's cache
directory as soon as it is built, and gives its ``TorchTensorRTModule`` the file memory-mapped as the plan. The
engines built so far are then held neither on the GPU nor in host memory while the next ones build.
``compile_module`` leaves them in their files: an ExecuTorch save streams each file into the program, and
``compile_and_save`` loads them before any other save. The files are deleted when the block ends.
"""

import contextlib
import logging
import os
import tempfile
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

import torch
from torch_tensorrt._utils import _cache_root

logger = logging.getLogger(__name__)

# The engine files of the enclosing offload_engines_to_disk() block, each with its mapping once it is written.
# Holding the mapping keeps its address from naming another tensor until the block ends.
_offloaded_engine_files: ContextVar[Optional[Dict[str, Optional[torch.Tensor]]]] = (
    ContextVar("offloaded_engine_files", default=None)
)


@contextlib.contextmanager
def offload_engines_to_disk() -> Iterator[None]:
    """Keep the engines of the compilations inside the block in files until it ends, then delete them."""
    files: Dict[str, Optional[torch.Tensor]] = {}
    token = _offloaded_engine_files.set(files)
    try:
        yield
    finally:
        _offloaded_engine_files.reset(token)
        paths = list(files)
        files.clear()
        for path in paths:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)


def defer_engine_setup_to_load() -> bool:
    """Whether engines built now stay in their files, not set up, until they are loaded. True inside
    ``offload_engines_to_disk()``."""
    return _offloaded_engine_files.get() is not None


def offload_engine(serialized_engine: bytes, name: str) -> torch.Tensor:
    """Write an engine to a file in the cache directory and return the file memory-mapped as a uint8 tensor."""
    files = _offloaded_engine_files.get()
    assert (
        files is not None
    ), "engines are offloaded only inside offload_engines_to_disk()"
    # Not the engine cache's directory: its eviction removes whatever it finds there.
    directory = _cache_root() / "offloaded_engines"
    directory.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(
        suffix=".engine", prefix=f"{name or 'engine'}_", dir=directory
    )
    files[path] = None  # before the write, so a partly written file is deleted too
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(serialized_engine)
    except OSError as e:
        raise OSError(
            e.errno,
            f"Could not write the engine of {name} to {path} ({e.strerror}). compile_and_save keeps every "
            "engine on disk until the save; set TMPDIR to a disk with room for them.",
        ) from e
    plan = torch.from_file(
        path, shared=False, size=os.path.getsize(path), dtype=torch.uint8
    )
    files[path] = plan
    logger.debug(f"Offloaded the engine of {name} to {path}")
    return plan


def get_offloaded_engine_path(tensor: Any) -> Optional[str]:
    """The path of the offloaded engine file ``tensor`` maps, or None if it maps none."""
    files = _offloaded_engine_files.get()
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
