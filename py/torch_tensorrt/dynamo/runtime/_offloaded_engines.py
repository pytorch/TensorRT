# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TensorRT engines kept in files instead of memory until an ExecuTorch save streams them into the program.

Inside ``offload_engines_to_disk()`` (``torch_tensorrt.compile_and_save``), ``convert_module`` writes each engine to a
file as soon as it is built, and gives its ``TorchTensorRTModule`` the file memory-mapped as the plan. The engines
built so far are then held neither on the GPU nor in host memory while the next ones build. ``compile_module`` leaves
them in their files, not set up. Inside ``offload_engines_for_save()`` (an ExecuTorch ``save()`` given an
``engine_offload_dir``), only the save writes the engines it lowers to files; compilations are unchanged.

Each block makes its own private directory (``tempfile.mkdtemp``: a random ``torch_tensorrt_engines_*`` name, readable
only by its owner) in the folder it is given, or in ``TMPDIR``, and removes it when the block ends, also on an
exception. A process killed during the block cannot remove it.
"""

import contextlib
import dataclasses
import logging
import os
import shutil
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterator, Optional, Union

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _EngineFiles:
    directory: str
    # The setting that moves the directory, named in errors.
    setting: str
    # Each file with its mapping once it is written. Holding the mapping keeps its address from naming another tensor
    # until the block ends.
    files: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


_engine_files: ContextVar[Optional[_EngineFiles]] = ContextVar(
    "offloaded_engine_files", default=None
)
_defer_engine_setup: ContextVar[bool] = ContextVar("defer_engine_setup", default=False)


@contextlib.contextmanager
def _keep_engines_in_files(
    defer_engine_setup: bool, folder: Optional[Union[str, os.PathLike[str]]]
) -> Iterator[None]:
    parent = Path(folder).expanduser().resolve() if folder else None
    setting = "engine_offload_dir" if parent else "engine_offload_dir (or TMPDIR)"
    try:
        if parent is not None:
            # Only the last level, so a mistyped parent fails instead of being created.
            parent.mkdir(mode=0o700, exist_ok=True)
        directory = tempfile.mkdtemp(prefix="torch_tensorrt_engines_", dir=parent)
    except OSError as e:
        raise OSError(
            e.errno,
            f"Could not make a directory for TensorRT engine files in {parent or tempfile.gettempdir()} "
            f"({e.strerror}); set {setting} to a writable folder.",
        ) from e
    block = _EngineFiles(directory, setting)
    files_token = _engine_files.set(block)
    defer_token = _defer_engine_setup.set(defer_engine_setup)
    try:
        yield
    finally:
        _defer_engine_setup.reset(defer_token)
        _engine_files.reset(files_token)
        block.files.clear()
        try:
            shutil.rmtree(block.directory)
        except OSError as e:
            logger.warning(
                f"Could not remove {block.directory}, which holds TensorRT engine files: {e}"
            )


def offload_engines_to_disk(
    folder: Optional[Union[str, os.PathLike[str]]] = None,
) -> ContextManager[None]:
    """Keep the engines of the compilations inside the block in files in ``folder`` (default ``TMPDIR``), not set up,
    until it ends, then delete them."""
    return _keep_engines_in_files(defer_engine_setup=True, folder=folder)


def offload_engines_for_save(
    folder: Union[str, os.PathLike[str]],
) -> ContextManager[None]:
    """Let an ExecuTorch save inside the block write the engines it lowers to files in ``folder``, deleted when the
    block ends."""
    return _keep_engines_in_files(defer_engine_setup=False, folder=folder)


def defer_engine_setup_to_load() -> bool:
    """Whether engines built now stay in their files, not set up, until they are loaded. True inside
    ``offload_engines_to_disk()``."""
    return _defer_engine_setup.get()


def engines_go_to_files() -> bool:
    """Whether ``offload_engine`` has a block to write into: inside either block."""
    return _engine_files.get() is not None


def offload_engine(serialized_engine: Any, name: str) -> torch.Tensor:
    """Write an engine to a file in the block's directory and return the file memory-mapped as a uint8 tensor."""
    block = _engine_files.get()
    assert block is not None, "engines are offloaded only inside an offload block"
    try:
        fd, path = tempfile.mkstemp(
            suffix=".engine", prefix=f"{name or 'engine'}_", dir=block.directory
        )
        with os.fdopen(fd, "wb") as f:
            f.write(serialized_engine)
    except OSError as e:
        raise OSError(
            e.errno,
            f"Could not write the engine of {name} to {block.directory} ({e.strerror}). Every engine stays in a file until "
            f"the program is saved; set {block.setting} to a disk with room for all of them.",
        ) from e
    plan = torch.from_file(
        path, shared=False, size=os.path.getsize(path), dtype=torch.uint8
    )
    block.files[path] = plan
    logger.debug(f"Offloaded the engine of {name} to {path}")
    return plan


def get_offloaded_engine_path(tensor: Any) -> Optional[str]:
    """The path of the offloaded engine file ``tensor`` maps, or None if it maps none."""
    block = _engine_files.get()
    files = block.files if block is not None else {}
    if not files or not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        return None
    return next(
        (
            path
            for path, plan in files.items()
            if plan.data_ptr() == tensor.data_ptr() and plan.numel() == tensor.numel()
        ),
        None,
    )
