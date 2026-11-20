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
import glob
import logging
import os
import tempfile
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional, Sequence, Set, Tuple

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
            try:
                os.remove(path)
            except FileNotFoundError:
                pass  # ExecuTorch deleted it, as the owner of a file it copied or moved.
            except OSError as error:
                # Logged, so it can't replace an error that ended the block.
                logger.warning("Could not delete the engine file %s: %s", path, error)


@contextlib.contextmanager
def moving_engines_into(
    outputs: Sequence[str], sources: Sequence[Tuple[str, int]]
) -> Iterator[None]:
    """Around an ExecuTorch save inside the block that moves the engines into ``outputs``.

    ``sources`` names each file the save may move data from, with the size of that data. If the save
    fails, this deletes each partial output (``<output>.<random>.partial``) it left, and each source
    other than an engine file that it moved in part, such as a file of CUDA weights. Neither can be
    finished: the program is gone and the block deletes the engine files. Afterwards the plan of each
    engine the save moved, in part or in full, is emptied, so a later read of it through its module
    raises instead of killing the process with a bus error.
    """
    files = _offloaded_engine_files.get()
    assert files is not None, "engines move only inside offload_engines_to_disk()"

    def partials() -> Set[str]:
        return {
            p for path in outputs for p in glob.glob(f"{glob.escape(path)}.*.partial")
        }

    earlier = partials()
    try:
        yield
    except BaseException as e:
        moved_in_part = [
            path
            for path, size in sources
            if path not in files and 0 < _size(path) < size
        ]
        deleted = []
        for path in [*sorted(partials() - earlier), *moved_in_part]:
            with contextlib.suppress(OSError):
                os.remove(path)
                deleted.append(path)
        if deleted:
            e.add_note(
                f"Deleted {', '.join(deleted)}: what this save moved can't be finished, since "
                "the files that hold the rest of it are deleted. Export again."
            )
        raise
    finally:
        for path, (plan, size) in _engine_files(files).items():
            if size < plan.numel():
                plan.set_()  # in place, so every module holding the plan sees it empty
                files[path] = None


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _engine_files(
    files: Dict[str, Optional[torch.Tensor]],
) -> Dict[str, Tuple[torch.Tensor, int]]:
    """Each mapped engine file whose size can be read, with its mapping and its size now."""
    found = {}
    for path, plan in files.items():
        if plan is not None:
            with contextlib.suppress(OSError):
                found[path] = plan, os.path.getsize(path)
    return found


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
