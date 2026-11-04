# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled engines kept in files while compilation runs (``offload_compiled_engines_to_disk``).

Each engine goes to a temporary file as soon as it is built, so the engines built so far are not held
on the GPU or the host while the next ones build. When the last one is built, ``compile_module`` loads
each engine from its file into a ``TorchTensorRTModule`` and deletes the file, so compilation returns
ordinary modules. ``torch_tensorrt.executorch.compile_and_save`` keeps the files until the program is
saved instead, and the save streams each file into the program.
"""

import contextlib
import logging
import os
import tempfile
import weakref
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

logger = logging.getLogger(__name__)

# The engine files of the enclosing compile_and_save, when compilation must leave them in place.
_kept_files: ContextVar[Optional[List[str]]] = ContextVar(
    "kept_engine_files", default=None
)

# Engine files mapped by engine_file_tensor, by the address of their mapping. Lowering hands the
# ExecuTorch backend another tensor object over the same storage, so an attribute would be lost.
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


def engine_file_tensor(path: str) -> torch.Tensor:
    """A uint8 tensor mapped from an engine file; ``engine_file_of`` finds the file again."""
    tensor = torch.from_file(
        path, shared=False, size=os.path.getsize(path), dtype=torch.uint8
    )
    _mapped_files[tensor.data_ptr()] = path
    return tensor


def engine_file_of(tensor: Any) -> Optional[str]:
    """The engine file ``tensor`` was mapped from by ``engine_file_tensor``, else None."""
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


class OffloadedTorchTensorRTModule(TorchTensorRTModule):
    """A compiled engine kept in a file until compilation ends.

    Never set up, run or returned by compilation: ``load()`` turns it into a ``TorchTensorRTModule``.
    Its engine, for the ExecuTorch export, is the file memory-mapped. The file is deleted when the module
    is loaded or garbage collected.
    """

    def __init__(self, serialized_engine: bytes, **module_kwargs: Any) -> None:
        super().__init__(**module_kwargs)
        fd, self.engine_file = tempfile.mkstemp(
            suffix=".engine", prefix=f"{module_kwargs.get('name') or 'engine'}_"
        )
        with os.fdopen(fd, "wb") as f:
            f.write(serialized_engine)
        logger.debug(f"Offloaded the engine of {self.name} to {self.engine_file}")
        kept = _kept_files.get()
        if kept is not None:
            kept.append(self.engine_file)
        self._module_kwargs = module_kwargs
        self._outputs_unowned = False
        weakref.finalize(self, _remove, self.engine_file)

    def _plan_buffer(self) -> torch.Tensor:
        return engine_file_tensor(self.engine_file)

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        self._outputs_unowned = enabled

    def are_output_tensors_unowned(self) -> bool:
        return self._outputs_unowned

    def load(self) -> TorchTensorRTModule:
        """This engine as a ``TorchTensorRTModule``, read from the file, which is then deleted."""
        with open(self.engine_file, "rb") as f:
            module = TorchTensorRTModule(
                serialized_engine=f.read(), **self._module_kwargs
            )
        _remove(self.engine_file)
        return module
