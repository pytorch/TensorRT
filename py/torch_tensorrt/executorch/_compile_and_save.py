# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile a model with TensorRT and save it as an ExecuTorch program, with no engine in memory."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Set, Union

from torch.export import ExportedProgram
from torch_tensorrt._compile import _EXECUTORCH_SAVE_OPTIONS, save
from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
from torch_tensorrt.dynamo.runtime._offloaded_engines import keep_offloaded_engines


def _keywords(fn: Callable[..., Any]) -> Set[str]:
    return {
        name
        for name, parameter in inspect.signature(fn).parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


_COMPILE_KEYWORDS = _keywords(dynamo_compile) - {"exported_program"}
_SAVE_KEYWORDS = (_keywords(save) | set(_EXECUTORCH_SAVE_OPTIONS)) - {
    "module",
    "file_path",
    "output_format",
}


def compile_and_save(
    exported_program: ExportedProgram,
    file_path: Union[str, "os.PathLike[str]"],
    **kwargs: Any,
) -> None:
    """Compile ``exported_program`` with TensorRT and save it as an ExecuTorch ``.pte``, without holding a
    compiled engine in memory.

    The same program as::

        trt_gm = torch_tensorrt.dynamo.compile(exported_program, ...)
        torch_tensorrt.save(trt_gm, file_path, output_format="executorch", ...)

    but each engine goes to a temporary file as soon as it is built, and is never loaded: the save streams
    the file into the program (or the ``.ptd`` with ``external_engine_data``). Neither compilation nor the
    save holds an engine on the GPU or in host memory. The files go to the system temporary directory
    (``TMPDIR``) and are deleted when the call returns, which returns nothing.

    Each keyword argument goes to ``torch_tensorrt.dynamo.compile`` or ``torch_tensorrt.save``, whichever
    takes it; ``inputs``, ``arg_inputs`` and ``kwarg_inputs`` go to both.
    """
    unknown = sorted(set(kwargs) - _COMPILE_KEYWORDS - _SAVE_KEYWORDS)
    if unknown:
        raise TypeError(
            f"compile_and_save() got unexpected keyword arguments: {unknown}"
        )

    with keep_offloaded_engines():
        trt_gm = dynamo_compile(
            exported_program,
            **{k: v for k, v in kwargs.items() if k in _COMPILE_KEYWORDS},
        )
        save(
            trt_gm,
            os.fspath(file_path),
            output_format="executorch",
            **{k: v for k, v in kwargs.items() if k in _SAVE_KEYWORDS},
        )
