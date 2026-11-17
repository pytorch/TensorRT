# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ops that a pre-export lowering pass must see before Torch-TRT decompositions.

``dynamo.trace`` applies the decomposition table during export, so the passes in
``pre_lowering_pass_list``, which run after it, see a decomposed graph. A pass that
matches an op the table rewrites, such as ``aten.stack``, would find nothing to match.

Each pre-export pass therefore declares, for given settings, the ops it needs to see
undecomposed, or ``ALL_OPS`` if it needs the graph as ``torch.export`` makes it.
``dynamo.trace`` leaves those ops out of the table, and ``compile`` decomposes them
after the passes have run. A pass with no declaration counts as ``ALL_OPS``, so
forgetting one costs a second trace, never a pass that silently does nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Literal, TypeVar, Union

from torch._ops import OpOverload
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)

ALL_OPS: Literal["all"] = "all"
UndecomposedOps = Union[frozenset[OpOverload], Literal["all"]]
Declaration = Callable[
    [CompilationSettings], Union[Iterable[OpOverload], Literal["all"]]
]

_ATTRIBUTE = "_torch_tensorrt_undecomposed_ops"

F = TypeVar("F", bound=Callable[..., Any])


def needs_undecomposed_ops(declare: Declaration) -> Callable[[F], F]:
    """Declare which ops a pre-export lowering pass must see undecomposed.

    ``declare`` takes the compilation settings and returns the ops, an empty
    collection if the pass needs none, or ``ALL_OPS``.
    """

    def mark(lowering_pass: F) -> F:
        setattr(lowering_pass, _ATTRIBUTE, declare)
        return lowering_pass

    return mark


def needs_no_undecomposed_ops(lowering_pass: F) -> F:
    """Declare that a pre-export lowering pass works on a decomposed graph."""
    return needs_undecomposed_ops(lambda settings: ())(lowering_pass)


def undecomposed_ops_for(
    passes: Iterable[Callable[..., Any]], settings: CompilationSettings
) -> UndecomposedOps:
    """The union of what ``passes`` declare for ``settings``."""
    ops: set[OpOverload] = set()
    for lowering_pass in passes:
        declare = getattr(lowering_pass, _ATTRIBUTE, None)
        if declare is None:
            logger.warning(
                "Pre-export lowering pass %s does not declare the ops it needs to see "
                "undecomposed (see needs_undecomposed_ops), so dynamo.trace exports "
                "without Torch-TRT decompositions",
                getattr(lowering_pass, "__name__", lowering_pass),
            )
            return ALL_OPS
        declared = declare(settings)
        if declared == ALL_OPS:
            return ALL_OPS
        ops.update(declared)
    return frozenset(ops)
