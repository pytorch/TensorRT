# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Whether a pre-export lowering pass needs the graph before Torch-TRT decompositions.

``dynamo.trace`` applies the decomposition table during export, so the passes in
``pre_lowering_pass_list``, which run after it, see a decomposed graph. A pass that
matches an op the table rewrites, such as ``aten.stack``, would find nothing to match
and silently do nothing.

Each pre-export pass therefore declares whether, under given settings, it needs the
graph as ``torch.export`` makes it. If any does, ``dynamo.trace`` exports without
Torch-TRT decompositions and ``compile`` decomposes after the passes have run. A pass
with no declaration counts as needing it, so a forgotten declaration costs a second
trace, never a pass that silently does nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, TypeVar

from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)

_ATTRIBUTE = "_torch_tensorrt_needs_undecomposed_graph"

F = TypeVar("F", bound=Callable[..., Any])


def needs_undecomposed_graph(
    when: Callable[[CompilationSettings], bool],
) -> Callable[[F], F]:
    """Declare that a pre-export lowering pass needs the graph before Torch-TRT
    decompositions whenever ``when(settings)`` is true, typically when the setting
    that enables the pass is on."""

    def mark(lowering_pass: F) -> F:
        setattr(lowering_pass, _ATTRIBUTE, when)
        return lowering_pass

    return mark


def runs_on_decomposed_graph(lowering_pass: F) -> F:
    """Declare that a pre-export lowering pass works on a decomposed graph."""
    return needs_undecomposed_graph(lambda settings: False)(lowering_pass)


def any_needs_undecomposed_graph(
    passes: Iterable[Callable[..., Any]], settings: CompilationSettings
) -> bool:
    """Whether any of ``passes`` needs the graph undecomposed under ``settings``.

    A pass without a declaration logs a warning and counts as needing it.
    """
    for lowering_pass in passes:
        when = getattr(lowering_pass, _ATTRIBUTE, None)
        if when is None:
            logger.warning(
                "Pre-export lowering pass %s does not declare whether it needs the "
                "graph before Torch-TRT decompositions (see needs_undecomposed_graph "
                "and runs_on_decomposed_graph), so dynamo.trace exports without them",
                getattr(lowering_pass, "__name__", lowering_pass),
            )
            return True
        if when(settings):
            return True
    return False
