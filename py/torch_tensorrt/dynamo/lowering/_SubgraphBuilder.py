"""Cursor-based FX graph node builder."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import List, Optional, Type

import torch
import torch.fx
from torch.fx.node import Node

logger = logging.getLogger(__name__)


def _fmt_node(n: object) -> str:
    """Return a compact string summary of an FX node or a plain value."""
    if not isinstance(n, Node):
        return repr(n)
    val = n.meta.get("val", None)
    if val is not None and hasattr(val, "shape") and hasattr(val, "dtype"):
        return f"%{n.name}[{tuple(val.shape)},{val.dtype}]"
    return f"%{n.name}"


def _fmt_args(args: tuple) -> str:
    return "(" + ", ".join(_fmt_node(a) for a in args) + ")"


# NB: Its pretty tedious to go through and hand write all the graph insert afters
# Could not find a Pytorch utility that simplifies this so we have this class. I want
# remove it if we find a PyTorch alternative
class SubgraphBuilder:
    """Cursor-based helper for inserting a sequence of FX ``call_function`` nodes.

    Construct it with the graph and an anchor node, then call it like a
    function to append each new node immediately after the current cursor::

        with SubgraphBuilder(graph, node) as b:
            re = b(aten.select.int, inp, -1, 0)
            im = b(aten.select.int, inp, -1, 1)
            out = b(aten.add.Tensor, re, im)

    Each call inserts one ``call_function`` node right after the cursor and
    advances the cursor to that node.  Scalar / list arguments are forwarded
    as-is.

    On ``__exit__`` the graph is linted to catch any malformed nodes inserted
    during the block.  Exceptions from user code propagate normally; lint
    errors are only raised when the block itself succeeds.
    """

    __slots__ = ("_g", "_anchor_desc", "_cursor", "_inserted")

    def __init__(self, graph: torch.fx.Graph, cursor: Node) -> None:
        self._g = graph
        # Snapshot the description now — the anchor node is erased inside the block.
        self._anchor_desc: str = _fmt_node(cursor)
        self._cursor = cursor
        self._inserted: List[Node] = []

    @property
    def cursor(self) -> Node:
        return self._cursor

    def __call__(self, op: object, *args: object) -> Node:
        with self._g.inserting_after(self._cursor):
            node = self._g.call_function(op, args=args)
        self._cursor = node
        self._inserted.append(node)
        return node

    def __enter__(self) -> "SubgraphBuilder":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is None:
            if logger.isEnabledFor(logging.DEBUG) and self._inserted:
                lines = [f"  rewrite  {self._anchor_desc}  ->"]
                for n in self._inserted:
                    op_name = getattr(n.target, "__name__", str(n.target))
                    lines.append(f"    {_fmt_node(n)} = {op_name}{_fmt_args(n.args)}")
                logger.debug("\n".join(lines))
            self._g.lint()
