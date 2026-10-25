# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import sympy
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import map_arg
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

# A mask is checked by computing it on the CPU at the sequence lengths it is
# compiled for, up to this one.
_MAX_CHECKED_SEQ_LEN = 8192
# ... and only if no tensor it is computed from is larger than this.
_MAX_CHECKED_NUMEL = _MAX_CHECKED_SEQ_LEN**2
# An additive bias at or below this masks a position out.
_MASKED_OUT_BIAS = -1e4

_AND_OPS = (
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.logical_and.default,
)
# Ops between a mask and the attention call that leave its values in place
# (the last two dims remain the query and key positions).
_BROADCAST_OPS = (
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.alias.default,
)


# Symbolic size -> the value (a sympy Integer) to compute a mask at.
_Sizes = Dict[Any, Any]


class _Unknown(Exception):
    """The mask cannot be computed at compile time."""


def _is_size(node: torch.fx.Node) -> bool:
    """A non-input node whose value is a (symbolic) integer, such as ``sym_size``
    and arithmetic on it: computed from its shape expression, not by running it."""
    val = node.meta.get("val")
    return (
        node.op != "placeholder"
        and isinstance(val, (torch.SymInt, int))
        and not isinstance(val, bool)
    )


def _cone(node: torch.fx.Node) -> Set[torch.fx.Node]:
    """``node`` and everything it is computed from, with sizes as leaves."""
    seen: Set[torch.fx.Node] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if not _is_size(n):
            stack.extend(n.all_input_nodes)
    return seen


class _MaskChecker:
    """Computes attention masks at compile time to tell whether they are causal."""

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.gm = gm
        # Symbols standing for the values of scalar inputs, not for shapes.
        self.input_symbols: Set[Any] = set()
        for n in gm.graph.nodes:
            val = n.meta.get("val")
            if n.op == "placeholder" and isinstance(val, torch.SymInt):
                self.input_symbols |= val.node.expr.free_symbols

    def reads_inputs(self, node: torch.fx.Node) -> bool:
        """Whether the value of a graph input, and not only shapes, feeds ``node``."""
        for n in _cone(node):
            if n.op == "placeholder":
                return True
            val = n.meta.get("val")
            if (
                _is_size(n)
                and isinstance(val, torch.SymInt)
                and val.node.expr.free_symbols & self.input_symbols
            ):
                return True
        return False

    def evaluate(self, node: torch.fx.Node, sizes: _Sizes) -> Any:
        """Run the part of the graph that computes ``node`` on the CPU, with the
        symbolic sizes set to ``sizes``."""
        cone = _cone(node)
        for n in cone:
            val = n.meta.get("val")
            if (
                isinstance(val, torch.Tensor)
                and n.target not in _BROADCAST_OPS
                and math.prod(_size(s, sizes) for s in val.shape) > _MAX_CHECKED_NUMEL
            ):
                raise _Unknown(f"{n} is too large to compute")
        env: Dict[torch.fx.Node, Any] = {}
        cpu = torch.device("cpu")
        for n in self.gm.graph.nodes:
            if n not in cone:
                continue
            if _is_size(n):
                env[n] = _size(n.meta["val"], sizes)
            elif n.op == "get_attr":
                value = functools.reduce(getattr, n.target.split("."), self.gm)
                env[n] = value.to(cpu) if isinstance(value, torch.Tensor) else value
            elif n.op == "call_function":
                args, kwargs = map_arg((n.args, n.kwargs), lambda a: env[a])
                if "device" in kwargs:
                    kwargs = {**kwargs, "device": cpu}
                env[n] = n.target(*args, **kwargs)
            else:
                raise _Unknown(f"cannot run {n.op} {n.target}")
        return env[node]

    def check(self, bias: torch.fx.Node, seq_len: Any) -> Optional[bool]:
        """Whether ``bias`` is the causal mask for every sequence length it is
        compiled for: True or False when that can be computed at compile time;
        None when the mask depends on runtime inputs, or on lengths too long to
        compute it at.

        A mask that reads runtime inputs (say, a padding mask) can still be shown
        not to be causal: when it is an AND of boolean masks, and the ones that
        can be computed already mask out a causal position, as a sliding window
        does."""
        core = bias
        while core.op == "call_function" and core.target in _BROADCAST_OPS:
            core = core.args[0]
        if not self.reads_inputs(core):
            return self._check_exact(core, seq_len)
        if core.op == "call_function" and core.target == torch.ops.aten.where.self:
            branches = core.args[1:]
            if any(
                isinstance(b, torch.fx.Node) and self.reads_inputs(b) for b in branches
            ):
                return None
            # where(mask, 0, -inf) is the additive form of a boolean mask.
            allowed_bias, masked_bias = (
                self.evaluate(b, {}) if isinstance(b, torch.fx.Node) else b
                for b in branches
            )
            if not (
                bool(torch.as_tensor(allowed_bias == 0).all())
                and bool(torch.as_tensor(masked_bias <= _MASKED_OUT_BIAS).all())
            ):
                return None
            core = core.args[0]
            while core.op == "call_function" and core.target in _BROADCAST_OPS:
                core = core.args[0]
        terms = _and_terms(core)
        known = [t for t in terms if not self.reads_inputs(t)]
        if not known:
            return None
        for sizes, _ in self._sizes(known, seq_len):
            n = _size(seq_len, sizes)
            values = [self.evaluate(t, sizes) for t in known]
            if not all(
                isinstance(v, torch.Tensor) and v.dtype == torch.bool for v in values
            ):
                return None
            allowed = functools.reduce(torch.logical_and, values)
            if bool((~allowed & _causal(n)).any()):
                return False
        return None

    def _check_exact(self, mask: torch.fx.Node, seq_len: Any) -> Optional[bool]:
        complete = False
        for sizes, exhaustive in self._sizes([mask], seq_len):
            n = _size(seq_len, sizes)
            allowed = _allowed(self.evaluate(mask, sizes))
            if allowed is None or not bool((allowed == _causal(n)).all()):
                return False
            complete = complete or exhaustive
        return True if complete else None

    def _sizes(
        self, nodes: List[torch.fx.Node], seq_len: Any
    ) -> List[Tuple[_Sizes, bool]]:
        """Values for the symbolic sizes to compute masks at: their example
        values, and the sequence length at its upper bound (at most
        ``_MAX_CHECKED_SEQ_LEN``). The flag says whether checking at these sizes
        covers every length the graph is compiled for."""
        symbols: Set[Any] = set()
        shape_env = None
        sizes = [n.meta["val"] for node in nodes for n in _cone(node) if _is_size(n)]
        for size in sizes + [seq_len]:
            if isinstance(size, torch.SymInt):
                symbols |= size.node.expr.free_symbols
                shape_env = size.node.shape_env
        if shape_env is None:
            if seq_len > _MAX_CHECKED_SEQ_LEN:
                raise _Unknown(f"sequence length {seq_len} is too long to check")
            return [({}, True)]
        try:
            hints = {s: sympy.Integer(shape_env.var_to_val[s]) for s in symbols}
        except KeyError as e:
            raise _Unknown(f"no example value for {e}")
        result = []
        if _size(seq_len, hints) <= _MAX_CHECKED_SEQ_LEN:
            # A static sequence length is covered by its one value.
            result.append((hints, not isinstance(seq_len, torch.SymInt)))
        expr = seq_len.node.expr if isinstance(seq_len, torch.SymInt) else None
        if expr in symbols:
            upper = shape_env.var_to_range[expr].upper
            covers_all = upper.is_finite and int(upper) <= _MAX_CHECKED_SEQ_LEN
            longest = int(upper) if covers_all else _MAX_CHECKED_SEQ_LEN
            result.append(({**hints, expr: sympy.Integer(longest)}, covers_all))
        if not result:
            raise _Unknown("sequence lengths too long to check")
        return result


def _size(size: Any, sizes: _Sizes) -> int:
    if isinstance(size, int):
        return size
    value = size.node.expr.xreplace(sizes)
    if not value.is_number:
        raise _Unknown(f"no value for {size}")
    return int(value)


def _causal(seq_len: int) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool).tril()


def _allowed(mask: Any) -> Optional[torch.Tensor]:
    """The positions a mask lets through, or None if it also changes their scores
    (an additive bias other than 0 / masked out, such as ALiBi)."""
    if not isinstance(mask, torch.Tensor):
        return None
    if mask.dtype == torch.bool:
        return mask
    if not mask.is_floating_point():
        return None
    allowed = mask > _MASKED_OUT_BIAS
    return None if bool(mask.masked_fill(~allowed, 0).ne(0).any()) else allowed


def _and_terms(node: torch.fx.Node) -> List[torch.fx.Node]:
    if node.op == "call_function" and node.target in _AND_OPS:
        return _and_terms(node.args[0]) + _and_terms(node.args[1])
    return [node]


def force_causal_efficient_attention(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Force efficient-attention calls to causal mode when enabled in settings.

    For square attention (seq_q == seq_k) whose attn_bias is a causal mask:
    replaces attn_bias with is_causal=True so IAttentionLayer can use its native
    causal path.

    The mask is computed at compile time when it can be -- at the example
    sequence length and at the longest one compiled for -- and a mask that is not
    causal (a sliding window, a bidirectional or padding mask built in the graph,
    an additive bias such as ALiBi) is left in place. A mask that depends on
    runtime inputs cannot be checked: it is assumed causal, as
    ``attn_bias_is_causal`` promises, and a warning names it.

    For decode-phase attention (seq_q != seq_k): skip the transformation.
    Applying is_causal=True is semantically wrong because it creates a lower-triangular
    mask aligned to position 0, so the query attends only to k[0] instead of all past keys.
    The node is left unchanged and passed to IAttentionLayer, which supports non-square Q/K natively.
    """
    if not settings.attn_bias_is_causal:
        return gm

    changed = False
    checker = _MaskChecker(gm)
    verdicts: Dict[torch.fx.Node, Optional[bool]] = {}
    unverified: List[str] = []
    for node in gm.graph.nodes:
        if (
            node.target
            != torch.ops.aten._scaled_dot_product_efficient_attention.default
        ):
            continue

        attn_bias = node.args[3] if len(node.args) > 3 else None
        if attn_bias is None:
            continue

        query_node, key_node = node.args[0], node.args[1]
        query_meta = query_node.meta.get("val") if hasattr(query_node, "meta") else None
        key_meta = key_node.meta.get("val") if hasattr(key_node, "meta") else None
        if (
            query_meta is not None
            and key_meta is not None
            and query_meta.size(-2) != key_meta.size(-2)
        ):
            logger.debug(
                f"Skipping forcing causal pass for node {node} because seq_q={query_meta.size(-2)} != seq_k={key_meta.size(-2)} (decode-phase, IAttentionLayer handles it)"
            )
            continue

        verdict = None
        if isinstance(attn_bias, torch.fx.Node) and query_meta is not None:
            if attn_bias not in verdicts:
                try:
                    # Real tensors, also when the graph is lowered under a fake
                    # mode (torch.compile).
                    with unset_fake_temporarily():
                        verdicts[attn_bias] = checker.check(
                            attn_bias, query_meta.size(-2)
                        )
                except Exception as e:  # whatever the mask is computed with
                    logger.debug(
                        f"Could not compute the attention mask {attn_bias}: {e}"
                    )
                    verdicts[attn_bias] = None
            verdict = verdicts[attn_bias]
        if verdict is False:
            logger.info(
                f"Keeping the attention mask of {node}: it is not causal (a sliding "
                "window or a bidirectional mask, say), so is_causal=True cannot "
                "replace it"
            )
            continue
        if verdict is None:
            unverified.append(node.name)

        node.args = (
            node.args[0],
            node.args[1],
            node.args[2],
            None,
            False,
            0.0,
            True,
        )
        changed = True
        logger.debug(f"Node {node} changed to causal mode: {node.args}")

    if unverified:
        logger.warning(
            f"attn_bias_is_causal=True: replaced the attention mask of "
            f"{len(unverified)} attention call(s) ({', '.join(unverified[:3])}"
            f"{', ...' if len(unverified) > 3 else ''}) with causal masking without "
            "being able to check that it is causal, since it depends on runtime "
            f"inputs or on sequences longer than {_MAX_CHECKED_SEQ_LEN}. If the model "
            "uses sliding-window, padding or bidirectional masks, compile with "
            "attn_bias_is_causal=False."
        )

    if changed:
        gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"After forcing causal efficient attention pass:\n{gm.graph}")
    return gm
