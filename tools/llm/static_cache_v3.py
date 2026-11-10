# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shape-static KV cache for single-token decode.

``static_cache_v1`` and ``static_cache_v2`` slice the cache to the current
length, so attention runs over a key length that changes every step. TensorRT
resolves those data-dependent shapes on the host: every layer and step costs a
device-to-host copy and a stream synchronization, CPU and GPU work cannot
overlap, and the engine cannot be captured in a CUDA graph.

This pass keeps every shape static for decode (batch 1, one query token):

* each layer's key and value caches are graph inputs of shape
  ``[1, kv_heads, max_seq_len, head_dim]``;
* the new key and value are written at ``position_ids`` with ``index_copy``,
  which TensorRT lowers to an in-place ``IKVCacheUpdateLayer``;
* attention runs over the whole cache with native grouped-query attention,
  instead of Hugging Face's repeated key and value heads, limited to the
  written slots in one of two ways (``configure(..., attention=...)``):

  - ``"mask"``: SDPA with the mask ``arange(max_seq_len) <= position_ids``.
    TensorRT 11.4 runs it with its ``_gemv_mha_v1`` decode kernel for short
    caches (192 slots), which honors the mask, but picks ``_gemv_mha_v2`` for
    long ones (1088 slots and more were measured), which ignores it: the
    unwritten slots then take part in the softmax (zeros dilute it, stale
    entries leak in), whether the mask is Boolean or additive.
  - ``"lengths"``: TensorRT's attention layer (``IAttention``) with
    ``key_value_lengths = position_ids + 1``, through the
    ``static_cache_v3::attention`` operator and its converter below. The
    kernel attends to (and reads) only the written slots, at any length.
  - ``"auto"`` (default): ``"mask"`` up to :data:`MASK_ATTENTION_MAX` slots,
    ``"lengths"`` above. Configuring ``"mask"`` for a longer cache warns.

Export :class:`DecodeStep` (the model with positional ``input_ids`` and
``position_ids``). The compiled module, and the program
``torch_tensorrt.save`` writes for it, take ``(input_ids, position_ids,
key_cache_0, value_cache_0, ...)`` and return ``(logits, key_cache_0,
value_cache_0, ...)``; the engine updates the caches in place, so keep passing
the same cache tensors. Prefill runs one token at a time through the same
engine. Import this module and call :func:`configure` before compiling;
importing registers the lowering pass for every compilation in the process.

Explicit caches: a model that must read its caches in the exported program
itself, such as one whose attention runs in a ``torch_tensorrt.region`` scope
(which only sees tensors that exist at export), takes them as inputs instead.
Export :func:`cache_decode_step`: the model writes each cache with
:func:`write`, a functional ``index_copy`` that TensorRT still lowers to the
in-place ``IKVCacheUpdateLayer`` (the cache is a graph input), and returns the
written caches, so export stays functional, as regions require. The calling
convention is the one above. Such a graph has no SDPA for the pass below,
which leaves it unchanged.
"""

import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tensorrt as trt
import torch
import torch.utils._pytree as pytree
from cache_utils import set_output_spec
from torch._export.utils import _detect_fake_mode_from_gm
from torch.fx import Node
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
    impl,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    SourceIR,
    cast_trt_tensor,
    get_trt_tensor,
)
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

ATTENTION_MODES = ("auto", "mask", "lengths")
# Longest cache that "auto" limits with a position mask: the 192 slots of the
# published 128-token decode results, which TensorRT 11.4 decodes with
# _gemv_mha_v1, a kernel that honors the mask. Its long-cache kernel does not.
MASK_ATTENTION_MAX = 192

SDPA = torch.ops.aten.scaled_dot_product_attention.default
_aten = torch.ops.aten
# Ops through which Hugging Face's repeat_kv expands key and value heads.
_REPEAT_KV_OPS = {
    _aten.expand.default,
    _aten.unsqueeze.default,
    _aten.reshape.default,
    _aten._reshape_copy.default,
    _aten.view.default,
    _aten._unsafe_view.default,
    _aten.clone.default,
    _aten.slice.Tensor,
}

_max_seq_len: Optional[int] = None
_attention = "mask"


@torch.library.custom_op("static_cache_v3::attention", mutates_args=())
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    """Attention of ``query`` over the first ``lengths[b]`` cache slots of batch ``b``.

    ``key`` and ``value`` are ``[batch, kv_heads, slots, head_dim]`` caches;
    the slots from ``lengths[b]`` on are ignored, whatever they hold. In
    TensorRT this is one ``IAttention`` layer with ``key_value_lengths``.
    """
    slots = torch.arange(key.shape[2], device=key.device)
    mask = slots < lengths.reshape(-1, 1, 1, 1)
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=enable_gqa
    )


@attention.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> torch.Tensor:
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


@dynamo_tensorrt_converter(torch.ops.static_cache_v3.attention.default)
def _convert_attention(
    ctx: ConversionContext,
    target: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    name: str,
) -> Any:
    query, key, value, lengths, scale, _ = args
    query = get_trt_tensor(ctx, query, f"{name}_query")
    key = get_trt_tensor(ctx, key, f"{name}_key")
    value = get_trt_tensor(ctx, value, f"{name}_value")
    scale = get_trt_tensor(ctx, scale, f"{name}_scale", query.dtype)
    query = impl.elementwise.mul(
        ctx, target, SourceIR.ATEN, f"{name}_scaled_query", query, scale
    )
    layer = ctx.net.add_attention_v2(
        query,
        key,
        value,
        trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.NONE,
    )
    lengths = get_trt_tensor(ctx, lengths, f"{name}_lengths")
    if lengths.dtype != trt.DataType.INT32:
        lengths = cast_trt_tensor(
            ctx, lengths, trt.DataType.INT32, f"{name}_lengths_int32", target
        )
    layer.key_value_lengths = lengths
    layer.decomposable = True
    return layer.get_output(0)


class DecodeStep(torch.nn.Module):
    """A causal LM as ``(input_ids, position_ids) -> logits``.

    Exporting Hugging Face's ``forward`` with ``position_ids`` as a keyword
    would leave the cache inputs this pass appends out of order with it in a
    saved program's call signature; positional inputs keep them in order.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> Any:
        return self.model(input_ids, position_ids=position_ids).logits


def write(
    cache: torch.Tensor, position_ids: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    """``cache`` with ``update`` written at ``position_ids`` (batch 1, one token).

    Functional (``index_copy``, not ``index_copy_``): with ``cache`` a graph
    input, TensorRT lowers it to an ``IKVCacheUpdateLayer`` that writes the
    cache in place and returns it, so readers of the result read the cache.
    """
    return cache.index_copy(2, position_ids.reshape(-1), update)


def cache_examples(
    config: Any, max_seq_len: int, device: torch.device, dtype: torch.dtype
) -> List[torch.Tensor]:
    """Zero ``key_cache_0, value_cache_0, ...`` tensors for a decoder ``config``."""
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    shape = (1, kv_heads, max_seq_len, head_dim)
    return [
        torch.zeros(shape, dtype=dtype, device=device)
        for _ in range(2 * config.num_hidden_layers)
    ]


class CacheDecodeStep(torch.nn.Module):
    """A causal LM that reads and writes explicit caches (see the module docstring).

    ``model(input_ids, position_ids=..., caches=[...])`` must write every cache
    with :func:`write` and return an object whose ``.logits`` are the logits
    and whose ``.past_key_values`` are the written caches, in input order.
    Construct it with :func:`cache_decode_step`, whose ``forward`` names every
    cache: ``torch_tensorrt.compile`` names inputs after ``forward``'s
    parameters, which ``*caches`` would not give.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: Sequence[torch.Tensor],
    ) -> Any:
        out = self.model(input_ids, position_ids=position_ids, caches=list(caches))
        written = tuple(out.past_key_values)
        if len(written) != len(caches):
            raise ValueError(
                f"The model returned {len(written)} caches for {len(caches)} inputs"
            )
        return (out.logits, *written)


def cache_decode_step(model: torch.nn.Module, layers: int) -> CacheDecodeStep:
    """A :class:`CacheDecodeStep` whose ``forward`` takes ``input_ids,
    position_ids, key_cache_0, value_cache_0, ...`` for ``layers`` layers."""
    names = [f"{kind}_cache_{i}" for i in range(layers) for kind in ("key", "value")]
    listed = ", ".join(names)
    source = (
        f"def forward(self, input_ids, position_ids, {listed}):\n"
        f"    return self.step(input_ids, position_ids, [{listed}])\n"
    )
    namespace: Dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 - a fixed template over generated names
    forward = namespace["forward"]
    name = f"CacheDecodeStep{layers}"
    step_class = type(name, (CacheDecodeStep,), {"forward": forward})
    return step_class(model)


def resolve_attention(max_seq_len: int, attention: str = "auto") -> str:
    """The attention (``"mask"`` or ``"lengths"``) that ``attention`` selects."""
    if attention not in ATTENTION_MODES:
        raise ValueError(
            f"attention must be one of {ATTENTION_MODES}; got {attention!r}"
        )
    if attention == "auto":
        return "mask" if max_seq_len <= MASK_ATTENTION_MAX else "lengths"
    return attention


def configure(max_seq_len: int, attention: str = "auto") -> None:
    """Set the cache length and attention of the next compilations (see above)."""
    global _max_seq_len, _attention
    if not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise ValueError("max_seq_len must be a positive integer")
    resolved = resolve_attention(max_seq_len, attention)
    if resolved == "mask" and max_seq_len > MASK_ATTENTION_MAX:
        warnings.warn(
            f"static_cache_v3: attention='mask' with {max_seq_len} cache slots. "
            "TensorRT 11.4 decodes caches of about 1k slots and more with a "
            "kernel that ignores the mask, so unwritten slots enter the softmax; "
            "use attention='lengths' (or 'auto').",
            stacklevel=2,
        )
    _max_seq_len = max_seq_len
    _attention = resolved


def _grouped_source(node: Node, heads: int) -> Optional[Node]:
    """The key or value before Hugging Face's repeat_kv, if ``node`` comes from it."""
    current: Any = node
    for _ in range(8):
        if not isinstance(current, Node) or current.op != "call_function":
            return None
        if current.target not in _REPEAT_KV_OPS:
            return None
        current = current.args[0]
        value = current.meta.get("val") if isinstance(current, Node) else None
        if (
            isinstance(value, torch.Tensor)
            and value.dim() == 4
            and value.shape[1] < heads
            and heads % value.shape[1] == 0
        ):
            return current
    return None


def _add_cache_input(gm: torch.fx.GraphModule, name: str, value: torch.Tensor) -> Node:
    from cache_utils import _add_graph_input

    return _add_graph_input(gm, name, value)


def _call(
    gm: torch.fx.GraphModule, fake_mode: Any, target: Any, args: Tuple, kwargs: dict
) -> Node:
    node = gm.graph.call_function(target, args, kwargs)

    def fake(argument: Any) -> Any:
        return argument.meta["val"] if isinstance(argument, Node) else argument

    with fake_mode:
        node.meta["val"] = target(
            *pytree.tree_map(fake, args), **pytree.tree_map(fake, kwargs)
        )
    return node


@_aten_lowering_pass
def insert_static_cache_v3(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Give every attention a static KV cache input and output (see the module docstring)."""
    if _max_seq_len is None:
        raise RuntimeError(
            "Call static_cache_v3.configure(max_seq_len) before compiling"
        )
    sdpa_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == SDPA
    ]
    if not sdpa_nodes:
        return gm
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    position_ids = next(
        (node for node in placeholders if node.name == "position_ids"), None
    )
    if position_ids is None:
        raise ValueError("static_cache_v3 needs a 'position_ids' graph input")
    query = sdpa_nodes[0].args[0].meta["val"]
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError(
            "static_cache_v3 compiles single-token decode for batch 1; got a query "
            f"of shape {tuple(query.shape)}"
        )
    fake_mode = _detect_fake_mode_from_gm(gm)
    max_seq_len = _max_seq_len

    caches: List[Tuple[Node, Node, Node, Node, bool]] = []
    for index, sdpa in enumerate(sdpa_nodes):
        q, k, v = sdpa.args[:3]
        heads = q.meta["val"].shape[1]
        k_source, v_source = _grouped_source(k, heads), _grouped_source(v, heads)
        grouped = k_source is not None and v_source is not None
        if not grouped:
            k_source, v_source = k, v
        value = k_source.meta["val"]
        shape = (value.shape[0], value.shape[1], max_seq_len, value.shape[3])
        k_cache = _add_cache_input(
            gm,
            f"key_cache_{index}",
            torch.zeros(shape, dtype=value.dtype, device=value.device),
        )
        v_cache = _add_cache_input(
            gm,
            f"value_cache_{index}",
            torch.zeros(shape, dtype=value.dtype, device=value.device),
        )
        caches.append((k_cache, v_cache, k_source, v_source, grouped))

    # Lowering can leave weight get_attr nodes ahead of the placeholders, so
    # anchor on the node after the last placeholder, not the first non-placeholder.
    last_input = [node for node in gm.graph.nodes if node.op == "placeholder"][-1]
    with gm.graph.inserting_before(last_input.next):
        write_index = _call(
            gm, fake_mode, _aten.reshape.default, (position_ids, [-1]), {}
        )
        if _attention == "lengths":
            # Batch 1, one token: the written slots are 0..position.
            lengths = _call(gm, fake_mode, _aten.add.Tensor, (write_index, 1), {})
        else:
            key_positions = _call(
                gm,
                fake_mode,
                _aten.arange.start_step,
                (0, max_seq_len),
                {"dtype": torch.int64, "device": query.device},
            )
            last_position = _call(
                gm,
                fake_mode,
                _aten.reshape.default,
                (position_ids, [-1, 1, 1, 1]),
                {},
            )
            mask = _call(
                gm, fake_mode, _aten.le.Tensor, (key_positions, last_position), {}
            )

    new_outputs = []
    for sdpa, (k_cache, v_cache, k_source, v_source, grouped) in zip(
        sdpa_nodes, caches
    ):
        with gm.graph.inserting_before(sdpa):
            keys = _call(
                gm,
                fake_mode,
                _aten.index_copy.default,
                (k_cache, 2, write_index, k_source),
                {},
            )
            values = _call(
                gm,
                fake_mode,
                _aten.index_copy.default,
                (v_cache, 2, write_index, v_source),
                {},
            )
            if _attention == "lengths":
                scale = sdpa.kwargs.get("scale")
                if scale is None:
                    scale = 1.0 / math.sqrt(sdpa.args[0].meta["val"].shape[-1])
                attention = _call(
                    gm,
                    fake_mode,
                    torch.ops.static_cache_v3.attention.default,
                    (sdpa.args[0], keys, values, lengths, float(scale), grouped),
                    {},
                )
            else:
                attention = _call(
                    gm,
                    fake_mode,
                    SDPA,
                    (sdpa.args[0], keys, values, mask, 0.0, False),
                    {"scale": sdpa.kwargs.get("scale"), "enable_gqa": grouped},
                )
        sdpa.replace_all_uses_with(attention)
        gm.graph.erase_node(sdpa)
        new_outputs += [keys, values]

    output = next(node for node in gm.graph.nodes if node.op == "output")
    outputs = output.args[0]
    outputs = tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)
    gm.graph.output(outputs + tuple(new_outputs))
    gm.graph.erase_node(output)
    gm = clean_up_graph_after_modifications(gm)
    examples = [
        node.meta["val"] if isinstance(node, Node) else node
        for node in outputs + tuple(new_outputs)
    ]
    set_output_spec(gm, pytree.tree_flatten(tuple(examples))[1])
    logger.debug("After inserting the static v3 KV cache: %s", gm.graph)
    return gm
