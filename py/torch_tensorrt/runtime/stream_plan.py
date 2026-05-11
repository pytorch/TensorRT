"""
Graph-level stream management for compiled Torch-TensorRT modules.

apply_stream_plan() / stream_plan() take a partitioned GraphModule produced
by torchtrt.compile() and insert explicit stream-switching and barrier nodes
derived from the DAG structure:

  - enter_compute_stream  — conditional fork at graph entry
  - set_stream            — switches current stream at each TRT subgraph boundary
  - sync_streams          — event-based barrier at every cross-stream data edge
  - exit_compute_stream   — restores caller's original stream at graph exit

All four ops are side-effectful so DCE cannot remove them.

The original GraphModule is never mutated: apply_stream_plan returns a new
GraphModule that shares TRT engine submodules (no re-build) but owns a new
graph.  stream_plan() is the RAII context-manager variant that releases
stream attributes on exit.

See docsrc/contributors/stream_management.rst for the full design.
"""

from __future__ import annotations

import copy
import logging
import operator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch
import torch.fx

logger = logging.getLogger(__name__)

# Import to trigger fake/effect registration for the four stream control ops
# and to bring call_trt_with_token into scope.
from torch_tensorrt.dynamo.runtime._stream_ops import call_trt_with_token  # noqa: F401


class StreamPlanError(ValueError):
    pass


@dataclass
class StreamPlan:
    """
    Resolved assignment of compiled subgraph nodes to streams.
    Built by _resolve_plan(); not constructed directly by callers.
    """

    assignment: dict[str, torch.cuda.Stream] = field(default_factory=dict)
    device: Optional[torch.device] = None


# ── Public API ────────────────────────────────────────────────────────────────


@contextmanager
def stream_plan(
    gm: torch.fx.GraphModule,
    streams: Optional[list[torch.cuda.Stream]] = None,
    hints: Optional[dict[str, torch.cuda.Stream]] = None,
) -> Iterator[torch.fx.GraphModule]:
    """
    RAII context manager.  Yields a new GraphModule with stream management
    baked in.  The original module is untouched.  Stream attributes are
    released on exit.

    Args:
        gm:      Compiled GraphModule from torchtrt.compile().
        streams: One stream per TRT subgraph in graph-traversal order.
                 Plain torch.cuda.Stream() created for any omitted entries.
        hints:   {submodule_name: stream} — overrides by name.  Names come
                 from inspecting model.named_modules() after compile().
                 Merged with positional streams; conflicts raise StreamPlanError.

    Raises:
        StreamPlanError: unknown node name, stream count mismatch, device
                         mismatch, conflicting assignments, or CUDA graphs active.
    """
    planned = apply_stream_plan(gm, streams, hints)
    try:
        yield planned
    finally:
        _release(planned)


def apply_stream_plan(
    gm: torch.fx.GraphModule,
    streams: Optional[list[torch.cuda.Stream]] = None,
    hints: Optional[dict[str, torch.cuda.Stream]] = None,
) -> torch.fx.GraphModule:
    """
    Permanent (non-RAII) variant of stream_plan().
    Returns a new GraphModule with stream management baked in.
    The original module is untouched.

    Works on both fresh-compile graphs (TRT call_module nodes) and post-serde
    graphs (call_function execute_engine nodes); the call_module form is
    inlined to execute_engine form before the plan is applied.
    """
    # Mutating inlining must happen on a copy so we don't touch the user's gm.
    planned = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    _inline_trt_call_modules(planned)
    plan = _resolve_plan(planned, streams, hints)
    return _apply_stream_plan(planned, plan)


# ── Plan resolution ───────────────────────────────────────────────────────────


def _is_trt_module(mod: torch.nn.Module) -> bool:
    try:
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        return isinstance(mod, TorchTensorRTModule)
    except ImportError:
        return False


def _is_execute_engine_node(n: torch.fx.Node) -> bool:
    """Return True if n is a call_function to torch.ops.tensorrt.execute_engine.

    This is the post-inline form produced by save+load via the dynamo exporter.
    """
    return (
        n.op == "call_function"
        and getattr(n.target, "_overloadpacket", None)
        is torch.ops.tensorrt.execute_engine
    )


def _trt_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Find TRT compute nodes in either form:
    - call_module → TorchTensorRTModule (fresh compile output)
    - call_function(torch.ops.tensorrt.execute_engine) (post-serde form)
    """
    out = []
    for n in gm.graph.nodes:
        if n.op == "call_module" and _is_trt_module(gm.get_submodule(n.target)):
            out.append(n)
        elif _is_execute_engine_node(n):
            out.append(n)
    return out


def _trt_node_key(n: torch.fx.Node) -> str:
    """Stable plan-assignment key for a TRT node, matched to the current form.

    For call_module form: the submodule name (e.g. ``_run_on_acc_0``).
    For execute_engine form: the engine attribute name (e.g.
    ``_run_on_acc_0_engine``).

    Hints accept both spellings — see ``_normalize_hint_name``.
    """
    if n.op == "call_module":
        return str(n.target)
    return str(n.args[1].target)  # engine_get_attr.target


def _engine_for_node(gm: torch.fx.GraphModule, n: torch.fx.Node) -> Any:
    """Return the engine object associated with a TRT node (TorchBind or mock).

    For call_module form: returns ``submod.engine`` of the TRT submodule.
    For execute_engine form: resolves the get_attr arg to the engine attribute.
    """
    if n.op == "call_module":
        submod = gm.get_submodule(n.target)
        return getattr(submod, "engine", None)
    engine_attr_name = n.args[1].target
    return getattr(gm, engine_attr_name, None)


def _engine_device_id(gm: torch.fx.GraphModule, n: torch.fx.Node) -> int:
    """Return the CUDA device index for the TRT node n.

    Handles both call_module (TorchTensorRTModule) and call_function
    (TorchBind Engine) forms, plus the mock-engine fallback used by tests.
    """
    if n.op == "call_module":
        submod = gm.get_submodule(n.target)
        if hasattr(submod, "settings") and hasattr(submod.settings, "device"):
            return int(submod.settings.device.gpu_id)
        try:
            return int(submod.engine.device_info.id)
        except AttributeError:
            return 0
    engine = _engine_for_node(gm, n)
    try:
        return int(engine.device_info.id)  # TorchBind engine or mock
    except AttributeError:
        return 0


def _inline_trt_call_modules(gm: torch.fx.GraphModule) -> None:
    """Mutate gm in place: replace each TRT call_module node whose engine is a
    real TorchBind ``torch.classes.tensorrt.Engine`` with a
    ``call_function(torch.ops.tensorrt.execute_engine, (inputs, engine_attr))``
    pair.

    This mirrors :func:`torch_tensorrt.dynamo._exporter.inline_trt_modules`
    but skips submodules whose ``.engine`` is not a real ScriptObject (e.g.
    mock TRT modules used in unit tests), letting tests keep working without
    real engines.

    Idempotent: post-serde graphs already have execute_engine nodes and no
    TRT call_module nodes, so this is a no-op for them.
    """
    graph = gm.graph
    targets = [
        n
        for n in graph.nodes
        if n.op == "call_module" and _is_trt_module(gm.get_submodule(n.target))
    ]
    for trt_module_node in targets:
        submod = gm.get_submodule(trt_module_node.target)
        engine = getattr(submod, "engine", None)
        if not isinstance(engine, torch._C.ScriptObject):
            # Mock or otherwise non-TorchBind engine — leave as call_module.
            continue
        if "val" not in trt_module_node.meta:
            # Without meta we can't determine number of outputs.  Skip.
            continue

        name = trt_module_node.target
        engine_name = f"{name}_engine"
        setattr(gm, engine_name, engine)

        with graph.inserting_before(trt_module_node):
            engine_node = graph.get_attr(engine_name)
            trt_node = graph.call_function(
                torch.ops.tensorrt.execute_engine.default,
                (trt_module_node.args, engine_node),
            )
        # Preserve meta on the new call_function and the engine get_attr.
        trt_node.meta["val"] = trt_module_node.meta["val"]
        # execute_engine returns a list; unwrap consistently with the exporter.
        num_outputs = len(trt_module_node.meta["val"])
        if num_outputs == 1:
            with graph.inserting_after(trt_node):
                getitem_output = graph.call_function(operator.getitem, (trt_node, 0))
                getitem_output.meta["val"] = trt_node.meta["val"]
            trt_module_node.replace_all_uses_with(getitem_output)
        else:
            trt_module_node.replace_all_uses_with(trt_node)
            for idx, getitem_node in enumerate(trt_node.users):
                getitem_node.meta["val"] = trt_node.meta["val"][idx]
        graph.erase_node(trt_module_node)


def _normalize_hint_name(name: str, valid_keys: set[str]) -> str:
    """Resolve a user-provided hint name to the canonical key.

    Accepts both the submodule name (``_run_on_acc_0``) and the engine attr
    name (``_run_on_acc_0_engine``) since the inlining adds the suffix.
    """
    if name in valid_keys:
        return name
    suffixed = f"{name}_engine"
    if suffixed in valid_keys:
        return suffixed
    return name  # let validation raise with the original name


def _resolve_plan(
    gm: torch.fx.GraphModule,
    streams: Optional[list[torch.cuda.Stream]],
    hints: Optional[dict[str, torch.cuda.Stream]],
) -> StreamPlan:
    from torch_tensorrt.runtime._cudagraphs import _PY_RT_CUDAGRAPHS, CudaGraphsMode

    if _PY_RT_CUDAGRAPHS != CudaGraphsMode.STANDARD:
        raise StreamPlanError(
            "stream_plan() is incompatible with CUDA Graphs. "
            "Disable cudagraphs before applying a stream plan."
        )

    nodes = _trt_nodes(gm)
    if not nodes:
        raise StreamPlanError("No TRT subgraphs found in compiled module.")

    keys = [_trt_node_key(n) for n in nodes]
    assignment: dict[str, torch.cuda.Stream] = {}

    if streams is not None:
        if len(streams) != len(nodes):
            raise StreamPlanError(
                f"streams has {len(streams)} entries but graph has "
                f"{len(nodes)} TRT nodes: {keys}"
            )
        for key, stream in zip(keys, streams):
            assignment[key] = stream

    if hints:
        valid = set(keys)
        for raw_name, stream in hints.items():
            name = _normalize_hint_name(raw_name, valid)
            if name not in valid:
                raise StreamPlanError(
                    f"hints references unknown node {raw_name!r}. " f"TRT nodes: {keys}"
                )
            if name in assignment and assignment[name] is not stream:
                raise StreamPlanError(
                    f"Conflicting stream for {raw_name!r}: "
                    "positional and hint disagree."
                )
            assignment[name] = stream

    for key in keys:
        if key not in assignment:
            assignment[key] = torch.cuda.Stream()

    device: Optional[torch.device] = None
    for node, key in zip(nodes, keys):
        engine_device_id = _engine_device_id(gm, node)
        stream = assignment[key]
        if device is None:
            device = torch.device("cuda", engine_device_id)
        if stream.device.index != engine_device_id:
            raise StreamPlanError(
                f"Stream for {key!r} is on cuda:{stream.device.index} "
                f"but engine targets cuda:{engine_device_id}."
            )

    return StreamPlan(assignment=assignment, device=device)


# ── FX pass ───────────────────────────────────────────────────────────────────


def _apply_stream_plan(
    planned: torch.fx.GraphModule,
    plan: StreamPlan,
) -> torch.fx.GraphModule:
    """
    Mutates ``planned`` in place to insert:
      - enter_compute_stream / exit_compute_stream at the graph boundary
      - set_stream at each stream transition
      - sync_streams at every cross-stream data-dependency edge
      - call_trt_with_token wrapping each TRT engine call

    The caller is responsible for handing in a deep copy of the user's
    GraphModule (apply_stream_plan/stream_plan handle that and the inlining).

    Streams are bound to ``torch.classes.tensorrt.StreamGuard`` ScriptObjects
    registered as top-level module attributes (``_trt_stream_guard_N``).  The
    FX graph references the guards via get_attr, never as raw int handles —
    so the planned module survives save/load, AOTI codegen, and re-export
    without dangling pointers.
    """
    device_index = (plan.device.index or 0) if plan.device else 0
    graph = planned.graph

    # ── Stream slots ─────────────────────────────────────────────────────────
    # One StreamGuard per unique stream object in the plan.  Each guard holds
    # the live cudaStream_t handle for the duration of the planned module's
    # lifetime.  Slot indexing is stable so positional save/load works.
    StreamGuardCls = torch.classes.tensorrt.StreamGuard
    slot_for_stream: dict[int, int] = {}  # id(stream) → slot_idx
    slot_to_keepalive: list[torch.cuda.Stream] = []  # slot_idx → Python Stream
    for stream in plan.assignment.values():
        if id(stream) in slot_for_stream:
            continue
        slot = len(slot_for_stream)
        slot_for_stream[id(stream)] = slot
        guard = StreamGuardCls(stream.device.index, False)
        guard.bind(stream.cuda_stream)
        setattr(planned, f"_trt_stream_guard_{slot}", guard)
        slot_to_keepalive.append(stream)
    # Keep Python torch.cuda.Stream wrappers alive as long as the planned
    # module is alive — otherwise the cudaStream_t under the guard could be
    # destroyed.  This list is intentionally not pickled; on load a fresh
    # set of streams is bound by bind_stream_plan_streams (or auto-bind).
    planned._trt_stream_keepalives = slot_to_keepalive

    # Caller-guard slot: a separate StreamGuard mutated at runtime by
    # enter_compute_stream to track the caller's current stream.  Lives as a
    # graph attribute so downstream sync_streams (caller↔compute) can
    # reference it via get_attr — and so it survives save/load identically to
    # compute slots.
    caller_guard = StreamGuardCls(device_index, False)
    planned._trt_caller_guard = caller_guard

    def slot_for_node(n: torch.fx.Node) -> Optional[int]:
        """Return the stream-slot index for a TRT node, or None if not TRT."""
        is_trt = (
            n.op == "call_module" and _is_trt_module(planned.get_submodule(n.target))
        ) or _is_execute_engine_node(n)
        if not is_trt:
            return None
        key = _trt_node_key(n)
        if key in plan.assignment:
            return slot_for_stream[id(plan.assignment[key])]
        return None

    # Pre-create one get_attr node per slot, hoisted to the top of the graph
    # right after the last placeholder.  These nodes are referenced by every
    # subsequent stream-control op insertion.
    last_placeholder = None
    for n in graph.nodes:
        if n.op == "placeholder":
            last_placeholder = n
        else:
            break

    slot_to_guard_node: dict[int, torch.fx.Node] = {}
    if last_placeholder is not None:
        with graph.inserting_after(last_placeholder):
            for slot in range(len(slot_for_stream)):
                slot_to_guard_node[slot] = graph.get_attr(f"_trt_stream_guard_{slot}")
            caller_guard_node = graph.get_attr("_trt_caller_guard")
    else:
        with graph.inserting_before(next(iter(graph.nodes))):
            for slot in range(len(slot_for_stream)):
                slot_to_guard_node[slot] = graph.get_attr(f"_trt_stream_guard_{slot}")
            caller_guard_node = graph.get_attr("_trt_caller_guard")

    # ── Step 1: barriers at cross-stream data-dependency edges ───────────────
    # For each TRT node N, for each unique source slot among its direct
    # predecessors that differs from N's slot, insert one sync_streams.
    # Predecessors on the caller stream are deferred until step 3 produces
    # the caller-guard node.

    deferred_caller_syncs: list[tuple[torch.fx.Node, int]] = []  # (consumer, dst_slot)

    for n in list(graph.nodes):
        if n.op in ("placeholder", "get_attr", "output"):
            continue
        n_slot = slot_for_node(n)
        if n_slot is None:
            continue
        synced_from: set[Optional[int]] = set()
        for pred in n.all_input_nodes:
            p_slot = slot_for_node(pred)
            if p_slot == n_slot or p_slot in synced_from:
                continue
            synced_from.add(p_slot)
            if p_slot is not None:
                with graph.inserting_before(n):
                    graph.call_function(
                        torch.ops.tensorrt.sync_streams.default,
                        args=(
                            slot_to_guard_node[p_slot],
                            slot_to_guard_node[n_slot],
                            device_index,
                        ),
                    )
            else:
                deferred_caller_syncs.append((n, n_slot))

    # ── Step 2: set_stream at each stream transition ─────────────────────────
    current_slot: Optional[int] = None
    for n in list(graph.nodes):
        if n.op in ("placeholder", "get_attr", "output"):
            continue
        n_slot = slot_for_node(n)
        if n_slot is not None and n_slot != current_slot:
            with graph.inserting_before(n):
                graph.call_function(
                    torch.ops.tensorrt.set_stream.default,
                    args=(slot_to_guard_node[n_slot], device_index),
                )
            current_slot = n_slot
        elif n_slot is None:
            current_slot = None

    # ── Step 3: graph entry — enter_compute_stream ───────────────────────────
    # Mutates the pre-registered caller_guard attribute at runtime to capture
    # the caller's current stream handle.  Returns the handle as an int token.
    # caller_guard_node was already created above and is referenced by steps
    # 4–5 as the caller side of cross-stream syncs.
    primary_trt = next(n for n in graph.nodes if slot_for_node(n) is not None)
    primary_slot_opt = slot_for_node(primary_trt)
    assert primary_slot_opt is not None  # `next` above guarantees this
    primary_slot: int = primary_slot_opt
    first_compute = next(
        n for n in graph.nodes if n.op not in ("placeholder", "get_attr")
    )
    with graph.inserting_before(first_compute):
        graph.call_function(
            torch.ops.tensorrt.enter_compute_stream.default,
            args=(slot_to_guard_node[primary_slot], caller_guard_node, device_index),
        )

    # ── Step 4: patch deferred caller-stream barriers ────────────────────────
    seen_deferred: set[tuple[int, int]] = set()
    for n, dst_slot in deferred_caller_syncs:
        key = (id(n), dst_slot)
        if key in seen_deferred:
            continue
        seen_deferred.add(key)
        with graph.inserting_before(n):
            graph.call_function(
                torch.ops.tensorrt.sync_streams.default,
                args=(caller_guard_node, slot_to_guard_node[dst_slot], device_index),
            )

    # ── Step 5: sync every compute slot back to caller, then exit ────────────
    # Sync ALL unique compute streams rather than just the last TRT node's
    # stream.  Fan-out topologies (x → eng0, x → eng1 → add) end with a
    # non-TRT op, so without an explicit per-slot sync the caller stream
    # would read tensors produced on the compute streams without a fence.
    output_node = next(n for n in graph.nodes if n.op == "output")
    with graph.inserting_before(output_node):
        for slot in range(len(slot_for_stream)):
            graph.call_function(
                torch.ops.tensorrt.sync_streams.default,
                args=(slot_to_guard_node[slot], caller_guard_node, device_index),
            )
        graph.call_function(
            torch.ops.tensorrt.exit_compute_stream.default,
            args=(caller_guard_node, device_index),
        )

    # ── Step 6: replace each TRT engine call with call_trt_with_token ────────
    # call_trt_with_token is a real dispatcher op (registered in
    # core/runtime/stream_ops.cpp) — not a Python HOP.  AOTI codegens it like
    # any other custom op.  Schema:
    #
    #   call_trt_with_token(int token,
    #                       __torch__.classes.tensorrt.Engine engine,
    #                       Tensor[] inputs) -> Tensor[]
    #
    # The token is an int returned by the preceding set_stream / sync_streams
    # node, creating a hard data-flow edge from stream setup to engine call.
    # The body discards the token and forwards to execute_engine.
    #
    # Two input forms handled:
    #   1. call_module → TorchTensorRTModule (mock submodules in unit tests):
    #      no rewrite — the call_module stays, and the preceding set_stream
    #      mutation has already switched the current stream.  Mocks don't go
    #      through the real op (their engine is not a TorchBind ScriptObject).
    #   2. call_function(execute_engine) (real engines after inline pass):
    #      replaced by call_trt_with_token consuming the same engine attr,
    #      with the preceding stream-control op's int return as token.
    _stream_ctrl_ops = {
        torch.ops.tensorrt.set_stream.default,
        torch.ops.tensorrt.sync_streams.default,
    }
    node_list = list(graph.nodes)
    for idx, n in enumerate(node_list):
        if not _is_execute_engine_node(n):
            continue
        if slot_for_node(n) is None:
            continue
        last_ctrl: Optional[torch.fx.Node] = None
        for prev in reversed(node_list[:idx]):
            if prev.op == "call_function" and prev.target in _stream_ctrl_ops:
                last_ctrl = prev
                break
        if last_ctrl is None:
            continue

        engine_node = n.args[1]
        engine_inputs_arg = n.args[0]  # already a list-typed arg per schema
        original_meta = dict(n.meta)

        # Detect the exporter's single-output getitem(n, 0) pattern so we
        # don't end up with a redundant getitem after rewriting.
        users = list(n.users)
        single_getitem: Optional[torch.fx.Node] = None
        if len(users) == 1:
            u = users[0]
            if (
                u.op == "call_function"
                and u.target is operator.getitem
                and u.args == (n, 0)
            ):
                single_getitem = u

        with graph.inserting_before(n):
            new_node = graph.call_function(
                torch.ops.tensorrt.call_trt_with_token.default,
                args=(last_ctrl, engine_node, engine_inputs_arg),
            )
        new_node.meta.update(original_meta)
        if single_getitem is not None:
            # Both the original execute_engine and call_trt_with_token return
            # Tensor[].  Forward all uses of n to new_node, then the existing
            # getitem(0) pattern keeps working without modification.
            pass
        n.replace_all_uses_with(new_node)
        graph.erase_node(n)

    graph.lint()
    planned.recompile()
    planned._stream_plan_applied = True

    logger.debug(
        "apply_stream_plan: inserted stream management into graph with "
        "%d TRT subgraphs across %d unique streams",
        len(plan.assignment),
        len(slot_for_stream),
    )

    return planned


def _release(planned: torch.fx.GraphModule) -> None:
    """Drop stream attributes from the planned module. TRT engines are untouched."""
    for attr in list(vars(planned)):
        if attr.startswith("_trt_stream_guard_") or attr in (
            "_trt_stream_keepalives",
            "_trt_caller_guard",
        ):
            delattr(planned, attr)
