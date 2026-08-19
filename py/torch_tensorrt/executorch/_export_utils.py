from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import is_fake
from torch.export.graph_signature import InputKind

logger = logging.getLogger(__name__)

# A lifted engine reaches the graph as a FakeScriptObject holding the real engine on
# .real_obj, and deepcopy of a real engine is a serialize plus a deserialize.
_SHARED_PAYLOAD_TYPES = (torch.Tensor, torch.ScriptObject, FakeScriptObject)


def _seed_graph_bound_leaves(value: Any, memo: dict[int, Any]) -> None:
    """Seed shape-environment-bound leaves into ``memo`` so deepcopy shares them.

    Fake tensors and symbolic sizes reach back to a live ``ShapeEnv`` that owns
    tensors deepcopy refuses to touch, and cloning them would also detach the copy
    from the symbols the graph is guarded on. Seeding only the leaves lets the
    surrounding container still be copied, so a container such as the ``list`` a
    multi-output op stores in ``meta["val"]`` is not shared with the caller.
    """
    for leaf in torch.utils._pytree.tree_leaves(value):
        if isinstance(leaf, (torch.SymInt, torch.SymFloat, torch.SymBool)) or (
            isinstance(leaf, torch.Tensor) and is_fake(leaf)
        ):
            memo[id(leaf)] = leaf


def get_engine_info_from_state(engine_obj: Any) -> list[Any]:
    """Normalize TensorRT engine state into the serialized engine-info list."""
    state = engine_obj.__getstate__()
    engine_info = state[0] if isinstance(state, tuple) else state
    return list(engine_info)


def validate_engine_info(engine_info: Sequence[Any], *, node_name: str = "") -> None:
    """Reject engine configurations unsupported by ExecuTorch export."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
        REQUIRES_OUTPUT_ALLOCATOR_IDX,
    )

    if (
        len(engine_info) > REQUIRES_OUTPUT_ALLOCATOR_IDX
        and str(engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX]) == "1"
    ):
        node_suffix = f" for node '{node_name}'" if node_name else ""
        raise RuntimeError(
            "ExecuTorch export does not support TensorRT engines that require "
            "an output allocator (data-dependent output shapes)"
            f"{node_suffix}."
        )


def _schema_name(node: Any) -> str:
    target = node.target
    return target._schema.name if hasattr(target, "_schema") else ""


def _resolve_engine_info(exported_program: Any, node: Any) -> list[Any]:
    if node.target is not torch.ops.tensorrt.execute_engine.default:
        return list(node.args[1:])

    graph_module = exported_program.graph_module
    engine_node = node.args[1]
    if engine_node.op == "get_attr":
        engine_obj = getattr(graph_module, engine_node.target, None)
        if engine_obj is None:
            raise RuntimeError(
                f"execute_engine node '{node.name}': get_attr target "
                f"'{engine_node.target}' not found on graph module"
            )
    elif engine_node.op == "placeholder":
        from torch_tensorrt.dynamo._exporter import _resolve_lifted_custom_obj

        engine_obj = _resolve_lifted_custom_obj(exported_program, engine_node)
        if engine_obj is None:
            raise RuntimeError(
                f"execute_engine node '{node.name}': placeholder engine "
                f"'{engine_node.name}' did not resolve to a lifted "
                f"custom-object constant (available: "
                f"{sorted(getattr(exported_program, 'constants', {}) or {})})"
            )
    else:
        raise RuntimeError(
            f"execute_engine node '{node.name}': unexpected engine arg op "
            f"'{engine_node.op}'"
        )
    return get_engine_info_from_state(engine_obj)


def validate_engine_program(
    exported_program: Any, resolved: dict[str, list[Any]] | None = None
) -> int:
    """Validate all TensorRT engine nodes before any input program is mutated.

    Resolving an engine serializes it, so pass a dict as ``resolved`` to keep what was
    resolved here and let the rewrite reuse it instead of serializing a second time.
    """
    count = 0
    # One engine can feed several execute_engine nodes, and resolving it again would
    # serialize it again and keep a second copy of the bytes.
    by_engine_node: dict[Any, list[Any]] = {}
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if (
            node.target is not torch.ops.tensorrt.execute_engine.default
            and _schema_name(node) != "tensorrt::no_op_placeholder_for_execute_engine"
        ):
            continue
        engine_node = (
            node.args[1]
            if node.target is torch.ops.tensorrt.execute_engine.default
            else None
        )
        engine_info = (
            by_engine_node.get(engine_node) if engine_node is not None else None
        )
        if engine_info is None:
            engine_info = _resolve_engine_info(exported_program, node)
            if engine_node is not None:
                by_engine_node[engine_node] = engine_info
        validate_engine_info(engine_info, node_name=node.name)
        if resolved is not None:
            resolved[node.name] = engine_info
        count += 1
    return count


def _payload_sharing_memo(exported_program: Any) -> dict[int, Any]:
    memo: dict[int, Any] = {}
    for value in (
        *exported_program.state_dict.values(),
        *exported_program.constants.values(),
    ):
        if isinstance(value, _SHARED_PAYLOAD_TYPES):
            memo[id(value)] = value
    for module in exported_program.graph_module.modules():
        for value in (
            *module.parameters(recurse=False),
            *module.buffers(recurse=False),
        ):
            memo[id(value)] = value
        for value in module.__dict__.values():
            if isinstance(value, _SHARED_PAYLOAD_TYPES):
                memo[id(value)] = value
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            for value in node.meta.values():
                for leaf in torch.utils._pytree.tree_leaves(value):
                    if isinstance(leaf, _SHARED_PAYLOAD_TYPES):
                        memo[id(leaf)] = leaf
    return memo


def _stage_graph_module(
    graph_module: torch.fx.GraphModule,
    payload_memo: dict[int, Any],
) -> torch.fx.GraphModule:
    staged = copy.deepcopy(graph_module, payload_memo)
    # FX gives each staged node a fresh meta dict but shares the values inside it,
    # so nested containers need their own copy to keep the source unaffected.
    for name, source_module in graph_module.named_modules():
        if not isinstance(source_module, torch.fx.GraphModule):
            continue
        staged_module = staged if not name else staged.get_submodule(name)
        if not isinstance(staged_module, torch.fx.GraphModule):
            continue
        # Node copying renames any node whose name shadows a Python builtin, so 'sum'
        # comes back as 'sum_1', which in turn pushes an existing 'sum_1' to 'sum_2'.
        # Copying preserves node order, so put the source names back position by
        # position before anything reads them. Renaming is what makes the name check
        # below pass, so pair the nodes on what copying does keep.
        for source_node, staged_node in zip(
            source_module.graph.nodes, staged_module.graph.nodes
        ):
            if (staged_node.op, staged_node.target) != (
                source_node.op,
                source_node.target,
            ):
                raise RuntimeError(
                    f"Staged GraphModule {name or '<root>'!r} reordered its nodes."
                )
            staged_node.name = source_node.name
        staged_nodes = {node.name: node for node in staged_module.graph.nodes}
        source_nodes = {node.name: node for node in source_module.graph.nodes}
        if staged_nodes.keys() != source_nodes.keys():
            raise RuntimeError(
                f"Staged GraphModule {name or '<root>'!r} changed node identities."
            )
        for node_name, source_node in source_nodes.items():
            staged_meta = {}
            for key, value in source_node.meta.items():
                # Seed the shape-bound leaves first so the container around them
                # is still copied while the leaves stay shared.
                _seed_graph_bound_leaves(value, payload_memo)
                # deepcopy records a copy in the memo before filling it, so a failed
                # attempt leaves half-built copies that a later value sharing the same
                # object would silently reuse. Everything added after this mark belongs
                # to this attempt, because deepcopy only ever appends to the memo.
                memo_mark = len(payload_memo)
                try:
                    staged_meta[key] = copy.deepcopy(value, payload_memo)
                except (
                    RuntimeError,
                    TypeError,
                    ValueError,
                    NotImplementedError,
                ) as exc:
                    for stale in list(payload_memo)[memo_mark:]:
                        del payload_memo[stale]
                    # A value deepcopy cannot reach through, for example a live
                    # ShapeEnv, has to be shared rather than lose the graph its guards
                    # refer to. Log it, since sharing a mutable value here is the one
                    # case that breaks the isolation the rest of this function provides.
                    logger.warning(
                        "sharing node %r meta %r with the source program: %s",
                        node_name,
                        key,
                        exc,
                    )
                    staged_meta[key] = value
            staged_nodes[node_name].meta = staged_meta
    return staged


def stage_exported_program(exported_program: Any) -> Any:
    """Copy program structure while sharing immutable tensor and engine payloads."""
    payload_memo = _payload_sharing_memo(exported_program)
    graph_module = _stage_graph_module(exported_program.graph_module, payload_memo)
    return exported_program._update(
        graph_module,
        # The signature holds the lifted engine on CustomObjArgument.fake_val, so it
        # needs the same memo or the copy reaches the engine the graph shares.
        copy.deepcopy(exported_program.graph_signature, payload_memo),
        state_dict=dict(exported_program.state_dict),
        constants=dict(exported_program.constants),
    )


def _unique_engine_buffer_name(
    exported_program: Any, graph_module: torch.fx.GraphModule
) -> str:
    """Pick an engine buffer name free on the module and on both lifted namespaces.

    torch.export lifts buffers into ``state_dict`` and script objects and non-persistent
    tensors into ``constants``, so a name can be taken in either while ``hasattr`` on the
    module reports nothing.
    """
    from torch.fx.experimental.const_fold import get_unique_attr_name_in_module

    index = 0
    while True:
        name: str = get_unique_attr_name_in_module(graph_module, f"_trt_engine_{index}")
        if name not in exported_program.state_dict and name not in (
            exported_program.constants
        ):
            return name
        index += 1


def _remove_lifted_engine_placeholder(
    exported_program: Any, engine_node: torch.fx.Node
) -> None:
    """Remove an unused lifted engine from the graph, signature, and constants."""
    if engine_node.op != "placeholder" or engine_node.users:
        raise RuntimeError(
            f"Engine placeholder {engine_node.name!r} must be unused before cleanup."
        )

    matching_specs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == InputKind.CUSTOM_OBJ
        and getattr(spec.arg, "name", None) == engine_node.name
    ]
    if len(matching_specs) != 1:
        raise RuntimeError(
            f"Engine placeholder {engine_node.name!r} must have exactly one "
            f"CUSTOM_OBJ input spec; found {len(matching_specs)}."
        )

    engine_spec = matching_specs[0]
    engine_target = engine_spec.target
    if not isinstance(engine_target, str):
        raise RuntimeError(
            f"Engine placeholder {engine_node.name!r} has no constant target."
        )

    exported_program._graph_signature.input_specs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec is not engine_spec
    ]
    engine_node.graph.erase_node(engine_node)

    if not any(
        spec.target == engine_target
        for spec in exported_program.graph_signature.input_specs
    ):
        exported_program.constants.pop(engine_target, None)


def replace_execute_engine(
    exported_program: Any, resolved: dict[str, list[Any]] | None = None
) -> Any:
    """Replace execute_engine nodes with ExecuTorch-safe placeholder calls.

    ExecuTorch's lowering runs passes that dispatch through the C++ schema validator, and
    that validator rejects the engine argument because it arrives as a custom-object
    placeholder rather than a real script object. The replacement node carries the same
    engine information as plain strings, so the passes never see a script object.

    The engine bytes are stored as a uint8 buffer on the graph module and referenced
    through a get_attr node. That keeps the payload out of the Python source the graph
    emits, because CPython's tokenizer cannot parse a string literal larger than about
    2 GB, so an inline base64 string breaks recompilation for any engine past that size.

    ``resolved`` carries what validate_engine_program already resolved, so the engine is
    not serialized again here.
    """
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
        ENGINE_IDX,
        SERIALIZATION_LEN,
    )

    graph_module = exported_program.graph_module
    execute_engine_op = torch.ops.tensorrt.execute_engine.default
    no_op = torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default
    nodes_to_replace = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is execute_engine_op
    ]
    if not nodes_to_replace:
        return exported_program

    materialized_engines: dict[torch.fx.Node, tuple[torch.fx.Node, list[str]]] = {}
    for node in nodes_to_replace:
        inputs_arg = node.args[0]
        engine_node = node.args[1]
        materialized = materialized_engines.get(engine_node)
        if materialized is None:
            engine_info = (resolved or {}).get(node.name)
            if engine_info is None:
                engine_info = _resolve_engine_info(exported_program, node)
            engine_bytes = engine_info[ENGINE_IDX]
            if isinstance(engine_bytes, str):
                import base64

                engine_bytes = base64.b64decode(engine_bytes)
            elif not isinstance(engine_bytes, (bytes, bytearray)):
                engine_bytes = bytes(engine_bytes)
            # frombuffer needs a writable buffer, and rebinding here releases the
            # read-only copy instead of holding both for the rest of the loop.
            engine_bytes = bytearray(engine_bytes)
            engine_tensor = torch.frombuffer(engine_bytes, dtype=torch.uint8)

            buffer_name = _unique_engine_buffer_name(exported_program, graph_module)
            graph_module.register_buffer(buffer_name, engine_tensor, persistent=True)
            exported_program.state_dict[buffer_name] = engine_tensor
            # Nothing reads the engine slot: engine_attr_node takes that position in
            # no_op_args below. Keep the slot so the indices around it still line up.
            str_args = [
                ("" if index == ENGINE_IDX else str(value) if value is not None else "")
                for index, value in enumerate(engine_info[:SERIALIZATION_LEN])
            ]

            # Reuse the graph's existing fake mode. A fresh one fails downstream with
            # "fake mode from input 0 doesn't match mode from input 1" as soon as a pass
            # mixes tensors from the two modes.
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(
                [
                    graph_node.meta["val"]
                    for graph_node in graph_module.graph.nodes
                    if "val" in graph_node.meta
                ]
            )
            fake_engine = (
                fake_mode.from_tensor(engine_tensor)
                if fake_mode is not None
                else engine_tensor
            )
            with graph_module.graph.inserting_before(node):
                engine_attr_node = graph_module.graph.get_attr(buffer_name)
                engine_attr_node.meta["val"] = fake_engine
            materialized = (engine_attr_node, str_args)
            materialized_engines[engine_node] = materialized
        engine_attr_node, str_args = materialized

        with graph_module.graph.inserting_before(node):
            no_op_args = (
                inputs_arg,
                *str_args[:ENGINE_IDX],
                engine_attr_node,
                *str_args[ENGINE_IDX + 1 :],
            )
            no_op_node = graph_module.graph.call_function(no_op, no_op_args)
            no_op_node.meta["val"] = node.meta.get("val")

        node.replace_all_uses_with(no_op_node)
        graph_module.graph.erase_node(node)
        if engine_node.op == "get_attr" and not engine_node.users:
            graph_module.graph.erase_node(engine_node)
            from torch.fx.graph_module import _del_attr, _has_attr

            if _has_attr(graph_module, engine_node.target):
                _del_attr(graph_module, engine_node.target)
        elif engine_node.op == "placeholder" and not engine_node.users:
            _remove_lifted_engine_placeholder(exported_program, engine_node)

    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    return exported_program
