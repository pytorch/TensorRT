from __future__ import annotations

import copy
from typing import Any, Sequence

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.export.graph_signature import InputKind


def _is_graph_bound_metadata(value: Any) -> bool:
    """True if a metadata value is tied to the exported program's shape environment.

    Fake tensors and symbolic sizes reach back to a live ``ShapeEnv`` that owns
    tensors deepcopy refuses to touch, and cloning them would also detach the copy
    from the symbols the graph is guarded on.
    """
    for leaf in torch.utils._pytree.tree_leaves(value):
        if isinstance(leaf, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return True
        if isinstance(leaf, torch.Tensor) and is_fake(leaf):
            return True
    return False


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


def validate_engine_program(exported_program: Any) -> int:
    """Validate all TensorRT engine nodes before any input program is mutated."""
    count = 0
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if (
            node.target is not torch.ops.tensorrt.execute_engine.default
            and _schema_name(node) != "tensorrt::no_op_placeholder_for_execute_engine"
        ):
            continue
        validate_engine_info(
            _resolve_engine_info(exported_program, node), node_name=node.name
        )
        count += 1
    return count


def _payload_sharing_memo(exported_program: Any) -> dict[int, Any]:
    memo: dict[int, Any] = {}
    for value in (
        *exported_program.state_dict.values(),
        *exported_program.constants.values(),
    ):
        memo[id(value)] = value
    for module in exported_program.graph_module.modules():
        for value in (
            *module.parameters(recurse=False),
            *module.buffers(recurse=False),
        ):
            memo[id(value)] = value
        for value in module.__dict__.values():
            if isinstance(value, (torch.Tensor, torch.ScriptObject)):
                memo[id(value)] = value
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            for value in node.meta.values():
                for leaf in torch.utils._pytree.tree_leaves(value):
                    if isinstance(leaf, (torch.Tensor, torch.ScriptObject)):
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
        staged_nodes = {node.name: node for node in staged_module.graph.nodes}
        source_nodes = {node.name: node for node in source_module.graph.nodes}
        if staged_nodes.keys() != source_nodes.keys():
            raise RuntimeError(
                f"Staged GraphModule {name or '<root>'!r} changed node identities."
            )
        for node_name, source_node in source_nodes.items():
            staged_nodes[node_name].meta = {
                key: (
                    value
                    if _is_graph_bound_metadata(value)
                    else copy.deepcopy(value, payload_memo)
                )
                for key, value in source_node.meta.items()
            }
    return staged


def stage_exported_program(exported_program: Any) -> Any:
    """Copy program structure while sharing immutable tensor and engine payloads."""
    graph_module = _stage_graph_module(
        exported_program.graph_module,
        _payload_sharing_memo(exported_program),
    )
    return exported_program._update(
        graph_module,
        copy.deepcopy(exported_program.graph_signature),
        state_dict=dict(exported_program.state_dict),
        constants=dict(exported_program.constants),
    )


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


def replace_execute_engine(exported_program: Any) -> Any:
    """Replace execute_engine nodes with ExecuTorch-safe placeholder calls."""
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
            engine_info = _resolve_engine_info(exported_program, node)
            engine_bytes = engine_info[ENGINE_IDX]
            if isinstance(engine_bytes, str):
                import base64

                engine_bytes = base64.b64decode(engine_bytes)
            elif not isinstance(engine_bytes, (bytes, bytearray)):
                engine_bytes = bytes(engine_bytes)
            engine_tensor = torch.frombuffer(bytearray(engine_bytes), dtype=torch.uint8)

            from torch.fx.experimental.const_fold import get_unique_attr_name_in_module

            buffer_name = get_unique_attr_name_in_module(graph_module, "_trt_engine_0")
            graph_module.register_buffer(buffer_name, engine_tensor, persistent=True)
            exported_program.state_dict[buffer_name] = engine_tensor
            # The engine slot is replaced by engine_attr_node below, so skip it:
            # str() on multi-gigabyte engine bytes would build a throwaway string
            # roughly four times their size.
            str_args = [
                ("" if index == ENGINE_IDX else str(value) if value is not None else "")
                for index, value in enumerate(engine_info[:SERIALIZATION_LEN])
            ]

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
