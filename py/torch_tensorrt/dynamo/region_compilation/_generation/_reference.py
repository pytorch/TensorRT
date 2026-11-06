# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Save and load a captured reference graph as JSON, without pickle.

Runs and caches are directories that can be shared, copied, or replayed, so
loading one must not execute code from it. The captured reference contains only
ATen and prims operations, a few ``operator`` functions, numbered inputs, and
JSON constants. It is written as data and rebuilt from operators looked up by
name in ``torch.ops``. FX generates and runs Python code for the rebuilt graph,
so saved node names only link values, and the constant and keyword names that
it writes must be plain identifiers.
"""

from __future__ import annotations

import functools
import json
import keyword
import operator
from pathlib import Path
from typing import Any

import torch

FILENAME = "reference.json"
_VERSION = 1
_NAMESPACES = ("aten", "prims")
# Python functions that captured ATen graphs use on tuple outputs and scalars.
_OPERATORS = {
    name: getattr(operator, name)
    for name in ("getitem", "add", "sub", "mul", "truediv", "floordiv", "neg")
}
# Operators that pass the schema checks in _supported but still have effects:
# from_file opens, creates, and maps a file, and the _copy_from variants write
# into dst without declaring it.
_UNSAFE = frozenset(
    {"aten::from_file", "aten::_copy_from", "aten::_copy_from_and_resize"}
)
# Bounds for files from shared runs. Captured arguments nest a few levels, and a
# reference this large could not fit in an author's prompt.
_MAX_DEPTH = 16
_MAX_BYTES = 16 << 20


def _supported(target: torch._ops.OpOverload) -> bool:
    """Whether a reference may call ``target``.

    Captured references are functional, deterministic dispatcher graphs. An
    operator whose schema declares mutation, that PyTorch tags as random, or that
    returns nothing (and so runs only for an effect, such as printing or saving)
    is refused, as are TorchScript builtins, whose schemas do not reliably
    declare mutation, and Inductor's internal primitives, whose random ones
    carry no random tag.
    """
    schema = target._schema
    if (
        target.namespace not in _NAMESPACES
        or schema.is_mutable
        or torch.Tag.nondeterministic_seeded in target.tags
        or not schema.returns
        or schema.name in _UNSAFE
        # Captured references never contain them; inductor_random is untagged.
        or schema.name.startswith("prims::inductor_")
    ):
        return False
    try:
        torch._C._dispatch_find_schema_or_throw(schema.name, schema.overload_name)
    except RuntimeError:
        return False
    return True


def _supported_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether arguments have the forms that captured graphs give ``target``."""
    if isinstance(target, torch._ops.OpOverload):
        # FX writes keyword names into the code it generates.
        names = {argument.name for argument in target._schema.arguments}
        return all(key in names and not keyword.iskeyword(key) for key in kwargs)
    # Python operators combine node values and numbers; a list or string
    # operand could be repeated into an arbitrarily large value.
    return (
        not kwargs
        and len(args) == (1 if target is operator.neg else 2)
        and all(isinstance(arg, (torch.fx.Node, bool, int, float)) for arg in args)
    )


@functools.cache
def _graph_module_attributes() -> frozenset[str]:
    return frozenset(dir(torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())))


def _constant_name(name: Any) -> bool:
    # FX writes the name into the generated code as an attribute of the module,
    # and one that GraphModule already uses, such as forward, would replace it.
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.startswith("__")
        and name not in _graph_module_attributes()
    )


def _encode(value: Any, depth: int = 0) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"node": value.name}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and value != value:
            return {"float": "nan"}
        if isinstance(value, float) and value in (float("inf"), float("-inf")):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (list, tuple)):
        if depth >= _MAX_DEPTH:
            raise TypeError("cannot save a reference argument nested this deeply")
        return {
            "list" if isinstance(value, list) else "tuple": [
                _encode(v, depth + 1) for v in value
            ]
        }
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"device": str(value)}
    if isinstance(value, torch.layout):
        return {"layout": str(value).removeprefix("torch.")}
    if isinstance(value, torch.memory_format):
        return {"memory_format": str(value).removeprefix("torch.")}
    raise TypeError(f"cannot save reference argument of type {type(value).__name__}")


def _torch_attribute(name: Any, kind: type) -> Any:
    # vars() rather than getattr, which imports some torch submodules on demand.
    value = vars(torch).get(name) if isinstance(name, str) else None
    if not isinstance(value, kind):
        raise ValueError(f"reference names an unknown {kind.__name__} {name!r}")
    return value


def _decode(value: Any, nodes: dict[str, torch.fx.Node], depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("reference argument has an unexpected form")
    ((kind, data),) = value.items()
    if kind == "node":
        if not isinstance(data, str) or data not in nodes:
            raise ValueError(f"reference uses an undefined value {data!r}")
        return nodes[data]
    if kind == "float" and data in ("nan", "inf", "-inf"):
        return float(data)
    if kind in ("list", "tuple") and isinstance(data, list):
        if depth >= _MAX_DEPTH:
            raise ValueError("reference argument is nested too deeply")
        items = [_decode(item, nodes, depth + 1) for item in data]
        return items if kind == "list" else tuple(items)
    if kind == "dtype":
        return _torch_attribute(data, torch.dtype)
    if kind == "device" and isinstance(data, str):
        try:
            return torch.device(data)
        except RuntimeError:
            raise ValueError(f"reference names an unknown device {data!r}") from None
    if kind == "layout":
        return _torch_attribute(data, torch.layout)
    if kind == "memory_format":
        return _torch_attribute(data, torch.memory_format)
    raise ValueError(f"reference argument has an unexpected form {kind!r}")


def _target_name(target: Any) -> str:
    if isinstance(target, torch._ops.OpOverload):
        if not _supported(target):
            raise TypeError(f"cannot save operation {target}")
        return f"{target.namespace}::{target._opname}.{target._overloadname}"
    for name, function in _OPERATORS.items():
        if target is function:
            return f"operator::{name}"
    raise TypeError(f"cannot save call target {target!r}")


def _resolve(name: Any) -> Any:
    if isinstance(name, str):
        namespace, _, qualified = name.partition("::")
        if namespace == "operator" and qualified in _OPERATORS:
            return _OPERATORS[qualified]
        packet, _, overload = qualified.partition(".")
        if (
            namespace in _NAMESPACES
            and packet.isidentifier()
            and overload.isidentifier()
        ):
            namespace_ops = getattr(torch.ops, namespace)
            target = getattr(getattr(namespace_ops, packet, None), overload, None)
            # torch.ops attributes include more than operators: accept only the
            # supported operator that this exact name describes.
            if (
                isinstance(target, torch._ops.OpOverload)
                and _supported(target)
                and _target_name(target) == name
            ):
                return target
    raise ValueError(f"reference names an unsupported operation {name!r}")


def encode_reference(reference: torch.fx.GraphModule) -> dict[str, Any]:
    """Describe a captured reference as JSON data."""
    nodes, constants = [], {}
    for node in reference.graph.nodes:
        entry: dict[str, Any] = {"name": node.name, "op": node.op}
        if node.op == "call_function":
            entry["target"] = _target_name(node.target)
            if not _supported_call(node.target, node.args, node.kwargs):
                raise TypeError(f"cannot save the arguments of {node.target}")
        elif node.op == "get_attr":
            target = str(node.target)
            if not _constant_name(target):
                raise TypeError(f"cannot save reference constant {target!r}")
            value = getattr(reference, target)
            if isinstance(value, torch.Tensor):
                raise TypeError("cannot save a tensor constant in a reference")
            constants[target] = _encode(value)
            entry["target"] = target
        elif node.op not in ("placeholder", "output"):
            raise TypeError(f"cannot save reference node {node.op!r}")
        entry["args"] = _encode(node.args)
        entry["kwargs"] = {key: _encode(value) for key, value in node.kwargs.items()}
        nodes.append(entry)
    return {"version": _VERSION, "nodes": nodes, "constants": constants}


def decode_reference(data: Any) -> torch.fx.GraphModule:
    """Rebuild a reference from :func:`encode_reference` data, running no code from it.

    Saved names only link values: inputs become ``arg0``, ``arg1``, ... and the
    other values ``value<position>``, as in the controller's own references.
    """
    if not isinstance(data, dict) or data.get("version") != _VERSION:
        raise ValueError("reference has an unsupported format version")
    constants, entries = data.get("constants"), data.get("nodes")
    if not isinstance(constants, dict) or not isinstance(entries, list) or not entries:
        raise ValueError("reference has an unexpected form")
    root = torch.nn.Module()
    for name, value in constants.items():
        if not _constant_name(name):
            raise ValueError(f"reference constant has an invalid name {name!r}")
        setattr(root, name, _decode(value, {}))
    graph = torch.fx.Graph()
    nodes: dict[str, torch.fx.Node] = {}
    inputs = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("reference node has an unexpected form")
        name, op, target = entry.get("name"), entry.get("op"), entry.get("target")
        args, kwargs = _decode(entry.get("args"), nodes), entry.get("kwargs")
        if not isinstance(name, str) or name in nodes:
            raise ValueError(f"reference node has an invalid or repeated name {name!r}")
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise ValueError("reference node has an unexpected form")
        kwargs = {key: _decode(value, nodes) for key, value in kwargs.items()}
        if (op == "output") != (index == len(entries) - 1):
            raise ValueError("reference must end with its only output")
        if op == "output" and len(args) == 1:
            node = graph.output(args[0])
        elif op == "placeholder":
            node = graph.placeholder(f"arg{inputs}")
            inputs += 1
        elif op == "get_attr":
            if not isinstance(target, str) or target not in constants:
                raise ValueError("reference reads a constant it does not define")
            node = graph.create_node(op, target, name=f"value{index + 1}")
        elif op == "call_function":
            function = _resolve(target)
            if not _supported_call(function, args, kwargs):
                raise ValueError(
                    f"reference calls {target!r} with unexpected arguments"
                )
            node = graph.create_node(
                op, function, args, kwargs, name=f"value{index + 1}"
            )
        else:
            raise ValueError(f"reference has an unsupported node {op!r}")
        nodes[name] = node
    graph.lint()
    return torch.fx.GraphModule(root, graph)


def save_reference(reference: torch.fx.GraphModule, path: Path) -> None:
    path.write_text(json.dumps(encode_reference(reference), indent=1), encoding="utf-8")


def load_reference(path: str | Path) -> torch.fx.GraphModule:
    with open(path, "rb") as file:
        content = file.read(_MAX_BYTES + 1)
    if len(content) > _MAX_BYTES:
        raise ValueError(f"{path} is larger than a saved reference can be")
    try:
        data = json.loads(content.decode("utf-8"))
    except RecursionError:
        raise ValueError(f"{path} is nested too deeply") from None
    return decode_reference(data)
