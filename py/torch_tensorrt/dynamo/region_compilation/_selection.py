# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Mark the regions that ``regions=`` rules select on a traced graph.

A rule selects module calls by path, class or module object. Every traced
operation records the modules it ran in, so the pass finds each selected
call's operations and writes the markers and tags that a scope written in
``forward`` would have left. Capture then treats both kinds of region alike.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch_tensorrt.region import _marker_ops
from torch_tensorrt.region._constants import (
    BACKEND_KEY,
    NAME_KEY,
    OPERATOR_KEY,
    PLUGIN_KEY,
    ROUTE_KEY,
    SELECTED_KEY,
    TOKEN_KEY,
)
from torch_tensorrt.region._context import _remember_operator, _remember_plugin
from torch_tensorrt.region._errors import RegionCaptureError
from torch_tensorrt.region._rules import (
    ExecuteInTorch,
    FusedKernel,
    FusedOp,
    RegionRule,
    check_rules,
)

from ._capture import (
    _GUARDS,
    _POSITION_TOKENS,
    _describe,
    _marker_kind,
    is_region_hop,
)

logger = logging.getLogger(__name__)

_CALLS = ("call_function", "call_method", "call_module")


@dataclass(frozen=True, eq=False)
class ModulePaths:
    """A selector matched against the model: the module paths it selects.

    An exported program records each operation's modules by path.
    """

    label: str
    paths: frozenset[str]


@dataclass(frozen=True, eq=False)
class ModuleSet:
    """A selector matched against the model: the module objects it selects.

    A torch.compile graph records each operation's modules by identity, and by
    a path that starts at whichever module's ``forward`` was traced, which is
    not always the model.
    """

    label: str
    # id(module) to the module's name in the model, or None when unknown.
    names: Mapping[int, Optional[str]]


@dataclass
class _Call:
    """The operations of one call of a selected module."""

    label: str
    name: str
    # The call's entry in the program's record of module calls: the path, plus
    # "@1" for the second call of a module.
    recorded_as: str
    path: str
    rule: RegionRule
    nodes: list[torch.fx.Node]


def module_path(raw: str) -> str:
    """Write a traced module path the way ``named_modules`` does.

    torch.compile records ``L['self'].layers[0].mlp`` or
    ``L['self']._modules['layers']._modules['0']``; export records
    ``layers.0.mlp``.
    """
    path = re.sub(r"^[LG]\[['\"][^'\"]*['\"]\]\.?", "", raw)
    path = re.sub(
        r"(^|\.)_(?:modules|parameters|buffers)\[['\"]([^'\"]+)['\"]\]", r"\1\2", path
    )
    path = re.sub(r"\[['\"]?([^'\"\]]+)['\"]?\]", r".\1", path)
    return path.strip(".")


def _glob(pattern: str, path: str) -> bool:
    """Match dotted paths: ``*`` is one name, ``**`` is any number of names."""
    wanted, names = pattern.split("."), path.split(".") if path else []

    def match(i: int, j: int) -> bool:
        if i == len(wanted):
            return j == len(names)
        if wanted[i] == "**":
            return any(match(i + 1, k) for k in range(j, len(names) + 1))
        return (
            j < len(names)
            and fnmatch.fnmatchcase(names[j], wanted[i])
            and match(i + 1, j + 1)
        )

    return match(0, 0)


def _class_name(module_type: Any) -> str:
    if isinstance(module_type, type):
        return f"{module_type.__module__}.{module_type.__qualname__}"
    return str(module_type)


def _recorded_class(recorded: Any) -> Optional[type]:
    """The class an exported program names, if this process has loaded it."""
    if isinstance(recorded, type):
        return recorded
    parts = str(recorded).split(".")
    for split in range(len(parts) - 1, 0, -1):
        found: Any = sys.modules.get(".".join(parts[:split]))
        for name in parts[split:]:
            found = getattr(found, name, None)
        if isinstance(found, type):
            return found
    return None


def _module_id(key: str) -> Optional[int]:
    """The module identity in a torch.compile module-stack key such as ``1234@1``."""
    number = str(key).partition("@")[0]
    return int(number) if number.isdigit() else None


def _selects(selector: Any, key: str, path: str, recorded_class: Any) -> bool:
    if isinstance(selector, ModuleSet):
        return _module_id(key) in selector.names
    if not path:
        # The whole model is not a region.
        return False
    if isinstance(selector, ModulePaths):
        return path in selector.paths
    if isinstance(selector, str):
        return _glob(selector, path)
    known = _recorded_class(recorded_class)
    if known is not None:
        return issubclass(known, selector)
    # The program names a class this process has not loaded.
    return _class_name(recorded_class) == _class_name(selector)


def _label(selector: Any) -> str:
    if isinstance(selector, (ModulePaths, ModuleSet)):
        return selector.label
    if isinstance(selector, str):
        return repr(selector)
    if isinstance(selector, torch.nn.Module):
        return f"<{type(selector).__name__} module>"
    return _class_name(selector)


def _sample(names: Any, limit: int = 8) -> str:
    unique = list(dict.fromkeys(names))
    shown = ", ".join(repr(name) for name in unique[:limit])
    return shown + (", ..." if len(unique) > limit else "") if unique else "none"


def _matched_modules(
    model: torch.nn.Module, regions: Mapping[Any, RegionRule]
) -> list[tuple[Any, RegionRule, dict[int, list[str]]]]:
    """Match each rule against a model: the names of every module it selects."""
    check_rules(regions)
    modules = [
        (name, module)
        for name, module in model.named_modules(remove_duplicate=False)
        if name
    ]
    matched = []
    for selector, rule in regions.items():
        if isinstance(selector, torch.nn.Module):
            selected = {id(module) for _, module in modules if module is selector}
        elif isinstance(selector, str):
            selected = {id(module) for name, module in modules if _glob(selector, name)}
        else:
            selected = {
                id(module) for _, module in modules if isinstance(module, selector)
            }
        if not selected:
            raise RegionCaptureError(
                f"Region rule {_label(selector)} matched no submodule of the model; "
                f"its submodules include {_sample(name for name, _ in modules)}"
            )
        containers = [
            name
            for name, module in modules
            if id(module) in selected
            and type(module).forward is torch.nn.Module.forward
        ]
        if len(containers) == len(
            [name for name, module in modules if id(module) in selected]
        ):
            # A ModuleList or ModuleDict holds modules and is never called.
            raise RegionCaptureError(
                f"Region rule {_label(selector)} selects only containers that "
                f"are never called ({_sample(containers, 3)}); select their "
                f"children, for example {containers[0] + '.*'!r}"
            )
        # A module registered under two names is called under either.
        names: dict[int, list[str]] = {}
        for name, module in modules:
            if id(module) in selected:
                names.setdefault(id(module), []).append(name)
        matched.append((selector, rule, names))
    return matched


def resolve_rules(
    model: torch.nn.Module, regions: Mapping[Any, RegionRule]
) -> tuple[dict[ModulePaths, RegionRule], tuple[str, ...]]:
    """Match rules against a model before it is exported.

    Returns the rules keyed by the paths they select, and the paths whose
    calls the export should record because a rule replaces those calls. The
    model knows subclasses and every name of a shared module; an exported
    program knows neither.
    """
    resolved: dict[ModulePaths, RegionRule] = {}
    replaced: list[str] = []
    for selector, rule, names in _matched_modules(model, regions):
        paths = frozenset(name for group in names.values() for name in group)
        resolved[ModulePaths(_label(selector), paths)] = rule
        if not isinstance(rule, ExecuteInTorch):
            replaced.extend(sorted(paths))
    return resolved, tuple(dict.fromkeys(replaced))


def bind_rules(
    model: torch.nn.Module, regions: Mapping[Any, RegionRule]
) -> dict[ModuleSet, RegionRule]:
    """Match rules against a model before torch.compile traces it."""
    return {
        ModuleSet(
            _label(selector), {key: group[0] for key, group in names.items()}
        ): rule
        for selector, rule, names in _matched_modules(model, regions)
    }


def find_compiled_model() -> Optional[torch.nn.Module]:
    """The model whose torch.compile call is running the Torch-TensorRT backend."""
    frame: Any = sys._getframe(1)
    while frame is not None:
        owner = frame.f_locals.get("self")
        if isinstance(owner, torch._dynamo.eval_frame.OptimizedModule):
            model = owner._orig_mod
            return model if isinstance(model, torch.nn.Module) else None
        frame = frame.f_back
    return None


def _traced_frame_module() -> Optional[torch.nn.Module]:
    """The module whose forward torch.compile is tracing into the current graph."""
    try:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
        owner = getattr(tx.output, "root_tx", tx).f_locals.get("self")
    except Exception:
        return None
    return owner if isinstance(owner, torch.nn.Module) else None


def _reject_split_modules(
    model: Optional[torch.nn.Module], rules: Mapping[Any, RegionRule]
) -> None:
    """Reject a rule whose module torch.compile traced in pieces.

    A graph break in a module's forward, or in its caller's, makes
    torch.compile trace the forward as graphs of its own. Their operations
    record the modules below that forward, not the forward's own module, so
    the rule would silently select nothing. ``model`` is None when a function
    that calls the model was compiled.
    """
    traced = _traced_frame_module()
    if traced is None or traced is model:
        return
    modules = {} if model is None else {id(part): part for part in model.modules()}
    for selector in rules:
        if isinstance(selector, ModuleSet):
            split = next(
                (
                    name or type(traced).__name__
                    for key, name in selector.names.items()
                    if key == id(traced)
                    or (
                        key in modules
                        and any(part is traced for part in modules[key].modules())
                    )
                ),
                None,
            )
        elif isinstance(selector, type) and isinstance(traced, selector):
            split = type(traced).__name__
        else:
            continue
        if split is not None:
            raise RegionCaptureError(
                f"Region rule {_label(selector)}: torch.compile traced part of "
                f"{split!r} as a graph of its own, which a graph break in "
                "that module's forward or in its caller's causes. The rule "
                "cannot place the whole call in a region; remove the graph "
                "break, or write the scope in the module's forward"
            )


def _frames(node: torch.fx.Node) -> list[tuple[str, str, Any]]:
    stack = node.meta.get("nn_module_stack") or {}
    return [(key, module_path(path), kind) for key, (path, kind) in stack.items()]


def _selected_calls(
    gm: torch.fx.GraphModule,
    regions: Mapping[Any, RegionRule],
    *,
    require_match: bool,
) -> list[_Call]:
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    operations = [
        node
        for node in gm.graph.nodes
        if node.op in _CALLS and node.target not in _GUARDS and not _marker_kind(node)
    ]
    frames = {node: _frames(node) for node in operations}
    calls: list[_Call] = []
    for selector, rule in regions.items():
        label = _label(selector)
        found: dict[tuple[str, str], list[torch.fx.Node]] = {}
        for node in operations:
            # The outermost selected module owns the operation.
            for key, path, kind in frames[node]:
                if _selects(selector, key, path, kind):
                    found.setdefault((key, path), []).append(node)
                    break
        if not found and require_match:
            raise RegionCaptureError(
                f"Region rule {label} matched no module call in the traced "
                "program; its modules include "
                + _sample(
                    path for stack in frames.values() for _, path, _ in stack if path
                )
                + ". A container such as a ModuleList is never called: select "
                "its children"
            )
        for (key, path), nodes in found.items():
            name = path
            if isinstance(selector, ModuleSet):
                name = selector.names.get(_module_id(key) or 0) or path
            repeat = key[key.rfind("@") :] if "@" in key else ""
            nodes = _call_operations(nodes, frames, name)
            calls.append(_Call(label, name, path + repeat, path, rule, nodes))
    calls.sort(key=lambda call: order[call.nodes[0]])
    owner: dict[torch.fx.Node, _Call] = {}
    for call in calls:
        for node in call.nodes:
            other = owner.setdefault(node, call)
            if other is not call:
                raise RegionCaptureError(
                    f"Region rules {other.label} and {call.label} both select "
                    f"{node.name!r}, in {other.name!r} and {call.name!r}; "
                    "regions cannot nest or overlap"
                )
    return calls


def _call_operations(
    nodes: list[torch.fx.Node],
    frames: Mapping[torch.fx.Node, list[tuple[str, str, Any]]],
    name: str,
) -> list[torch.fx.Node]:
    """One module call's operations, with what the trace recorded between them.

    A trace records a call's operations one after another and numbers repeated
    calls of a module, so the operations between a call's first and last ran
    inside it. Some record no module, such as a grad-mode switch.
    """
    selected = set(nodes)
    members = []
    strays = []
    node = nodes[0]
    while True:
        if node in selected:
            members.append(node)
        elif _marker_kind(node) or is_region_hop(node):
            begin = node if _marker_kind(node) != "end" else node.args[0]
            scope = begin.args[0] if _marker_kind(begin) else None
            raise RegionCaptureError(
                f"A region rule selects {name!r}, whose operations surround "
                f"{scope or 'a scope'!r}, a region written in the model's "
                "source; regions cannot nest"
            )
        elif node.op in _CALLS and node.target not in _GUARDS:
            if frames.get(node):
                strays.append(node)
            else:
                members.append(node)
        if node is nodes[-1]:
            break
        node = node.next
    if strays:
        raise RegionCaptureError(
            f"The operations of {name!r} are not one connected piece of the "
            "program; these run between its parts: "
            + ", ".join(f"{stray.name} ({stray.target})" for stray in strays[:3])
        )
    return members


def _fake_mode(gm: torch.fx.GraphModule) -> Any:
    for node in gm.graph.nodes:
        value = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(value, torch._subclasses.FakeTensor):
            return value.fake_mode
    return None


def _listed_inputs(
    call: _Call,
    live_ins: Sequence[torch.fx.Node],
    state: Mapping[str, str],
    call_order: Optional[Sequence[str]],
    order_hint: str,
    constants: frozenset[str] = frozenset(),
) -> Optional[list[torch.fx.Node]]:
    """Resolve a rule's ``inputs`` to the graph values the module reads.

    ``constants`` names the placeholders of tensors created in ``forward``.
    """
    rule = call.rule
    assert isinstance(rule, (FusedOp, FusedKernel))
    where = f"Region rule {call.label} at {call.name!r}"
    read = {node.name: node for node in live_ins}
    names = {node.name: state[node.name] for node in live_ins if node.name in state}
    received = [node for node in live_ins if node.name not in state]

    def how_to_list(node: torch.fx.Node) -> str:
        if node.name in names:
            fqn = names[node.name]
            prefix = f"{call.path}."
            listed_as = repr(fqn[len(prefix) :] if fqn.startswith(prefix) else fqn)
            if node.name in constants:
                listed_as += " (a tensor the module creates in forward)"
            return listed_as
        if call_order is not None and node.name in call_order:
            return f"position {call_order.index(node.name)}"
        if len(received) == 1:
            return "position 0"
        return f"the tensor {_describe([node])}"

    def unrecorded_order() -> RegionCaptureError:
        # First-use order is not argument order: do not guess.
        return RegionCaptureError(
            f"{where}: the module reads {len(received)} tensors from its "
            f"caller ({_describe(received)}), and this program does not record "
            f"the order its call passes them. {order_hint}"
        )

    if rule.inputs is None:
        if len(live_ins) > 1:
            if call_order is None and len(received) > 1:
                raise unrecorded_order()
            # The caller's tensors in call order, then the module's own.
            ways = sorted(
                (how_to_list(node) for node in received),
                key=lambda way: (not way.startswith("position"), way),
            ) + [how_to_list(node) for node in live_ins if node.name in names]
            raise RegionCaptureError(
                f"{where}: the module reads {len(live_ins)} tensors; list them "
                "with inputs=[...] in the order the replacement takes them. "
                "They are: " + ", ".join(ways)
            )
        return None
    listed = []
    for item in rule.inputs:
        if isinstance(item, int):
            if call_order is not None:
                if item >= len(call_order):
                    raise RegionCaptureError(
                        f"{where}: input position {item} is out of range; the "
                        f"call passes {len(call_order)} tensors"
                    )
                if call_order[item] not in read:
                    raise RegionCaptureError(
                        f"{where}: the module does not read the tensor at "
                        f"position {item} of its call, so the replacement "
                        "cannot take it"
                    )
                listed.append(read[call_order[item]])
            elif len(received) <= 1:
                if item >= len(received):
                    raise RegionCaptureError(
                        f"{where}: input position {item} is out of range; the "
                        f"module reads {len(received)} tensor(s) from its caller"
                    )
                listed.append(received[item])
            else:
                raise unrecorded_order()
            continue
        candidates = [
            name for name, fqn in names.items() if fqn in (f"{call.path}.{item}", item)
        ] or [
            # A tied weight is recorded once, under another module's name.
            name
            for name, fqn in names.items()
            if fqn.endswith(f".{item}") and not fqn.startswith(f"{call.path}.")
        ]
        if len(candidates) != 1:
            raise RegionCaptureError(
                f"{where}: cannot find the parameter or buffer {item!r}; the "
                "module reads "
                + (", ".join(how_to_list(read[name]) for name in names) or "none")
            )
        listed.append(read[candidates[0]])
    missing = [node for node in live_ins if node not in listed]
    if missing:
        raise RegionCaptureError(
            f"{where}: inputs must list every tensor the module reads; add "
            + ", ".join(how_to_list(node) for node in missing)
        )
    return listed


def select_regions(
    gm: torch.fx.GraphModule,
    regions: Mapping[Any, RegionRule],
    *,
    state: Mapping[str, str],
    call_inputs: Optional[Callable[[str], Optional[Sequence[str]]]] = None,
    call_outputs: Optional[Callable[[str], Optional[Sequence[str]]]] = None,
    require_match: bool = True,
    order_hint: str = "",
    constants: frozenset[str] = frozenset(),
) -> list[str]:
    """Insert region markers for the module calls that ``regions`` selects.

    Args:
        gm: An exported program's graph, or the graph torch.compile traced.
        regions: ``{selector: rule}``; a selector is a module path, a module
            class, a :class:`ModulePaths` or a :class:`ModuleSet`.
        state: Placeholder name to parameter or buffer name.
        call_inputs: Returns the names of the tensors a module call passes, in
            call order, or None when the trace did not record them.
        call_outputs: The same for the tensors a module call returns.
        require_match: Whether a rule that selects nothing is an error. A
            model that torch.compile splits into several graphs has each
            module in one of them.
        order_hint: How to make the call order known, for the error raised
            when a rule needs it.
        constants: Placeholder names of tensors a module creates in forward.

    Returns:
        The names of the regions, in program order.
    """
    if _marker_ops.CAPTURE_ERROR:
        raise RegionCaptureError(_marker_ops.CAPTURE_ERROR)
    calls = _selected_calls(gm, regions, require_match=require_match)
    fake_mode = _fake_mode(gm)
    names = []
    for call in calls:
        span = set(call.nodes)
        live_ins: list[torch.fx.Node] = []
        for node in call.nodes:
            for source in node.all_input_nodes:
                if source not in span and source not in live_ins:
                    live_ins.append(source)
        rule = call.rule
        name = rule.name or call.name
        backend, route, plugin, operator, listed = "eager", "torch", "", "", None
        returned = None
        if isinstance(rule, ExecuteInTorch):
            backend = rule.backend
        else:
            order = None if call_inputs is None else call_inputs(call.recorded_as)
            listed = _listed_inputs(call, live_ins, state, order, order_hint, constants)
            if call_outputs is not None:
                returned = call_outputs(call.recorded_as)
            if isinstance(rule, FusedOp):
                operator = _remember_operator(rule.op)
            else:
                route = "plugin"
                plugin = _remember_plugin(
                    rule.kernel,
                    rule.signature,
                    rule.constexprs,
                    rule.grid,
                    rule.extra_args_fn,
                    rule.num_warps,
                    rule.num_stages,
                )
        first, last = call.nodes[0], call.nodes[-1]
        with gm.graph.inserting_before(first):
            begin = gm.graph.call_function(
                _marker_ops.BEGIN, (name, backend, route, plugin, operator, listed)
            )
        if fake_mode is not None:
            with fake_mode:
                begin.meta["val"] = torch.empty((), dtype=torch.uint8, device="cpu")
        with gm.graph.inserting_after(last):
            end = gm.graph.call_function(_marker_ops.END, (begin,))
        end.meta["val"] = None
        tags = {
            TOKEN_KEY: -next(_POSITION_TOKENS),
            NAME_KEY: name,
            BACKEND_KEY: backend,
            ROUTE_KEY: route,
            PLUGIN_KEY: plugin,
            OPERATOR_KEY: operator,
        }
        # Every operation between the markers is a member, guards included.
        node = begin.next
        while node is not end:
            if node.op in _CALLS:
                node.meta["custom"] = {**node.meta.get("custom", {}), **tags}
            node = node.next
        # What capture needs to know about the module call a rule selected.
        end.meta["custom"] = {
            **tags,
            SELECTED_KEY: {
                "outputs": None if returned is None else tuple(returned),
                "hint": order_hint,
            },
        }
        names.append(name)
    if calls:
        gm.graph.lint()
        gm.recompile()
    logger.debug("Region rules selected: %s", names)
    return names


def _check_given_rules(regions: Any) -> None:
    """Check the rules a caller passed, apart from those already matched."""
    check_rules(
        {
            key: rule
            for key, rule in regions.items()
            if not isinstance(key, (ModulePaths, ModuleSet))
        }
        if isinstance(regions, Mapping)
        else regions
    )


def select_exported_regions(
    ep: torch.export.ExportedProgram, regions: Mapping[Any, RegionRule]
) -> list[str]:
    """Mark the regions that rules select in an exported program, in place."""
    _check_given_rules(regions)
    for selector in regions:
        if isinstance(selector, (torch.nn.Module, ModuleSet)):
            raise RegionCaptureError(
                f"Region rule {_label(selector)} selects a module object, which "
                "an exported program does not hold; select it by path or "
                "class, or compile the nn.Module"
            )
    signature = ep.graph_signature
    state = {
        **signature.inputs_to_parameters,
        **signature.inputs_to_buffers,
        **signature.inputs_to_lifted_tensor_constants,
    }
    tensor = torch.export.graph_signature.TensorArgument
    recorded = {
        entry.fqn: entry.signature
        for entry in ep.module_call_graph
        if entry.signature is not None
    }

    def names(arguments: Any) -> list[str]:
        return [argument.name for argument in arguments if isinstance(argument, tensor)]

    from torch_tensorrt.region._session import current_session

    # Set when torch_tensorrt.compile traced the model but torch.export could
    # not record how the selected modules are called.
    unrecorded = getattr(current_session(), "call_record_failure", None)
    return select_regions(
        ep.graph_module,
        regions,
        state=state,
        call_inputs=lambda call: (
            names(recorded[call].inputs) if call in recorded else None
        ),
        call_outputs=lambda call: (
            names(recorded[call].outputs) if call in recorded else None
        ),
        order_hint=(
            f"torch.export could not record it ({unrecorded}). Select a module "
            "that takes and returns one tensor, or write the scope in the "
            "module's forward"
            if unrecorded
            else "Select a module that takes and returns one tensor, or export "
            "with torch.export.export(..., preserve_module_call_signature="
            "(<module path>,)); torch_tensorrt.compile records the order "
            "itself when it is given the nn.Module"
        ),
        constants=frozenset(signature.inputs_to_lifted_tensor_constants),
    )


def _rules_without_a_model(regions: Mapping[Any, RegionRule]) -> dict[Any, RegionRule]:
    """Rules for a compiled function, where no model says what a path starts from."""
    rules: dict[Any, RegionRule] = {}
    for selector, rule in regions.items():
        if isinstance(selector, str):
            # The graph's paths start at the module whose forward was traced,
            # which need not be the model.
            raise RegionCaptureError(
                f"Region rule {selector!r} selects by path, and the "
                "Torch-TensorRT backend cannot tell which model the path "
                "starts from. Compile the model itself, with "
                "torch.compile(model, backend='torch_tensorrt', ...) or "
                "torch_tensorrt.compile(model, ir='torch_compile', "
                "regions=...), or select by module class or module object"
            )
        if isinstance(selector, torch.nn.Module):
            rules[ModuleSet(_label(selector), {id(selector): None})] = rule
        else:
            rules[selector] = rule
    return rules


def select_traced_regions(
    gm: torch.fx.GraphModule,
    regions: Mapping[Any, RegionRule],
    model: Optional[torch.nn.Module] = None,
) -> list[str]:
    """Mark the regions that rules select in a graph torch.compile traced.

    ``model`` is the model being compiled, when the caller knows it.
    """
    _check_given_rules(regions)
    bound: dict[Any, RegionRule] = {}
    unbound: dict[Any, RegionRule] = {}
    for selector, rule in regions.items():
        (bound if isinstance(selector, ModuleSet) else unbound)[selector] = rule
    model = model if model is not None else find_compiled_model()
    if unbound:
        bound.update(
            bind_rules(model, unbound)
            if model is not None
            else _rules_without_a_model(unbound)
        )
    _reject_split_modules(model, bound)
    if any(isinstance(key, ModuleSet) for key in bound) and any(
        _module_id(key) is None
        for node in gm.graph.nodes
        for key in (node.meta.get("nn_module_stack") or {})
    ):
        raise RegionCaptureError(
            "This PyTorch version does not record which module object each "
            "torch.compile operation ran in, so rules cannot select modules; "
            "write the scopes in forward instead"
        )
    state = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        source = getattr(node.meta.get("grapharg"), "source", None)
        name = getattr(source, "name", None)
        raw = name() if callable(name) else name
        if isinstance(raw, str) and re.search(r"_(?:parameters|buffers)\[", raw):
            state[node.name] = module_path(raw)
    return select_regions(
        gm,
        bound,
        state=state,
        require_match=False,
        order_hint=(
            "Select a module that takes and returns one tensor, write the "
            "scope in the module's forward with torch_tensorrt.region.fused_op "
            "or fused_kernel, or compile ahead of time with "
            "torch_tensorrt.compile(model, ir='dynamo', regions=...)"
        ),
    )
