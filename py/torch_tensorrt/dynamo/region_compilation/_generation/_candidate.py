# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Python kernel DSL backends that validate generated source and reuse frontends."""

from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import importlib
import importlib.util
import keyword
import logging
import math
import sys
from abc import ABC, abstractmethod
from functools import cached_property, partial
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch

from ._backends import (
    SharedMemoryLimitError,
    _dimensions,
    _proposal_schema,
    _target_metadata,
)
from ._util import sha256_text

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec

_LOGGER = logging.getLogger(__name__)
# Kernel code may use only these builtins, and cannot rebind them, since the
# call check trusts their names.
_BUILTINS = {"abs", "bool", "float", "int", "len", "max", "min", "range", "tuple"}
# Names that reach a builtin wherever kernel code has not assigned them. Those
# other than _BUILTINS, such as open, act on the host when a DSL calls them
# while compiling, so a variable named like one is usable only where kernel
# code certainly assigned it.
_PYTHON_BUILTINS = frozenset(name for name in dir(builtins) if not name.startswith("_"))
# Tensor methods accepted even when the installed DSL's types cannot be inspected.
_METHODS = frozenset(
    {
        "astype",
        "broadcast_to",
        "expand_dims",
        "max",
        "min",
        "permute",
        "reshape",
        "split",
        "sum",
        "to",
        "trans",
    }
)
_ALIASES = frozenset({"triton", "tl", "ct", "libdevice"})
# Keyword arguments JITFunction.run binds next to the kernel's own parameters.
_TRITON_LAUNCH_KEYWORDS = frozenset(
    {"num_warps", "num_stages", "debug", "instrumentation_mode"}
)
_FORBIDDEN = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Match,
    ast.Nonlocal,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def _identifier(value: Any, aliases: frozenset[str] = _ALIASES) -> bool:
    return (
        isinstance(value, str)
        and value.isidentifier()
        and not keyword.iskeyword(value)
        and "__" not in value
        and value not in aliases
    )


def _targets(target: ast.expr) -> set[str]:
    """The names that an assignment, ``for`` or ``del`` target binds or unbinds."""
    names: set[str] = set()
    pending = [target]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            pending += node.elts
        elif isinstance(node, ast.Starred):
            pending.append(node.value)
    return names


def _unassigned_reads(
    function: ast.FunctionDef, assigned: set[str] | frozenset[str]
) -> set[ast.Name]:
    """Uses of builtin names where the function's own assignment may not apply.

    A variable named like a builtin is certainly the function's own only after
    an assignment earlier in the same block or in one enclosing it, or within
    the comprehension that binds it, past its first iterable. Kernel compilers
    do not keep Python's function-wide scope: CuTe DSL moves loop and branch
    bodies into nested functions, and Triton drops what a block assigned when
    the block ends. Elsewhere, a use can reach the builtin itself. ``assigned``
    holds the names bound throughout, such as parameters. Walrus targets do not
    count, and an augmented assignment uses its target.
    """
    unassigned: set[ast.Name] = set()

    def use(name: ast.Name, bound: frozenset[str]) -> None:
        if name.id in _PYTHON_BUILTINS and name.id not in bound:
            unassigned.add(name)

    # Stacks rather than recursion: expressions such as long sums and chains
    # of elif nest deeper than Python's recursion limit.
    def read(root: ast.AST, outer: frozenset[str]) -> None:
        pending = [(root, outer)]
        while pending:
            node, bound = pending.pop()
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    use(node, bound)
            elif isinstance(
                node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
            ):
                # The first iterable is evaluated outside the comprehension.
                for generator in node.generators:
                    pending.append((generator.iter, bound))
                    bound = bound | _targets(generator.target)
                    pending.append((generator.target, bound))
                    pending += [(condition, bound) for condition in generator.ifs]
                parts = (
                    [node.key, node.value]
                    if isinstance(node, ast.DictComp)
                    else [node.elt]
                )
                pending += [(part, bound) for part in parts]
            else:
                pending += [(child, bound) for child in ast.iter_child_nodes(node)]

    # A block starts from what its enclosing blocks assigned before it, and
    # what it assigns ends with it, so each block is checked on its own.
    blocks = [(function.body, frozenset(assigned))]
    while blocks:
        statements, bound = blocks.pop()
        for statement in statements:
            if isinstance(statement, ast.For):
                read(statement.iter, bound)
                read(statement.target, bound)
                blocks.append((statement.body, bound | _targets(statement.target)))
                blocks.append((statement.orelse, bound))
            elif isinstance(statement, (ast.While, ast.If)):
                read(statement.test, bound)
                blocks += [(statement.body, bound), (statement.orelse, bound)]
            else:
                read(statement, bound)
                if isinstance(statement, ast.AugAssign):
                    if isinstance(statement.target, ast.Name):
                        use(statement.target, bound)
                elif isinstance(statement, ast.Assign):
                    bound = bound.union(*map(_targets, statement.targets))
                elif isinstance(statement, ast.AnnAssign):
                    if statement.value is not None:
                        bound = bound | _targets(statement.target)
                elif isinstance(statement, ast.Delete):
                    bound = bound.difference(*map(_targets, statement.targets))
    return unassigned


def _dsl_schema(**options: Any) -> dict[str, Any]:
    # anyOf rather than a list of types: structured-output validators in
    # strict mode reject union "type" keywords.
    return _proposal_schema(
        source={"type": "string"},
        kernel_name={"type": "string"},
        grid=_dimensions(2147483647),
        constants={
            "type": "array",
            "items": _proposal_schema(
                name={"type": "string"},
                value={"anyOf": [{"type": "boolean"}, {"type": "number"}]},
            ),
        },
        **options,
    )


def _line(node: ast.stmt) -> str:
    """Name a module-level statement for an error message."""
    text = ast.unparse(node).strip().splitlines()
    snippet = text[0] if text else type(node).__name__
    if len(snippet) > 80:
        snippet = snippet[:77] + "..."
    return f"line {node.lineno}: {snippet!r}"


def _constants(candidate: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item["value"] for item in candidate["constants"]}


def _shared_memory_error(
    candidate: dict[str, Any], required: int, limit: int, compiled: Any
) -> SharedMemoryLimitError:
    from torch_tensorrt.kernels._triton import shared_memory_allocations

    allocations = shared_memory_allocations(compiled)
    stages = candidate["num_stages"]
    message = (
        f"Generated Triton kernel needs {required} bytes of shared memory with "
        f"num_stages={'null (Triton defaults to 3)' if stages is None else stages}, "
        f"but a TensorRT plugin launch allows at most {limit} "
        "(target.plugin_max_shared_memory_bytes); Triton's own launch could use "
        "more, the plugin cannot. "
        + (
            f"Largest shared-memory buffers: {'; '.join(allocations)}. "
            if allocations
            else ""
        )
        + "Keep fewer or smaller tiles in shared memory, such as tl.dot operand "
        "and accumulator tiles, or use num_stages=1"
    )
    return SharedMemoryLimitError(
        message, required=required, limit=limit, allocations=allocations
    )


def _program_index(node: ast.expr, pids: set[str], tiles: set[str]) -> bool:
    """Whether ``node`` is scalar arithmetic on program ids and non-tile names."""
    found = False

    def visit(item: ast.expr) -> bool:
        nonlocal found
        if isinstance(item, ast.Call):
            found = found or ast.unparse(item.func) == "tl.program_id"
            return ast.unparse(item.func) == "tl.program_id"
        if isinstance(item, ast.Name):
            found = found or item.id in pids
            return item.id not in tiles
        if isinstance(item, ast.Constant):
            return type(item.value) is int
        if isinstance(item, ast.BinOp):
            return visit(item.left) and visit(item.right)
        if isinstance(item, ast.UnaryOp):
            return visit(item.operand)
        return False

    return visit(node) and found


def _halvable_tiles(
    candidate: dict[str, Any], dimensions: set[int]
) -> tuple[dict[str, str | None], list[str]]:
    """Tile constants that shared-memory fitting may halve, with their trip counts.

    A tile is an int constant naming a ``tl.arange`` extent. It is kept when it
    equals a boundary dimension, or when it multiplies a program id, directly
    or through scalar index arithmetic, because the literal grid would then
    leave part of the output unwritten. A non-tile constant that bounds a loop
    and whose product with the tile is a boundary dimension is the tile's trip
    count, and doubles with it.
    """
    tree = ast.parse(candidate["source"])
    values = {
        name: value
        for name, value in _constants(candidate).items()
        if type(value) is int
    }
    tiles = {
        name.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "tl.arange"
        and len(node.args) == 2
        for name in ast.walk(node.args[1])
        if isinstance(name, ast.Name) and name.id in values
    }
    pids: set[str] = set()
    assignments = [
        (node.targets[0].id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ] + [
        (node.target.id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ]
    while True:
        derived = {
            name
            for name, value in assignments
            if name not in pids and _program_index(value, pids, tiles)
        }
        if not derived:
            break
        pids |= derived
    gridded = {
        other.id
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        for side, other in ((node.left, node.right), (node.right, node.left))
        if isinstance(other, ast.Name)
        and other.id in tiles
        and _program_index(side, pids, tiles)
    }
    # Names bounding a loop: only these can be a tile's trip count.
    bounds = {
        name.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("range", "tl.range", "tl.static_range")
        for argument in node.args
        for name in ast.walk(argument)
        if isinstance(name, ast.Name)
    }
    log = []
    halvable: dict[str, str | None] = {}
    for name in values:
        if name not in tiles:
            continue
        if values[name] in dimensions:
            log.append(f"{name} kept: it equals a boundary dimension")
            continue
        if name in gridded:
            log.append(f"{name} kept: it scales tl.program_id, and the grid is fixed")
            continue
        counts = [
            other
            for other, value in values.items()
            if other not in tiles
            and other in bounds
            and value * values[name] in dimensions
        ]
        if len(counts) > 1:
            log.append(f"{name} kept: {counts} could each be its trip count")
            continue
        halvable[name] = counts[0] if counts else None
    return halvable, log


def _halve_largest(
    candidate: dict[str, Any], tiles: dict[str, str | None]
) -> tuple[dict[str, Any], str, str] | None:
    """Halve the largest tile that stays at least 16, doubling its trip count."""
    values = _constants(candidate)
    choices = [name for name in tiles if values[name] // 2 >= 16]
    if not choices:
        return None
    name = max(choices, key=lambda item: values[item])
    updates = {name: values[name] // 2}
    change = f"{name} {values[name]} -> {updates[name]}"
    count = tiles[name]
    if count is not None:
        updates[count] = values[count] * 2
        change += f" and its trip count {count} {values[count]} -> {updates[count]}"
    constants = [
        {"name": item["name"], "value": updates.get(item["name"], item["value"])}
        for item in candidate["constants"]
    ]
    return {**candidate, "num_stages": 1, "constants": constants}, change, name


def _aligned_io(options: dict[str, Any] | None) -> bool:
    """Whether ``make_registration`` options promise 16-byte aligned pointers."""
    value = (options or {}).get("aligned_io", False)
    if type(value) is not bool:
        raise ValueError(f"Backend option aligned_io must be a bool; got {value!r}")
    return value


class _DslBackend(ABC):
    """Source and ABI contract shared by the Python kernel DSL backends.

    A kernel takes tensors ``arg0..argN-1`` and ``out0..outM-1``, then the
    candidate's constants in order. These checks reject unsupported source and
    launch contracts; they are not a hostile-code sandbox.
    """

    name: str
    version: str = "1"
    schema: ClassVar[dict[str, Any]]
    packages: ClassVar[tuple[str, ...]]
    # Permitted ``import module as alias`` statements.
    imports: ClassVar[frozenset[tuple[str, str | None]]]
    # Permitted ``from module import name as alias`` statements, as
    # (module, name, alias), and the DSL alias each one binds.
    from_imports: ClassVar[dict[tuple[str, str, str | None], str]] = {}
    # Source of the kernel decorator and of each constant's annotation.
    decorator: ClassVar[str]
    annotation: ClassVar[str]
    # Decorator lists that mark a module-level helper function, and the
    # annotations its parameters may carry.
    helper_decorators: ClassVar[tuple[tuple[str, ...], ...]] = ()
    helper_annotations: ClassVar[frozenset[str]] = frozenset()
    # Kernels may call any function of these namespaces, and these functions.
    namespaces: ClassVar[frozenset[tuple[str, ...]]]
    functions: ClassVar[frozenset[tuple[str, ...]]] = frozenset()
    # Module names that calls are rooted at and that generated names cannot shadow.
    aliases: ClassVar[frozenset[str]] = _ALIASES
    # Builtins that the DSL compiler's rewrite of a kernel calls inside it by
    # name, which kernel and helper bodies therefore cannot rebind either.
    compiler_builtins: ClassVar[frozenset[str]] = frozenset()
    # Types whose public methods kernels may call, as ``module.Type`` names.
    tensor_types: ClassVar[tuple[str, ...]]

    @property
    def proposal_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self.schema)

    @property
    def instruction_resources(self) -> tuple[str, ...]:
        return (f"references/{self.name}.md",)

    def target_metadata(self, device: Any) -> dict[str, Any]:
        return _target_metadata(device, self._toolchain(), self.packages)

    @cached_property
    def methods(self) -> frozenset[str]:
        """Callable public attributes of the installed DSL's tensor types."""
        try:
            types = [
                getattr(importlib.import_module(module), name)
                for module, _, name in (
                    path.rpartition(".") for path in self.tensor_types
                )
            ]
        except ImportError:
            types = []
        return _METHODS.union(
            name
            for kind in types
            for name in dir(kind)
            if not name.startswith("_") and callable(getattr(kind, name))
        )

    def validate_candidate(
        self, proposal: dict[str, Any], num_inputs: int, num_outputs: int
    ) -> dict[str, Any]:
        if any(
            type(count) is not int or count < 1 for count in (num_inputs, num_outputs)
        ):
            raise ValueError(
                "Generated kernels need at least one tensor input and output"
            )
        fields = self.schema["required"]
        if not isinstance(proposal, dict) or set(proposal) != set(fields):
            raise ValueError(
                f"Generated {self.name} candidate must contain exactly {sorted(fields)}"
            )
        source = proposal["source"]
        if not isinstance(source, str) or not source.strip() or len(source) > 131072:
            raise ValueError(
                "Generated source must be nonempty Python text under 128 KiB"
            )
        kernel_name = proposal["kernel_name"]
        # As for helpers, a kernel named like a permitted builtin would replace
        # it in every call that the source check allows.
        if not _identifier(kernel_name, self.aliases) or kernel_name in _BUILTINS:
            raise ValueError(
                "Generated kernel_name must be a nonreserved Python identifier"
            )
        grid = proposal["grid"]
        if (
            not isinstance(grid, list)
            or not 1 <= len(grid) <= 3
            or any(
                type(value) is not int or not 1 <= value <= limit
                for value, limit in zip(grid, (2147483647, 65535, 65535))
            )
        ):
            raise ValueError(
                "Generated grid must contain one to three valid positive CUDA dimensions"
            )
        if not isinstance(proposal["constants"], list):
            raise ValueError("Generated constants must be a list of name/value objects")
        runtime_names = [f"arg{i}" for i in range(num_inputs)] + [
            f"out{i}" for i in range(num_outputs)
        ]
        names = set(runtime_names)
        constants = []
        for item in proposal["constants"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "value"}
                or not _identifier(item["name"], self.aliases)
                or item["name"] in names
            ):
                raise ValueError("Generated constants need unique, nonreserved names")
            value = item["value"]
            if type(value) not in {bool, int, float} or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(
                    "Generated constant values must be finite bool/int/float scalars"
                )
            names.add(item["name"])
            constants.append(dict(item))
        try:
            self._check_source(source, kernel_name, constants, runtime_names)
        except (MemoryError, RecursionError) as error:
            # Python parses and unparses nested expressions recursively. Code
            # nested deeper than that can go is rejected like other code the
            # check cannot accept, which costs the author a turn instead of
            # ending the search.
            raise ValueError(
                "Generated kernel nests expressions too deeply to check "
                f"({type(error).__name__}); split long chains of calls, "
                "attributes or operators into separate statements"
            ) from None
        return {**proposal, "grid": list(grid), "constants": constants}

    def _check_source(
        self,
        source: str,
        kernel_name: str,
        constants: list[dict[str, Any]],
        runtime_names: list[str],
    ) -> None:
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError) as error:
            raise ValueError(
                f"Generated kernel has invalid Python syntax: {error}"
            ) from error
        imported = set()
        functions: dict[str, ast.FunctionDef] = {}
        for index, node in enumerate(tree.body):
            if (
                index == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported |= self._imported(node)
            elif isinstance(node, ast.FunctionDef):
                if node.name in functions:
                    raise ValueError(
                        f"line {node.lineno}: function {node.name!r} is defined twice"
                    )
                functions[node.name] = node
            else:
                raise ValueError(
                    f"{_line(node)} is not allowed at module scope. Generated source "
                    "permits only its kernel DSL imports, the kernel and helper "
                    f"functions decorated {self._helper_rule()}; put compile-time "
                    "values in the proposal's constants or inside the kernel"
                )
        function = functions.pop(kernel_name, None)
        if function is None:
            raise ValueError(
                f"Generated source must define exactly the named kernel "
                f"{kernel_name!r}; it defines {sorted(functions) or 'no functions'}"
            )
        helpers = self._check_helpers(
            list(functions.values()),
            reserved={kernel_name, *runtime_names, *(c["name"] for c in constants)},
        )
        roots = [self.decorator, self.annotation] if constants else [self.decorator]
        roots += [
            ast.unparse(node)
            for helper in functions.values()
            for node in [
                *helper.decorator_list,
                *(arg.annotation for arg in helper.args.args if arg.annotation),
            ]
        ]
        if not {root.partition(".")[0] for root in roots} <= imported:
            raise ValueError("Generated source is missing its kernel DSL imports")
        if [ast.unparse(item) for item in function.decorator_list] != [self.decorator]:
            raise ValueError(f"Generated kernel requires exactly @{self.decorator}")
        args = function.args
        if (
            args.posonlyargs
            or args.vararg
            or args.kwonlyargs
            or args.kwarg
            or args.defaults
            or function.returns is not None
            or getattr(function, "type_params", [])
        ):
            raise ValueError(
                "Generated kernel requires plain positional parameters without defaults"
            )
        expected_names = runtime_names + [item["name"] for item in constants]
        if [arg.arg for arg in args.args] != expected_names:
            raise ValueError(f"Generated kernel parameters must be {expected_names}")
        if any(arg.annotation is not None for arg in args.args[: len(runtime_names)]):
            raise ValueError(
                "Generated tensor parameters must not have Python annotations"
            )
        for argument, constant in zip(args.args[len(runtime_names) :], constants):
            value = constant["value"]
            expected = self.annotation.format(type=type(value).__name__)
            as_float = self.annotation.format(type="float")
            annotation = (
                None
                if argument.annotation is None
                else ast.unparse(argument.annotation)
            )
            if type(value) is int and annotation == as_float and as_float != expected:
                # JSON need not keep 2.0 apart from 2, and the claude CLI returns
                # it as 2, so a float annotation makes a whole number a float.
                # The constants are validate_candidate's copies.
                try:
                    constant["value"] = float(value)
                except OverflowError:
                    raise ValueError(
                        f"Generated constant {argument.arg!r} is too large for a float"
                    ) from None
            elif annotation != expected:
                raise ValueError(
                    f"Generated constant {argument.arg!r} needs the annotation {expected}"
                )
        for checked in (function, *functions.values()):
            self._check_body(checked, helpers)

    def _imported(self, node: ast.Import | ast.ImportFrom) -> set[str]:
        """Return the DSL aliases an import binds, rejecting any other import."""
        bound = set()
        for alias in node.names:
            if isinstance(node, ast.Import):
                name = (
                    alias.asname or alias.name
                    if (alias.name, alias.asname) in self.imports
                    else None
                )
            else:
                name = (
                    self.from_imports.get((node.module or "", alias.name, alias.asname))
                    if node.level == 0
                    else None
                )
            if name is None:
                raise ValueError(
                    f"{_line(node)} is not allowed: generated source may import only "
                    f"its kernel DSL, as {self._import_rule()}"
                )
            bound.add(name)
        return bound

    def _import_rule(self) -> str:
        """List the permitted import statements, without redundant aliases."""
        spellings = [
            f"import {module}" + ("" if alias is None else f" as {alias}")
            for module, alias in sorted(self.imports, key=lambda i: (i[0], i[1] or ""))
            if alias != module
        ] + [
            f"from {module} import {name}" + ("" if alias is None else f" as {alias}")
            for module, name, alias in sorted(
                self.from_imports, key=lambda i: (i[0], i[1], i[2] or "")
            )
            if alias != name
        ]
        return ", ".join(f"`{spelling}`" for spelling in spellings)

    def _helper_rule(self) -> str:
        return (
            " or ".join(
                "@" + " @".join(decorators) if decorators else "without a decorator"
                for decorators in self.helper_decorators
            )
            or "(none are allowed)"
        )

    def _check_helpers(
        self, helpers: list[ast.FunctionDef], reserved: set[str]
    ) -> frozenset[str]:
        """Check module-level helper functions and return their names."""
        for helper in helpers:
            where = f"line {helper.lineno}: helper function {helper.name!r}"
            decorators = tuple(ast.unparse(item) for item in helper.decorator_list)
            if decorators not in self.helper_decorators:
                raise ValueError(
                    f"{where} must be decorated {self._helper_rule()}; only the "
                    "named kernel takes the kernel decorator"
                )
            if (
                not _identifier(helper.name)
                or helper.name in reserved
                or helper.name in _BUILTINS
            ):
                raise ValueError(
                    f"{where} needs a name other than the kernel, its parameters, "
                    "the DSL aliases and the permitted builtins"
                )
            args = helper.args
            if (
                args.posonlyargs
                or args.vararg
                or args.kwonlyargs
                or args.kwarg
                or args.defaults
                or helper.returns is not None
                or getattr(helper, "type_params", [])
            ):
                raise ValueError(
                    f"{where} requires plain positional parameters without "
                    "defaults or a return annotation"
                )
            for arg in args.args:
                if "__" in arg.arg:
                    raise ValueError(
                        f"{where} cannot use the private parameter name {arg.arg!r}"
                    )
                if (
                    arg.annotation is not None
                    and ast.unparse(arg.annotation) not in self.helper_annotations
                ):
                    raise ValueError(
                        f"{where} parameter {arg.arg!r} may be unannotated or "
                        f"annotated {' or '.join(sorted(self.helper_annotations))}"
                    )
        return frozenset(helper.name for helper in helpers)

    def _check_body(self, function: ast.FunctionDef, helpers: frozenset[str]) -> None:
        """Keep a kernel or helper body within its DSL, helpers and _BUILTINS.

        Kernel compilers run kernel code as Python while compiling; CuTe DSL
        traces it. So besides restricting calls, the body may not rebind a
        permitted builtin, a DSL alias, a helper or a builtin the compiler
        calls, use other builtins, use an alias except to name a DSL function,
        type or constant, or assign attributes. A variable named like another
        builtin is usable only where its assignment certainly applies. This
        guards against mistakes; it is not a sandbox.
        """
        where = f"line {{line}}: {function.name!r}"
        # The names whose calls _callable or the DSL compiler trusts.
        reserved = self.aliases | helpers | _BUILTINS | self.compiler_builtins
        parents = {
            child: node
            for node in ast.walk(function)
            for child in ast.iter_child_nodes(node)
        }
        # Parameters, locals, helpers and _BUILTINS: the names a body may use.
        known = {arg.arg for arg in function.args.args} | helpers | _BUILTINS
        unassigned = _unassigned_reads(function, known)
        known |= {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
        }
        for arg in function.args.args:
            if arg.arg in reserved:
                raise ValueError(
                    f"{where.format(line=arg.lineno)}: parameter {arg.arg!r} would "
                    "rebind a DSL module, a helper function or a builtin that "
                    f"the kernel or the {self.name} compiler calls; rename it"
                )
        for statement in function.body:
            for inner in ast.walk(statement):
                line = getattr(inner, "lineno", function.lineno)
                if isinstance(inner, _FORBIDDEN):
                    raise ValueError(
                        f"{where.format(line=line)}: generated kernels must contain "
                        f"device computation only, not {type(inner).__name__} "
                        "statements; define helpers at module scope"
                    )
                if isinstance(inner, ast.Name) and "__" in inner.id:
                    raise ValueError(
                        "Generated kernel cannot access private Python names"
                    )
                if isinstance(inner, ast.Attribute) and (
                    inner.attr.startswith("_") or "__" in inner.attr
                ):
                    raise ValueError(
                        "Generated kernel cannot access private Python attributes"
                    )
                if isinstance(inner, ast.Name):
                    self._check_name(
                        inner,
                        parents[inner],
                        known,
                        reserved,
                        where.format(line=line),
                        unassigned,
                    )
                elif isinstance(inner, (ast.Attribute, ast.Subscript)):
                    self._check_reference(
                        inner, parents[inner], where.format(line=line)
                    )
                if not isinstance(inner, ast.Call):
                    continue
                if ast.unparse(inner.func) == "tl.inline_asm_elementwise":
                    raise ValueError("Generated kernels cannot contain inline assembly")
                if not self._callable(inner.func, helpers):
                    raise ValueError(
                        f"Generated kernel calls must use the supported {self.name} "
                        f"DSL: {ast.unparse(inner.func)!r} is not allowed "
                        f"({where.format(line=line)})"
                    )

    def _check_name(
        self,
        node: ast.Name,
        parent: ast.AST,
        known: set[str] | frozenset[str],
        reserved: frozenset[str],
        where: str,
        unassigned: set[ast.Name],
    ) -> None:
        """Keep each name meaning its parameter, local, helper, alias or builtin."""
        if not isinstance(node.ctx, ast.Load):
            if node.id in reserved:
                raise ValueError(
                    f"{where}: generated kernels cannot bind {node.id!r}, which "
                    "names a DSL module, a helper function or a builtin that the "
                    f"kernel or the {self.name} compiler calls; use another name"
                )
        elif node.id in self.aliases:
            if not (isinstance(parent, ast.Attribute) and parent.value is node):
                raise ValueError(
                    f"{where}: generated kernels may use the DSL module {node.id!r} "
                    "only to name its functions, types and constants, as in "
                    f"{node.id}.<name>"
                )
        elif node.id not in known:
            raise ValueError(
                f"{where}: {node.id!r} is not a parameter, a local variable, a "
                "helper function or a builtin that generated kernels may use "
                f"({', '.join(sorted(_BUILTINS))})"
            )
        if node in unassigned:
            raise ValueError(
                f"{where}: {node.id!r} names a Python builtin wherever it is not "
                "certainly assigned: assign the variable earlier in this block "
                "or an enclosing one (an assignment inside an if, a loop or a "
                "comprehension ends with it), or use another name"
            )

    def _check_reference(
        self, node: ast.Attribute | ast.Subscript, parent: ast.AST, where: str
    ) -> None:
        """Reject assigning attributes or DSL state, and naming past the DSL."""
        if not isinstance(node.ctx, ast.Load):
            root: ast.expr = node
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(node, ast.Attribute) or (
                isinstance(root, ast.Name) and root.id in self.aliases
            ):
                raise ValueError(
                    f"{where}: generated kernels cannot assign or delete "
                    f"attributes or DSL module state: {ast.unparse(node)!r}"
                )
            return
        # Only a whole attribute chain names something, and a call target is
        # checked as a call.
        if (
            not isinstance(node, ast.Attribute)
            or (isinstance(parent, ast.Attribute) and parent.value is node)
            or (isinstance(parent, ast.Call) and parent.func is node)
        ):
            return
        qualified = self._dsl_path(node)
        if qualified is not None and not self._dsl_name(qualified, call=False):
            raise ValueError(
                "Generated kernels may name only functions, types and constants of "
                f"the supported {self.name} DSL, attributes of its types such as "
                "enum members, and data attributes of its other members, not "
                "modules or other objects: "
                f"{ast.unparse(node)!r} is not allowed ({where})"
            )

    def _callable(
        self, target: ast.expr, helpers: frozenset[str] = frozenset()
    ) -> bool:
        """Allow DSL functions, tensor methods, helpers, and a few pure builtins.

        Kernel compilers can evaluate Python calls while tracing, so a call
        rooted at a module must name a DSL function, not reach other modules.
        """
        if isinstance(target, ast.Name):
            return target.id in _BUILTINS or target.id in helpers
        if not isinstance(target, ast.Attribute):
            return False
        qualified = self._dsl_path(target)
        if qualified is None:
            return target.attr in self.methods
        return self._dsl_name(qualified, call=True)

    def _dsl_path(self, node: ast.Attribute) -> tuple[str, ...] | None:
        """The name an attribute chain rooted at a DSL alias spells, else None."""
        path: list[str] = []
        receiver: ast.expr = node
        while isinstance(receiver, ast.Attribute):
            path.insert(0, receiver.attr)
            receiver = receiver.value
        if not isinstance(receiver, ast.Name) or receiver.id not in self.aliases:
            return None
        return (receiver.id, *path)

    def _dsl_name(self, qualified: tuple[str, ...], call: bool) -> bool:
        """Whether kernel code may call, or otherwise use, a DSL name.

        A call must name a member of a DSL namespace. Any other use may also
        name an attribute of a namespace class, such as an enum member, or a
        data attribute of another namespace member, such as
        tl.float32.primitive_bitwidth, which a variable holding the member
        reaches anyway. Each must be the installed DSL's own: data, or code
        that the DSL package defines, never a module or what a namespace
        merely imports, such as dataclasses.dataclass, through which code
        could reach past the DSL. Other uses must also resolve: a name that a
        later import adds could be a module. A call to a missing function just
        fails to compile.
        """
        if qualified in self.functions:
            return True
        if qualified[:-1] in self.namespaces:
            owners: list[Any] | None = None
        elif not call and qualified[:-2] in self.namespaces:
            owners = self._resolve(qualified[:-1])
        else:
            return False
        found = self._resolve(qualified)
        if found is None:
            # The DSL does not import here, so no kernel that uses it can run.
            return True
        return (
            (call or bool(found))
            and all(self._owned(value) for value in found)
            and all(
                self._owned(owner)
                and (isinstance(owner, type) or not any(map(callable, found)))
                for owner in owners or ()
            )
        )

    def _owned(self, value: Any) -> bool:
        """Whether ``value`` is data, or a function or class the DSL defines."""
        if isinstance(value, ModuleType):
            return False
        if isinstance(value, partial):
            value = value.func
        if not callable(value):
            return True
        module = getattr(value, "__module__", None)
        if not isinstance(module, str):
            module = type(value).__module__
        return any(
            module == root or module.startswith(f"{root}.")
            for paths in self._alias_modules.values()
            for root in paths
        )

    @cached_property
    def _alias_modules(self) -> dict[str, tuple[str, ...]]:
        """The modules each DSL alias can name under the permitted imports."""
        modules: dict[str, set[str]] = {}
        for module, alias in self.imports:
            modules.setdefault(alias or module, set()).add(module)
        for (module, name, _), alias in self.from_imports.items():
            modules.setdefault(alias, set()).add(f"{module}.{name}")
        return {alias: tuple(sorted(paths)) for alias, paths in modules.items()}

    def _resolve(self, qualified: tuple[str, ...]) -> list[Any] | None:
        """The objects ``qualified`` names in the modules its alias can import.

        None when none of those modules imports. A name that one of them
        lacks resolves to nothing there.
        """
        found: list[Any] = []
        importable = False
        for path in self._alias_modules.get(qualified[0], ()):
            try:
                value = importlib.import_module(path)
            except ImportError:
                continue
            importable = True
            try:
                for part in qualified[1:]:
                    value = getattr(value, part)
            except Exception:  # noqa: BLE001 (a module __getattr__ may raise anything)
                continue
            found.append(value)
        return found if importable else None

    def _kernel(
        self,
        candidate: dict[str, Any],
        num_inputs: int,
        num_outputs: int,
        directory: Path,
    ) -> Any:
        """Import the candidate's kernel from ``directory`` once per process."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sha256_text(candidate['source'])}.py"
        # cuTile rereads kernel source from this file whenever it compiles.
        path.write_text(candidate["source"], encoding="utf-8")
        module_name = f"_torch_tensorrt_generated_{sha256_text(str(path.resolve()))}"
        module = sys.modules.get(module_name)
        if module is None:
            # Importing runs module-level statements; recheck the source here even
            # though callers validate, since it is the last step before execution.
            self.validate_candidate(candidate, num_inputs, num_outputs)
            _LOGGER.info("Generated %s kernel source: %s", self.name, path)
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ValueError("Could not load generated kernel source")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        return getattr(module, candidate["kernel_name"])

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None = None,
    ) -> KernelSpec | partial[None]:
        kernel = self._kernel(candidate, len(input_specs), len(output_specs), directory)
        dimensions = tuple(candidate["grid"])

        def grid(inputs: Any, outputs: Any) -> tuple[int, ...]:
            return dimensions

        return self._registration(kernel, candidate, grid, dict(options or {}))

    @abstractmethod
    def _toolchain(self) -> dict[str, Any]:
        """Probe the toolchain and return the versions that identify it."""

    @abstractmethod
    def _registration(
        self,
        kernel: Any,
        candidate: dict[str, Any],
        grid: Callable[..., Any],
        options: dict[str, Any],
    ) -> KernelSpec | partial[None]:
        """Bind the loaded kernel to the existing frontend's registration.

        ``options`` are the generator's code-generation options, such as
        ``aligned_io``; see ``GenerationBackend``.
        """

    @abstractmethod
    def launch(
        self,
        candidate: dict[str, Any],
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        directory: Path,
    ) -> None:
        """Launch once on the current stream; see ``GenerationBackend``."""


class _TritonBackend(_DslBackend):
    name = "triton"
    # 2: helper functions, from-imports, plugin shared-memory fitting.
    version = "2"
    schema = _dsl_schema(
        # The values validate_compile_options accepts. The outer enum and
        # minimum repeat the branches' limits for readers of the bare keywords.
        num_warps={
            "anyOf": [
                {"type": "integer", "enum": [1, 2, 4, 8, 16, 32]},
                {"type": "null"},
            ],
            "enum": [None, 1, 2, 4, 8, 16, 32],
        },
        num_stages={
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
            "minimum": 1,
        },
    )
    packages = ("triton",)
    imports = frozenset(
        {
            ("triton", None),
            ("triton", "triton"),
            ("triton.language", "tl"),
            ("triton.language.extra.cuda.libdevice", "libdevice"),
            ("triton.language.extra.libdevice", "libdevice"),
        }
    )
    from_imports = {
        ("triton", "language", "tl"): "tl",
        ("triton.language.extra.cuda", "libdevice", None): "libdevice",
        ("triton.language.extra.cuda", "libdevice", "libdevice"): "libdevice",
        ("triton.language.extra", "libdevice", None): "libdevice",
        ("triton.language.extra", "libdevice", "libdevice"): "libdevice",
    }
    decorator = "triton.jit"
    annotation = "tl.constexpr"
    helper_decorators = (("triton.jit",),)
    helper_annotations = frozenset({"tl.constexpr"})
    namespaces = frozenset({("tl",), ("tl", "math"), ("libdevice",)})
    functions = frozenset({("triton", "cdiv"), ("triton", "next_power_of_2")})
    tensor_types = ("triton.language.tensor",)

    def _toolchain(self) -> dict[str, Any]:
        from torch_tensorrt.kernels._triton import _driver_max_ptx_version

        return {"driver_ptx": _driver_max_ptx_version()}

    def validate_candidate(
        self, proposal: dict[str, Any], num_inputs: int, num_outputs: int
    ) -> dict[str, Any]:
        from torch_tensorrt.kernels._triton import validate_compile_options

        candidate = super().validate_candidate(proposal, num_inputs, num_outputs)
        validate_compile_options(candidate["num_warps"], candidate["num_stages"])
        reserved = _TRITON_LAUNCH_KEYWORDS & set(_constants(candidate))
        if reserved:
            raise ValueError(
                f"Triton constants cannot be named {sorted(reserved)}; Triton binds "
                f"{sorted(_TRITON_LAUNCH_KEYWORDS)} by keyword when it launches"
            )
        return candidate

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None = None,
    ) -> KernelSpec | partial[None]:
        """Register the kernel after checking the plugin's compiled launch.

        Triton's own launcher opts a kernel into more shared memory than a
        block gets by default, but a TensorRT plugin launch does not, so such a
        kernel passes a direct launch and then fails inside an engine.
        """
        registration, artifact, compiled = self._plugin_artifact(
            candidate, input_specs, output_specs, directory, options
        )
        limit = torch.cuda.get_device_properties(
            input_specs[0].device
        ).shared_memory_per_block
        if artifact.shared_mem > limit:
            raise _shared_memory_error(candidate, artifact.shared_mem, limit, compiled)
        return registration

    def _plugin_artifact(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None,
    ) -> tuple[partial[None], Any, Any]:
        """Compile what the plugin registration compiles; Triton caches it."""
        from torch_tensorrt.kernels._triton import _compile_triton

        from .._plugins import _triton_signature

        registration = super().make_registration(
            candidate, input_specs, output_specs, directory, options
        )
        assert isinstance(registration, partial)
        with torch.cuda.device(input_specs[0].device):
            compiled, artifact = _compile_triton(
                registration.keywords["kernel"],
                _triton_signature(registration, input_specs, output_specs, "generated"),
                registration.keywords["constexprs"],
                num_warps=candidate["num_warps"],
                num_stages=candidate["num_stages"],
                pointer_alignment=registration.keywords.get("pointer_alignment"),
            )
        return registration, artifact, compiled

    def fit_shared_memory(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None,
        error: SharedMemoryLimitError,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Find a variant whose plugin fits in shared memory, compiling only.

        Tries one pipeline stage, then halves the largest ``tl.arange`` tile
        constant, at most three times and never below 16. A constant equal to
        a boundary dimension, or one that scales ``tl.program_id``, is kept,
        since the grid is a literal. A constant bounding a loop whose product
        with the tile is a boundary dimension, the tile's trip count, is
        doubled with it. Returns the first variant that fits, or None, and a
        line per decision or compilation.
        """
        dimensions = {
            int(extent)
            for spec in (*input_specs, *output_specs)
            for extent in spec.shape
        }
        tiles, log = _halvable_tiles(candidate, dimensions)
        current, needed = candidate, error.required
        steps = ["stages"] if candidate["num_stages"] != 1 else []
        for step in [*steps, "tile", "tile", "tile"]:
            name = None
            if step == "stages":
                trial = {**current, "num_stages": 1}
                change = f"num_stages {current['num_stages']} -> 1"
            else:
                halved = _halve_largest(current, tiles)
                if halved is None:
                    break
                trial, change, name = halved
            try:
                _, artifact, _ = self._plugin_artifact(
                    trial, input_specs, output_specs, directory, options
                )
            except Exception as failure:  # this variant does not compile
                lines = str(failure).strip().splitlines()
                reason = lines[-1][:200] if lines else repr(failure)
                log.append(f"{change}: does not compile: {reason}")
                if name is not None:
                    tiles.pop(name, None)
                continue
            log.append(f"{change}: {needed} -> {artifact.shared_mem} bytes")
            current, needed = trial, int(artifact.shared_mem)
            if needed <= error.limit:
                return current, log
        return None, log

    def _registration(
        self,
        kernel: Any,
        candidate: dict[str, Any],
        grid: Callable[..., Any],
        options: dict[str, Any],
    ) -> KernelSpec | partial[None]:
        from torch_tensorrt.region._plugin import prepare_plugin

        # Region registration derives the pointer signature from the boundaries,
        # exactly as it does for a user-supplied fused_kernel.
        return prepare_plugin(
            kernel,
            constexprs=_constants(candidate),
            grid=grid,
            num_warps=candidate["num_warps"],
            num_stages=candidate["num_stages"],
            pointer_alignment=16 if _aligned_io(options) else None,
        )

    def launch(
        self,
        candidate: dict[str, Any],
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        directory: Path,
    ) -> None:
        kernel = self._kernel(candidate, len(inputs), len(outputs), directory)
        options = {
            name: candidate[name]
            for name in ("num_warps", "num_stages")
            if candidate[name] is not None
        }
        # Constants follow the tensors positionally, as the ABI orders them, so
        # a constant named grid or warmup cannot collide with Triton's run().
        with torch.cuda.device(inputs[0].device):
            kernel[tuple(candidate["grid"])](
                *inputs, *outputs, *_constants(candidate).values(), **options
            )


class _CutileBackend(_DslBackend):
    name = "cutile"
    schema = _dsl_schema()
    packages = ("cuda-tile",)
    imports = frozenset({("cuda.tile", "ct")})
    from_imports = {("cuda", "tile", "ct"): "ct"}
    decorator = "ct.kernel"
    annotation = "ct.Constant[{type}]"
    # cuTile compiles plain functions a kernel calls as tile functions.
    helper_decorators = (("ct.function",), ())
    helper_annotations = frozenset(
        {"ct.Constant[int]", "ct.Constant[float]", "ct.Constant[bool]"}
    )
    namespaces = frozenset({("ct",)})
    tensor_types = ("cuda.tile.Tile", "cuda.tile.Array", "cuda.tile.TiledView")

    def _toolchain(self) -> dict[str, Any]:
        from torch_tensorrt.kernels._cutile import (
            _cutile_compiler_version,
            validate_cutile_toolchain,
        )

        # The toolchain gate returns the loaded driver's CUDA version.
        driver = validate_cutile_toolchain("agent_generation")
        return {
            "driver": list(driver),
            "cutile_compiler": list(_cutile_compiler_version()),
        }

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
        options: dict[str, Any] | None = None,
    ) -> KernelSpec | partial[None]:
        """Register the kernel with the layout facts of the captured boundaries.

        A region plugin is a static specialization of contiguous tensors, so
        cuTile may assume each array's extent. ``options["aligned_io"]`` adds
        16-byte base addresses, which lets cuTile vectorize loads and stores.
        """
        registration = super().make_registration(
            candidate, input_specs, output_specs, directory
        )
        assert isinstance(registration, partial)
        shapes = {f"arg{i}": tuple(spec.shape) for i, spec in enumerate(input_specs)}
        shapes.update(
            {f"out{i}": tuple(spec.shape) for i, spec in enumerate(output_specs)}
        )
        return partial(
            registration,
            static_shapes=shapes,
            pointer_alignment=16 if _aligned_io(options) else None,
        )

    def _registration(
        self,
        kernel: Any,
        candidate: dict[str, Any],
        grid: Callable[..., Any],
        options: dict[str, Any],
    ) -> KernelSpec | partial[None]:
        from torch_tensorrt import kernels

        # Arrays are flattened rank-one views; region registration supplies
        # their names and dtypes from the captured boundaries.
        return partial(
            kernels.cutile_op,
            kernel=kernel,
            constants=_constants(candidate),
            ndim=1,
            grid=grid,
        )

    def launch(
        self,
        candidate: dict[str, Any],
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        directory: Path,
    ) -> None:
        import cuda.tile as ct
        from torch_tensorrt.kernels._cutile import _tma_hints, _without_tma

        kernel = self._kernel(candidate, len(inputs), len(outputs), directory)
        device = inputs[0].device
        arrays = [tensor.view(-1) for tensor in (*inputs, *outputs)]
        # The plugin never uses TMA: its descriptors are built on the host and
        # a TensorRT AOT launch cannot pass them. cuTile's own launch would use
        # TMA wherever it can, and the two codes can differ by 100x in speed
        # (a tile that TMA stages through shared memory otherwise lives in
        # registers), so check and time the code the plugin runs.
        context = _tma_hints()
        if context is None:
            _LOGGER.warning(
                "This cuda-tile release offers no way to disable TMA; the "
                "direct launch may be faster than the plugin."
            )
        with (
            torch.cuda.device(device),
            contextlib.nullcontext() if context is None else _without_tma(context),
        ):
            ct.launch(
                torch.cuda.current_stream(device).cuda_stream,
                tuple(candidate["grid"]),
                kernel,
                (*arrays, *_constants(candidate).values()),
            )


TRITON = _TritonBackend()
CUTILE = _CutileBackend()
