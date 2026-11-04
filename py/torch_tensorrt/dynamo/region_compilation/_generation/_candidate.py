# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Python kernel DSL backends that validate generated source and reuse frontends."""

from __future__ import annotations

import ast
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
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch

from ._backends import _dimensions, _proposal_schema, _target_metadata
from ._util import sha256_text

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec

_LOGGER = logging.getLogger(__name__)
_BUILTINS = {"abs", "bool", "float", "int", "len", "max", "min", "range", "tuple"}
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
    ast.Nonlocal,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def _identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.isidentifier()
        and not keyword.iskeyword(value)
        and "__" not in value
        and value not in _ALIASES
    )


def _dsl_schema(**options: Any) -> dict[str, Any]:
    return _proposal_schema(
        source={"type": "string"},
        kernel_name={"type": "string"},
        grid=_dimensions(2147483647),
        constants={
            "type": "array",
            "items": _proposal_schema(
                name={"type": "string"}, value={"type": ["boolean", "number"]}
            ),
        },
        **options,
    )


def _constants(candidate: dict[str, Any]) -> dict[str, Any]:
    return {item["name"]: item["value"] for item in candidate["constants"]}


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
    # Source of the kernel decorator and of each constant's annotation.
    decorator: ClassVar[str]
    annotation: ClassVar[str]
    # Kernels may call any function of these namespaces, and these functions.
    namespaces: ClassVar[frozenset[tuple[str, ...]]]
    functions: ClassVar[frozenset[tuple[str, ...]]] = frozenset()
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
        if not _identifier(kernel_name):
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
                or not _identifier(item["name"])
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
        self._check_source(source, kernel_name, constants, runtime_names)
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
        functions = []
        for index, node in enumerate(tree.body):
            if (
                index == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.name, alias.asname) not in self.imports:
                        raise ValueError(
                            "Generated source may import only its kernel DSL"
                        )
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node)
            else:
                raise ValueError(
                    "Generated source permits imports and one kernel definition only"
                )
        if len(functions) != 1 or functions[0].name != kernel_name:
            raise ValueError("Generated source must define exactly the named kernel")
        function = functions[0]
        roots = [self.decorator, self.annotation] if constants else [self.decorator]
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
            expected = self.annotation.format(type=type(constant["value"]).__name__)
            if argument.annotation is None or (
                ast.unparse(argument.annotation) != expected
            ):
                raise ValueError(
                    f"Generated constant {argument.arg!r} needs the annotation {expected}"
                )
        for statement in function.body:
            for inner in ast.walk(statement):
                if isinstance(inner, _FORBIDDEN):
                    raise ValueError(
                        "Generated kernel must contain device computation only"
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
                if not isinstance(inner, ast.Call):
                    continue
                if ast.unparse(inner.func) == "tl.inline_asm_elementwise":
                    raise ValueError("Generated kernels cannot contain inline assembly")
                if not self._callable(inner.func):
                    raise ValueError(
                        f"Generated kernel calls must use the supported {self.name} "
                        f"DSL: {ast.unparse(inner.func)!r} is not allowed"
                    )

    def _callable(self, target: ast.expr) -> bool:
        """Allow DSL functions, tensor methods, and a few pure builtins.

        Kernel compilers can evaluate Python calls while tracing, so a call
        rooted at a module must name a DSL function, not reach other modules.
        """
        if isinstance(target, ast.Name):
            return target.id in _BUILTINS
        if not isinstance(target, ast.Attribute):
            return False
        path = [target.attr]
        receiver = target.value
        while isinstance(receiver, ast.Attribute):
            path.insert(0, receiver.attr)
            receiver = receiver.value
        if not isinstance(receiver, ast.Name) or receiver.id not in _ALIASES:
            return target.attr in self.methods
        qualified = (receiver.id, *path)
        return qualified[:-1] in self.namespaces or qualified in self.functions

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
    ) -> KernelSpec | partial[None]:
        kernel = self._kernel(candidate, len(input_specs), len(output_specs), directory)
        dimensions = tuple(candidate["grid"])

        def grid(inputs: Any, outputs: Any) -> tuple[int, ...]:
            return dimensions

        return self._registration(kernel, candidate, grid)

    @abstractmethod
    def _toolchain(self) -> dict[str, Any]:
        """Probe the toolchain and return the versions that identify it."""

    @abstractmethod
    def _registration(
        self, kernel: Any, candidate: dict[str, Any], grid: Callable[..., Any]
    ) -> KernelSpec | partial[None]:
        """Bind the loaded kernel to the existing frontend's registration."""

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
    schema = _dsl_schema(
        # The values validate_compile_options accepts.
        num_warps={"type": ["integer", "null"], "enum": [None, 1, 2, 4, 8, 16, 32]},
        num_stages={"type": ["integer", "null"], "minimum": 1},
    )
    packages = ("triton",)
    imports = frozenset(
        {
            ("triton", None),
            ("triton", "triton"),
            ("triton.language", "tl"),
            ("triton.language.extra.cuda.libdevice", "libdevice"),
        }
    )
    decorator = "triton.jit"
    annotation = "tl.constexpr"
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

    def _registration(
        self, kernel: Any, candidate: dict[str, Any], grid: Callable[..., Any]
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
    decorator = "ct.kernel"
    annotation = "ct.Constant[{type}]"
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

    def _registration(
        self, kernel: Any, candidate: dict[str, Any], grid: Callable[..., Any]
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

        kernel = self._kernel(candidate, len(inputs), len(outputs), directory)
        device = inputs[0].device
        arrays = [tensor.view(-1) for tensor in (*inputs, *outputs)]
        with torch.cuda.device(device):
            ct.launch(
                torch.cuda.current_stream(device).cuda_stream,
                tuple(candidate["grid"]),
                kernel,
                (*arrays, *_constants(candidate).values()),
            )


TRITON = _TritonBackend()
CUTILE = _CutileBackend()
