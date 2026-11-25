# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Place whole submodules in regions with ``regions=`` rules, ahead of time."""

import importlib.util
import logging
import re
import sys
import textwrap

import pytest
import torch
import torch_tensorrt
from torch.utils._triton import has_triton
from torch_tensorrt import region
from torch_tensorrt._compile import _trace_with_regions
from torch_tensorrt.dynamo._compiler import _copy_program_graph
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
    ConverterSupport,
)
from torch_tensorrt.dynamo.lowering._buffer_lifting import erase_export_guards
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    check_decomposed_regions,
    get_region_records,
    has_region_markers,
    materialize_regions,
    normalize_region_scopes,
    select_exported_regions,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region import (
    ExecuteInTorch,
    FusedKernel,
    FusedOp,
    RegionCaptureError,
    RegionError,
)
from torch_tensorrt.region._session import RegionCompilationSession

from ._utils import run_python

# Use test-level skips: collection skips omit the JUnit ``type`` attribute and
# cannot be consumed by the CI results reporter on Windows and TensorRT RTX.
try:
    import tensorrt.plugin as trtp
    import triton
    import triton.language as tl
except ImportError:
    triton = None

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)
requires_cuda_qdp = pytest.mark.skipif(
    triton is None
    or not torch.cuda.is_available()
    or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="CUDA, Triton and TensorRT QDP plugin support are required",
)
BLOCK = 128
EPS = 1e-6
_calls = []


# The operators record each call. plus_hundred deliberately differs from every
# module it replaces; the others compute what their module computes, and their
# results change when their arguments are bound in another order.
@torch.library.custom_op("ttrt_region_rules_test::plus_hundred", mutates_args=())
def _plus_hundred(x: torch.Tensor) -> torch.Tensor:
    _calls.append("plus_hundred")
    return x + 100


@_plus_hundred.register_fake
def _plus_hundred_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_region_rules_test::weighted_difference", mutates_args=())
def _weighted_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    _calls.append("weighted_difference")
    return first - 2 * second


@_weighted_difference.register_fake
def _weighted_difference_fake(first, second):
    return torch.empty_like(first)


@torch.library.custom_op("ttrt_region_rules_test::scale_shift", mutates_args=())
def _scale_shift(
    x: torch.Tensor, weight: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    _calls.append("scale_shift")
    return x * weight + shift


@_scale_shift.register_fake
def _scale_shift_fake(x, weight, shift):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_region_rules_test::affine", mutates_args=())
def _affine(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _calls.append("affine")
    return x @ weight.T + bias


@_affine.register_fake
def _affine_fake(x, weight, bias):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("ttrt_region_rules_test::project", mutates_args=())
def _project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _calls.append("project")
    return x @ weight.T


@_project.register_fake
def _project_fake(x, weight):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("ttrt_region_rules_test::project_scale_shift", mutates_args=())
def _project_scale_shift(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    _calls.append("project_scale_shift")
    return (x @ weight.T) * scale + shift


@_project_scale_shift.register_fake
def _project_scale_shift_fake(x, weight, scale, shift):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("ttrt_region_rules_test::rms_norm", mutates_args=())
def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _calls.append("rms_norm")
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


@_rms_norm.register_fake
def _rms_norm_fake(x, weight):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_region_rules_test::feed_forward", mutates_args=())
def _feed_forward(
    hidden: torch.Tensor, residual: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    _calls.append("feed_forward")
    return residual + torch.nn.functional.silu(hidden @ up.T) @ down.T


@_feed_forward.register_fake
def _feed_forward_fake(hidden, residual, up, down):
    return torch.empty_like(residual)


@torch.library.custom_op("ttrt_region_rules_test::halves", mutates_args=())
def _halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _calls.append("halves")
    return x.clamp(min=0), x.clamp(max=0)


@_halves.register_fake
def _halves_fake(x):
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("ttrt_region_rules_test::first_half", mutates_args=())
def _first_half(x: torch.Tensor) -> torch.Tensor:
    _calls.append("first_half")
    return x[:, : x.shape[1] // 2].clone()


@_first_half.register_fake
def _first_half_fake(x):
    return x.new_empty(x.shape[0], x.shape[1] // 2)


@torch.library.custom_op("ttrt_region_rules_test::reshaped", mutates_args=())
def _reshaped(x: torch.Tensor) -> torch.Tensor:
    _calls.append("reshaped")
    return x.reshape(x.shape[0] * 2, -1).clone()


@_reshaped.register_fake
def _reshaped_fake(x):
    return x.new_empty(x.shape[0] * 2, x.shape[1] // 2)


@torch.library.custom_op("ttrt_region_rules_test::transposed", mutates_args=())
def _transposed(x: torch.Tensor) -> torch.Tensor:
    _calls.append("transposed")
    return x.transpose(0, 1).contiguous()


@_transposed.register_fake
def _transposed_fake(x):
    return x.new_empty(x.shape[1], x.shape[0])


def _two_halves_fake(x):
    return tuple(x.new_empty(x.shape[0], x.shape[1] // 2) for _ in range(2))


@torch.library.custom_op("ttrt_region_rules_test::chunks", mutates_args=())
def _chunks(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _calls.append("chunks")
    low, high = x.chunk(2, dim=-1)
    return low.clone(), high.clone()


_chunks.register_fake(_two_halves_fake)


@torch.library.custom_op("ttrt_region_rules_test::doubled_chunks", mutates_args=())
def _doubled_chunks(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _calls.append("doubled_chunks")
    low, high = (x * 2).chunk(2, dim=-1)
    return low.clone(), high.clone()


_doubled_chunks.register_fake(_two_halves_fake)

# Operators that a rule must refuse when it is created.
_invalid = torch.library.Library("ttrt_region_rules_invalid", "DEF")
_invalid.define("scalar(Tensor x, int count) -> Tensor")
_invalid.define("alias(Tensor(a) x) -> Tensor(a)")
_invalid.define("mutate(Tensor(a!) x) -> Tensor")


if triton is not None:

    @triton.jit
    def _bias_kernel(x, bias, n, y, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < n, other=0.0)
        values += tl.load(bias + offsets, offsets < n, other=0.0)
        tl.store(y + offsets, tl.where(values < 0, 0.0, values), offsets < n)

    @triton.jit
    def _subtract_kernel(first, second, n, result, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        lhs = tl.load(first + offsets, offsets < n, other=0.0)
        rhs = tl.load(second + offsets, offsets < n, other=0.0)
        tl.store(result + offsets, lhs - rhs, offsets < n)

    @triton.jit
    def _weighted_difference_kernel(first, second, n, result, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        lhs = tl.load(first + offsets, offsets < n, other=0.0)
        rhs = tl.load(second + offsets, offsets < n, other=0.0)
        tl.store(result + offsets, lhs - 2.0 * rhs, offsets < n)

    @triton.jit
    def _halves_kernel(x, n, high, low, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < n, other=0.0)
        tl.store(high + offsets, tl.where(values > 0, values, 0.0), offsets < n)
        tl.store(low + offsets, tl.where(values < 0, values, 0.0), offsets < n)

    @triton.jit
    def _rms_norm_kernel(x, weight, cols, y, BLOCK: tl.constexpr, EPS: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        values = tl.load(x + row * cols + offsets, offsets < cols, other=0.0)
        scale = tl.rsqrt(tl.sum(values * values, 0) / cols + EPS)
        gains = tl.load(weight + offsets, offsets < cols, other=0.0)
        tl.store(y + row * cols + offsets, gains * values * scale, offsets < cols)


def _grid(inputs, outputs):
    return (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),)


def _extras(inputs, outputs):
    return [trtp.SymInt32(inputs[0].shape_expr.numel())]


def _row_grid(inputs, outputs):
    return (trtp.cdiv(inputs[0].shape_expr.numel(), inputs[1].shape_expr.numel()),)


def _row_extras(inputs, outputs):
    return [trtp.SymInt32(inputs[1].shape_expr.numel())]


def _kernel_rule(kernel, **overrides):
    options = dict(constexprs={"BLOCK": BLOCK}, grid=_grid, extra_args_fn=_extras)
    options.update(overrides)
    return FusedKernel(kernel, **options)


class Activation(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class SmoothActivation(Activation):
    def forward(self, x):
        return torch.tanh(x)


class Scale(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 2.0, width))
        self.register_buffer("shift", torch.linspace(-1.0, 1.0, width))

    def forward(self, x):
        return x * self.weight + self.shift


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = Activation()
        self.scale = Scale()

    def forward(self, x):
        return self.scale(self.act(x) + 1)


class Stack(torch.nn.Module):
    """Blocks in a ModuleList, a module that runs no operations and a Sequential."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.skip = torch.nn.Identity()
        self.head = torch.nn.Sequential(torch.nn.Linear(8, 8), SmoothActivation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x - 0.5)
        return self.head(self.skip(x)) * 2


class Sandwich(torch.nn.Module):
    """Two modules, each between operations that TensorRT converts."""

    def __init__(self, width=8):
        super().__init__()
        self.act = Activation()
        self.scale = Scale(width)

    def forward(self, x):
        x = self.act(x + 1)
        x = self.scale(x * 3)
        return x - 2


class WeightedDifference(torch.nn.Module):
    def forward(self, first, second, factor=2.0):
        # The product reads second before the sum reads first.
        return second * -factor + first


class DifferenceCaller(torch.nn.Module):
    def __init__(self, layout="positional"):
        super().__init__()
        self.difference = WeightedDifference()
        self.layout = layout

    def forward(self, x):
        first, second = x + 1, x * 3
        if self.layout == "positional":
            result = self.difference(first, second)
        elif self.layout == "keyword":
            result = self.difference(first, second=second)
        elif self.layout == "keywords reversed":
            result = self.difference(second=second, first=first)
        else:
            # A number between the tensors is not a position.
            result = self.difference(first, factor=2.0, second=second)
        return result * 2


class FirstOnly(torch.nn.Module):
    def forward(self, first, second):
        return torch.sin(first)


class FirstOnlyCaller(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_only = FirstOnly()

    def forward(self, x):
        return self.first_only(x + 1, x * 3) * 2


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


class Wrapped(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapper = Projection()

    def forward(self, x):
        return self.wrapper(x + 1) * 2


class RMSNorm(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, width))

    def forward(self, x):
        # For a float32 input, both casts return their input itself.
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
        return self.weight * y.type_as(x)


class Annotated(torch.nn.Module):
    """One scope written in forward between modules that rules can select."""

    def __init__(self):
        super().__init__()
        self.act = Activation()
        self.difference = WeightedDifference()
        self.scale = Scale()

    def forward(self, x):
        x = self.act(x + 1)
        with region.execute_in_torch(name="written"):
            y = torch.cos(x) * 3
        return self.scale(self.difference(x, second=y)) - 1


class Around(torch.nn.Module):
    """Operations before and after one module."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x + 1) * 2


class InPlaceBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.linear(x))


class HalfActivation(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) * torch.tensor(0.5)


class Untraceable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = Activation()

    def forward(self, x):
        raise AssertionError("This model cannot be traced")


def _materialize(program, rules=None):
    """Select and capture regions as dynamo.compile does, without building engines."""
    program = _copy_program_graph(program)
    if rules:
        select_exported_regions(program, rules)
    for entry in program.module_call_graph:
        if entry.fqn:
            entry.signature = None
    program = normalize_region_scopes(program)
    captured = program.graph_module
    program = program.run_decompositions()
    check_decomposed_regions(captured, program.graph_module)
    graph = materialize_regions(program.module(), captured)
    erase_export_guards(graph)
    graph.graph.lint()
    graph.recompile()
    return graph


def _trace(model, rules, *inputs, strict=False):
    """Trace a model as torch_tensorrt.compile(model, regions=rules) does."""
    options = {"regions": rules, "strict": strict, "device": inputs[0].device}
    return _trace_with_regions(model, list(inputs), {}, options)


SOURCES = ["program", "strict program", "model", "strict model"]


def _capture(source, model, rules, *inputs, preserve=()):
    """Capture rules on a program exported from the model, or on the model itself."""
    strict = source.startswith("strict")
    if source.endswith("model"):
        return _materialize(*_trace(model, rules, *inputs, strict=strict))
    program = torch.export.export(
        model, inputs, strict=strict, preserve_module_call_signature=preserve
    )
    return _materialize(program, rules)


def _compile(model_or_program, inputs, **options):
    return torch_tensorrt.compile(
        model_or_program, inputs=list(inputs), ir="dynamo", min_block_size=1, **options
    )


def _summary(graph):
    return [
        (record.id, record.backend, record.route, record.operator_key)
        for record in get_region_records(graph)
    ]


def _names(graph):
    return [record.name for record in get_region_records(graph)]


def _operations(module):
    return [
        node.target._schema.name.split("::")[-1]
        for node in module.graph.nodes
        if isinstance(node.target, torch._ops.OpOverload)
    ]


def _bodies(graph):
    """The operations each region runs, in program order."""
    return [
        _operations(graph.get_submodule(record.child_target))
        for record in get_region_records(graph)
    ]


def _compiled_body(compiled, record):
    """The operations of a region that a compiled model keeps in PyTorch."""
    owner = compiled.get_submodule(record.owner)
    return _operations(owner.get_submodule(record.child_target))


def _assert_compiled_close(actual, expected):
    # TensorRT may merge float32 arithmetic. The operators that differ from
    # their modules on purpose differ by 100.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def _engines(compiled):
    return [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]


def _assert_outside_engines(compiled, records):
    """The regions that stay in PyTorch are called once, outside every engine."""
    audit_region_placement(compiled, get_region_records(compiled))
    for record in records:
        assert record.route == "torch"
        assert record.owner is not None
        assert not record.owner.startswith("_run_on_acc")


def _assert_inside_engines(compiled, records):
    """Each kernel region is one plugin layer of an engine, called nowhere else."""
    audit_region_placement(compiled, get_region_records(compiled))
    for record in records:
        assert record.route == "plugin"
        assert record.owner.startswith("_run_on_acc")
        engine = compiled.get_submodule(record.owner)
        assert record.id in engine._torch_tensorrt_plugin_regions
        assert str(record.plugin_op) in engine.get_layer_info()
    for module in compiled.modules():
        if isinstance(module, torch.fx.GraphModule):
            assert not any(
                node.target == record.plugin_op
                for node in module.graph.nodes
                for record in records
            )


@pytest.mark.parametrize(
    "selector, names, bodies",
    [
        ("layers.0.act", ["layers.0.act"], [["relu"]]),
        ("layers.*.act", ["layers.0.act", "layers.1.act"], [["relu"]] * 2),
        ("**.act", ["layers.0.act", "layers.1.act"], [["relu"]] * 2),
        ("lay*.1.sc*", ["layers.1.scale"], [["mul", "add"]]),
        (
            "layers.1.*",
            ["layers.1.act", "layers.1.scale"],
            [["relu"], ["mul", "add"]],
        ),
        # Selecting a module takes the modules inside it.
        (
            "layers.*",
            ["layers.0", "layers.1"],
            [["relu", "add", "mul", "add"]] * 2,
        ),
        ("head", ["head"], [["linear", "tanh"]]),
        ("head.*", ["head.0", "head.1"], [["linear"], ["tanh"]]),
        (torch.nn.Sequential, ["head"], [["linear", "tanh"]]),
        (torch.nn.Linear, ["head.0"], [["linear"]]),
        # The ModuleList and the Identity also match but run no operations.
        ("*", ["head"], [["linear", "tanh"]]),
        # Every module matches; the outermost ones take their operations.
        (
            "**",
            ["layers.0", "layers.1", "head"],
            [["relu", "add", "mul", "add"]] * 2 + [["linear", "tanh"]],
        ),
    ],
    ids=[
        "path",
        "one name",
        "any names",
        "part of a name",
        "children",
        "parents",
        "sequential",
        "sequential children",
        "sequential class",
        "class",
        "some without operations",
        "outermost",
    ],
)
def test_selector_places_each_selected_module_call_in_a_region(selector, names, bodies):
    x = torch.randn(2, 8)
    model = Stack().eval()
    selected = sum(bodies, [])
    for source in SOURCES:
        graph = _capture(source, model, {selector: ExecuteInTorch()}, x)
        records = get_region_records(graph)
        # A region is named after its module's path.
        assert [record.name for record in records] == names, source
        assert [record.id for record in records] == [
            f"{name}#{index}" for index, name in enumerate(names)
        ]
        assert all(record.route == "torch" for record in records)
        assert _bodies(graph) == bodies, source
        # The selected activations run only in their regions.
        remaining = _operations(graph)
        assert remaining.count("relu") == 2 - selected.count("relu"), source
        assert remaining.count("tanh") == 1 - selected.count("tanh"), source
        torch.testing.assert_close(graph(x), model(x))
        audit_region_placement(graph, records)


def test_class_lookup_does_not_import_the_module_a_program_names(tmp_path, monkeypatch):
    # A module that could be imported and holds a class of the recorded name.
    name = "region_rules_module_nobody_imported"
    (tmp_path / f"{name}.py").write_text(
        "import torch\n\n\nclass Activation(torch.nn.Module):\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    x = torch.ones(2, 8)
    program = torch.export.export(Sandwich().eval(), (x,))
    # Make the program name that class for one of its modules.
    for node in program.graph.nodes:
        stack = node.meta.get("nn_module_stack")
        if stack:
            node.meta["nn_module_stack"] = {
                key: (path, f"{name}.Activation" if path == "act" else kind)
                for key, (path, kind) in stack.items()
            }
    with pytest.raises(RegionCaptureError, match="matched no module call"):
        _materialize(program, {Activation: ExecuteInTorch()})
    assert name not in sys.modules
    # Once the module is loaded, the class is found by the recorded name.
    spec = importlib.util.spec_from_file_location(name, tmp_path / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    graph = _materialize(program, {module.Activation: ExecuteInTorch()})
    assert _names(graph) == ["act"]
    assert _bodies(graph) == [["relu"]]


@pytest.mark.parametrize(
    "selector, container",
    [
        ("layers", "'layers'"),
        ("skip", None),
        (torch.nn.ModuleList, "torch.nn.modules.container.ModuleList"),
        (torch.nn.Identity, None),
    ],
    ids=["module list", "identity", "module list class", "identity class"],
)
def test_rule_that_selects_only_modules_without_operations_is_rejected(
    selector, container
):
    # A ModuleList is never called, and an Identity runs no operations.
    no_call = (
        "matched no module call in the traced program; its modules include "
        "'layers.0', 'layers.0.act'.* A container such as a ModuleList is never "
        "called: select its children"
    )
    for source in SOURCES:
        message = no_call
        if container and source.endswith("model"):
            # The model shows that the rule selected nothing but containers.
            message = "^" + re.escape(
                f"Region rule {container} selects only containers that are never "
                "called ('layers'); select their children, for example 'layers.*'"
            )
        with pytest.raises(RegionCaptureError, match=message + "$"):
            _capture(
                source, Stack().eval(), {selector: ExecuteInTorch()}, torch.ones(2, 8)
            )


@requires_cuda
@pytest.mark.parametrize(
    "entry", ["model", "model, module object", "program", "dynamo program"]
)
def test_execute_in_torch_rule_keeps_the_module_between_engines(entry):
    x = torch.randn(2, 8, device="cuda")
    model = Sandwich().cuda().eval()
    rules = {"act": ExecuteInTorch()}
    with torch.no_grad():
        if entry == "model":
            compiled = _compile(model, [x], regions=rules)
        elif entry == "model, module object":
            compiled = _compile(model, [x], regions={model.act: ExecuteInTorch()})
        elif entry == "program":
            compiled = _compile(torch.export.export(model, (x,)), [x], regions=rules)
        else:
            compiled = torch_tensorrt.dynamo.compile(
                torch.export.export(model, (x,)),
                inputs=[x],
                min_block_size=1,
                regions=rules,
            )
        # The activation splits what would otherwise be one engine.
        assert len(_engines(compiled)) == 2
        (record,) = get_region_records(compiled)
        assert (record.id, record.backend) == ("act#0", "eager")
        _assert_outside_engines(compiled, [record])
        assert _compiled_body(compiled, record) == ["relu"]
        _assert_compiled_close(compiled(x), model(x))
        if entry == "model":
            assert len(_engines(_compile(model, [x]))) == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="CUDA Inductor requires Triton support on the device",
)
def test_inductor_rule_compiles_the_module_on_first_call(observed_inductor):
    x = torch.randn(3, 257, device="cuda")
    model = Sandwich(257).cuda().eval()
    compiled = _compile(model, [x], regions={Scale: ExecuteInTorch(backend="inductor")})
    assert len(_engines(compiled)) == 2
    (record,) = get_region_records(compiled)
    assert (record.id, record.backend) == ("scale#0", "inductor")
    _assert_outside_engines(compiled, [record])
    assert observed_inductor.counter.frame_count == 0
    with torch.no_grad():
        for value in (x, x + 1):
            _assert_compiled_close(compiled(value), model(value))
    assert observed_inductor.counter.frame_count == 1
    assert observed_inductor.executions == 2


@requires_cuda
@pytest.mark.parametrize("entry", ["model", "program"])
def test_fused_op_rule_runs_the_operator_between_engines(entry, monkeypatch):
    def unexpected_converter(*args, **kwargs):
        raise AssertionError("The rule's operator was incorrectly sent to TensorRT")

    monkeypatch.setitem(
        DYNAMO_ATEN_CONVERTERS,
        torch.ops.ttrt_region_rules_test.plus_hundred.default,
        [ConverterSupport(unexpected_converter)],
    )
    x = torch.randn(2, 8, device="cuda")
    model = Sandwich().cuda().eval()
    source = model if entry == "model" else torch.export.export(model, (x,))
    with torch.no_grad():
        compiled = _compile(source, [x], regions={Activation: FusedOp(_plus_hundred)})
        assert len(_engines(compiled)) == 2
        (record,) = get_region_records(compiled)
        assert record.operator_key == "ttrt_region_rules_test::plus_hundred.default"
        _assert_outside_engines(compiled, [record])
        _calls.clear()
        # The operator's result replaces the module's.
        _assert_compiled_close(compiled(x), model.scale((x + 1 + 100) * 3) - 2)
        assert _calls == ["plus_hundred"]


class BiasedActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(3, 17))

    def forward(self, x):
        return torch.relu(x.float() + self.bias)


class KernelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.biased = BiasedActivation()
        self.difference = WeightedDifference()

    def forward(self, x):
        hidden = x * 2
        # hidden is read again after the first module's no-op cast returned it.
        y = self.biased(hidden)
        return self.difference(y, factor=1.0, second=hidden + 1) - 1


@requires_cuda_qdp
@pytest.mark.parametrize("entry", ["model", "program"])
def test_fused_kernel_rules_run_as_plugins_in_an_engine(entry):
    x = torch.randn(3, 17, device="cuda")
    model = KernelModel().cuda().eval()
    rules = {
        "biased": _kernel_rule(lambda: _bias_kernel, inputs=[0, "bias"]),
        "difference": _kernel_rule(
            lambda: _subtract_kernel, inputs=[0, 1], name="subtract"
        ),
    }
    with torch.no_grad():
        if entry == "model":
            source = model
        else:
            source = torch.export.export(
                model, (x,), preserve_module_call_signature=("difference",)
            )
        compiled = _compile(source, [x], regions=rules)
        records = get_region_records(compiled)
        assert [record.id for record in records] == ["biased#0", "subtract#1"]
        _assert_inside_engines(compiled, records)
        _assert_compiled_close(compiled(x), model(x))
        other = torch.randn_like(x)
        _assert_compiled_close(compiled(other), model(other))
        # The kernels take their arguments in the order the rules list them.
        bias, subtract = (record.plugin_op for record in records)
        torch.testing.assert_close(
            bias(x * 2, model.biased.bias), torch.relu(x * 2 + model.biased.bias)
        )
        torch.testing.assert_close(subtract(x, other), x - other)


@pytest.mark.parametrize(
    "make, error, message",
    [
        (lambda: ExecuteInTorch(backend="unknown"), ValueError, "backend"),
        (lambda: ExecuteInTorch(backend=3), TypeError, "backend must be a string"),
        (lambda: ExecuteInTorch(name=3), TypeError, "name must be a string or None"),
        (
            lambda: FusedOp("ttrt_region_rules_test::plus_hundred"),
            TypeError,
            "custom_op function or torch.ops",
        ),
        # A built-in operator is not a replacement the caller wrote.
        (
            lambda: FusedOp(torch.ops.aten.relu),
            TypeError,
            "fused_op requires a registered torch custom operator",
        ),
        (
            lambda: FusedOp(torch.ops.aten.relu.default),
            TypeError,
            "fused_op requires a registered torch custom operator",
        ),
        (
            lambda: FusedOp(torch.ops.ttrt_region_rules_invalid.mutate),
            RegionCaptureError,
            "fused_op does not support a mutating operator schema",
        ),
        (
            lambda: FusedOp(torch.ops.ttrt_region_rules_invalid.alias.default),
            RegionCaptureError,
            "fused_op does not support an aliasing operator schema",
        ),
        (
            lambda: FusedOp(torch.ops.ttrt_region_rules_invalid.scalar),
            RegionCaptureError,
            "fused_op requires positional Tensor inputs and Tensor outputs",
        ),
        (lambda: FusedOp(_plus_hundred, inputs="weight"), TypeError, "tuple or list"),
        (lambda: FusedOp(_plus_hundred, inputs=[0.5]), TypeError, "positions"),
        (lambda: FusedOp(_plus_hundred, inputs=[True]), TypeError, "positions"),
        (lambda: FusedOp(_plus_hundred, inputs=[-1]), ValueError, "negative"),
        (lambda: FusedOp(_plus_hundred, name=3), TypeError, "name must be a string"),
        (
            lambda: FusedKernel(_plus_hundred),
            TypeError,
            "kernel=lambda: my_kernel; got CustomOpDef",
        ),
        (
            lambda: FusedKernel(lambda: None, grid=3),
            TypeError,
            "grid must be a function or functools.partial; got int",
        ),
        (
            lambda: FusedKernel(lambda: None, inputs=[None]),
            TypeError,
            "FusedKernel inputs must be a tuple or list",
        ),
    ],
)
def test_rule_arguments_are_checked_when_the_rule_is_created(make, error, message):
    with pytest.raises(error, match=message):
        make()


def test_inputs_name_parameters_and_buffers():
    x = torch.randn(2, 8)
    model = Stack().eval()
    rules = {
        "layers.*.scale": FusedOp(_scale_shift, inputs=[0, "weight", "shift"]),
        # The affine operator stands in for the whole head, activation included.
        "head": FusedOp(_affine, inputs=[0, "0.weight", "0.bias"]),
    }
    hidden = x
    for layer in model.layers:
        hidden = layer(hidden - 0.5)
    expected = model.head[0](hidden) * 2
    for source in SOURCES:
        graph = _capture(source, model, rules, x)
        assert _bodies(graph) == [["scale_shift"], ["scale_shift"], ["affine"]]
        _calls.clear()
        torch.testing.assert_close(graph(x), expected)
        assert _calls == ["scale_shift", "scale_shift", "affine"], source


def test_inputs_name_a_parameter_of_a_module_inside_the_selected_one():
    x = torch.randn(2, 8)
    model = Wrapped().eval()
    rules = {"wrapper": FusedOp(_affine, inputs=[0, "linear.weight", "linear.bias"])}
    for source in SOURCES:
        graph = _capture(source, model, rules, x)
        assert _bodies(graph) == [["affine"]], source
        _calls.clear()
        torch.testing.assert_close(graph(x), model(x))
        assert _calls == ["affine"], source


class ScaledProjection(torch.nn.Module):
    """A projection whose child has a parameter that is also named weight."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))
        self.scale = Scale()

    def forward(self, x):
        return self.scale(x @ self.weight.T)


class TiedAboveAChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = torch.nn.Linear(8, 8, bias=False)
        self.decode = ScaledProjection()
        self.decode.weight = self.encode.weight

    def forward(self, x):
        return self.decode(self.encode(x) * 2)


def test_tied_weight_is_not_mistaken_for_a_parameter_of_a_child():
    x = torch.randn(2, 8)
    model = TiedAboveAChild().eval()
    # "weight" is the module's own weight, recorded under either tied name;
    # the child's weight is "scale.weight".
    inputs = [0, "weight", "scale.weight", "scale.shift"]
    rules = {"decode": FusedOp(_project_scale_shift, inputs=inputs)}
    for source in SOURCES:
        graph = _capture(source, model, rules, x)
        assert _bodies(graph) == [["project_scale_shift"]], source
        _calls.clear()
        torch.testing.assert_close(graph(x), model(x))
        assert _calls == ["project_scale_shift"], source


def test_parameter_name_is_not_looked_up_in_a_child_of_the_selected_module():
    # Only a tied weight, which lies outside the selected module, is found
    # under another module's name: the wrapper has no weight of its own.
    rules = {"wrapper": FusedOp(_affine, inputs=[0, "weight", "bias"])}
    for source in SOURCES:
        with pytest.raises(
            RegionCaptureError,
            match="Region rule 'wrapper' at 'wrapper': cannot find the parameter "
            r"or buffer 'weight'; the module reads .*linear\.weight.*linear\.bias",
        ):
            _capture(source, Wrapped().eval(), rules, torch.ones(2, 8))


@pytest.mark.parametrize(
    "layout, inputs",
    [
        ("positional", [0, 1]),
        ("keyword", [0, 1]),
        # Keyword arguments count in the order the call passes them.
        ("keywords reversed", [1, 0]),
        ("number between", [0, 1]),
    ],
)
def test_positions_follow_the_call_when_the_model_is_traced(layout, inputs):
    x = torch.randn(2, 8)
    model = DifferenceCaller(layout).eval()
    for source in ("model", "strict model"):
        rules = {"difference": FusedOp(_weighted_difference, inputs=inputs)}
        graph = _capture(source, model, rules, x)
        assert _bodies(graph) == [["weighted_difference"]], source
        _calls.clear()
        # The module reads its second argument first; the operator still gets
        # the first argument first.
        torch.testing.assert_close(graph(x), model(x))
        assert _calls == ["weighted_difference"], source
        # Listing the positions the other way round swaps the arguments.
        rules = {"difference": FusedOp(_weighted_difference, inputs=inputs[::-1])}
        graph = _capture(source, model, rules, x)
        torch.testing.assert_close(graph(x), ((x * 3) - 2 * (x + 1)) * 2)


# What the errors say when a program does not record a module's calls.
RECORD_THE_CALL = (
    r"Select a module that takes and returns one tensor, or export with "
    r"torch\.export\.export\(\.\.\., preserve_module_call_signature=\(<module "
    r"path>,\)\); torch_tensorrt\.compile records the order itself when it is "
    r"given the nn\.Module$"
)


@pytest.mark.parametrize(
    "source, model, selector, op, inputs, message",
    [
        (
            "model",
            DifferenceCaller,
            "difference",
            _weighted_difference,
            [0, 2],
            "input position 2 is out of range; the call passes 2 tensors$",
        ),
        (
            "program",
            Sandwich,
            "act",
            _weighted_difference,
            [0, 1],
            r"input position 1 is out of range; the module reads 1 tensor\(s\) "
            "from its caller$",
        ),
        (
            "model",
            FirstOnlyCaller,
            "first_only",
            _weighted_difference,
            [0, 1],
            "the module does not read the tensor at position 1 of its call",
        ),
        (
            "program",
            Sandwich,
            "scale",
            _scale_shift,
            [0, "weight", "bias"],
            "cannot find the parameter or buffer 'bias'; the module reads "
            "'weight', 'shift'$",
        ),
        (
            "program",
            Wrapped,
            "wrapper",
            _affine,
            [0, "linear.weight", "weight"],
            "cannot find the parameter or buffer 'weight'; the module reads "
            r"'linear\.weight', 'linear\.bias'$",
        ),
        # Each tensor the rule left out is named the way inputs= lists it.
        (
            "program",
            Sandwich,
            "scale",
            _scale_shift,
            None,
            r"the module reads 3 tensors; list them with inputs=\[...\] in the "
            r"order the replacement takes them\. They are: position 0, 'weight', "
            "'shift'$",
        ),
        (
            "program",
            Sandwich,
            "scale",
            _weighted_difference,
            [0, "weight"],
            "inputs must list every tensor the module reads; add 'shift'$",
        ),
        (
            "program",
            Sandwich,
            "scale",
            _scale_shift,
            [0, "weight", "weight"],
            "inputs must list every tensor the module reads; add 'shift'$",
        ),
        (
            "program",
            Sandwich,
            "scale",
            _weighted_difference,
            ["weight", "shift"],
            "inputs must list every tensor the module reads; add position 0$",
        ),
        (
            "program",
            Sandwich,
            "scale",
            _plus_hundred,
            [0],
            "inputs must list every tensor the module reads; add 'weight', " "'shift'$",
        ),
        (
            "program",
            Wrapped,
            "wrapper",
            _affine,
            [0, "linear.weight"],
            "inputs must list every tensor the module reads; add 'linear.bias'$",
        ),
        # A recorded call tells the position of each tensor it passes.
        (
            "model",
            DifferenceCaller,
            "difference",
            _weighted_difference,
            [1],
            "inputs must list every tensor the module reads; add position 0$",
        ),
        (
            "model",
            DifferenceCaller,
            "difference",
            _weighted_difference,
            None,
            # The positions are in call order: the module reads the second first.
            r"the module reads 2 tensors; list them with inputs=\[...\] in the "
            r"order the replacement takes them\. They are: position 0, position 1$",
        ),
    ],
    ids=[
        "position past the call",
        "position past the only tensor",
        "position the module does not read",
        "unknown name",
        "name of a child without its path",
        "omitted",
        "missing",
        "repeated",
        "missing position",
        "missing several",
        "missing name inside a child",
        "missing position of a recorded call",
        "omitted with a recorded call",
    ],
)
def test_inputs_that_do_not_match_the_module_are_rejected(
    source, model, selector, op, inputs, message
):
    with pytest.raises(RegionCaptureError, match=message):
        _capture(
            source,
            model().eval(),
            {selector: FusedOp(op, inputs=inputs)},
            torch.ones(2, 8),
        )


class TwoLayouts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.difference = WeightedDifference()

    def forward(self, x):
        y = self.difference(x + 1, x * 3)
        return self.difference(x - 1, second=y) * 2


def test_positions_are_not_guessed_when_calls_pass_arguments_differently(caplog):
    # torch.export cannot record calls of one module with different layouts.
    rules = {"difference": FusedOp(_weighted_difference, inputs=[0, 1])}
    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            RegionCaptureError,
            match="the module reads 2 tensors from its caller .* and this "
            "program does not record the order its call passes them",
        ):
            _capture("model", TwoLayouts().eval(), rules, torch.ones(2, 8))
    # The compile call says why it could not record the calls.
    assert (
        "torch.export could not record how the modules selected by regions= "
        "are called" in caplog.text
    )
    # Keeping the module in PyTorch needs no order.
    x = torch.randn(2, 8)
    model = TwoLayouts().eval()
    graph = _capture("model", model, {"difference": ExecuteInTorch()}, x)
    assert _names(graph) == ["difference", "difference"]
    torch.testing.assert_close(graph(x), model(x))


def test_unrecorded_call_error_does_not_suggest_what_the_caller_did():
    rules = {"difference": FusedOp(_weighted_difference, inputs=[0, 1])}
    # torch_tensorrt.compile traces the model and captures in one session.
    with RegionCompilationSession():
        with pytest.raises(
            RegionCaptureError,
            match="Region rule 'difference' at 'difference': the module reads 2 "
            r"tensors from its caller \(mul\[2, 8\], add\[2, 8\]\), and this "
            r"program does not record the order its call passes them\. "
            r"torch\.export could not record it \(\w+: [^\n]+\)\. Select a module "
            "that takes and returns one tensor, or write the scope in the "
            "module's forward$",
        ) as error:
            _capture("model", TwoLayouts().eval(), rules, torch.ones(2, 8))
    assert "when it is given the nn.Module" not in str(error.value)


def test_trace_failure_is_not_blamed_on_recording_the_calls(caplog):
    rules = {"act": FusedOp(_plus_hundred)}
    with caplog.at_level(logging.WARNING):
        with pytest.raises(AssertionError, match="This model cannot be traced"):
            _trace(Untraceable().eval(), rules, torch.ones(2, 8))
    assert "could not record how the modules selected by regions=" not in caplog.text


class SameTensorTwice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.difference = WeightedDifference()

    def forward(self, x):
        y = x + 1
        return self.difference(y, y) * 2


@requires_cuda_qdp
def test_kernel_takes_one_tensor_passed_twice_as_two_arguments():
    x = torch.randn(3, 17, device="cuda")
    model = SameTensorTwice().cuda().eval()
    rules = {
        "difference": _kernel_rule(lambda: _weighted_difference_kernel, inputs=[0, 1])
    }
    with torch.no_grad():
        compiled = _compile(model, [x], regions=rules)
        (record,) = get_region_records(compiled)
        assert len(record.input_names) == 2
        _assert_inside_engines(compiled, [record])
        assert len(_engines(compiled)) == 1
        # first - 2 * second, with the one tensor as both.
        _assert_compiled_close(compiled(x), -(x + 1) * 2)
        _assert_compiled_close(compiled(x), model(x))


class Halves(torch.nn.Module):
    def forward(self, x):
        # The second result is computed first.
        low = x.clamp(max=0)
        high = x.clamp(min=0)
        return high, low


class HalvesCaller(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.halves = Halves()

    def forward(self, x):
        high, low = self.halves(x + 0.1)
        return high * 2 - low * 3


class Pair(torch.nn.Module):
    """Operations before and after one module that returns two tensors."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        first, second = self.inner(x + 1)
        return first * 3 - second


def test_fused_op_rule_outputs_follow_the_order_the_module_returns():
    x = torch.randn(2, 8)
    model = HalvesCaller().eval()
    rules = {"halves": FusedOp(_halves)}
    key = "ttrt_region_rules_test::halves.default"
    for source in SOURCES:
        # The model's trace records the call; a program has to be asked to.
        graph = _capture(source, model, rules, x, preserve=("halves",))
        assert _summary(graph) == [("halves#0", "eager", "torch", key)], source
        assert _bodies(graph) == [["halves"]], source
        assert "clamp" not in _operations(graph), source
        _calls.clear()
        # The operator returns (high, low) as the module does.
        torch.testing.assert_close(graph(x), model(x))
        assert _calls == ["halves"], source


def test_outputs_are_not_ordered_by_guess_when_a_program_did_not_record_the_call():
    x = torch.randn(2, 8)
    model = HalvesCaller().eval()
    for source in ("program", "strict program"):
        with pytest.raises(
            RegionCaptureError,
            match="fused_op region 'halves#0': the selected module returns 2 "
            r"tensors \(clamp\[2, 8\], clamp_1\[2, 8\]\), and this program does "
            r"not record the order it returns them\. " + RECORD_THE_CALL,
        ):
            _capture(source, model, {"halves": FusedOp(_halves)}, x)
        # Keeping the module in PyTorch needs no order.
        graph = _capture(source, model, {"halves": ExecuteInTorch()}, x)
        assert _bodies(graph) == [["clamp", "clamp"]], source
        torch.testing.assert_close(graph(x), model(x))


@requires_cuda_qdp
@pytest.mark.parametrize("entry", ["model", "program"])
def test_fused_kernel_rule_outputs_follow_the_order_the_module_returns(entry):
    x = torch.randn(3, 17, device="cuda")
    model = HalvesCaller().cuda().eval()
    rules = {"halves": _kernel_rule(lambda: _halves_kernel)}
    with torch.no_grad():
        if entry == "model":
            source = model
        else:
            source = torch.export.export(
                model, (x,), preserve_module_call_signature=("halves",)
            )
        compiled = _compile(source, [x], regions=rules)
        (record,) = get_region_records(compiled)
        assert record.id == "halves#0"
        assert len(record.output_names) == 2
        _assert_inside_engines(compiled, [record])
        assert len(_engines(compiled)) == 1
        # The kernel writes (high, low) in the order the module returns them.
        _assert_compiled_close(compiled(x), model(x))
        other = torch.randn_like(x)
        _assert_compiled_close(compiled(other), model(other))
        if entry == "program":
            with pytest.raises(
                RegionCaptureError,
                match="fused_kernel region 'halves#0': the selected module "
                "returns 2 tensors .* does not record the order it returns them",
            ):
                _compile(torch.export.export(model, (x,)), [x], regions=rules)


@requires_cuda_qdp
def test_fused_kernel_rule_over_a_module_that_only_hands_its_input_through():
    model = Around(torch.nn.Dropout(0.5)).eval()
    with pytest.raises(
        RegionCaptureError,
        match="fused_kernel region 'inner#0': the selected module returns its "
        "input unchanged, so its operator or kernel would never run",
    ):
        _capture(
            "model",
            model,
            {"inner": _kernel_rule(lambda: _bias_kernel)},
            torch.ones(2, 8),
        )


class FirstHalf(torch.nn.Module):
    def forward(self, x):
        return x[:, : x.shape[1] // 2]


class Reshaped(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0] * 2, -1)


class Transposed(torch.nn.Module):
    def forward(self, x):
        return x.transpose(0, 1)


class Chunks(torch.nn.Module):
    def forward(self, x):
        return x.chunk(2, dim=-1)


class DoubledChunks(torch.nn.Module):
    def forward(self, x):
        return (x * 2).chunk(2, dim=-1)


# A module whose results share storage, the operator that computes the same,
# and the operations of the module once capture has copied what is shared.
VIEWS = {
    "part of the input": (lambda: Around(FirstHalf()), _first_half, ["slice", "clone"]),
    "reshaped input": (lambda: Around(Reshaped()), _reshaped, ["reshape", "clone"]),
    "transposed input": (
        lambda: Around(Transposed()),
        _transposed,
        ["transpose", "clone"],
    ),
    "chunks of the input": (
        lambda: Pair(Chunks()),
        _chunks,
        ["chunk", "clone", "clone"],
    ),
    "chunks of a result": (
        lambda: Pair(DoubledChunks()),
        _doubled_chunks,
        ["mul", "chunk", "clone"],
    ),
}


def _view_rule(case, kind):
    make, op, body = VIEWS[case]
    name = op._opoverload._schema.name.split("::")[-1]
    if kind == "torch":
        return make, ExecuteInTorch(), body, []
    return make, FusedOp(op), [name], [name]


@pytest.mark.parametrize(
    "case, kind",
    [
        ("part of the input", "torch"),
        ("part of the input", "operator"),
        ("reshaped input", "torch"),
        ("reshaped input", "operator"),
        ("transposed input", "torch"),
        # The operator returns the ordinary layout, not the strides of a view.
        ("transposed input", "operator"),
        ("chunks of the input", "torch"),
        ("chunks of the input", "operator"),
        ("chunks of a result", "torch"),
        ("chunks of a result", "operator"),
    ],
)
def test_rule_selects_a_module_whose_results_share_storage(case, kind):
    x = torch.randn(2, 8)
    make, rule, body, calls = _view_rule(case, kind)
    model = make().eval()
    for source in SOURCES:
        # A region's results have storage of their own: capture copies them.
        graph = _capture(source, model, {"inner": rule}, x, preserve=("inner",))
        assert _names(graph) == ["inner"], source
        assert _bodies(graph) == [body], source
        _calls.clear()
        torch.testing.assert_close(graph(x), model(x))
        assert _calls == calls, source


class WritesInput(torch.nn.Module):
    def forward(self, x):
        return x.relu_() * 3


class WritesSharedTemporary(torch.nn.Module):
    def forward(self, x):
        y = x * 3
        flat = y.view(-1)
        y.relu_()
        # flat reads the tensor that the activation wrote.
        return y + flat.view(y.shape)


def test_fused_rule_replaces_a_module_that_writes_a_tensor_it_just_created():
    x = torch.randn(2, 8)
    model = Around(InPlaceBlock()).eval()
    rules = {"inner": FusedOp(_affine, inputs=[0, "linear.weight", "linear.bias"])}
    for source in SOURCES:
        graph = _capture(source, model, rules, x)
        assert _bodies(graph) == [["affine"]], source
        _calls.clear()
        # The affine operator stands in for the layer and its activation.
        torch.testing.assert_close(graph(x), model.inner.linear(x + 1) * 2)
        assert _calls == ["affine"], source


@pytest.mark.parametrize(
    "inner",
    [lambda: torch.nn.ReLU(inplace=True), WritesInput, WritesSharedTemporary],
    ids=["in-place activation", "its input", "a tensor that is read elsewhere"],
)
def test_rule_over_a_module_that_writes_a_tensor_others_read_is_rejected(inner):
    for rule in (ExecuteInTorch(), FusedOp(_plus_hundred)):
        for source in SOURCES:
            with pytest.raises(
                RegionCaptureError,
                match="Mutation or effects inside regions are unsupported: "
                "aten.relu_.default in 'inner#0'$",
            ):
                _capture(
                    source, Around(inner()).eval(), {"inner": rule}, torch.ones(2, 8)
                )


@pytest.mark.parametrize(
    "inputs, message",
    [
        ([0], "inputs must list every tensor the module reads; add "),
        (
            None,
            r"the module reads 2 tensors; list them with inputs=\[...\] in the "
            r"order the replacement takes them\. They are: position 0, ",
        ),
    ],
    ids=["position", "omitted"],
)
def test_fused_rule_selects_a_module_that_creates_a_tensor_constant(inputs, message):
    x = torch.randn(2, 8)
    model = Around(HalfActivation()).eval()
    for source in SOURCES:
        # The constant is a tensor the module reads: the rule has to list it,
        # under the name torch.export gave it.
        with pytest.raises(
            RegionCaptureError,
            match="Region rule 'inner' at 'inner': " + message + "'lifted_tensor_0' "
            r"\(a tensor the module creates in forward\)$",
        ):
            _capture(source, model, {"inner": FusedOp(_plus_hundred, inputs=inputs)}, x)
        # An operator that takes the constant gets it where the rule lists it.
        rules = {"inner": FusedOp(_weighted_difference, inputs=[0, "lifted_tensor_0"])}
        graph = _capture(source, model, rules, x)
        assert _bodies(graph) == [["weighted_difference"]], source
        _calls.clear()
        torch.testing.assert_close(graph(x), (x + 1 - 2 * 0.5) * 2)
        assert _calls == ["weighted_difference"], source


@pytest.mark.parametrize("source", SOURCES)
def test_rule_around_a_scope_written_in_forward_is_rejected(source):
    x = torch.randn(2, 8)
    model = Around(Annotated()).eval()
    with pytest.raises(
        RegionCaptureError,
        match="A region rule selects 'inner', whose operations surround "
        "'written', a region written in the model's source; regions cannot nest$",
    ):
        _capture(source, model, {"inner": ExecuteInTorch()}, x)
    # A rule may select a module beside the scope.
    graph = _capture(source, model, {"inner.act": ExecuteInTorch()}, x)
    assert [record.id for record in get_region_records(graph)] == [
        "inner.act#0",
        "written#1",
    ]
    torch.testing.assert_close(graph(x), model(x))


NOT_A_MAPPING = (
    "^regions must map module paths, module classes or modules to region rules, "
    r"such as \{.+\}$"
)


@requires_cuda
@pytest.mark.parametrize(
    "regions, error, message",
    [
        ([("act", ExecuteInTorch())], TypeError, NOT_A_MAPPING),
        (
            {3: ExecuteInTorch()},
            TypeError,
            "by module class, or by the module itself; got 3$",
        ),
        ({"act": "eager"}, TypeError, "must be ExecuteInTorch, FusedOp or FusedKernel"),
        ({"": ExecuteInTorch()}, ValueError, "cannot select the whole model"),
        (
            {Activation(): ExecuteInTorch()},
            RegionCaptureError,
            "selects a module object, which an exported program does not hold",
        ),
    ],
    ids=["list", "number selector", "string rule", "empty path", "module object"],
)
def test_compiling_a_program_checks_the_rules(regions, error, message):
    x = torch.ones(2, 8, device="cuda")
    program = torch.export.export(Sandwich().cuda().eval(), (x,))
    with pytest.raises(error, match=message):
        torch_tensorrt.dynamo.compile(program, inputs=[x], regions=regions)
    with pytest.raises(error, match=message):
        torch_tensorrt.compile(program, inputs=[x], ir="dynamo", regions=regions)


def test_trace_does_not_take_rules():
    # Rules select modules of the model being compiled; a trace has no use for
    # them and would otherwise drop them without a word.
    with pytest.raises(
        ValueError,
        match="regions= selects modules when a model is compiled: pass it to "
        "torch_tensorrt.compile or torch_tensorrt.dynamo.compile, not to "
        "torch_tensorrt.dynamo.trace$",
    ):
        torch_tensorrt.dynamo.trace(
            Untraceable().eval(),
            arg_inputs=[torch.ones(2, 8)],
            regions={"act": ExecuteInTorch()},
        )


@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.fx_frontend,
    reason="The FX frontend is required",
)
@pytest.mark.filterwarnings("ignore:FX frontend is deprecated")
def test_regions_are_rejected_by_frontends_that_cannot_honor_them():
    with pytest.raises(
        ValueError, match="regions= requires ir='dynamo' or ir='torch_compile'"
    ):
        torch_tensorrt.compile(
            Sandwich().eval(),
            inputs=[torch.ones(2, 8)],
            ir="fx",
            regions={"act": ExecuteInTorch()},
        )


@requires_cuda
def test_rules_are_rejected_where_regions_are_unsupported():
    x = torch.ones(2, 8, device="cuda")
    model = Sandwich().cuda().eval()
    program = torch.export.export(model, (x,))
    rules = {"act": ExecuteInTorch()}
    dynamic = torch_tensorrt.Input(
        min_shape=(1, 8), opt_shape=(2, 8), max_shape=(4, 8), dtype=torch.float32
    )
    with pytest.raises(RegionCaptureError, match="requires static shapes"):
        _compile(model, [dynamic], regions=rules)
    # Engine-only output takes no rules; they are refused before the model is traced.
    with pytest.raises(TypeError, match="unexpected keyword argument 'regions'"):
        torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            program, inputs=[x], regions=rules
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'regions'"):
        torch_tensorrt.convert_method_to_trt_engine(
            Untraceable().cuda().eval(), inputs=[x], ir="dynamo", regions=rules
        )
    with pytest.raises(
        RegionError,
        match=r"require_full_compilation=True \(region 'act#0' must run outside",
    ):
        _compile(program, [x], regions=rules, require_full_compilation=True)
    with pytest.raises(RegionError, match="immutable_weights=False"):
        _compile(program, [x], regions=rules, immutable_weights=False)


@requires_cuda
def test_rules_cannot_be_used_where_weights_are_rebuilt_in_place():
    x = torch.randn(2, 8, device="cuda")
    model = Sandwich().cuda().eval()
    rules = {"act": ExecuteInTorch()}
    mutable = torch_tensorrt.MutableTorchTensorRTModule(
        model, min_block_size=1, regions=rules
    )
    with pytest.raises(RegionError, match="incompatible with immutable_weights=False"):
        mutable(x)
    with torch.no_grad():
        compiled = _compile(model, [x], regions=rules)
        with pytest.raises(
            RegionError,
            match="A model compiled with regions cannot be refitted; compile it "
            "again with the new weights$",
        ):
            torch_tensorrt.dynamo.refit_module_weights(
                compiled, torch.export.export(model, (x,)), arg_inputs=[x]
            )
        # The refused refit left the compiled model as it was.
        _assert_compiled_close(compiled(x), model(x))


@requires_cuda
@pytest.mark.parametrize("selector", ["scale", Scale], ids=["path", "class"])
def test_saved_program_without_region_code_compiles_with_rules(tmp_path, selector):
    x = torch.randn(2, 8, device="cuda")
    model = Sandwich().cuda().eval()
    path = tmp_path / "plain.pt2"
    torch.export.save(torch.export.export(model, (x,)), path)
    program = torch.export.load(path)
    assert not has_region_markers(program.graph_module)
    rules = {
        "act": ExecuteInTorch(),
        selector: FusedOp(_scale_shift, inputs=[0, "weight", "shift"]),
    }
    with torch.no_grad():
        compiled = _compile(program, [x], regions=rules)
        key = "ttrt_region_rules_test::scale_shift.default"
        assert _summary(compiled) == [
            ("act#0", "eager", "torch", ""),
            ("scale#1", "eager", "torch", key),
        ]
        _assert_outside_engines(compiled, get_region_records(compiled))
        # The engines hold what runs before, between and after the two modules.
        assert len(_engines(compiled)) == 3
        _calls.clear()
        _assert_compiled_close(compiled(x), model(x))
        assert _calls == ["scale_shift"]


class Recurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(8, 8, batch_first=True)

    def forward(self, x):
        output, _ = self.rnn(x + 1)
        return output * 2


@requires_cuda
@pytest.mark.parametrize("selector", ["rnn", torch.nn.GRU], ids=["path", "class"])
def test_program_loaded_from_disk_compiles_a_recurrent_module(tmp_path, selector):
    x = torch.randn(2, 5, 8, device="cuda")
    model = Recurrent().cuda().eval()
    path = tmp_path / "recurrent.pt2"
    torch.export.save(torch.export.export(model, (x,)), path)
    program = torch.export.load(path)
    # On the GPU the layer's weights are parts of one block of memory, which
    # the loaded program no longer shows.
    weights = [model.rnn.weight_ih_l0, model.rnn.weight_hh_l0]
    assert torch._C._is_alias_of(*weights)
    with torch.no_grad():
        compiled = _compile(program, [x], regions={selector: ExecuteInTorch()})
        (record,) = get_region_records(compiled)
        assert record.id == "rnn#0"
        _assert_outside_engines(compiled, [record])
        assert "gru" in _compiled_body(compiled, record)
        assert len(_engines(compiled)) == 2
        _assert_compiled_close(compiled(x), model(x))


@requires_cuda
def test_saved_program_compiles_with_rules_in_a_fresh_process(tmp_path):
    x = torch.linspace(-2, 2, 16, device="cuda").reshape(2, 8)
    model = Sandwich().cuda().eval()
    path = tmp_path / "plain.pt2"
    torch.export.save(torch.export.export(model, (x,)), path)
    expected = tmp_path / "expected.pt"
    with torch.no_grad():
        torch.save(model(x).cpu(), expected)
    child = textwrap.dedent("""
        import sys

        import torch
        import torch_tensorrt
        from torch_tensorrt import region
        from torch_tensorrt.dynamo.region_compilation import get_region_records
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        calls = []


        @torch.library.custom_op("fresh_process::scale_shift", mutates_args=())
        def scale_shift(
            x: torch.Tensor, weight: torch.Tensor, shift: torch.Tensor
        ) -> torch.Tensor:
            calls.append("scale_shift")
            return x * weight + shift


        @scale_shift.register_fake
        def scale_shift_fake(x, weight, shift):
            return torch.empty_like(x)


        program = torch.export.load(sys.argv[1])
        x = torch.linspace(-2, 2, 16, device="cuda").reshape(2, 8)
        rules = {
            "act": region.ExecuteInTorch(),
            "scale": region.FusedOp(scale_shift, inputs=[0, "weight", "shift"]),
        }
        with torch.no_grad():
            compiled = torch_tensorrt.dynamo.compile(
                program, inputs=[x], min_block_size=1, regions=rules
            )
            records = get_region_records(compiled)
            assert [(record.id, record.operator_key) for record in records] == [
                ("act#0", ""),
                ("scale#1", "fresh_process::scale_shift.default"),
            ], records
            assert not any(r.owner.startswith("_run_on_acc") for r in records)
            engines = [
                m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)
            ]
            assert len(engines) == 3, engines
            calls.clear()
            torch.testing.assert_close(
                compiled(x).cpu(), torch.load(sys.argv[2]), rtol=1e-4, atol=1e-4
            )
            assert calls == ["scale_shift"], calls
        print("fresh-process-rules-ok")
        """)
    assert "fresh-process-rules-ok" in run_python("-c", child, path, expected)


class Attention(torch.nn.Module):
    def __init__(self, width, heads=2):
        super().__init__()
        self.heads = heads
        self.inputs = torch.nn.Linear(width, 3 * width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = (
            part.reshape(batch, length, self.heads, -1).transpose(1, 2)
            for part in self.inputs(x).chunk(3, dim=-1)
        )
        mixed = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        return self.output(mixed.transpose(1, 2).reshape(batch, length, width))


class FeedForward(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = torch.nn.Linear(width, 2 * width, bias=False)
        self.down = torch.nn.Linear(2 * width, width, bias=False)

    def forward(self, hidden, residual):
        return residual + self.down(torch.nn.functional.silu(self.up(hidden)))


class DecoderBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = Attention(width)
        self.mlp_norm = RMSNorm(width)
        self.mlp = FeedForward(width)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return self.mlp(residual=x, hidden=self.mlp_norm(x))


class Decoder(torch.nn.Module):
    """A small decoder: embedding, blocks, final norm and a tied projection."""

    def __init__(self, vocabulary=32, width=16, layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocabulary, width)
        self.layers = torch.nn.ModuleList(DecoderBlock(width) for _ in range(layers))
        self.norm = RMSNorm(width)
        self.head = torch.nn.Linear(width, vocabulary, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


@requires_cuda
@pytest.mark.parametrize(
    "source",
    [
        "model",
        "strict model",
        "program",
        pytest.param("model with a kernel", marks=requires_cuda_qdp),
    ],
)
def test_decoder_compiles_with_a_mix_of_rules(source):
    torch.manual_seed(0)
    model = Decoder().cuda().eval()
    ids = torch.randint(0, 32, (2, 6), device="cuda")
    rules = {
        "layers.*.attention_norm": FusedOp(_rms_norm, inputs=[0, "weight"]),
        "layers.*.mlp_norm": ExecuteInTorch(),
        # The block calls its MLP with keywords: residual first, then hidden.
        FeedForward: FusedOp(_feed_forward, inputs=[1, 0, "up.weight", "down.weight"]),
        "norm": ExecuteInTorch(name="final_norm"),
        "head": FusedOp(_project, inputs=[0, "weight"]),
    }
    if source == "model with a kernel":
        rules["norm"] = FusedKernel(
            lambda: _rms_norm_kernel,
            inputs=[0, "weight"],
            name="final_norm",
            constexprs={"BLOCK": 16, "EPS": EPS},
            grid=_row_grid,
            extra_args_fn=_row_extras,
        )
    with torch.no_grad():
        if source == "program":
            program = torch.export.export(
                model,
                (ids,),
                preserve_module_call_signature=("layers.0.mlp", "layers.1.mlp"),
            )
            compiled = torch_tensorrt.dynamo.compile(
                program, inputs=[ids], min_block_size=1, regions=rules
            )
        else:
            compiled = _compile(
                model, [ids], strict=source == "strict model", regions=rules
            )
        records = get_region_records(compiled)
        assert [record.id for record in records] == [
            "layers.0.attention_norm#0",
            "layers.0.mlp_norm#1",
            "layers.0.mlp#2",
            "layers.1.attention_norm#3",
            "layers.1.mlp_norm#4",
            "layers.1.mlp#5",
            "final_norm#6",
            "head#7",
        ]
        operators = [record.operator_key.split("::")[-1] for record in records]
        assert operators == [
            "rms_norm.default",
            "",
            "feed_forward.default",
            "rms_norm.default",
            "",
            "feed_forward.default",
            "",
            "project.default",
        ]
        kernels = [record for record in records if record.route == "plugin"]
        assert len(kernels) == (source == "model with a kernel")
        _assert_inside_engines(compiled, kernels)
        _assert_outside_engines(
            compiled, [record for record in records if record.route == "torch"]
        )
        # The attention and the embedding stay in TensorRT.
        assert len(_engines(compiled)) > 1
        _calls.clear()
        torch.testing.assert_close(compiled(ids), model(ids), rtol=1e-3, atol=5e-3)
        assert _calls == [
            "rms_norm",
            "feed_forward",
            "rms_norm",
            "feed_forward",
            "project",
        ]
