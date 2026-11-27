# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Scopes written in forward are honored on every ahead-of-time entry path.

Entry points that cannot honor regions say so, for scopes and for rules.
"""

import io
import json
import platform
import re
from types import SimpleNamespace

import pytest
import torch
import torch_tensorrt
from torch.utils._triton import has_triton
from torch_tensorrt import region
from torch_tensorrt.dynamo._compiler import _copy_program_graph
from torch_tensorrt.dynamo.lowering._buffer_lifting import erase_export_guards
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    check_decomposed_regions,
    get_region_records,
    has_region_markers,
    materialize_regions,
    normalize_region_scopes,
    region_hop_records,
)
from torch_tensorrt.dynamo.region_compilation._types import REGION_ID_META_KEY
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region import RegionCaptureError, RegionError

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
requires_cuda_inductor = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="CUDA Inductor requires Triton support on the device",
)
requires_triton = pytest.mark.skipif(
    triton is None, reason="Triton and TensorRT QDP plugin support are required"
)
requires_cuda_qdp = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="CUDA and TensorRT QDP plugin support are required",
)
BLOCK = 128
_OPTIONS = dict(min_block_size=1)
_PLUS_HUNDRED = "ttrt_entry_test::plus_hundred.default"
_calls = []
# One entry per run of CountsForwardRuns.forward: tracing a model runs it.
_forward_runs = []


# The operators and kernels deliberately differ from the scope bodies they
# replace, so a result shows which of the two ran.
@torch.library.custom_op("ttrt_entry_test::plus_hundred", mutates_args=())
def _plus_hundred(x: torch.Tensor) -> torch.Tensor:
    _calls.append("plus_hundred")
    return x + 100


@_plus_hundred.register_fake
def _plus_hundred_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_entry_test::weighted_difference", mutates_args=())
def _weighted_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    _calls.append("weighted_difference")
    return first - 2 * second


@_weighted_difference.register_fake
def _weighted_difference_fake(first, second):
    return torch.empty_like(first)


# The fresh process registers this name with two inputs.
@torch.library.custom_op("ttrt_entry_test::one_input_here", mutates_args=())
def _one_input_here(x: torch.Tensor) -> torch.Tensor:
    return x + 100


@_one_input_here.register_fake
def _one_input_here_fake(x):
    return torch.empty_like(x)


if triton is not None:

    @triton.jit
    def _plus_seven_kernel(x, n, y, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < n, other=0.0)
        tl.store(y + offsets, values + 7.0, offsets < n)

    @triton.jit
    def _minus_seven_kernel(x, n, y, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < n, other=0.0)
        tl.store(y + offsets, values - 7.0, offsets < n)


def _grid(inputs, outputs):
    return (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),)


def _extras(inputs, outputs):
    return [trtp.SymInt32(inputs[0].shape_expr.numel())]


def _plus_seven():
    return _plus_seven_kernel


def _unexpected_kernel():
    raise AssertionError("A trace that ignores regions resolved the kernel")


class Annotated(torch.nn.Module):
    """TensorRT work around a scope kept in PyTorch and a scope an operator replaces."""

    def __init__(self, backend="eager", op=_plus_hundred):
        super().__init__()
        self.backend = backend
        self.op = op

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="kept", backend=self.backend):
            x = torch.tanh(x) * 3
        with region.fused_op(self.op, name="replaced"):
            x = torch.sigmoid(x)
        return x * 2

    @staticmethod
    def with_regions(x, shift=100):
        """What the model computes once the operator replaces its scope."""
        return (torch.tanh(x + 1) * 3 + shift) * 2


class KernelScope(torch.nn.Module):
    def __init__(self, kernel=_plus_seven):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        x = x * 2
        with region.fused_kernel(
            kernel=self.kernel,
            name="shifted",
            constexprs={"BLOCK": BLOCK},
            grid=_grid,
            extra_args_fn=_extras,
        ):
            x = torch.relu(x)
        return x + 1

    @staticmethod
    def with_regions(x, shift=7):
        return x * 2 + shift + 1


class CastThenReuse(torch.nn.Module):
    """In float32 the cast returns its input, which the model reads again."""

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="cast"):
            x = x.float()
            y = torch.tanh(x)
        return y + x


class EveryScope(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="kept"):
            x = torch.tanh(x) * 3
        with region.fused_op(_plus_hundred, name="replaced"):
            x = torch.sigmoid(x)
        with region.fused_kernel(kernel=_unexpected_kernel, name="shifted"):
            x = torch.relu(x - 0.5)
        return x * 2


class Unannotated(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        return self.activation(x + 1) * 2


class CountsForwardRuns(Unannotated):
    def forward(self, x):
        _forward_runs.append(type(self).__name__)
        return super().forward(x)


class DropoutOutside(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x + 1)
        with region.execute_in_torch(name="kept"):
            x = torch.tanh(x) * 3
        return x * 2


class ScopeAfterLinear(torch.nn.Module):
    """Weights in TensorRT, then a scope an operator replaces."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        x = self.linear(x)
        with region.fused_op(_plus_hundred, name="replaced"):
            x = torch.tanh(x)
        return x * 2

    def with_regions(self, x):
        return (self.linear(x) + 100) * 2


def _dynamo_compile(program, x, **options):
    return torch_tensorrt.dynamo.compile(program, inputs=[x], **_OPTIONS, **options)


def _compile_program(program, x, **options):
    return torch_tensorrt.compile(
        program, ir="dynamo", inputs=[x], **_OPTIONS, **options
    )


def _export_then(compile_program, strict):
    def entry(model, x):
        program = torch.export.export(model, (x,), strict=strict)
        return compile_program(program, x)

    return entry


def _trace_then(compile_program):
    def entry(model, x):
        return compile_program(torch_tensorrt.dynamo.trace(model, inputs=[x]), x)

    return entry


def _compile_model(**options):
    def entry(model, x):
        return torch_tensorrt.compile(
            model, ir="dynamo", inputs=[x], **options, **_OPTIONS
        )

    return entry


_ENTRY_PATHS = {
    "export_strict-dynamo_compile": _export_then(_dynamo_compile, True),
    "export_nonstrict-dynamo_compile": _export_then(_dynamo_compile, False),
    "export_strict-compile": _export_then(_compile_program, True),
    "export_nonstrict-compile": _export_then(_compile_program, False),
    "dynamo_trace-dynamo_compile": _trace_then(_dynamo_compile),
    "dynamo_trace-compile": _trace_then(_compile_program),
    "compile_model": _compile_model(),
    "compile_model_nonstrict": _compile_model(strict=False),
    "compile_model_strict": _compile_model(strict=True),
}
entry_paths = pytest.mark.parametrize("entry", list(_ENTRY_PATHS))
strict_modes = pytest.mark.parametrize(
    "strict", [False, True], ids=["nonstrict", "strict"]
)


def _assert_close(actual, expected):
    # TensorRT and Inductor may round elementwise math unlike eager PyTorch.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def _assert_roughly_close(actual, expected):
    # TensorRT may round a matrix product more coarsely than PyTorch does.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def _engines(compiled):
    return [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]


def _region_modules(compiled):
    """The PyTorch module each region runs as, and its path, by region id."""
    return {
        module.meta[REGION_ID_META_KEY]: (path, module)
        for path, module in compiled.named_modules()
        if REGION_ID_META_KEY in getattr(module, "meta", {})
    }


def _placement(compiled):
    return [
        (r.id, r.route, r.backend, r.owner, r.plugin_key, r.operator_key)
        for r in get_region_records(compiled)
    ]


def _targets(graph):
    return [
        node.target
        for child in graph.modules()
        if isinstance(child, torch.fx.GraphModule)
        for node in child.graph.nodes
        if node.op == "call_function"
    ]


def _marker_count(program):
    return sum("torch_tensorrt_region" in str(target) for target in _targets(program))


def _has_capture_leftovers(program):
    return any("hints_wrapper" in str(target) for target in _targets(program))


def _materialize(program):
    """Capture a traced program's regions as compilation does, without TensorRT.

    No session is open, and the caller's program stays as it was traced.
    """
    program = normalize_region_scopes(_copy_program_graph(program))
    captured = program.graph_module
    program = program.run_decompositions()
    check_decomposed_regions(captured, program.graph_module)
    graph = materialize_regions(program.module(), captured)
    erase_export_guards(graph)
    graph.graph.lint()
    graph.recompile()
    return graph, captured


def _scope_summary(captured):
    """What capture kept of each scope: its identity, policy and body."""
    summary = []
    for record in region_hop_records(captured):
        body = captured.get_submodule(record.child_target)
        output = next(node for node in body.graph.nodes if node.op == "output")
        summary.append(
            (
                record.id,
                record.name,
                record.backend,
                record.route,
                record.operator_key,
                [str(n.target) for n in body.graph.nodes if n.op == "call_function"],
                sum(node.op == "placeholder" for node in body.graph.nodes),
                len(output.args[0]),
            )
        )
    return summary


def _save_and_load_program(program):
    artifact = io.BytesIO()
    torch.export.save(program, artifact)
    artifact.seek(0)
    return torch.export.load(artifact)


@requires_cuda
@entry_paths
@pytest.mark.parametrize(
    "backend", ["eager", pytest.param("inductor", marks=requires_cuda_inductor)]
)
def test_scopes_in_forward_are_honored(entry, backend, observed_inductor):
    model = Annotated(backend).eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    _calls.clear()
    with torch.no_grad():
        plain = model(x)
    assert not _calls
    compiled = _ENTRY_PATHS[entry](model, x)

    records = get_region_records(compiled)
    assert [(r.id, r.route, r.backend, r.operator_key) for r in records] == [
        ("kept#0", "torch", backend, ""),
        ("replaced#1", "torch", "eager", _PLUS_HUNDRED),
    ]
    audit_region_placement(compiled, records)
    # Both regions sit between two engines, in modules PyTorch runs.
    calls = [str(n.target) for n in compiled.graph.nodes if n.op == "call_module"]
    assert [
        isinstance(compiled.get_submodule(target), TorchTensorRTModule)
        for target in calls
    ] == [True, False, True]
    modules = _region_modules(compiled)
    assert sorted(modules) == ["kept#0", "replaced#1"]
    for record in records:
        path, module = modules[record.id]
        assert isinstance(module, torch.fx.GraphModule)
        assert path == f"{record.owner}.{record.child_target}"
        assert path.startswith(calls[1] + ".")
    kept, replaced = modules["kept#0"][1], modules["replaced#1"][1]
    assert torch.ops.aten.tanh.default in _targets(kept)
    assert _targets(replaced) == [torch.ops.ttrt_entry_test.plus_hundred.default]

    seen = []
    if backend == "eager":
        kept.register_forward_hook(lambda _, args, output: seen.append((args, output)))
    with torch.no_grad():
        actual = compiled(x)
    _assert_close(actual, Annotated.with_regions(x))
    assert not torch.allclose(actual, plain, atol=1)
    assert _calls == ["plus_hundred"]
    if backend == "eager":
        ((inputs, outputs),) = seen
        torch.testing.assert_close(outputs[0], torch.tanh(inputs[0]) * 3)
        assert not observed_inductor.requests
    else:
        assert len(observed_inductor.requests) == 1
        assert observed_inductor.counter.frame_count == 1
        assert observed_inductor.executions == 1


@requires_triton
@requires_cuda_qdp
@entry_paths
def test_kernel_scope_in_forward_is_honored(entry):
    model = KernelScope().eval().cuda()
    x = torch.randn(4, 16, device="cuda")
    compiled = _ENTRY_PATHS[entry](model, x)

    (record,) = get_region_records(compiled)
    assert (record.id, record.route, record.operator_key) == ("shifted#0", "plugin", "")
    # The kernel is recorded by a random key, not by a position in a table.
    assert isinstance(record.plugin_key, str) and len(record.plugin_key) == 32
    audit_region_placement(compiled, [record], require_plugins_in_trt=True)
    (engine,) = _engines(compiled)
    assert compiled.get_submodule(record.owner) is engine
    assert engine._torch_tensorrt_plugin_regions == (record.id,)
    assert str(record.plugin_op) in engine.get_layer_info()
    assert not _region_modules(compiled)
    with torch.no_grad():
        _assert_close(compiled(x), KernelScope.with_regions(x))
        assert not torch.allclose(model(x), KernelScope.with_regions(x), atol=1)


class TwoScopes(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="first"):
            y = torch.relu(x) + 1
        z = y * 2
        with region.execute_in_torch(name="second", backend="inductor"):
            z = torch.sigmoid(z) - y
        return z + 1


class ReadsParameters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.register_buffer("offset", torch.randn(8))

    def forward(self, x):
        x = x * 2
        with region.execute_in_torch(name="affine"):
            y = self.linear(x) + self.offset
        return y.tanh()


class CastInScope(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="cast"):
            y = x.float()
            z = torch.relu(y.double()).float() + y
        return z + x


class ListedInputs(torch.nn.Module):
    def forward(self, first, second):
        # The body reads ``second`` before ``first``.
        with region.fused_op(
            _weighted_difference, inputs=(first, second), name="weighted"
        ):
            result = second * -2 + first
        return result.cos()


class ReturnInsideScope(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="tail"):
            return torch.relu(x), x.sum(dim=-1)


class EmptyScope(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="empty"):
            pass
        return x + 1


class DeadScope(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="dead"):
            torch.relu(x)
        return x + 1


class NestedScopes(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="outer"):
            x = x + 1
            with region.execute_in_torch(name="inner"):
                x = torch.relu(x)
        return x


class ScopeAroundBranch(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="branch"):
            y = torch.cond(x.sum() > 0, torch.sin, torch.cos, (x,))
        return y + 1


@pytest.mark.parametrize(
    "factory, input_count, names",
    [
        (TwoScopes, 1, ["first", "second"]),
        (ReadsParameters, 1, ["affine"]),
        (CastInScope, 1, ["cast"]),
        (ListedInputs, 2, ["weighted"]),
        (ReturnInsideScope, 1, ["tail"]),
        (EmptyScope, 1, []),
        (DeadScope, 1, []),
    ],
)
def test_strict_nonstrict_and_saved_programs_capture_alike(factory, input_count, names):
    model = factory().eval()
    inputs = tuple(torch.randn(2, 8) for _ in range(input_count))
    expected = model(*inputs)
    summaries = []
    for strict in (True, False):
        program = torch.export.export(model, inputs, strict=strict)
        assert has_region_markers(program.graph_module)
        # A saved program keeps what capture needs, in either mode.
        for candidate in (program, _save_and_load_program(program)):
            graph, captured = _materialize(candidate)
            summaries.append(_scope_summary(captured))
            _calls.clear()
            torch.testing.assert_close(graph(*inputs), expected)
            assert _calls == ["weighted_difference"] * (factory is ListedInputs)
            assert [r.name for r in get_region_records(graph)] == names
    assert [scope[1] for scope in summaries[0]] == names
    assert all(summary == summaries[0] for summary in summaries)


@strict_modes
@pytest.mark.parametrize(
    "factory, message",
    [
        (NestedScopes, "Nested region scopes are unsupported"),
        (ScopeAroundBranch, "Higher-order operator 'cond' inside a region"),
    ],
)
def test_unsupported_scopes_are_rejected_in_both_modes(factory, message, strict):
    program = torch.export.export(factory().eval(), (torch.randn(2, 8),), strict=strict)
    with pytest.raises(RegionCaptureError, match=message):
        normalize_region_scopes(program)


class SharedStorageInputs(torch.nn.Module):
    """The scope reads the two halves of one tensor, which share its storage."""

    def __init__(self, split="chunk"):
        super().__init__()
        self.split = split

    def forward(self, x):
        first, second = getattr(x, self.split)(2)
        with region.execute_in_torch(name="halves"):
            result = second * -2 + first
        return result.cos()


class ReturnsView(torch.nn.Module):
    def __init__(self, view="view"):
        super().__init__()
        self.view = view

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="view"):
            y = getattr(x, self.view)(4, 4)
        return y * 2


def _exported_and_loaded(*views):
    """Programs as exported and as loaded from a file, for some view operators.

    A loaded program no longer says which values share storage. Capture works
    it out again, whichever view operator the program holds.
    """
    return pytest.mark.parametrize(
        "view, saved",
        [
            pytest.param(view, saved, id=f"{view}-{label}")
            for view in views
            for saved, label in [(False, "exported"), (True, "loaded")]
        ],
    )


@_exported_and_loaded("view", "reshape")
@strict_modes
def test_scope_returning_a_view_of_its_input_is_explained(strict, view, saved):
    program = torch.export.export(
        ReturnsView(view).eval(), (torch.randn(2, 8),), strict=strict
    )
    if saved:
        program = _save_and_load_program(program)
    with pytest.raises(
        RegionCaptureError, match="outputs must not alias region inputs; clone"
    ):
        _materialize(program)


class RecurrentScope(torch.nn.Module):
    """On CUDA the weights of an RNN are views of one flat tensor."""

    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.GRU(8, 8, batch_first=True)

    def forward(self, x):
        x = x * 2
        with region.execute_in_torch(name="recurrent"):
            y, _ = self.recurrent(x)
        return y + 1


class RecurrentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.GRU(8, 8, batch_first=True)

    def forward(self, x):
        y, _ = self.recurrent(x * 2)
        return y + 1


class Halves(torch.nn.Module):
    def forward(self, x):
        return x.chunk(2)


class HalvesModule(torch.nn.Module):
    """A rule selects a module that returns two views of its input."""

    def __init__(self):
        super().__init__()
        self.halves = Halves()

    def forward(self, x):
        first, second = self.halves(x * 2)
        return (second * -2 + first).cos()


# Model, rules, input shape, and an operation of the region.
_SHARED_STORAGE = {
    "scope_reads_chunks": (SharedStorageInputs, None, (4, 8), "aten.mul"),
    "scope_around_rnn": (RecurrentScope, None, (2, 4, 8), "aten.gru"),
    "rule_selects_rnn": (
        RecurrentModule,
        {"recurrent": region.ExecuteInTorch()},
        (2, 4, 8),
        "aten.gru",
    ),
    "rule_selects_views": (
        HalvesModule,
        {"halves": region.ExecuteInTorch()},
        (4, 8),
        "aten.chunk",
    ),
}


@requires_cuda
@strict_modes
@pytest.mark.parametrize("case", list(_SHARED_STORAGE))
def test_loaded_program_with_shared_storage_compiles_as_exported(case, strict):
    factory, rules, shape, operation = _SHARED_STORAGE[case]
    model, x = factory().eval().cuda(), torch.randn(*shape, device="cuda")
    if "rnn" in case:
        input_weight, hidden_weight, *_ = model.parameters()
        assert torch._C._is_alias_of(input_weight, hidden_weight)
    program = torch.export.export(model, (x,), strict=strict)
    with torch.no_grad():
        expected = model(x)
    placements = []
    for candidate in (program, _save_and_load_program(program)):
        compiled = _dynamo_compile(candidate, x, regions=rules)
        placements.append(_placement(compiled))
        # The region runs in PyTorch, between two TensorRT engines.
        ((_, body),) = _region_modules(compiled).values()
        assert any(operation in str(target) for target in _targets(body))
        assert len(_engines(compiled)) == 2
        with torch.no_grad():
            _assert_close(compiled(x), expected)
    exported, loaded = placements
    assert len(exported) == 1 and loaded == exported


_FRESH_PROCESS = """
import json
import sys

import torch
import torch_tensorrt

# The programs load with nothing but the import above.
directory, names = sys.argv[1], sys.argv[2:]
programs = {name: torch.export.load(f"{directory}/{name}.pt2") for name in names}

from torch_tensorrt.dynamo.region_compilation import (
    get_region_records,
    has_region_markers,
)

x = torch.load(f"{directory}/input.pt", weights_only=True).cuda()
calls = []


def compile_program(name):
    try:
        compiled = torch_tensorrt.dynamo.compile(
            programs[name], inputs=[x], min_block_size=1
        )
        calls.clear()
        with torch.no_grad():
            output = compiled(x)
    except Exception as error:
        return {
            "error": [base.__name__ for base in type(error).__mro__],
            "message": str(error),
        }
    return {
        "records": [
            [r.id, r.route, r.owner, r.operator_key]
            for r in get_region_records(compiled)
        ],
        "output": output.cpu().tolist(),
        "operator_calls": len(calls),
    }


report = {
    "markers": {
        name: has_region_markers(program.graph_module)
        for name, program in programs.items()
    },
    "unregistered": {name: compile_program(name) for name in names},
}


@torch.library.custom_op("ttrt_entry_test::plus_hundred", mutates_args=())
def plus_hundred(x: torch.Tensor) -> torch.Tensor:
    calls.append("plus_hundred")
    return x + 100


@plus_hundred.register_fake
def plus_hundred_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_entry_test::one_input_here", mutates_args=())
def two_inputs_there(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@two_inputs_there.register_fake
def two_inputs_there_fake(x, y):
    return torch.empty_like(x)


report["registered"] = {
    name: compile_program(name)
    for name in names
    if name.startswith(("scopes", "changed"))
}
print("REPORT" + json.dumps(report))
"""


@pytest.fixture(scope="module")
def fresh_process(tmp_path_factory):
    """Save annotated programs, then compile them in a process that starts clean."""
    directory = tmp_path_factory.mktemp("annotated_programs")
    x = torch.randn(4, 16, device="cuda")
    torch.save(x.cpu(), directory / "input.pt")
    programs = {
        "cast_strict": (CastThenReuse(), True),
        "cast_nonstrict": (CastThenReuse(), False),
        "scopes_strict": (Annotated(), True),
        "scopes_nonstrict": (Annotated(), False),
        "changed": (Annotated(op=_one_input_here), True),
    }
    if triton is not None and torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
        programs["kernel"] = (KernelScope(), True)
    for name, (model, strict) in programs.items():
        program = torch.export.export(model.eval().cuda(), (x,), strict=strict)
        torch.export.save(program, directory / f"{name}.pt2")
    output = run_python("-c", _FRESH_PROCESS, directory, *programs, timeout=300)
    (line,) = (line for line in output.splitlines() if line.startswith("REPORT"))
    return SimpleNamespace(
        x=x, names=list(programs), **json.loads(line.removeprefix("REPORT"))
    )


def _assert_honored_elsewhere(result, expected, operator_keys):
    assert "error" not in result, result
    assert [(route, key) for _, route, _, key in result["records"]] == [
        ("torch", key) for key in operator_keys
    ]
    assert not any(
        owner.startswith("_run_on_acc") for _, _, owner, _ in result["records"]
    )
    _assert_close(torch.tensor(result["output"]), expected.cpu())


@requires_cuda
def test_saved_programs_load_with_only_torch_tensorrt_imported(fresh_process):
    assert fresh_process.markers == dict.fromkeys(fresh_process.names, True)


@requires_cuda
@pytest.mark.parametrize("mode", ["strict", "nonstrict"])
def test_fresh_process_keeps_a_saved_scope_in_pytorch(fresh_process, mode):
    # The scope casts in float32, which used to fail once the program was loaded.
    result = fresh_process.unregistered[f"cast_{mode}"]
    with torch.no_grad():
        expected = CastThenReuse()(fresh_process.x)
    _assert_honored_elsewhere(result, expected, [""])
    assert result["records"][0][0] == "cast#0"


@requires_cuda
@pytest.mark.parametrize("mode", ["strict", "nonstrict"])
def test_fresh_process_uses_the_operator_registered_there(fresh_process, mode):
    missing = fresh_process.unregistered[f"scopes_{mode}"]
    assert "RegionError" in missing["error"]
    assert (
        f"fused_op region 'replaced#1' uses the operator {_PLUS_HUNDRED!r}, which "
        "is not registered in this process" in missing["message"]
    )
    result = fresh_process.registered[f"scopes_{mode}"]
    expected = Annotated.with_regions(fresh_process.x)
    _assert_honored_elsewhere(result, expected, ["", _PLUS_HUNDRED])
    assert result["operator_calls"] == 1


@requires_cuda
def test_fresh_process_checks_the_operator_registered_there(fresh_process):
    result = fresh_process.registered["changed"]
    assert "RegionCaptureError" in result["error"]
    assert "fused_op region 'replaced#1' input count does not match" in (
        result["message"]
    )


@requires_triton
@requires_cuda_qdp
def test_fresh_process_explains_that_the_kernel_was_not_saved(fresh_process):
    result = fresh_process.unregistered["kernel"]
    assert "RegionError" in result["error"]
    assert "fused_kernel region 'shifted#0' has no kernel in this process" in (
        result["message"]
    )
    assert "saving an exported program does not save the kernel" in result["message"]


def _normalize(program, x):
    return normalize_region_scopes(program)


@strict_modes
@pytest.mark.parametrize(
    "compile_program",
    [
        _normalize,
        pytest.param(_dynamo_compile, marks=requires_cuda),
        pytest.param(_compile_program, marks=requires_cuda),
    ],
    ids=["capture", "dynamo_compile", "compile"],
)
def test_decomposed_program_names_run_decompositions(compile_program, strict):
    device = "cpu" if compile_program is _normalize else "cuda"
    model, x = Annotated().eval().to(device), torch.randn(4, 16, device=device)
    program = torch.export.export(model, (x,), strict=strict).run_decompositions()
    with pytest.raises(
        RegionCaptureError, match=r"run_decompositions\(\) rewrote the region markers"
    ) as caught:
        compile_program(program, x)
    assert "membership survived" not in str(caught.value)
    assert str(caught.value).endswith(
        "Compile the program as torch.export produced it, before run_decompositions()"
    )


def _engine_from_program(model, x, **options):
    return torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        torch.export.export(model, (x,)), inputs=[x], **_OPTIONS, **options
    )


def _engine_from_model(model, x, **options):
    return torch_tensorrt.convert_method_to_trt_engine(
        model, inputs=[x], ir="dynamo", **_OPTIONS, **options
    )


def _model_with_regions(regions):
    """An annotated model, or a plain model and the rules that select a region."""
    if regions == "scopes":
        return Annotated().eval().cuda(), {}
    rules = {"activation": region.ExecuteInTorch()}
    return CountsForwardRuns().eval().cuda(), dict(regions=rules)


def _refusal(regions, path):
    """How a call that cannot honor regions refuses scopes, and how it refuses rules."""
    if regions == "scopes":
        return pytest.raises(
            RegionError, match=f"^Explicit regions do not support {path}$"
        )
    # The call takes no regions= option at all.
    return pytest.raises(TypeError, match="unexpected keyword argument 'regions'")


@requires_cuda
@pytest.mark.parametrize("regions", ["scopes", "rules"])
@pytest.mark.parametrize(
    "convert", [_engine_from_program, _engine_from_model], ids=["program", "model"]
)
def test_engine_only_output_rejects_regions(convert, regions):
    model, options = _model_with_regions(regions)
    x = torch.randn(4, 16, device="cuda")
    _forward_runs.clear()
    with _refusal(regions, "serialized-engine-only output"):
        convert(model, x, **options)
    if regions == "rules":
        if convert is _engine_from_model:
            # The rules are rejected before the model is traced.
            assert not _forward_runs
        # The same model converts once nothing asks for a region.
        _forward_runs.clear()
        assert isinstance(convert(model, x), bytes)
        assert _forward_runs


@pytest.fixture
def cross_compilation(monkeypatch):
    """Open the cross-compilation entry points in a build without the feature.

    They reject regions before anything needs the feature.
    """
    features = torch_tensorrt._features
    monkeypatch.setattr(
        features,
        "ENABLED_FEATURES",
        features.ENABLED_FEATURES._replace(windows_cross_compile=True),
    )


@requires_cuda
@pytest.mark.skipif(
    platform.system() != "Linux" or platform.architecture()[0] != "64bit",
    reason="Cross-compilation for Windows starts from 64-bit Linux",
)
@pytest.mark.parametrize("regions", ["scopes", "rules"])
@pytest.mark.parametrize("entry", ["program", "model"])
def test_cross_compilation_rejects_regions(entry, regions, cross_compilation, tmp_path):
    model, options = _model_with_regions(regions)
    x = torch.randn(4, 16, device="cuda")
    destination = tmp_path / "cross_compiled.ep"
    program = torch.export.export(model, (x,)) if entry == "program" else None
    _forward_runs.clear()
    with _refusal(regions, "cross-compilation"):
        if entry == "model":
            torch_tensorrt.cross_compile_for_windows(
                model, str(destination), inputs=[x], **_OPTIONS, **options
            )
        else:
            torch_tensorrt.dynamo.cross_compile_for_windows(
                program, inputs=[x], **_OPTIONS, **options
            )
    if regions == "rules":
        # The rules are rejected before the model is traced.
        assert not _forward_runs
    assert not destination.exists()


@requires_cuda
def test_model_in_training_mode_is_rejected_before_rules_trace_it():
    model, options = _model_with_regions("rules")
    x = torch.randn(4, 16, device="cuda")
    model.train()
    _forward_runs.clear()
    with pytest.raises(
        RegionCaptureError,
        match=r"^Regions currently require model\.eval\(\); still in training "
        r"mode: <the model>, activation$",
    ):
        _compile_model(**options)(model, x)
    assert not _forward_runs
    # Without rules the model has no region, and its mode is its own business.
    compiled = _compile_model()(model, x)
    assert _forward_runs and not get_region_records(compiled)


@requires_cuda
@strict_modes
@pytest.mark.parametrize(
    "compile_program", [_dynamo_compile, _compile_program], ids=["dynamo", "compile"]
)
def test_program_exported_in_training_mode_is_rejected(compile_program, strict):
    x = torch.randn(4, 16, device="cuda")
    model = DropoutOutside().train().cuda()
    program = torch.export.export(model, (x,), strict=strict)
    with pytest.raises(
        RegionCaptureError,
        match=r"require a model in eval mode: aten\.dropout.*Call model\.eval\(\)",
    ):
        compile_program(program, x)


@requires_cuda
def test_program_updating_batch_statistics_is_rejected():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(16)

        def forward(self, x):
            x = self.norm(x)
            with region.execute_in_torch(name="kept"):
                x = torch.tanh(x)
            return x * 2

    x = torch.randn(4, 16, device="cuda")
    program = torch.export.export(Model().train().cuda(), (x,))
    with pytest.raises(RegionCaptureError, match="mutation of buffer 'norm.running_"):
        _dynamo_compile(program, x)


@requires_cuda
@pytest.mark.parametrize(
    "compile_program", [_dynamo_compile, _compile_program], ids=["dynamo", "compile"]
)
def test_eval_mode_dropout_is_accepted(compile_program):
    x = torch.randn(4, 16, device="cuda")
    model = DropoutOutside().eval().cuda()
    program = torch.export.export(model, (x,))
    assert torch.ops.aten.dropout.default in _targets(program.graph_module)
    compiled = compile_program(program, x)
    assert [r.id for r in get_region_records(compiled)] == ["kept#0"]
    with torch.no_grad():
        _assert_close(compiled(x), model(x))


def _torch_compile_model(model, x, **options):
    return torch_tensorrt.compile(
        model, ir="torch_compile", inputs=[x], **_OPTIONS, **options
    )


@requires_cuda
def test_training_mode_without_regions_is_not_a_region_error():
    x = torch.randn(4, 16, device="cuda")
    with region.ignore_regions():
        program = torch.export.export(DropoutOutside().train().cuda(), (x,))
    assert not get_region_records(_dynamo_compile(program, x))


@requires_cuda
def test_mutable_module_takes_an_annotated_model_inside_ignore_regions():
    model, x = ScopeAfterLinear().eval().cuda(), torch.randn(4, 16, device="cuda")
    mutable = torch_tensorrt.MutableTorchTensorRTModule(model, **_OPTIONS)
    _calls.clear()
    with region.ignore_regions(), torch.no_grad():
        first = mutable(x)
        _assert_roughly_close(first, model(x))
    assert not get_region_records(mutable.gm)
    assert len(_engines(mutable.gm)) == 1
    # Later calls and weight updates need no block: the module keeps the
    # program it traced without regions.
    with torch.no_grad():
        mutable.linear.weight.mul_(-1)
        second = mutable(x)
        _assert_roughly_close(second, model(x))
    assert not torch.allclose(second, first, atol=0.1)
    assert not get_region_records(mutable.gm)
    assert not _calls


@requires_cuda
@strict_modes
def test_ignore_regions_exports_an_ordinary_model(strict):
    model, x = Annotated().eval().cuda(), torch.randn(4, 16, device="cuda")
    with region.ignore_regions():
        program = torch.export.export(model, (x,), strict=strict)
    assert not _marker_count(program.graph_module)
    compiled = _dynamo_compile(program, x)
    assert not get_region_records(compiled) and not _region_modules(compiled)
    # Nothing stays in PyTorch: the scope bodies are TensorRT layers.
    assert len(_engines(compiled)) == 1
    assert not any(isinstance(m, torch.fx.GraphModule) for m in compiled.children())
    _calls.clear()
    with torch.no_grad():
        _assert_close(compiled(x), model(x))
    assert not _calls


def _trace_ignoring_scopes(kind, model, x, graphs):
    if kind == "torchscript":
        return torch.jit.trace(model, (x,))

    def inspect(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    backend = inspect if kind == "function" else kind
    # fullgraph: an ignored scope must not add a graph break.
    return torch.compile(model, backend=backend, fullgraph=True)


# Stands in for what torch_tensorrt.region reports on a PyTorch that cannot
# record regions.
_NO_REGION_SUPPORT = (
    "Regions (execute_in_torch, fused_op, fused_kernel and regions= rules) "
    "require PyTorch features that this test pretends are missing"
)


@pytest.fixture
def pytorch_without_region_support(monkeypatch):
    monkeypatch.setattr(region._marker_ops, "CAPTURE_ERROR", _NO_REGION_SUPPORT)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class OneScope(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == "execute_in_torch":
            scope = region.execute_in_torch(name="only")
        elif self.kind == "fused_op":
            scope = region.fused_op(_plus_hundred, name="only")
        else:
            scope = region.fused_kernel(kernel=_unexpected_kernel, name="only")
        with scope:
            x = torch.tanh(x)
        return x * 2


@strict_modes
@pytest.mark.parametrize("kind", ["execute_in_torch", "fused_op", "fused_kernel"])
def test_export_reports_missing_pytorch_support_for_every_scope(
    kind, strict, pytorch_without_region_support
):
    model, x = OneScope(kind).eval(), torch.randn(2, 8)
    with pytest.raises(RegionCaptureError) as caught:
        torch.export.export(model, (x,), strict=strict)
    assert str(caught.value) == _NO_REGION_SUPPORT
    # Eager execution needs none of it, and ignore_regions() avoids the error.
    _calls.clear()
    torch.testing.assert_close(model(x), torch.tanh(x) * 2)
    with region.ignore_regions():
        program = torch.export.export(model, (x,), strict=strict)
    assert not _marker_count(program.graph_module)
    torch.testing.assert_close(program.module()(x), torch.tanh(x) * 2)
    assert not _calls


def _torch_compile(backend):
    def entry(model, x, **options):
        compiled = torch.compile(
            model, backend=backend, options=dict(_OPTIONS, **options)
        )
        return compiled(x)

    return entry


def _call_torch_compile_model(model, x, **options):
    return _torch_compile_model(model, x, **options)(x)


_TORCH_COMPILE_PATHS = {
    "torch_compile-torch_tensorrt": _torch_compile("torch_tensorrt"),
    "torch_compile-tensorrt": _torch_compile("tensorrt"),
    "compile_model-torch_compile": _call_torch_compile_model,
}


@requires_cuda
@pytest.mark.parametrize("entry", [*_ENTRY_PATHS, *_TORCH_COMPILE_PATHS])
def test_every_trace_that_records_scopes_reports_missing_pytorch_support(
    entry, pytorch_without_region_support
):
    model, x = Annotated().eval().cuda(), torch.randn(4, 16, device="cuda")
    if entry in _ENTRY_PATHS:
        with pytest.raises(RegionCaptureError) as caught:
            _ENTRY_PATHS[entry](model, x)
        assert str(caught.value) == _NO_REGION_SUPPORT
    else:
        # torch.compile reports an error raised while it traces as its own,
        # and does not run the model without its regions instead.
        with pytest.raises(
            torch._dynamo.exc.InternalTorchDynamoError,
            match=re.escape(f"RegionCaptureError: {_NO_REGION_SUPPORT}"),
        ):
            _TORCH_COMPILE_PATHS[entry](model, x)


@pytest.mark.parametrize("kind", ["eager", "function", "torchscript"])
def test_trace_that_ignores_scopes_needs_no_pytorch_region_support(
    kind, pytorch_without_region_support
):
    model, x = EveryScope().eval(), torch.randn(2, 8)
    _calls.clear()
    traced = _trace_ignoring_scopes(kind, model, x, [])
    with torch.no_grad():
        _assert_close(traced(x), model(x))
    assert not _calls


_PYTORCH_WITHOUT_ORDERED_EFFECTS = """
import json

import torch

# Take away what torch_tensorrt.region registers its markers with.
assert vars(torch.library).pop("_register_effectful_op", None) is not None

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.region import _marker_ops


class Model(torch.nn.Module):
    def forward(self, x):
        with region.execute_in_torch(name="kept"):
            x = torch.tanh(x)
        return x * 2


model, x = Model().eval(), torch.ones(2, 8)
report = {"reason": _marker_ops.CAPTURE_ERROR}
report["eager"] = bool(torch.equal(model(x), torch.tanh(x) * 2))
try:
    torch.export.export(model, (x,))
    report["export"] = None
except Exception as error:
    report["export"] = [type(error).__name__, str(error)]
with region.ignore_regions():
    program = torch.export.export(model, (x,))
report["ignored"] = [str(node.target) for node in program.graph.nodes]
report["ignored_result"] = bool(torch.equal(program.module()(x), torch.tanh(x) * 2))
print("REPORT" + json.dumps(report))
"""


def test_pytorch_without_ordered_effects_explains_what_regions_need():
    output = run_python("-c", _PYTORCH_WITHOUT_ORDERED_EFFECTS, timeout=300)
    (line,) = (line for line in output.splitlines() if line.startswith("REPORT"))
    report = json.loads(line.removeprefix("REPORT"))
    reason = report["reason"]
    assert reason.startswith(
        "Regions (execute_in_torch, fused_op, fused_kernel and regions= rules) "
        "require PyTorch ordered-effect markers and graph-break guards; this "
        "PyTorch version lacks them: "
    )
    assert "_register_effectful_op" in reason
    assert reason.endswith(
        "To trace an annotated model without its regions, use "
        "torch_tensorrt.region.ignore_regions()"
    )
    # Importing and running the model work; only recording a region fails.
    assert report["eager"] is True
    assert report["export"] == ["RegionCaptureError", reason]
    assert "aten.tanh.default" in report["ignored"]
    assert not any("torch_tensorrt_region" in target for target in report["ignored"])
    assert report["ignored_result"] is True


class _Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Annotated()

    def forward(self, x):
        return self.inner(x) - 1


@requires_cuda
@pytest.mark.parametrize(
    "strict, call_signatures",
    [(False, False), (True, False), (False, True)],
    ids=["nonstrict", "strict", "call_signatures"],
)
def test_compile_leaves_the_callers_program_as_exported(strict, call_signatures):
    model, x = _Wrapper().eval().cuda(), torch.randn(4, 16, device="cuda")
    program = torch.export.export(
        model,
        (x,),
        strict=strict,
        preserve_module_call_signature=("inner",) if call_signatures else (),
    )

    def state():
        return (
            str(program.graph_module.graph),
            str(program.graph_signature),
            str(program.module_call_graph),
            sorted(program.state_dict),
            sorted(program.constants),
        )

    exported = state()
    expected = Annotated.with_regions(x) - 1
    compiled = [_dynamo_compile(program, x), _compile_program(program, x)]
    assert state() == exported
    assert _marker_count(program.graph_module) == 4
    assert not _has_capture_leftovers(program.graph_module)
    # The program still runs as the model, and saves its markers.
    with torch.no_grad():
        _assert_close(program.module()(x), model(x))
    loaded = _save_and_load_program(program)
    assert _marker_count(loaded.graph_module) == 4
    assert not _has_capture_leftovers(loaded.graph_module)
    compiled.append(_dynamo_compile(loaded, x))

    placement = _placement(compiled[0])
    assert [region_id for region_id, *_ in placement] == ["kept#0", "replaced#1"]
    for module in compiled:
        assert _placement(module) == placement
        _calls.clear()
        with torch.no_grad():
            _assert_close(module(x), expected)
        assert _calls == ["plus_hundred"]
