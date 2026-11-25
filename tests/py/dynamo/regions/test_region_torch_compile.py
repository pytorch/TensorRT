# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Regions under torch.compile with the Torch-TensorRT backend.

Scopes written in forward and ``regions=`` rules must survive every way of
reaching the backend, recompiles and graph breaks. A failure must reach the
caller instead of running the model without the regions it asked for.
"""

import contextlib
import logging
import operator
import re
import threading

import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt
from torch._dynamo.exc import BackendCompilerFailed, InternalTorchDynamoError
from torch.utils._triton import has_triton
from torch_tensorrt import region
from torch_tensorrt.dynamo.backend import backends
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    get_region_records,
)
from torch_tensorrt.dynamo.region_compilation._types import REGION_ID_META_KEY
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region import RegionCaptureError, RegionError

# Use test-level skips: collection skips omit the JUnit ``type`` attribute and
# cannot be consumed by the CI results reporter on Windows and TensorRT RTX.
try:
    import tensorrt.plugin as trtp
    import triton
    import triton.language as tl
except ImportError:
    triton = None

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)
requires_cuda_qdp = pytest.mark.skipif(
    triton is None
    or not torch.cuda.is_available()
    or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="Triton, CUDA and TensorRT QDP plugin support are required",
)
requires_cuda_inductor = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="CUDA Inductor requires Triton support on the device",
)
BLOCK = 128
# Every way to reach the backend: its two names, with and without fullgraph,
# and torch_tensorrt.compile.
ENTRY_POINTS = [
    "torch_tensorrt",
    "tensorrt",
    "torch_tensorrt-fullgraph",
    "tensorrt-fullgraph",
    "ir_torch_compile",
]

_calls = []


# The operators add 100, which the code they replace does not: a result that
# holds it, and the call list, show that the operator ran.
@torch.library.custom_op("ttrt_torch_compile_test::plus_hundred", mutates_args=())
def _plus_hundred(x: torch.Tensor) -> torch.Tensor:
    _calls.append("plus_hundred")
    return x + 100


@_plus_hundred.register_fake
def _plus_hundred_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_torch_compile_test::scale_plus_hundred", mutates_args=())
def _scale_plus_hundred(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _calls.append("scale_plus_hundred")
    return x * weight + 100


@_scale_plus_hundred.register_fake
def _scale_plus_hundred_fake(x, weight):
    return torch.empty_like(x)


@torch.library.custom_op(
    "ttrt_torch_compile_test::copy_and_plus_hundred", mutates_args=()
)
def _copy_and_plus_hundred(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _calls.append("copy_and_plus_hundred")
    return x.clone(), x + 100


@_copy_and_plus_hundred.register_fake
def _copy_and_plus_hundred_fake(x):
    return torch.empty_like(x), torch.empty_like(x)


if triton is not None:

    @triton.jit
    def _shifted_relu_kernel(x, n, y, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < n, other=0.0) + 0.25
        # The extra 100 is not in the replaced code: it shows the kernel ran.
        tl.store(y + offsets, tl.where(values < 0, 0.0, values) + 100.0, offsets < n)


def _grid(inputs, outputs):
    return (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),)


def _extras(inputs, outputs):
    return [trtp.SymInt32(inputs[0].shape_expr.numel())]


@pytest.fixture
def graphs(monkeypatch):
    """Every module the Torch-TensorRT backend hands back to torch.compile."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    _calls.clear()
    recorded = []
    compile_module = backends.compile_module

    def record(*args, **kwargs):
        recorded.append(compile_module(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(backends, "compile_module", record)
    yield recorded
    torch._dynamo.reset()


def _compile(model, *, backend="torch_tensorrt", **options):
    """torch.compile with the backend; ``fullgraph`` and ``dynamic`` pass through."""
    arguments = {
        key: options.pop(key) for key in ("fullgraph", "dynamic") if key in options
    }
    return torch.compile(
        model, backend=backend, options={"min_block_size": 1, **options}, **arguments
    )


def _build(entry, model, x, **options):
    if entry == "ir_torch_compile":
        return torch_tensorrt.compile(
            model, ir="torch_compile", inputs=[x], min_block_size=1, **options
        )
    backend, _, fullgraph = entry.partition("-")
    return _compile(model, backend=backend, fullgraph=bool(fullgraph), **options)


def _run(compiled, *inputs):
    with torch.no_grad():
        return compiled(*inputs)


def _close(actual, expected):
    torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-4, atol=1e-4)


@contextlib.contextmanager
def _backend_raises(error, message):
    """A call fails in the backend with ``error``, whose text matches ``message``."""
    with pytest.raises(BackendCompilerFailed, match=message) as raised:
        yield raised
    assert isinstance(raised.value.inner_exception, error)


def _another_thread_can_take(lock):
    """Whether no thread holds the lock, asked from a thread that cannot hold it."""
    taken = []

    def attempt():
        taken.append(lock.acquire(blocking=False))
        if taken[0]:
            lock.release()

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()
    return taken[0]


def _writes_in_place(body):
    """Whether a call in the region's body still asks to work in place."""
    return any(
        node.kwargs.get("inplace") is True or any(arg is True for arg in node.args)
        for node in body.graph.nodes
    )


def _engine_count(graph):
    return sum(isinstance(module, TorchTensorRTModule) for module in graph.modules())


def _targets(body):
    return [
        node.target
        for node in body.graph.nodes
        if node.op in ("call_function", "call_method")
    ]


def _torch_regions(graph, ids):
    """Check each region is one PyTorch call outside TensorRT; return the bodies."""
    records = get_region_records(graph)
    assert [record.id for record in records] == ids
    # Raises if a region entered a TensorRT engine or lost its call.
    audit_region_placement(graph, records)
    bodies = []
    for record in records:
        assert record.route == "torch"
        assert not record.owner.startswith("_run_on_acc")
        owner = graph if record.owner == "<root>" else graph.get_submodule(record.owner)
        assert not isinstance(owner, TorchTensorRTModule)
        body = owner.get_submodule(record.child_target)
        assert body.meta[REGION_ID_META_KEY] == record.id
        bodies.append(body)
    return bodies


def _plugin_region(graph, region_id):
    """Check that the region is one kernel layer inside a TensorRT engine."""
    (record,) = get_region_records(graph)
    assert (record.id, record.route) == (region_id, "plugin")
    assert record.owner.startswith("_run_on_acc")
    engine = graph.get_submodule(record.owner)
    assert isinstance(engine, TorchTensorRTModule)
    assert list(engine._torch_tensorrt_plugin_regions) == [region_id]
    assert str(record.plugin_op) in engine.get_layer_info()
    # No call of the kernel may remain outside the engine.
    for module in graph.modules():
        if isinstance(module, torch.fx.GraphModule):
            assert record.plugin_op not in _targets(module)


def _ignored_warnings(caplog):
    logger = region._context.logger.name
    return [r.getMessage() for r in caplog.records if r.name == logger]


class Scoped(torch.nn.Module):
    """One execute_in_torch scope between operations that TensorRT converts."""

    def __init__(self, backend="eager"):
        super().__init__()
        self.backend = backend

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="activation", backend=self.backend):
            x = torch.relu(x) * 3
        return x * 2


class FusedScope(torch.nn.Module):
    """One fused_op scope: eager PyTorch takes a sigmoid, the operator adds 100."""

    def forward(self, x):
        x = x + 1
        with region.fused_op(_plus_hundred, inputs=[x], name="shift"):
            x = torch.sigmoid(x)
        return x * 2


class Body(torch.nn.Module):
    """One execute_in_torch scope around ``body``; ``after`` reads its result."""

    def __init__(self, body, after=None):
        super().__init__()
        self.body = body
        self.after = after

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="activation"):
            y = self.body(x)
        return y * 2 if self.after is None else self.after(y)


class Activation(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) * 3


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(16))

    def forward(self, x):
        # In fp32 the cast returns ``x`` itself.
        return torch.sigmoid(x.float() * self.weight)


class Plain(torch.nn.Module):
    """No scope in the source: its regions come from rules."""

    def __init__(self, activation=None):
        super().__init__()
        self.activation = Activation() if activation is None else activation
        self.scale = Scale()

    def forward(self, x):
        x = self.activation(x + 1)
        x = self.scale(x)
        return x * 2


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = Scale()
        self.activation = Activation()

    def forward(self, x):
        return self.activation(self.scale(x))


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + 1
        return x


class GraphBreak(torch.nn.Module):
    """A graph break before, inside or after the region of ``forward``."""

    def __init__(self, position, scoped=True):
        super().__init__()
        self.position = position
        self.scoped = scoped
        self.activation = Activation()

    def forward(self, x):
        x = x + 1
        if self.position == "before":
            torch._dynamo.graph_break()
        x = x * 2
        if self.scoped:
            with region.execute_in_torch(name="activation"):
                x = torch.relu(x) * 3
                if self.position == "inside":
                    torch._dynamo.graph_break()
                x = x + 0.5
        else:
            x = self.activation(x) + 0.5
        x = x - 1
        if self.position == "after":
            torch._dynamo.graph_break()
        return x * 2


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_every_entry_point_keeps_a_scope_outside_tensorrt(entry, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Scoped().eval().cuda()
    compiled = _build(entry, model, x)
    _close(_run(compiled, x), model(x))
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["activation#0"])
    assert torch.relu in _targets(body)
    # The operations before and after the scope still run in TensorRT.
    assert _engine_count(graph) == 2


@pytest.mark.parametrize("entry", ["torch_tensorrt", "ir_torch_compile"])
def test_rules_place_modules_in_regions(entry, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Plain().eval().cuda()
    rules = {
        "activation": region.ExecuteInTorch(),
        Scale: region.FusedOp(_scale_plus_hundred, inputs=[0, "weight"]),
    }
    compiled = _build(entry, model, x, regions=rules)
    expected = (torch.relu(x + 1) * 3 * model.scale.weight + 100) * 2
    _close(_run(compiled, x), expected)
    assert _calls == ["scale_plus_hundred"]
    (graph,) = graphs
    activation, scale = _torch_regions(graph, ["activation#0", "scale#1"])
    assert torch.relu in _targets(activation)
    assert _targets(scale) == [
        torch.ops.ttrt_torch_compile_test.scale_plus_hundred.default
    ]
    assert _engine_count(graph) == 2


class LogProbabilities(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, x):
        x = x + 1
        with region.execute_in_torch(name="log_probabilities", backend=self.backend):
            # Lowered for TensorRT these become log(softmax(x)) and a copy.
            y = torch.log_softmax(x, dim=-1).view(2, 2)
        return y * 2


@pytest.mark.parametrize(
    "backend, grad",
    [
        ("eager", False),
        ("eager", True),
        pytest.param("inductor", False, marks=requires_cuda_inductor),
    ],
)
def test_region_runs_the_operations_the_model_wrote(backend, grad, graphs):
    # log(softmax(x)) underflows to -inf for these logits; log_softmax does not.
    x = torch.tensor([[0.0, -200.0, 50.0, -1000.0]], device="cuda")
    model = LogProbabilities(backend).eval().cuda()
    compiled = _compile(model)
    with torch.set_grad_enabled(grad):
        result = compiled(x)
    assert torch.isfinite(result).all()
    _close(result, model(x))
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["log_probabilities#0"])
    assert get_region_records(graph)[0].backend == backend
    assert _targets(body) == [torch.log_softmax, "view"]


@requires_cuda_inductor
@pytest.mark.parametrize("source", ["scope", "rule"])
def test_inductor_region_compiles_once_and_is_reused(source, graphs, monkeypatch):
    compiled_bodies, runs, lock_was_held = [], [], []
    inductor_compile = torch._inductor.compile
    # Inductor and torch.compile share tracing state, which this lock protects.
    compile_lock = torch._dynamo.convert_frame.compile_lock

    def compile_spy(body, example_inputs, **kwargs):
        compiled_bodies.append(body)
        lock_was_held.append(not _another_thread_can_take(compile_lock))
        run_body = inductor_compile(body, example_inputs, **kwargs)

        def run(*args):
            runs.append(len(args))
            return run_body(*args)

        return run

    monkeypatch.setattr(torch._inductor, "compile", compile_spy)
    x = torch.randn(4, 16, device="cuda")
    if source == "scope":
        model = Scoped(backend="inductor").eval().cuda()
        compiled = _compile(model)
    else:
        model = Plain().eval().cuda()
        rules = {"activation": region.ExecuteInTorch(backend="inductor")}
        compiled = _compile(model, regions=rules)
    for value in (x, x + 1, x - 1):
        _close(_run(compiled, value), model(value))
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["activation#0"])
    assert get_region_records(graph)[0].backend == "inductor"
    # Inductor compiled the region's own body, once, and ran it on every call.
    assert compiled_bodies == [body]
    assert runs == [1, 1, 1]
    assert _engine_count(graph) == 2
    # The body compiles on the first call, after torch.compile let go of the
    # lock: the region took it again for the time Inductor ran.
    assert lock_was_held == [True]
    assert _another_thread_can_take(compile_lock)


def test_fused_op_scope_calls_its_operator_on_every_run(graphs):
    x = torch.randn(4, 16, device="cuda")
    model = FusedScope().eval().cuda()
    # Eager PyTorch runs the code in the scope, not the operator.
    _close(model(x), torch.sigmoid(x + 1) * 2)
    assert not _calls
    compiled = _compile(model)
    for value in (x, x + 1, x - 1):
        _close(_run(compiled, value), (value + 1 + 100) * 2)
    assert _calls == ["plus_hundred"] * 3
    # Three calls with one shape compile once.
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["shift#0"])
    assert _targets(body) == [torch.ops.ttrt_torch_compile_test.plus_hundred.default]
    assert _engine_count(graph) == 2


class KernelScope(torch.nn.Module):
    def forward(self, x):
        x = x * 2
        with region.fused_kernel(
            kernel=lambda: _shifted_relu_kernel,
            name="activation",
            constexprs={"BLOCK": BLOCK},
            grid=_grid,
            extra_args_fn=_extras,
        ):
            y = torch.relu(x + 0.25)
        return y - 1


class ShiftedRelu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 0.25)


class KernelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = ShiftedRelu()

    def forward(self, x):
        return self.activation(x * 2) - 1


@requires_cuda_qdp
@pytest.mark.parametrize("source", ["scope", "torch_tensorrt", "ir_torch_compile"])
def test_fused_kernel_is_a_plugin_layer_inside_an_engine(source, graphs):
    x = torch.randn(3, 129, device="cuda")
    if source == "scope":
        model = KernelScope().eval().cuda()
        compiled = _compile(model)
    else:
        model = KernelModule().eval().cuda()
        rule = region.FusedKernel(
            lambda: _shifted_relu_kernel,
            constexprs={"BLOCK": BLOCK},
            grid=_grid,
            extra_args_fn=_extras,
        )
        compiled = _build(source, model, x, regions={"activation": rule})
    for value in (x, torch.randn_like(x)):
        # Eager PyTorch runs the replaced code, which adds no 100.
        _close(_run(compiled, value), model(value) + 100)
    (graph,) = graphs
    _plugin_region(graph, "activation#0")
    assert _engine_count(graph) == 1


@pytest.mark.parametrize("fused", [False, True], ids=["execute_in_torch", "fused_op"])
def test_scope_may_cast_to_the_type_it_already_has(fused, graphs):
    class Cast(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            # In fp32 the cast returns ``x`` itself.
            if fused:
                with region.fused_op(_plus_hundred, inputs=[x], name="cast"):
                    x = torch.sigmoid(x.float())
            else:
                with region.execute_in_torch(name="cast"):
                    x = torch.sigmoid(x.float())
            return x * 2

    x = torch.randn(4, 16, device="cuda")
    model = Cast().eval().cuda()
    compiled = _compile(model)
    _close(_run(compiled, x), (x + 1 + 100) * 2 if fused else model(x))
    assert _calls == ["plus_hundred"] * fused
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["cast#0"])
    if not fused:
        assert _targets(body) == ["float", torch.sigmoid]


class PassThrough(torch.nn.Module):
    """A region that returns its result and, uncopied, the tensor it was given."""

    class Cast(torch.nn.Module):
        def forward(self, x):
            # In fp32 the cast returns ``x`` itself.
            hidden = x.float()
            return torch.sigmoid(hidden), hidden

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.cast = self.Cast()

    def forward(self, x):
        x = x + 1
        if self.source == "fused_op":
            with region.fused_op(_plus_hundred, inputs=[x], name="cast"):
                hidden = x.float()
                y = torch.sigmoid(hidden)
        elif self.source == "fused_op_with_clone":
            # The copy is a result of the scope, so the operator returns it too.
            with region.fused_op(_copy_and_plus_hundred, inputs=[x], name="cast"):
                hidden = x.clone().float()
                y = torch.sigmoid(hidden)
        elif self.source == "execute_in_torch":
            with region.execute_in_torch(name="cast"):
                hidden = x.float()
                y = torch.sigmoid(hidden)
        else:
            y, hidden = self.cast(x)
        return y * 2 + hidden


@pytest.mark.parametrize("source", ["execute_in_torch", "rule"])
def test_region_may_pass_a_tensor_through(source, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = PassThrough(source).eval().cuda()
    options = {"regions": {"cast": region.FusedOp(_plus_hundred)}}
    compiled = _compile(model, **(options if source == "rule" else {}))
    # Later reads of the tensor take the region's input, not a region output.
    expected = (x + 1 + 100) * 2 + (x + 1) if source == "rule" else model(x)
    _close(_run(compiled, x), expected)
    assert _calls == ["plus_hundred"] * (source == "rule")
    (graph,) = graphs
    _torch_regions(graph, ["cast#0"])


def test_fused_scope_that_returns_its_input_asks_for_a_clone(graphs):
    x = torch.randn(4, 16, device="cuda")
    compiled = _compile(PassThrough("fused_op").eval().cuda())
    with _backend_raises(
        RegionCaptureError, "would output its input 'x' unchanged.*clone 'x'"
    ):
        _run(compiled, x)
    assert not graphs and not _calls
    # With the clone the error asks for, the scope compiles: the backend
    # captures regions before it drops the clones of the rest of the graph.
    compiled = _compile(PassThrough("fused_op_with_clone").eval().cuda())
    _close(_run(compiled, x), (x + 1 + 100) * 2 + (x + 1))
    assert _calls == ["copy_and_plus_hundred"]
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["cast#0"])
    operator_call = torch.ops.ttrt_torch_compile_test.copy_and_plus_hundred.default
    assert operator_call in _targets(body)


class Halves(torch.nn.Module):
    """Returns two views of one tensor, which a region cannot."""

    def forward(self, x):
        return torch.relu(x).chunk(2, dim=-1)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1)


@pytest.mark.parametrize("views", ["of_its_input", "of_one_result"])
def test_rule_copies_the_views_its_module_returns(views, graphs):
    class Caller(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = Flatten() if views == "of_its_input" else Halves()

        def forward(self, x):
            x = x + 1
            if views == "of_its_input":
                return self.activation(x) * 2 + x.sum()
            first, second = self.activation(x)
            return first * 2 + second

    x = torch.randn(4, 16, device="cuda")
    model = Caller().eval().cuda()
    # A rule selects a module that cannot be edited to clone what it returns.
    compiled = _compile(model, regions={"activation": region.ExecuteInTorch()})
    _close(_run(compiled, x), model(x))
    (graph,) = graphs
    (body,) = _torch_regions(graph, ["activation#0"])
    # One copy gives the results storage of their own: the view of the input,
    # or the second of the two views of one tensor.
    assert _targets(body).count(torch.clone) == 1


@pytest.mark.parametrize(
    "selector",
    ["layers.*.activation", "**.activation", Activation],
    ids=["one_name_glob", "any_depth_glob", "class"],
)
def test_selector_matches_every_module_it_names(selector, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Stack().eval().cuda()
    compiled = _compile(model, regions={selector: region.ExecuteInTorch()})
    _close(_run(compiled, x), model(x))
    (graph,) = graphs
    bodies = _torch_regions(graph, ["layers.0.activation#0", "layers.1.activation#1"])
    # Each region is the selected module only: its neighbor stays in TensorRT.
    for body in bodies:
        assert torch.relu in _targets(body)
        assert torch.sigmoid not in _targets(body)
    assert _engine_count(graph) == 3


def test_class_selector_matches_subclasses(graphs):
    class Special(Activation):
        pass

    x = torch.randn(4, 16, device="cuda")
    model = Plain(Special()).eval().cuda()
    compiled = _compile(model, regions={Activation: region.ExecuteInTorch()})
    _close(_run(compiled, x), model(x))
    _torch_regions(graphs[0], ["activation#0"])


class Nested(torch.nn.Module):
    """``activation`` at the top and in ``block``, whose forward has a graph break."""

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = Activation()

        def forward(self, x):
            x = x - 1
            # torch.compile traces the rest of this forward as a graph of its
            # own, in which the path of the module below is "activation" too.
            torch._dynamo.graph_break()
            return self.activation(x) + 0.5

    def __init__(self):
        super().__init__()
        self.activation = Activation()
        self.block = self.Inner()

    def forward(self, x):
        x = self.activation(x + 1)
        return self.block(x) * 2


@pytest.mark.parametrize(
    "entry, selector, replaced",
    [
        ("torch_tensorrt", "activation", ["activation"]),
        ("ir_torch_compile", "activation", ["activation"]),
        ("torch_tensorrt", "block.activation", ["block.activation"]),
        ("torch_tensorrt", "the_module", ["block.activation"]),
        ("torch_tensorrt", Activation, ["activation", "block.activation"]),
    ],
    ids=["path", "path_ir_torch_compile", "nested_path", "module", "class"],
)
def test_rule_selects_modules_of_the_model_in_every_graph(
    entry, selector, replaced, graphs
):
    x = torch.randn(4, 16, device="cuda")
    model = Nested().eval().cuda()
    if selector == "the_module":
        selector = model.block.activation
    compiled = _build(
        entry, model, x, regions={selector: region.FusedOp(_plus_hundred)}
    )

    # The operator adds 100 where a rule replaced a module, and nowhere else.
    def call(module, value):
        return value + 100 if module in replaced else torch.relu(value) * 3

    hidden = call("activation", x + 1)
    _close(_run(compiled, x), (call("block.activation", hidden - 1) + 0.5) * 2)
    assert _calls == ["plus_hundred"] * len(replaced)
    # The break makes several graphs. A rule applies in each graph that holds
    # one of its modules, and a path is the module's path in the model: the
    # graph traced from the inner forward names its module "activation" too.
    assert len(graphs) > 1
    regions = [record.name for graph in graphs for record in get_region_records(graph)]
    assert regions == replaced


def test_rules_for_a_module_and_its_submodule_raise(graphs):
    x = torch.randn(4, 16, device="cuda")
    rules = {
        "layers.0": region.ExecuteInTorch(),
        "layers.0.activation": region.ExecuteInTorch(backend="inductor"),
    }
    compiled = _compile(Stack().eval().cuda(), regions=rules)
    with _backend_raises(
        RegionCaptureError,
        "Region rules 'layers.0' and 'layers.0.activation' both select .*"
        "in 'layers.0' and 'layers.0.activation'; regions cannot nest or overlap",
    ):
        _run(compiled, x)
    assert not graphs


# Rules that cannot apply to the model: the rules, whether the model is in
# training mode, the error and its message.
_RULE_PROBLEMS = {
    "no_match": (
        lambda: {"missing": region.ExecuteInTorch()},
        False,
        RegionCaptureError,
        "Region rule 'missing' matched no submodule of the model; "
        "its submodules include 'activation', 'scale'",
    ),
    "module_of_another_model": (
        lambda: {Activation(): region.ExecuteInTorch()},
        False,
        RegionCaptureError,
        "Region rule <Activation module> matched no submodule of the model",
    ),
    "training_mode": (
        lambda: {"activation": region.ExecuteInTorch()},
        True,
        RegionCaptureError,
        r"Regions currently require model.eval\(\); still in training mode: "
        "<the model>, activation, scale",
    ),
    "not_a_mapping": (
        lambda: ["activation"],
        False,
        TypeError,
        "regions must map module paths, module classes or modules to region "
        r"rules, such as \{",
    ),
}


@pytest.mark.parametrize("problem", list(_RULE_PROBLEMS))
@pytest.mark.parametrize("entry", ["torch_tensorrt", "ir_torch_compile"])
def test_rules_that_cannot_apply_to_the_model_raise(entry, problem, graphs):
    make_rules, training, error, message = _RULE_PROBLEMS[problem]
    x = torch.randn(4, 16, device="cuda")
    model = Plain().cuda().train(training)
    if entry == "ir_torch_compile":
        # torch_tensorrt.compile matches the rules against the model at once,
        # before the first call compiles anything.
        with pytest.raises(error, match=message):
            _build(entry, model, x, regions=make_rules())
    else:
        # torch.compile learns of the model only when the first call compiles.
        compiled = _build(entry, model, x, regions=make_rules())
        with _backend_raises(error, message):
            _run(compiled, x)
    assert not graphs


class Interrupted(torch.nn.Module):
    """A module whose forward has a graph break."""

    def forward(self, x):
        x = torch.relu(x)
        torch._dynamo.graph_break()
        return x * 3


@pytest.mark.parametrize("selector", ["class", "module"])
def test_module_traced_in_pieces_raises_when_a_function_is_compiled(selector, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Plain(Interrupted()).eval().cuda()
    selected = Interrupted if selector == "class" else model.activation
    compiled = _compile(
        lambda value: model(value), regions={selected: region.FusedOp(_plus_hundred)}
    )
    # The backend sees a function, not the model, and cannot tell the module's
    # path: the message names its class.
    with _backend_raises(
        RegionCaptureError,
        "torch.compile traced part of 'Interrupted' as a graph of its own.*"
        "remove the graph break, or write the scope in the module's forward",
    ):
        _run(compiled, x)
    assert not _calls
    assert not any(get_region_records(graph) for graph in graphs)


@pytest.mark.parametrize(
    "inputs, message",
    [
        (
            None,
            r"the module reads 2 tensors; list them with inputs=\[...\] in the "
            "order the replacement takes them. They are: position 0, 'weight'$",
        ),
        ([0], "inputs must list every tensor the module reads; add 'weight'$"),
        (
            [1, "weight"],
            "input position 1 is out of range; "
            r"the module reads 1 tensor\(s\) from its caller$",
        ),
        (
            [0, "bias"],
            "cannot find the parameter or buffer 'bias'; the module reads 'weight'$",
        ),
    ],
    ids=["inputs_omitted", "parameter_not_listed", "no_such_position", "no_such_name"],
)
def test_rule_inputs_error_names_what_to_list(inputs, message, graphs):
    x = torch.randn(4, 16, device="cuda")
    rules = {"scale": region.FusedOp(_scale_plus_hundred, inputs=inputs)}
    compiled = _compile(Plain().eval().cuda(), regions=rules)
    with pytest.raises(BackendCompilerFailed) as raised:
        _run(compiled, x)
    failure = raised.value.inner_exception
    assert isinstance(failure, RegionCaptureError)
    assert re.search("^Region rule 'scale' at 'scale': " + message, str(failure))
    assert not graphs and not _calls


@pytest.mark.parametrize("selector", ["class", "module"])
def test_class_and_module_rules_select_in_a_function_that_calls_the_model(
    selector, graphs
):
    x = torch.randn(4, 16, device="cuda")
    model = Plain().eval().cuda()
    chosen = Activation if selector == "class" else model.activation
    compiled = _compile(
        lambda value: model(value) + 1,
        regions={chosen: region.FusedOp(_plus_hundred)},
    )
    expected = torch.sigmoid((x + 1 + 100) * model.scale.weight) * 2 + 1
    _close(_run(compiled, x), expected)
    assert _calls == ["plus_hundred"]
    (graph,) = graphs
    _torch_regions(graph, ["activation#0"])
    assert _engine_count(graph) == 2


def test_path_rule_needs_the_model_itself_to_be_compiled(graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Plain().eval().cuda()
    compiled = _compile(
        lambda value: model(value), regions={"activation": region.ExecuteInTorch()}
    )
    # The backend sees a function: it cannot tell which module the path names.
    with _backend_raises(
        RegionCaptureError,
        "Region rule 'activation' selects by path, and the Torch-TensorRT backend "
        "cannot tell which model the path starts from.*"
        "or select by module class or module object",
    ):
        _run(compiled, x)
    assert not graphs


@pytest.mark.parametrize("entry", ["dynamic_false", "ir_torch_compile"])
def test_new_shape_recompiles_and_keeps_the_region(entry, graphs):
    small = torch.randn(4, 16, device="cuda")
    large = torch.randn(8, 16, device="cuda")
    model = Scoped().eval().cuda()
    if entry == "dynamic_false":
        compiled = _compile(model, dynamic=False)
    else:
        compiled = _build(entry, model, small)
    for value in (small, large, small, large):
        _close(_run(compiled, value), model(value))
    # One compilation for each shape, and none when a shape comes back.
    assert len(graphs) == 2
    for graph in graphs:
        _torch_regions(graph, ["activation#0"])
        assert _engine_count(graph) == 2


@pytest.mark.parametrize("source", ["scope", "rule"])
def test_recompile_with_dynamic_shapes_asks_for_static_shapes(source, graphs):
    small = torch.randn(4, 16, device="cuda")
    large = torch.randn(8, 16, device="cuda")
    if source == "scope":
        model = Scoped().eval().cuda()
        compiled = _compile(model)
    else:
        model = Plain().eval().cuda()
        compiled = _compile(model, regions={"activation": region.ExecuteInTorch()})
    _close(_run(compiled, small), model(small))
    # By default torch.compile makes the changed size dynamic when it recompiles.
    with pytest.raises(
        BackendCompilerFailed, match=r"requires static shapes.*dynamic=False"
    ) as raised:
        _run(compiled, large)
    assert isinstance(raised.value.inner_exception, RegionCaptureError)
    assert len(graphs) == 1
    # The first shape still runs its compiled graph, with its region.
    _close(_run(compiled, small), model(small))
    _torch_regions(graphs[0], ["activation#0"])


def test_dynamic_shapes_are_rejected_on_the_first_call(graphs):
    x = torch.randn(4, 16, device="cuda")
    compiled = _compile(Scoped().eval().cuda(), dynamic=True)
    with pytest.raises(
        BackendCompilerFailed, match=r"requires static shapes.*dynamic=False"
    ):
        _run(compiled, x)
    assert not graphs


@pytest.mark.parametrize("position", ["before", "after"])
def test_graph_break_outside_a_scope_keeps_the_region(position, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = GraphBreak(position).eval().cuda()
    compiled = _compile(model)
    _close(_run(compiled, x), model(x))
    # The break splits forward into two graphs; one of them holds the scope.
    first, second = graphs
    with_region, without = (second, first) if position == "before" else (first, second)
    _torch_regions(with_region, ["activation#0"])
    assert _engine_count(with_region) == 2
    assert not get_region_records(without)
    assert _engine_count(without) == 1


def test_graph_break_inside_a_scope_raises(graphs):
    x = torch.randn(4, 16, device="cuda")
    compiled = _compile(GraphBreak("inside").eval().cuda())
    with pytest.raises(RuntimeError, match="[Gg]raph.break"):
        _run(compiled, x)
    assert not graphs


class Skipped(torch.nn.Module):
    """A graph break inside a loop makes torch.compile skip this forward."""

    def forward(self, x):
        for _ in range(2):
            x = x + 1
            torch._dynamo.graph_break()
        with region.fused_op(_plus_hundred, inputs=[x], name="shift"):
            x = torch.sigmoid(x)
        return x * 2


def test_scope_in_a_forward_torch_compile_skips_runs_as_written(
    graphs, monkeypatch, caplog
):
    monkeypatch.setattr(region._context, "_warned_ignored", set())
    x = torch.randn(4, 16, device="cuda")
    compiled = _compile(Skipped().eval().cuda())
    with caplog.at_level(logging.WARNING, logger=region._context.logger.name):
        # The forward runs in Python, and the scope with it: the code in the
        # scope runs, not the operator, as in eager PyTorch.
        for value in (x, x + 1):
            _close(_run(compiled, value), torch.sigmoid(value + 2) * 2)
    assert not _calls
    assert not any(get_region_records(graph) for graph in graphs)
    # One warning says so, however often the forward runs.
    (message,) = _ignored_warnings(caplog)
    assert "ignored in a forward that torch.compile did not trace" in message


def test_pytorch_without_the_needed_features_is_reported(graphs, monkeypatch):
    missing = (
        "Regions (execute_in_torch, fused_op, fused_kernel and regions= rules) "
        "require PyTorch features that this test pretends are missing"
    )
    monkeypatch.setattr(region._marker_ops, "CAPTURE_ERROR", missing)
    x = torch.randn(4, 16, device="cuda")
    scoped = FusedScope().eval().cuda()
    # A scope raises while torch.compile traces it.
    with pytest.raises(InternalTorchDynamoError, match=re.escape(missing)):
        _run(_compile(scoped), x)
    # A rule raises in the backend.
    plain = Plain().eval().cuda()
    compiled = _compile(plain, regions={"activation": region.ExecuteInTorch()})
    with _backend_raises(RegionCaptureError, re.escape(missing)):
        _run(compiled, x)
    assert not graphs and not _calls
    # ignore_regions() traces the scope as ordinary code, which needs nothing.
    torch._dynamo.reset()
    with region.ignore_regions():
        _close(_run(_compile(scoped), x), scoped(x))
    assert not _calls
    (graph,) = graphs
    assert not get_region_records(graph)


@pytest.mark.parametrize(
    "backend", ["eager", pytest.param("inductor", marks=requires_cuda_inductor)]
)
def test_only_other_backends_warn_that_they_ignore_scopes(
    backend, graphs, monkeypatch, caplog
):
    warned = set()
    monkeypatch.setattr(region._context, "_warned_ignored", warned)
    x = torch.randn(4, 16, device="cuda")
    model = FusedScope().eval().cuda()
    with caplog.at_level(logging.WARNING, logger=region._context.logger.name):
        # The Torch-TensorRT backend records the scope: nothing to warn about.
        _close(_run(_compile(model), x), (x + 1 + 100) * 2)
        assert _calls == ["plus_hundred"]
        assert not _ignored_warnings(caplog) and not warned
        _calls.clear()
        # Another backend runs the code in the scope and says so once.
        for _ in range(2):
            torch._dynamo.reset()
            compiled = torch.compile(model, backend=backend, fullgraph=True)
            _close(_run(compiled, x), torch.sigmoid(x + 1) * 2)
        assert not _calls
        (message,) = _ignored_warnings(caplog)
        assert "ignored by this torch.compile backend" in message
        assert "backend='torch_tensorrt'" in message
    assert len(graphs) == 1


def _training_dropout(x):
    return F.dropout(x, 0.5, training=True)


class DropoutAfter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        with region.execute_in_torch(name="activation"):
            x = torch.relu(x + 1)
        return self.dropout(x) * 2


def _calls_a_model_in_training_mode():
    model = DropoutAfter().train().cuda()
    return lambda value: model(value)


_RANDOM = "Random operations are supported only in fused_op regions: "
_RANDOM_BODIES = {
    "dropout": (_training_dropout, "aten.native_dropout.default"),
    "rand_like": (lambda x: x + torch.rand_like(x), "aten.rand_like.default"),
    "randn": (
        lambda x: x + torch.randn(4, 16, device="cuda"),
        "aten.randn.default",
    ),
    "bernoulli": (
        lambda x: x * torch.bernoulli(torch.sigmoid(x)),
        "aten.bernoulli.default",
    ),
}


@pytest.mark.parametrize(
    "build, options, error, message",
    [
        *(
            pytest.param(
                lambda body=body: Body(body).eval().cuda(),
                {},
                RegionCaptureError,
                _RANDOM + operation + " in 'activation#0'",
                id=f"{name}_in_scope",
            )
            for name, (body, operation) in _RANDOM_BODIES.items()
        ),
        pytest.param(
            lambda: Plain(torch.nn.Dropout(0.5)).eval().cuda(),
            {"regions": {"activation": region.FusedOp(_plus_hundred)}},
            RegionCaptureError,
            "fused_op region 'activation#0': the selected module returns its "
            "input unchanged, so its operator or kernel would never run",
            id="fused_rule_over_a_module_that_computes_nothing",
        ),
        pytest.param(
            # The selected module's forward holds a scope of its own.
            lambda: Plain(Scoped()).eval().cuda(),
            {"regions": {"activation": region.ExecuteInTorch()}},
            RegionCaptureError,
            "A region rule selects 'activation', whose operations surround "
            "'activation', a region written in the model's source; regions "
            "cannot nest",
            id="rule_around_a_scope",
        ),
        pytest.param(
            lambda: Scoped().eval().cuda(),
            {"use_distributed_mode_trace": True},
            RegionError,
            "incompatible with use_distributed_mode_trace=True",
            id="distributed_trace",
        ),
        pytest.param(
            # Weights that can be refitted later.
            lambda: Scoped().eval().cuda(),
            {"immutable_weights": False},
            RegionError,
            "Regions are incompatible with immutable_weights=False",
            id="refittable_weights",
        ),
        pytest.param(
            lambda: DropoutAfter().train().cuda(),
            {},
            RegionCaptureError,
            r"Regions currently require model.eval\(\); still in training mode: "
            "<the model>, dropout",
            id="model_in_training_mode",
        ),
        pytest.param(
            # The backend sees a function, not the model: the dropout that
            # the model's graph records says that it was in training mode.
            _calls_a_model_in_training_mode,
            {},
            RegionCaptureError,
            "Regions currently require a model in eval mode: "
            r"aten.native_dropout.default at '\w+' samples at random",
            id="function_that_calls_a_model_in_training_mode",
        ),
    ],
)
def test_region_failure_reaches_the_caller(build, options, error, message, graphs):
    x = torch.randn(4, 16, device="cuda")
    # pass_through_build_failures keeps its default, which falls back to
    # PyTorch after a failed engine build: a region failure must not.
    compiled = _compile(build(), **options)
    # Every call raises: the model never runs without its region.
    for _ in range(2):
        with _backend_raises(error, message):
            _run(compiled, x)
    assert not graphs


def _write_with_a_method(x):
    return torch.relu(x).mul_(3)


def _write_with_an_assignment(x):
    y = x * 3
    y += 1
    return y


def _write_with_the_inplace_keyword(x):
    return F.relu(x * 3, inplace=True)


def _write_with_the_inplace_argument(x):
    return F.leaky_relu(x * 3, 0.1, True)


def _write_then_read(x):
    y = x * 3
    y.relu_()
    return y


def _write_what_is_read_twice(x):
    y = x * 3
    return y.relu_() + y


@pytest.mark.parametrize(
    "body, operations",
    [
        pytest.param(_write_with_a_method, [torch.relu, "mul"], id="method"),
        pytest.param(
            _write_with_an_assignment,
            [operator.mul, operator.add],
            id="augmented_assignment",
        ),
        pytest.param(
            _write_with_the_inplace_keyword,
            [operator.mul, F.relu],
            id="inplace_keyword",
        ),
        pytest.param(
            _write_with_the_inplace_argument,
            [operator.mul, F.leaky_relu],
            id="inplace_positional_argument",
        ),
        pytest.param(
            _write_then_read, [operator.mul, "relu"], id="statement_then_read"
        ),
        # Both operands of the addition read the values that were written.
        pytest.param(
            _write_what_is_read_twice,
            [operator.mul, "relu", operator.add],
            id="tensor_read_twice",
        ),
    ],
)
def test_write_to_a_tensor_the_scope_made_runs_out_of_place(body, operations, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Body(body).eval().cuda()
    _close(_run(_compile(model), x), model(x))
    (graph,) = graphs
    (region_body,) = _torch_regions(graph, ["activation#0"])
    # Nothing else can see the write, so the region computes a new tensor.
    assert _targets(region_body) == operations
    assert not _writes_in_place(region_body)
    assert _engine_count(graph) == 2


def _add_to_the_input(x):
    x += 1
    return x * 3


def _assign_into_the_input(x):
    x[0] = 0
    return x * 3


def _write_through_a_view(x):
    y = x * 3
    y.view(-1).relu_()
    return y


_WRITES = "Mutation or effects inside regions are unsupported: "


@pytest.mark.parametrize(
    "body, operation",
    [
        pytest.param(lambda x: x.mul_(3), "mul_", id="method"),
        pytest.param(_add_to_the_input, "<built-in function iadd>", id="assignment"),
        pytest.param(_assign_into_the_input, "<built-in function setitem>", id="item"),
        pytest.param(
            lambda x: F.relu(x, inplace=True), "<function relu", id="inplace_keyword"
        ),
        pytest.param(
            lambda x: F.leaky_relu(x, 0.1, True),
            "<function leaky_relu",
            id="inplace_positional_argument",
        ),
        pytest.param(
            lambda x: torch.add(x, 1, out=x), "<built-in method add", id="out_keyword"
        ),
        # In fp32 the cast returns ``x`` itself, so this writes the input too.
        pytest.param(lambda x: x.float().relu_(), "relu_", id="after_a_cast"),
        pytest.param(lambda x: x.detach().relu_(), "relu_", id="after_detach"),
        pytest.param(_write_through_a_view, "relu_", id="view_of_a_new_tensor"),
        pytest.param(lambda x: x.__iadd__(1), "__iadd__", id="operator_method"),
    ],
)
def test_write_to_a_region_input_is_rejected(body, operation, graphs):
    x = torch.randn(4, 16, device="cuda")
    compiled = _compile(Body(body).eval().cuda())
    # The write is neither run nor lost without a word.
    with _backend_raises(RegionCaptureError, _WRITES + re.escape(operation)):
        _run(compiled, x)
    assert not graphs


@pytest.mark.parametrize(
    "rule",
    [region.ExecuteInTorch(), region.FusedOp(_plus_hundred)],
    ids=["execute_in_torch", "fused_op"],
)
def test_rule_over_a_module_that_overwrites_its_input_is_rejected(rule, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Plain(torch.nn.ReLU(inplace=True)).eval().cuda()
    compiled = _compile(model, regions={"activation": rule})
    with _backend_raises(RegionCaptureError, _WRITES + "<function relu"):
        _run(compiled, x)
    assert not graphs and not _calls


@pytest.mark.parametrize("failing_step", ["before_capture", "engine_build"])
@pytest.mark.parametrize("source", ["scope", "rule"])
def test_failed_compilation_does_not_run_the_model_without_its_region(
    source, failing_step, graphs, monkeypatch
):
    def fail(*args, **kwargs):
        raise RuntimeError("this step deliberately failed")

    # The first thing the backend does with a graph, and the last.
    step = "detect_fake_mode" if failing_step == "before_capture" else "compile_module"
    monkeypatch.setattr(backends, step, fail)
    x = torch.randn(4, 16, device="cuda")
    # Without regions the backend's default is to run the graph in PyTorch.
    plain = Plain().eval().cuda()
    _close(_run(_compile(plain), x), plain(x))
    # With regions that would run the model without what it asked for.
    if source == "scope":
        compiled = _compile(Scoped().eval().cuda())
    else:
        compiled = _compile(plain, regions={"activation": region.ExecuteInTorch()})
    with _backend_raises(
        RegionError,
        "Compilation of a torch.compile graph with regions failed: "
        "RuntimeError: this step deliberately failed",
    ) as raised:
        _run(compiled, x)
    assert isinstance(raised.value.inner_exception.__cause__, RuntimeError)
    assert not graphs


@pytest.mark.parametrize("body", ["rand_like", "dropout"])
def test_fused_op_scope_may_replace_random_operations(body, graphs):
    class Noise(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with region.fused_op(_plus_hundred, inputs=[x], name="noise"):
                x = _RANDOM_BODIES[body][0](x)
            return x * 2

    x = torch.randn(4, 16, device="cuda")
    compiled = _compile(Noise().eval().cuda())
    _close(_run(compiled, x), (x + 1 + 100) * 2)
    assert _calls == ["plus_hundred"]
    (body,) = _torch_regions(graphs[0], ["noise#0"])
    assert _targets(body) == [torch.ops.ttrt_torch_compile_test.plus_hundred.default]


class Writes(torch.nn.Module):
    """One region in a model that also writes to its input or to a buffer."""

    def __init__(self, write, scoped):
        super().__init__()
        self.write = write
        self.scoped = scoped
        self.activation = Activation()
        self.norm = torch.nn.BatchNorm1d(16)
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x):
        y = x * 2
        if self.write == "input_before_the_region":
            x.add_(1)
        elif self.write == "buffer":
            self.count += 1
        elif self.write == "batch_norm_statistics":
            y = self.norm(y)
        if self.scoped:
            with region.execute_in_torch(name="activation"):
                y = torch.relu(y) * 3
        else:
            y = self.activation(y)
        if self.write == "input_after_the_region":
            x.add_(1)
        return y * 2


@pytest.mark.parametrize("scoped", [True, False], ids=["scope", "rule"])
@pytest.mark.parametrize(
    "write, written",
    [
        ("input_before_the_region", "mutation of input or buffer 'l_x_'"),
        ("input_after_the_region", "mutation of input or buffer 'l_x_'"),
        ("buffer", "mutation of input or buffer 'l_self_buffers_count_'"),
        ("batch_norm_statistics", r".*the running statistics of batch norm '\w+'"),
    ],
    ids=["input_before_the_region", "input_after_the_region", "buffer", "batch_norm"],
)
def test_write_outside_a_region_is_rejected(write, written, scoped, graphs):
    x = torch.randn(4, 16, device="cuda")
    model = Writes(write, scoped).eval().cuda()
    if write == "batch_norm_statistics":
        # In training mode batch norm updates its running statistics.
        model.norm.train()
    options = {} if scoped else {"regions": {"activation": region.ExecuteInTorch()}}
    compiled = _compile(model, **options)
    before = (x.clone(), model.count.clone(), model.norm.running_mean.clone())
    # Compilation would drop the write, so a model with regions refuses it.
    with _backend_raises(
        RegionCaptureError,
        f"Explicit regions require a functional model: {written} is unsupported "
        "anywhere in a model containing live regions",
    ):
        _run(compiled, x)
    assert not graphs
    for value, saved in zip((x, model.count, model.norm.running_mean), before):
        torch.testing.assert_close(value, saved)


def _block(kind):
    if kind == "autocast":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.no_grad() if kind == "no_grad" else torch.enable_grad()


class Blocked(torch.nn.Module):
    """An autocast or grad-mode block around a region, or inside it."""

    class Inside(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        def forward(self, x):
            x = x * 0.5
            with _block(self.kind):
                x = torch.relu(x) * 3
            return x.float()

    def __init__(self, kind, inside, scoped):
        super().__init__()
        self.kind = kind
        self.inside = inside
        self.scoped = scoped
        self.activation = self.Inside(kind) if inside else Activation()

    def forward(self, x):
        x = x + 1
        if self.inside and self.scoped:
            with region.execute_in_torch(name="activation"):
                x = x * 0.5
                with _block(self.kind):
                    x = torch.relu(x) * 3
                x = x.float()
        elif self.inside:
            x = self.activation(x)
        elif self.scoped:
            with _block(self.kind):
                with region.execute_in_torch(name="activation"):
                    x = torch.relu(x) * 3
        else:
            with _block(self.kind):
                x = self.activation(x)
        return x.float() * 2


_GRAD_BLOCK = r"torch.no_grad\(\) or torch.enable_grad\(\)"


@pytest.mark.parametrize("scoped", [True, False], ids=["scope", "rule"])
@pytest.mark.parametrize("inside", [False, True], ids=["around", "inside"])
@pytest.mark.parametrize(
    "kind, grad, block",
    [
        ("autocast", False, "torch.autocast"),
        # A grad-mode block is recorded when it changes the mode of the call.
        ("no_grad", True, _GRAD_BLOCK),
        ("enable_grad", False, _GRAD_BLOCK),
    ],
    ids=["autocast", "no_grad", "enable_grad"],
)
def test_block_in_the_model_does_not_mix_with_a_region(
    kind, grad, block, inside, scoped, graphs
):
    x = torch.randn(4, 16, device="cuda")
    model = Blocked(kind, inside, scoped).eval().cuda()
    options = {} if scoped else {"regions": {"activation": region.ExecuteInTorch()}}
    compiled = _compile(model, **options)
    # The region's body runs later, outside the block that the model wrote.
    if inside:
        message = f"A {block} block inside a region is unsupported: 'activation#0'. "
    else:
        message = (
            f"A region inside a {block} block is unsupported under torch.compile. "
        )
    advice = "[Ee]nter the block around the call of the compiled model"
    with torch.set_grad_enabled(grad):
        with _backend_raises(RegionCaptureError, message + ".*" + advice):
            compiled(x)
    assert not graphs
