# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise automatic region registration through the public compile API."""

from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import kernels, region
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    get_region_records,
)
from torch_tensorrt.region import RegionCaptureError, RegionError

from ._utils import compile_regions as _compile

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
trtp = pytest.importorskip("tensorrt.plugin")

requires_cuda_qdp = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="CUDA and TensorRT QDP plugin support are required",
)
BLOCK = 128


@triton.jit
def _unary_kernel(x, n, y, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < n, other=0.0) + 0.25
    tl.store(y + offsets, tl.where(values < 0, 0.0, values), offsets < n)


@triton.jit
def _row_kernel(x, cols, y, row_sum, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + row * cols + offsets, offsets < cols, other=0.0)
    shifted = values + 0.25
    tl.store(
        y + row * cols + offsets,
        tl.where(shifted < 0, 0.0, shifted),
        offsets < cols,
    )
    tl.store(row_sum + row, tl.sum(values, 0))


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


def _grid(inputs, outputs):
    return (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),)


def _extras(inputs, outputs):
    return [trtp.SymInt32(inputs[0].shape_expr.numel())]


def _fused(kernel=lambda: _unary_kernel, **overrides):
    options = dict(constexprs={"BLOCK": BLOCK}, grid=_grid, extra_args_fn=_extras)
    options.update(overrides)
    return region.fused_kernel(kernel=kernel, **options)


class Unary(torch.nn.Module):
    def forward(self, x):
        with _fused(name="activation"):
            y = torch.relu(x + 0.25)
        return y


def _assert_plugin_ownership(compiled, count):
    records = [
        record for record in get_region_records(compiled) if record.route == "plugin"
    ]
    assert len(records) == count
    engine_ids = [
        region_id
        for module in compiled.modules()
        for region_id in getattr(module, "_torch_tensorrt_plugin_regions", ())
    ]
    assert sorted(engine_ids) == sorted(record.id for record in records)
    assert all(record.owner.startswith("_run_on_acc") for record in records)
    # No plugin call may remain outside an engine.
    for module in compiled.modules():
        if isinstance(module, torch.fx.GraphModule):
            assert not any(
                node.op == "call_function" and node.target == record.plugin_op
                for node in module.graph.nodes
                for record in records
            )
    return records


def test_unowned_context_does_not_resolve_or_register_a_kernel(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("An unowned region attempted kernel registration")

    monkeypatch.setattr(kernels, "triton_op", unexpected)
    monkeypatch.setattr(kernels, "cuda_kernel_op", unexpected)

    class Eager(torch.nn.Module):
        def forward(self, x):
            with region.fused_kernel(kernel=unexpected):
                result = x.sin() + 1
            return result

    x = torch.randn(2, 8)
    expected = x.sin() + 1
    torch.testing.assert_close(Eager()(x), expected)
    exported = torch.export.export(Eager(), (x,), strict=True)
    torch.testing.assert_close(exported.module()(x), expected)
    assert not get_region_records(exported.module())


@requires_cuda_qdp
@pytest.mark.parametrize("use_fast_partitioner", [True, False])
def test_triton_region_is_one_plugin_in_a_real_engine(use_fast_partitioner):
    x = torch.randn(3, 129, device="cuda")
    model = Unary().eval()
    with torch.inference_mode():
        compiled = _compile(
            model,
            [x],
            use_fast_partitioner=use_fast_partitioner,
            require_full_compilation=True,
        )
        records = _assert_plugin_ownership(compiled, 1)
        torch.testing.assert_close(compiled(x), model(x))
        other = torch.randn_like(x)
        torch.testing.assert_close(compiled(other), model(other))

        # Eager dispatch must respect the captured shape and dtype.
        op = records[0].plugin_op
        torch.testing.assert_close(op(x), model(x))
        with pytest.raises(RegionError, match="captured static shapes"):
            op(torch.randn(4, 129, device="cuda"))
        with pytest.raises(RegionError, match="captured static shapes"):
            op(x.to(torch.float16))

        engine = compiled.get_submodule(records[0].owner)
        graph = torch.fx.Graph()
        graph.output(graph.call_module("nested", (graph.placeholder("x"),)))
        wrapper = torch.fx.GraphModule({"nested": compiled}, graph)
        nested_record = replace(records[0], owner=None)
        audit_region_placement(wrapper, [nested_record], require_plugins_in_trt=True)
        assert wrapper.get_submodule(nested_record.owner) is engine


@requires_cuda_qdp
def test_plugin_specializations_keep_shapes_and_storage_offsets_independent():
    model = Unary().eval()
    retained = []
    with torch.inference_mode():
        for rows, columns in ((3, 17), (5, 29)):
            x = torch.randn(rows + 1, columns, device="cuda")[1:]
            assert x.is_contiguous() and x.storage_offset() > 0
            compiled = _compile(model, [x], require_full_compilation=True)
            op = _assert_plugin_ownership(compiled, 1)[0].plugin_op
            retained.append((compiled, op, x))

        # New registrations must preserve earlier shape and offset bindings.
        for index, (compiled, op, x) in enumerate(retained):
            torch.testing.assert_close(compiled(x), model(x))
            torch.testing.assert_close(op(x), model(x))
            with pytest.raises(RegionError, match="captured static shapes"):
                op(retained[1 - index][2])
        with torch._subclasses.fake_tensor.FakeTensorMode():
            synthetic = torch.empty(5, 5, device="cuda")
            for _, op, x in retained:
                assert tuple(op(synthetic).shape) == tuple(x.shape)


@requires_cuda_qdp
@pytest.mark.parametrize("with_parameter", [False, True])
def test_plugin_inputs_follow_first_use_not_exported_placeholder_order(with_parameter):
    class ReversedInputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(
                torch.full((3, 17), 2.0, device="cuda"), requires_grad=False
            )

        def forward(self, x, y):
            # First use puts y first; subtraction exposes reversed bindings.
            with _fused(lambda: _subtract_kernel):
                result = y - (self.offset if with_parameter else x)
            return result

    x = torch.full((3, 17), -3.0, device="cuda")
    y = torch.arange(51, device="cuda", dtype=torch.float32).reshape(3, 17) + 10
    model = ReversedInputs().eval()
    with torch.inference_mode():
        compiled = _compile(model, [x, y], require_full_compilation=True)
        record = _assert_plugin_ownership(compiled, 1)[0]
        expected = model(x, y)
        torch.testing.assert_close(compiled(x, y), expected)
        second = model.offset if with_parameter else x
        torch.testing.assert_close(record.plugin_op(y, second), expected)


@requires_cuda_qdp
@pytest.mark.parametrize("use_fast_partitioner", [True, False])
def test_multioutput_plugin_derives_different_output_shapes(use_fast_partitioner):
    class MultiOutput(torch.nn.Module):
        def forward(self, x):
            with region.fused_kernel(
                kernel=lambda: _row_kernel,
                constexprs={"BLOCK": BLOCK},
                grid=lambda inputs, outputs: (inputs[0].shape_expr[0],),
                extra_args_fn=lambda inputs, outputs: [
                    trtp.SymInt32(inputs[0].shape_expr[1])
                ],
            ):
                activated = torch.relu(x + 0.25)
                reduced = torch.sum(x, dim=-1)
            # The kernel follows production order, independent of return order.
            return reduced, activated

    x = torch.randn(3, 17, device="cuda")
    model = MultiOutput().eval()
    with torch.inference_mode():
        compiled = _compile(
            model,
            [x],
            require_full_compilation=True,
            use_fast_partitioner=use_fast_partitioner,
        )
        record = _assert_plugin_ownership(compiled, 1)[0]
        assert len(record.output_names) == 2
        torch.testing.assert_close(compiled(x), model(x), atol=1e-5, rtol=1e-5)
        with torch._subclasses.fake_tensor.FakeTensorMode():
            first, second = record.plugin_op(torch.empty(5, 5, device="cuda"))
            assert tuple(first.shape) == (3, 17)
            assert tuple(second.shape) == (3,)


@requires_cuda_qdp
def test_cuda_kernel_spec_infers_inputs_and_outputs():
    spec = kernels.KernelSpec(
        kernel_source=r"""
        extern "C" __global__ void relu_add(const float* x, int n, float* y) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = fmaxf(x[i] + 0.25f, 0.0f);
        }
        """,
        kernel_name="relu_add",
        extras=[kernels.Numel("arg0")],
        geometry=kernels.Elementwise(),
    )

    class CudaRegion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Local kernel descriptors stay outside Dynamo's graph.
            self.kernel_factory = lambda: spec

        def forward(self, x):
            with region.fused_kernel(kernel=self.kernel_factory):
                y = torch.relu(x + 0.25)
            return y

    x = torch.randn(3, 129, device="cuda")
    model = CudaRegion().eval()
    with torch.inference_mode():
        compiled = _compile(model, [x], require_full_compilation=True)
        _assert_plugin_ownership(compiled, 1)
        torch.testing.assert_close(compiled(x), model(x))


@requires_cuda_qdp
def test_cuda_region_checks_engine_pointer_dtypes_before_launch(monkeypatch):
    import tensorrt as trt
    from torch.fx.experimental.proxy_tensor import make_fx

    from torch_tensorrt.dynamo.region_compilation._plugins import register_region_plugin
    from torch_tensorrt.kernels import _derive, _register
    from torch_tensorrt.region._plugin import prepare_plugin

    class Launch:
        def __init__(self):
            self.lock = Lock()

        def __call__(self, *args):
            raise RuntimeError("launch callback reached")

    launch = Launch()
    spec = kernels.KernelSpec(
        "unused source",
        "kernel",
        extras=[],
        include_paths=[],
        geometry=kernels.Custom(launch),
    )
    captured_spec = prepare_plugin(spec)
    spec.extras.append(kernels.Numel("arg0"))
    spec.include_paths.append("changed_after_capture")
    assert not captured_spec.extras and not captured_spec.include_paths
    assert captured_spec.geometry.fn is launch
    with torch._subclasses.fake_tensor.FakeTensorMode():
        child = make_fx(lambda x: (x + 1,))(torch.empty(8, device="cuda"))
    captured = {}

    def capture_registration(**kwargs):
        captured.update(kwargs)
        namespace, name = kwargs["op_name"].split("::")
        monkeypatch.setattr(
            getattr(torch.ops, namespace),
            name,
            SimpleNamespace(default=object()),
            raising=False,
        )

    monkeypatch.setattr(
        _derive, "_compile_kernel", lambda spec: (b"unused PTX", None, None)
    )
    monkeypatch.setattr(
        _register, "register_precompiled_qdp_plugin", capture_registration
    )
    register_region_plugin(child, captured_spec, region_id="pointer-dtype")
    actual = SimpleNamespace(dtype=trt.float32)
    wrong = SimpleNamespace(dtype=trt.float16)
    for inputs, outputs in (([wrong], [actual]), ([actual], [wrong])):
        with pytest.raises(ValueError, match="descriptor has dtype"):
            captured["aot_fn"](inputs, outputs, 0)
    with pytest.raises(RuntimeError, match="launch callback reached"):
        captured["aot_fn"]([actual], [actual], 0)


@requires_cuda_qdp
def test_repeated_plugin_scopes_with_parameter_and_torch_region_recompile():
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(
                torch.randn(3, 17, device="cuda"), requires_grad=False
            )

        def forward(self, x):
            with region.execute_in_torch(name="torch"):
                x = torch.sin(x)
            with _fused(lambda: _bias_kernel, name="repeated"):
                x = torch.relu(x + self.bias)
            with _fused(lambda: _bias_kernel, name="repeated"):
                x = torch.relu(x + self.bias)
            return x

    x = torch.randn(3, 17, device="cuda")
    model = Mixed().eval()
    previous_ops = set()
    with torch.inference_mode():
        for _ in range(2):
            compiled = _compile(model, [x])
            records = _assert_plugin_ownership(compiled, 2)
            assert len({record.id for record in records}) == 2
            assert all(len(record.input_names) == 2 for record in records)
            assert not previous_ops.intersection(record.plugin_op for record in records)
            previous_ops.update(record.plugin_op for record in records)
            torch_records = [
                record
                for record in get_region_records(compiled)
                if record.route == "torch"
            ]
            assert len(torch_records) == 1
            assert not torch_records[0].owner.startswith("_run_on_acc")
            torch.testing.assert_close(compiled(x), model(x))


@requires_cuda_qdp
@pytest.mark.parametrize("use_fast_partitioner", [True, False])
def test_plugin_value_reused_after_torch_region_has_one_owner(use_fast_partitioner):
    class Reused(torch.nn.Module):
        def forward(self, x):
            with _fused():
                y = torch.relu(x + 0.25)
            with region.execute_in_torch():
                z = torch.sigmoid(y)
            # Reusing y in another engine must not duplicate plugin ownership.
            return y + z

    x = torch.randn(3, 17, device="cuda")
    model = Reused().eval()
    with torch.inference_mode():
        compiled = _compile(model, [x], use_fast_partitioner=use_fast_partitioner)
        _assert_plugin_ownership(compiled, 1)
        torch.testing.assert_close(compiled(x), model(x))


@requires_cuda_qdp
def test_constant_input_plugin_remains_an_engine_layer():
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(3, 17, device="cuda"))

        def forward(self, x):
            with _fused():
                result = torch.relu(self.weight + 0.25)
            return x + result

    x = torch.randn(3, 17, device="cuda")
    model = Constant().eval()
    compiled = _compile(model, [x])
    record = _assert_plugin_ownership(compiled, 1)[0]
    engine = compiled.get_submodule(record.owner)
    assert str(record.plugin_op) in engine.get_layer_info()
    torch.testing.assert_close(compiled(x), model(x))


@requires_cuda_qdp
def test_dead_plugin_scope_does_not_register_a_kernel(monkeypatch):
    class Dead(torch.nn.Module):
        def forward(self, x):
            with _fused():
                _unused = torch.relu(x + 0.25)
            return x * 2

    def unexpected(*args, **kwargs):
        raise AssertionError("Dead region attempted kernel registration")

    monkeypatch.setattr(kernels, "triton_op", unexpected)
    x = torch.randn(3, 17, device="cuda")
    compiled = _compile(Dead().eval(), [x])
    assert not get_region_records(compiled)
    torch.testing.assert_close(compiled(x), x * 2)


@requires_cuda_qdp
def test_plugin_region_rejects_final_pytorch_fallback():
    x = torch.randn(3, 17, device="cuda")
    with pytest.raises(RegionError, match="must run in a TensorRT plugin"):
        # The one custom op is smaller than the requested minimum partition.
        _compile(Unary().eval(), [x], min_block_size=10)


@pytest.mark.parametrize("field,value", [("route", "torch"), ("plugin_key", 123)])
def test_lowering_cannot_change_plugin_policy(field, value):
    from torch_tensorrt.dynamo._settings import CompilationSettings
    from torch_tensorrt.dynamo.lowering.passes.pass_manager import DynamoPassManager
    from torch_tensorrt.dynamo.region_compilation import (
        is_region_hop,
        normalize_region_scopes,
    )
    from torch_tensorrt.region._session import RegionCompilationSession

    with RegionCompilationSession(strict=True):
        exported = normalize_region_scopes(
            torch.export.export(Unary(), (torch.ones(2, 8),), strict=True)
        )

        def change_policy(graph, settings):
            hop = next(node for node in graph.graph.nodes if is_region_hop(node))
            hop.kwargs = {"hints": {**hop.kwargs["hints"], field: value}}
            return graph

        with pytest.raises(RegionError, match="changed or lost a region boundary"):
            DynamoPassManager([change_policy])(
                exported.graph_module, CompilationSettings()
            )


@requires_cuda_qdp
@pytest.mark.parametrize("kind", ["input_layout", "output_layout", "cpu"])
def test_plugin_region_rejects_unsupported_tensor_boundaries(kind):
    class InvalidBoundary(torch.nn.Module):
        def forward(self, x):
            if kind == "input_layout":
                x = x.transpose(0, 1)
            elif kind == "cpu":
                x = x.cpu()
            with _fused():
                y = torch.relu(x + 0.25)
                if kind == "output_layout":
                    y = y.transpose(0, 1)
            return y

    error = "requires CUDA tensors" if kind == "cpu" else "requires contiguous"
    with pytest.raises(RegionError, match=error):
        _compile(InvalidBoundary().eval(), [torch.randn(3, 17, device="cuda")])


@requires_cuda_qdp
def test_plugin_region_rejects_dynamic_shapes():
    shape = torch_tensorrt.Input(
        min_shape=(2, 17),
        opt_shape=(3, 17),
        max_shape=(4, 17),
        dtype=torch.float32,
    )
    with pytest.raises(RegionCaptureError, match="requires static shapes"):
        _compile(Unary().eval(), [shape])


@requires_cuda_qdp
@pytest.mark.parametrize("invalid", ["dtype", "extras", "constexpr"])
def test_plugin_region_rejects_invalid_launch_contract(invalid):
    class InvalidLaunch(torch.nn.Module):
        def forward(self, x):
            with region.fused_kernel(
                kernel=lambda: _unary_kernel,
                signature=(
                    {"x": "*fp16", "n": "i32", "y": "*fp32"}
                    if invalid == "dtype"
                    else None
                ),
                constexprs={} if invalid == "constexpr" else {"BLOCK": BLOCK},
                grid=_grid,
                extra_args_fn=None if invalid == "extras" else _extras,
            ):
                y = torch.relu(x + 0.25)
            return y

    error = {
        "dtype": "signature parameter",
        "extras": "no extra_args_fn",
        "constexpr": "missing constexpr values",
    }[invalid]
    with pytest.raises((RegionError, ValueError), match=error):
        _compile(InvalidLaunch().eval(), [torch.randn(3, 17, device="cuda")])
