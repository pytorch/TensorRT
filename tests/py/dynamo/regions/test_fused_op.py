# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Execute supplied custom operators while preserving explicit Torch placement."""

import io

import pytest
import torch
from torch_tensorrt import region
from torch_tensorrt.dynamo._exporter import export
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
    ConverterSupport,
)
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    get_region_records,
)
from torch_tensorrt.dynamo.region_compilation._capture import _ALIAS_COPY_KEY
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

from ._utils import capture_regions as _capture
from ._utils import compile_regions

_calls = []


@torch.library.custom_op("ttrt_fused_op_test::difference", mutates_args=())
def _difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    _calls.append("difference")
    return first - second


@_difference.register_fake
def _difference_fake(first, second):
    return torch.empty_like(first)


@torch.library.custom_op("ttrt_fused_op_test::multiple", mutates_args=())
def _multiple(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.relu(x), x.sum(dim=-1).to(torch.float64)


@_multiple.register_fake
def _multiple_fake(x):
    return torch.empty_like(x), x.new_empty(x.shape[:-1], dtype=torch.float64)


@torch.library.custom_op("ttrt_fused_op_test::single_tuple", mutates_args=())
def _single_tuple(x: torch.Tensor) -> tuple[torch.Tensor]:
    return (x.sin(),)


@_single_tuple.register_fake
def _single_tuple_fake(x):
    return (torch.empty_like(x),)


@torch.library.custom_op("ttrt_fused_op_test::upcast_relu", mutates_args=())
def _upcast_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x.double())


@_upcast_relu.register_fake
def _upcast_relu_fake(x):
    return torch.empty_like(x, dtype=torch.float64)


@torch.library.custom_op("ttrt_fused_op_test::noise", mutates_args=())
def _noise(x: torch.Tensor) -> torch.Tensor:
    return x + torch.rand_like(x)


@_noise.register_fake
def _noise_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("ttrt_fused_op_test::gated", mutates_args=())
def _gated(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return up * torch.sigmoid(gate)


@_gated.register_fake
def _gated_fake(up, gate):
    return torch.empty_like(up)


@torch.library.custom_op("ttrt_fused_op_test::affine", mutates_args=())
def _affine(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight.T + bias


@_affine.register_fake
def _affine_fake(x, weight, bias):
    return x.new_empty(x.shape[0], weight.shape[0])


_invalid = torch.library.Library("ttrt_fused_op_invalid", "DEF")
_invalid.define("scalar(Tensor x, int count) -> Tensor")
_invalid.define("alias(Tensor(a) x) -> Tensor(a)")
_invalid.define("mutate(Tensor(a!) x) -> Tensor")


def _targets(graph):
    return [
        node.target
        for child in graph.modules()
        if isinstance(child, torch.fx.GraphModule)
        for node in child.graph.nodes
        if node.op == "call_function"
    ]


class _Difference(torch.nn.Module):
    def __init__(self, op=torch.ops.ttrt_fused_op_test.difference):
        super().__init__()
        self.op = op

    def forward(self, second, first):
        with region.fused_op(self.op, inputs=(first, second), name="subtract"):
            result = first - second
        return result.sin()


@pytest.mark.parametrize(
    "op",
    [torch.ops.ttrt_fused_op_test.difference, _difference],
    ids=["ops", "function"],
)
def test_listed_input_order_and_actual_custom_op_execution(op):
    first, second = torch.randn(2, 3), torch.randn(2, 3)
    model = _Difference(op).eval()
    _calls.clear()
    expected = model(second, first)
    unowned = torch.export.export(model, (second, first), strict=True).module()
    torch.testing.assert_close(unowned(second, first), expected)
    assert not _calls

    graph = _capture(model, second, first)
    records = get_region_records(graph)
    assert len(records) == 1 and records[0].route == "torch"
    assert torch.ops.ttrt_fused_op_test.difference.default in _targets(graph)
    _calls.clear()
    torch.testing.assert_close(graph(second, first), expected)
    assert _calls == ["difference"]
    audit_region_placement(graph, records)


def test_multiple_outputs_preserve_shapes_and_dtypes():
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_op(torch.ops.ttrt_fused_op_test.multiple.default):
                full = torch.relu(x)
                rows = x.sum(dim=-1).to(torch.float64)
            return full, rows

    x = torch.randn(2, 3)
    graph = _capture(Model(), x)
    actual = graph(x)
    torch.testing.assert_close(actual, Model()(x))
    assert actual[0].shape == (2, 3) and actual[0].dtype == torch.float32
    assert actual[1].shape == (2,) and actual[1].dtype == torch.float64
    assert torch.ops.ttrt_fused_op_test.multiple.default in _targets(graph)


def test_input_and_output_casts_keep_the_custom_op():
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_op(torch.ops.ttrt_fused_op_test.upcast_relu.default):
                y = torch.relu(x.double())
            return y.float() + 1

    x = torch.randn(2, 3)
    graph = _capture(Model(), x)
    assert torch.ops.ttrt_fused_op_test.upcast_relu.default in _targets(graph)
    torch.testing.assert_close(graph(x), Model()(x))


def test_singleton_tuple_operator_preserves_tensor_output():
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_op(torch.ops.ttrt_fused_op_test.single_tuple.default):
                result = x.sin()
            return result

    x = torch.randn(2, 3)
    torch.testing.assert_close(_capture(Model(), x)(x), Model()(x))


@pytest.mark.parametrize(
    "op, message",
    [
        (torch.ops.aten.sin.default, "custom"),
        (torch.ops.ttrt_fused_op_invalid.scalar.default, "Tensor|tensor|schema"),
        (torch.ops.ttrt_fused_op_invalid.alias.default, "alias"),
        (torch.ops.ttrt_fused_op_invalid.mutate.default, "mutat"),
        (torch.ops.ttrt_fused_op_test.difference.default, "input|argument|schema"),
        (torch.ops.ttrt_fused_op_test.multiple.default, "output|return"),
    ],
)
def test_incompatible_operator_contract_is_rejected(op, message):
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_op(op):
                result = x.sin()
            return result

    with pytest.raises(
        (region.RegionError, RuntimeError, TypeError, ValueError), match=message
    ):
        _capture(Model(), torch.randn(2, 3))


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_fake_output_must_match_reference(mismatch):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            with region.fused_op(
                torch.ops.ttrt_fused_op_test.difference.default, inputs=(x, y)
            ):
                result = x - y
                result = (
                    result.sum(dim=-1)
                    if mismatch == "shape"
                    else result.to(torch.float64)
                )
            return result

    with pytest.raises(region.RegionError, match="output|fake"):
        _capture(Model(), torch.randn(2, 3), torch.randn(2, 3))


class _Gated(torch.nn.Module):
    def forward(self, up, gate):
        with region.execute_in_torch():
            up = up.cos()  # The earlier region replaces this listed tensor.
        # sigmoid reads gate before the product reads up.
        with region.fused_op(torch.ops.ttrt_fused_op_test.gated, inputs=(up, gate)):
            result = up * torch.sigmoid(gate)
        return result


class _Affine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.randn(3), requires_grad=False)

    def forward(self, x):
        # The transpose reads the weight before the product reads x.
        with region.fused_op(
            torch.ops.ttrt_fused_op_test.affine, inputs=(x, self.weight, self.bias)
        ):
            result = x @ self.weight.T + self.bias
        return result


@pytest.mark.parametrize(
    "model, op, shapes",
    [
        (_Gated, torch.ops.ttrt_fused_op_test.gated.default, [(2, 3), (2, 3)]),
        (_Affine, torch.ops.ttrt_fused_op_test.affine.default, [(4, 3)]),
    ],
)
def test_inputs_bind_in_listed_order_not_traced_order(model, op, shapes):
    inputs = [torch.randn(shape) for shape in shapes]
    model = model().eval()
    graph = _capture(model, *inputs)
    assert op in _targets(graph)
    torch.testing.assert_close(graph(*inputs), model(*inputs))


def test_inputs_sharing_storage_are_copied_without_capture_metadata():
    class Halves(torch.nn.Module):
        def forward(self, x):
            first, second = x.chunk(2)
            with region.fused_op(_difference, inputs=(first, second)):
                result = first - second
            return result

    x = torch.randn(4, 3)
    graph = _capture(Halves(), x)
    # The operator reads a copy of the second half, which keeps no capture mark.
    assert torch.ops.aten.clone.default in _targets(graph)
    assert not any(
        _ALIAS_COPY_KEY in node.meta.get("custom", {})
        for child in graph.modules()
        if isinstance(child, torch.fx.GraphModule)
        for node in child.graph.nodes
    )
    torch.testing.assert_close(graph(x), Halves()(x))


@pytest.mark.parametrize(
    "inputs, message",
    [
        (lambda up, gate, other: None, "reads 2 tensors .* pass inputs="),
        (lambda up, gate, other: (), "exactly once"),
        (lambda up, gate, other: (up,), "exactly once"),
        (lambda up, gate, other: (up, gate, up), "exactly once"),
        (lambda up, gate, other: (up, gate, other), "exactly once"),
    ],
    ids=["omitted", "empty", "missing", "repeated", "unread"],
)
def test_ambiguous_or_mismatched_inputs_are_rejected(inputs, message):
    class Model(torch.nn.Module):
        def forward(self, up, gate, other):
            with region.fused_op(
                torch.ops.ttrt_fused_op_test.gated, inputs=inputs(up, gate, other)
            ):
                result = up * torch.sigmoid(gate)
            return result + other

    with pytest.raises(region.RegionCaptureError, match=message):
        _capture(Model(), *(torch.randn(2, 3) for _ in range(3)))


@pytest.mark.parametrize("inputs", [torch.ones(2), [torch.ones(2), 1.0]])
def test_inputs_must_be_a_tuple_or_list_of_tensors(inputs):
    with pytest.raises(TypeError, match="tuple or list of tensors"):
        with region.fused_op(torch.ops.ttrt_fused_op_test.gated, inputs=inputs):
            pass


@pytest.mark.parametrize(
    "op", [_difference_fake, "ttrt_fused_op_test::difference"], ids=["fake", "name"]
)
def test_op_must_be_a_custom_op_function_or_operator(op):
    with pytest.raises(TypeError, match="custom_op function or torch.ops"):
        with region.fused_op(op):
            pass


def test_random_region_preserves_seeded_custom_op_execution():
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_op(torch.ops.ttrt_fused_op_test.noise.default):
                result = x + torch.rand_like(x)
            return result

    x = torch.randn(2, 3)
    state = torch.random.get_rng_state()
    graph = _capture(Model(), x)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.ops.ttrt_fused_op_test.noise.default in _targets(graph)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        expected = Model()(x)
        torch.manual_seed(17)
        torch.testing.assert_close(graph(x), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape_only", [False, True], ids=["capture", "decomposition"])
def test_dead_random_region_is_rejected(shape_only):
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_op(torch.ops.ttrt_fused_op_test.noise.default):
                result = x + torch.rand_like(x)
            # Torch-TensorRT decomposes full_like into a full of result's shape.
            return torch.full_like(result, 3.0) + x if shape_only else x * 2

    with pytest.raises(
        region.RegionCaptureError, match="random.*live output; 'region#0' is dead"
    ):
        compile_regions(Model().eval(), [torch.randn(2, 3, device="cuda")])


@pytest.mark.parametrize(
    "executorch", [False, True], ids=["exported_program", "executorch"]
)
def test_export_retains_custom_op(executorch):
    first, second = torch.randn(2, 3), torch.randn(2, 3)
    graph = _capture(_Difference(), second, first)
    if executorch:
        pytest.importorskip("executorch.exir")
        import torch_tensorrt.executorch as tte

        program = tte.export(graph).exported_program()
    else:
        artifact = io.BytesIO()
        torch.export.save(export(graph, use_legacy_exporter=True), artifact)
        artifact.seek(0)
        program = torch.export.load(artifact)
    assert any(
        "ttrt_fused_op_test" in str(target) and "difference" in str(target)
        for target in _targets(program.graph_module)
    )
    _calls.clear()
    torch.testing.assert_close(
        program.module()(second, first), _Difference()(second, first)
    )
    assert _calls == ["difference"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_fast_partitioner", [True, False])
def test_custom_op_stays_between_tensorrt_engines(use_fast_partitioner, monkeypatch):
    def unexpected_converter(*args, **kwargs):
        raise AssertionError("fused_op was incorrectly sent to TensorRT")

    monkeypatch.setitem(
        DYNAMO_ATEN_CONVERTERS,
        torch.ops.ttrt_fused_op_test.difference.default,
        [ConverterSupport(unexpected_converter)],
    )

    class Model(torch.nn.Module):
        def forward(self, x, bias):
            before = x + 1
            with region.fused_op(
                torch.ops.ttrt_fused_op_test.difference.default, inputs=(before, bias)
            ):
                result = before - bias
            return result * 2

    inputs = [torch.randn(2, 8, device="cuda") for _ in range(2)]
    model = Model().eval()
    with torch.inference_mode():
        compiled = compile_regions(
            model, inputs, use_fast_partitioner=use_fast_partitioner
        )
        assert sum(isinstance(m, TorchTensorRTModule) for m in compiled.modules()) == 2
        assert torch.ops.ttrt_fused_op_test.difference.default in _targets(compiled)
        audit_region_placement(compiled, get_region_records(compiled))
        _calls.clear()
        torch.testing.assert_close(compiled(*inputs), model(*inputs))
        assert _calls == ["difference"]
