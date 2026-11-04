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


@torch.library.custom_op("ttrt_fused_op_test::noise", mutates_args=())
def _noise(x: torch.Tensor) -> torch.Tensor:
    return x + torch.rand_like(x)


@_noise.register_fake
def _noise_fake(x):
    return torch.empty_like(x)


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
    def forward(self, second, first):
        with region.fused_op(torch.ops.ttrt_fused_op_test.difference, name="subtract"):
            result = first - second
        return result.sin()


def test_first_use_order_and_actual_custom_op_execution():
    first, second = torch.randn(2, 3), torch.randn(2, 3)
    model = _Difference().eval()
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
            with region.fused_op(torch.ops.ttrt_fused_op_test.difference.default):
                result = x - y
                result = (
                    result.sum(dim=-1)
                    if mismatch == "shape"
                    else result.to(torch.float64)
                )
            return result

    with pytest.raises(region.RegionError, match="output|fake"):
        _capture(Model(), torch.randn(2, 3), torch.randn(2, 3))


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
            with region.fused_op(torch.ops.ttrt_fused_op_test.difference.default):
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
