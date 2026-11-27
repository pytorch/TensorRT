# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from unittest import mock

import pytest
import torch
from torch.fx.passes.shape_prop import ShapeProp
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo._compiler import compile_module
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.region_compilation import (
    RegionRecord,
    attach_region_records,
    audit_region_placement,
    get_region_records,
    is_torch_region_node,
    validate_region_settings,
)
from torch_tensorrt.region import RegionError


def make_region_graph(
    *, neighbors=True, frozen_state=False, constant_only=False, tuple_output=False
):
    body_graph = torch.fx.Graph()
    body_input = body_graph.placeholder("x")
    if frozen_state:
        weight = body_graph.placeholder("weight")
        body_input = body_graph.call_function(
            torch.ops.aten.add.Tensor, (body_input, weight)
        )
    body_output = body_graph.call_function(torch.ops.aten.relu.default, (body_input,))
    body_graph.output((body_output,) if tuple_output else body_output)
    body = torch.fx.GraphModule(torch.nn.Module(), body_graph)
    body.meta["torch_tensorrt_region_id"] = "test#0"

    root = torch.nn.Module()
    root.add_module("region_body", body)
    if frozen_state or constant_only:
        root.register_buffer("_frozen_param0", torch.ones(2, 3))
    graph = torch.fx.Graph()
    x = graph.get_attr("_frozen_param0") if constant_only else graph.placeholder("x")
    before = graph.call_function(torch.ops.aten.relu.default, (x,)) if neighbors else x
    args = (before,)
    if frozen_state:
        args += (graph.get_attr("_frozen_param0"),)
    region = graph.call_module("region_body", args)
    region.meta["torch_tensorrt_region"] = {"id": "test#0", "route": "torch"}
    value = (
        graph.call_function(operator.getitem, (region, 0)) if tuple_output else region
    )
    result = (
        graph.call_function(torch.ops.aten.sigmoid.default, (value,))
        if neighbors
        else value
    )
    graph.output(result)
    gm = torch.fx.GraphModule(root, graph)
    ShapeProp(gm).propagate(*(() if constant_only else (torch.randn(2, 3),)))
    records = [
        RegionRecord(
            id="test#0",
            name="test",
            child_target="region_body",
            input_names=tuple(
                node.name for node in body.graph.nodes if node.op == "placeholder"
            ),
            output_names=(body_output.name,),
            reference_graph=str(body.graph),
        )
    ]
    attach_region_records(gm, records)
    return gm


@pytest.fixture(autouse=True)
def preserve_converter_settings():
    previous = CONVERTERS.compilation_settings
    previous_targets = CONVERTERS.disallowed_targets
    CONVERTERS.set_compilation_settings(
        CompilationSettings(offload_module_to_cpu=False)
    )
    yield
    CONVERTERS.compilation_settings = previous
    CONVERTERS.set_disallowed_targets(previous_targets)


def test_region_counts_as_unsupported_computation():
    gm = make_region_graph()
    supported, total, overview = partitioning.get_graph_converter_support_overview(
        gm, set()
    )
    assert (supported, total) == (2, 3)
    assert overview.fallback_reasons["region_body"] == {
        "explicit execute_in_torch region"
    }


@pytest.mark.parametrize("tuple_output", [False, True])
def test_constant_input_region_is_not_folded(tuple_output):
    gm = make_region_graph(
        neighbors=False, constant_only=True, tuple_output=tuple_output
    )
    expected = gm()
    with mock.patch.object(
        gm.region_body,
        "forward",
        side_effect=AssertionError("region executed while folding"),
    ):
        result = constant_fold(gm, CompilationSettings(offload_module_to_cpu=False))
    assert sum(is_torch_region_node(result, node) for node in result.graph.nodes) == 1
    audit_region_placement(result, get_region_records(result))
    torch.testing.assert_close(result(), expected)


def test_fast_full_support_shortcut_cannot_absorb_region():
    with pytest.raises(ValueError, match="cannot bypass an execute_in_torch region"):
        partitioning.fast_partition(make_region_graph(), assume_full_support=True)


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
def test_compile_dryrun_preserves_regions_and_live_parent_state(use_fast_partitioner):
    gm = make_region_graph(frozen_state=True)
    inputs = torch.randn(2, 3)
    expected = gm(inputs)
    settings = CompilationSettings(
        min_block_size=1,
        use_fast_partitioner=use_fast_partitioner,
        offload_module_to_cpu=False,
        dryrun=True,
    )
    with mock.patch("torch_tensorrt.dynamo._compiler.convert_module") as convert:
        result = compile_module(gm, [], settings=settings)
        convert.assert_not_called()
    assert get_region_records(result)
    audit_region_placement(result, get_region_records(result))
    torch.testing.assert_close(result(inputs), expected)


def test_route_specific_conflicts_name_the_conflicting_region():
    gm = make_region_graph()
    eager = get_region_records(gm)[0]
    plugin = RegionRecord(id="test#1", name="test", child_target="", route="plugin")
    hardware_compatible = CompilationSettings(hardware_compatible=True)
    # Regions kept in PyTorch run on any GPU the engine supports.
    validate_region_settings(hardware_compatible, [eager])
    # Each conflict names a region on its own route, even when another comes first.
    with pytest.raises(RegionError, match="fused_kernel region 'test#1' targets"):
        validate_region_settings(hardware_compatible, [eager, plugin])
    full_compilation = CompilationSettings(require_full_compilation=True)
    with pytest.raises(RegionError, match="region 'test#0' must run outside TensorRT"):
        validate_region_settings(full_compilation, [plugin, eager])


def make_plugin_graph():
    graph = torch.fx.Graph()
    # A converted operator stands in for the registered plugin operator.
    plugin = graph.call_function(torch.ops.aten.relu.default, (graph.placeholder("x"),))
    plugin.meta["torch_tensorrt_region"] = {"id": "test#0", "route": "plugin"}
    # Returning the plugin too gives its partition several outputs.
    graph.output(
        (plugin, graph.call_function(torch.ops.aten.sigmoid.default, (plugin,)))
    )
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    ShapeProp(gm).propagate(torch.randn(2, 3))
    record = RegionRecord(
        id="test#0",
        name="test",
        child_target="",
        route="plugin",
        plugin_op=torch.ops.aten.relu.default,
    )
    attach_region_records(gm, [record])
    return gm


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_block_size": 10, "use_fast_partitioner": True},
        {"min_block_size": 10, "use_fast_partitioner": False},
        # Nothing converts, so compile_module returns before partitioning.
        {
            "torch_executed_ops": {
                torch.ops.aten.relu.default,
                torch.ops.aten.sigmoid.default,
            }
        },
    ],
    ids=["fast", "global", "unpartitioned"],
)
def test_plugin_region_left_outside_tensorrt_is_rejected(overrides):
    message = "fused_kernel region 'test#0' must run in a TensorRT plugin"
    with pytest.raises(RegionError, match=message):
        compile_module(
            make_plugin_graph(),
            [],
            settings=CompilationSettings(offload_module_to_cpu=False, **overrides),
        )
