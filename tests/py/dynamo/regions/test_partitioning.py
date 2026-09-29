# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from copy import deepcopy
from unittest import mock

import pytest
import torch
from torch.fx.passes.shape_prop import ShapeProp

from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo._compiler import compile_module
from torch_tensorrt.dynamo._DryRunTracker import parse_non_trt_nodes
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.partitioning._adjacency_partitioner import OpSupportTester
from torch_tensorrt.dynamo.partitioning._global_partitioner import (
    TorchTensorRTOperatorSupport,
)
from torch_tensorrt.dynamo.regions import (
    RegionRecord,
    attach_region_records,
    audit_region_placement,
    get_region_records,
    is_torch_region_node,
)


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
    body._torch_tensorrt_region_id = "test#0"

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
            reference=deepcopy(body),
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
    assert any("test#0" in entry for entry in parse_non_trt_nodes(gm))


@pytest.mark.parametrize(
    "support_type", [OpSupportTester, TorchTensorRTOperatorSupport]
)
def test_region_support_policy_precedes_converter_lookup(support_type):
    gm = make_region_graph()
    node = next(node for node in gm.graph.nodes if node.op == "call_module")
    support = support_type()
    assert not support.is_node_supported(dict(gm.named_modules()), node)
    assert support.unsupported_operators["region_body"] == 1
    assert support.fallback_reasons["region_body"] == {
        "explicit execute_in_torch region"
    }


@pytest.mark.parametrize(
    "partition", [partitioning.fast_partition, partitioning.global_partition]
)
def test_region_remains_outside_accelerated_children(partition):
    gm = make_region_graph()
    inputs = torch.randn(2, 3)
    expected = gm(inputs)
    records = get_region_records(gm)
    result, _ = partition(gm, min_block_size=1)
    audit_region_placement(result, records)
    assert (
        len([name for name, _ in result.named_children() if "_run_on_acc" in name]) == 2
    )
    torch.testing.assert_close(result(inputs), expected)


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


@pytest.mark.parametrize("neighbors", [False, True])
def test_full_compilation_rejected_before_shortcuts(neighbors):
    gm = make_region_graph(neighbors=neighbors)
    settings = CompilationSettings(
        require_full_compilation=True, offload_module_to_cpu=False
    )
    with mock.patch("torch_tensorrt.dynamo._compiler.convert_module") as convert:
        with pytest.raises(RuntimeError, match="require_full_compilation"):
            compile_module(gm, [], settings=settings)
        convert.assert_not_called()


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


@pytest.mark.parametrize("neighbors,min_block_size", [(False, 1), (True, 10)])
def test_compile_early_return_preserves_region(neighbors, min_block_size):
    gm = make_region_graph(neighbors=neighbors)
    settings = CompilationSettings(
        offload_module_to_cpu=False, min_block_size=min_block_size
    )
    inputs = torch.randn(2, 3)
    expected = gm(inputs)
    result = compile_module(gm, [], settings=settings)
    audit_region_placement(result, get_region_records(result))
    torch.testing.assert_close(result(inputs), expected)


def test_conflicting_dryrun_stops_before_partitioning():
    gm = make_region_graph()
    settings = CompilationSettings(
        require_full_compilation=True,
        enable_resource_partitioning=True,
        offload_module_to_cpu=False,
        dryrun=True,
    )
    with mock.patch(
        "torch_tensorrt.dynamo._compiler.partitioning.fast_partition"
    ) as split:
        result = compile_module(gm, [], settings=settings)
        split.assert_not_called()
    assert result is gm
    assert (
        "require_full_compilation" in result.meta["torch_tensorrt_region_conflicts"][0]
    )


def test_fast_partitioner_failure_retains_region_in_global_fallback():
    gm = make_region_graph()
    inputs = torch.randn(2, 3)
    expected = gm(inputs)
    settings = CompilationSettings(
        min_block_size=1, offload_module_to_cpu=False, dryrun=True
    )
    with mock.patch(
        "torch_tensorrt.dynamo._compiler.partitioning.fast_partition",
        side_effect=torch.fx.passes.splitter_base.FxNetSplitterInternalError(
            "exercise global fallback"
        ),
    ):
        result = compile_module(gm, [], settings=settings)
    assert settings.use_fast_partitioner
    audit_region_placement(result, get_region_records(result))
    torch.testing.assert_close(result(inputs), expected)
