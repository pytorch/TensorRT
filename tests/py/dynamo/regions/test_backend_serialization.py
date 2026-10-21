# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""A saved region must not silently lose its requested execution backend."""

from unittest import mock

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo._exporter import (
    export,
    inline_torch_region_modules,
    transform,
)
from torch_tensorrt.dynamo.lowering._buffer_lifting import erase_export_guards
from torch_tensorrt.dynamo.regions import (
    attach_region_records,
    audit_region_placement,
    get_region_records,
    materialize_regions,
    normalize_region_scopes,
)
from torch_tensorrt.region._session import RegionCompilationSession

pytestmark = pytest.mark.unit


class _InductorRegion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.ones(3))

    def forward(self, x):
        with region.execute_in_torch(name="activation", backend="inductor"):
            y = torch.relu(x + self.bias)
        return y * 2


@pytest.fixture
def source():
    model, x = _InductorRegion().eval(), torch.randn(2, 3)
    with RegionCompilationSession(strict=True):
        ep = torch.export.export(model, (x,), strict=True)
        ep = normalize_region_scopes(ep)
        gm = materialize_regions(ep.run_decompositions().module())
    gm = erase_export_guards(gm)
    gm.recompile()
    return gm, x


def _snapshot(gm):
    """Capture graph, backend metadata and state identity without cloning."""
    return (
        {
            name: str(child.graph)
            for name, child in gm.named_modules()
            if isinstance(child, torch.fx.GraphModule)
        },
        {name: id(value) for name, value in gm.named_buffers()},
        [
            (record.id, record.backend, record.owner)
            for record in get_region_records(gm)
        ],
    )


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "fallback_child"])
def test_save_rejects_inductor_before_writing_or_mutating(source, tmp_path, nested):
    gm, x = source
    if nested:
        root = torch.nn.Module()
        root.add_module("_run_on_gpu_0", gm)
        graph = torch.fx.Graph()
        arg = graph.placeholder("x")
        graph.output(graph.call_module("_run_on_gpu_0", (arg,)))
        parent = torch.fx.GraphModule(root, graph)
        attach_region_records(parent, get_region_records(gm))
        gm = parent
    audit_region_placement(gm, get_region_records(gm))
    before = _snapshot(gm)
    expected = gm(x)
    destination = tmp_path / "region.artifact"

    # Test both a new destination and an existing file that must not be overwritten.
    for existing in (False, True):
        if existing:
            destination.write_bytes(b"existing artifact")
        with pytest.raises(
            region.RegionError, match="backend='inductor'.*cannot preserve"
        ):
            torch_tensorrt.save(
                gm,
                str(destination),
                inputs=[x],
                output_format="exported_program",
                retrace=False,
                use_legacy_exporter=True,
            )
        if existing:
            assert destination.read_bytes() == b"existing artifact"
        else:
            assert not destination.exists()
        assert _snapshot(gm) == before
        audit_region_placement(gm, get_region_records(gm))
        torch.testing.assert_close(gm(x), expected)


@pytest.mark.parametrize("entrypoint", [export, transform, inline_torch_region_modules])
def test_export_helpers_reject_inductor_before_cloning_or_mutating(source, entrypoint):
    gm, x = source
    before = _snapshot(gm)
    expected = gm(x)
    with mock.patch(
        "torch_tensorrt.dynamo._exporter.copy.deepcopy",
        side_effect=AssertionError("Inductor region was cloned before rejection"),
    ):
        with pytest.raises(
            region.RegionError, match="backend='inductor'.*cannot preserve"
        ):
            entrypoint(gm)
    assert _snapshot(gm) == before
    audit_region_placement(gm, get_region_records(gm))
    torch.testing.assert_close(gm(x), expected)
