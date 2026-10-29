# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""A saved region must not silently lose its requested execution backend."""

from unittest import mock

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo._exporter import export, transform
from torch_tensorrt.dynamo.region_compilation import (
    audit_region_placement,
    get_region_records,
)

from ._utils import capture_regions, graph_snapshot

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
    x = torch.randn(2, 3)
    return capture_regions(_InductorRegion(), x), x


def test_save_rejects_inductor_before_writing_or_mutating(source, tmp_path):
    gm, x = source
    audit_region_placement(gm, get_region_records(gm))
    before = graph_snapshot(gm)
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
        assert graph_snapshot(gm) == before
        audit_region_placement(gm, get_region_records(gm))
        torch.testing.assert_close(gm(x), expected)


@pytest.mark.parametrize("entrypoint", [export, transform])
def test_export_helpers_reject_inductor_before_cloning_or_mutating(source, entrypoint):
    gm, x = source
    before = graph_snapshot(gm)
    expected = gm(x)
    with mock.patch(
        "torch_tensorrt.dynamo._exporter.copy.deepcopy",
        side_effect=AssertionError("Inductor region was cloned before rejection"),
    ):
        with pytest.raises(
            region.RegionError, match="backend='inductor'.*cannot preserve"
        ):
            entrypoint(gm)
    assert graph_snapshot(gm) == before
    audit_region_placement(gm, get_region_records(gm))
    torch.testing.assert_close(gm(x), expected)
