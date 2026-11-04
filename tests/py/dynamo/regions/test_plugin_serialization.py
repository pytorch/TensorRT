# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""A compiled region plugin is self-contained in a serialized TensorRT engine."""

import importlib.util
import textwrap
from pathlib import Path

import pytest
import torch

import torch_tensorrt

from ._utils import compile_regions, run_python


@pytest.fixture
def plugin_example():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")
    if not torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
        pytest.skip("TensorRT QDP plugins are required")
    example = (
        Path(__file__).resolve().parents[4] / "examples" / "dynamo" / "fused_kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "region_kernel_serialization", example
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_export_loads_in_fresh_process_without_kernel_registration(
    tmp_path, plugin_example
):
    x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(2, 8) - 8
    inputs = (x, torch.ones_like(x), torch.full_like(x, 2))
    compiled = compile_regions(plugin_example.Model().eval(), inputs)
    destination = tmp_path / "plugin_region.pt2"
    torch_tensorrt.save(
        compiled,
        str(destination),
        output_format="exported_program",
        use_legacy_exporter=True,
        retrace=False,
    )
    with torch.no_grad():
        torch.testing.assert_close(compiled(*inputs), plugin_example.Model()(*inputs))
    child = textwrap.dedent("""
        import sys
        import torch
        import torch_tensorrt

        assert not any(
            name.startswith("torch_tensorrt_region_fused::")
            for name in torch._C._dispatch_get_all_op_names()
        ), "The region kernel must not be registered in this fresh process"
        loaded = torch_tensorrt.load(sys.argv[1])
        x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(2, 8) - 8
        inputs = (x, torch.ones_like(x), torch.full_like(x, 2))
        with torch.no_grad():
            torch.testing.assert_close(
                loaded.module()(*inputs), torch.relu(x + 2) * 2 - 0.5
            )
        print("fresh-process-plugin-ok")
        """)
    assert "fresh-process-plugin-ok" in run_python("-c", child, destination)


def test_dryrun_plugin_save_rejected_before_file_or_graph_changes(
    tmp_path, monkeypatch, plugin_example
):
    from torch_tensorrt.dynamo import _exporter
    from torch_tensorrt.region import RegionError

    inputs = [torch.randn(2, 8, device="cuda") for _ in range(3)]
    compiled = compile_regions(plugin_example.Model().eval(), inputs, dryrun=True)
    destination = tmp_path / "existing.pt2"
    destination.write_bytes(b"untouched")
    graph = str(compiled.graph)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Unresolved plugin graph reached exporter deepcopy")

    monkeypatch.setattr(_exporter.copy, "deepcopy", unexpected_copy)
    with pytest.raises(RegionError, match="requires a built TensorRT engine"):
        torch_tensorrt.save(
            compiled,
            str(destination),
            output_format="exported_program",
            use_legacy_exporter=True,
            retrace=False,
        )
    with pytest.raises(RegionError, match="requires a built TensorRT engine"):
        _exporter.export(compiled)
    assert destination.read_bytes() == b"untouched"
    assert str(compiled.graph) == graph
