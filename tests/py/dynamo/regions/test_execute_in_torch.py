# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Start here: one model, then eager execution, region capture, and compilation.

The model computes relu(x + 1) * 2. Only relu is inside the PyTorch region.
Extended cases live in test_execute_in_torch_regressions.py.
"""

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo.regions import (
    get_region_records,
    materialize_torch_regions,
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.region._session import RegionCompilationSession


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        x = x + 1  # Outside: eligible for TensorRT.
        with region.execute_in_torch():  # A name is optional; none is needed here.
            x = torch.relu(x)  # Inside: must stay in PyTorch.
        return x * 2  # Outside: eligible for TensorRT.


def test_eager_execution():
    """Without compilation, the annotation does not change normal PyTorch."""
    x = torch.randn(2, 8)
    model = SimpleModel().eval()
    torch.testing.assert_close(model(x), torch.relu(x + 1) * 2)


def test_region_capture():
    """Extract relu into a child graph. No TensorRT engines are built yet."""
    x = torch.randn(2, 8)
    model = SimpleModel().eval()

    # These are the capture steps that the public compile API normally runs.
    with RegionCompilationSession(strict=True):
        exported = torch.export.export(model, (x,), strict=True)  # Record operations.
        exported = normalize_region_scopes(exported)  # Extract the marked region.
        exported = exported.run_decompositions()  # Lower operations.
        graph_model = materialize_torch_regions(exported.module())  # Child call.

    records = get_region_records(graph_model)
    assert len(records) == 1
    child = graph_model.get_submodule(records[0].child_target)

    # The relu moved into the child; it is no longer an operation in the parent.
    assert any(node.target == torch.ops.aten.relu.default for node in child.graph.nodes)
    assert not any(
        node.target == torch.ops.aten.relu.default for node in graph_model.graph.nodes
    )
    torch.testing.assert_close(graph_model(x), model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_fast_partitioner", [True, False], ids=["fast", "global"])
def test_tensorrt_compilation(use_fast_partitioner):
    """Build and run: TensorRT add -> PyTorch relu -> TensorRT multiply."""
    x = torch.randn(2, 8, device="cuda")
    model = SimpleModel().eval().cuda()
    compiled = torch_tensorrt.compile(
        model,
        inputs=[x],
        ir="dynamo",
        strict=True,
        offload_module_to_cpu=False,
        min_block_size=1,  # Allow the single add and multiply to become engines.
        use_fast_partitioner=use_fast_partitioner,
    )

    engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
    assert len(engines) == 2
    assert len(get_region_records(compiled)) == 1
    torch.testing.assert_close(compiled(x), model(x))
