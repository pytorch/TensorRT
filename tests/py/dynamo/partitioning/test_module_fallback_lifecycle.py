# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Release coverage for module fallback across runtime and persistence APIs."""

from unittest.mock import patch

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.dynamo.backend import backends


class _LinearBetweenEngines(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.sigmoid(self.linear(x * 0.5 + 0.25))


class _WeightedEngines(_LinearBetweenEngines):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.full((8,), 0.5))

    def forward(self, x):
        return torch.sigmoid(self.linear(x * self.scale + 0.25))


def _assert_linear_fallback(compiled):
    """Numerical agreement alone cannot establish that exclusion was honored."""
    assert any("_run_on_acc" in name for name, _ in compiled.named_modules())
    targets = {
        str(node.target)
        for module in compiled.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if node.op == "call_function"
    }
    assert targets & {
        "aten.linear.default",
        "aten.addmm.default",
        "aten.mm.default",
    }, f"The excluded Linear disappeared into TensorRT: {targets}"


def _graph_read_state(compiled):
    # Global partitioning can retain unused source parameters. Only the state
    # read by remaining FX operations must stay on the execution device.
    state = []
    for module in compiled.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op != "get_attr" or not node.users:
                continue
            value = module
            for name in str(node.target).split("."):
                value = getattr(value, name)
            if isinstance(value, torch.Tensor):
                state.append(value)
    return state


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@pytest.mark.parametrize(
    "use_python_runtime",
    [
        pytest.param(True, id="legacy_python_flag"),
        pytest.param(False, id="native_default"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@torch.no_grad()
def test_module_fallback_dynamic_runtime(
    use_fast_partitioner, use_python_runtime, dtype
):
    # The runtime flag is deprecated; exercise both legacy option values.
    # Actual runtime selection is automatic in current Torch-TensorRT.
    torch.manual_seed(17)
    model = _LinearBetweenEngines().eval().cuda().to(dtype)
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 8),
                opt_shape=(3, 8),
                max_shape=(7, 8),
                dtype=dtype,
            )
        ],
        enabled_precisions={dtype},
        min_block_size=1,
        torch_executed_modules=["torch.nn.modules.linear.Linear"],
        use_fast_partitioner=use_fast_partitioner,
        use_python_runtime=use_python_runtime,
        cache_built_engines=False,
    )
    _assert_linear_fallback(compiled)
    # Revisit earlier shapes to catch stale bindings or outputs after a resize.
    for batch in [1, 3, 7, 2, 1, 7]:
        sample = torch.randn(batch, 8, device="cuda", dtype=dtype)
        tolerance = 3e-3 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(
            compiled(sample), model(sample), rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@pytest.mark.parametrize("retrace", [True, False])
@torch.no_grad()
def test_module_fallback_save_load(tmp_path, use_fast_partitioner, retrace):
    torch.manual_seed(23)
    model = _LinearBetweenEngines().eval().cuda()
    sample = torch.randn(3, 8, device="cuda")
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[sample],
        min_block_size=1,
        torch_executed_modules=["torch.nn.modules.linear.Linear"],
        use_fast_partitioner=use_fast_partitioner,
        use_python_runtime=False,
        cache_built_engines=False,
    )
    _assert_linear_fallback(compiled)
    path = str(tmp_path / "module_fallback.ep")
    torch_tensorrt.save(compiled, path, arg_inputs=[sample], retrace=retrace)
    restored = torch_tensorrt.load(path).module()
    targets = {str(node.target) for node in restored.graph.nodes}
    assert any("execute_engine" in target for target in targets)
    assert targets & {"aten.linear.default", "aten.addmm.default", "aten.mm.default"}
    for _ in range(3):
        fresh_sample = torch.randn_like(sample)
        torch.testing.assert_close(
            restored(fresh_sample), model(fresh_sample), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@torch.no_grad()
def test_module_fallback_torch_compile(use_fast_partitioner):
    model = _LinearBetweenEngines().eval().cuda()
    sample = torch.randn(3, 8, device="cuda")
    compiled_graphs = []
    original_compile_module = backends.compile_module

    def capture_compile(*args, **kwargs):
        result = original_compile_module(*args, **kwargs)
        compiled_graphs.append(result)
        return result

    torch._dynamo.reset()
    try:
        with patch.object(backends, "compile_module", capture_compile):
            compiled = torch.compile(
                model,
                backend="torch_tensorrt",
                options={
                    "min_block_size": 1,
                    "torch_executed_modules": ["torch.nn.modules.linear.Linear"],
                    "use_fast_partitioner": use_fast_partitioner,
                    "pass_through_build_failures": True,
                    "cache_built_engines": False,
                },
            )
            torch.testing.assert_close(
                compiled(sample), model(sample), rtol=1e-4, atol=1e-4
            )
        assert compiled_graphs, "The TensorRT backend was never invoked"
        for graph in compiled_graphs:
            _assert_linear_fallback(graph)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@pytest.mark.parametrize("exclude_root", [True, False])
@torch.no_grad()
def test_module_fallback_cpu_offload(use_fast_partitioner, exclude_root):
    model = _WeightedEngines().eval().cuda()
    sample = torch.randn(3, 8, device="cuda")
    expected = model(sample)
    excluded = (
        f"{_WeightedEngines.__module__}.{_WeightedEngines.__qualname__}"
        if exclude_root
        else "torch.nn.modules.linear.Linear"
    )
    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[sample],
        min_block_size=1,
        torch_executed_modules=[excluded],
        use_fast_partitioner=use_fast_partitioner,
        offload_module_to_cpu=True,
        cache_built_engines=False,
    )
    if exclude_root:
        assert not any("_run_on_acc" in name for name, _ in compiled.named_modules())
    else:
        _assert_linear_fallback(compiled)
    torch.testing.assert_close(compiled(sample), expected, rtol=1e-4, atol=1e-4)
    state = _graph_read_state(compiled)
    assert state, "The test must exercise fallback weights"
    assert all(tensor.device.type == "cuda" for tensor in state)
    if not exclude_root:
        assert (
            model.scale.device.type == "cpu"
        ), "TensorRT-only weights must stay offloaded"


@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.refit, reason="Refit is unavailable"
)
@pytest.mark.parametrize("use_fast_partitioner", [True, False])
@pytest.mark.parametrize("exclude_root", [True, False])
@torch.no_grad()
def test_module_fallback_mutable_refit_with_offload(use_fast_partitioner, exclude_root):
    torch.manual_seed(31)
    model = _WeightedEngines().eval().cuda()
    sample = torch.randn(3, 8, device="cuda")
    expected = model(sample)
    excluded = (
        f"{_WeightedEngines.__module__}.{_WeightedEngines.__qualname__}"
        if exclude_root
        else "torch.nn.modules.linear.Linear"
    )
    mutable = torch_tensorrt.MutableTorchTensorRTModule(
        model,
        min_block_size=1,
        torch_executed_modules=[excluded],
        use_fast_partitioner=use_fast_partitioner,
        offload_module_to_cpu=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )
    torch.testing.assert_close(mutable(sample), expected, rtol=1e-4, atol=1e-4)
    if exclude_root:
        assert not any("_run_on_acc" in name for name, _ in mutable.gm.named_modules())
    else:
        _assert_linear_fallback(mutable.gm)
    layout = {
        name: type(module)
        for name, module in mutable.gm.named_children()
        if "_run_on" in name
    }
    replacement = _WeightedEngines().eval().cuda()
    replacement.scale.uniform_(0.8, 1.2)
    new_expected = replacement(sample)
    assert not torch.allclose(expected, new_expected)
    mutable.load_state_dict(replacement.state_dict())
    torch.testing.assert_close(mutable(sample), new_expected, rtol=1e-4, atol=1e-4)
    assert layout == {
        name: type(module)
        for name, module in mutable.gm.named_children()
        if "_run_on" in name
    }
    state = _graph_read_state(mutable.gm)
    assert state and all(tensor.device.type == "cuda" for tensor in state)
    if not exclude_root:
        _assert_linear_fallback(mutable.gm)
        assert mutable.original_model.scale.device.type == "cpu"
