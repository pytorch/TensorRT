import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import DYNAMO_CONVERTERS as CONVERTERS
from torch_tensorrt.dynamo.partitioning._global_partitioner import (
    TorchTensorRTOperatorSupport,
)


def test_fallback_data_dependent_ops_setting_default():
    # Off by default; opt-in only.
    assert CompilationSettings().fallback_data_dependent_ops is False
    assert (
        CompilationSettings(
            fallback_data_dependent_ops=True
        ).fallback_data_dependent_ops
        is True
    )


def test_old_pickle_without_field_defaults_to_false():
    # A settings object serialized before fallback_data_dependent_ops existed (no
    # such key in the state) must restore to the default instead of raising.
    state = CompilationSettings().__dict__.copy()
    state.pop("fallback_data_dependent_ops", None)
    restored = CompilationSettings.__new__(CompilationSettings)
    restored.__setstate__(state)
    assert restored.fallback_data_dependent_ops is False


def _nonzero_node() -> torch.fx.Node:
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nonzero(x)

    ep = torch.export.export(Model(), (torch.tensor([0, 3, 0, 5]),))
    gm = ep.module()
    return next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.nonzero.default
    )


def test_output_allocator_node_falls_back_only_when_enabled():
    # nonzero's selected converter requires an output allocator, so the partitioner
    # marks the node unsupported only when the flag is on. This is decided per node
    # (via the selected converter), not by op target.
    node = _nonzero_node()
    original = CONVERTERS.compilation_settings
    try:
        CONVERTERS.set_compilation_settings(
            CompilationSettings(fallback_data_dependent_ops=False)
        )
        assert TorchTensorRTOperatorSupport._requires_output_allocator(node) is False
        CONVERTERS.set_compilation_settings(
            CompilationSettings(fallback_data_dependent_ops=True)
        )
        assert TorchTensorRTOperatorSupport._requires_output_allocator(node) is True
    finally:
        CONVERTERS.compilation_settings = original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_data_dependent_ops_conflicts_with_require_full_compilation():
    # Running data-dependent ops in PyTorch contradicts require_full_compilation.
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nonzero(x)

    inputs = (torch.tensor([0, 3, 0, 5, 7], dtype=torch.int32, device="cuda"),)
    ep = torch.export.export(Model().cuda(), inputs)
    with pytest.raises(ValueError):
        torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=list(inputs),
            min_block_size=1,
            fallback_data_dependent_ops=True,
            require_full_compilation=True,
            use_python_runtime=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fallback_data_dependent_ops_routes_output_allocator_op_to_torch():
    # End to end on GPU: with fallback_data_dependent_ops=True an output-allocator op
    # (nonzero) falls back to PyTorch instead of being absorbed into a TensorRT engine.
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nonzero(x)

    inputs = (torch.tensor([0, 3, 0, 5, 7], dtype=torch.int32, device="cuda"),)
    ep = torch.export.export(Model().cuda(), inputs)
    gm = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=list(inputs),
        min_block_size=1,
        fallback_data_dependent_ops=True,
        use_python_runtime=True,
    )
    targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
    assert torch.ops.aten.nonzero.default in targets
