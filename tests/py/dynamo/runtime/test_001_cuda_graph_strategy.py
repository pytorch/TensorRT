import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import CUDA_GRAPH_STRATEGY
from torch_tensorrt.dynamo._settings import CompilationSettings


class CudaGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def _compile_cpp(strategy):
    model = CudaGraphModel().eval().cuda()
    inputs = [torch.randn(2, 3, 16, 16).cuda()]
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float32},
        use_python_runtime=False,
        min_block_size=1,
        cuda_graph_strategy=strategy,
    )
    torch._dynamo.reset()
    return compiled, inputs


class TestCudaGraphStrategySettings(TestCase):
    """Setting-level validation that runs on every build (RTX and non-RTX)."""

    def test_default_value(self):
        settings = CompilationSettings()
        self.assertEqual(settings.cuda_graph_strategy, CUDA_GRAPH_STRATEGY)

    def test_settable_values(self):
        for value in ("disabled", "whole_graph_capture"):
            settings = CompilationSettings(cuda_graph_strategy=value)
            self.assertEqual(settings.cuda_graph_strategy, value)


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy is a TensorRT-RTX feature",
)
class TestCudaGraphStrategyCpp(TestCase):
    """End-to-end: compile + infer through the C++ runtime with each strategy."""

    def tearDown(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def test_disabled(self):
        compiled, inputs = _compile_cpp("disabled")
        y = compiled(*[inp.clone() for inp in inputs])
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_whole_graph_capture(self):
        compiled, inputs = _compile_cpp("whole_graph_capture")
        y = compiled(*[inp.clone() for inp in inputs])
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_whole_graph_capture_with_subgraph_cudagraphs(self):
        """Subgraph cudagraph mode + RTX strategy: RTX-native should take over without errors."""
        compiled, inputs = _compile_cpp("whole_graph_capture")
        torchtrt.runtime.set_cudagraphs_mode(True)
        y = compiled(*[inp.clone() for inp in inputs])
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_repeated_inference(self):
        """Repeated inference exercises the RTX-native capture/replay path."""
        compiled, inputs = _compile_cpp("whole_graph_capture")
        ref = compiled(*[inp.clone() for inp in inputs])
        for _ in range(4):
            out = compiled(*[inp.clone() for inp in inputs])
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.isfinite(out).all().item())


@unittest.skipIf(
    not ENABLED_FEATURES.torch_tensorrt_runtime,
    "C++ runtime is not available",
)
class TestCudaGraphStrategyInvalidValue(TestCase):
    """Invalid strategy names raise ValueError."""

    def test_invalid_strategy_raises(self):
        model = CudaGraphModel().eval().cuda()
        inputs = [torch.randn(2, 3, 16, 16).cuda()]
        with self.assertRaises((ValueError, RuntimeError)):
            torchtrt.compile(
                model,
                ir="dynamo",
                inputs=inputs,
                enabled_precisions={torch.float32},
                use_python_runtime=False,
                min_block_size=1,
                cuda_graph_strategy="not_a_real_strategy",
            )


if __name__ == "__main__":
    run_tests()
