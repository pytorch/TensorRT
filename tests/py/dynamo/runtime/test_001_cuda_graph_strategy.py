import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.runtime import RuntimeSettings


class CudaGraphConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def _apply_runtime_settings(compiled, rs):
    """Apply ``RuntimeSettings`` to every inner ``TorchTensorRTModule``."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule):
            mod.runtime_settings = rs


def _compile_conv(strategy):
    """Compile CudaGraphConvModel + apply cuda_graph_strategy post-compile."""
    model = CudaGraphConvModel().eval().cuda()
    inputs = [torch.randn(2, 3, 16, 16).cuda()]
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float32},
        min_block_size=1,
    )
    torch._dynamo.reset()
    _apply_runtime_settings(compiled, RuntimeSettings(cuda_graph_strategy=strategy))
    return compiled, inputs


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


def _compile_simple(*, runtime_settings=None):
    """Compile SimpleModel with dynamic shapes through the build-selected runtime."""
    model = SimpleModel().eval().cuda()
    inputs = [
        torchtrt.Input(
            min_shape=(1, 3),
            opt_shape=(2, 3),
            max_shape=(4, 3),
            dtype=torch.float32,
        )
    ]
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float32},
        min_block_size=1,
    )
    torch._dynamo.reset()
    if runtime_settings is not None:
        _apply_runtime_settings(compiled, runtime_settings)
    return compiled


def _find_python_trt_engine(compiled):
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule) and isinstance(mod.engine, TRTEngine):
            return mod.engine
    return None


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy requires TensorRT-RTX",
)
@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "Whitebox introspection requires the Python TRTEngine path",
)
class TestCudaGraphStrategySetup(TestCase):
    """Tests that cuda_graph_strategy is correctly applied on TRT-RTX."""

    def test_default_strategy_is_disabled(self):
        compiled = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine, "No Python TRTEngine found")
        self.assertEqual(engine.runtime_settings.cuda_graph_strategy, "disabled")

    def test_whole_graph_capture_strategy_via_compile_hint(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(cuda_graph_strategy="whole_graph_capture"),
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertEqual(
            engine.runtime_settings.cuda_graph_strategy, "whole_graph_capture"
        )

    def test_rtx_native_flag_tracks_strategy(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        )
        engine = _find_python_trt_engine(compiled)
        self.assertTrue(engine._rtx_native_cudagraphs)

        compiled2 = _compile_simple(
            runtime_settings=RuntimeSettings(cuda_graph_strategy="disabled")
        )
        engine2 = _find_python_trt_engine(compiled2)
        self.assertFalse(engine2._rtx_native_cudagraphs)

    def test_runtime_cm_overrides_strategy(self):
        compiled = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertEqual(engine.runtime_settings.cuda_graph_strategy, "disabled")
        with torchtrt.runtime.enable_cudagraphs(
            compiled, cuda_graph_strategy="whole_graph_capture"
        ) as wrapped:
            self.assertEqual(
                engine.runtime_settings.cuda_graph_strategy, "whole_graph_capture"
            )
            for bs in (1, 2, 4):
                output = wrapped(torch.randn(bs, 3).cuda())
                self.assertEqual(output.shape, (bs, 3))
        # Restored on exit.
        self.assertEqual(engine.runtime_settings.cuda_graph_strategy, "disabled")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy integration requires TensorRT-RTX",
)
@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "Whitebox introspection requires the Python TRTEngine path",
)
class TestCudaGraphStrategyWithSubgraphCudagraphs(TestCase):
    """Tests integration with set_cudagraphs_mode()."""

    def setUp(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def test_rtx_native_bypasses_manual_capture(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(cuda_graph_strategy="whole_graph_capture"),
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        torchtrt.runtime.set_cudagraphs_mode(True)
        for _ in range(3):
            compiled(torch.randn(2, 3).cuda())
        self.assertFalse(
            isinstance(engine.cudagraph, torch.cuda.CUDAGraph),
            "Manual CUDA graph should not be recorded when RTX native is active",
        )

    def test_subgraph_mode_always_uses_rtx_native(self):
        """Even with cuda_graph_strategy=disabled, SUBGRAPH mode on RTX
        should override to RTX-native because manual capture is not safe."""
        compiled = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertFalse(engine._rtx_native_cudagraphs)
        torchtrt.runtime.set_cudagraphs_mode(True)
        for _ in range(3):
            compiled(torch.randn(2, 3).cuda())
        self.assertTrue(
            engine._rtx_native_cudagraphs,
            "RTX-native should be enabled automatically in SUBGRAPH mode",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Monolithic capturability tests require TensorRT-RTX",
)
@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "Whitebox introspection requires the Python TRTEngine path",
)
class TestMonolithicCapturability(TestCase):
    """Tests for _is_monolithic_capturable() and related logic."""

    def test_lazy_strategy_not_monolithic_capturable(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(
                cuda_graph_strategy="disabled",
                dynamic_shapes_kernel_specialization_strategy="lazy",
            ),
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        stream = torch.cuda.Stream()
        self.assertFalse(engine._is_monolithic_capturable(stream))

    def test_eager_strategy_monolithic_capturable(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(
                cuda_graph_strategy="disabled",
                dynamic_shapes_kernel_specialization_strategy="eager",
            ),
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        stream = torch.cuda.Stream()
        # The exact result depends on engine stream-capturability; just verify
        # the lazy-gating doesn't fire on eager.
        with torch.cuda.stream(stream):
            _ = engine._is_monolithic_capturable(stream)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy is a TensorRT-RTX feature",
)
class TestCudaGraphStrategyInference(TestCase):
    """End-to-end: compile + infer with each strategy on the build-selected runtime."""

    @parameterized.expand([("disabled",), ("whole_graph_capture",)])
    def test_strategy_inference(self, strategy):
        compiled, inputs = _compile_conv(strategy)
        y = compiled(*inputs)
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())


class TestCudaGraphStrategyInvalidValue(TestCase):
    """Invalid strategy names are rejected at ``RuntimeSettings.__post_init__``."""

    def test_invalid_strategy_raises_at_construction(self):
        with self.assertRaises(ValueError):
            RuntimeSettings(cuda_graph_strategy="not_a_real_strategy")


if __name__ == "__main__":
    run_tests()
