import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


def _compile_simple(**extra_kwargs):
    """Helper: compile SimpleModel with dynamic shapes and Python runtime."""
    model = SimpleModel().eval().cuda()
    inputs = [
        torchtrt.Input(
            min_shape=(1, 3),
            opt_shape=(2, 3),
            max_shape=(4, 3),
            dtype=torch.float32,
        )
    ]
    kwargs = {
        "ir": "dynamo",
        "inputs": inputs,
        "enabled_precisions": {torch.float32},
        "use_python_runtime": True,
        "min_block_size": 1,
    }
    kwargs.update(extra_kwargs)
    compiled = torchtrt.compile(model, **kwargs)
    torch._dynamo.reset()
    return compiled


def _find_python_trt_module(compiled):
    """Walk the compiled graph module to find PythonTorchTensorRTModule instances."""
    from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
        PythonTorchTensorRTModule,
    )

    for name, mod in compiled.named_modules():
        if isinstance(mod, PythonTorchTensorRTModule):
            return mod
    return None


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy requires TensorRT-RTX",
)
class TestCudaGraphStrategySetup(TestCase):
    """Tests that cuda_graph_strategy is correctly applied on TRT-RTX."""

    def test_default_strategy_is_disabled(self):
        import tensorrt as trt

        compiled = _compile_simple()
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod, "No PythonTorchTensorRTModule found")
        self.assertIsNotNone(mod.runtime_config, "runtime_config should be set for RTX")
        self.assertEqual(
            mod.runtime_config.cuda_graph_strategy,
            trt.CudaGraphStrategy.DISABLED,
        )

    def test_whole_graph_capture_strategy(self):
        import tensorrt as trt

        compiled = _compile_simple(cuda_graph_strategy="whole_graph_capture")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertEqual(
            mod.runtime_config.cuda_graph_strategy,
            trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE,
        )

    def test_rtx_native_flag_set(self):
        compiled = _compile_simple(cuda_graph_strategy="whole_graph_capture")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertTrue(mod._rtx_native_cudagraphs)

    def test_rtx_native_flag_disabled(self):
        compiled = _compile_simple(cuda_graph_strategy="disabled")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertFalse(mod._rtx_native_cudagraphs)

    def test_inference_with_each_strategy(self):
        for strategy in ("disabled", "whole_graph_capture"):
            with self.subTest(strategy=strategy):
                compiled = _compile_simple(cuda_graph_strategy=strategy)
                mod = _find_python_trt_module(compiled)
                self.assertIsNotNone(
                    mod.context,
                    f"Execution context should be created for {strategy}",
                )
                for bs in (1, 2, 4):
                    output = compiled(torch.randn(bs, 3).cuda())
                    self.assertEqual(output.shape, (bs, 3))

    def test_setting_in_compilation_settings(self):
        for strategy in ("disabled", "whole_graph_capture"):
            settings = CompilationSettings(cuda_graph_strategy=strategy)
            self.assertEqual(settings.cuda_graph_strategy, strategy)

    def test_default_compilation_settings(self):
        settings = CompilationSettings()
        self.assertEqual(settings.cuda_graph_strategy, "disabled")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy integration requires TensorRT-RTX",
)
class TestCudaGraphStrategyWithSubgraphCudagraphs(TestCase):
    """Tests integration with set_cudagraphs_mode()."""

    def setUp(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def test_rtx_native_bypasses_manual_capture(self):
        compiled = _compile_simple(cuda_graph_strategy="whole_graph_capture")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)

        torchtrt.runtime.set_cudagraphs_mode(True)

        # Run inference a few times to ensure capture would have happened
        for _ in range(3):
            compiled(torch.randn(2, 3).cuda())

        # Manual cudagraph should NOT have been recorded (RTX handles it natively)
        self.assertFalse(
            hasattr(mod, "cudagraph")
            and isinstance(mod.cudagraph, torch.cuda.CUDAGraph),
            "Manual CUDA graph should not be recorded when RTX native is active",
        )

    def test_subgraph_mode_always_uses_rtx_native(self):
        """Even with cuda_graph_strategy=disabled, SUBGRAPH mode on RTX
        should override to RTX-native because manual capture is not safe."""
        compiled = _compile_simple(cuda_graph_strategy="disabled")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        # Initially, _rtx_native_cudagraphs is False (disabled strategy)
        self.assertFalse(mod._rtx_native_cudagraphs)

        torchtrt.runtime.set_cudagraphs_mode(True)

        # Run inference — should trigger override to RTX-native
        for _ in range(3):
            compiled(torch.randn(2, 3).cuda())

        # Should have been overridden to RTX-native
        self.assertTrue(
            mod._rtx_native_cudagraphs,
            "RTX-native should be enabled automatically in SUBGRAPH mode",
        )
        # Manual cudagraph should NOT have been recorded
        self.assertFalse(
            hasattr(mod, "cudagraph")
            and isinstance(mod.cudagraph, torch.cuda.CUDAGraph),
            "Manual CUDA graph should not be recorded on RTX",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Monolithic capturability tests require TensorRT-RTX",
)
class TestMonolithicCapturability(TestCase):
    """Tests for _is_monolithic_capturable() and related logic."""

    def test_lazy_strategy_not_monolithic_capturable(self):
        """Lazy kernel specialization makes monolithic capture unsafe."""
        compiled = _compile_simple(
            cuda_graph_strategy="disabled",
            dynamic_shapes_kernel_specialization_strategy="lazy",
        )
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        stream = torch.cuda.Stream()
        self.assertFalse(mod._is_monolithic_capturable(stream))

    def test_eager_strategy_monolithic_capturable(self):
        """Eager strategy with capturable stream should be monolithic capturable."""
        compiled = _compile_simple(
            cuda_graph_strategy="disabled",
            dynamic_shapes_kernel_specialization_strategy="eager",
        )
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        stream = torch.cuda.Stream()
        # is_stream_capturable depends on engine properties.
        # With eager strategy, the strategy check passes.
        if mod.context.is_stream_capturable(stream.cuda_stream):
            self.assertTrue(mod._is_monolithic_capturable(stream))

    def test_none_strategy_monolithic_capturable(self):
        """None strategy (always fallback) should be monolithic capturable."""
        compiled = _compile_simple(
            cuda_graph_strategy="disabled",
            dynamic_shapes_kernel_specialization_strategy="none",
        )
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        stream = torch.cuda.Stream()
        if mod.context.is_stream_capturable(stream.cuda_stream):
            self.assertTrue(mod._is_monolithic_capturable(stream))


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Context recreation tests require TensorRT-RTX",
)
class TestContextRecreation(TestCase):
    """Tests for _enable_rtx_native_cudagraphs() context recreation."""

    def test_enable_rtx_native_recreates_context(self):
        """Calling _enable_rtx_native_cudagraphs recreates the execution context."""
        import tensorrt as trt

        compiled = _compile_simple(cuda_graph_strategy="disabled")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertFalse(mod._rtx_native_cudagraphs)

        old_context_id = id(mod.context)
        mod._enable_rtx_native_cudagraphs()

        self.assertTrue(mod._rtx_native_cudagraphs)
        self.assertNotEqual(
            id(mod.context),
            old_context_id,
            "Context should be recreated",
        )
        self.assertEqual(
            mod.runtime_config.cuda_graph_strategy,
            trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE,
        )

    def test_explicit_whole_graph_capture_no_override_needed(self):
        """With explicit whole_graph_capture, SUBGRAPH mode should not
        need to override (already RTX-native)."""
        compiled = _compile_simple(cuda_graph_strategy="whole_graph_capture")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertTrue(mod._rtx_native_cudagraphs)

        old_context_id = id(mod.context)

        torchtrt.runtime.set_cudagraphs_mode(True)
        compiled(torch.randn(2, 3).cuda())
        torchtrt.runtime.set_cudagraphs_mode(False)

        # Context should NOT have been recreated (was already RTX-native)
        self.assertEqual(
            id(mod.context),
            old_context_id,
            "Context should not be recreated if already RTX-native",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Cudagraph mode toggle tests require TensorRT-RTX",
)
class TestCudagraphModeToggle(TestCase):
    """Tests for toggling cudagraph mode with RTX-native."""

    def setUp(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def test_cudagraphs_off_after_rtx_native_override(self):
        """After RTX-native override, disabling cudagraphs should still
        produce correct results (RTX-native continues transparently)."""
        compiled = _compile_simple(cuda_graph_strategy="disabled")

        torchtrt.runtime.set_cudagraphs_mode(True)
        compiled(torch.randn(2, 3).cuda())  # triggers override

        torchtrt.runtime.set_cudagraphs_mode(False)

        # Should still work -- RTX-native is transparent
        for bs in (1, 2, 4):
            output = compiled(torch.randn(bs, 3).cuda())
            self.assertEqual(output.shape, (bs, 3))

    def test_no_cudagraphs_with_whole_graph_capture(self):
        """With cuda_graph_strategy='whole_graph_capture' but no
        set_cudagraphs_mode, RTX-native runs transparently."""
        compiled = _compile_simple(cuda_graph_strategy="whole_graph_capture")
        mod = _find_python_trt_module(compiled)
        self.assertTrue(mod._rtx_native_cudagraphs)

        # No set_cudagraphs_mode(True) -- RTX-native still active transparently
        for bs in (1, 2, 4):
            output = compiled(torch.randn(bs, 3).cuda())
            self.assertEqual(output.shape, (bs, 3))

    def test_toggle_on_off_on(self):
        """Toggle cudagraphs on -> off -> on, verify correctness each time."""
        compiled = _compile_simple(cuda_graph_strategy="disabled")
        inp = torch.randn(2, 3).cuda()

        # Phase 1: on
        torchtrt.runtime.set_cudagraphs_mode(True)
        out1 = compiled(inp)
        self.assertEqual(out1.shape, (2, 3))

        # Phase 2: off
        torchtrt.runtime.set_cudagraphs_mode(False)
        out2 = compiled(inp)
        self.assertEqual(out2.shape, (2, 3))

        # Phase 3: on again
        torchtrt.runtime.set_cudagraphs_mode(True)
        out3 = compiled(inp)
        self.assertEqual(out3.shape, (2, 3))


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "This test verifies standard TRT behavior (non-RTX)",
)
class TestCudaGraphStrategyNonRTX(TestCase):
    """Tests that the setting is ignored on non-RTX builds."""

    def test_setting_ignored_on_non_rtx(self):
        compiled = _compile_simple(cuda_graph_strategy="whole_graph_capture")
        mod = _find_python_trt_module(compiled)
        if mod is not None:
            self.assertIsNone(
                mod.runtime_config,
                "runtime_config should be None for standard TRT",
            )
            self.assertFalse(mod._rtx_native_cudagraphs)
        output = compiled(torch.randn(2, 3).cuda())
        self.assertEqual(output.shape, (2, 3))


if __name__ == "__main__":
    run_tests()
