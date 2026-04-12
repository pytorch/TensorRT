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
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
class TestDynamicShapesKernelStrategySetup(TestCase):
    """Tests that the dynamic shapes kernel specialization strategy is correctly applied."""

    def test_default_strategy_is_lazy(self):
        import tensorrt as trt

        compiled = _compile_simple()
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod, "No PythonTorchTensorRTModule found")
        self.assertIsNotNone(mod.runtime_config, "runtime_config should be set for RTX")
        self.assertEqual(
            mod.runtime_config.dynamic_shapes_kernel_specialization_strategy,
            trt.DynamicShapesKernelSpecializationStrategy.LAZY,
        )

    def test_eager_strategy(self):
        import tensorrt as trt

        compiled = _compile_simple(
            dynamic_shapes_kernel_specialization_strategy="eager"
        )
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertEqual(
            mod.runtime_config.dynamic_shapes_kernel_specialization_strategy,
            trt.DynamicShapesKernelSpecializationStrategy.EAGER,
        )

    def test_none_strategy(self):
        import tensorrt as trt

        compiled = _compile_simple(dynamic_shapes_kernel_specialization_strategy="none")
        mod = _find_python_trt_module(compiled)
        self.assertIsNotNone(mod)
        self.assertEqual(
            mod.runtime_config.dynamic_shapes_kernel_specialization_strategy,
            trt.DynamicShapesKernelSpecializationStrategy.NONE,
        )

    def test_context_created_with_each_strategy(self):
        for strategy in ("lazy", "eager", "none"):
            with self.subTest(strategy=strategy):
                compiled = _compile_simple(
                    dynamic_shapes_kernel_specialization_strategy=strategy
                )
                mod = _find_python_trt_module(compiled)
                self.assertIsNotNone(
                    mod.context, f"Execution context should be created for {strategy}"
                )
                # Test inference with multiple dynamic batch sizes
                for bs in (1, 2, 4):
                    output = compiled(torch.randn(bs, 3).cuda())
                    self.assertEqual(output.shape, (bs, 3))

    def test_setting_in_compilation_settings(self):
        for strategy in ("lazy", "eager", "none"):
            settings = CompilationSettings(
                dynamic_shapes_kernel_specialization_strategy=strategy
            )
            self.assertEqual(
                settings.dynamic_shapes_kernel_specialization_strategy, strategy
            )

    def test_default_compilation_settings(self):
        settings = CompilationSettings()
        self.assertEqual(settings.dynamic_shapes_kernel_specialization_strategy, "lazy")


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "This test verifies standard TRT behavior (non-RTX)",
)
class TestDynamicShapesKernelStrategyNonRTX(TestCase):
    """Tests that the setting is ignored on non-RTX builds."""

    def test_setting_ignored_on_non_rtx(self):
        compiled = _compile_simple(
            dynamic_shapes_kernel_specialization_strategy="eager"
        )
        mod = _find_python_trt_module(compiled)
        if mod is not None:
            self.assertIsNone(
                mod.runtime_config,
                "runtime_config should be None for standard TRT",
            )
        # Inference should still work
        output = compiled(torch.randn(2, 3).cuda())
        self.assertEqual(output.shape, (2, 3))


if __name__ == "__main__":
    run_tests()
