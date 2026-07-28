import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.runtime import RuntimeSettings

_STRATEGIES = [("lazy",), ("eager",), ("none",)]


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


class DynamicConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


def _compile_dynamic_conv(strategy):
    """Compile DynamicConvModel with the given strategy as a compile-time hint."""
    model = DynamicConvModel().eval().cuda()
    inp = torchtrt.Input(
        min_shape=(1, 3, 16, 16),
        opt_shape=(2, 3, 16, 16),
        max_shape=(4, 3, 16, 16),
        dtype=torch.float32,
    )
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=[inp],
        enabled_precisions={torch.float32},
        min_block_size=1,
    )
    torch._dynamo.reset()
    _apply_runtime_settings(
        compiled,
        RuntimeSettings(dynamic_shapes_kernel_specialization_strategy=strategy),
    )
    return compiled


def _apply_runtime_settings(compiled, rs):
    """Apply ``RuntimeSettings`` to every inner ``TorchTensorRTModule``."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule):
            mod.runtime_settings = rs


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
        min_block_size=1,
    )
    torch._dynamo.reset()
    if runtime_settings is not None:
        _apply_runtime_settings(compiled, runtime_settings)
    return compiled


def _find_python_trt_engine(compiled):
    """Walk the compiled graph module and return the Python ``TRTEngine`` instance,
    or ``None`` if the build selected the C++ runtime (whitebox tests guard with
    a class-level ``skipIf(ENABLED_FEATURES.torch_tensorrt_runtime)``).
    """
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
    from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

    for _, mod in compiled.named_modules():
        if isinstance(mod, TorchTensorRTModule) and isinstance(mod.engine, TRTEngine):
            return mod.engine
    return None


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
@unittest.skipIf(
    ENABLED_FEATURES.torch_tensorrt_runtime,
    "Whitebox introspection requires the Python TRTEngine path",
)
class TestDynamicShapesKernelStrategySetup(TestCase):
    """Tests that the dynamic shapes kernel specialization strategy is correctly applied."""

    def test_default_strategy_is_lazy(self):
        compiled = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine, "No Python TRTEngine found")
        self.assertEqual(
            engine.runtime_settings.dynamic_shapes_kernel_specialization_strategy,
            "lazy",
        )

    def test_eager_strategy_via_compile_hint(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(
                dynamic_shapes_kernel_specialization_strategy="eager"
            )
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertEqual(
            engine.runtime_settings.dynamic_shapes_kernel_specialization_strategy,
            "eager",
        )

    def test_none_strategy_via_compile_hint(self):
        compiled = _compile_simple(
            runtime_settings=RuntimeSettings(
                dynamic_shapes_kernel_specialization_strategy="none"
            )
        )
        engine = _find_python_trt_engine(compiled)
        self.assertIsNotNone(engine)
        self.assertEqual(
            engine.runtime_settings.dynamic_shapes_kernel_specialization_strategy,
            "none",
        )

    def test_runtime_cm_overrides_strategy(self):
        """`set_dynamic_shapes_kernel_strategy` CM overrides the active strategy."""
        compiled = _compile_simple()
        engine = _find_python_trt_engine(compiled)
        self.assertEqual(
            engine.runtime_settings.dynamic_shapes_kernel_specialization_strategy,
            "lazy",
        )
        with torchtrt.runtime.set_dynamic_shapes_kernel_strategy(compiled, "eager"):
            self.assertEqual(
                engine.runtime_settings.dynamic_shapes_kernel_specialization_strategy,
                "eager",
            )
            for bs in (1, 2, 4):
                output = compiled(torch.randn(bs, 3).cuda())
                self.assertEqual(output.shape, (bs, 3))
        # Restored on exit.
        self.assertEqual(
            engine.runtime_settings.dynamic_shapes_kernel_specialization_strategy,
            "lazy",
        )

    def test_context_created_with_each_strategy(self):
        for strategy in ("lazy", "eager", "none"):
            with self.subTest(strategy=strategy):
                compiled = _compile_simple(
                    runtime_settings=RuntimeSettings(
                        dynamic_shapes_kernel_specialization_strategy=strategy
                    )
                )
                engine = _find_python_trt_engine(compiled)
                self.assertFalse(
                    engine.has_context(),
                    f"Lazy: context should NOT yet exist for {strategy}",
                )
                for bs in (1, 2, 4):
                    output = compiled(torch.randn(bs, 3).cuda())
                    self.assertEqual(output.shape, (bs, 3))
                self.assertTrue(
                    engine.has_context(),
                    f"Context should exist after first forward for {strategy}",
                )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel strategy is a TensorRT-RTX feature",
)
class TestDynamicShapesKernelStrategyInference(TestCase):
    """End-to-end: compile + infer with each strategy on the build-selected runtime."""

    @parameterized.expand(_STRATEGIES)
    def test_strategy_inference(self, strategy):
        compiled = _compile_dynamic_conv(strategy)
        x = torch.randn(2, 3, 16, 16, device="cuda")
        y = compiled(x)
        self.assertEqual(tuple(y.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_dynamic_shape_with_eager(self):
        """Exercise shape changes under eager kernel specialization."""
        compiled = _compile_dynamic_conv("eager")
        for batch in (1, 2, 3, 4):
            x = torch.randn(batch, 3, 16, 16, device="cuda")
            y = compiled(x)
            self.assertEqual(tuple(y.shape), (batch, 8, 16, 16))


class TestDynamicShapesKernelStrategyInvalidValue(TestCase):
    """Invalid strategy names are rejected at ``RuntimeSettings.__post_init__``."""

    def test_invalid_strategy_raises_at_construction(self):
        with self.assertRaises(ValueError):
            RuntimeSettings(
                dynamic_shapes_kernel_specialization_strategy="not_a_real_strategy",
            )


if __name__ == "__main__":
    run_tests()
