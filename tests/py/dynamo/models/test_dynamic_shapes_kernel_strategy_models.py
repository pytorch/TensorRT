import importlib
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

# Combinations of (strategy, runtime_name, use_python_runtime). Tests use parameterized
# so the strategy sweep runs on both runtimes with a single test body.
_STRATEGY_RUNTIMES = [
    ("lazy_python", "lazy", True),
    ("eager_python", "eager", True),
    ("none_python", "none", True),
    ("lazy_cpp", "lazy", False),
    ("eager_cpp", "eager", False),
    ("none_cpp", "none", False),
]


def _skip_if_cpp_unavailable(testcase, use_python_runtime):
    if not use_python_runtime and not ENABLED_FEATURES.torch_tensorrt_runtime:
        testcase.skipTest("C++ runtime is not available")


def _compile_with_strategy(model, inputs, *, use_python_runtime, strategy):
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        use_python_runtime=use_python_runtime,
        min_block_size=1,
        dynamic_shapes_kernel_specialization_strategy=strategy,
    )
    torch._dynamo.reset()
    return compiled


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
class TestDynamicShapesKernelStrategyModels(TestCase):
    """End-to-end model tests with each strategy across both runtimes."""

    @parameterized.expand(_STRATEGY_RUNTIMES)
    def test_resnet18_strategy(self, _name, strategy, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        input_tensor = torch.randn(4, 3, 224, 224).cuda()
        compiled = _compile_with_strategy(
            model,
            [
                torchtrt.Input(
                    min_shape=(1, 3, 224, 224),
                    opt_shape=(4, 3, 224, 224),
                    max_shape=(8, 3, 224, 224),
                    dtype=torch.float32,
                )
            ],
            use_python_runtime=use_python_runtime,
            strategy=strategy,
        )
        ref_output = model(input_tensor)
        trt_output = compiled(input_tensor)
        cos_sim = cosine_similarity(ref_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"Cosine similarity {cos_sim} below threshold {COSINE_THRESHOLD} "
            f"(strategy={strategy}, python_runtime={use_python_runtime})",
        )


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
class TestDynamicShapesKernelStrategyDynamic(TestCase):
    """Tests kernel specialization strategies with dynamic input shapes, both runtimes."""

    @parameterized.expand(_STRATEGY_RUNTIMES)
    def test_dynamic_batch_with_strategy(self, _name, strategy, use_python_runtime):
        _skip_if_cpp_unavailable(self, use_python_runtime)

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = ConvModel().eval().cuda()

        compiled = _compile_with_strategy(
            model,
            [
                torchtrt.Input(
                    min_shape=(1, 3, 32, 32),
                    opt_shape=(4, 3, 32, 32),
                    max_shape=(8, 3, 32, 32),
                    dtype=torch.float32,
                )
            ],
            use_python_runtime=use_python_runtime,
            strategy=strategy,
        )

        for batch_size in (1, 4, 8):
            input_tensor = torch.randn(batch_size, 3, 32, 32).cuda()
            ref_output = model(input_tensor)
            trt_output = compiled(input_tensor)
            cos_sim = cosine_similarity(ref_output, trt_output)
            self.assertTrue(
                cos_sim > COSINE_THRESHOLD,
                f"BS={batch_size}, strategy={strategy}, python_runtime={use_python_runtime}: "
                f"cosine similarity {cos_sim} below threshold {COSINE_THRESHOLD}",
            )


if __name__ == "__main__":
    run_tests()
