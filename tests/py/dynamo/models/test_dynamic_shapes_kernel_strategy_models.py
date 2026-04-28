import importlib
import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
class TestDynamicShapesKernelStrategyModels(TestCase):
    """End-to-end model tests with different kernel specialization strategies."""

    def tearDown(self):
        torch._dynamo.reset()

    def _compile_and_verify(self, model, strategy):
        input_tensor = torch.randn(4, 3, 224, 224).cuda()
        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=[
                torchtrt.Input(
                    min_shape=(1, 3, 224, 224),
                    opt_shape=(4, 3, 224, 224),
                    max_shape=(8, 3, 224, 224),
                    dtype=torch.float32,
                )
            ],
            use_python_runtime=True,
            min_block_size=1,
            dynamic_shapes_kernel_specialization_strategy=strategy,
        )
        ref_output = model(input_tensor)
        trt_output = compiled(input_tensor)
        cos_sim = cosine_similarity(ref_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"Cosine similarity {cos_sim} below threshold {COSINE_THRESHOLD} "
            f"with strategy={strategy}",
        )

    def test_resnet18_lazy_strategy(self):
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        self._compile_and_verify(model, "lazy")

    def test_resnet18_eager_strategy(self):
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        self._compile_and_verify(model, "eager")

    def test_resnet18_none_strategy(self):
        import torchvision.models as models

        model = models.resnet18(pretrained=True).eval().cuda()
        self._compile_and_verify(model, "none")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Dynamic shapes kernel specialization strategy requires TensorRT-RTX",
)
class TestDynamicShapesKernelStrategyDynamic(TestCase):
    """Tests kernel specialization strategies with dynamic input shapes."""

    def tearDown(self):
        torch._dynamo.reset()

    def _test_dynamic_batch_with_strategy(self, strategy):
        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = ConvModel().eval().cuda()

        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=[
                torchtrt.Input(
                    min_shape=(1, 3, 32, 32),
                    opt_shape=(4, 3, 32, 32),
                    max_shape=(8, 3, 32, 32),
                    dtype=torch.float32,
                )
            ],
            use_python_runtime=True,
            min_block_size=1,
            dynamic_shapes_kernel_specialization_strategy=strategy,
        )

        for batch_size in (1, 4, 8):
            with self.subTest(batch_size=batch_size, strategy=strategy):
                input_tensor = torch.randn(batch_size, 3, 32, 32).cuda()
                ref_output = model(input_tensor)
                trt_output = compiled(input_tensor)
                cos_sim = cosine_similarity(ref_output, trt_output)
                self.assertTrue(
                    cos_sim > COSINE_THRESHOLD,
                    f"BS={batch_size}, strategy={strategy}: cosine similarity "
                    f"{cos_sim} below threshold {COSINE_THRESHOLD}",
                )

    def test_dynamic_batch_lazy(self):
        self._test_dynamic_batch_with_strategy("lazy")

    def test_dynamic_batch_eager(self):
        self._test_dynamic_batch_with_strategy("eager")

    def test_dynamic_batch_none(self):
        self._test_dynamic_batch_with_strategy("none")


if __name__ == "__main__":
    run_tests()
