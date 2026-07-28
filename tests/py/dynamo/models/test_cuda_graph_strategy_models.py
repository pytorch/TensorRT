import unittest

import torch
import torch.nn.functional as F
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.runtime import RuntimeSettings


def _apply_runtime_settings(compiled, rs):
    """Walk a compiled module and apply RuntimeSettings to every TRT submodule."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
        TorchTensorRTModule,
    )

    for _, m in compiled.named_modules():
        if isinstance(m, TorchTensorRTModule):
            m.runtime_settings = rs


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy models require TensorRT-RTX",
)
class TestCudaGraphStrategyModels(TestCase):
    """End-to-end model tests with cuda_graph_strategy."""

    def _check_cosine_similarity(self, output, ref_output, threshold=0.99):
        cos_sim = F.cosine_similarity(
            output.flatten().unsqueeze(0),
            ref_output.flatten().unsqueeze(0),
        )
        self.assertTrue(
            cos_sim.item() > threshold,
            f"Cosine similarity {cos_sim.item():.4f} below threshold {threshold}",
        )

    def test_resnet18_whole_graph_capture(self):
        try:
            from torchvision.models import resnet18
        except ImportError:
            self.skipTest("torchvision not available")

        model = resnet18(weights=None).eval().cuda()
        input_tensor = torch.randn(4, 3, 224, 224).cuda()
        ref_output = model(input_tensor)

        inputs = [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
            )
        ]
        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
        )
        _apply_runtime_settings(
            compiled, RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        )
        torch._dynamo.reset()

        output = compiled(input_tensor)
        self._check_cosine_similarity(output, ref_output)

    def test_resnet18_disabled_strategy(self):
        try:
            from torchvision.models import resnet18
        except ImportError:
            self.skipTest("torchvision not available")

        model = resnet18(weights=None).eval().cuda()
        input_tensor = torch.randn(4, 3, 224, 224).cuda()
        ref_output = model(input_tensor)

        inputs = [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
            )
        ]
        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
        )
        _apply_runtime_settings(
            compiled, RuntimeSettings(cuda_graph_strategy="disabled")
        )
        torch._dynamo.reset()

        output = compiled(input_tensor)
        self._check_cosine_similarity(output, ref_output)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CUDA graph strategy models require TensorRT-RTX",
)
class TestCudaGraphStrategyDynamic(TestCase):
    """Tests with dynamic batch sizes and cudagraph mode integration."""

    def setUp(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torchtrt.runtime.set_cudagraphs_mode(False)

    def test_dynamic_batch_whole_graph_capture(self):
        model = ConvModel().eval().cuda()
        inputs = [
            torchtrt.Input(
                min_shape=(1, 3, 32, 32),
                opt_shape=(4, 3, 32, 32),
                max_shape=(8, 3, 32, 32),
                dtype=torch.float32,
            )
        ]
        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
        )
        _apply_runtime_settings(
            compiled, RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        )
        torch._dynamo.reset()

        for bs in (1, 4, 8):
            input_tensor = torch.randn(bs, 3, 32, 32).cuda()
            ref_output = model(input_tensor)
            output = compiled(input_tensor)
            cos_sim = F.cosine_similarity(
                output.flatten().unsqueeze(0),
                ref_output.flatten().unsqueeze(0),
            )
            self.assertTrue(
                cos_sim.item() > 0.99,
                f"Batch size {bs}: cosine similarity {cos_sim.item():.4f} too low",
            )

    def test_dynamic_batch_with_subgraph_cudagraphs(self):
        model = ConvModel().eval().cuda()
        inputs = [
            torchtrt.Input(
                min_shape=(1, 3, 32, 32),
                opt_shape=(4, 3, 32, 32),
                max_shape=(8, 3, 32, 32),
                dtype=torch.float32,
            )
        ]
        compiled = torchtrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions={torch.float32},
            use_python_runtime=True,
            min_block_size=1,
        )
        _apply_runtime_settings(
            compiled, RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        )
        torch._dynamo.reset()

        torchtrt.runtime.set_cudagraphs_mode(True)

        for bs in (1, 4, 8):
            input_tensor = torch.randn(bs, 3, 32, 32).cuda()
            ref_output = model(input_tensor)
            output = compiled(input_tensor)
            cos_sim = F.cosine_similarity(
                output.flatten().unsqueeze(0),
                ref_output.flatten().unsqueeze(0),
            )
            self.assertTrue(
                cos_sim.item() > 0.99,
                f"Batch size {bs}: cosine similarity {cos_sim.item():.4f} too low",
            )


if __name__ == "__main__":
    run_tests()
