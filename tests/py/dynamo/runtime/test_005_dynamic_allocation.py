import importlib
import unittest

import torch
import torch_tensorrt as torch_trt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


class TestDynamicAllocation(TestCase):
    def test_dynamic_allocation(self):
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]

        settings = {
            "ir": "dynamo",
            "use_python_runtime": False,
            "enabled_precisions": {torch.float32},
            "immutable_weights": False,
            "lazy_engine_init": True,
            "dynamically_allocate_resources": True,
        }

        model = models.resnet152(pretrained=True).eval().to("cuda")
        compiled_module = torch_trt.compile(model, inputs=inputs, **settings)
        compiled_module(*inputs)

        # Inference on PyTorch model
        model_output = model(*inputs)
        cos_sim = cosine_similarity(model_output, compiled_module(*inputs))
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        # Clean up model env
        torch._dynamo.reset()
