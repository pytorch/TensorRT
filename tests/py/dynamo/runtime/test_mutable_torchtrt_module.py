# type: ignore
import os
import tempfile
import unittest

import pytest
import timm
import torch
import torch.nn.functional as F
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch import nn
from torch_tensorrt.dynamo._compiler import (
    convert_exported_program_to_serialized_trt_engine,
)
from torch_tensorrt.dynamo.utils import (
    COSINE_THRESHOLD,
    cosine_similarity,
    prepare_inputs,
)

assertions = unittest.TestCase()


# @pytest.mark.unit
def test_resnet18_and_save():

    torch.manual_seed(0)
    inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

    # %%
    # Compile the module for the first time and save it.
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    compile_spec = {
        "use_python": False,
        "enabled_precisions": {torch.float32},
        "make_refitable": True,
    }

    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*inputs)

    mutable_module.load_state_dict(model2.state_dict())

    # Check the output
    expected_outputs, refitted_outputs = model2(*inputs), mutable_module(*inputs)
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            msg=f"The output of refitted Mutable Module is not correct.",
        )

    print("Refit successfully!")
    save_path = os.path.join(tempfile.gettempdir(), "mutable_module.pkl")
    torch_trt.MutableTorchTensorRTModule.save(mutable_module, save_path)
    reload = torch_trt.MutableTorchTensorRTModule.load(save_path)

    expected_outputs, refitted_outputs = reload(*inputs), mutable_module(*inputs)
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            msg=f"The output of saved and reloaded Mutable Module is not correct.",
        )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_custom_model_with_kwarg():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    torch.manual_seed(0)
    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "d": {"value": torch.tensor(8).to("cuda")},
        "b": torch.tensor(6).to("cuda"),
    }

    compile_spec = {
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
        "make_refitable": True,
    }

    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*args, **kwargs)

    torch.manual_seed(2)
    model2 = net().eval().to("cuda")
    mutable_module.load_state_dict(model2.state_dict())

    # Check the output
    expected_outputs, refitted_outputs = model2(*args, **kwargs), mutable_module(
        *args, **kwargs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):

        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            msg=f"The output of refitted Mutable Module is not correct.",
        )

    print("Refit successfully!")
    # Clean up model env
    torch._dynamo.reset()
