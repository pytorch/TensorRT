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
from torch_tensorrt.dynamo.runtime._MutableTorchTensorRTModule import RefitFlag
from torch_tensorrt.dynamo.utils import check_output_equal

assertions = unittest.TestCase()


@pytest.mark.unit
def test_check_output_equal():
    torch.manual_seed(0)
    a = {
        "a": torch.rand(10, 30),
        "b": [torch.rand(10, 30), torch.rand(5, 5)],
        "c": {"a": torch.rand(10, 30), "b": [torch.rand(10, 30), torch.rand(5, 5)]},
    }
    torch.manual_seed(0)
    b = {
        "a": torch.rand(10, 30),
        "b": [torch.rand(10, 30), torch.rand(5, 5)],
        "c": {"a": torch.rand(10, 30), "b": [torch.rand(10, 30), torch.rand(5, 5)]},
    }
    assertions.assertTrue(
        check_output_equal(a, b),
        msg=f"test_check_output_equal is not correct.",
    )


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_resnet18():

    torch.manual_seed(0)
    inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

    compile_spec = {
        "use_python_runtime": False,
        "enabled_precisions": {torch.float32},
        "make_refittable": True,
    }

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*inputs)

    mutable_module.load_state_dict(model2.state_dict())
    assertions.assertTrue(
        mutable_module.refit_state.get_state() == RefitFlag.NEEDS_REFIT,
        msg=f"Changing the attribute did not trigger the flag. Test failed.",
    )
    # Check the output
    expected_outputs, refitted_outputs = model2(*inputs), mutable_module(*inputs)
    assertions.assertTrue(
        check_output_equal(expected_outputs, refitted_outputs),
        msg=f"The output of saved and reloaded Mutable Module is not correct.",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_save():

    torch.manual_seed(0)
    inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

    # %%
    # Compile the module for the first time and save it.
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    compile_spec = {
        "use_python_runtime": False,
        "enabled_precisions": {torch.float32},
        "make_refittable": True,
    }

    model = models.resnet18(pretrained=True).eval().to("cuda")
    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*inputs)

    save_path = os.path.join(tempfile.gettempdir(), "mutable_module.pkl")
    torch_trt.MutableTorchTensorRTModule.save(mutable_module, save_path)
    reload = torch_trt.MutableTorchTensorRTModule.load(save_path)

    loaded_outputs, trt_gm_outputs = reload(*inputs), mutable_module(*inputs)
    assertions.assertTrue(
        check_output_equal(loaded_outputs, trt_gm_outputs),
        msg=f"The output of saved and reloaded Mutable Module is not correct.",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_resnet18_modify_attribute():

    torch.manual_seed(0)
    inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

    compile_spec = {
        "use_python_runtime": False,
        "enabled_precisions": {torch.float32},
        "make_refittable": True,
    }

    model = models.resnet18(pretrained=True).eval().to("cuda")
    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*inputs)

    mutable_module.conv1.weight = nn.Parameter(
        torch.rand_like(mutable_module.conv1.weight)
    )
    assertions.assertEqual(
        mutable_module.refit_state.get_state(),
        RefitFlag.UNKNOWN,
        msg=f"Changing the attribute did not trigger the flag. Test failed.",
    )
    # Check the output
    model.to("cuda")
    expected_outputs, refitted_outputs = model(*inputs), mutable_module(*inputs)
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 5e-2, 5e-2),
            msg=f"The output of refitted Mutable Module is not correct.",
        )

    # # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_resnet18_modify_attribute_no_refit():

    torch.manual_seed(0)
    inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

    compile_spec = {
        "use_python_runtime": False,
        "enabled_precisions": {torch.float32},
        "make_refittable": True,
    }

    model = models.resnet18(pretrained=True).eval().to("cuda")
    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*inputs)

    mutable_module.conv1.weight = nn.Parameter(
        mutable_module.original_model.conv1.weight
    )
    assertions.assertEqual(
        mutable_module.refit_state.get_state(),
        RefitFlag.UNKNOWN,
        msg=f"Changing the attribute did not trigger the flag. Test failed.",
    )
    # Check the output
    mutable_module.update_refit_condition()
    assertions.assertEqual(
        mutable_module.refit_state.get_state(),
        RefitFlag.LIVE,
        msg=f"update_refit_condition() failed to set the flag to LIVE.",
    )
    # Check the output
    model.to("cuda")
    expected_outputs, refitted_outputs = model(*inputs), mutable_module(*inputs)
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            msg=f"The output of refitted Mutable Module is not correct.",
        )

    # # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
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
        "make_refittable": True,
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
    assertions.assertTrue(
        check_output_equal(expected_outputs, refitted_outputs),
        msg=f"The output of saved and reloaded Mutable Module is not correct.",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_custom_model_with_inplace_init():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            return self.fc1(x)

        def set_weights(self):
            nn.init.normal_(self.conv1.weight)

    torch.manual_seed(0)
    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]

    compile_spec = {
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
        "make_refittable": True,
    }

    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*args)

    mutable_module.set_weights()
    assertions.assertEqual(
        mutable_module.refit_state.get_state(),
        RefitFlag.UNKNOWN,
        msg=f"Changing the attribute did not trigger the flag. Test failed.",
    )

    # Check the output
    model.cuda()
    expected_outputs, refitted_outputs = model(*args), mutable_module(*args)
    assertions.assertTrue(
        check_output_equal(expected_outputs, refitted_outputs),
        msg=f"The output of saved and reloaded Mutable Module is not correct.",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_custom_model_with_init_recompile():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            return self.fc1(x)

        def set_layer(self):
            self.fc1 = nn.Linear(12 * 56 * 56, 3)

    torch.manual_seed(0)
    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]

    compile_spec = {
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
        "make_refittable": True,
    }

    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*args)

    mutable_module.set_layer()
    assertions.assertEqual(
        mutable_module.refit_state.get_state(),
        RefitFlag.UNKNOWN,
        msg=f"Changing the attribute did not trigger the flag. Test failed.",
    )

    # Check the output
    model.cuda()  # move offloaded model from cpu to cuda
    expected_outputs, refitted_outputs = model(*args), mutable_module(*args)
    assertions.assertTrue(
        check_output_equal(expected_outputs, refitted_outputs),
        msg=f"The output of saved and reloaded Mutable Module is not correct.",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_custom_model_with_kwarg_different_input():
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
        "make_refittable": True,
    }

    mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
    mutable_module(*args, **kwargs)

    kwargs = {
        "d": {"value": torch.tensor(8).to("cuda")},
        "b": torch.tensor(6).to("cuda"),
        "c": torch.tensor(0.5).to("cuda"),
    }

    # Check the output
    model.cuda()
    expected_outputs, refitted_outputs = model(*args, **kwargs), mutable_module(
        *args, **kwargs
    )
    assertions.assertTrue(
        check_output_equal(expected_outputs, refitted_outputs),
        msg=f"The output of saved and reloaded Mutable Module is not correct.",
    )

    # Clean up model env
    torch._dynamo.reset()
