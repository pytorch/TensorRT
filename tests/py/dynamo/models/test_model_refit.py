import importlib
import os
import sys
import unittest

import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torch_tensorrt as torch_trt
from torch import nn
from torch_tensorrt.dynamo import refit_module_weights
from torch_tensorrt.dynamo._refit import (
    construct_refit_mapping,
    get_engine_from_encoded_engine,
)
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.logging import TRT_LOGGER

import tensorrt as trt

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_mapping():
    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    trt_input = [
        torchtrt.Input(i.shape, dtype=torch.float, format=torch.contiguous_format)
        for i in inputs
    ]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )
    settings = trt_gm._run_on_acc_0.settings
    runtime = trt.Runtime(TRT_LOGGER)

    engine_info = trt_gm._run_on_acc_0.engine.__getstate__()[0]
    engine = get_engine_from_encoded_engine(engine_info[3], runtime)

    exp_program2 = pre_export_lowering(exp_program2, settings)
    exp_program2 = exp_program2.run_decompositions(
        get_decompositions(settings.enable_experimental_decompositions)
    )
    new_gm = exp_program2.module()
    new_gm = post_lowering(new_gm, settings)
    mapping = construct_refit_mapping(new_gm, trt_input, settings)

    refitter = trt.Refitter(engine, TRT_LOGGER)
    weight_list = refitter.get_all_weights()
    for weight in weight_list:
        assertions.assertTrue(
            weight in mapping,
            msg=f"Weight is not found in mapping. Test failed",
        )
    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
@pytest.mark.critical
def test_conv_refit_with_weightmap():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.conv(x)

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_batch_norm_refit_one_engine_with_weightmap():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(self.conv(x))

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_batch_norm_refit_one_engine_without_weightmap():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(self.conv(x))

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
        verify_output=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_with_weightmap():
    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_no_map_with_weightmap():
    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    trt_gm._run_on_acc_0.weight_name_map = None

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@unittest.skipIf(
    torch_trt.ENABLED_FEATURES.tensorrt_rtx,
    "Refit with wrong weightmap is not supported on TensorRT-RTX",
)
@pytest.mark.unit
def test_refit_one_engine_with_wrong_weightmap():
    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )
    # Manually Deleted all batch norm layer. This suppose to fail the fast refit
    trt_gm._run_on_acc_0.weight_name_map = {
        k: v
        for k, v in trt_gm._run_on_acc_0.weight_name_map.items()
        if "[SCALE]" not in k
    }

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not importlib.util.find_spec("transformers"),
    "transformers is required to run this test",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_refit_one_engine_bert_with_weightmap():
    from transformers import BertModel

    inputs = [
        torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    ]
    model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    model2 = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    nn.init.xavier_normal_(model2.embeddings.word_embeddings.weight)
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        if not isinstance(expected_output, torch.Tensor) or not isinstance(
            refitted_output, torch.Tensor
        ):
            continue
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_inline_runtime_with_weightmap(tmpdir):

    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs), strict=False)
    exp_program2 = torch.export.export(model2, tuple(inputs), strict=False)

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )
    torchtrt.save(trt_gm, trt_ep_path, arg_inputs=inputs, retrace=True)
    trt_gm = torch.export.load(trt_ep_path)

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )

    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_python_runtime_with_weightmap():
    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_refit_multiple_engine_with_weightmap():
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

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")

    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    torch_executed_ops = {"torch.ops.aten.convolution.default"}
    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
        torch_executed_ops=torch_executed_ops,
        reuse_cached_engines=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@pytest.mark.unit
def test_refit_multiple_engine_with_weightmap_cpu_offload():
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

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")

    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    torch_executed_ops = {"torch.ops.aten.convolution.default"}
    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
        torch_executed_ops=torch_executed_ops,
        reuse_cached_engines=False,
        offload_module_to_cpu=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )
    model2.cuda()
    # Check the output
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_without_weightmap():
    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not importlib.util.find_spec("transformers"),
    "transformers is required to run this test",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_refit_one_engine_bert_without_weightmap():
    from transformers import BertModel

    inputs = [
        torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    ]
    model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    model2 = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    nn.init.xavier_normal_(model2.embeddings.word_embeddings.weight)
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        if not isinstance(expected_output, torch.Tensor) or not isinstance(
            refitted_output, torch.Tensor
        ):
            continue
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_inline_runtime_without_weightmap(tmpdir):
    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )
    torchtrt.save(trt_gm, trt_ep_path, arg_inputs=inputs)
    trt_gm = torch.export.load(trt_ep_path)
    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
@pytest.mark.unit
def test_refit_one_engine_python_runtime_without_weightmap():
    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_refit_multiple_engine_without_weightmap():
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

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")

    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    torch_executed_ops = {"torch.ops.aten.convolution.default"}
    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        min_block_size=min_block_size,
        immutable_weights=False,
        torch_executed_ops=torch_executed_ops,
        reuse_cached_engines=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

    # Check the output
    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@unittest.skipIf(
    torch_trt.ENABLED_FEATURES.tensorrt_rtx and sys.platform == "win32",
    "cumsum refit errors out on TensorRT-RTX on Windows",
)
@pytest.mark.unit
def test_refit_cumsum():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 16 * 16, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = torch.cumsum(self.fc1(x), 1)
            x = x**2
            return x

    model = net().eval().to("cuda")
    model2 = net().eval().to("cuda")
    inputs = [torch.randn((1, 3, 16, 16)).to("cuda")]
    model(*inputs)
    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))
    with torchtrt.logging.debug():
        trt_gm = torchtrt.dynamo.compile(
            exp_program,
            tuple(inputs),
            enabled_precisions={torch.float},
            min_block_size=1,
            immutable_weights=False,
        )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    model2.to("cuda")
    expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(
        *inputs
    )
    for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
        assertions.assertTrue(
            torch.allclose(expected_output, refitted_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_complex_buffer_refit():
    """Refit a model whose weights include a complex-valued buffer (e.g. RoPE freqs).

    Exercises the combined complex_graph_detection + refit_module_weights path:
      - complex get_attr buffer is unpacked to real by the lowering pass
      - complex placeholder input goes through view_as_real at the TRT boundary
      - after refitting with new frequencies the output matches the new model
    """

    class ComplexFreqModel(nn.Module):
        def __init__(self, freqs: torch.Tensor):
            super().__init__()
            self.register_buffer("freqs", freqs.cuda())

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            # complex mul then expose as real so TRT can produce a real output
            return torch.view_as_real(z * self.freqs)

    SEQ, DIM = 8, 32

    def make_freqs() -> torch.Tensor:
        angles = torch.rand(SEQ, DIM // 2)
        return torch.polar(torch.ones_like(angles), angles).cuda()

    freqs1 = make_freqs()
    freqs2 = make_freqs()

    model1 = ComplexFreqModel(freqs1).eval()
    model2 = ComplexFreqModel(freqs2).eval()

    z = torch.randn(SEQ, DIM // 2, dtype=torch.complex64).cuda()
    inputs = [z]

    exp_program1 = torch.export.export(model1, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program1,
        tuple(inputs),
        use_python_runtime=True,
        min_block_size=1,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

    expected_output = exp_program2.module()(*inputs)
    refitted_output = new_trt_gm(*inputs)

    assertions.assertTrue(
        torch.allclose(expected_output, refitted_output, atol=1e-2, rtol=1e-2),
        "Refit with complex buffer failed: output mismatch after refit",
    )

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_complex_buffer_with_real_param_refit():
    """Refit a model that mixes a complex buffer with a real nn.Linear weight.

    Verifies that Stage 3 slice-matching for complex buffer constants coexists
    correctly with ordinary weight-name-map entries for real parameters.
    After refitting both the frequencies and the projection matrix, the output
    should match the new model exactly.
    """

    SEQ, DIM = 8, 32

    class ComplexRotateAndProject(nn.Module):
        def __init__(self, freqs: torch.Tensor):
            super().__init__()
            self.register_buffer("freqs", freqs.cuda())
            self.proj = nn.Linear(DIM, DIM, bias=False)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            rotated = z * self.freqs  # complex mul, (SEQ, DIM//2)
            r = torch.view_as_real(rotated)  # (SEQ, DIM//2, 2)
            flat = r.reshape(z.shape[0], -1)  # (SEQ, DIM)
            return self.proj(flat)  # (SEQ, DIM) real output

    def make_freqs() -> torch.Tensor:
        angles = torch.rand(SEQ, DIM // 2)
        return torch.polar(torch.ones_like(angles), angles).cuda()

    model1 = ComplexRotateAndProject(make_freqs()).eval().cuda()
    model2 = ComplexRotateAndProject(make_freqs()).eval().cuda()

    z = torch.randn(SEQ, DIM // 2, dtype=torch.complex64).cuda()
    inputs = [z]

    exp_program1 = torch.export.export(model1, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program1,
        tuple(inputs),
        use_python_runtime=True,
        min_block_size=1,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

    expected_output = exp_program2.module()(*inputs)
    refitted_output = new_trt_gm(*inputs)

    assertions.assertTrue(
        torch.allclose(expected_output, refitted_output, atol=1e-2, rtol=1e-2),
        "Refit with complex buffer + real param failed: output mismatch",
    )

    torch._dynamo.reset()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.refit,
    "Refit feature is not supported in Python 3.13 or higher",
)
@pytest.mark.unit
def test_dual_complex_buffer_refit():
    """Refit a model with two independent complex buffers.

    Ensures Stage 3 value-based matching correctly distinguishes the real and
    imaginary slices of freqs_a from those of freqs_b when both are unpacked to
    separate _unpacked_complex state-dict entries with the same shape.
    """

    SEQ, DIM = 8, 32

    class DualComplexFreqModel(nn.Module):
        def __init__(self, freqs_a: torch.Tensor, freqs_b: torch.Tensor):
            super().__init__()
            self.register_buffer("freqs_a", freqs_a.cuda())
            self.register_buffer("freqs_b", freqs_b.cuda())

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            ra = torch.view_as_real(z * self.freqs_a)  # (SEQ, DIM//2, 2)
            rb = torch.view_as_real(z * self.freqs_b)  # (SEQ, DIM//2, 2)
            return ra + rb  # real output

    def make_freqs() -> torch.Tensor:
        angles = torch.rand(SEQ, DIM // 2)
        return torch.polar(torch.ones_like(angles), angles).cuda()

    model1 = DualComplexFreqModel(make_freqs(), make_freqs()).eval()
    model2 = DualComplexFreqModel(make_freqs(), make_freqs()).eval()

    z = torch.randn(SEQ, DIM // 2, dtype=torch.complex64).cuda()
    inputs = [z]

    exp_program1 = torch.export.export(model1, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program1,
        tuple(inputs),
        use_python_runtime=True,
        min_block_size=1,
        immutable_weights=False,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

    expected_output = exp_program2.module()(*inputs)
    refitted_output = new_trt_gm(*inputs)

    assertions.assertTrue(
        torch.allclose(expected_output, refitted_output, atol=1e-2, rtol=1e-2),
        "Refit with dual complex buffers failed: output mismatch",
    )

    torch._dynamo.reset()
