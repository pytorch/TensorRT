import os
import tempfile
import time
import unittest

import numpy as np
import pytest
import tensorrt as trt
import torch
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torch_tensorrt as torch_trt
import torchvision.models as models
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
from transformers import BertModel

assertions = unittest.TestCase()


@unittest.skipIf(
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_mapping():

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    trt_input = [
        torchtrt.Input(i.shape, dtype=torch.float, format=torch.contiguous_format)
        for i in inputs
    ]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )
    settings = trt_gm._run_on_acc_0.settings
    runtime = trt.Runtime(TRT_LOGGER)

    engine_info = trt_gm._run_on_acc_0.engine.__getstate__()[0]
    engine = get_engine_from_encoded_engine(engine_info[3], runtime)

    exp_program2 = pre_export_lowering(exp_program2)
    exp_program2 = exp_program2.run_decompositions(
        get_decompositions(settings.enable_experimental_decompositions)
    )
    new_gm = exp_program2.module()
    new_gm = post_lowering(new_gm)
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
@pytest.mark.unit
def test_refit_one_engine_with_weightmap():

    model = models.resnet18(pretrained=False).eval().to("cuda")
    model2 = models.resnet18(pretrained=True).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
        verify_output=True,
    )

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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_refit_one_engine_no_map_with_weightmap():

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    trt_gm._run_on_acc_0.weight_name_map = None

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_refit_one_engine_with_wrong_weightmap():

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
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
@pytest.mark.unit
def test_refit_one_engine_bert_with_weightmap():
    inputs = [
        torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    ]
    model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    model2 = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    nn.init.xavier_normal_(model2.embeddings.word_embeddings.weight)
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

    # Check the output
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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_refit_one_engine_inline_runtime__with_weightmap():
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )
    torchtrt.save(trt_gm, trt_ep_path, inputs=inputs)
    trt_gm = torch.export.load(trt_ep_path)
    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

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


@pytest.mark.unit
def test_refit_one_engine_python_runtime_with_weightmap():

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
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
    debug = False
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
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
        torch_executed_ops=torch_executed_ops,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=True,
    )

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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_refit_one_engine_without_weightmap():

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_refit_one_engine_bert_without_weightmap():
    inputs = [
        torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    ]
    model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    model2 = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
    nn.init.xavier_normal_(model2.embeddings.word_embeddings.weight)
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

    # Check the output
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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
)
@pytest.mark.unit
def test_refit_one_engine_inline_runtime_without_weightmap():
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = False

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )
    torchtrt.save(trt_gm, trt_ep_path, inputs=inputs)
    trt_gm = torch.export.load(trt_ep_path)
    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

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


@pytest.mark.unit
def test_refit_one_engine_python_runtime_without_weightmap():

    model = models.resnet18(pretrained=True).eval().to("cuda")
    model2 = models.resnet18(pretrained=False).eval().to("cuda")
    inputs = [torch.randn((1, 3, 224, 224)).to("cuda")]
    enabled_precisions = {torch.float}
    debug = False
    min_block_size = 1
    use_python_runtime = True

    exp_program = torch.export.export(model, tuple(inputs))
    exp_program2 = torch.export.export(model2, tuple(inputs))

    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        tuple(inputs),
        use_python_runtime=use_python_runtime,
        enabled_precisions=enabled_precisions,
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

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
    not torch_trt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "TorchScript Frontend is not available",
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
    debug = False
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
        debug=debug,
        min_block_size=min_block_size,
        make_refittable=True,
        torch_executed_ops=torch_executed_ops,
    )

    new_trt_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=exp_program2,
        arg_inputs=inputs,
        use_weight_map_cache=False,
    )

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


@pytest.mark.unit
def test_refit_cumsum_fallback():

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
    inputs = [torch.randn((1, 3, 16, 16)).to("cuda")]
    model(*inputs)
    exp_program = torch.export.export(model, tuple(inputs))
    with torchtrt.logging.debug():
        trt_gm = torchtrt.dynamo.compile(
            exp_program,
            tuple(inputs),
            enabled_precisions={torch.float},
            debug=True,
            min_block_size=1,
            make_refittable=True,
        )

    num_pyt_segments = len(
        [1 for submod in list(trt_gm.named_children()) if "_run_on_gpu" in submod[0]]
    )

    # Number of pyt segments should be 1 (because of cumsum being non-refitable)
    assertions.assertTrue(
        num_pyt_segments == 1,
        f"test_refit_cumsum_fallback test found {num_pyt_segments} pytorch segments but expected 1",
    )

    # Check the output
    pyt_outputs, trt_outputs = exp_program.module()(*inputs), trt_gm(*inputs)
    for pyt_output, trt_output in zip(pyt_outputs, trt_outputs):
        assertions.assertTrue(
            torch.allclose(pyt_output, trt_output, 1e-2, 1e-2),
            "Refit Result is not correct. Refit failed",
        )
        # Clean up model env

    torch._dynamo.reset()
