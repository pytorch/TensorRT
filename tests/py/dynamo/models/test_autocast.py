# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import base64
import logging
import pickle

import pytest
import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import pre_export_lowering
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import SERIALIZED_METADATA_IDX


@pytest.mark.unit
@pytest.mark.critical
def test_no_pytorch_autocast():
    class NoPytorchAutocastModel(nn.Module):
        def __init__(self):
            super(NoPytorchAutocastModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
            )
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            out1 = self.conv1(x)
            out2 = self.relu1(out1)
            out3 = self.pool1(out2)
            out4 = self.conv2(out3)
            out5 = self.relu2(out4)
            out6 = self.pool2(out5)
            out7 = self.flatten(out6)
            out8 = self.fc1(out7)
            out9 = torch.add(out8, out8)
            return x, out1, out2, out3, out4, out5, out6, out7, out8, out9

    model = NoPytorchAutocastModel().cuda().eval()
    BS = 8
    inputs = (torch.randn((BS, 3, 32, 32), dtype=torch.float32, device="cuda"),)
    ep = torch.export.export(model, inputs)
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*inputs), batch_size=BS, shuffle=False
    )

    with torch_tensorrt.dynamo.Debugger(
        "graphs",
        logging_dir=".",
        engine_builder_monitor=False,
    ):
        trt_autocast_mod = torch_tensorrt.compile(
            ep.module(),
            arg_inputs=inputs,
            min_block_size=1,
            enable_autocast=True,
            autocast_low_precision_type=torch.float16,
            autocast_excluded_nodes={"^conv1$", "relu"},
            autocast_excluded_ops={"torch.ops.aten.flatten.using_ints"},
            autocast_max_output_threshold=512,
            autocast_max_depth_of_reduction=None,
            autocast_calibration_dataloader=calibration_dataloader,
        )

        autocast_outs = trt_autocast_mod(*inputs)
        pytorch_outs = model(*inputs)

        should_be_fp32 = [
            autocast_outs[0],
            autocast_outs[1],
            autocast_outs[2],
            autocast_outs[5],
            autocast_outs[7],
        ]
        should_be_fp16 = [
            autocast_outs[3],
            autocast_outs[4],
            autocast_outs[6],
            autocast_outs[8],
            autocast_outs[9],
        ]
        assert all(
            a.dtype == torch.float32 for a in should_be_fp32
        ), "Some Autocast outputs are not float32!"
        assert all(
            a.dtype == torch.float16 for a in should_be_fp16
        ), "Some Autocast outputs are not float16!"
        for i, (a, w) in enumerate(zip(autocast_outs, pytorch_outs)):
            assert torch.allclose(
                a.to(torch.float32), w.to(torch.float32), atol=1e-2, rtol=1e-2
            ), f"Autocast and Pytorch outputs do not match! autocast_outs[{i}] = {a}, pytorch_outs[{i}] = {w}"


@pytest.mark.unit
@pytest.mark.critical
def test_whole_pytorch_autocast():
    class WholePytorchAutocastModel(nn.Module):
        def __init__(self):
            super(WholePytorchAutocastModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
            )
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
                out1 = self.conv1(x)
                out2 = self.relu1(out1)
                out3 = self.pool1(out2)
                out4 = self.conv2(out3)
                out5 = self.relu2(out4)
                out6 = self.pool2(out5)
                out7 = self.flatten(out6)
                out8 = self.fc1(out7)
                out9 = torch.log(
                    torch.abs(out8) + 1
                )  # log is fp32 due to Pytorch Autocast requirements
                return x, out1, out2, out3, out4, out5, out6, out7, out8, out9

    model = WholePytorchAutocastModel().cuda().eval()
    BS = 4
    inputs = (torch.randn((BS, 3, 32, 32), dtype=torch.float32, device="cuda"),)
    ep = torch.export.export(model, inputs)
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*inputs), batch_size=BS, shuffle=False
    )

    with torch_tensorrt.dynamo.Debugger(
        "graphs",
        logging_dir=".",
        engine_builder_monitor=False,
    ):
        trt_autocast_mod = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=inputs,
            min_block_size=1,
            # Torch-TensorRT's autocast doesn't affect layers inside Pytorch autocast
            enable_autocast=True,
            autocast_low_precision_type=torch.bfloat16,
            autocast_excluded_nodes={"^conv1$", "relu"},
            autocast_excluded_ops={"torch.ops.aten.flatten.using_ints"},
            autocast_max_output_threshold=512,
            autocast_max_depth_of_reduction=None,
            autocast_calibration_dataloader=calibration_dataloader,
        )

        autocast_outs = trt_autocast_mod(*inputs)
        pytorch_outs = model(*inputs)

        should_be_fp32 = [autocast_outs[0], autocast_outs[9]]
        should_be_fp16 = [autocast_outs[i] for i in range(1, 9)]
        assert all(
            a.dtype == torch.float32 for a in should_be_fp32
        ), "Some Autocast outputs are not float32!"
        assert all(
            a.dtype == torch.float16 for a in should_be_fp16
        ), "Some Autocast outputs are not float16!"
        for i, (a, w) in enumerate(zip(autocast_outs, pytorch_outs)):
            assert torch.allclose(
                a.to(torch.float32), w.to(torch.float32), atol=1e-2, rtol=1e-2
            ), f"Autocast and Pytorch outputs do not match! autocast_outs[{i}] = {a}, pytorch_outs[{i}] = {w}"


@pytest.mark.unit
@pytest.mark.critical
def test_mixed_pytorch_autocast():
    class MixedPytorchAutocastModel(nn.Module):
        def __init__(self):
            super(MixedPytorchAutocastModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
            )
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            out1 = self.conv1(x)
            out2 = self.relu1(out1)
            out3 = self.pool1(out2)
            out4 = self.conv2(out3)
            out5 = self.relu2(out4)
            out6 = self.pool2(out5)
            out7 = self.flatten(out6)
            with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
                out8 = self.fc1(out7)
                out9 = torch.log(
                    torch.abs(out8) + 1
                )  # log is fp32 due to Pytorch Autocast requirements
            return x, out1, out2, out3, out4, out5, out6, out7, out8, out9

    model = MixedPytorchAutocastModel().cuda().eval()
    BS = 2
    inputs = (torch.randn((BS, 3, 32, 32), dtype=torch.float32, device="cuda"),)
    ep = torch.export.export(model, inputs)
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*inputs), batch_size=BS, shuffle=False
    )

    with torch_tensorrt.dynamo.Debugger(
        "graphs",
        logging_dir=".",
        engine_builder_monitor=False,
    ):
        trt_autocast_mod = torch_tensorrt.compile(
            ep.module(),
            arg_inputs=inputs,
            min_block_size=1,
            # Torch-TensorRT's autocast doesn't affect layers inside Pytorch autocast
            enable_autocast=True,
            autocast_low_precision_type=torch.bfloat16,
            autocast_excluded_nodes={"^conv1$", "relu"},
            autocast_excluded_ops={"torch.ops.aten.flatten.using_ints"},
            autocast_max_output_threshold=512,
            autocast_max_depth_of_reduction=None,
            autocast_calibration_dataloader=calibration_dataloader,
        )

        autocast_outs = trt_autocast_mod(*inputs)
        pytorch_outs = model(*inputs)

        should_be_fp32 = [
            autocast_outs[0],
            autocast_outs[1],
            autocast_outs[2],
            autocast_outs[5],
            autocast_outs[7],
            autocast_outs[9],
        ]
        should_be_fp16 = [
            autocast_outs[8],
        ]
        should_be_bf16 = [autocast_outs[3], autocast_outs[4], autocast_outs[6]]
        assert all(
            a.dtype == torch.float32 for a in should_be_fp32
        ), "Some Autocast outputs are not float32!"
        assert all(
            a.dtype == torch.float16 for a in should_be_fp16
        ), "Some Autocast outputs are not float16!"
        assert all(
            a.dtype == torch.bfloat16 for a in should_be_bf16
        ), "Some Autocast outputs are not bfloat16!"
        for i, (a, w) in enumerate(zip(autocast_outs, pytorch_outs)):
            assert torch.allclose(
                a.to(torch.float32), w.to(torch.float32), atol=1e-2, rtol=1e-2
            ), f"Autocast and Pytorch outputs do not match! autocast_outs[{i}] = {a}, pytorch_outs[{i}] = {w}"


@pytest.mark.unit
@pytest.mark.critical
def test_nested_pytorch_autocast():
    class NestedPytorchAutocastModel(nn.Module):
        def __init__(self):
            super(NestedPytorchAutocastModel, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
            )
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x, y):
            out1 = self.conv1(
                x
            )  # fp32 because of "^conv1$" in `autocast_excluded_nodes`
            out2 = self.relu1(
                out1
            )  # fp32 because of "relu" in `autocast_excluded_nodes`
            out3 = self.pool1(out2)  # bf16
            out4 = self.conv2(out3)  # bf16
            out5 = self.relu2(
                out4
            )  # fp32 because of "relu" in `autocast_excluded_nodes`
            out6 = self.pool2(out5)  # bf16
            out7 = self.flatten(
                out6
            )  # fp32 because of `torch.ops.aten.flatten.using_ints` in `autocast_excluded_ops`
            # Respect the precisions in the pytorch autocast context
            with torch.autocast(x.device.type, enabled=True, dtype=torch.float32):
                out8 = self.fc1(out7)  # fp32
                with torch.autocast(x.device.type, enabled=False):
                    out9 = torch.sub(out8.half(), y)  # fp16
                    out10 = torch.add(out9, out9)  # fp16
            with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
                out11 = torch.log(
                    torch.abs(out10) + 1
                )  # fp32 because Pytorch Autocast requires `log` to be fp32
            return x, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11

    model = NestedPytorchAutocastModel().cuda().eval()
    inputs = (
        torch.randn((1, 3, 32, 32), dtype=torch.float32, device="cuda"),
        torch.randn((1,), dtype=torch.float16, device="cuda"),
    )
    ep = torch.export.export(model, inputs)
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*inputs), batch_size=2, shuffle=False
    )

    with torch_tensorrt.dynamo.Debugger(
        "graphs",
        logging_dir=".",
        engine_builder_monitor=False,
    ):
        trt_autocast_mod = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=inputs,
            min_block_size=1,
            # Torch-TensorRT's autocast doesn't affect layers inside Pytorch autocast
            enable_autocast=True,
            autocast_low_precision_type=torch.bfloat16,
            autocast_excluded_nodes={"^conv1$", "relu"},
            autocast_excluded_ops={"torch.ops.aten.flatten.using_ints"},
            autocast_max_output_threshold=512,
            autocast_max_depth_of_reduction=None,
            autocast_calibration_dataloader=calibration_dataloader,
        )

        autocast_outs = trt_autocast_mod(*inputs)
        pytorch_outs = model(*inputs)

        should_be_fp32 = [
            autocast_outs[0],
            autocast_outs[1],
            autocast_outs[2],
            autocast_outs[5],
            autocast_outs[7],
            autocast_outs[8],
            autocast_outs[11],
        ]
        should_be_fp16 = [autocast_outs[9], autocast_outs[10]]
        should_be_bf16 = [autocast_outs[3], autocast_outs[4], autocast_outs[6]]
        assert all(
            a.dtype == torch.float32 for a in should_be_fp32
        ), "Some Autocast outputs are not float32!"
        assert all(
            a.dtype == torch.float16 for a in should_be_fp16
        ), "Some Autocast outputs are not float16!"
        assert all(
            a.dtype == torch.bfloat16 for a in should_be_bf16
        ), "Some Autocast outputs are not bfloat16!"
        for i, (a, w) in enumerate(zip(autocast_outs, pytorch_outs)):
            assert torch.allclose(
                a.to(torch.float32), w.to(torch.float32), atol=1e-2, rtol=1e-2
            ), f"Autocast and Pytorch outputs do not match! autocast_outs[{i}] = {a}, pytorch_outs[{i}] = {w}"


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "split",
    [
        lambda y: torch.chunk(y, 2, dim=-1),
        lambda y: torch.split(y, 16, dim=-1),
        lambda y: torch.unbind(y.view(4, 2, 16), dim=1),
    ],
    ids=["chunk", "split", "unbind"],
)
def test_autocast_ops_with_list_outputs(split):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 32)

        def forward(self, x):
            a, b = split(self.linear(x))
            return a * b

    model = Model().cuda().eval()
    x = torch.randn(4, 16, device="cuda")
    ep = torch.export.export(model, (x,))
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    out = trt_model(x)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), model(x), rtol=1e-2, atol=1e-2)

    # An excluded op that reads a list element must get float32 inputs:
    # 300 * 300 overflows float16.
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(300.0)
    ep = torch.export.export(model, (x,))
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_excluded_ops={"torch.ops.aten.mul.Tensor"},
    )
    out = trt_model(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, model(x))


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "split",
    [
        lambda y: torch.chunk(y, 2, dim=-1),
        lambda y: torch.split(y, 16, dim=-1),
        lambda y: torch.unbind(y.view(4, 2, 16), dim=1),
    ],
    ids=["chunk", "split", "unbind"],
)
def test_autocast_calibration_checks_values_through_list_outputs(split):
    class Model(nn.Module):
        def forward(self, x):
            # The multiply feeding the list op and the one reading its elements
            # both overflow float16.
            y = x * 300.0
            a, b = split(y * y)
            return a * b / 8.1e9

    model = Model().cuda().eval()
    x = torch.ones(4, 32, device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x), model(x))


@pytest.mark.unit
@pytest.mark.parametrize(
    "indices",
    [
        lambda y: torch.max(y, dim=-1)[1],
        lambda y: torch.topk(y, 2, dim=-1).indices,
    ],
    ids=["max", "topk"],
)
def test_autocast_calibration_keeps_large_indices_exact(indices):
    class Model(nn.Module):
        def forward(self, x):
            # Indices above 2048 cannot be stored exactly in float16.
            return indices(x * 2.0) + 1

    model = Model().cuda().eval()
    x = torch.randn(256, 4096, device="cuda") * 1000.0
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x), model(x))


@pytest.mark.unit
@pytest.mark.parametrize("calibrate", [True, False])
def test_autocast_names_torch_cond_as_unsupported(calibrate):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            y = self.linear(x)
            # Only inside the branch do the values leave the float16 range.
            return torch.cond(
                y.sum() > -1e9, lambda t: t * 1e5 / 1e5, lambda t: -t, (y,)
            )

    model = Model().cuda().eval()
    x = torch.randn(4, 16, device="cuda")
    loader = torch.utils.data.DataLoader([(x,)], batch_size=None) if calibrate else None
    with pytest.raises(RuntimeError, match="does not support models with torch.cond"):
        torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            arg_inputs=[x],
            min_block_size=1,
            enable_autocast=True,
            autocast_low_precision_type=torch.float16,
            autocast_calibration_dataloader=loader,
        )


@pytest.mark.unit
def test_autocast_calibration_joins_0_dim_values_across_batches():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.affine = nn.Parameter(torch.tensor([2.0, 0.5]))

        def forward(self, x):
            # unbind of a 1-D tensor and sum give 0-dim values.
            scale, shift = self.affine.unbind(0)
            return x * scale + shift + x / x.sum()

    model = Model().cuda().eval()
    x = torch.rand(4, 16, device="cuda") + 1.0
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,), (x * 2.0,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
def test_autocast_depth_of_reduction_reads_a_recorded_convolution_weight():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(2, 8, 4, 3, 3))

        def forward(self, x):
            # unbind makes the weight a recorded value instead of a parameter.
            return nn.functional.conv2d(torch.relu(x), self.weights.unbind(0)[0])

    x = torch.randn(1, 4, 16, 16, device="cuda")
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_max_depth_of_reduction=8,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    pre_export_lowering(torch.export.export(Model().cuda().eval(), (x,)), settings)
    # 4 input channels times a 3x3 kernel is a reduction depth of 36.
    assert "conv2d" in settings._autocast_calibrated_high_precision_nodes


@pytest.mark.unit
def test_rule_based_autocast_runs_without_traced_outputs():
    from torch_tensorrt.dynamo.lowering.passes.rule_based_autocast import (
        rule_based_autocast,
    )

    class Model(nn.Module):
        def forward(self, x):
            return torch.relu(x) * 2.0

    gm = torch.export.export(Model(), (torch.randn(4, 8),)).graph_module
    settings = CompilationSettings(
        enable_autocast=True, autocast_low_precision_type=torch.float16
    )
    gm = rule_based_autocast(gm, settings)
    assert torch.ops.aten.to.dtype in [n.target for n in gm.graph.nodes]


@pytest.mark.unit
def test_autocast_calibration_ignores_indices_at_the_op_feeding_topk():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 1024)

        def forward(self, x):
            return torch.topk(self.linear(x), 8, dim=-1).values

    x = torch.randn(4, 16, device="cuda")
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    pre_export_lowering(torch.export.export(Model().cuda().eval(), (x,)), settings)
    # topk's indices reach 1023, but they say nothing about the range of the linear layer.
    assert "linear" not in settings._autocast_calibrated_high_precision_nodes


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_warns_without_calibration_dataloader(caplog):
    class LargeIntermediateModel(nn.Module):
        def forward(self, x):
            # The intermediates exceed the FP16 range, but the output does not.
            y = x * 1000.0
            return torch.sqrt(torch.mean(y * y, dim=-1, keepdim=True)) / 1000.0

    model = LargeIntermediateModel().cuda().eval()
    inputs = (torch.randn((8, 64), dtype=torch.float32, device="cuda"),)
    ep = torch.export.export(model, inputs)

    with caplog.at_level(
        logging.WARNING,
        logger="torch_tensorrt.dynamo.lowering.passes.rule_based_autocast",
    ):
        torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=inputs,
            min_block_size=1,
            enable_autocast=True,
            autocast_low_precision_type=torch.float16,
        )
    assert any("autocast_calibration_dataloader" in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Refit of a loaded program needs the Torch-TensorRT C++ runtime",
)
def test_autocast_saves_calibrated_precisions_not_calibration_data(tmp_path):
    class MaskedScores(nn.Module):
        def __init__(self, fill):
            super(MaskedScores, self).__init__()
            self.linear = nn.Linear(16, 16)
            self.fill = nn.Parameter(torch.tensor(fill))

        def forward(self, x, mask):
            # A -1e9 fill is out of the float16 range, so the calibration data
            # keeps these nodes in float32.
            return torch.softmax(self.linear(x) + (1.0 - mask) * self.fill, dim=-1)

    inputs = (
        torch.randn((2, 8, 16), dtype=torch.float32, device="cuda"),
        (torch.rand((2, 8, 16), device="cuda") > 0.3).float(),
    )
    batch = torch.randn((64, 8, 16), dtype=torch.float32, device="cuda")
    batch_mask = (torch.rand((64, 8, 16), device="cuda") > 0.3).float()
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(batch, batch_mask), batch_size=2
    )
    model = MaskedScores(-1e9).cuda().eval()
    trt_autocast_mod = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs),
        arg_inputs=inputs,
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=calibration_dataloader,
        immutable_weights=False,
    )

    (engine,) = [
        m for m in trt_autocast_mod.modules() if isinstance(m, TorchTensorRTModule)
    ]
    encoded = engine._pack_engine_info()[SERIALIZED_METADATA_IDX]
    # Every saved program carries this metadata, so it must not carry the batch.
    settings = TorchTensorRTModule.decode_metadata(encoded)["settings"]
    assert settings.autocast_calibration_dataloader is None
    metadata = base64.b64decode(encoded)
    assert len(metadata) < batch.nbytes
    assert batch[0, 0, :16].cpu().numpy().tobytes() not in metadata

    # Refit cannot change the engine's precisions. An in-memory refit with weights
    # that calibration would treat differently must keep the recorded choice, and
    # so must refit after save and load.
    small_fill = MaskedScores(-1.0).cuda().eval()
    small_fill.load_state_dict(model.state_dict() | {"fill": torch.tensor(-1.0)})
    refitted = torch_tensorrt.dynamo.refit_module_weights(
        trt_autocast_mod, torch.export.export(small_fill, inputs), arg_inputs=inputs
    )
    assert torch.allclose(
        refitted(*inputs).to(torch.float32), small_fill(*inputs), atol=1e-3, rtol=1e-3
    ), "In-memory refit does not match PyTorch!"

    path = tmp_path / "model.ep"
    torch_tensorrt.save(refitted, path, arg_inputs=list(inputs))
    new_model = MaskedScores(-1e9).cuda().eval()
    refitted = torch_tensorrt.dynamo.refit_module_weights(
        torch_tensorrt.load(path),
        torch.export.export(new_model, inputs),
        arg_inputs=inputs,
    )
    assert torch.allclose(
        refitted(*inputs).to(torch.float32), new_model(*inputs), atol=1e-3, rtol=1e-3
    ), "Refit after load does not match PyTorch!"


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_refuses_calibrated_precisions_for_another_graph():
    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            return self.linear(x)

    inputs = (torch.randn((2, 16), dtype=torch.float32, device="cuda"),)
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        _autocast_calibrated_high_precision_nodes=frozenset({"not_in_this_graph"}),
    )
    with pytest.raises(RuntimeError, match="not_in_this_graph"):
        pre_export_lowering(
            torch.export.export(LinearModel().cuda().eval(), inputs), settings
        )


@pytest.mark.unit
def test_autocast_excluded_ops_accepts_op_overloads():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x):
            return torch.flatten(self.linear(x), 1)

    model = Model().cuda().eval()
    x = torch.randn(4, 2, 8, device="cuda")
    trt_model = torch_tensorrt.compile(
        model,
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_excluded_ops={torch.ops.aten.flatten.using_ints},
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    out = trt_model(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, model(x), rtol=1e-2, atol=1e-2)

    single = CompilationSettings(autocast_excluded_ops="torch.ops.aten.softmax.int")
    assert single.autocast_excluded_ops == {"torch.ops.aten.softmax.int"}
    assert (
        CompilationSettings(autocast_excluded_ops=None).autocast_excluded_ops == set()
    )
    # Settings pickled by older versions, for example in an engine cache, hold
    # the entries as given.
    old = CompilationSettings()
    old.autocast_excluded_ops = ["torch.ops.aten.softmax.int"]
    loaded = pickle.loads(pickle.dumps(old))
    assert loaded.autocast_excluded_ops == {"torch.ops.aten.softmax.int"}


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "factory",
    [
        lambda x: torch.zeros(x.shape[0], 1, device=x.device),
        lambda x: torch.ones(x.shape[0], 1, device=x.device) - 1.0,
        lambda x: torch.full((x.shape[0], 1), 0.0, device=x.device),
        lambda x: torch.arange(x.shape[0], dtype=torch.float32, device=x.device)[
            :, None
        ]
        * 0.0,
    ],
    ids=["zeros", "ones", "full", "arange"],
)
def test_autocast_factory_ops_keep_their_dtype(factory):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 32)
            self.norm = nn.LayerNorm(32)
            self.conv = nn.Conv1d(1, 4, 3, padding=1)

        def forward(self, x):
            keep = 1.0 - factory(x)
            y = self.norm(self.linear(x) * keep)
            return self.conv(y[:, None, :] * keep[:, :, None])

    model = Model().cuda().eval()
    x = torch.randn(4, 16, device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize("calibrate", [True, False], ids=["calib", "no_calib"])
@pytest.mark.parametrize(
    "lookup",
    [
        lambda m, x, ids: m.table[ids],
        lambda m, x, ids: m.embedding(ids),
        lambda m, x, ids: m.weight[ids],
        lambda m, x, ids: m.table[ids % 10],
        lambda m, x, ids: torch.gather(x, 1, ids[:, None].expand(-1, 16) % 16),
    ],
    ids=["buffer", "embedding", "parameter", "integer_op", "gather"],
)
def test_autocast_keeps_integer_indices(lookup, calibrate):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.randn(100, 16))
            self.embedding = nn.Embedding(100, 16)
            self.weight = nn.Parameter(torch.randn(100, 16))
            self.linear = nn.Linear(16, 16)

        def forward(self, x, ids):
            return self.linear(x + lookup(self, x, ids))

    model = Model().cuda().eval()
    inputs = (torch.randn(4, 16, device="cuda"), torch.tensor([3, 1, 4, 1]).cuda())
    settings = {}
    if calibrate:
        settings["autocast_calibration_dataloader"] = torch.utils.data.DataLoader(
            [inputs], batch_size=None
        )
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs),
        arg_inputs=list(inputs),
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        **settings,
    )
    torch.testing.assert_close(
        trt_model(*inputs).float(), model(*inputs), rtol=1e-2, atol=1e-2
    )


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "to_float",
    [
        lambda ids: ids / 16,
        lambda ids: ids * 0.5,
        lambda ids: torch.sin(ids * 0.1),
        lambda ids: (1.0 - ids % 2) * -1e4,
        lambda ids: ids.float().square(),
    ],
    ids=["div", "mul", "sin", "mask_bias", "float_square"],
)
def test_autocast_integers_that_become_floats(to_float):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 16)

        def forward(self, x, ids):
            y = to_float(ids)
            return self.linear(torch.cat([x + y, y], dim=-1))

    model = Model().cuda().eval()
    inputs = (
        torch.randn(4, 16, device="cuda"),
        torch.randint(0, 400, (4, 16), device="cuda"),
    )
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs),
        arg_inputs=list(inputs),
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [inputs], batch_size=None
        ),
    )
    ref = model(*inputs)
    torch.testing.assert_close(
        trt_model(*inputs).float(), ref, rtol=5e-2, atol=5e-2 * ref.abs().max().item()
    )


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_keeps_integer_comparisons_exact():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x, ids):
            # float16 rounds ids above 2048 and makes ids above 65504 inf.
            keep = (ids != 128001).unsqueeze(-1)
            return self.linear(x) * keep

    model = Model().cuda().eval()
    x = torch.randn(1, 6, 8, device="cuda")
    ids = torch.tensor([[5, 2049, 70000, 128001, 128002, 130000]], device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x, ids)),
        arg_inputs=[x, ids],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    torch.testing.assert_close(
        trt_model(x, ids).float(), model(x, ids), rtol=1e-2, atol=1e-2
    )


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_keeps_embedding_bag_offsets():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bag = nn.EmbeddingBag(5000, 8, mode="sum")

        def forward(self, ids, offsets):
            return self.bag(ids, offsets)

    model = Model().cuda().eval()
    ids = torch.randint(0, 5000, (3000,), device="cuda")
    # float16 rounds the offset 2501 to 2500, which moves an id to another bag.
    offsets = torch.tensor([0, 2501], device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (ids, offsets)),
        arg_inputs=[ids, offsets],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    torch.testing.assert_close(
        trt_model(ids, offsets).float(), model(ids, offsets), rtol=1e-2, atol=1e-2
    )


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_attention_with_integer_mask():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(16, 48)
            self.out = nn.Linear(16, 16)

        def forward(self, x, mask):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            bias = (1.0 - mask[:, None, :].to(x.dtype)) * -10000.0
            scores = q @ k.transpose(-1, -2) / 4.0 + bias
            return self.out(scores.softmax(dim=-1) @ v)

    model = Model().cuda().eval()
    mask = torch.ones(2, 8, dtype=torch.int64, device="cuda")
    mask[:, 6:] = 0
    inputs = (torch.randn(2, 8, 16, device="cuda"), mask)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs),
        arg_inputs=list(inputs),
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [inputs], batch_size=None
        ),
    )
    torch.testing.assert_close(
        trt_model(*inputs).float(), model(*inputs), rtol=1e-2, atol=1e-2
    )


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize("calibrate", [True, False], ids=["calib", "no_calib"])
def test_autocast_combines_masks_from_topk(calibrate):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)

        def forward(self, x):
            values, indices = (x * 2).topk(8, dim=-1)
            keep = (indices > 3001) & (values > 0)
            return self.linear(torch.where(keep, values, 0.0))

    def batch():
        # Distinct values that float16 holds exactly, so topk picks the same ones.
        return (
            torch.stack([torch.randperm(4096) for _ in range(4)]).cuda() - 2048
        ) / 2048

    model = Model().cuda().eval()
    batches = [(batch(),), (batch(),)]
    settings = {}
    if calibrate:
        settings["autocast_calibration_dataloader"] = torch.utils.data.DataLoader(
            batches, batch_size=None
        )
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, batches[0]),
        arg_inputs=list(batches[0]),
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        **settings,
    )
    for x in batches:
        torch.testing.assert_close(
            trt_model(*x).float(), model(*x), rtol=1e-2, atol=1e-2
        )


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "masked",
    [
        lambda x, mask: torch.where(
            ~torch.zeros(x.shape, dtype=torch.bool, device=x.device), x, 0.0
        ),
        lambda x, mask: x.masked_fill(mask, 0.0),
        lambda x, mask: torch.where(mask, x, -x),
        lambda x, mask: x + mask * -2.0,
        lambda x, mask: x * torch.where(x > 0, 1.0, 0.5),
    ],
    ids=["factory_mask", "masked_fill", "where", "mask_times_scalar", "comparison"],
)
def test_autocast_keeps_bool_tensors_bool(masked):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)
            self.conv = nn.Conv1d(1, 4, 3, padding=1)

        def forward(self, x, mask):
            y = masked(self.linear(x), mask)
            return self.conv(y[:, None, :])

    model = Model().cuda().eval()
    inputs = (torch.randn(4, 16, device="cuda"), torch.rand(4, 16, device="cuda") > 0.5)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs),
        arg_inputs=list(inputs),
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [inputs], batch_size=None
        ),
    )
    torch.testing.assert_close(
        trt_model(*inputs).float(), model(*inputs), rtol=1e-2, atol=1e-2
    )


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_gives_autocast_regions_their_traced_dtypes():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.linear = nn.Linear(64, 16)

        def forward(self, x):
            y = self.conv(x).flatten(2)
            with torch.autocast(x.device.type, enabled=False):
                y = nn.functional.normalize(y.float(), dim=1)
                sim = torch.einsum("bci,bcj->bij", y, y)
            return self.linear(sim)

    model = Model().cuda().eval()
    x = torch.randn(2, 3, 8, 8, device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "given",
    [
        lambda y: y.cumsum(2, dtype=torch.float32) / 16,
        lambda y: y + torch.arange(y.shape[2], device=y.device).to(torch.float32) / 16,
        lambda y: nn.functional.normalize(y.float(), dim=1),
    ],
    ids=["cumsum_dtype", "to_dtype", "float"],
)
def test_autocast_keeps_the_dtype_an_op_is_given(given):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 4, 3, padding=1)
            self.conv2 = nn.Conv1d(4, 4, 3, padding=1)

        def forward(self, x):
            return self.conv2(given(self.conv1(x)))

    model = Model().cuda().eval()
    x = torch.randn(2, 4, 16, device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_keeps_tiny_constants_nonzero():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 8, 3, padding=1)

        def forward(self, x):
            # F.normalize divides by max(norm, 1e-12), and 1e-12 rounds to 0 in float16.
            return self.conv(nn.functional.normalize(x, dim=1))

    model = Model().cuda().eval()
    x = torch.randn(1, 16, 16, 16, device="cuda")
    x[..., :4] = 0
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "scale",
    [lambda y: y * 1e-12, lambda y: y / 1e12],
    ids=["mul", "div"],
)
def test_autocast_leaves_tiny_factors_alone(scale):
    class Model(nn.Module):
        def forward(self, x):
            return scale(x + 1.0)

    model = Model().cuda().eval()
    # Below the default output threshold of 512, so the scaling runs in float16.
    x = torch.rand(4, 16, device="cuda") * 400 + 100
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    # In float16 the result is 0, as in eager; it must not become x * 6e-8.
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast_keeps_the_upcast_a_view_needs(dtype):
    class Model(nn.Module):
        def forward(self, x):
            # x[::2] is strided; its .float() copy is contiguous, so view(-1) works.
            return x[::2].float().view(-1) * 2

    model = Model().cuda().eval()
    x = torch.randn(4, 4, device="cuda", dtype=dtype)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=dtype,
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_keeps_the_upcast_an_in_place_op_writes():
    class Model(nn.Module):
        def forward(self, x):
            # add_ writes into the float32 copy, so x itself must stay unchanged.
            y = x.float()
            y.add_(1)
            return y * 2, x * 3

    model = Model().cuda().eval()
    x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float16)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x.clone(),)),
        arg_inputs=[x.clone()],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    doubled, tripled = trt_model(x.clone())
    torch.testing.assert_close(doubled.float(), torch.tensor([4.0, 6.0], device="cuda"))
    torch.testing.assert_close(tripled.float(), torch.tensor([3.0, 6.0], device="cuda"))
