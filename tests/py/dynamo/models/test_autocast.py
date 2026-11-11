# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import base64
import copy
import logging
import operator
import pickle
import warnings

import pytest
import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import pre_export_lowering
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import _copy_program
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    trace_intermediate_node_outputs,
)
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
def test_autocast_calibrates_0_dim_values_across_batches():
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
def test_autocast_depth_rule_reads_one_calibration_batch():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8, 4))

        def forward(self, x):
            # The matmul reduces over 8 values, whatever the number of batches.
            return (x.amax(dim=(0, 1)) * 1e-3) @ self.weight

    x = torch.randn(2, 3, 8, device="cuda")
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_max_depth_of_reduction=10,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,), (x,)], batch_size=None
        ),
    )
    pre_export_lowering(torch.export.export(Model().cuda().eval(), (x,)), settings)
    assert "matmul" not in settings._autocast_calibrated_high_precision_nodes


@pytest.mark.unit
@pytest.mark.parametrize("long_first", [True, False], ids=["long_first", "short_first"])
def test_autocast_depth_rule_sees_the_longest_calibration_batch(long_first):
    class Attention(nn.Module):
        def forward(self, q, k, v):
            # The second matmul reduces over the sequence length.
            return torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v

    def batch(length, scale):
        q = torch.randn(1, length, 8, device="cuda") * scale
        return q, torch.randn(1, length, 8, device="cuda"), torch.randn_like(q)

    # The short batch has the larger probabilities, the long one the deeper reduction.
    short, long = batch(4, 3.0), batch(32, 0.01)
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_max_depth_of_reduction=16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [long, short] if long_first else [short, long], batch_size=None
        ),
    )
    length = torch.export.Dim("length", min=2, max=64)
    exported = torch.export.export(
        Attention().cuda().eval(),
        short,
        dynamic_shapes=({1: length}, {1: length}, {1: length}),
    )
    pre_export_lowering(exported, settings)
    high_precision_nodes = settings._autocast_calibrated_high_precision_nodes
    assert "matmul" not in high_precision_nodes
    assert "matmul_1" in high_precision_nodes


@pytest.mark.unit
@pytest.mark.parametrize("max_depth_of_reduction", [None, 2])
def test_autocast_calibration_of_outputs_of_different_ranks(max_depth_of_reduction):
    class Model(nn.Module):
        def forward(self, x):
            return torch.squeeze(x, 1) @ torch.ones(4, 4, device=x.device)

    dim = torch.export.Dim("dim", min=1, max=8)
    batches = [
        (torch.randn(2, 1, 4, device="cuda"),),
        (torch.randn(2, 3, 4, device="cuda"),),
    ]
    exported = torch.export.export(
        Model().eval(),
        (torch.randn(2, 3, 4, device="cuda"),),
        dynamic_shapes=({1: dim},),
    )
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_max_depth_of_reduction=max_depth_of_reduction,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            batches, batch_size=None
        ),
    )
    # squeeze gives shape (2, 4) for the first batch and (2, 3, 4) for the second, so
    # the depth rule cannot tell which dims matmul reduces. The range rule needs no
    # shapes.
    if max_depth_of_reduction is None:
        pre_export_lowering(exported, settings)
    else:
        with pytest.raises(ValueError, match="different ranks"):
            pre_export_lowering(exported, settings)


@pytest.mark.unit
def test_autocast_calibration_measures_each_batch_as_it_runs():
    class Model(nn.Module):
        def forward(self, x):
            # x.float() returns x itself, so its value is the batch tensor.
            return x.float() * 1e-5

    class RefilledBuffer(torch.utils.data.IterableDataset):
        # A data loader can fill one buffer for every batch.
        def __iter__(self):
            buffer = torch.empty(4, device="cuda")
            for value in (1e5, 1.0, 1.0):
                buffer.fill_(value)
                yield (buffer,)

    model = Model().eval()
    inputs = [torch.full((4,), 1e5, device="cuda")]
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, tuple(inputs)),
        arg_inputs=inputs,
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            RefilledBuffer(), batch_size=None
        ),
    )
    # In float16, the multiply would read inf for an input of 1e5.
    torch.testing.assert_close(trt_model(*inputs), model(*inputs))


@pytest.mark.unit
def test_autocast_calibration_measures_a_buffer_view_before_its_update():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(2, 4))

        def forward(self, x):
            past = self.cache[0]
            out = past * 2 + x
            # Like a static cache, the buffer changes later in the same forward.
            self.cache[0].copy_(x * 100)
            return out

    x = torch.ones(4, device="cuda")
    magnitudes, _ = trace_intermediate_node_outputs(
        torch.export.export(Model().cuda().eval(), (x,)).module(), [(x,)]
    )
    assert magnitudes["select"] == 0.0


def _calibrate_with_a_late_rank(rank: int, store: str) -> None:
    import time

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", store=dist.FileStore(store, 2), rank=rank, world_size=2
    )
    group = dist.group.WORLD.group_name

    class Model(nn.Module):
        def forward(self, x):
            y = torch.ops._c10d_functional.all_reduce.default(x, "sum", group)
            return torch.ops._c10d_functional.wait_tensor.default(y) * 1e-5

    x = torch.tensor([1.0 if rank == 0 else 1e5])
    gm = torch.export.export(Model(), (x,)).module()
    if rank == 1:
        # Rank 0 runs the all_reduce first and has to wait for this rank to join.
        time.sleep(1)
    magnitudes, _ = trace_intermediate_node_outputs(gm, [(x,)])
    dist.destroy_process_group()
    assert magnitudes["all_reduce"] == 100001.0, magnitudes


@pytest.mark.unit
def test_autocast_calibration_measures_a_collective_once_it_is_done(tmp_path):
    torch.multiprocessing.spawn(
        _calibrate_with_a_late_rank, args=(str(tmp_path / "store"),), nprocs=2
    )


@pytest.mark.unit
def test_autocast_does_not_cast_a_collective_before_its_wait():
    from torch_tensorrt.dynamo.lowering.passes.rule_based_autocast import (
        rule_based_autocast,
    )

    class Model(nn.Module):
        def forward(self, x):
            y = torch.ops._c10d_functional.all_reduce.default(x, "sum", "0")
            return torch.ops._c10d_functional.wait_tensor.default(y) * 2.0

    gm = torch.export.export(Model(), (torch.randn(4, 8),)).graph_module
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_excluded_ops={torch.ops._c10d_functional.all_reduce.default},
    )
    gm = rule_based_autocast(gm, settings)
    wait = next(
        n
        for n in gm.graph.nodes
        if n.target == torch.ops._c10d_functional.wait_tensor.default
    )
    # The all_reduce stays in float32, and the wait must get its output uncast.
    assert wait.args[0].target == torch.ops._c10d_functional.all_reduce.default


@pytest.mark.unit
def test_autocast_calibration_waits_for_the_gpu_once_per_batch():
    # Reading each node's largest value as it runs would make the host wait for the
    # GPU at every node of every batch.
    class Model(nn.Module):
        def forward(self, x):
            return torch.relu(x @ x.T) * 2 + x.sum()

    x = torch.randn(8, 8, device="cuda")
    gm = torch.export.export(Model().eval(), (x,)).module()
    batches = [(torch.randn(8, 8, device="cuda"),) for _ in range(3)]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            magnitudes, _ = trace_intermediate_node_outputs(gm, batches)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    syncs = [str(w.message) for w in caught if "synchroniz" in str(w.message)]
    assert len(syncs) == len(batches), syncs
    assert magnitudes["relu"] == max(
        float(torch.relu(b @ b.T).max()) for (b,) in batches
    )


@pytest.mark.unit
def test_autocast_calibration_reads_a_batch_before_the_loader_refills_it():
    class Model(nn.Module):
        def forward(self, x):
            return x.to("cuda", non_blocking=True) * 1e-5

    class PinnedRefill(torch.utils.data.IterableDataset):
        # A loader can refill one pinned buffer on the host for every batch. The copy
        # to the GPU waits behind the work queued before it, so it reads the next
        # batch unless calibration waits for the GPU before asking for one.
        def __iter__(self):
            buffer = torch.empty(4, pin_memory=True)
            busy = torch.full((2048, 2048), 2048**-1, device="cuda")
            for value in (1e5, 1.0, 1.0):
                buffer.fill_(value)
                for _ in range(10):
                    busy = busy @ busy
                yield (buffer,)

    gm = torch.export.export(Model().eval(), (torch.empty(4, pin_memory=True),))
    loader = torch.utils.data.DataLoader(PinnedRefill(), batch_size=None)
    # The first run in a process can wait for the GPU while CUDA sets up, which
    # lets the copy finish before the refill anyway.
    for _ in range(2):
        magnitudes, _ = trace_intermediate_node_outputs(gm.module(), loader)
        assert magnitudes["mul"] == 1.0


@pytest.mark.unit
def test_autocast_calibration_counts_an_overflow_to_inf():
    class Model(nn.Module):
        def forward(self, x):
            return x * 2 * 0.5

    # In float16, x * 2 is the largest finite value in the first batch and inf in
    # the second. The range rule keeps the op that reads it in high precision.
    first, second = (
        torch.full((4,), v, dtype=torch.float16, device="cuda")
        for v in (32752.0, 40000.0)
    )
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_max_output_threshold=65504,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(first,), (second,)], batch_size=None
        ),
    )
    pre_export_lowering(torch.export.export(Model().eval(), (first,)), settings)
    assert "mul_1" in settings._autocast_calibrated_high_precision_nodes


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype", [torch.float8_e5m2, torch.uint16], ids=["float8", "uint16"]
)
def test_autocast_calibrates_types_without_abs(dtype):
    # float8_e5m2 has no nan_to_num or amax kernel, and uint16 none for abs or amax.
    class Model(nn.Module):
        def forward(self, x):
            return x.to(dtype).float() * 2

    x = torch.tensor([0.0, 1.0, 1024.0, 3.0], device="cuda")
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
    )
    pre_export_lowering(torch.export.export(Model().eval(), (x,)), settings)
    # 1024 is above the default threshold of 512, so the multiply stays in float32.
    assert "mul" in settings._autocast_calibrated_high_precision_nodes


def _quantize_float8(x):
    scale = x.abs().amax() / 448
    return [(x / scale).to(torch.float8_e4m3fn), scale]


@pytest.mark.unit
def test_autocast_calibrates_a_list_of_float8_data_and_its_scale():
    # Calibration gives the graph each batch as one tuple.
    graph = torch.fx.Graph()
    x = graph.call_function(operator.getitem, (graph.placeholder("batch"), 0))
    out = graph.call_function(_quantize_float8, (x,))
    data, scale = (graph.call_function(operator.getitem, (out, i)) for i in (0, 1))
    graph.output(
        graph.call_function(torch.mul, (graph.call_method("float", (data,)), scale))
    )
    gm = torch.fx.GraphModule(nn.Module(), graph)
    x = torch.tensor([1.0, -2000.0, 3.0])
    magnitudes, _ = trace_intermediate_node_outputs(gm, [(x,)])
    # The data reaches 448, the largest float8_e4m3fn value, and the scale is 4.5.
    assert magnitudes["_quantize_float8"] == 448


@pytest.mark.unit
def test_autocast_calibrates_a_cpu_program_with_a_bool_mask():
    class Model(nn.Module):
        def forward(self, x, mask):
            return x * torch.chunk(mask, 2)[0].float()

    # The program and the calibration batch stay on the CPU.
    model = Model().eval()
    x, mask = torch.randn(4, 8), torch.rand(8, 8) > 0.5
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x, mask)),
        arg_inputs=[x, mask],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x, mask)], batch_size=None
        ),
    )
    out = trt_model(x.cuda(), mask.cuda())
    torch.testing.assert_close(out.float().cpu(), model(x, mask), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
def test_autocast_calibrates_a_list_output_on_two_devices():
    class Model(nn.Module):
        def forward(self, x, y):
            a, b = torch._foreach_add([x * 1.0, y * 1.0], 2.0)
            return a * 3.0, b * 3.0

    y = torch.randn(2, 4)
    y[0, 0] = 1000.0
    inputs = (torch.randn(2, 4, device="cuda"), y)
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [inputs], batch_size=None
        ),
    )
    pre_export_lowering(torch.export.export(Model().eval(), inputs), settings)
    # Only the CPU tensor holds a value above the default threshold of 512. The range
    # rule reads the list of _foreach_add at the multiply of the GPU input too.
    assert "mul" in settings._autocast_calibrated_high_precision_nodes


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
def test_autocast_runs_without_distributed_support(monkeypatch):
    from torch_tensorrt.dynamo.lowering.passes.rule_based_autocast import (
        rule_based_autocast,
    )

    class Model(nn.Module):
        def forward(self, x):
            return torch.relu(x) * 2.0

    x = torch.randn(4, 8)
    gm = torch.export.export(Model(), (x,)).module()
    # A PyTorch build without distributed support has no _c10d_functional ops.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    monkeypatch.setattr(torch.ops, "_c10d_functional", object())
    trace_intermediate_node_outputs(gm, [(x,)])
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
def test_autocast_leaves_the_exported_program_unchanged():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        # Named like a Python builtin, which a plain copy of the graph renames.
        def forward(self, input):
            return torch.relu(self.conv(input))

    x = torch.randn(1, 3, 16, 16, device="cuda")
    ep = torch.export.export(Model().cuda().eval(), (x,))
    graph = ep.graph_module.print_readable(print_output=False)
    torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    assert ep.graph_module.print_readable(print_output=False) == graph


@pytest.mark.unit
def test_copy_program_keeps_node_names():
    class Model(nn.Module):
        # Named like a Python builtin. A plain copy of the graph renames it to max_1,
        # which moves the node of x.max() from max_1 to max_2.
        def forward(self, x, max):
            return torch.minimum(x, max) * x.max()

    ep = torch.export.export(Model(), (torch.randn(4), torch.randn(4)))
    names = [n.name for n in ep.graph.nodes]
    assert [n.name for n in _copy_program(ep).graph.nodes] == names


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
def test_autocast_excluded_node_names():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)
            self.norm = nn.LayerNorm(16)

        def forward(self, x):
            y = self.norm(self.linear(x)) * 2.0
            return nn.functional.layer_norm(y, (16,))

    x = torch.randn(4, 16, device="cuda")

    def dtypes(**kwargs):
        settings = CompilationSettings(
            enable_autocast=True, autocast_low_precision_type=torch.float16, **kwargs
        )
        ep = pre_export_lowering(
            torch.export.export(Model().cuda().eval(), (x,)), settings
        )
        return {
            n.name: n.meta["val"].dtype
            for n in ep.graph.nodes
            if n.op == "call_function" and isinstance(n.meta.get("val"), torch.Tensor)
        }

    # The nodes layer_norm and layer_norm_1, whose modules are norm and the model.
    by_name = dtypes(autocast_excluded_node_names={".*layer_norm.*"})
    assert by_name["layer_norm"] == by_name["layer_norm_1"] == torch.float32
    assert by_name["linear"] == by_name["mul"] == torch.float16
    # A single pattern given as a string is one pattern, not one per character.
    assert dtypes(autocast_excluded_node_names=".*layer_norm.*") == by_name
    # autocast_excluded_nodes matches only module paths, which here are norm and "".
    by_path = dtypes(autocast_excluded_nodes={".*layer_norm.*"})
    assert by_path["layer_norm"] == by_path["layer_norm_1"] == torch.float16


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
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        model.bag.weight.copy_(torch.randn(5000, 8, device="cuda", generator=generator))
    ids = torch.randint(0, 5000, (3000,), device="cuda", generator=generator)
    # float16 rounds the offset 2501 to 2500, which moves an id to another bag.
    offsets = torch.tensor([0, 2501], device="cuda")
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (ids, offsets)),
        arg_inputs=[ids, offsets],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    # Autocast runs the bag in float16, which sums float16 rows in float32 and rounds the
    # result. The float32 model differs from that by up to about 0.1 for a 2501-row bag,
    # so compare with PyTorch in float16. An id in the wrong bag changes an element by
    # about 1.6.
    expected = nn.functional.embedding_bag(
        ids, model.bag.weight.half(), offsets, mode="sum"
    )
    torch.testing.assert_close(
        trt_model(ids, offsets), expected, rtol=2**-10, atol=2**-10
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
def test_autocast_model_regions():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 64)
            self.norm = nn.LayerNorm(64)
            self.fc2 = nn.Linear(64, 64)

        def forward(self, x):
            with torch.autocast(x.device.type, dtype=torch.float16):
                h = self.norm(self.fc1(x) * 1000.0)
                with torch.autocast(x.device.type, enabled=False):
                    h = torch.softmax(h, dim=-1)
                y = self.fc2(h)
            return y.float() + x

    model = Model().cuda().eval()
    x = torch.randn(2, 8, 64, device="cuda")
    kwargs = dict(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
        autocast_excluded_nodes={"^fc2$"},
        autocast_model_regions="reclassify",
    )
    ep = pre_export_lowering(
        torch.export.export(model, (x,)), CompilationSettings(**kwargs)
    )
    vals = {n.target: n.meta.get("val") for n in ep.graph.nodes}
    # Calibration keeps the layer norm, which reads values above 512, in
    # float32.
    assert vals[torch.ops.aten.layer_norm.default].dtype == torch.float32
    # The last linear, fc2, is excluded by the path of its module.
    assert vals[torch.ops.aten.linear.default].dtype == torch.float32
    # Only the nested region with enabled=False is left, and its softmax still
    # runs in float32.
    (region,) = ep.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.wrap_with_autocast
    )
    assert region.args[2] is False
    body = getattr(ep.graph_module, region.args[4].target)
    (softmax,) = body.graph.find_nodes(
        op="call_function", target=torch.ops.aten.softmax.int
    )
    assert softmax.meta["val"].dtype == torch.float32

    # Without a low precision type autocast changes nothing, so the regions stay.
    settings = CompilationSettings(**{**kwargs, "autocast_low_precision_type": None})
    ep = pre_export_lowering(torch.export.export(model, (x,)), settings)
    assert torch.ops.higher_order.wrap_with_autocast in (
        n.target for n in ep.graph.nodes
    )

    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1, **kwargs
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
def test_inline_autocast_regions():
    from torch_tensorrt.dynamo.lowering.passes.inline_autocast_regions import (
        inline_autocast_regions,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.register_buffer("cache", torch.zeros(4, 8, dtype=torch.float16))

        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                # A 0-dim tensor leaves the division in the dtype of the linear.
                eight = torch.tensor(8.0, dtype=torch.float64, device=x.device)
                y = self.linear(x) / eight
                # lerp needs its inputs in one dtype, as they had under autocast.
                z = torch.lerp(y, self.cache[pos], 0.5)
            with torch.autocast(x.device.type, dtype=torch.float16):
                # The same 0-dim tensor leaves pow, which autocast casts, in float32.
                # matmul, which autocast casts too, gets the cache in float32.
                z = torch.pow(z, eight) @ self.cache.T
                # Export asserts here that z is float16, which the inlined z is not.
                return z.float()

    x = torch.randn(2, 8, device="cuda")
    pos = torch.tensor([1, 3], device="cuda")
    gm = torch.export.export(Model().cuda().eval(), (x, pos)).graph_module
    inline_autocast_regions(gm)
    nodes = {n.target: n for n in gm.graph.nodes}
    # Under the region the linear and the division ran in float16, now in float32
    # like the inputs of the linear. Autocast does not cast the inputs of a
    # division, so it keeps its float64 divisor.
    div = nodes[torch.ops.aten.div.Tensor]
    assert div.meta["val"].dtype == torch.float32
    assert div.args[1].meta["val"].dtype == torch.float64
    # lerp gets the cache in float32, like y.
    lerp = nodes[torch.ops.aten.lerp.Scalar]
    assert lerp.meta["val"].dtype == torch.float32
    # The second region gets z as the first now gives it, without a cast to float16.
    power = nodes[torch.ops.aten.pow.Tensor_Tensor]
    assert power.args[0] is lerp
    assert power.meta["val"].dtype == torch.float32
    matmul = nodes[torch.ops.aten.matmul.default]
    assert matmul.args[1].meta["val"].dtype == torch.float32
    assert not dict(gm.named_children())


@pytest.mark.unit
@pytest.mark.critical
def test_autocast_model_regions_with_mixed_dtypes():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.register_buffer("weight", torch.randn(64, 64).half() / 8)

        def forward(self, x):
            with torch.autocast(x.device.type, dtype=torch.float16):
                # Autocast gives both inputs of this matmul float16.
                y = self.linear(x) @ self.weight
            # A float16 matmul, so it needs y in float16 too.
            return (y @ self.weight).float()

    model = Model().cuda().eval()
    x = torch.randn(2, 8, 64, device="cuda")
    kwargs = dict(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_model_regions="reclassify",
        autocast_calibration_dataloader=torch.utils.data.DataLoader(
            [(x,)], batch_size=None
        ),
        autocast_excluded_nodes={"^linear$"},
    )
    ep = pre_export_lowering(
        torch.export.export(model, (x,)), CompilationSettings(**kwargs)
    )
    assert torch.ops.higher_order.wrap_with_autocast not in (
        n.target for n in ep.graph.nodes
    )

    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1, **kwargs
    )
    torch.testing.assert_close(trt_model(x).float(), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
def test_autocast_model_regions_casts_an_output_only_for_ops_outside():
    class Model(nn.Module):
        def forward(self, x):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x @ x
            with torch.autocast(x.device.type, dtype=torch.float16):
                z = (y @ x).float()
            # This op outside the regions gets y in float16, as before.
            return z, y.float()

    x = torch.full((2, 2), 256.0, device="cuda")
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_model_regions="reclassify",
        autocast_excluded_node_names={"matmul.*"},
    )
    ep = pre_export_lowering(torch.export.export(Model().eval(), (x,)), settings)
    # Both matmuls stay in float32. x @ x is 131072, above the float16 range, so the
    # second one must not read it through the cast for the op outside.
    z, y = ep.module()(x)
    torch.testing.assert_close(z, x @ x @ x)
    # The op outside gets y in float16, where 131072 is inf.
    torch.testing.assert_close(y, torch.full_like(x, float("inf")))


@torch.library.custom_op("autocast_regions_test::read_first_rows", mutates_args=())
def read_first_rows(t: torch.Tensor) -> torch.Tensor:
    # Reads from the start of the storage, which can lie before the view.
    return t.as_strided(t.shape, t.stride(), 0).clone()


@read_first_rows.register_fake
def _(t: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(t)


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "cache",
        "total",
        "read_view",
        "write_view",
        "dtype_view",
        "output_views",
        "foreach_output_views",
        "foreach_then_read",
        "nested_views",
        "view_of_input",
        "view_then_region_write",
        "read_only_views",
        "storage_read",
        "contiguous_copy",
        "storage_copy",
        "storage_in_place",
        "storage_custom",
        "storage_no_grad",
        "read_in_disabled",
        "nested",
        "after_inlined",
        "nested_float",
        "new_input",
        "read_then_write",
        "write_region_output",
        "to_copy",
        "pow",
        "fresh",
    ],
)
def test_autocast_model_regions_keeps_values(case):
    class Cache(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.register_buffer("cache", torch.zeros(4, 8, dtype=torch.float16))

        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.bfloat16):
                self.cache.index_copy_(0, pos, self.linear(x).half())
            return self.cache.float()

    class Total(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.register_buffer("total", torch.zeros(1, 8))

        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                self.total.add_(self.linear(x))
            return self.total.clone()

    class ReadView(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x @ x.T
                # The view taken before add_ sees what add_ writes into y.
                v = y[:, :2]
                y.add_(1.0)
                return v.float()

    class WriteView(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                y[:, :2].add_(1.0)
                return y.float()

    class DtypeView(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                return (x.T @ x).view(torch.int32).float()

    class OutputViews(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x @ x.T
                v = y[:, :1]
            v.add_(10.0)
            return y.float()

    class ForeachOutputViews(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[:2]
            # y sees what _foreach_add_ writes into v through a list.
            torch._foreach_add_([v], 10.0)
            return y.float()

    class ForeachThenRead(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                # _foreach_add_ returns nothing, so mul reads y itself.
                torch._foreach_add_([y], 1.0)
                return (y * 2.0).float()

    class NestedViews(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x @ x.T
                with torch.autocast(x.device.type, enabled=False):
                    v = y[:, :1]
            v.add_(10.0)
            return y.float()

    class ViewOfInput(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                v = x[:, :2]
            # v sees what add_ writes into x.
            x.add_(1.0)
            return v.float()

    class ViewThenRegionWrite(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.register_buffer("cache", torch.zeros(4, 8))

        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                v = self.cache[:2]
            # This region writes into the cache, and v sees the write.
            with torch.autocast(x.device.type, dtype=torch.float16):
                self.cache.index_copy_(0, pos, self.linear(x).float())
            return v.float()

    class ReadOnlyViews(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                a, b = y[:4], y[2:6]
            # Nothing writes into y, so its slices can be separate copies.
            return (a * b).float()

    class StorageRead(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[::2]
            # as_strided reads all of y through its view v.
            return torch.as_strided(v, (8, 8), (8, 1), 0).float()

    class ContiguousCopy(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[::2]
            # contiguous() copies the strided v, so c misses the write into v.
            c = v.contiguous()
            v.add_(1.0)
            return c.float()

    class StorageCopy(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[::2]
            return torch.as_strided_copy(v, (8, 8), (8, 1), 0).float()

    class StorageInPlace(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[::2]
            return v.as_strided_((8, 8), (8, 1), 0).float()

    class StorageCustom(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[2:6]
            return read_first_rows(v).float()

    class StorageNoGrad(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[::2]
            with torch.no_grad():
                return torch.as_strided(v, (8, 8), (8, 1), 0).float()

    class ReadInDisabled(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                y = x.T @ x
                v = y[::2]
            # This region reads only the elements of v, so v can be a separate copy.
            with torch.autocast(x.device.type, enabled=False):
                return v.float().softmax(-1)

    class Nested(Cache):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                h = self.linear(x)
                with torch.autocast(x.device.type, enabled=False):
                    self.cache.index_copy_(0, pos, h)
            return self.cache.float()

    class AfterInlined(Cache):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                h = self.linear(x)
            # This region keeps its autocast and needs h in float16, as before.
            with torch.autocast(x.device.type, dtype=torch.float16):
                self.cache.index_copy_(0, pos, h)
            return self.cache.float()

    class ToCopy(nn.Module):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                # The float() copy of the strided y[::2] is contiguous, so view works.
                y = x.T @ x
                return y[::2].float().view(-1)

    class Pow(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("exponent", torch.tensor(65505.0))

        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                # pow runs in float32, where -1 ** 65505 is -1. In float16 the
                # exponent would round to 65504, which is even.
                return torch.pow(x.sign().half(), self.exponent)

    class Fresh(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(8, 8)
            self.relu = nn.ReLU(inplace=True)
            self.linear2 = nn.Linear(8, 8)

        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                # In-place ops on new values that nothing else reads.
                h = self.linear2(self.relu(self.linear1(x)))
                h.add_(1.0).mul_(2.0)
            return h.float()

    class NewInput(Fresh):
        def forward(self, x, pos):
            # Only the region uses h, a new value, so add_ can write into a cast.
            h = self.linear1(x)
            with torch.autocast(x.device.type, dtype=torch.float16):
                h.add_(1.0)
                return self.linear2(h).float()

    class ReadThenWrite(Fresh):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                # norm and half() read y before div_ writes into it, and keep no
                # view of it.
                y = self.linear1(x).float()
                y.div_(y.norm(dim=-1, keepdim=True) + y.half().abs().max())
                return y

    class WriteRegionOutput(Fresh):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                h = self.linear1(x)
            with torch.autocast(x.device.type, dtype=torch.float16):
                return self.linear2(h.relu_()).float()

    class NestedFloat(Fresh):
        def forward(self, x, pos):
            with torch.autocast(x.device.type, dtype=torch.float16):
                h = self.linear1(x)
                # This region keeps its autocast. It gets h in float16, as traced,
                # also when linear1 runs in float32, so float() still copies h.
                with torch.autocast(x.device.type, enabled=False):
                    y = h.float()
                    y.add_(1.0)
                return (self.linear2(y) + h).float()

    models = {
        "cache": (
            Cache,
            {"autocast_excluded_ops": {torch.ops.aten.index_copy_.default}},
        ),
        "total": (Total, {}),
        "read_view": (ReadView, {"autocast_excluded_node_names": {"^add_$"}}),
        "write_view": (WriteView, {"autocast_excluded_node_names": {"^add_$"}}),
        "dtype_view": (DtypeView, {}),
        "output_views": (
            OutputViews,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "foreach_output_views": (
            ForeachOutputViews,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "foreach_then_read": (
            ForeachThenRead,
            {"autocast_excluded_node_names": {"matmul.*"}},
        ),
        "nested_views": (
            NestedViews,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "view_of_input": (
            ViewOfInput,
            {"autocast_excluded_ops": {torch.ops.aten.add_.Tensor}},
        ),
        "view_then_region_write": (ViewThenRegionWrite, {}),
        "read_only_views": (
            ReadOnlyViews,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "storage_read": (
            StorageRead,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "contiguous_copy": (
            ContiguousCopy,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "storage_copy": (
            StorageCopy,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "storage_in_place": (
            StorageInPlace,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "storage_custom": (
            StorageCustom,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "storage_no_grad": (
            StorageNoGrad,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "read_in_disabled": (
            ReadInDisabled,
            {"autocast_excluded_node_names": {"matmul.*", "slice.*"}},
        ),
        "nested": (
            Nested,
            {"autocast_excluded_ops": {torch.ops.aten.index_copy_.default}},
        ),
        "after_inlined": (AfterInlined, {}),
        "nested_float": (NestedFloat, {"autocast_excluded_node_names": {"^linear$"}}),
        "new_input": (NewInput, {}),
        "read_then_write": (ReadThenWrite, {}),
        "write_region_output": (WriteRegionOutput, {}),
        "to_copy": (ToCopy, {}),
        "pow": (Pow, {"autocast_excluded_ops": {torch.ops.aten.pow.Tensor_Tensor}}),
        "fresh": (Fresh, {}),
    }
    cls, kwargs = models[case]
    model = cls().cuda().eval()

    def inputs(i):
        x = torch.randn(
            1, 8, device="cuda", generator=torch.Generator("cuda").manual_seed(i)
        )
        return x, torch.tensor([i], device="cuda")

    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_model_regions="reclassify",
        **kwargs,
    )
    ep = pre_export_lowering(
        torch.export.export(copy.deepcopy(model), inputs(0)), settings
    )
    # A region keeps its autocast when inlining would change its values. Regions
    # with enabled=False (args[2]) always keep theirs.
    kept = any(
        n.target == torch.ops.higher_order.wrap_with_autocast and n.args[2]
        for n in ep.graph.nodes
    )
    inlined = (
        "new_input",
        "read_then_write",
        "write_region_output",
        "to_copy",
        "read_only_views",
        "nested_float",
        "read_in_disabled",
    )
    assert kept == (case not in (*inlined, "pow", "fresh"))
    lowered = ep.module()
    # A second call shows whether the first call's write into a buffer was kept.
    for i in range(2):
        torch.testing.assert_close(
            lowered(*inputs(i)).float(), model(*inputs(i)), rtol=1e-2, atol=1e-2
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern, dtype",
    [
        ("^matmul$", torch.float32),
        ("^matmul_1$", torch.float32),
        ("^mul$", torch.float16),
    ],
    ids=["name_in_region", "name_after_inlining", "other_name"],
)
def test_autocast_model_regions_excludes_nodes_by_name(pattern, dtype):
    class Model(nn.Module):
        def forward(self, x):
            with torch.autocast(x.device.type, dtype=torch.float16):
                return x @ x

    x = torch.randn(4, 4, device="cuda")
    settings = CompilationSettings(
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_model_regions="reclassify",
        autocast_excluded_node_names={pattern},
    )
    ep = pre_export_lowering(torch.export.export(Model(), (x,)), settings)
    (matmul,) = ep.graph.find_nodes(
        op="call_function", target=torch.ops.aten.matmul.default
    )
    # The region's output already has the name matmul, so inlining renames the op.
    assert matmul.name == "matmul_1"
    assert matmul.meta["val"].dtype == dtype


@pytest.mark.unit
def test_autocast_model_regions_rejects_unknown_values():
    assert CompilationSettings().autocast_model_regions == "keep"
    with pytest.raises(ValueError, match="autocast_model_regions"):
        CompilationSettings(autocast_model_regions="inline")


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
def test_autocast_keeps_the_input_dtype_of_a_dtype_view():
    class Model(nn.Module):
        def forward(self, x):
            return (x + 0).view(torch.int32)

    model = Model().cuda().eval()
    # Values that float16 holds exactly, so the float32 bits match eager.
    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )
    torch.testing.assert_close(trt_model(x), model(x))
    # x + 0 still runs in float16, so the view reads the bits of the rounded value.
    y = torch.randn(2, 4, device="cuda")
    torch.testing.assert_close(trt_model(y), y.half().float().view(torch.int32))


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


def _add_one_under_autocast(y, bits):
    with torch.autocast(y.device.type, dtype=torch.float16):
        y.add_(1.0)


def _add_one_under_no_grad(y, bits):
    with torch.no_grad():
        y.add_(1.0)


def _double_after_a_block(y, bits):
    _add_one_under_autocast(y, bits)
    y.mul_(2)


def _add_one_to_a_row_under_autocast(y, bits):
    # The block writes y only through y[0], so it returns nothing and nothing uses it.
    with torch.autocast(y.device.type, dtype=torch.float16):
        y[0].add_(1.0)


@pytest.mark.unit
@pytest.mark.critical
@pytest.mark.parametrize(
    "write",
    [
        lambda y, bits: bits.fill_(0),
        lambda y, bits: bits[0].fill_(0x3F800000),
        lambda y, bits: y[0].view(torch.int32).fill_(0x3F800000),
        lambda y, bits: y.mul_(2),
        # y[:1] * 3 runs in float16, and index_copy_, which runs in PyTorch below,
        # needs its source in the dtype of y.
        lambda y, bits: y.index_copy_(0, torch.tensor([1], device=y.device), y[:1] * 3),
        # The products run in float16, and addmm_, which runs in PyTorch below, needs
        # its matrices in the dtype of y.
        lambda y, bits: y.addmm_(y[:, :2] * 1, y[:2] * 2),
        _add_one_under_autocast,
        _add_one_under_no_grad,
        _double_after_a_block,
        lambda y, bits: torch._foreach_add_([y], 1.0),
        # y * 2 runs in float16, and _foreach_lerp_ needs it in the dtype of y.
        lambda y, bits: torch._foreach_lerp_([y], [y * 2], 0.5),
        _add_one_to_a_row_under_autocast,
    ],
    ids=[
        "view",
        "select_of_view",
        "view_of_select",
        "viewed_tensor",
        "index_copy",
        "addmm",
        "autocast_block",
        "no_grad_block",
        "after_block",
        "list",
        "list_with_a_source",
        "write_only_block",
    ],
)
def test_autocast_keeps_writes_into_the_memory_of_a_dtype_view(write):
    class Model(nn.Module):
        def forward(self, x):
            y = x + 0
            bits = y.view(torch.int32)
            # 0x3F800000 are the bits of 1.0 in float32.
            write(y, bits)
            return y * 2, bits + 0

    model = Model().cuda().eval()
    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x.clone(),)),
        arg_inputs=[x.clone()],
        min_block_size=1,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        torch_executed_ops={
            "torch.ops.aten.index_copy.default",
            "torch.ops.aten.addmm.default",
        },
    )
    doubled, bits = trt_model(x.clone())
    expected_doubled, expected_bits = model(x.clone())
    torch.testing.assert_close(doubled.float(), expected_doubled)
    torch.testing.assert_close(bits, expected_bits)
