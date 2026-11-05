# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import base64
import logging

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
        def __init__(self):
            super(MaskedScores, self).__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x, mask):
            # -1e9 is out of the float16 range, so the calibration data keeps
            # these nodes in float32.
            return torch.softmax(self.linear(x) + (1.0 - mask) * -1e9, dim=-1)

    inputs = (
        torch.randn((2, 8, 16), dtype=torch.float32, device="cuda"),
        (torch.rand((2, 8, 16), device="cuda") > 0.3).float(),
    )
    batch = torch.randn((64, 8, 16), dtype=torch.float32, device="cuda")
    batch_mask = (torch.rand((64, 8, 16), device="cuda") > 0.3).float()
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(batch, batch_mask), batch_size=2
    )
    trt_autocast_mod = torch_tensorrt.dynamo.compile(
        torch.export.export(MaskedScores().cuda().eval(), inputs),
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
    assert engine.settings.autocast_calibration_dataloader is not None

    # Refit of the loaded program must pick the precisions the engine was built with.
    path = tmp_path / "model.ep"
    torch_tensorrt.save(trt_autocast_mod, path, arg_inputs=list(inputs))
    new_model = MaskedScores().cuda().eval()
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
