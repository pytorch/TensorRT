import pytest
import torch
import torch.nn as nn
import torch_tensorrt


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
            use_python_runtime=True,
            use_explicit_typing=True,
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
            use_python_runtime=True,
            use_explicit_typing=True,
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
            use_python_runtime=False,
            use_explicit_typing=True,
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
            use_python_runtime=False,
            use_explicit_typing=True,
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
