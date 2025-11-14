import torch
import torch.nn as nn
import torch_tensorrt


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
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
            x = self.fc1(x)
            out = torch.log(
                torch.abs(x) + 1
            )  # log is fp32 due to Pytorch Autocast requirements
        return out


if __name__ == "__main__":
    model = MixedPytorchAutocastModel().cuda().eval()
    inputs = (torch.randn((8, 3, 32, 32), dtype=torch.float32, device="cuda"),)
    ep = torch.export.export(model, inputs)
    calibration_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*inputs), batch_size=2, shuffle=False
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
            ##### weak typing #####
            # use_explicit_typing=False,
            # enabled_precisions={torch.float16},
            ##### strong typing + autocast #####
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
