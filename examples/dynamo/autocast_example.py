import torch
import torch.nn as nn
import torch_tensorrt


class AutocastExample(nn.Module):
    def __init__(self):
        super(AutocastExample, self).__init__()
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
        x = self.conv1(x)  # fp32 because of "^conv1$" in `autocast_excluded_nodes`
        x = self.relu1(x)  # fp32 because of "relu" in `autocast_excluded_nodes`
        out = self.pool1(x)  # fp16
        x = self.conv2(out)  # fp16
        x = self.relu2(x)  # fp32 because of "relu" in `autocast_excluded_nodes`
        x = self.pool2(x)  # fp16
        x = self.flatten(
            x
        )  # fp32 because of `torch.ops.aten.flatten.using_ints` in `autocast_excluded_ops`
        # Respect the precisions in the pytorch autocast context
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float32):
            x = self.fc1(x)
            with torch.autocast(x.device.type, enabled=False):
                x = torch.sub(x.half(), y)
                out2 = torch.add(x, x)
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
            out2 = torch.log(out2)
        return x, out, out2


if __name__ == "__main__":
    model = AutocastExample().cuda().eval()
    inputs = (
        torch.randn((1, 3, 32, 32), dtype=torch.float32, device="cuda"),
        torch.randn((1,), dtype=torch.float16, device="cuda"),
    )

    ep = torch.export.export(model, inputs)

    with torch_tensorrt.dynamo.Debugger(
        "graphs",
        logging_dir=".",
        engine_builder_monitor=False,
    ):
        trt_mod = torch_tensorrt.compile(
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
            autocast_excluded_ops={torch.ops.aten.flatten.using_ints},
            autocast_data_max=512,
            autocast_max_depth_of_reduction=None,
        )

        trt_out = trt_mod(*inputs)
