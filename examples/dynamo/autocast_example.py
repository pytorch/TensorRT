import torch
import torch.nn as nn
import torch_tensorrt
import torchvision


class MyModule(torch.nn.Module):
    def forward(self, a_float32, b_float32, c_float32, d_float32):
        with torch.autocast(device_type="cuda"):
            e_float16 = torch.mm(a_float32, b_float32)
            with torch.autocast(device_type="cuda", enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = torch.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the autocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = torch.mm(d_float32, f_float32)
        return g_float16


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
        out = self.pool1(self.relu1(self.conv1(x)))  # fp16
        x = self.pool2(self.relu2(self.conv2(out)))  # fp16
        x = self.flatten(x)
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float32):
            x = self.fc1(x)  # fp32
            with torch.autocast(x.device.type, enabled=False):
                x = torch.sub(x.half(), y)  # fp16
                out2 = torch.add(x, x)  # fp16
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
            out2 = torch.log(out2)  # fp32
        return x, out, out2


class MyResNet18Wrapper(torch.nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(MyResNet18Wrapper, self).__init__()
        self.resnet = torchvision.models.resnet18(
            num_classes=num_classes, weights="IMAGENET1K_V1" if pretrained else None
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    # model = MyModule().cuda().eval()
    # inputs = (torch.randn((8, 8), device="cuda"),
    #           torch.randn((8, 8), device="cuda"),
    #           torch.randn((8, 8), device="cuda"),
    #           torch.randn((8, 8), device="cuda"),)

    # model = AutocastExample().cuda().eval()
    # inputs = (torch.randn((1, 3, 32, 32), dtype=torch.float32, device="cuda"),
    #           torch.randn((1,), dtype=torch.float16, device="cuda"),)

    model = MyResNet18Wrapper().cuda().eval()
    inputs = (torch.randn((1, 3, 224, 224), dtype=torch.float32, device="cuda"),)

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
            low_precision_type=torch.float16,
            # nodes_to_exclude={"^conv2d$"},
            targets_to_exclude={},
            data_max=512,
            max_depth_of_reduction=None,
        )

        trt_out = trt_mod(*inputs)
