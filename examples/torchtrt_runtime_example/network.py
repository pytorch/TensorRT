import torch
import torch.nn as nn
import torch_tensorrt as torchtrt

# create a simple norm layer.
# This norm layer uses NormalizePlugin from Torch-TensorRT
class Norm(torch.nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        return torch.norm(x, 2, None, False)


# Create a sample network with a conv and gelu node.
# Gelu layer in Torch-TensorRT is converted to CustomGeluPluginDynamic from TensorRT plugin registry.
class ConvGelu(torch.nn.Module):
    def __init__(self):
        super(ConvGelu, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        return x


def main():

    model = ConvGelu().eval().cuda()
    scripted_model = torch.jit.script(model)

    compile_settings = {
        "inputs": [torchtrt.Input([1, 3, 5, 5])],
        "enabled_precisions": {torch.float32},
    }

    trt_ts_module = torchtrt.compile(scripted_model, **compile_settings)
    torch.jit.save(trt_ts_module, "conv_gelu.jit")

    norm_model = Norm().eval().cuda()
    norm_ts_module = torch.jit.script(norm_model)
    norm_trt_ts = torchtrt.compile(norm_ts_module, **compile_settings)
    torch.jit.save(norm_trt_ts, "norm.jit")
    print("Generated Torchscript-TRT models.")


if __name__ == "__main__":
    main()
