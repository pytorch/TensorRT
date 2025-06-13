import torch
import torch.nn as nn
import torch_tensorrt


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
    torch_ex_input = torch.randn([1, 3, 5, 5], device="cuda")
    compile_settings = {
        "arg_inputs": [torch_ex_input],
        "ir": "dynamo",
        "enabled_precisions": {torch.float32},
        "min_block_size": 1,
    }

    cg_trt_module = torch_tensorrt.compile(model, **compile_settings)
    torch_tensorrt.save(
        cg_trt_module,
        file_path="torchtrt_aoti_conv_gelu.pt2",
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=[torch_ex_input],
    )

    norm_model = Norm().eval().cuda()
    norm_trt_module = torch_tensorrt.compile(norm_model, **compile_settings)
    torch_tensorrt.save(
        norm_trt_module,
        file_path="torchtrt_aoti_norm.pt2",
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=[torch_ex_input],
    )
    print("Generated TorchTRT-AOTI models.")

    loaded_cg_trt_module = torch._inductor.aoti_load_package(
        "torchtrt_aoti_conv_gelu.pt2"
    )
    loaded_norm_trt_module = torch._inductor.aoti_load_package("torchtrt_aoti_norm.pt2")
    with torch.inference_mode():
        print(loaded_cg_trt_module(torch_ex_input))
        print(loaded_norm_trt_module(torch_ex_input))


if __name__ == "__main__":
    main()
