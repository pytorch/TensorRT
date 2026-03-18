import torch
import torch_tensorrt


class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


with torch.no_grad():
    model = MyModel().eval().cuda()
    example_input = (torch.randn((2, 3, 4, 4)).cuda(),)

    exported_program = torch.export.export(model, example_input)
    compile_settings = {
        "arg_inputs": [
            torch_tensorrt.Input(shape=(2, 3, 4, 4), dtype=torch.float32),
        ],
        "min_block_size": 1,
    }
    trt_gm = torch_tensorrt.dynamo.compile(exported_program, **compile_settings)

    # Save as ExecuTorch .pte format (loadable by ExecuTorch runtime)
    torch_tensorrt.save(
        trt_gm, "model.pte", output_format="executorch", arg_inputs=example_input
    )
