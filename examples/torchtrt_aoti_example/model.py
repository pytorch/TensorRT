import os
import torch
import torch_tensorrt


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs = (torch.randn(8, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    # [Optional] Specify the first dimension of the input x as dynamic.
    exported = torch.export.export(
        model, example_inputs, dynamic_shapes={"x": {0: batch_dim}}
    )
    # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
    # Depending on your use case, e.g. if your training platform and inference platform
    # are different, you may choose to save the exported model using torch.export.save and
    # then load it back using torch.export.load on your inference platform to run AOT compilation.
    compile_settings = {
        "arg_inputs": [
            torch_tensorrt.Input(
                min_shape=(1, 10),
                opt_shape=(8, 10),
                max_shape=(1014, 10),
                dtype=torch.float32,
            )
        ],
        "enabled_precisions": {torch.float32},
        "min_block_size": 1,
    }
    cg_trt_module = torch_tensorrt.dynamo.compile(exported, **compile_settings)
    torch_tensorrt.save(
        cg_trt_module,
        file_path=os.path.join(os.getcwd(), "model.pt2"),
        output_format="aot_inductor",
        retrace=True,
        arg_inputs=example_inputs,
    )

    # output_path = torch._inductor.aoti_compile_and_package(
    #     exported,
    #     # [Optional] Specify the generated shared library path. If not specified,
    #     # the generated artifact is stored in your system temp directory.
    #     package_path=os.path.join(os.getcwd(), "model.pt2"),
    # )
