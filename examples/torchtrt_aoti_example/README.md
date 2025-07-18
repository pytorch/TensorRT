# torchtrt_aoti_example

## An example application to run Torch-Tensorrt/AOT-Inductor Models in C++

This sample is a demonstration on how to use Torch-TensorRT runtime library `libtorchtrt_runtime.so` along with AOT-Inductor

### Generating AOT-Inductor modules with TRT Engines

The following command will generate `model.pt2` AOT-Inductor modules which contain TensorRT engines.

```sh
python model.py
```

### `torchtrt_aoti_example`
The main goal is to use Torch-TensorRT runtime library `libtorchtrt_runtime.so`, a lightweight library sufficient enough to deploy your AOT-Inductor programs containing TRT engines.

1) Install PyTorch and Torch-TensorRT

```sh
pip install torch torch-tensorrt
```

2) Generate a AOT-Inductor Model with TensorRT engines embedded.

```sh
python model.py
```

`model.py` generates a AOT-inductor package with TensorRT embedded for a simple MLP. It sets a dynamic input dimension for the batch dimension meaning that recompilation is not required for different input shapes

```py
model = Model().to(device=device)
example_inputs=(torch.randn(8, 10, device=device),)
batch_dim = torch.export.Dim("batch", min=1, max=1024)
# [Optional] Specify the first dimension of the input x as dynamic.
exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
# [Note] In this example we directly feed the exported module to aoti_compile_and_package.
# Depending on your use case, e.g. if your training platform and inference platform
# are different, you may choose to save the exported model using torch.export.save and
# then load it back using torch.export.load on your inference platform to run AOT compilation.
compile_settings = {
    "arg_inputs": [torch_tensorrt.Input(min_shape=(1, 10), opt_shape=(8,10), max_shape=(1014, 10), dtype=torch.float32)],
    "enabled_precisions": {torch.float32},
    "min_block_size": 1,
}
cg_trt_module = torch_tensorrt.dynamo.compile(exported, **compile_settings)
```

The module is saved using `torch_tensorrt.save` that is a convience wrapper around a `torch._inductor.aoti_compile_and_package`. More complex workflows are supported through direct use of `torch.export`, `torch._inductor.aoti_compile` and `torch._inductor.package.package_aoti`.

```py
torch_tensorrt.save(
    cg_trt_module,
    file_path=os.path.join(os.getcwd(), "model.pt2"),
    output_format="aot_inductor",
    retrace=True,
    arg_inputs=example_inputs,
)
```

2) Run `inference.py`

`inference.py` is a simple python script to load and run `model.pt2`

> **NOTE**: Remember that while no torch_tensorrt APIs will get used in `inference.py` it is important to import torch_tensorrt as it will load the runtime extentions for PyTorch.

3) Build and run `torchtrt_aoti_example`

```sh
make
```

`torchtrt_aoti_example` is a binary which loads the `model.pt2` that is generated by `model.py`. It runs the binary once with 8x10 inputs and once with 1x10 inputs to demonstrate the dynamic shape behavior.

The Makefile will print a list of directories to add to your `LD_LIBRARY_PATH` so that the binary can link against `pytorch`, `tensorrt` and `torch_tensorrt`
