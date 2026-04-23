# ExecuTorch Export for Torch-TensorRT

Torch-TensorRT can export TRT-backed `ExportedProgram`s to ExecuTorch `.pte`
files. The TensorRT engine payload is embedded into the `.pte` and executed
through the TensorRT ExecuTorch backend at runtime.

## Requirements

- Linux
- `executorch` installed in the Python environment
- Torch-TensorRT built with `torch_tensorrt_runtime`
- NVIDIA GPU runtime dependencies available at inference time

## Installation

Install Torch-TensorRT with the ExecuTorch extra:

```bash
pip install -e ".[executorch]"
```

If your local ExecuTorch source tree lives outside the active Python
environment, point builds at it explicitly:

```bash
export EXECUTORCH_PATH=/home/lanl/git/executorch
```

## Quickstart

Export, compile with TensorRT, and save as `.pte`:

```python
import torch
import torch_tensorrt

class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)

model = MyModel().eval().cuda()
example_input = (torch.randn(1, 3, 224, 224, device="cuda"),)

exported_program = torch.export.export(model, example_input)
compile_settings = {
    "arg_inputs": [
        torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float32),
    ],
    "min_block_size": 1,
}
trt_gm = torch_tensorrt.dynamo.compile(exported_program, **compile_settings)

torch_tensorrt.save(
    trt_gm,
    "model.pte",
    output_format="executorch",
    arg_inputs=example_input,
    retrace=False,
)
```

## Lowering API

If you want direct ExecuTorch control, use the public helpers:

```python
from executorch.exir import to_edge_transform_and_lower
from torch_tensorrt.executorch import (
    TensorRTPartitioner,
    get_edge_compile_config,
)

edge_program = to_edge_transform_and_lower(
    trt_program,
    partitioner=[TensorRTPartitioner()],
    compile_config=get_edge_compile_config(),
)
executorch_program = edge_program.to_executorch()
with open("model.pte", "wb") as f:
    executorch_program.write_to_file(f)
```

## Notes

- Multi-engine `.pte` exports are supported, but each delegate boundary can add
  overhead.
- Engines that require an output allocator, such as data-dependent output shape
  cases, are rejected by the current export path.
- Additional ExecuTorch partitioners can be passed through
  `torch_tensorrt.save(..., partitioners=[...])`.

## C++ Runner

This repo includes a minimal C++ loader example in
`examples/torchtrt_executorch_example/executor_runner.cpp`.

The corresponding Bazel target statically links the TensorRT backend into the
application binary:

```bash
bazel build //examples/torchtrt_executorch_example:trt_executor_runner
```

There is also a matching CMake path for local source builds:

```bash
cmake -S . -B build-executorch \
  -DBUILD_TORCHTRT_EXECUTORCH=ON \
  -DEXECUTORCH_ROOT=/home/lanl/git/executorch \
  -DEXECUTORCH_CORE_LIBRARY=/home/lanl/git/executorch/cmake-out/libexecutorch_core.a
cmake --build build-executorch --target trt_executor_runner -j
```

Use it with a generated `.pte` file:

```bash
./bazel-bin/examples/torchtrt_executorch_example/trt_executor_runner --model_path=/path/to/model.pte
# or
./build-executorch/examples/torchtrt_executorch_example/trt_executor_runner --model_path=/path/to/model.pte
```
