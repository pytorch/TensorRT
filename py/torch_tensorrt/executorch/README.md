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
export EXECUTORCH_ROOT=/home/lanl/git/executorch
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

This repo includes a C++ reference runner in
`examples/executorch_reference_runner/main.cpp`. It follows the ExecuTorch
CMake integration pattern: clone ExecuTorch next to the unpacked
`libtorchtrt_executorch` source package, add both directories with CMake, and
link the runner against `torchtrt::executorch_backend`.

The corresponding Bazel target statically links the TensorRT backend into the
application binary:

```bash
bazel build //examples/torchtrt_executorch_example:trt_executor_runner
```

For the CMake source package flow:

```bash
mkdir user_runner_project
cd user_runner_project
git clone https://github.com/pytorch/executorch.git
tar -xzf /path/to/libtorchtrt_executorch.tar.gz

export EXECUTORCH_ROOT="${PWD}/executorch"
export CMAKE_PREFIX_PATH=/home/lanl/miniconda3/envs/torch_tensorrt_py310/lib/python3.10/site-packages/torch/share/cmake

# 1. Build the ExecuTorch runtime static library.
cmake -S "${EXECUTORCH_ROOT}" -B "${EXECUTORCH_ROOT}/cmake-out" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON
cmake --build "${EXECUTORCH_ROOT}/cmake-out" --target executorch_core -j

# 2. Build the standalone TensorRT backend archive.
cmake -S libtorchtrt_executorch -B build-libtorchtrt-executorch \
  -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
cmake --build build-libtorchtrt-executorch --target executorch_trt_backend -j

# 3. Configure and build the reference runner.
cmake -S libtorchtrt_executorch/examples/executorch_reference_runner \
  -B build-executorch-reference-runner \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
cmake --build build-executorch-reference-runner --target my_runner -j
```

This produces:

- `EXECUTORCH_ROOT/cmake-out/libexecutorch_core.a`
- `build-libtorchtrt-executorch/lib/libexecutorch_trt_backend.a`
- `build-executorch-reference-runner/lib/libexecutorch_trt_backend.a`
- `build-executorch-reference-runner/my_runner`

The `my_runner` binary statically links the application code,
ExecuTorch runtime integration, and the TensorRT ExecuTorch backend. It still
depends on the TensorRT, CUDA, and LibTorch shared libraries present on the
system, but it does not require the `torch_tensorrt` Python wheel at runtime.

Use it with a generated `.pte` file:

```bash
./bazel-bin/examples/torchtrt_executorch_example/trt_executor_runner --model_path=/path/to/model.pte
# or
./build-executorch-reference-runner/my_runner --model_path=/path/to/model.pte
```

## Export vs Runtime

Exporting a `.pte` file requires the Python packages:

- `torch_tensorrt`
- `executorch`

For example:

```bash
python examples/torchtrt_executorch_example/export_static_shape.py --model_path=model.pte
```

Loading and executing that saved `.pte` file only requires
`trt_executor_runner` plus the native TensorRT/CUDA/LibTorch dependencies. The
Python `torch_tensorrt` wheel is not needed for inference.
