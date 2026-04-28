# Torch-TensorRT ExecuTorch Reference Runner

This directory contains a minimal C++ reference runner for loading and executing
Torch-TensorRT compiled models saved in ExecuTorch `.pte` format.

The `.pte` file contains an ExecuTorch program with embedded TensorRT engine
payloads. The runner links the TensorRT ExecuTorch backend, loads the `.pte`
with the ExecuTorch C++ runtime, prepares input tensors, calls `execute()`, and
prints output shapes and sample values.

This is reference code. It fills all inputs with `1.0f`; replace that input
setup with your application's real input buffers.

## Input Model

Create or obtain a Torch-TensorRT compiled ExecuTorch model:

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

You can also generate a sample `.pte` from the Torch-TensorRT source tree:

```bash
python examples/torchtrt_executorch_example/export_static_shape.py --model_path=model.pte
```

## 1. Build `libexecutorch_core.a`

Build the ExecuTorch core runtime from an ExecuTorch source checkout:

```bash
git clone https://github.com/pytorch/executorch.git

export EXECUTORCH_ROOT=/path/to/executorch

cmake -S "${EXECUTORCH_ROOT}" -B "${EXECUTORCH_ROOT}/cmake-out" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON

cmake --build "${EXECUTORCH_ROOT}/cmake-out" --target executorch_core -j
```

Expected artifact:

```text
${EXECUTORCH_ROOT}/cmake-out/libexecutorch_core.a
```

## 2. Build `libexecutorch_trt_backend.a`

Build the Torch-TensorRT ExecuTorch backend from the
`libtorchtrt_executorch.tar.gz` release package:

```bash

# Download libtorchtrt_executorch.tar.gz
tar xvf libtorchtrt_executorch.tar.gz

export EXECUTORCH_ROOT=/path/to/executorch
export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake -S libtorchtrt_executorch -B build-libtorchtrt-executorch \
  -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"

cmake --build build-libtorchtrt-executorch \
  --target executorch_trt_backend \
  -j
```

Expected artifact:

```text
build-libtorchtrt-executorch/lib/libexecutorch_trt_backend.a
```

## 3. Build The Reference Runner

Build `my_runner` from the unpacked source package:

```bash
export EXECUTORCH_ROOT=/path/to/executorch
export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
export EXECUTORCH_SOURCE_DIR="${EXECUTORCH_ROOT}"
export TORCHTRT_EXECUTORCH_SOURCE_DIR=/path/to/untarred/libtorchtrt_executorch

cmake -S libtorchtrt_executorch/examples/executorch_reference_runner \
  -B build-executorch-reference-runner \
  -DEXECUTORCH_SOURCE_DIR="${EXECUTORCH_SOURCE_DIR}" \
  -DTORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCHTRT_EXECUTORCH_SOURCE_DIR}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"

cmake --build build-executorch-reference-runner --target my_runner -j
```

Expected artifact:

```text
build-executorch-reference-runner/my_runner
```

## 4. Load And Run A `.pte` Model

Run the reference runner against a Torch-TensorRT compiled ExecuTorch model:

```bash
./build-executorch-reference-runner/my_runner --model_path=/path/to/model.pte --num_runs=1
```

The runner demonstrates this C++ loading sequence:

```text
executorch::runtime::runtime_init()
FileDataLoader::from(model_path)
Program::load(loader)
program.method_meta(method_name)
allocate planned ExecuTorch memory
program.load_method(method_name, memory_manager)
method.set_input(...)
method.execute()
method.get_outputs(...)
```

Loading the method initializes the TensorRT ExecuTorch backend for any
Torch-TensorRT delegate subgraphs embedded in the `.pte`. The Python
`torch_tensorrt` package is needed when exporting the `.pte`; it is not needed
by this native runner at inference time.
