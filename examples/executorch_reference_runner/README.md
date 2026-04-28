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

You can also generate a sample `.pte` from the Torch-TensorRT source tree:

```bash
python examples/torchtrt_executorch_example/export_static_shape.py --model_path=model.pte
```

## Build The Reference Runner

A normal reference runner build does not need separate steps for
`libexecutorch_core.a` and `libexecutorch_trt_backend.a`. The runner CMake adds
both ExecuTorch and the Torch-TensorRT ExecuTorch source package, and linking
`torchtrt::executorch_backend` makes the backend archive a dependency of
`my_runner`.

```bash
# get the executorch source code
git clone https://github.com/pytorch/executorch.git
# download the libtorchtrt.tar.gz
tar xvf libtorchtrt.tar.gz

export EXECUTORCH_SOURCE_DIR=/path/to/executorch
export TORCHTRT_EXECUTORCH_SOURCE_DIR="${PWD}/libtorchtrt_executorch"
export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake -S "${TORCHTRT_EXECUTORCH_SOURCE_DIR}/examples/executorch_reference_runner" \
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

The build also creates the executorch core and tensorrt backend archive as a dependency:

```text
build-executorch-reference-runner/executorch/libexecutorch_core.a
build-executorch-reference-runner/libtorchtrt_executorch/lib/libexecutorch_trt_backend.a
```

## Load And Run A `.pte` Model

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
