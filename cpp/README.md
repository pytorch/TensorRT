# Torch-TensorRT native runtimes

The C++ TorchScript compiler and `torchtrtc` CLI have been removed. Compile PyTorch
models with the Python Dynamo frontend and deploy using AOTInductor or ExecuTorch.

`libtorchtrt.so` / `torchtrt.dll` contain the TensorRT runtime operators and
TorchBind engine registrations used by Dynamo and `torch.export`. The
`libtorchtrt_runtime.so` / `torchtrt_runtime.dll` runtime names are also available.

The public `torch_tensorrt/torch_tensorrt.h`, `logging.h`, and `macros.h` headers
remain available for native runtime applications, including AOTInductor callers.
Both runtime libraries export build information, CUDA device selection, and
logging controls. The logging API configures the same logger used by the native
TensorRT runtime. Execution controls and engine settings remain under
`torch_tensorrt/core/runtime/`; shared utilities remain under
`torch_tensorrt/core/util/`.

With CMake, link `torchtrt` or `torchtrt_runtime` after `find_package(torchtrt)`
to get the public and core include paths. These APIs configure the Torch-TensorRT
native runtime; a self-contained AOTInductor package that does not call that
runtime uses its own generated execution path.

The standalone ExecuTorch backend and its CMake integration are in
[`src/torch_tensorrt/executorch`](src/torch_tensorrt/executorch/README.md).
