# Torch-TensorRT native runtimes

The C++ TorchScript compiler and `torchtrtc` CLI have been removed. Compile PyTorch
models with the Python Dynamo frontend and deploy using AOTInductor or ExecuTorch.

`libtorchtrt.so` / `torchtrt.dll` contain the TensorRT runtime operators and
TorchBind engine registrations used by Dynamo and `torch.export`. The
`libtorchtrt_runtime.so` / `torchtrt_runtime.dll` runtime names are also available.

The standalone ExecuTorch backend and its CMake integration are in
[`src/torch_tensorrt/executorch`](src/torch_tensorrt/executorch/README.md).
