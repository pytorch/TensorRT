# Torch-TensorRT ExecuTorch Backend

This package builds the TensorRT backend delegate for ExecuTorch from source.
It is intended to live next to an ExecuTorch checkout and be consumed with
`add_subdirectory`.

```text
user_runner_project/
  executorch/
  libtorchtrt_executorch/
```

Build the ExecuTorch core runtime first:

```bash
export EXECUTORCH_ROOT="${PWD}/executorch"
export CMAKE_PREFIX_PATH=/path/to/torch/share/cmake

cmake -S "${EXECUTORCH_ROOT}" -B "${EXECUTORCH_ROOT}/cmake-out" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON

cmake --build "${EXECUTORCH_ROOT}/cmake-out" --target executorch_core -j
```

Build the TensorRT backend archive from this package:

```bash
cmake -S libtorchtrt_executorch -B build-libtorchtrt-executorch \
  -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"

cmake --build build-libtorchtrt-executorch --target executorch_trt_backend -j
```

Then add both projects from your runner CMake:

```cmake
add_subdirectory("executorch")
add_subdirectory("libtorchtrt_executorch")

target_link_libraries(
  my_runner
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend)
```

The backend archive is available as the `executorch_trt_backend` CMake target
and is written to `lib/libexecutorch_trt_backend.a` in the build tree.
