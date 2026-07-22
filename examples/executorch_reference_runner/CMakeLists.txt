cmake_minimum_required(VERSION 3.17)
project(my_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_default_torch_tensorrt_root "${CMAKE_CURRENT_LIST_DIR}/../..")
set(_default_project_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(_default_torchtrt_executorch_source_dir
    "${_default_torch_tensorrt_root}/src/torch_tensorrt/executorch"
)
if(NOT EXISTS "${_default_torchtrt_executorch_source_dir}/CMakeLists.txt")
  set(_default_torchtrt_executorch_source_dir
      "${_default_torch_tensorrt_root}/cpp/src/torch_tensorrt/executorch"
  )
endif()

set(EXECUTORCH_SOURCE_DIR
    "${_default_project_root}/executorch"
    CACHE PATH
    "Path to the ExecuTorch source tree"
)
set(TORCHTRT_EXECUTORCH_SOURCE_DIR
    "${_default_torchtrt_executorch_source_dir}"
    CACHE PATH
    "Path to the Torch-TensorRT ExecuTorch backend source directory"
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
# Keep this reference runner libtorch-free. The Torch-TensorRT backend builds
# only ExecuTorch's minimal shared caller-stream extension from source below.
set(EXECUTORCH_BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)

if(NOT EXECUTORCH_SOURCE_DIR)
  message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR is empty. Set it to an ExecuTorch source checkout, for example -DEXECUTORCH_SOURCE_DIR=/path/to/executorch")
endif()

if(NOT EXISTS "${EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR must point to an ExecuTorch source checkout with a CMakeLists.txt: ${EXECUTORCH_SOURCE_DIR}")
endif()

if(NOT TORCHTRT_EXECUTORCH_SOURCE_DIR)
  message(FATAL_ERROR "TORCHTRT_EXECUTORCH_SOURCE_DIR is empty. Set it to torch_tensorrt/src/torch_tensorrt/executorch")
endif()

if(NOT EXISTS "${TORCHTRT_EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "TORCHTRT_EXECUTORCH_SOURCE_DIR must point to the Torch-TensorRT ExecuTorch backend source directory with a CMakeLists.txt: ${TORCHTRT_EXECUTORCH_SOURCE_DIR}")
endif()

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/executorch")
add_subdirectory("${TORCHTRT_EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/torch_tensorrt_executorch")

add_executable(example_executorch_runner main.cpp)
target_link_libraries(
  example_executorch_runner
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend)
