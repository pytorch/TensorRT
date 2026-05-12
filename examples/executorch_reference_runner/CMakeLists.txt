cmake_minimum_required(VERSION 3.17)
project(my_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_torchtrt_package_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(_default_project_root "${_torchtrt_package_root}/.." ABSOLUTE)
set(EXECUTORCH_SOURCE_DIR
    "${_default_project_root}/executorch"
    CACHE PATH
    "Path to the ExecuTorch source tree"
)
set(TORCHTRT_EXECUTORCH_SOURCE_DIR
    "${_torchtrt_package_root}/src/torch_tensorrt"
    CACHE PATH
    "Path to the unpacked Torch-TensorRT ExecuTorch source package"
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
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
  message(FATAL_ERROR "TORCHTRT_EXECUTORCH_SOURCE_DIR is empty. Set it to the unpacked Torch-TensorRT ExecuTorch source package")
endif()

if(NOT EXISTS "${TORCHTRT_EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "TORCHTRT_EXECUTORCH_SOURCE_DIR must point to the unpacked Torch-TensorRT ExecuTorch source package with a CMakeLists.txt: ${TORCHTRT_EXECUTORCH_SOURCE_DIR}")
endif()

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/executorch")
add_subdirectory("${TORCHTRT_EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/torchtrt_executorch")

add_executable(my_runner main.cpp)
target_link_libraries(
  my_runner
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend)
