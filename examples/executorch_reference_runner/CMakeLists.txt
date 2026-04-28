cmake_minimum_required(VERSION 3.17)
project(my_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_default_project_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(EXECUTORCH_SOURCE_DIR
    "${_default_project_root}/executorch"
    CACHE PATH
    "Path to the ExecuTorch source tree"
)
set(TORCHTRT_EXECUTORCH_SOURCE_DIR
    "${_default_project_root}/libtorchtrt_executorch"
    CACHE PATH
    "Path to the unpacked libtorchtrt_executorch source package"
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/executorch")
add_subdirectory("${TORCHTRT_EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/libtorchtrt_executorch")

add_executable(my_runner main.cpp)
target_link_libraries(
  my_runner
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend)
