cmake_minimum_required(VERSION 3.24)
project(torch_tensorrt_executorch_delegate LANGUAGES CXX)

if(NOT EXECUTORCH_SOURCE_DIR OR NOT TORCH_TENSORRT_SOURCE_DIR)
  message(FATAL_ERROR "EXECUTORCH_SOURCE_DIR and TORCH_TENSORRT_SOURCE_DIR are required")
endif()

list(APPEND CMAKE_MODULE_PATH "${TORCH_TENSORRT_SOURCE_DIR}/cmake/Modules")
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" executorch)

add_library(torch_tensorrt_executorch_backend STATIC
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBackend.cpp"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp")
target_compile_features(torch_tensorrt_executorch_backend PUBLIC cxx_std_17)
target_compile_definitions(torch_tensorrt_executorch_backend
  PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(torch_tensorrt_executorch_backend PRIVATE
  "${TORCH_TENSORRT_SOURCE_DIR}"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/include"
  "${EXECUTORCH_SOURCE_DIR}/.."
  "${EXECUTORCH_SOURCE_DIR}/runtime/core/portable_type/c10")
target_link_libraries(torch_tensorrt_executorch_backend PRIVATE
  CUDA::cudart TensorRT::nvinfer Threads::Threads executorch)

if(NOT TARGET portable_lib)
  message(FATAL_ERROR "ExecuTorch did not define portable_lib")
endif()

# ExecuTorch portable c10 delegates selected headers to torch/headeronly. The
# pybind bridge must resolve those delegated headers from the exact PyTorch
# wheel it links, not ExecuTorch's vendored snapshot.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/torch_header_shim/torch")
file(CREATE_LINK
  "${TORCH_PRIMARY_INCLUDE_DIR}/torch/headeronly"
  "${CMAKE_BINARY_DIR}/torch_header_shim/torch/headeronly"
  SYMBOLIC COPY_ON_ERROR)
target_include_directories(util BEFORE PRIVATE "${CMAKE_BINARY_DIR}/torch_header_shim")
target_include_directories(portable_lib BEFORE PRIVATE "${CMAKE_BINARY_DIR}/torch_header_shim")

# TensorRTBackend registers through a static initializer. Force its complete
# archive into the same pybind module that owns ExecuTorch's backend registry.
target_link_libraries(portable_lib PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,torch_tensorrt_executorch_backend>"
  extension_threadpool
  CUDA::cudart TensorRT::nvinfer Threads::Threads)
set_target_properties(portable_lib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DELEGATE_LIBRARY_OUTPUT_DIRECTORY}"
  OUTPUT_NAME "_portable_lib")
set_target_properties(data_loader PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${DELEGATE_LIBRARY_OUTPUT_DIRECTORY}")

add_custom_target(torch_tensorrt_executorch_portable_lib DEPENDS portable_lib data_loader)
