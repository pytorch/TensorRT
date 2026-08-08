cmake_minimum_required(VERSION 3.24)
project(torch_tensorrt_executorch_runtime LANGUAGES CXX)

if(NOT EXECUTORCH_SOURCE_DIR AND EXECUTORCH_CMAKE_FILE)
  get_filename_component(EXECUTORCH_SOURCE_DIR "${EXECUTORCH_CMAKE_FILE}" DIRECTORY)
endif()
if(NOT TORCH_PRIMARY_INCLUDE_DIR AND TORCH_HEADER_MARKER)
  get_filename_component(TORCH_PRIMARY_INCLUDE_DIR "${TORCH_HEADER_MARKER}/../../../.." ABSOLUTE)
endif()
if(NOT EXECUTORCH_SOURCE_DIR OR NOT TORCH_TENSORRT_SOURCE_DIR OR NOT TORCH_PRIMARY_INCLUDE_DIR)
  message(FATAL_ERROR
    "EXECUTORCH_SOURCE_DIR, TORCH_TENSORRT_SOURCE_DIR, and TORCH_PRIMARY_INCLUDE_DIR are required")
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

# The pybind bridge includes both ATen and portable ExecuTorch headers. Keep the
# ATen/c10/torch header set internally consistent by resolving all of it from
# the exact PyTorch wheel that supplies the linked libraries.
if(NOT EXISTS "${TORCH_PRIMARY_INCLUDE_DIR}/torch/headeronly")
  message(FATAL_ERROR
    "torch/headeronly not found under TORCH_PRIMARY_INCLUDE_DIR=${TORCH_PRIMARY_INCLUDE_DIR}")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/torch_header_shim/torch")
file(COPY "${TORCH_PRIMARY_INCLUDE_DIR}/c10/"
  DESTINATION "${CMAKE_BINARY_DIR}/torch_header_shim/c10")
file(COPY "${TORCH_PRIMARY_INCLUDE_DIR}/torch/headeronly/"
  DESTINATION "${CMAKE_BINARY_DIR}/torch_header_shim/torch/headeronly")
if(NOT EXISTS "${CMAKE_BINARY_DIR}/torch_header_shim/c10/util/complex.h" OR
   NOT EXISTS "${CMAKE_BINARY_DIR}/torch_header_shim/torch/headeronly/macros/Macros.h")
  message(FATAL_ERROR "Failed to stage the PyTorch header shim")
endif()
target_include_directories(util BEFORE PRIVATE "${CMAKE_BINARY_DIR}/torch_header_shim")
target_include_directories(portable_lib BEFORE PRIVATE "${CMAKE_BINARY_DIR}/torch_header_shim")

# TensorRTBackend registers through a static initializer. Force its complete
# archive into the same pybind module that owns ExecuTorch's backend registry.
target_link_libraries(portable_lib PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,torch_tensorrt_executorch_backend>"
  extension_threadpool
  CUDA::cudart TensorRT::nvinfer Threads::Threads)

# The runtime wheel intentionally does not bundle PyTorch, TensorRT, or CUDA.
# TensorRT and CUDA pip packages install their shared libraries in sibling
# directories under site-packages, relative to this extension package.
# Preserve these paths in both the Bazel-collected build output and installed
# extension so importing the wheel does not depend on LD_LIBRARY_PATH.
set(_torch_tensorrt_executorch_runtime_rpath
  "$ORIGIN"
  "$ORIGIN/../tensorrt_libs"
  "$ORIGIN/../nvidia/cuda_runtime/lib"
  "$ORIGIN/../nvidia/cu13/lib")
set_target_properties(portable_lib PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${_torch_tensorrt_executorch_runtime_rpath}"
  OUTPUT_NAME "_portable_lib"
  SUFFIX ".so")
set_target_properties(data_loader PROPERTIES SUFFIX ".so")

install(TARGETS portable_lib data_loader LIBRARY DESTINATION lib)

add_custom_target(torch_tensorrt_executorch_portable_lib DEPENDS portable_lib data_loader)
