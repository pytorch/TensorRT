cmake_minimum_required(VERSION 3.24)
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
# Not removable: only this backend registers the allocator the exported program's
# device copies look up, even when TensorRT is the sole delegate.
set(EXECUTORCH_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
option(
  TORCHTRT_ENABLE_DRIVE_OS_COMPAT
  "Enable DRIVE OS compatibility for the ExecuTorch CUDA backend"
  OFF)
set(
  TORCHTRT_DRIVE_CUDA_ROOT
  "$ENV{TORCHTRT_DRIVE_CUDA_ROOT}"
  CACHE PATH
  "CUDA root injected into the DRIVE build container")

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

# Keep the DRIVE setting as a fallback; use CUDAToolkit_ROOT for both toolkit
# discovery and the DRIVE header paths below.
if(TORCHTRT_ENABLE_DRIVE_OS_COMPAT
   AND TORCHTRT_DRIVE_CUDA_ROOT
   AND NOT CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${TORCHTRT_DRIVE_CUDA_ROOT}" CACHE PATH "" FORCE)
endif()

# Validate DRIVE SDK inputs before configuring dependency projects.
if(TORCHTRT_ENABLE_DRIVE_OS_COMPAT)
  message(STATUS "Torch-TensorRT DRIVE OS compatibility: enabled")

  if(NOT CUDAToolkit_ROOT)
    message(FATAL_ERROR
      "Set CUDAToolkit_ROOT (or TORCHTRT_DRIVE_CUDA_ROOT) to the CUDA root injected into the DRIVE build container")
  endif()

  set(_torchtrt_drive_common_include
      "${CUDAToolkit_ROOT}/targets/aarch64-linux/include")
  set(_torchtrt_drive_thor_include
      "${CUDAToolkit_ROOT}/thor/targets/aarch64-linux/include")
  if(NOT EXISTS "${_torchtrt_drive_common_include}/cuda_runtime_api.h")
    message(FATAL_ERROR
      "DRIVE common CUDA headers not found under ${_torchtrt_drive_common_include}")
  endif()
  if(NOT EXISTS "${_torchtrt_drive_thor_include}/curand_kernel.h")
    message(FATAL_ERROR
      "Thor CUDA headers not found under ${_torchtrt_drive_thor_include}")
  endif()
endif()

find_package(CUDAToolkit REQUIRED)

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/executorch")
add_subdirectory("${TORCHTRT_EXECUTORCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/torch_tensorrt_executorch")

if(TORCHTRT_ENABLE_DRIVE_OS_COMPAT)
  # ExecuTorch's CUDA shim target is defined in the source tree added above.
  # DRIVE exposes common and Thor-specific headers as separate target trees.
  target_include_directories(
    aoti_cuda_shims
    PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
      "${_torchtrt_drive_common_include}"
      "${_torchtrt_drive_thor_include}")

  # The pinned ExecuTorch release links the cuRAND host library even though its
  # CUDA shims use only the device-side API from curand_kernel.h. DRIVE does not
  # ship a conventional libcurand development target, so remove that unused
  # link. Keep this local to the reference runner instead of modifying the
  # pinned ExecuTorch checkout.
  foreach(_drive_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(_drive_links aoti_cuda_shims ${_drive_property})

    if(_drive_links)
      list(REMOVE_ITEM _drive_links CUDA::curand)
      set_property(TARGET aoti_cuda_shims
        PROPERTY ${_drive_property} "${_drive_links}")
    endif()
  endforeach()
endif()

add_executable(example_executorch_runner main.cpp)
target_link_libraries(
  example_executorch_runner
  PRIVATE
    ${CMAKE_DL_LIBS}
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend)

# Caller-owned KV-cache persistence check (see kv_cache_decode_check.cpp). It
# cudaMalloc's the device-tagged planned arenas that hold the KV buffers and
# copies the logits back to host, so it links the CUDA runtime directly.
add_executable(kv_cache_decode_check kv_cache_decode_check.cpp)
target_link_libraries(
  kv_cache_decode_check
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend
    CUDA::cudart)
