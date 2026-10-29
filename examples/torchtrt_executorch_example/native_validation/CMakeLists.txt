# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.24)
project(drive_native_validation LANGUAGES CXX)

get_filename_component(_torchtrt_source "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(TORCHTRT_SOURCE_DIR "${_torchtrt_source}" CACHE PATH "Torch-TensorRT source checkout")
if(NOT EXISTS "${TORCHTRT_SOURCE_DIR}/examples/executorch_reference_runner/CMakeLists.txt")
  message(FATAL_ERROR "Set TORCHTRT_SOURCE_DIR to the Torch-TensorRT checkout")
endif()

# Reuse the supported runtime/delegate targets and the opt-in DRIVE adjustments.
# No LibTorch, Python bindings, or model-specific engine loading code.
add_subdirectory(
  "${TORCHTRT_SOURCE_DIR}/examples/executorch_reference_runner"
  "${CMAKE_BINARY_DIR}/reference_runner")
find_package(CUDAToolkit REQUIRED)
add_executable(validate_pte main.cpp)
target_compile_features(validate_pte PRIVATE cxx_std_17)
target_link_libraries(validate_pte PRIVATE
  executorch executorch::backends executorch::extensions executorch::kernels
  executorch::backend_tensorrt CUDA::cudart)

# Separate from the recurrence/correctness checker: fixed-input latency samples.
add_executable(benchmark_pte benchmark.cpp)
target_compile_features(benchmark_pte PRIVATE cxx_std_17)
target_link_libraries(benchmark_pte PRIVATE
  executorch executorch::backends executorch::extensions executorch::kernels
  executorch::backend_tensorrt CUDA::cudart)

install(TARGETS validate_pte benchmark_pte
  RUNTIME DESTINATION bin COMPONENT native_validation)
foreach(_support aoti_cuda_shims extension_cuda)
  if(TARGET ${_support})
    get_target_property(_support_type ${_support} TYPE)
    if(_support_type STREQUAL "SHARED_LIBRARY")
      install(FILES "$<TARGET_FILE:${_support}>"
        DESTINATION lib COMPONENT native_validation)
    endif()
  endif()
endforeach()
