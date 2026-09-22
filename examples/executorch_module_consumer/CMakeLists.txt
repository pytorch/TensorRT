# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Consumes the two installed wheels and nothing from this repository's source tree. That is the point
# of it: the delegate comes out of its wheel already built, found by name, and the ExecuTorch runtime
# comes out of its own wheel the same way.
cmake_minimum_required(VERSION 3.28)
project(executorch_module_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
# Only components the package actually exports. The module and tensor extensions are not separate
# targets: their code arrives through the runtime target and its headers, so asking for them by name
# fails the configure step.
find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
find_package(executorch_backend_tensorrt REQUIRED)

add_executable(executorch_module_consumer main.cpp)
# The runtime headers reach for a generated macro header that only exists in a source tree, so a
# consumer of the wheel has to compile with the definition that turns that include off. The package
# supplies it rather than leaving a caller to discover it from a failing compile.
target_compile_definitions(executorch_module_consumer PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
target_link_libraries(executorch_module_consumer PRIVATE
  executorch::runtime
  executorch::backend_cuda
  executorch::backend_tensorrt
  executorch::kernels_optimized
  CUDA::cudart
)
