# SPDX-License-Identifier: BSD-3-Clause
# Keep the native runtime ABI matched to the pinned Python exporter.
set(TORCHTRT_SOURCE_DIR "" CACHE PATH "Pinned Torch-TensorRT runtime checkout")
set(EXECUTORCH_SOURCE_DIR "" CACHE PATH "Pinned ExecuTorch checkout")
if(NOT EXISTS "${TORCHTRT_SOURCE_DIR}/examples/executorch_reference_runner/CMakeLists.txt"
   OR NOT EXISTS "${EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Run build_native.sh to fetch the pinned source dependencies")
endif()
set(TORCHTRT_ENABLE_DRIVE_OS_COMPAT ON CACHE BOOL "" FORCE)
set(EXECUTORCH_ENABLE_LOGGING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK OFF CACHE BOOL "" FORCE)
add_subdirectory("${TORCHTRT_SOURCE_DIR}/examples/executorch_reference_runner"
                 "${CMAKE_BINARY_DIR}/reference_runner" EXCLUDE_FROM_ALL)
target_link_libraries(benchmark_et PRIVATE
  executorch executorch::backends executorch::extensions executorch::kernels
  executorch::backend_tensorrt)
target_compile_options(benchmark_et PRIVATE -Wno-deprecated-declarations)
install(FILES "$<TARGET_FILE:aoti_cuda_shims>" "$<TARGET_FILE:extension_cuda>"
        DESTINATION lib)
