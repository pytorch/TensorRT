cmake_minimum_required(VERSION 3.24)
project(driveos_pbr LANGUAGES CXX)
set(PBR_BUNDLE "" CACHE PATH "Extracted PBR artifact bundle")
if(NOT EXISTS "${PBR_BUNDLE}/sdk/lib/libexecutorch.a")
  message(FATAL_ERROR "Set PBR_BUNDLE to the checked PBR bundle (including sdk/)")
endif()
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE NvInfer.h PATH_SUFFIXES aarch64-linux-gnu REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer PATH_SUFFIXES aarch64-linux-gnu REQUIRED)
add_executable(benchmark_trt tensorrt.cpp)
add_executable(benchmark_et executorch.cpp)
foreach(target benchmark_trt benchmark_et)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE -O3 -Wall -Wextra)
  target_link_libraries(${target} PRIVATE CUDA::cudart ${TENSORRT_LIBRARY} ${CMAKE_DL_LIBS})
  set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN/../lib")
endforeach()
target_include_directories(benchmark_trt PRIVATE ${TENSORRT_INCLUDE})
set(sdk "${PBR_BUNDLE}/sdk")
target_include_directories(benchmark_et SYSTEM PRIVATE "${sdk}/include"
  "${sdk}/include/executorch/runtime/core/portable_type/c10")
target_compile_definitions(benchmark_et PRIVATE C10_USING_CUSTOM_GENERATED_MACROS ET_LOG_ENABLED=0)
target_compile_options(benchmark_et PRIVATE -Wno-deprecated-declarations)
target_link_options(benchmark_et PRIVATE -Wl,--export-dynamic)
# Static registration must survive the linker's unused-symbol elimination.
target_link_libraries(benchmark_et PRIVATE
  -Wl,--whole-archive
  "${sdk}/lib/libexecutorch.a" "${sdk}/lib/libaoti_cuda_backend.a"
  "${sdk}/lib/libportable_ops_lib.a" "${sdk}/lib/libexecutorch_trt_backend.a"
  -Wl,--no-whole-archive
  "${sdk}/lib/libcuda_platform.a"
  "${sdk}/lib/libextension_module_static.a" "${sdk}/lib/libextension_data_loader.a"
  "${sdk}/lib/libextension_flat_tensor.a" "${sdk}/lib/libextension_named_data_map.a"
  "${sdk}/lib/libextension_runner_util.a" "${sdk}/lib/libextension_tensor.a"
  "${sdk}/lib/libportable_kernels.a" "${sdk}/lib/libkernels_util_all_deps.a"
  "${sdk}/lib/libexecutorch_core.a"
  -Wl,--no-as-needed "${PBR_BUNDLE}/lib/libaoti_cuda_shims.so"
  "${PBR_BUNDLE}/lib/libextension_cuda.so" ${TENSORRT_LIBRARY} -Wl,--as-needed)
install(TARGETS benchmark_trt benchmark_et RUNTIME DESTINATION bin)
