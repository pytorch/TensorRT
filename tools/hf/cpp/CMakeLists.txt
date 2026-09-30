cmake_minimum_required(VERSION 3.19)
project(torch_tensorrt_edge_llm LANGUAGES CXX)

# Applications add ExecuTorch and Torch-TensorRT's native backend first. Use the
# same extension_cuda shared library through that target for caller streams.
if(NOT TARGET torchtrt::executorch_backend OR NOT TARGET executorch_trt_backend)
    message(FATAL_ERROR "Add Torch-TensorRT's ExecuTorch backend before torch_tensorrt_edge_llm")
endif()

set(EDGELLM_INCLUDE_DIR "" CACHE PATH "TensorRT-Edge-LLM cpp include root")
set(EDGELLM_EXECUTORCH_LIBRARY "" CACHE FILEPATH "Path to libedgellmExecutorch.so")
if(NOT EXISTS "${EDGELLM_INCLUDE_DIR}/executorch/vitExecutorchAdapter.h")
    message(FATAL_ERROR "EDGELLM_INCLUDE_DIR must contain executorch/vitExecutorchAdapter.h")
endif()
if(NOT EXISTS "${EDGELLM_EXECUTORCH_LIBRARY}")
    message(FATAL_ERROR "EDGELLM_EXECUTORCH_LIBRARY must name libedgellmExecutorch.so")
endif()

add_library(edge_llm_adapter SHARED IMPORTED)
set_target_properties(edge_llm_adapter PROPERTIES IMPORTED_LOCATION "${EDGELLM_EXECUTORCH_LIBRARY}")

add_library(executorch_edge_llm_backend STATIC src/EdgeLLMBackend.cpp src/EdgeLLMBlobHeader.cpp)
set_target_properties(executorch_edge_llm_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(executorch_edge_llm_backend PUBLIC cxx_std_17)
target_include_directories(executorch_edge_llm_backend PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include" "${EDGELLM_INCLUDE_DIR}")
target_link_libraries(executorch_edge_llm_backend PUBLIC executorch_trt_backend edge_llm_adapter)

add_library(edge_llm_executorch_backend INTERFACE)
add_library(tensorrt_edge_llm::executorch_backend ALIAS edge_llm_executorch_backend)
add_dependencies(edge_llm_executorch_backend executorch_edge_llm_backend)
target_link_libraries(edge_llm_executorch_backend INTERFACE torchtrt::executorch_backend)
# Keep registration alive when the application links a static delegate archive.
if(MSVC)
    target_link_libraries(edge_llm_executorch_backend INTERFACE executorch_edge_llm_backend)
    target_link_options(edge_llm_executorch_backend INTERFACE "/WHOLEARCHIVE:$<TARGET_FILE:executorch_edge_llm_backend>")
else()
    target_link_libraries(edge_llm_executorch_backend INTERFACE
        "-Wl,--whole-archive" "$<TARGET_FILE:executorch_edge_llm_backend>" "-Wl,--no-whole-archive" edge_llm_adapter)
endif()
