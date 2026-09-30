cmake_minimum_required(VERSION 3.19)
project(torch_tensorrt_edge_llm LANGUAGES CXX)

# Applications add ExecuTorch and Torch-TensorRT's native backend first. Use the
# same extension_cuda shared library through that target for caller streams.
if(NOT TARGET torchtrt::executorch_backend OR NOT TARGET executorch_trt_backend)
    message(FATAL_ERROR "Add Torch-TensorRT's ExecuTorch backend before torch_tensorrt_edge_llm")
endif()

add_library(executorch_edge_llm_backend STATIC src/EdgeLLMBackend.cpp src/EdgeLLMBlobHeader.cpp)
set_target_properties(executorch_edge_llm_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(executorch_edge_llm_backend PUBLIC cxx_std_17)
target_include_directories(executorch_edge_llm_backend PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(executorch_edge_llm_backend PUBLIC executorch_trt_backend)

add_library(edge_llm_executorch_backend INTERFACE)
add_library(tensorrt_edge_llm::executorch_backend ALIAS edge_llm_executorch_backend)
add_dependencies(edge_llm_executorch_backend executorch_edge_llm_backend)
# Keep registration alive when the application links a static delegate archive.
if(MSVC)
    target_link_libraries(edge_llm_executorch_backend INTERFACE executorch_edge_llm_backend executorch_trt_backend)
    target_link_options(edge_llm_executorch_backend INTERFACE
        "/WHOLEARCHIVE:$<TARGET_FILE:executorch_edge_llm_backend>"
        "/WHOLEARCHIVE:$<TARGET_FILE:executorch_trt_backend>")
else()
    # One group avoids CMake deduplicating the two backends' archive flags or
    # ordering the TensorRT archive before the Edge wrapper that calls it.
    target_link_libraries(edge_llm_executorch_backend INTERFACE
        "-Wl,--whole-archive"
        "$<TARGET_FILE:executorch_edge_llm_backend>"
        "$<TARGET_FILE:executorch_trt_backend>"
        "-Wl,--no-whole-archive"
        executorch_trt_backend)
endif()
