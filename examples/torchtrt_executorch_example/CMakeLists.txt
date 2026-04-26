get_filename_component(EXECUTORCH_INCLUDE_PARENT "${EXECUTORCH_ROOT}" DIRECTORY)

set(executorch_file_data_loader_lib "executorch_file_data_loader_local")
add_library(${executorch_file_data_loader_lib} STATIC
    "${EXECUTORCH_ROOT}/extension/data_loader/file_data_loader.cpp"
)

target_include_directories(${executorch_file_data_loader_lib}
    PUBLIC
        "${EXECUTORCH_INCLUDE_PARENT}"
)

target_link_libraries(${executorch_file_data_loader_lib}
    PUBLIC
        "${EXECUTORCH_CORE_LIBRARY}"
        torch
)

add_executable(trt_executor_runner executor_runner.cpp)
target_sources(trt_executor_runner
    PRIVATE
        $<TARGET_OBJECTS:core_util>
        $<TARGET_OBJECTS:core_util_logging>
        $<TARGET_OBJECTS:core_runtime>
)
target_link_libraries(trt_executor_runner
    PRIVATE
        ${TORCH_LIBRARIES}
        "${EXECUTORCH_CORE_LIBRARY}"
        -Wl,--whole-archive executorch_trt_backend -Wl,--no-whole-archive
        CUDA::cudart
        TensorRT::nvinfer
        ${executorch_file_data_loader_lib}
)

set_property(TARGET trt_executor_runner PROPERTY CXX_STANDARD 17)
