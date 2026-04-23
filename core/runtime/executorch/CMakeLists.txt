get_filename_component(EXECUTORCH_INCLUDE_PARENT "${EXECUTORCH_ROOT}" DIRECTORY)

set(lib_name "core_runtime_executorch_backend")
add_library(${lib_name} OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/TensorRTBackend.cpp"
)

target_include_directories(${lib_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "${EXECUTORCH_INCLUDE_PARENT}"
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        core_runtime
)
