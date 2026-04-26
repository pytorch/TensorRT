get_filename_component(EXECUTORCH_INCLUDE_PARENT "${EXECUTORCH_ROOT}" DIRECTORY)

set(lib_name "executorch_trt_backend")
add_library(${lib_name} STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/TensorRTBackend.cpp"
)

set_target_properties(${lib_name}
    PROPERTIES
        OUTPUT_NAME "executorch_trt_backend"
)

target_include_directories(${lib_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "${EXECUTORCH_INCLUDE_PARENT}"
)

target_link_libraries(${lib_name}
    PUBLIC
        CUDA::cudart
        TensorRT::nvinfer
        torch
)

install(
    TARGETS ${lib_name}
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
