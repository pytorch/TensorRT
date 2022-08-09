set(lib_name "core_runtime")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/CudaDevice.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DeviceList.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/execute_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/TRTEngine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/register_jit_hooks.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime.h"
)

target_sources(${lib_name}
    PRIVATE ${CXX_SRCS}
            $<TARGET_OBJECTS:core_util>
)

target_include_directories(${lib_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        torch
        core_util
)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/runtime")
