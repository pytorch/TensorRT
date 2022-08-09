set(lib_name "core_util_logging")
add_library(${lib_name} OBJECT)

target_sources(${lib_name}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/TorchTRTLogger.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/TorchTRTLogger.h"
)

target_include_directories(${lib_name}
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        torch
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/util/logging")
