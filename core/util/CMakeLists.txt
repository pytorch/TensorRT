set(lib_name "core_util")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/Exception.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/trt_util.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/Exception.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/build_info.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit_util.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/macros.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/prelude.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/trt_util.h"
)

target_sources(${lib_name}
    PRIVATE
        ${CXX_SRCS}
    PUBLIC
        "$<TARGET_OBJECTS:core_util_logging>"
)

target_include_directories(${lib_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        torch
        core_util_logging
)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/util")

add_subdirectory(logging)
