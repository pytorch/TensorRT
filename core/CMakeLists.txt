set(lib_name "core")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/compiler.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/compiler.h"
)

target_sources(${lib_name}
    PRIVATE
        ${CXX_SRCS}
    PUBLIC
        $<TARGET_OBJECTS:core_lowering>
        $<TARGET_OBJECTS:core_conversion>
        $<TARGET_OBJECTS:core_runtime>
        $<TARGET_OBJECTS:core_partitioning>
        $<TARGET_OBJECTS:core_util_logging>
)

target_link_libraries(${lib_name}
    PUBLIC
        torch
        TensorRT::nvinfer
        core_runtime
        core_conversion
        core_lowering
        core_partitioning
        core_util_logging
)

target_include_directories(${lib_name}
    PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)

add_subdirectory(util)
add_subdirectory(conversion)
add_subdirectory(runtime)
add_subdirectory(lowering)
add_subdirectory(partitioning)
add_subdirectory(plugins)
add_subdirectory(ir)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/")
