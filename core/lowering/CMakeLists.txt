set(lib_name "core_lowering")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/drop_unused_nodes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lowering.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/register_trt_placeholder_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/LowerInfo.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/lowering.h"
)

target_sources(${lib_name}
    PRIVATE
        ${CXX_SRCS}
        $<TARGET_OBJECTS:core_util>
)

target_include_directories(${lib_name}
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

target_link_libraries(${lib_name}
    PUBLIC
        torch
    PRIVATE
        core_util
)

# add sublibraries
add_subdirectory(passes)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/lowering")
