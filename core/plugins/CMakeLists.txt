set(lib_name "core_plugins")
add_library(${lib_name} OBJECT)

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/impl/interpolate_plugin.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/normalize_plugin.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/register_plugins.cpp"
    PUBLIC  $<TARGET_OBJECTS:core_util>
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/plugins.h"
)

target_include_directories(${lib_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        TensorRT::nvinfer_plugin
        torch
        core_util
    PRIVATE
        Threads::Threads
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/plugins/")
install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/interpolate_plugin.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/normalize_plugin.h"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/plugins/impl"
)
