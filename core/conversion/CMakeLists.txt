set(lib_name "core_conversion")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/conversion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conversion_ignorelist.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/conversion.h"
)

target_sources(${lib_name}
    PRIVATE
        ${CXX_SRCS}
    PUBLIC
        $<TARGET_OBJECTS:core_ir>
        $<TARGET_OBJECTS:core_util>
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        torch
        core_ir
        core_util
)
target_include_directories(${lib_name}
    PUBLIC  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

# add sublibraries
add_subdirectory(conversionctx)
add_subdirectory(converters)
add_subdirectory(evaluators)
add_subdirectory(tensorcontainer)
add_subdirectory(var)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/conversion")
