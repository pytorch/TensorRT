set(lib_name "core_ir")
add_library(${lib_name} OBJECT)

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Input.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/ir.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/StaticParams.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/GraphInputs.cpp"
            $<TARGET_OBJECTS:core_util>
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ir.h"
)

target_include_directories(${lib_name}
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        torch
    PRIVATE
        core_util
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/ir/")
