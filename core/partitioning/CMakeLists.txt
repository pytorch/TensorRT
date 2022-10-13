set(lib_name "core_partitioning")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shape_analysis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stitching.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning.h"
)

target_sources(${lib_name}
    PRIVATE
        ${CXX_SRCS}
    PUBLIC
        $<TARGET_OBJECTS:core_conversion>
        $<TARGET_OBJECTS:core_ir>
        $<TARGET_OBJECTS:core_util>
)

target_link_libraries(${lib_name}
    PUBLIC
        TensorRT::nvinfer
        torch
        core_ir
        core_util
        core_conversion
)

target_include_directories(${lib_name}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
)

add_subdirectory(partitioningctx)
add_subdirectory(partitioninginfo)
add_subdirectory(segmentedblock)

install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/partitioning")
