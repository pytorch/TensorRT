set(lib_name "core_partitioning")
add_library(${lib_name} OBJECT)

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/SegmentedBlock.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shape_analysis.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/partitioning.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/PartitionInfo.cpp"
            $<TARGET_OBJECTS:core_conversion>
    PUBLIC  $<TARGET_OBJECTS:core_ir>
            $<TARGET_OBJECTS:core_util>
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/SegmentedBlock.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/shape_analysis.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/PartitionInfo.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning.h"
)

target_include_directories(${lib_name} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>")
target_link_libraries(${lib_name}
    PUBLIC
        torch
        TensorRT::nvinfer
        core_ir
        core_util
    PRIVATE
        core_conversion
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/partitioning/")
