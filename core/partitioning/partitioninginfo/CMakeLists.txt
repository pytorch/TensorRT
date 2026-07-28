set(sub_lib_name "partitioninginfo")

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/PartitioningInfo.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/PartitioningInfo.h"
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/partitioning/${sub_lib_name}")
