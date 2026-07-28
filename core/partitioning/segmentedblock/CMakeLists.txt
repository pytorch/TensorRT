set(sub_lib_name "segmentedblock")

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/SegmentedBlock.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/SegmentedBlock.h"
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/partitioning/${sub_lib_name}")
