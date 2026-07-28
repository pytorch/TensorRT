set(sub_lib_name "tensorcontainer")

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/TensorContainer.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/TensorContainer.h"
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/conversion/${sub_lib_name}")
