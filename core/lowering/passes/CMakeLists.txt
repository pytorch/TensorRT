target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/convNd_to_convolution.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/exception_elimination.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/fuse_addmm_branches.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/linear_to_addmm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/module_fallback.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/op_aliasing.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/reduce_gelu.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/reduce_remainder.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/reduce_to.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/remove_bn_dim_check.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/remove_contiguous.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/remove_dropout.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/remove_nops.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/remove_set_attrs.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/remove_unnecessary_casts.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/silu_to_sigmoid_multiplication.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/unpack_addmm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/unpack_batch_norm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/unpack_hardswish.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/unpack_log_softmax.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/unpack_std.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/unpack_var.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/view_to_reshape.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/passes.h"
)

# Install headers
set(sub_lib_name "passes")
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/lowering/${sub_lib_name}")
