set(sub_lib_name "converters")

target_sources(${lib_name}
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/NodeConverterRegistry.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/converter_util.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Weights.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/activation.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/batch_norm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/cast.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/concat.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/constant.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/constant_pad.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/conv_deconv.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/cumsum.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/element_wise.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/expand.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/interpolate.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/layer_norm.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/linear.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/lstm_cell.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/matrix_multiply.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/max.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/normalize.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/pooling.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/quantization.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/reduce.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/reflection_pad.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/replication_pad.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/select.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/shuffle.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/softmax.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/squeeze.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/stack.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/topk.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/unary.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/impl/unsqueeze.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/Weights.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/converters.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/converter_util.h"
)

# Install headers
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/core/conversion/${sub_lib_name}")
