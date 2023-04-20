load("//tests/core/conversion/converters:converter_test.bzl", "converter_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

converter_test(
    name = "test_activation",
)

converter_test(
    name = "test_add_sub_mul",
)

converter_test(
    name = "test_atan2",
)

converter_test(
    name = "test_batch_norm",
)

converter_test(
    name = "test_bitwise",
)

converter_test(
    name = "test_instance_norm",
)

converter_test(
    name = "test_cast",
)

converter_test(
    name = "test_clone",
)

converter_test(
    name = "test_clamp",
)

converter_test(
    name = "test_concat",
)

converter_test(
    name = "test_constant_pad",
)

converter_test(
    name = "test_conv_deconv",
)

converter_test(
    name = "test_copy",
)

converter_test(
    name = "test_comparators",
)

converter_test(
    name = "test_cumsum",
)

converter_test(
    name = "test_div",
)

converter_test(
    name = "test_einsum",
)

converter_test(
    name = "test_element_wise",
)

converter_test(
    name = "test_expand",
)

converter_test(
    name = "test_layer_norm",
)

converter_test(
    name = "test_linear",
)

converter_test(
    name = "test_matrix_multiply",
)

converter_test(
    name = "test_masked_fill",
)

converter_test(
    name = "test_max",
)

converter_test(
    name = "test_normalize",
)

converter_test(
    name = "test_pooling",
)

converter_test(
    name = "test_quantization",
)

converter_test(
    name = "test_reduce",
)

converter_test(
    name = "test_roll",
)

converter_test(
    name = "test_reflection_pad",
)

converter_test(
    name = "test_replication_pad",
)

converter_test(
    name = "test_scatter",
)

converter_test(
    name = "test_shuffle",
)

converter_test(
    name = "test_softmax",
)

converter_test(
    name = "test_unary",
)

converter_test(
    name = "test_interpolate",
)

converter_test(
    name = "test_index",
)

converter_test(
    name = "test_select",
)

converter_test(
    name = "test_stack",
)

converter_test(
    name = "test_slice",
)

converter_test(
    name = "test_split",
)

converter_test(
    name = "test_topk",
)

converter_test(
    name = "test_lstm_cell",
)

converter_test(
    name = "test_unsqueeze",
)

converter_test(
    name = "test_unbind",
)

converter_test(
    name = "test_unpack",
)

converter_test(
    name = "test_squeeze",
)

converter_test(
    name = "test_where",
)

test_suite(
    name = "converter_tests",
    tests = [
        ":test_activation",
        ":test_add_sub_mul",
        ":test_atan2",
        ":test_batch_norm",
        ":test_bitwise",
        ":test_cast",
        ":test_clamp",
        ":test_clone",
        ":test_comparators",
        ":test_concat",
        ":test_constant_pad",
        ":test_conv_deconv",
        ":test_copy",
        ":test_cumsum",
        ":test_div",
        ":test_einsum",
        ":test_expand",
        ":test_instance_norm",
        ":test_interpolate",
        ":test_index",
        ":test_layer_norm",
        ":test_linear",
        ":test_lstm_cell",
        ":test_matrix_multiply",
        ":test_masked_fill",
        ":test_max",
        ":test_normalize",
        ":test_pooling",
        ":test_reduce",
        ":test_roll",
        ":test_replication_pad",
        ":test_scatter",
        ":test_select",
        ":test_shuffle",
        ":test_softmax",
        ":test_squeeze",
        ":test_stack",
        ":test_split",
        ":test_slice",
        ":test_topk",
        ":test_unary",
        ":test_unsqueeze",
        ":test_unbind",
        ":test_unpack",
        ":test_where",
    ],
)
