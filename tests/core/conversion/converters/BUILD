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
    name = "test_cumsum",
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
    name = "test_reflection_pad",
)

converter_test(
    name = "test_replication_pad",
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
    name = "test_select",
)

converter_test(
    name = "test_stack",
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
    name = "test_squeeze",
)

test_suite(
    name = "converter_tests",
    tests = [
        ":test_activation",
        ":test_batch_norm",
        ":test_bitwise",
        ":test_cast",
        ":test_clone",
        ":test_concat",
        ":test_constant_pad",
        ":test_conv_deconv",
        ":test_copy",
        ":test_cumsum",
        ":test_element_wise",
        ":test_expand",
        ":test_instance_norm",
        ":test_interpolate",
        ":test_layer_norm",
        ":test_linear",
        ":test_lstm_cell",
        ":test_matrix_multiply",
        ":test_normalize",
        ":test_pooling",
        ":test_reduce",
        ":test_replication_pad",
        ":test_select",
        ":test_shuffle",
        ":test_softmax",
        ":test_squeeze",
        ":test_stack",
        ":test_topk",
        ":test_unary",
        ":test_unsqueeze",
    ],
)
