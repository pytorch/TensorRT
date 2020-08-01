load("//tests/core/converters:converter_test.bzl", "converter_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

converter_test(
  name = "test_activation"
)

converter_test(
  name = "test_batch_norm"
)

converter_test(
  name = "test_concat"
)

converter_test(
  name = "test_conv_deconv"
)

converter_test(
  name = "test_element_wise"
)

converter_test(
  name = "test_linear"
)

converter_test(
  name = "test_matrix_multiply"
)

converter_test(
  name = "test_pooling"
)

converter_test(
  name = "test_reduce"
)

converter_test(
  name = "test_shuffle"
)

converter_test(
  name = "test_softmax"
)

converter_test(
  name = "test_unary"
)

converter_test(
  name = "test_interpolate"
)

converter_test(
  name = "test_select"
)

converter_test(
  name = "test_stack"
)

converter_test(
  name = "test_loop"
)

test_suite(
  name = "test_converters",
  tests = [
    ":test_activation",
    ":test_batch_norm",
    ":test_conv_deconv",
    ":test_element_wise",
    ":test_linear",
    ":test_matrix_multiply",
    ":test_pooling",
    ":test_reduce",
    ":test_shuffle",
    ":test_softmax",
    ":test_unary",
    ":test_interpolate",
    ":test_select",
    ":test_stack",
    ":test_loop"
  ]
)
