load("//tests/core/converters:converter_test.bzl", "converter_test")

converter_test(
  name = "test_activation"
)

converter_test(
  name = "test_batch_norm"
)

converter_test(
  name = "test_conv"
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

test_suite(
  name = "test_converters",
  tests = [
    ":test_activation",
    ":test_batch_norm",
    ":test_conv",
    ":test_element_wise",
    ":test_linear",
    ":test_matrix_multiply",
    ":test_pooling",
    ":test_reduce",
    ":test_shuffle",
    ":test_softmax",
    ":test_unary",
  ]
)


