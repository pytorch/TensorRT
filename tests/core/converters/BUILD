load("//tests/core/converters:converter_test.bzl", "converter_test")

converter_test(
    name = "test_softmax"
)

converter_test(
    name = "test_activation"
)

converter_test(
    name = "test_pooling"
)

converter_test(
    name = "test_unary"
)

converter_test(
    name = "test_linear"
)

converter_test(
    name = "test_element_wise"
)

converter_test(
    name = "test_conv"
)

converter_test(
    name = "test_reduce"
)

test_suite(
    name = "test_converters",
    tests = [
        ":test_softmax",
        ":test_activation",
        ":test_pooling",
        ":test_unary",
        ":test_linear",
        ":test_element_wise",
        ":test_conv",
        ":test_reduce"
    ]
)


