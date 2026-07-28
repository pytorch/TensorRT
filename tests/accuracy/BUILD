load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

filegroup(
    name = "jit_models",
    srcs = glob(["**/*.jit.pt"]),
)

filegroup(
    name = "data",
    srcs = glob(["datasets/**/*"]),
)

test_suite(
    name = "aarch64_accuracy_tests",
    tests = [
        ":test_dla_fp16_accuracy",
        ":test_dla_int8_accuracy",
        ":test_fp16_accuracy",
        ":test_fp32_accuracy",
        ":test_int8_accuracy",
    ],
)

test_suite(
    name = "accuracy_tests",
    tests = [
        ":test_fp16_accuracy",
        ":test_fp32_accuracy",
        ":test_int8_accuracy",
    ],
)

cc_test(
    name = "test_int8_accuracy",
    srcs = ["test_int8_accuracy.cpp"],
    data = [
        ":data",
        ":jit_models",
    ],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10",
    ],
)

cc_test(
    name = "test_fp16_accuracy",
    srcs = ["test_fp16_accuracy.cpp"],
    data = [
        ":data",
        ":jit_models",
    ],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10",
    ],
)

cc_test(
    name = "test_fp32_accuracy",
    srcs = ["test_fp32_accuracy.cpp"],
    data = [
        ":data",
        ":jit_models",
    ],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10",
    ],
)

cc_test(
    name = "test_dla_int8_accuracy",
    srcs = ["test_dla_int8_accuracy.cpp"],
    data = [
        ":data",
        ":jit_models",
    ],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10",
    ],
)

cc_test(
    name = "test_dla_fp16_accuracy",
    srcs = ["test_dla_fp16_accuracy.cpp"],
    data = [
        ":data",
        ":jit_models",
    ],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10",
    ],
)

cc_binary(
    name = "test",
    srcs = ["test.cpp"],
    data = [
        ":jit_models",
    ],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10",
    ],
)

cc_library(
    name = "accuracy_test",
    hdrs = ["accuracy_test.h"],
    deps = [
        "//cpp:torch_tensorrt",
        "//tests/util",
        "@googletest//:gtest_main",
        "@libtorch",
    ],
)
