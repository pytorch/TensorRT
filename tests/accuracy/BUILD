filegroup(
    name = "jit_models",
    srcs = glob(["**/*.jit.pt"])
)

test_suite(
    name = "test_accuracy_aarch64",
    tests = [
        ":test_dla_int8_accuracy",
	":test_dla_fp16_accuracy",
        ":test_int8_accuracy",
        ":test_fp16_accuracy",
        ":test_fp32_accuracy",
    ]
)

test_suite(
    name = "test_accuracy",
    tests = [
        ":test_int8_accuracy",
        ":test_fp16_accuracy",
        ":test_fp32_accuracy",
    ]
)

cc_test(
    name = "test_int8_accuracy",
    srcs = ["test_int8_accuracy.cpp"],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10"
    ],
    data = [
        ":jit_models",
    ]
)

cc_test(
    name = "test_fp16_accuracy",
    srcs = ["test_fp16_accuracy.cpp"],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10"
    ],
    data = [
        ":jit_models",
    ]
)

cc_test(
    name = "test_fp32_accuracy",
    srcs = ["test_fp32_accuracy.cpp"],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10"
    ],
    data = [
        ":jit_models",
    ]
)

cc_test(
    name = "test_dla_int8_accuracy",
    srcs = ["test_dla_int8_accuracy.cpp"],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10"
    ],
    data = [
        ":jit_models",
    ]
)

cc_test(
    name = "test_dla_fp16_accuracy",
    srcs = ["test_dla_fp16_accuracy.cpp"],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10"
    ],
    data = [
        ":jit_models",
    ]
)

cc_binary(
    name = "test",
    srcs = ["test.cpp"],
    deps = [
        ":accuracy_test",
        "//tests/accuracy/datasets:cifar10"
    ],
    data = [
        ":jit_models",
    ]
)


cc_library(
    name = "accuracy_test",
    hdrs = ["accuracy_test.h"],
    deps = [
        "//cpp/api:trtorch",
        "//tests/util",
        "@libtorch//:libtorch",
        "@googletest//:gtest_main",
    ],
)
