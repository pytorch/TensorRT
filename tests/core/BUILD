load("@rules_cc//cc:defs.bzl", "cc_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

filegroup(
    name = "jit_models",
    srcs = ["//tests/modules:mobilenet_v2_scripted.jit.pt"],
)

cc_test(
    name = "test_detecting_input_type",
    srcs = ["test_detecting_input_type.cpp"],
    data = [
        ":jit_models",
    ],
    deps = [
        "//tests/util",
        "@googletest//:gtest_main",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
)

test_suite(
    name = "core_tests",
    tests = [
        ":test_detecting_input_type",
        "//tests/core/conversion:conversion_tests",
        "//tests/core/lowering:lowering_tests",
        "//tests/core/partitioning:partitioning_tests",
    ],
)
