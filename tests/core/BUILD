load("@rules_cc//cc:defs.bzl", "cc_test")

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default"
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack"
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
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
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_torch_whl": ["@torch_whl//:libtorch"],
        "//conditions:default": ["@libtorch"],
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
