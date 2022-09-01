load("@rules_cc//cc:defs.bzl", "cc_test")
load("//tests/core/partitioning:partitioning_test.bzl", "partitioning_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

filegroup(
    name = "jit_models",
    srcs = [
        "//tests/modules:conditional_scripted.jit.pt",
        "//tests/modules:inplace_op_if_scripted.jit.pt",
        "//tests/modules:loop_fallback_eval_scripted.jit.pt",
        "//tests/modules:loop_fallback_no_eval_scripted.jit.pt",
        "//tests/modules:mobilenet_v2_traced.jit.pt",
        "//tests/modules:resnet50_traced.jit.pt",
    ],
)

partitioning_test(
    name = "test_segmentation",
)

partitioning_test(
    name = "test_shape_analysis",
)

partitioning_test(
    name = "test_tensorrt_conversion",
)

partitioning_test(
    name = "test_stitched_graph",
)

partitioning_test(
    name = "test_resolve_nontensor_inputs",
)

cc_test(
    name = "test_loading_model",
    srcs = ["test_loading_model.cpp"],
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

cc_test(
    name = "test_fallback_graph_output",
    srcs = ["test_fallback_graph_output.cpp"],
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

cc_test(
    name = "test_loop_fallback",
    srcs = ["test_loop_fallback.cpp"],
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

cc_test(
    name = "test_conditionals",
    srcs = ["test_conditionals.cpp"],
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
    name = "partitioning_tests",
    tests = [
        ":test_conditionals",
        ":test_fallback_graph_output",
        ":test_loading_model",
        ":test_loop_fallback",
        ":test_resolve_nontensor_inputs",
        ":test_segmentation",
        ":test_shape_analysis",
        ":test_stitched_graph",
        ":test_tensorrt_conversion",
    ],
)
