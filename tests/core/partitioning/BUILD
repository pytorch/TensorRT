load("//tests/core/partitioning:partitioning_test.bzl", "partitioning_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
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

cc_test(
  name = "test_fallback_graph_output",
  srcs = ["test_fallback_graph_output.cpp"],
  deps = [
      "//tests/util",
      "//core",
      "@googletest//:gtest_main",
  ] + select({
      ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
      "//conditions:default":  ["@libtorch//:libtorch"],
  }),
  data = [
      "//tests/modules:jit_models"
  ]
)

test_suite(
    name = "partitioning_tests",
    tests = [
        ":test_segmentation",
        ":test_shape_analysis",
        ":test_tensorrt_conversion",
        ":test_stitched_graph",
        ":test_fallback_graph_output"
    ]
)