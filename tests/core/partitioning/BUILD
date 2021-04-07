load("//tests/core/partitioning:partitioning_test.bzl", "partitioning_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

filegroup(
    name = "jit_models",
    srcs = glob(["**/*.jit"])
)

partitioning_test(
  name = "test_segmentation",
)

partitioning_test(
  name = "test_shape_analysis",
)

test_suite(
    name = "partitioning_test",
    tests = [
        ":test_segmentation",
        ":test_shape_analysis",
    ]
)